from mpl_toolkits.mplot3d import Axes3D
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()


# Configure plot defaults
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'


data_dir = '/home/chaztikov/Documents/work/uq/data/'
fname = 'aortic_valve_piv_data.txt'
fname = 'pericardial_valve_piv_data.txt'

# atrial_pressure upstream_pressure pump_flow flow downstream_pressure
label1, label2 = 'atrial_pressure', 'upstream_pressure'
label1 = 'flow'

aortic_labels = ['upstream_pressure' 'flow' 'downstream_pressure' 'pdva']
pericardial_labels = [
    'atrial_pressure' 'upstream_pressure' 'pump_flow' 'flow' 'downstream_pressure']

kernel_type = 'ExponentiatedQuadratic'
ii0, dii, iif = 0, 10, 2650
dii = 1
num_optimizer_iters = 10000
num_predictive_samples = 10

NUM_TRAINING_POINTS = 0

BPM = 60
InitialFrame = 12000
StartFFTFrame = 2000
NFFT = 512*4
FPS = 5000
PixelsPerCm = 100


FramesPerPulse = 60/BPM*FPS
# Area = A{1,2}/(PixelsPerCm^2);
# Time =(1:FPS)/FPS*1000;
# Pulses = floor((size(Area,1)-InitialFrame)/FramesPerPulse);

filenames = ['aortic__valve_piv_data.csv','pericar_valve_piv_data.csv']
times = np.arange(0, 0.855, 0.000334)

paramslist = []
paramslist.append(['amplitude','length_scale','observation_noise_variance'])

for filename in filenames:
    
    df = pd.read_csv(filename)
    times = df['index'].values
    times = np.atleast_2d(times).T
    
    for val in df.columns[1:]:

        observation_index_points_ = times
        observations_ = df[val].values

        observation_index_points_ = observation_index_points_[::dii]
        observation_index_points_ = np.array(observation_index_points_, np.float32)

        observations_ = observations_[::dii]
        #  , np.float32)

        print(observations_.shape, '\n ', observation_index_points_.shape)

        checkpoints_iterator_ = tf.train.checkpoints_iterator('.')

        def build_gp(amplitude, length_scale, observation_noise_variance):
            """Defines the conditional dist. of GP outputs, given kernel parameters."""
            # Create the covariance kernel, which will be shared between the prior (which we
            # use for maximum likelihood training) and the posterior (which we use for
            # posterior predictive sampling)

            mean_fn = None
            kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

            # Create the GP prior distribution, which we will use to train the model
            # parameters.
            model = tfd.GaussianProcess(
                kernel=kernel,
                index_points=observation_index_points_,
                mean_fn=mean_fn,
                observation_noise_variance=observation_noise_variance)

            return model

        gp_joint_model = tfd.JointDistributionNamed({
            'amplitude': tfd.Normal(loc=400., scale=np.float32(1.)),
            'length_scale': tfd.Normal(loc=0.03, scale=np.float32(1.)),
            'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float32(1.)),
            'observations': build_gp,
        })

        # Create the trainable model parameters, which we'll subsequently optimize.
        # Note that we constrain them to be strictly positive.
        constrain_positive = tfb.Shift(np.finfo(np.float32).tiny)(tfb.Exp())

        amplitude_var = tfp.util.TransformedVariable(
            initial_value=400.,
            bijector=constrain_positive,
            name='amplitude',
            dtype=np.float32)

        length_scale_var = tfp.util.TransformedVariable(
            initial_value=0.01,
            bijector=constrain_positive,
            name='length_scale',
            dtype=np.float32)

        observation_noise_variance_var = tfp.util.TransformedVariable(
            initial_value=1.,
            bijector=constrain_positive,
            name='observation_noise_variance',
            dtype=np.float32)

        trainable_variables = [v.trainable_variables[0] for v in
                            [amplitude_var,
                                length_scale_var,
                                observation_noise_variance_var]]

        # Use `tf.function` to trace the loss for more efficient evaluation.
        @tf.function(autograph=False, experimental_compile=False)
        def target_log_prob(amplitude, length_scale, observation_noise_variance):
            return gp_joint_model.log_prob({
                'amplitude': amplitude,
                'length_scale': length_scale,
                'observation_noise_variance': observation_noise_variance,
                'observations': observations_
            })
        
        # Now we optimize the model parameters.
        optimizer = tf.optimizers.Adam(learning_rate=.01)

        # Store the likelihood values during training, so we can plot the progress
        lls_ = np.zeros(num_optimizer_iters, np.float32)
        for i in range(num_optimizer_iters):
            with tf.GradientTape() as tape:
                loss = -target_log_prob(amplitude_var,
                                        length_scale_var,
                                        observation_noise_variance_var)
            grads = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            lls_[i] = loss

            # checkpoints_iterator_;
            checkpoint_ = tf.train.Checkpoint(optimizer=optimizer)
            manager = tf.train.CheckpointManager(
                checkpoint_, './.tf_ckpts', checkpoint_name='checkpoint_'+str(i), max_to_keep=3)

        def write_trained_parameters():
            params = [amplitude_var._value().numpy(), length_scale_var._value(
            ).numpy(), observation_noise_variance_var._value().numpy()]
            # params = np.array(params)
            return params
        # checkpoint = tf.train.Checkpoint(optimizer=optimizer)
        # manager = tf.train.CheckpointManager(
        #     checkpoint, './.tf_ckpts',
        #     checkpoint_name=checkpoint_name, max_to_keep=3)

        x = gp_joint_model.sample()
        lp = gp_joint_model.log_prob(x)

        print("sampled {}".format(x))
        print("log_prob of sample: {}".format(lp))

        params = write_trained_parameters()
        print(params, '\n')
        paramslist.append(params)

        np.savetxt(val+'_paramslist.txt', paramslist)

        np.savetxt(val+'_loglikelihood.txt', lls_)
