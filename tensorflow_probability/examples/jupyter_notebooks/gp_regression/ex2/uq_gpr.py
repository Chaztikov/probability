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

filename = 'aortic__valve_piv_data.csv'

times = np.arange(0,0.855,0.000334)
df = pd.read_csv(filename)
df['index']*=0.000334
observation_index_points_ = times
observations_ = df['flow'].values

observation_index_points_ = np.atleast_2d(observation_index_points_).T;

observation_index_points_.shape
observations_.shape

# np.array(df['flow'],dtype=np.float64)


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
    'amplitude': tfd.Normal(loc=400., scale=np.float64(1.)),
    'length_scale': tfd.Normal(loc=0.03, scale=np.float64(1.)),
    'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observations': build_gp,
})


# Create the trainable model parameters, which we'll subsequently optimize.
# Note that we constrain them to be strictly positive.
constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

amplitude_var = tfp.util.TransformedVariable(
    initial_value=400.,
    bijector=constrain_positive,
    name='amplitude',
    dtype=np.float64)

length_scale_var = tfp.util.TransformedVariable(
    initial_value=0.01,
    bijector=constrain_positive,
    name='length_scale',
    dtype=np.float64)

observation_noise_variance_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='observation_noise_variance_var',
    dtype=np.float64)

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
num_iters = 100
optimizer = tf.optimizers.Adam(learning_rate=.01)

# Store the likelihood values during training, so we can plot the progress
lls_ = np.zeros(num_iters, np.float64)
for i in range(num_iters):
  with tf.GradientTape() as tape:
    loss = -target_log_prob(amplitude_var,
                            length_scale_var,
                            observation_noise_variance_var)
  grads = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(grads, trainable_variables))
  lls_[i] = loss


# Plot the loss evolution
plt.figure(figsize=(12, 4))
plt.plot(lls_)
plt.grid()
plt.title('Log marginal likelihood')
plt.xlabel("Training iteration")
plt.ylabel("Log marginal likelihood")
plt.savefig('lml_out.png')


'''
USE MODEL
'''


predictive_index_points_ = np.array(times[:], dtype=np.float64)
# Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
predictive_index_points_ = predictive_index_points_[..., np.newaxis]

optimized_kernel = tfk.ExponentiatedQuadratic(
    amplitude_var, length_scale_var)
gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=predictive_index_points_,
    observation_index_points=observation_index_points_,
    observations=observations_,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)

# Create op to draw  50 independent samples, each of which is a *joint* draw
# from the posterior at the predictive_index_points_. Since we have 200 input
# locations as defined above, this posterior distribution over corresponding
# function values is a 200-dimensional multivariate Gaussian distribution!
samples = gprm.sample(num_predictive_samples)


# Plot the true function, observations, and posterior samples.
plt.figure(figsize=(12, 4))
plt.scatter(observation_index_points_[:, 0],
            observations_,
            c='b',
            marker='o',
            label='Observations')
for i in range(num_predictive_samples):
  plt.plot(predictive_index_points_,
           samples[i, :],
           c='r', alpha=.1, marker='.',
           label='Posterior Sample' if i == 0 else None)
leg = plt.legend(loc='upper right')
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.xlabel(r"Index points ($\mathbb{R}^1$)")
plt.ylabel("Observation space")
plt.savefig('gpm_out.png')


def print_trained_parameters():
    print('Trained parameters:')
    print('amplitude: {}'.format(amplitude_var._value().numpy()))
    print('length_scale: {}'.format(length_scale_var._value().numpy()))
    print('observation_noise_variance: {}'.format(
        observation_noise_variance_var._value().numpy()))

# checkpoint = tf.train.Checkpoint(optimizer=optimizer)
# manager = tf.train.CheckpointManager(
#     checkpoint, './.tf_ckpts',
#     checkpoint_name=checkpoint_name, max_to_keep=3)


x = gp_joint_model.sample()
lp = gp_joint_model.log_prob(x)

print("sampled {}".format(x))
print("log_prob of sample: {}".format(lp))


print_trained_parameters()


