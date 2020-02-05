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
dii = 5
num_optimizer_iters = 100
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
df = pd.read_csv(filename)


times = np.arange(0, 0.855, 0.000334)
times = df['index'].values
times = np.atleast_2d(times).T
print(times.shape)
paramslist = []
for val in df.columns[-2:-1]:
    
    if(0):
        # Plot the loss evolution
        plt.figure(figsize=(12, 4))
        plt.plot(lls_)
        plt.grid()
        plt.title('Log marginal likelihood')
        plt.xlabel("Training iteration")
        plt.ylabel("Log marginal likelihood")
        plt.savefig(val+'_lml_out.png')
        plt.close()
    '''
    USE MODEL
    '''

    predictive_index_points_ = np.array(times[::dii, 0], dtype=np.float64)
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
                c='r', alpha=.9,
                label='Posterior Sample' if i == 0 else None)
    leg = plt.legend(loc='upper right')
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.grid()
    plt.xlabel(r"Index points ($\mathbb{R}^1$)")
    plt.ylabel("Observation space")
    plt.savefig(val+'_gpm_out.png')
    plt.close()
