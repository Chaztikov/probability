from mpl_toolkits.mplot3d import Axes3D
import time,os,sys
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


kernel_type = 'ExponentiatedQuadratic'
ii0, dii, iif = 0, 10, 2650
dii = 10
dt = .000334
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

paramfilenames = ['pericar__paramslists.txt','pericar__paramslists.txt']

tfinal = 0.855
times = np.arange(0, tfinal, 0.000334)

paramslist = []
header = ['amplitude','length_scale','observation_noise_variance']
# paramslist.append(['amplitude','length_scale','observation_noise_variance'])

for filename,pfilename in zip(filenames[:1],paramfilenames[:1]):
    paramslists = np.loadtxt('./save/'+pfilename)

    df = pd.read_csv('./save/'+filename)
    times = df['index'].values
    times = np.atleast_2d(times).T
    
    
    for val,params in zip(df.columns[1:2],paramslists):
        lls_ = np.loadtxt('./save/'+filename[:7]+'_'+val+'_loglikelihood.txt')
        
        amplitude_var, length_scale_var, observation_noise_variance_var = params
        
        input_points_ = times
        observations_ = df[val].values

        input_points_ = input_points_[::dii]
        input_points_ = np.array(input_points_, np.float64)

        observations_ = observations_[::dii]

        predictive_index_points_ = np.array(times[::dii, 0], dtype=np.float64)
        # Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
        predictive_index_points_ = predictive_index_points_[..., np.newaxis]

        optimized_kernel = tfk.ExponentiatedQuadratic(
            amplitude_var, length_scale_var)
        
        gprm = tfd.GaussianProcessRegressionModel(
            kernel=optimized_kernel,
            index_points=predictive_index_points_,
            observation_index_points=input_points_,
            observations=observations_,
            observation_noise_variance=observation_noise_variance_var,
            predictive_noise_variance=0.)

        samples = gprm.sample(num_predictive_samples)

        tau = 10./256.
        tau = tfinal / 10.
        print(tau,'\n\n')
        # Plot the true function, observations, and posterior samples.
        plt.figure(figsize=(12, 4))
        plt.scatter(
                    np.mod(input_points_[:, 0] * dt, tau),
                    observations_,
                    c='b',
                    marker='o',
                    label='Observations')
        for i in range(num_predictive_samples):
            plt.scatter( np.mod(input_points_*dt, tau),
                    samples[i, :],
                    c='r', alpha=.2,marker='.',
                    label='Posterior Sample' if i == 0 else None)
        leg = plt.legend(loc='upper right')
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        plt.grid()
        # plt.xlim(0,dt*2)
        plt.xlabel(r"Index points ($\mathbb{R}^1$)")
        plt.ylabel("Observation space")
        plt.savefig('./save/'+val+'_gpm_critic.png')
        plt.show()


