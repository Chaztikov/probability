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
pericardial_labels = ['atrial_pressure' 'upstream_pressure' 'pump_flow' 'flow' 'downstream_pressure']




kernel_type = 'ExponentiatedQuadratic'
ii0, dii, iif = 0, 10, 2650
dii = 10
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
paramfilenames = ['aortic___paramslists.txt','pericar__paramslists.txt']
times = np.arange(0, 0.855, 0.000334)

paramslist = []
header = ['amplitude','length_scale','observation_noise_variance']
# paramslist.append(['amplitude','length_scale','observation_noise_variance'])

for filename,pfilename in zip(filenames,paramfilenames):
    paramslists = np.loadtxt('./save/'+pfilename)

    df = pd.read_csv(filename)
    times = df['index'].values
    times = np.atleast_2d(times).T
    
    plt.figure(figsize=(12, 4))
    for val,params in zip(df.columns[1:],paramslists):
        lls_ = np.loadtxt('./save/'+filename[:7]+'_'+val+'_loglikelihood.txt')
        
        # Plot the loss evolution
        
        plt.plot(lls_)
    plt.grid()
    plt.title('Log marginal likelihood')
    plt.xlabel("Training iteration")
    plt.ylabel("Log marginal likelihood")
    plt.legend(df.columns[1:])
    plt.savefig('./save/'+filename[:7]+'_lml_out.png')
    plt.close()