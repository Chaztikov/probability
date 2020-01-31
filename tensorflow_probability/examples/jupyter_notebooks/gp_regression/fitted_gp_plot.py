import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()

from mpl_toolkits.mplot3d import Axes3D
# Configure plot defaults
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'

tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels

params = {'amplitude': 21.553632877877284,
'length_scale': 0.053389071478613485,
'period': 0.7718989498338754,
'observation_noise_variance': 3.165786981797053}


kernel_type='ExpSinSquared'
ii0,dii,iif=0,2,2650
nt_max = np.floor((iif-ii0)/dii).astype(int)
print(nt_max)
NUM_TRAINING_POINTS = 0
data_dir = '/home/chaztikov/Documents/work/uq/data/'
fname = 'aortic_valve_piv_data.txt'
fname = 'pericardial_valve_piv_data.txt'

# atrial_pressure upstream_pressure pump_flow flow downstream_pressure
label1,label2 = 'atrial_pressure', 'upstream_pressure'
label1='flow'

def get_header(data_dir='',fname='',comment=0):
    with open(data_dir+'/'+fname,'r') as f:
        head = f.readline()
        head = head.split()
        if comment:
            head = head[1:]
    return head

def sinusoid(x):
  return np.sin(3 * np.pi * x[..., 0])

def generate_1d_data(num_training_points, observation_noise_variance):
  """Generate noisy sinusoidal observations at a random set of points.

  Returns:
     observation_index_points, observations
  """
  index_points_ = np.random.uniform(-1., 1., (num_training_points, 1))
  index_points_ = index_points_.astype(np.float64)
  # y = f(x) + noise
  observations_ = (sinusoid(index_points_) +
                   np.random.normal(loc=0,
                                    scale=np.sqrt(observation_noise_variance),
                                    size=(num_training_points)))
  return index_points_, observations_
  
# Generate training data with a known noise level (we'll later try to recover
# this value from the data).

def generate_data(NUM_TRAINING_POINTS=0, value1='',value2='',data_dir = '', fname=''):   
    if(NUM_TRAINING_POINTS>1):
        observation_index_points_, observations_ = generate_1d_data(num_training_points=NUM_TRAINING_POINTS, observation_noise_variance=.1)
    else:
        data = np.loadtxt(data_dir+'/'+fname)
        
        head = get_header(data_dir,fname,1)
        head = np.array(head)
        iname = np.where(head==value1)[0]
        
        observations_ = data[ii0:iif:dii, iname]

        nt = observations_.shape[0]

        observation_index_points_ =  np.loadtxt(data_dir+'/'+'time_data.txt')
        
        t0 =observation_index_points_.min()
        tf = observation_index_points_.max()
        observation_index_points_ = np.linspace(t0,tf,nt)
        observation_index_points_ = np.atleast_2d(observation_index_points_).T
        
    return observation_index_points_, observations_
    
observation_index_points_, observations_ = generate_data(NUM_TRAINING_POINTS,label1,label2,data_dir,fname)



kernel = tfk.ExpSinSquared(amplitude=tf.constant(params['amplitude'],dtype=tf.float32),
                           length_scale=tf.constant(params['length_scale'],dtype=tf.float32),
                           period=tf.constant(params['period'],dtype=tf.float32))
                           

# kernel = psd_kernels.ExponentiatedQuadratic(
#     amplitude=tf.Variable(1., dtype=np.float64, name='amplitude'),
#     length_scale=tf.Variable(1., dtype=np.float64, name='length_scale'))

nsamples = 10
# Index points should be a collection (100, here) of feature vectors. In this
# example, we're using 1-d vectors, so we just need to reshape the output from
# np.linspace, to give a shape of (100, 1).
index_points_ = np.random.uniform(0, 1., (nsamples, 1))
# index_points_ = index_points_.astype(tf.float32)


X = tf.cast(index_points_,tf.float32)
gpm = tfd.GaussianProcess(kernel,index_points=X)

gpm.sample(nsamples)
Y = gpm.sample(nsamples)
# print(np.dtype(Y))
Y = tf.cast(Y,tf.float32)

print(X.shape, Y.shape)
plt.plot(X, Y,'.')
plt.show()