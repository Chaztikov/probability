
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow.compat.v2 as tf
# import tensorflow_probability as tfp
# tfb = tfp.bijectors
# tfd = tfp.distributions
# tfk = tfp.math.psd_kernels
# tf.enable_v2_behavior()

from mpl_toolkits.mplot3d import Axes3D
# Configure plot defaults
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'





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
        print(head)
        iname = np.where(head==value1)
        iname=iname[0]
        print(iname,'\n \n')
        data = data[:,iname]
        # data = 
        
        observations_ = data
        print(observations_.shape)
        observation_index_points_ =  np.loadtxt(data_dir+'/'+'time_data.txt')
        nt = max(data.shape)
        t0 =observation_index_points_.min()
        tf = observation_index_points_.max()
        observation_index_points_ = np.linspace(t0,tf,nt)
        np.arange(np.max(data.shape))
    return observation_index_points_, observations_
    
observation_index_points_, observations_ = generate_data(NUM_TRAINING_POINTS,label1,label2,data_dir,fname)
plt.plot(observation_index_points_,observations_)
plt.show()
print(observations_.shape, observation_index_points_.shape)
