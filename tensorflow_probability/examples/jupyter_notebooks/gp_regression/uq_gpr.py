import time

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
        
        observations_ = data[:,0]
        # print(observations_.shape)
        observation_index_points_ =  np.loadtxt(data_dir+'/'+'time_data.txt')
        
        nt = max(data.shape)
        t0 =observation_index_points_.min()
        tf = observation_index_points_.max()
        observation_index_points_ = np.linspace(t0,tf,nt)
        observation_index_points_ = np.atleast_2d(observation_index_points_).T
        
    return observation_index_points_, observations_
    
observation_index_points_, observations_ = generate_data(NUM_TRAINING_POINTS,label1,label2,data_dir,fname)
# plt.plot(observation_index_points_,observations_)
# plt.show()
print(observations_.shape, observation_index_points_.shape)

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()

def build_gp(amplitude, length_scale, observation_noise_variance):
  """Defines the conditional dist. of GP outputs, given kernel parameters."""

  # Create the covariance kernel, which will be shared between the prior (which we
  # use for maximum likelihood training) and the posterior (which we use for
  # posterior predictive sampling)
  kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

  # Create the GP prior distribution, which we will use to train the model
  # parameters.
  return tfd.GaussianProcess(
      kernel=kernel,
      index_points=observation_index_points_,
      observation_noise_variance=observation_noise_variance)

gp_joint_model = tfd.JointDistributionNamed({
    'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observations': build_gp,
})

x = gp_joint_model.sample()
lp = gp_joint_model.log_prob(x)

print("sampled {}".format(x))
print("log_prob of sample: {}".format(lp))


# Create the trainable model parameters, which we'll subsequently optimize.
# Note that we constrain them to be strictly positive.

constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

amplitude_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='amplitude',
    dtype=np.float64)

length_scale_var = tfp.util.TransformedVariable(
    initial_value=1.,
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
num_iters = 1000
optimizer = tf.optimizers.Adam(learning_rate=.01)

# Store the likelihood values during training, so we can plot the progress
lls_ = np.zeros(num_iters, np.float64)
for i in range(num_iters):
  with tf.GradientTape() as tape:
    loss = -target_log_prob(amplitude_var, length_scale_var,
                            observation_noise_variance_var)
  grads = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(grads, trainable_variables))
  lls_[i] = loss

print('Trained parameters:')
print('amplitude: {}'.format(amplitude_var._value().numpy()))
print('length_scale: {}'.format(length_scale_var._value().numpy()))
print('observation_noise_variance: {}'.format(observation_noise_variance_var._value().numpy()))


# Plot the loss evolution
plt.figure(figsize=(12, 4))
plt.plot(lls_)
plt.grid()
plt.title('Log marginal likelihood')
plt.xlabel("Training iteration")
plt.ylabel("Log marginal likelihood")
plt.savefig('out.png')


print(observation_index_points_)
print(observations_)