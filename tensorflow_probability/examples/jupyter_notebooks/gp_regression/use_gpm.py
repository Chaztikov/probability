import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels
# Suppose we have some data from a known function. Note the index points in
# general have shape `[b1, ..., bB, f1, ..., fF]` (here we assume `F == 1`),
# so we need to explicitly consume the feature dimensions (just the last one
# here).
f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)
observed_index_points = np.expand_dims(np.random.uniform(-1., 1., 50), -1)
# Squeeze to take the shape from [50, 1] to [50].
observed_values = f(observed_index_points)

# Define a kernel with trainable parameters.
kernel = psd_kernels.ExponentiatedQuadratic(
    amplitude=tf.Variable(1., dtype=np.float64, name='amplitude'),
    length_scale=tf.Variable(1., dtype=np.float64, name='length_scale'))

gp = tfd.GaussianProcess(kernel, observed_index_points)

optimizer = tf.optimizers.Adam()

@tf.function
def optimize():
  with tf.GradientTape() as tape:
    loss = -gp.log_prob(observed_values)
  grads = tape.gradient(loss, gp.trainable_variables)
  optimizer.apply_gradients(zip(grads, gp.trainable_variables))
  return loss

for i in range(1000):
  neg_log_likelihood = optimize()
  if i % 100 == 0:
    print("Step {}: NLL = {}".format(i, neg_log_likelihood))
print("Final NLL = {}".format(neg_log_likelihood))



# Suppose we have some data from a known function. Note the index points in
# general have shape `[b1, ..., bB, f1, ..., fF]` (here we assume `F == 1`),
# so we need to explicitly consume the feature dimensions (just the last one
# here).
f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)
observed_index_points = np.expand_dims(np.random.uniform(-1., 1., 50), -1)
# Squeeze to take the shape from [50, 1] to [50].
observed_values = f(observed_index_points)

# Define a kernel with trainable parameters.
kernel = psd_kernels.ExponentiatedQuadratic(
    amplitude=tf.Variable(1., dtype=np.float64, name='amplitude'),
    length_scale=tf.Variable(1., dtype=np.float64, name='length_scale'))

gp = tfd.GaussianProcess(kernel, observed_index_points)

optimizer = tf.optimizers.Adam()

@tf.function
def optimize():
  with tf.GradientTape() as tape:
    loss = -gp.log_prob(observed_values)
  grads = tape.gradient(loss, gp.trainable_variables)
  optimizer.apply_gradients(zip(grads, gp.trainable_variables))
  return loss

for i in range(1000):
  neg_log_likelihood = optimize()
  if i % 100 == 0:
    print("Step {}: NLL = {}".format(i, neg_log_likelihood))
print("Final NLL = {}".format(neg_log_likelihood))