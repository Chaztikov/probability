from mpl_toolkits.mplot3d import Axes3D
import time

import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()

tfp.math.bije