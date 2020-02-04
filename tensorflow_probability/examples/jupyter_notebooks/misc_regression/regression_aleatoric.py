
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

'''
In the previous section we’ve seen that there is variability in y for any particular value of x. We can treat this variability as being inherent to the problem. This means that even if we had an infinite training set, we still wouldn’t be able to predict the labels perfectly. A common example of this kind of uncertainty is the outcome of a fair coin flip (assuming you are not equipped with a detailed model of physics etc.). No matter how many flips we’ve seen in the past, we cannot predict what the flip will be in the future.

We will assume that this variability has a known functional relationship to the value of x. Let us model this relationship using the same linear function as we did for the mean of y.
'''

# Build model.
model = tfk.Sequential([
  tf.keras.layers.Dense(1 + 1),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),
])

# Do inference.
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
model.fit(x, y, epochs=500, verbose=False)

# Make predictions.
yhat = model(x_tst)
#Now, in addition to predicting the mean of the label distribution, we also predict its scale (standard deviation). After training and forming the predictions the same way, we can get meaningful predictions about the variability of y as a function of x. Like before:
mean = yhat.mean()
stddev = yhat.stddev()
mean_plus_2_stddev = mean - 2. * stddev
mean_minus_2_stddev = mean + 2. * stddev
#new model graph
#Much better! Our model is now less certain about what y should be as x gets larger. This kind of uncertainty is called aleatoric uncertainty, because it represents variation inherent to the underlying process . Though we’ve made progress, aleatoric uncertainty is not the only source of uncertainty in this problem. Before going further, let us consider the other source of uncertainty that we’ve hitherto ignored.