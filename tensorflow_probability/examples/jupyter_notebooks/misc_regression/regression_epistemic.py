import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

'''
The noise in the data means that we can not be fully certain of the parameters of the linear relationship between x and y. For example, the slope we’ve found in the previous section seems reasonable, but we don’t know for sure, and perhaps a slightly shallower or steeper slope would also be reasonable. This kind of uncertainty is called the epistemic uncertainty; unlike aleatoric uncertainty, epistemic uncertainty can be reduced if we get more data. To get a sense of this uncertainty we shall replace the standard Keras Dense layer with TFP’s DenseVariational layer.

The DenseVariational layer uses a variational posterior Q(w) over the weights to represent the uncertainty in their values. This layer regularizes Q(w) to be close to the prior distribution P(w), which models the uncertainty in the weights before we look into the data.

For Q(w) we’ll use a multivariate normal distribution for the variational posterior with a trainable diagonal covariance matrix centered on a trainable location. For P(w) we’ll use a standard multivariate normal distribution for the prior with a trainable location and fixed scale. See Appendix B for more details about how this layer works.

Let’s put that all together:

'''
# Build model.
model = tf.keras.Sequential([
  tfp.layers.DenseVariational(1, posterior_mean_field, prior_trainable),
  tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
])

# Do inference.
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
model.fit(x, y, epochs=500, verbose=False)

# Make predictions.
yhats = [model(x_tst) for i in range(100)]


'''

Each line represents a different random draw of the model parameters from the posterior distribution. As we can see, there is in fact quite a bit of uncertainty about the linear relationship. Even if we don’t care about the variability of y for any particular value of x, the uncertainty in the slope should give us pause if we’re making predictions for x’s too far from 0.

Note that in this example we are training both P(w) and Q(w). This training corresponds to using Empirical Bayes or Type-II Maximum Likelihood. We used this method so that we wouldn’t need to specify the location of the prior for the slope and intercept parameters, which can be tough to get right if we do not have prior knowledge about the problem. Moreover, if you set the priors very far from their true values, then the posterior may be unduly affected by this choice. A caveat of using Type-II Maximum Likelihood is that you lose some of the regularization benefits over the weights. If you wanted to do a proper Bayesian treatment of uncertainty (if you had some prior knowledge, or a more sophisticated prior), you could use a non-trainable prior (see Appendix B).


'''
