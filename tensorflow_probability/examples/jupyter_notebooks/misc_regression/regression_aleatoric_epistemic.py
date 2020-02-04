

#Now that we looked at aleatoric and epistemic uncertainty in isolation, we can use TFP layers’ composable API to create a model that reports both types of uncertainty:

# Build model.
model = tf.keras.Sequential([
  tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_trainable),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:]))),
])

# Do inference.
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
model.fit(x, y, epochs=500, verbose=False);

# Make predictions.
yhats = [model(x_tst) for _ in range(100)]

#The only change we’ve made to the previous model is that we added an extra output to DenseVariational layer to also model the scale of the label distribution. As in our previous solution, we get an ensemble of models, but this time they all also report the variability of y as a function of x. Let us plot this ensemble:

#Note the qualitative difference between the predictions of this model compared to those from the model that considered only aleatoric uncertainty: this model predicts more variability as x gets more negative in addition to getting more positive — something that is not possible to do with a simple linear model of aleatoric uncertainty.