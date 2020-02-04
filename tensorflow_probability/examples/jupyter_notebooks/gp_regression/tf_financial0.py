#from
# https://blog.tensorflow.org/2018/09/the-trinity-of-errors-in-financial-models.html

# First we encode our assumptions.
num_times_fed_meets_per_year = 8.
possible_fed_increases = tf.range(
    start=0.,
    limit=num_times_fed_meets_per_year + 1)
possible_cc_interest_rates = 2. + 10. + 0.25 * possible_fed_increases 
prob_fed_raises_rates = tf.constant([0.6, 0.7, 0.8, 0.9])
# Now we use TFP to compute probabilities in a vectorized manner.
# Pad a dim so we broadcast fed probs against CC interest rates.
prob_fed_raises_rates = prob_fed_raises_rates[…, tf.newaxis]
prob_cc_interest_rate = tfd.Binomial(
    total_count=num_times_fed_meets_per_year,
    probs=prob_fed_raises_rates).prob(possible_fed_increases)