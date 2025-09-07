# photopost/nima/loss.py
import tensorflow as tf


@tf.function
def emd_loss(y_true, y_pred):
    # Squared Earth Mover's Distance (EMD) between two distributions.
    # y_true, y_pred: shape (B, 10), rows sum to 1.
    # Returns scalar loss.
    cdf_true = tf.cumsum(y_true, axis=1)
    cdf_pred = tf.cumsum(y_pred, axis=1)
    cdf_diff = cdf_true - cdf_pred
    sample_emd = tf.reduce_sum(tf.square(cdf_diff), axis=1)
    return tf.reduce_mean(sample_emd)