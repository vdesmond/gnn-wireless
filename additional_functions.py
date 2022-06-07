from locale import normalize
import math
import sys

import tensorflow as tf

NOISE_POWER = tf.constant(4e-12)
P_DBM = 40
P = 10  # math.pow(10, P_DBM / 10)
NOISE_DENSITY = -169
BANDWIDTH = 5e6
P_NOISE_DBM = NOISE_DENSITY + 10 * math.log10(BANDWIDTH)
P_NOISE = math.pow(10, (P_NOISE_DBM - 30) / 10)


@tf.function()
def d2d_dims(N):
    return tf.cast((-1 + tf.sqrt(tf.cast(1 + 4 * N, dtype=tf.float32))) // 2, tf.int32)


@tf.function()
def sum_diag(A):
    return tf.reduce_sum(tf.square(A), axis=0)


@tf.function()
def log2(A):
    numerator = tf.math.log(A)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator


@tf.function()
def bce_loss(y_true, y_pred):
    """Binary Cross Entropy Loss with changes to be compatibile
       with IGNNITION

    Args:
        y_true (tf.Tensor): True labels {0, 1}
        y_pred (tf.Tensor): Predicted labels [0., 1.]

    Returns:
        loss value
    """

    N = int(tf.shape(y_pred)[0])
    X = d2d_dims(N)
    temp = tf.squeeze(y_pred)[:X]
    y_pred_split = tf.expand_dims(temp, axis=1)

    bce = tf.keras.losses.BinaryCrossentropy()
    loss_value = bce(y_true, y_pred_split)
    return loss_value


@tf.function
def object_rate_sum(H, X, L):
    """Function to calculate object sum rate

    Args:
        H (tf.Tensor): Channel Tensor of size (L, L)
        X (tf.Tensor): Link scheduling 1D Tensor
        L (tf.Tensor): Singleton Tensor denoting number of D2D pairs

    Returns:
        tf.Tensor: object sum rate
    """
    R = tf.TensorArray(tf.float32, size=L)

    for i in range(L):
        sum_arr = tf.TensorArray(tf.float32, size=L)
        for j in range(L):
            sum_arr = sum_arr.write(j, H[j, i] * P * X[j])
        sum_all = tf.reduce_sum(sum_arr.stack())
        sum_ij = sum_all - H[i, i] * P * X[i]
        R = R.write(i, log2(1 + H[i, i] * P * X[i] / (sum_ij + P_NOISE)))

    return tf.reduce_sum(R.stack())


@tf.function()
def sum_rate_metric(y_true, y_pred):
    """Average Sum Rate metric."""

    N = int(tf.shape(y_pred)[0])
    L = d2d_dims(N)

    channel_int = tf.squeeze(y_pred[2 * L :])
    channel_d2d = y_pred[L : 2 * L]

    indices = tf.sets.difference(
        tf.expand_dims(tf.range(0, L * L, 1), axis=0),
        tf.expand_dims(tf.range(0, L * L, 11), axis=0),
    ).values

    shape = tf.expand_dims(L * L, axis=0)

    Ht = tf.reshape(tf.scatter_nd(tf.expand_dims(indices, axis=1), channel_int, shape), [L, L])
    diag = tf.squeeze(tf.linalg.tensor_diag(channel_d2d))

    y_predicted = y_pred[:L]

    H = tf.add(Ht, diag)

    fpr = object_rate_sum(H, y_true, L)
    nnr = object_rate_sum(H, y_predicted, L)
    normalized_sum_rate = tf.divide(nnr, fpr)

    return normalized_sum_rate


def evaluation_metric(y_true, y_pred):
    return sum_rate_metric(y_true, y_pred)
