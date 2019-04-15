# -*- coding: utf_8 -*-

import tensorflow as tf
import math


def ArcFaceloss(labels, features):
    """
    ArcFace loss function 
    """

    N = tf.shape(labels)[0]
    s = 64.
    m1 = 1.
    m2 = 0.5
    m3 = 0.

    target_cos = tf.reduce_sum(tf.cast(labels, tf.float32) * features, axis=-1)
    target_cos = tf.cos(tf.math.acos(target_cos) * m1 + m2) - m3
    target_cos = tf.exp(s * target_cos)

    others = -1. * tf.multiply(tf.subtract(tf.cast(labels, tf.float32), 1.0), features)
    others = tf.exp(s * others)
    others = tf.subtract(tf.reduce_sum(others, axis=-1), 1.0)

    log_ = tf.log(tf.divide(target_cos, tf.add(tf.add(target_cos, others), 1e-5)))

    output = -1. * tf.divide(tf.reduce_sum(log_), tf.cast(N, tf.float32))

    return output


if __name__ == '__main__':
    pass
