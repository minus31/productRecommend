# -*- coding: utf_8 -*-
import os
import cv2
import pickle
import tensorflow as tf
import math
import numpy as np
import keras
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from keras.engine.input_layer import Input
from keras.models import Model

# from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import BatchNormalization, Lambda, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, Activation, concatenate
from keras.regularizers import l2
from keras.utils import multi_gpu_model


"""
Model Architecture for global descriptor 
"""


def style_considered_model(inputs_shape, num_classes=None):

    input_shape, sbow_shape = inputs_shape

    model = keras.applications.DenseNet169(
        input_shape=input_shape, include_top=False)
    # frozen
    model.trainable = False

    x1_1 = GlobalAveragePooling2D()(model.layers[-1].output)
    x1_2 = GlobalAveragePooling2D()(model.layers[-9].output)
    x1_3 = GlobalAveragePooling2D()(model.layers[-23].output)

    x2_1 = Dense(512, activation='elu')(x1_1)
    x2_2 = Dense(512, activation='elu')(x1_2)
    x2_3 = Dense(512, activation='elu')(x1_3)

    sbow = Input(shape=sbow_shape, name="style_bow")

    con = concatenate([x2_1, x2_2, x2_3, sbow], axis=-1)
    sdes = Dense(512, kernel_regularizer='l2')(con)
    cos = CosineTheta(num_classes, 512)(sdes)

    model_new = Model(inputs=[model.input, sbow], outputs=cos)

    return model_new


def base_model(input_shape, num_classes=None):

    model = keras.applications.DenseNet169(
        input_shape=input_shape, include_top=False)
    # frozen
    model.trainable = False

    x1_1 = GlobalAveragePooling2D()(model.layers[-1].output)
    x1_2 = GlobalAveragePooling2D()(model.layers[-9].output)
    x1_3 = GlobalAveragePooling2D()(model.layers[-23].output)

    x2_1 = Dense(512, activation='elu')(x1_1)
    x2_2 = Dense(512, activation='elu')(x1_2)
    x2_3 = Dense(512, activation='elu')(x1_3)

    con = concatenate([x2_1, x2_2, x2_3], axis=-1)
    des = Dense(512, kernel_regularizer='l2')(con)
    cos = CosineTheta(num_classes, 512)(des)

    model_new = Model(inputs=model.input, outputs=cos)

    return model_new


class CosineTheta(keras.layers.Layer):

    def __init__(self, num_classes, embedding_dim, **kwargs):

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        super(CosineTheta, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.var_weights = tf.get_variable(name="weight", shape=(
            self.num_classes, self.embedding_dim), initializer=constant_xavier_initializer)

        # Be sure to call this at the end
        super(CosineTheta, self).build(input_shape)

    def call(self, x):

        self.normed_features = tf.nn.l2_normalize(
            x, 1, 1e-10, name='features_norm')

        with tf.variable_scope("cos", reuse=tf.AUTO_REUSE):
            self.normed_weights = tf.nn.l2_normalize(
                self.var_weights, 1, 1e-10, name='weights_norm')

            cosine = tf.matmul(
                self.normed_features, self.normed_weights, transpose_a=False, transpose_b=True)

        return cosine

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes)


def constant_xavier_initializer(shape, dtype=tf.float32, uniform=True, **kwargs):
    """Initializer function."""

    if shape:
        fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
        fan_out = float(shape[-1])
    else:
        fan_in = 1.0
        fan_out = 1.0
    for dim in shape[:-2]:
        fan_in *= float(dim)
        fan_out *= float(dim)
    # Average number of inputs and output connections.
    n = (fan_in + fan_out) / 2.0
    if uniform:
        # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
        limit = math.sqrt(3.0 * 1.0 / n)
        return tf.random_uniform(shape, -limit, limit, dtype, seed=None)
    else:
        # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
        trunc_stddev = math.sqrt(1.3 * 1.0 / n)
        return tf.truncated_normal(shape, 0.0, trunc_stddev, dtype, seed=None)


if __name__ == '__main__':

    pass