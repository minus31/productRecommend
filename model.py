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

from keras.engine.base_layer import InputSpec, Layer
from keras.utils import conv_utils
from keras.legacy import interfaces


"""
Model Architecture for global descriptor 

"""

def cgd_model(input_shape, num_classes=None):
    """
    backbone model : ResNet50
    
    descriptor dim : 1024 (512 * 2) 
    """
    model = keras.applications.ResNet50(input_shape=input_shape, include_top=False)

    gd1 = GlobalAveragePooling2D()(model.layers[-1].output)
    gd2 = GlobalGeMPooling2D()(model.layers[-1].output)
 
    des1 = Dense(512, activation='elu', kernel_regularizer='l2')(gd1)
    des2 = Dense(512, activation='elu', kernel_regularizer='l2')(gd2)
    
    aux = Dense(num_classes, activation='softmax')(des1)
    # temperature scaling 
    aux = Lambda(lambda x: x / 0.5)(aux)
    
    con = concatenate([des1, des2])
    cos = CosineTheta(num_classes, 512*2)(con)

    model_new = Model(inputs=model.input, outputs=[aux, cos])

    return model_new


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

def base_densenet_model(input_shape, num_classes=None):
    """
    backbone model : DenseNet121
    """
    model = keras.applications.DenseNet121(
        input_shape=input_shape, include_top=False)
    # frozen
    model.trainable = False

    x = GlobalAveragePooling2D()(model.layers[-1].output)
 
    des = Dense(512, activation='elu', kernel_regularizer='l2')(x)
    
    cos = CosineTheta(num_classes, 512)(des)

    model_new = Model(inputs=model.input, outputs=cos)

    return model_new

def base_multihead_model(input_shape, num_classes=None):

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
    
    def get_config(self):
        config = {
        'num_classes': self.num_classes,
        'embedding_dim': self.embedding_dim
        }
        
        base_config = super(CosineTheta, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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


class _GlobalPooling2D(Layer):
    """Abstract class for different global pooling 2D layers.
    """

    @interfaces.legacy_global_pooling_support
    def __init__(self, data_format=None, **kwargs):
        super(_GlobalPooling2D, self).__init__(**kwargs)
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        self.pk = K.variable(value=1.5, dtype='float')

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return (input_shape[0], input_shape[3])
        else:
            return (input_shape[0], input_shape[1])

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(_GlobalPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalGeMPooling2D(_GlobalPooling2D):
    """
    """

    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.pow(K.mean(K.pow(inputs, self.pk), axis=[1, 2]), 1/self.pk)
        else:
            return K.pow(K.mean(K.pow(inputs, self.pk), axis=[2, 3]), 1/self.pk)



if __name__ == '__main__':

    pass