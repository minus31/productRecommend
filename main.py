# -*- coding: utf_8 -*-
import os
import argparse
import cv2
import pickle
import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from keras.engine.input_layer import Input
from keras.models import Model
from keras.applications.densenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from model import *
from custom_loss import *
from utils import *

def preprocess(img):
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    return preprocess_input(img)


def get_feature(model, DB_path):

    img_size = (256, 256)

    intermediate_model = Model(
        inputs=model.input, outputs=model.layers[-2].output)

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess, dtype='float32')

    db_generator = test_datagen.flow_from_directory(
        directory=DB_path,
        classes=["db"],
        target_size=(256, 256),
        color_mode="rgb",
        batch_size=32,
        class_mode=None,
        shuffle=False)

    db_vecs = intermediate_model.predict_generator(db_generator,
                                                   steps=len(db_generator),
                                                   verbose=1)
   

    return l2_normalize(db_vecs)

def join_generators(generators):
    while True: # keras requires all generators to be infinite
        data = [g for g in generators]

        x = [d[0] for d in data]
        
        # label smoothing 
        epsilon = 1e-1
        y1 = [(1 - epsilon)*d[1] + (epsilon/len(d[1])) for d in data]
        
        y2 = [d[1] for d in data]

        yield x, y1, y2
        
# class CustomHistory(keras.callbacks.Callback):
#     def init(self):
#         self.losses = []
#         self.val_losses = []
        
#     def on_epoch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))
#         self.val_losses.append(logs.get('val_loss'))


class Descriptor():
    def __init__(self, config):

        self.input_shape = config.input_shape
        self.sbow_shape = config.sbow_shape
        self.num_classes = config.num_classes
        self.batch_size = config.batch_size
        self.nb_epoch = config.epoch

        self.model = gcd_model(self.input_shape, self.num_classes)

    def train(self, dataset_path, datagen, checkpoint_path, checkpoint_inteval):

        opt = keras.optimizers.Adam(amsgrad=True)
        model = self.model
        model.compile(loss=['categorical_crossentropy', ArcFaceloss], optimizer=opt)

        train_generator = datagen.flow_from_directory(
            directory=dataset_path,
            target_size=self.input_shape[:2],
            color_mode="rgb",
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=True,
            subset='training')

        val_generator = datagen.flow_from_directory(
            directory=dataset_path,
            target_size=self.input_shape[:2],
            color_mode="rgb",
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=True,
            subset='validation')
        
        
        """ Callback """
        monitor = 'loss'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=4)
        
        """callback for Tensorboard"""
#         custom_hist = CustomHistory()
#         custom_hist.init()
        tb = keras.callbacks.TensorBoard(log_dir="./logs/", update_freq='batch')

        """ Training loop """
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VAL = val_generator.n // val_generator.batch_size

        t0 = time.time()

        for epoch in range(self.nb_epoch):
            t1 = time.time()
            res = model.fit_generator(generator=join_generators(train_generator),
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      initial_epoch=epoch,
                                      validation_data=join_generators(val_generator),
                                      validation_steps=STEP_SIZE_VAL,
                                      epochs=epoch + 1,
                                      callbacks=[reduce_lr, tb],
                                      verbose=1,
                                      shuffle=True)
            t2 = time.time()
            print(res.history)
            print('Training time for one epoch : %.1f' % ((t2 - t1)))

            if epoch % checkpoint_inteval == 0:
                model.save_weights(os.path.join(checkpoint_path, str(epoch)))
                
                
        model.save_weights(os.path.join(checkpoint_path, "finish.hdf5"))
        print('Total training time : %.1f' % (time.time() - t0))

    def updateDB(self, model_path, DB_path, reference_path):
        files = sorted(os.listdir(DB_path + "db"))
        db = [file for file in files if file.endswith(".png")]
        print("db file:", len(db))

        self.model.load_weights(model_path)

        features = get_feature(self.model, DB_path)
        
        print("feature's shape", features.shape)
        
        reference = {}
        
        reference["img"] = db
        reference["feature"] = list(features)
        
        with open(reference_path, "wb") as f:
            pickle.dump(reference, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("UPDATE COMPLETE")

        return None

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=1000)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--num_classes', type=int, default=600)
    args.add_argument('--input_shape', type=int, default=(256, 256, 3))
    args.add_argument('--sbow_shape', type=int, default=(128,))
    args.add_argument('--train', type=bool, default=False)
    args.add_argument('--updateDB', type=bool, default=False)
    args.add_argument('--DB_path', type=str, default=None)
    args.add_argument('--model_path', type=str,
                      default="./checkpoint/finish.hdf5")
    args.add_argument('--dataset_path', type=str, default="./data/images/")
    args.add_argument('--checkpoint_path', type=str, default="./checkpoint/")
    args.add_argument('--checkpoint_inteval', type=int, default=10)
    args.add_argument('--reference_path', type=str, default="./reference_part.p")

    config = args.parse_args()

    descriptor = Descriptor(config)

    if config.train:

        datagen = ImageDataGenerator(preprocessing_function=preprocess,
                                     zoom_range=0.2, vertical_flip=True, horizontal_flip=True,
                                     validation_split=0.1)

        descriptor.train(config.dataset_path, datagen,
                         checkpoint_path=config.checkpoint_path, checkpoint_inteval=config.checkpoint_inteval)

    if config.updateDB:
        
        descriptor.updateDB(config.model_path, config.DB_path, config.reference_path)