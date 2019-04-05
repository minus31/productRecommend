# -*- coding: utf_8 -*-
import os
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
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import BatchNormalization, Lambda, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, Activation, concatenate
from keras.regularizers import l2
from keras.utils import multi_gpu_model
from keras.applications.densenet import preprocess_input
import cv2

from model import *
from custom_loss import *

def preprocess(img):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    return preprocess_input(img)


def get_feature(model, DB_path):

    img_size = (224, 224)

    intermediate_model = Model(
        inputs=model.input, outputs=model.layers[-2].output)

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess, dtype='float32')

    db_generator = test_datagen.flow_from_directory(
        directory=DB_path,
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=32,
        class_mode=None,
        shuffle=False)

    db_vecs = intermediate_model.predict_generator(db_generator,
                                                   steps=len(
                                                       reference_generator),
                                                   verbose=1)

    return db_vecs

class Descriptor():
    def __init__(self, config):

        self.input_shape = config.input_shape
        self.sbow_shape = config.sbow_shape
        self.num_classes = config.num_classes
        self.batch_size = config.batch_size
        self.nb_epoch = config.epoch

        self.model = base_model(self.input_shape, self.num_classes)

    def train(self, dataset_path, datagen, checkpoint_path, checkpoint_inteval):

        opt = keras.optimizers.Adam(amsgrad=True)

        model.compile(loss=ArcFaceloss, optimizer=opt)

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
            target_size=input_shape[:2],
            color_mode="rgb",
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=True,
            subset='validation')

        """ Callback """
        monitor = 'loss'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=4)

        """ Training loop """
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VAL = val_generator.n // val_generator.batch_size

        t0 = time.time()

        for epoch in range(nb_epoch):
            t1 = time.time()
            res = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      initial_epoch=epoch,
                                      validation_data=val_generator,
                                      validation_steps=STEP_SIZE_VAL,
                                      epochs=epoch + 1,
                                      callbacks=[reduce_lr],
                                      verbose=1,
                                      shuffle=True)
            t2 = time.time()
            print(res.history)
            print('Training time for one epoch : %.1f' % ((t2 - t1)))

            if epoch % checkpoint_inteval == 0:
                model.save(checkpoint_path + str(epoch) + ".hdf5")
        model.save(checkpoint_path + "finish.hdf5")
        print('Total training time : %.1f' % (time.time() - t0))

    def updateDB(model_path, DB_path, reference_path):

        db = [os.path.join(DB_path, path) for path in os.listdir(DB_path)]

        model = load_model(model_path)

        features = get_feature(model, DB_path)

        reference = pd.DataFrame()

        reference["img"] = db
        reference["feature"] = features

        reference.to_csv(reference_path, index=False)

        print("UPDATE COMPLETE")

        return None

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=100)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--num_classes', type=int, default=1383)
    args.add_argument('--input_shape', type=int, default=(224, 224, 3))
    args.add_argument('--sbow', type=int, default=(128,))
    args.add_argument('--train', type=bool, default=False)
    args.add_argument('--updateDB', type=str, default=None)
    args.add_argument('--DB_path', type=str, default=None)
    args.add_argument('--model_path', type=str,
                      default="./checkpoint/finish.hdf5")
    args.add_argument('--dataset_path', type=str, default="./data/images/")
    args.add_argument('--checkpoint_path', type=str, default="./checkpoint/")
    args.add_argument('--checkpoint_inteval', type=int, default=10)
    args.add_argument('--reference_path', type=str, default="./")

    config = args.parse_args()

    descriptor = Descriptor(config)

    if config.train:

        datagen = ImageDataGenerator(preprocessing_function=preprocess,
                                     zoom_range=0.2, vertical_flip=True, horizontal_flip=True,
                                     validation_split=0.1)

        descriptor.train(config.dataset_path, datagen,
                         check_point=config.checkpoint_path, check_interval=config.checkpoint_inteval)

    if config.updateDB:
        descriptor.updateDB(config.model_path,
                            config.DB_path, config.reference_path)