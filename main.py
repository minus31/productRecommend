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
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from model import *
from custom_loss import *
from utils import *
from metric import *

def preprocess(img):
    """
    resize - 256,256, interpolation = cv2.INTER_AREA(which is recommended in a case of sizeing down)
    """
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    return preprocess_input(img)


def get_feature(model, DB_path):
    """
    Arguments
     model : keras model
     DB_path : path where dataset locate

    return
     l2_normalized global descriptor
    """

    img_size = (256, 256)

    intermediate_model = Model(
        inputs=model.input, outputs=model.layers[-3].output)

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess, dtype='float32')

    db_generator = test_datagen.flow_from_directory(
        directory=DB_path,
        target_size=(256, 256),
        color_mode="rgb",
        batch_size=32,
        class_mode=None,
        shuffle=False)

    db_vecs = intermediate_model.predict_generator(db_generator,
                                                   steps=len(db_generator),
                                                   verbose=1)

    return l2_normalize(db_vecs)

def gen_multiOutput(generators):
    """
    wrapper of data generator for multiple outputs
    """

    while True:
        data = next(generators)

        x = data[0]

        # label smoothing
        epsilon = 1e-1
        y1 = (1 - epsilon) * data[1] + (epsilon/data[1].shape[-1])

        y2 = data[1]

        yield x, [y1, y2]


class Descriptor():
    def __init__(self, config):

        self.input_shape = config.input_shape
        self.sbow_shape = config.sbow_shape
        self.num_classes = config.num_classes
        self.batch_size = config.batch_size
        self.nb_epoch = config.epoch
        self.k = config.k
        
        self.model = cgd_model(self.input_shape, self.num_classes)

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

        """Callback for Tensorboard"""
        tb = keras.callbacks.TensorBoard(log_dir="./logs/", update_freq='batch')

        """ Training loop """
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VAL = val_generator.n // val_generator.batch_size

        t0 = time.time()

        for epoch in range(self.nb_epoch):
            t1 = time.time()
            res = model.fit_generator(generator=gen_multiOutput(train_generator),
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      initial_epoch=epoch,
                                      validation_data=gen_multiOutput(val_generator),
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


        model.save_weights(os.path.join(checkpoint_path, "finish"))
        print('Total training time : %.1f' % (time.time() - t0))

    def updateDB(self, model_path):
        """
        model_path : model weight file path 
        """
        snap_path = "../Fashion_items_recommendation_demo/static/db/snap/"
        part_path = "../Fashion_items_recommendation_demo/static/db/part/"
        
        snap_files = sorted(os.listdir(snap_path + "snap"))
        snap_db = [file for file in snap_files if file.endswith(".png")]
        snap_db = sorted(snap_db)
        
        part_files = sorted(os.listdir(part_path + "part"))
        part_db = [file for file in part_files if file.endswith(".png")]
        part_db = sorted(part_db)

        self.model.load_weights(model_path)
        
        # feature : snapshot, reference : item
        features = get_feature(self.model, snap_path)
        reference = get_feature(self.model, part_path)
        
        print("feature's shape", features.shape)
        print("reference's shape", reference.shape)

        snap = {}
        snap["img"] = snap_db
        snap["feature"] = list(features)
        
        with open("./reference_snap.p", "wb") as f:
            pickle.dump(snap, f, protocol=pickle.HIGHEST_PROTOCOL)
                  
        part = {}
        part["img"] = part_db
        part["feature"] = list(reference)
                  
        with open("./reference_part.p", "wb") as f:
            pickle.dump(part, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("UPDATE COMPLETE!")

        return None
    
    def evaluate(self, model_path):
        """
        model_path : model weight file path 
        """
        
        self.model.load_weights(model_path)
        
        snap_path = "./data/db/snap/"
        part_path = "./data/db/db/"
        
        snap_files = sorted(os.listdir(snap_path + "snap"))
        snap_db = [file for file in snap_files if file.endswith(".png")]
        snap_db = sorted(snap_db)
        
        part_files = sorted(os.listdir(part_path + "part"))
        part_db = [file for file in part_files if file.endswith(".png")]
        part_db = sorted(part_db)
        
        features = get_feature(self.model, snap_path)
        reference = get_feature(self.model, part_path)

            
        sim_vector = np.dot(features.reshape(-1, 1024), reference.reshape(-1, 1024).T)
        indice = np.argsort(sim_vector, axis=-1)
        indice = np.flip(indice, axis=-1)
        
        results = []
        
        for i in range(features.shape[0]):
            
            ranked_list = [part_db[k] for k in indice[i]]
            results.append(ranked_list)
            
        mAP_, APs = mAP(results, self.k)     
        
        APs = {k : v for k, v in zip(snap_db, APs)}

        return mAP_, APs

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
    args.add_argument('--eval', type=bool, default=False)
    args.add_argument('--model_path', type=str,
                      default="./checkpoint/finish")
    args.add_argument('--dataset_path', type=str, default="./data/images/")
    args.add_argument('--checkpoint_path', type=str, default="./checkpoint/")
    args.add_argument('--checkpoint_inteval', type=int, default=10)
    args.add_argument('--k', type=int, default=21)

    config = args.parse_args()
    
    descriptor = Descriptor(config)

    if config.train:

        datagen = ImageDataGenerator(preprocessing_function=preprocess,
                                     zoom_range=0.2, 
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     fill_mode='nearest',
                                     vertical_flip=True, 
                                     shear_range=0.2,
                                     horizontal_flip=True,
                                     validation_split=0.1)

        descriptor.train(config.dataset_path, datagen,
                         checkpoint_path=config.checkpoint_path, checkpoint_inteval=config.checkpoint_inteval)

    if config.updateDB:

        descriptor.updateDB(config.model_path)
        
    if config.eval:
        
        """only for the snapshot images"""
        mAP_, APs = descriptor.evaluate(config.model_path)
        
        with open("AP_log.p", "wb") as f: 
            pickle.dump(APs, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        print("mAP : {}".format(mAP_))
        