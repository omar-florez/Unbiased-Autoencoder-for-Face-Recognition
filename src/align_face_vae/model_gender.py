'''
    Automatic bias removal for Face Recognition
    Author: Omar U. Florez
    November, 2018
''' 

import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import shutil
import imageio

import keras.utils as utils
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Lambda, Conv2DTranspose, BatchNormalization
from keras.models import Model, model_from_json
from keras import metrics
import keras.backend as K
from keras.callbacks import Callback

import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense

from src.align_face_vae.dataset_celeba import CelebaDatasetSimple
import ipdb
import warnings
warnings.filterwarnings('ignore')
import argparse

data_folder     = '/Users/ost437/Documents/OneDrive/workspace/datasets/celebrity/'
parser = argparse.ArgumentParser()
parser.add_argument('--saved_folder',   default='./saved')
parser.add_argument('--image_path',     default=data_folder + 'data/images-dpmcrop-test/')
parser.add_argument('--metadata_path',  default=data_folder + 'data/list_attr_celeba.txt')
parser.add_argument('--img_rows',       default=256, type=int)
parser.add_argument('--img_cols',       default=256, type=int)
parser.add_argument('--img_chans',      default=3, type=int)
parser.add_argument('--batch_size',     default=64, type=int)
parser.add_argument('--load_model',     default=False, action='store_true')
args = parser.parse_args()

def CNN_Gender(img_rows, img_cols, img_chans):
    input_shape  = (img_rows, img_cols, img_chans)
    learning_rate = 0.001

    reg = keras.regularizers.l2(0.0)
    # Start building our model
    model = Sequential()

    model.add(Conv2D(8, (3, 3), input_shape=input_shape, strides=2, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3), strides=2, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(32, (3, 3), strides=2, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))

    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    return model

def run():
    dataset = CelebaDatasetSimple(args.image_path, args.metadata_path)
    dataset.get_batch(batch_size=32)
    learning_rate = 0.001

    K.clear_session()
    if args.load_model:
        print("Load json and create model")
        #vae, encoder, decoder = load_models()
    else:
        model = CNN_Gender(args.img_rows, args.img_cols, args.img_chans)

        # Specification of our optimizer, this one is a industry favorite
        optimizer = keras.optimizers.Adam(lr=learning_rate)

        print("Training model")
        #loss = K.mean()
        #model.add_loss(loss)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit_generator(
            dataset.classification_generator(is_validation=False),
            steps_per_epoch=10000,
            epochs=1,
            validation_data=dataset.classification_generator(is_validation=True),
            validation_steps=1,
            ##callbacks=[tb_viz, checkpoint],
            #class_weight={0: 3.6, 1: 1.0}
            # There are roughly 3.6x as many male faces in the dataset
            # so up-weight the female faces (class 0)
            # This will cut down on some of the network's
            # bias towards predicting males.
        )
        save_model(model, os.path.join(args.saved_folder, 'model'))

def save_model(model, save_folder):
    with open(os.path.join(save_folder, 'model_gender.json'), 'w') as json_file:
        json_file.write(model.to_json())
    model.save_weights(os.path.join(save_folder, 'model_gender.h5'))
    print('Saved models to disk')

if __name__ == '__main__':
    run()