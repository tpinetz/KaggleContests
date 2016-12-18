# Base Script by ZFTurbo: https://kaggle.com/zfturbo
__author__ = 'Thomas Pinetz'


import numpy as np
np.random.seed(2016)

import os
import time
import json

from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras.preprocessing.image import ImageDataGenerator
from keras import __version__ as keras_version
from lib.fish_dataset import FishDataset
import lib.models as models
from lib.utils import (create_submission, get_train_data, 
                       to_categorical, normalize_per_channel,
                       get_test_data)
from sklearn.model_selection import train_test_split

classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def run_script(path):
    # input image dimensions
    batch_size = 16
    nb_epoch = 100
    random_state = 51

    # Creating Model
    print('Instancing model')
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model = models.InceptionModel(Adam(lr=0.0045), (32, 32)).getModel()

    X, y, _ = get_train_data(path + '/../train', classes)
    y = to_categorical(y)

    print('Finished Loading')
    X = normalize_per_channel(X)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                    test_size=0.2, random_state=23, 
                                                    stratify=y)
   
    datagen.fit(X_train)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
    ]
    
    print('Start training')

    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
              samples_per_epoch=len(X_train), nb_epoch=nb_epoch, verbose=2, 
              validation_data=(X_valid, y_valid),
              callbacks=callbacks)

    print('Generating model dump in best_model.h5')
    model.save('best_model.h5')

    return history, model


def run_submission(path, history, model):
    batch_size = 16
    num_fold = 0
    yfull_test = []
    test_id = []

    X, ids = get_test_data(path + '/../test')
    X = normalize_per_channel(X)
    test_prediction = model.predict(X, batch_size=batch_size, verbose=2)

    create_submission(test_prediction, ids, str(np.min(history.history['loss'])))


if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    path = os.path.dirname(os.path.realpath(__file__))

    history, model = run_script(path)
    run_submission(path, history, model)
    