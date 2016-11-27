# Base Script by ZFTurbo: https://kaggle.com/zfturbo
__author__ = 'Thomas Pinetz'


import numpy as np
np.random.seed(2016)

import os
import time
import json

from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version
from lib.fish_dataset import FishDataset
import lib.models as models
from lib.utils import create_submission


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

    train_set = FishDataset(path + '/../train', 'train')
    val_set = FishDataset(path + '/../train', 'val')

    train_set.data /= 255
    val_set.data /= 255

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model = models.MyTestModel(sgd, (32, 32)).getModel()
   
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, verbose=0),
    ]
    
    history = model.fit(train_set.data, np_utils.to_categorical(train_set.labels, train_set.nclasses()),
              batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=2, 
              validation_data=(val_set.data, np_utils.to_categorical(val_set.labels, val_set.nclasses())),
              callbacks=callbacks)

    print('Generating model dump in best_model.h5')
    model.save('best_model.h5')

    return history, model


def run_submission(path, history, model):
    batch_size = 16
    num_fold = 0
    yfull_test = []
    test_id = []

    test = FishDataset(path + '/../test', 'test')
    test_prediction = model.predict(test.data, batch_size=batch_size, verbose=2)

    create_submission(test_prediction, test.ids, str(np.min(history.history['loss'])))


if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    path = os.path.dirname(os.path.realpath(__file__))

    history, model = run_script(path)
    run_submission(path, history, model)
    