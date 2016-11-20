# Base Script by ZFTurbo: https://kaggle.com/zfturbo
__author__ = 'Thomas Pinetz'


import numpy as np
np.random.seed(2016)

import os
import glob
import scipy.misc
import datetime
import pandas as pd
import time
import warnings
import json
warnings.filterwarnings("ignore")

from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version
from lib.fish_dataset import FishDataset
import lib.models as models


def read_and_normalize_train_data():
    train_data, train_target, train_id = load_train()

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, 8)

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def run_script():
    # input image dimensions
    batch_size = 16
    nb_epoch = 30
    random_state = 51

    train_set = FishDataset('../train', 'train')
    val_set = FishDataset('../train', 'val')

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model = models.MyTestModel(sgd, (32, 32))
   
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, verbose=0),
    ]
    history = model.fit(train_set.data, np_utils.to_categorical(train_set.labels),
              batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=2, 
              validation_data=(val_set.data, np_utils.to_categorical(val_set.labels)),
              callbacks=callbacks)

    print('Generating model dump in best_model.h5')
    model.save('best_model.h5')

    return history, model


def run_submission(history, models):
    batch_size = 16
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)

    test = FishDataset('../test', 'test')
    test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
    
    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + history.history['loss'] \
                + '_folds_' + str(nfolds)
    create_submission(test.data, test.ids, info_string)


if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    history, models = run_script()
    run_submission(history, models)
    