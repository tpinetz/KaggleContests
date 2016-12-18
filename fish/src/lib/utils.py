import scipy.misc
import pandas as pd
import numpy as np
import datetime
import os, glob
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def get_im(path):
    img = scipy.misc.imread(path)
    img = scipy.misc.imresize(img, (90, 160))

    return img

def create_submission(predictions, ids, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(ids, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)


def get_train_data(fdir, classes):
    data = []
    labels = []
    ids = []
    for fld in classes:
        index = classes.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join(fdir, fld, '*.jpg')
        files = glob.glob(path)

        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im(fl)
            data.append(img.tolist())
            labels.append(index)
            ids.append(flbase)
    
    return (np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32), ids)


def get_test_data(fdir):
    data = []
    ids = []
    path = os.path.join(fdir, '*.jpg')
    files = glob.glob(path)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl)
        data.append(img.tolist())
        ids.append(flbase)    
    return (np.array(data, dtype=np.float32), ids)

def to_categorical(y):
        # One Hot Encoding Labels
    y = LabelEncoder().fit_transform(y)
    return np_utils.to_categorical(y)

def normalize_per_channel(X):
    X[:,:,1] = (X[:,:,1] - np.mean(X[:,:,1])) / np.std(X[:,:,1])
    X[:,:,2] = (X[:,:,2] - np.mean(X[:,:,2])) / np.std(X[:,:,2])
    X[:,:,3] = (X[:,:,3] - np.mean(X[:,:,3])) / np.std(X[:,:,3])

    return X