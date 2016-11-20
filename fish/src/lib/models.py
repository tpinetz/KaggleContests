from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

class Abstract_Model(object):
    def __init__(self, model, optimizer, loss, metrics):
        self.model = model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def getModel(self):
        return self.model


class MyTestModel(Abstract_Model):
    def __init__(self, optimizer, input_shape):
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(3,) + input_shape))
        model.add(Convolution2D(4, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(4, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(8, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(8, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8, activation='softmax'))

        Abstract_Model.__init__(self, model, optimizer, 'categorical_crossentropy', ['acc'])
