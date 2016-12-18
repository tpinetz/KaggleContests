from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from lib.layer_blocks import inception_block



class Abstract_Model(object):
    def __init__(self, model, optimizer, loss, metrics):
        self.model = model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def getModel(self):
        return self.model


class MyTestModel(Abstract_Model):
    def __init__(self, optimizer, input_shape):
        model = Sequential()

        model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf', input_shape=(90, 160, 3)))
        model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf'))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

        model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
        model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

        model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
        model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

        model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
        model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(8))
        model.add(Activation('sigmoid'))


        Abstract_Model.__init__(self, model, optimizer, 'categorical_crossentropy', ['acc'])

class InceptionModel(Abstract_Model):
    def __init__(self, optimizer, input_shape):
        input = Input(shape=(90, 160, 3))

        cv1_1 = Convolution2D(32, 5, 5, border_mode='same', activation='relu')(input)
        cv1_2 = Convolution2D(32, 5, 5, border_mode='same', activation='relu')(cv1_1)
        
        pool1 = MaxPooling2D(pool_size=(2, 2))(cv1_2)

        cv2 = inception_block(pool1, 64, 3)
        pool2 = MaxPooling2D(pool_size=(2, 2))(cv2)

        cv3 = inception_block(pool2, 128, 5)
        pool3 = MaxPooling2D(pool_size=(2, 2))(cv3)

        cv4 = inception_block(pool3, 256, 3)
        pool4 = MaxPooling2D(pool_size=(2, 2))(cv4)

        fl = Flatten()(pool4)
        fc1 = Dense(256, activation='relu')(fl)
        drop1 = Dropout(0.5)(fc1)

        fc2 = Dense(64, activation='relu')(drop1)
        drop2 = Dropout(0.5)(fc2)

        fc3 = Dense(8)(drop2)
        result = Activation('softmax')(fc3)

        model = Model(input=input, output=result)

        Abstract_Model.__init__(self, model, optimizer, 'categorical_crossentropy', ['acc'])