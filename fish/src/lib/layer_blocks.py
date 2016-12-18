from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import merge

def NConvolution(input, filter, kernel):
    cv = Convolution2D(filter, kernel, kernel, border_mode='same', activation='relu', init='he_uniform')(input)
    norm = BatchNormalization(mode=2, axis=1)(cv)
    return norm

def inception_block(input, filters, filter):
    cv1 = NConvolution(input, filters, filter)

    cv2 = NConvolution(input, int(filters / 2), 1)
    cv2 = NConvolution(cv2, filters, filter)

    cv3 = NConvolution(input, int(filters / 4), 1)
    cv3 = NConvolution(cv3, int(filters / 2), filter)
    cv3 = NConvolution(cv3, int(filters), filter)

    mr = merge([cv1, cv2, cv3], mode='sum')
    norm = BatchNormalization(axis=1, mode=2)(mr)
    return Activation('relu')(norm)