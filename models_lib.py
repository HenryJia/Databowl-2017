import numpy as np
import scipy
#import theano.tensor as T

from keras.models import Model, Sequential
from keras.layers import Input, Convolution3D, Convolution2D, MaxPooling3D, MaxPooling2D, Dense
from keras.layers import Dropout, Reshape, Lambda, Flatten
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.callbacks import ModelCheckpoint

def convnet3d(input_shape = (1, 64, 64, 64)):

    model = Sequential()

    model.add(Convolution3D(2, 3, 3, 3, border_mode = 'same', input_shape = (1, ) + input_shape[1:]))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Convolution3D(4, 3, 3, 3, border_mode = 'same'))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Convolution3D(8, 3, 3, 3, border_mode = 'same'))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Convolution3D(16, 3, 3, 3, border_mode = 'same'))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Convolution3D(32, 3, 3, 3, border_mode = 'same'))
    model.add(ELU())
    model.add(Convolution3D(32, 3, 3, 3, border_mode = 'same'))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Convolution3D(32, 3, 3, 3, border_mode = 'same'))
    model.add(ELU())
    model.add(Convolution3D(32, 3, 3, 3, border_mode = 'same'))
    model.add(ELU())

    model.add(Flatten())

    model.add(Dense(200))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(200))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

    return model
