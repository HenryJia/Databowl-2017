import numpy as np
import scipy

from keras.models import Model, Sequential
from keras.layers import Input, Conv3D, Conv2D, Dense
from keras.layers import MaxPooling3D, MaxPooling2D
from keras.layers import GlobalMaxPooling3D, GlobalMaxPooling2D, GlobalMaxPooling1D
from keras.layers import GlobalAveragePooling3D, GlobalAveragePooling2D, GlobalAveragePooling1D
from keras.layers import Dropout, Reshape, Lambda, Flatten, TimeDistributed
from keras.layers import LSTM, SimpleRNN, Bidirectional
from keras.layers.advanced_activations import LeakyReLU, ELU

from keras.optimizers import Adam

import keras.backend as K

def np_dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def convnet3d(input_shape = (64, 64, 64)):

    model = Sequential()

    model.add(Conv3D(2, (3, 3, 3), padding = 'same', input_shape = (1, ) + input_shape))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Conv3D(4, (3, 3, 3), padding = 'same'))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Conv3D(8, (3, 3, 3), padding = 'same'))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Conv3D(16, (3, 3, 3), padding = 'same'))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Conv3D(32, (3, 3, 3), padding = 'same'))
    model.add(ELU())
    model.add(Conv3D(32, (3, 3, 3), padding = 'same'))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Conv3D(32, (3, 3, 3), padding = 'same'))
    model.add(ELU())
    model.add(Conv3D(32, (3, 3, 3), padding = 'same'))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Conv3D(32, (3, 3, 3), padding = 'same'))
    model.add(ELU())
    model.add(Conv3D(32, (3, 3, 3), padding = 'same'))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Conv3D(32, (3, 3, 3), padding = 'same'))
    model.add(ELU())
    model.add(Conv3D(32, (3, 3, 3), padding = 'same'))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2, 2)))

    #model.add(GlobalMaxPooling3D())

    model.add(Flatten())

    model.add(Dense(128))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Dense(2, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    return model

def slicewise_convnet(input_shape = (512, 512)):

    model = Sequential()

    #model.add(Lambda(lambda x: x.dimshuffle((0, 1, 'x', 2, 3)), lambda input_shape: input_shape[:2] + (1, ) + input_shape[2:],
                     #batch_input_shape = (1, ) + input_shape))
    #print(model.output_shape)
    model.add(Conv2D(2, (3, 3), padding = 'same', input_shape = (1, ) + input_shape))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(4, (3, 3), padding = 'same'))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(8, (3, 3), padding = 'same'))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(16, (3, 3), padding = 'same'))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), padding = 'same'))
    model.add(ELU())
    model.add(Conv2D(32, (3, 3), padding = 'same'))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), padding = 'same'))
    model.add(ELU())
    model.add(Conv2D(32, (3, 3), padding = 'same'))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), padding = 'same'))
    model.add(ELU())
    model.add(Conv2D(32, (3, 3), padding = 'same'))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), padding = 'same'))
    model.add(ELU())
    model.add(Conv2D(32, (3, 3), padding = 'same'))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(ELU())
    model.add(Dense(32))
    model.add(ELU())

    model.add(Dense(1, activation = 'sigmoid'))

    #model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 1e-5))
    model.compile(loss = dice_coef_loss, optimizer = Adam(lr = 1e-5), metrics = [dice_coef])

    return model
