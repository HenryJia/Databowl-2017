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

def slicewise_convnet(input_shape = (None, 512, 512)):

    model = Sequential()

    model.add(Lambda(lambda x: x.dimshuffle((0, 1, 'x', 2, 3)), lambda input_shape: input_shape[:2] + (1, ) + input_shape[2:],
                     batch_input_shape = (1, ) + input_shape))
    #print(model.output_shape)
    model.add(TimeDistributed(Conv2D(2, (3, 3), padding = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(4, (3, 3), padding = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(8, (3, 3), padding = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(16, (3, 3), padding = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Flatten()))

    model.add(GlobalAveragePooling1D())

    model.add(Dense(32))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 1e-5))

    return model
