import numpy as np
import scipy
#import theano.tensor as T

from keras.models import Model, Sequential
from keras.layers import Input, Convolution3D, Convolution2D, Dense
from keras.layers import MaxPooling3D, MaxPooling2D
from keras.layers import GlobalMaxPooling3D, GlobalMaxPooling2D, GlobalMaxPooling1D
from keras.layers import GlobalAveragePooling3D, GlobalAveragePooling2D, GlobalAveragePooling1D
from keras.layers import Dropout, Reshape, Lambda, Flatten, TimeDistributed
from keras.layers import LSTM, SimpleRNN, Bidirectional
from keras.layers.advanced_activations import LeakyReLU, ELU

def convnet3d(input_shape = (64, 64, 64)):

    model = Sequential()

    model.add(Convolution3D(2, 3, 3, 3, border_mode = 'same', input_shape = (1, ) + input_shape))
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

def slicewise_convnet(input_shape = (None, 1, 64, 64)):

    model = Sequential()

    model.add(TimeDistributed(Convolution2D(2, 3, 3, border_mode = 'same'), input_shape = input_shape))
    model.add(ELU())
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Convolution2D(4, 3, 3, border_mode = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Convolution2D(8, 3, 3, border_mode = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Convolution2D(16, 3, 3, border_mode = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode = 'same')))
    model.add(ELU())
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Flatten()))

    #model.add(GlobalMaxPooling1D())
    #model.add(GlobalAveragePooling1D())
    model.add(Bidirectional(SimpleRNN(32, return_sequences = False, stateful = False, consume_less = 'gpu'), merge_mode = 'concat'))

    #model.add(Dense(32))
    #model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Dense(2, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return model
