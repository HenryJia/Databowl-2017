from __future__ import print_function

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, TimeDistributed, Lambda, concatenate, Activation, Flatten
from keras.layers import Conv2D, Conv3D
from keras.layers import AveragePooling2D, AveragePooling3D, MaxPooling2D, MaxPooling3D
from keras.layers import UpSampling2D, UpSampling3D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.advanced_activations import ELU

from keras.optimizers import Adam

import keras.backend as K

# Not sure if theano/tensorflow/Keras softmax will do it along axis 1, so we write our own
def semantic_softmax(x):
    e_x = K.exp(x - K.max(x, axis = 1, keepdims = True))
    out = e_x / K.sum(e_x, axis = 1, keepdims = True)

    return out

def weighted_categorical_crossentropy(y_true, y_pred):
    l = -y_true * K.log(K.clip(y_pred, 1e-8, 1 - 1e-8))
    out = l[:, :, 0] + l[:, :, 1] # Hard code this, since we're only ever gonna have 2 categories
    return out

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

def slicewise_convnet(input_shape = (512, 512)):
    input_var = Input(shape = (1, ) + input_shape)

    blocks = []

    conv = Conv2D(4, (3, 3), padding = 'same')(input_var)
    conv = ELU()(conv)
    blocks += [conv]

    conv = MaxPooling2D((2, 2))(conv) # 256
    conv = Conv2D(8, (3, 3), padding = 'same')(conv)
    conv = ELU()(conv)
    blocks += [conv]

    conv = MaxPooling2D((2, 2))(conv) # 128
    conv = Conv2D(16, (3, 3), padding = 'same')(conv)
    conv = ELU()(conv)
    blocks += [conv]

    conv = MaxPooling2D((2, 2))(conv) # 64
    conv = Conv2D(32, (3, 3), padding = 'same')(conv)
    conv = ELU()(conv)
    blocks += [conv]

    conv = MaxPooling2D((2, 2))(conv) # 32
    conv = Conv2D(64, (3, 3), padding = 'same')(conv)
    conv = ELU()(conv)
    blocks += [conv]

    conv = MaxPooling2D((2, 2))(conv) # 16
    conv = Conv2D(128, (3, 3), padding = 'same')(conv)
    conv = ELU()(conv)

    deconv = conv

    deconv = concatenate([UpSampling2D((2, 2))(deconv), blocks[-1]], axis = 1) # 32
    deconv = Conv2D(64, (3, 3), padding = 'same')(deconv)
    deconv = ELU()(deconv)

    deconv = concatenate([UpSampling2D((2, 2))(deconv), blocks[-2]], axis = 1) # 64
    deconv = Conv2D(32, (3, 3), padding = 'same')(deconv)
    deconv = ELU()(deconv)

    deconv = concatenate([UpSampling2D((2, 2))(deconv), blocks[-3]], axis = 1) # 128
    deconv = Conv2D(16, (3, 3), padding = 'same')(deconv)
    deconv = ELU()(deconv)

    deconv = concatenate([UpSampling2D((2, 2))(deconv), blocks[-4]], axis = 1) # 256
    deconv = Conv2D(8, (3, 3), padding = 'same')(deconv)
    deconv = ELU()(deconv)

    deconv = concatenate([UpSampling2D((2, 2))(deconv), blocks[-5]], axis = 1) # 512 
    deconv = Conv2D(4, (3, 3), padding = 'same')(deconv)
    deconv = ELU()(deconv)

    deconv = Conv2D(1, (1, 1), padding = 'same', activation = 'sigmoid')(deconv)

    model = Model(inputs = input_var, outputs = deconv)
    model.compile(loss = dice_coef_loss, optimizer = Adam(lr=1.0e-5), metrics = [dice_coef])

    return model

def convnet_3d(input_shape = (64, 64, 64)):
    model = Sequential()

    model.add(Conv3D(4, (3, 3, 3), padding = 'same', input_shape = (1, ) + input_shape))
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
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Conv3D(64, (3, 3, 3), padding = 'same'))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Conv3D(64, (3, 3, 3), padding = 'same'))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2, 2)))

    # Equivalent of Dense, but for the sake of invariance to dimensions at runtime, we use conv
    model.add(Conv3D(128, 1, 1, 1, padding = 'same'))
    model.add(ELU())
    model.add(Conv3D(2, 1, 1, 1, padding = 'same'))

    model_flat = model
    model_flat.add(Flatten())
    model_flat.add(Activation('softmax'))

    return model, model_flat
