from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Input, TimeDistributed, Lambda, merge
from keras.layers import Convolution2D, Convolution3D
from keras.layers import AveragePooling2D, AveragePooling3D, MaxPooling2D, MaxPooling3D
from keras.layers import UpSampling2D, UpSampling3D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.advanced_activations import ELU

import keras.backend as K

# Not sure if theano/tensorflow/Keras softmax will do it along axis 1, so we write our own
def semantic_softmax(x):
    e_x = K.exp(x - K.max(x, axis = 1, keepdims = True))
    out = e_x / K.sum(e_x, axis = 1, keepdims = True)

    return out

def weighted_categorical_crossentropy(y_true, y_pred):
    l = -y_true * K.log(K.clip(y_pred, 1e-8, 1 - 1e-8))
    #out = (1.0 / (2.5e5 + 1.0)) * l[:, :, 0] + (2.5e5 / (2.5e5 + 1)) * l[:, :, 1] # Hard code this, since we're only ever gonna have 2 categories
    #out = l[:, :, 0] + 2e5 * l[:, :, 1] # Hard code this, since we're only ever gonna have 2 categories
    out = l[:, :, 0] + l[:, :, 1] # Hard code this, since we're only ever gonna have 2 categories
    return out

def slicewise_convnet(input_shape = (512, 512)):
    input_var = Input(batch_shape = (1, None) + input_shape)

    blocks = []

    conv = Lambda(lambda x: K.expand_dims(x, dim = 2), lambda shape: shape[:2] + (1, ) + shape[2:])(input_var)
    conv = TimeDistributed(Convolution2D(2, 3, 3, border_mode = 'same'))(conv)
    conv = ELU()(conv)
    blocks += [conv]

    conv = TimeDistributed(MaxPooling2D((2, 2)))(conv) # 256
    conv = TimeDistributed(Convolution2D(4, 3, 3, border_mode = 'same'))(conv)
    conv = ELU()(conv)
    blocks += [conv]

    conv = TimeDistributed(MaxPooling2D((2, 2)))(conv) # 128
    conv = TimeDistributed(Convolution2D(8, 3, 3, border_mode = 'same'))(conv)
    conv = ELU()(conv)
    blocks += [conv]

    conv = TimeDistributed(MaxPooling2D((2, 2)))(conv) # 64
    conv = TimeDistributed(Convolution2D(16, 3, 3, border_mode = 'same'))(conv)
    conv = ELU()(conv)
    blocks += [conv]

    conv = TimeDistributed(MaxPooling2D((2, 2)))(conv) # 32
    conv = TimeDistributed(Convolution2D(32, 3, 3, border_mode = 'same'))(conv)
    conv = ELU()(conv)
    blocks += [conv]

    conv = TimeDistributed(MaxPooling2D((2, 2)))(conv) # 16
    conv = TimeDistributed(Convolution2D(64, 3, 3, border_mode = 'same'))(conv)
    conv = ELU()(conv)

    deconv = conv

    deconv = merge([TimeDistributed(UpSampling2D((2, 2)))(deconv), blocks[-1]], mode = 'concat', concat_axis = 2) # 32
    deconv = TimeDistributed(Convolution2D(32, 3, 3, border_mode = 'same'))(deconv)
    deconv = ELU()(deconv)

    deconv = merge([TimeDistributed(UpSampling2D((2, 2)))(deconv), blocks[-2]], mode = 'concat', concat_axis = 2) # 64
    deconv = TimeDistributed(Convolution2D(16, 3, 3, border_mode = 'same'))(deconv)
    deconv = ELU()(deconv)

    deconv = merge([TimeDistributed(UpSampling2D((2, 2)))(deconv), blocks[-3]], mode = 'concat', concat_axis = 2) # 128
    deconv = TimeDistributed(Convolution2D(8, 3, 3, border_mode = 'same'))(deconv)
    deconv = ELU()(deconv)

    deconv = merge([TimeDistributed(UpSampling2D((2, 2)))(deconv), blocks[-4]], mode = 'concat', concat_axis = 2) # 256
    deconv = TimeDistributed(Convolution2D(4, 3, 3, border_mode = 'same'))(deconv)
    deconv = ELU()(deconv)

    deconv = merge([TimeDistributed(UpSampling2D((2, 2)))(deconv), blocks[-5]], mode = 'concat', concat_axis = 2) # 512 
    deconv = TimeDistributed(Convolution2D(2, 3, 3, border_mode = 'same'))(deconv)
    deconv = ELU()(deconv)

    deconv = TimeDistributed(Convolution2D(2, 3, 3, border_mode = 'same', activation = semantic_softmax))(deconv)

    model = Model(input = input_var, output = deconv)
    model.compile(loss = weighted_categorical_crossentropy, optimizer = 'adam', metrics = ['accuracy'])

    return model

def convnet_3d(input_shape = (64, 64, 64)):
    model = Sequential()

    model.add(Convolution3D(4, 3, 3, 3, border_mode = 'same', input_shape = (1, ) + input_shape))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2)))

    model.add(Convolution2D(8, 3, 3, 3, border_mode = 'same'))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2)))

    model.add(Convolution3D(16, 3, 3, 3, border_mode = 'same'))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2)))

    model.add(Convolution3D(32, 3, 3, 3, border_mode = 'same'))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2)))

    model.add(Convolution3D(64, 3, 3, 3, border_mode = 'same'))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2)))

    model.add(Convolution3D(64, 3, 3, 3, border_mode = 'same'))
    model.add(ELU())
    model.add(MaxPooling3D((2, 2)))

    # Equivalent of Dense, but for the sake of invariance to dimensions at runtime, we use conv
    model.add(Convolution3D(128, 1, 1, 1, border_mode = 'same'))
    model.add(ELU())
    model.add(Convolution3D(2, 1, 1, 1, border_mode = 'same'))

    model_flat = model
    model_flat.add(Flatten())

    return model
