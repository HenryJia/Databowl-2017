from __future__ import print_function
import os
import random
import time
from collections import OrderedDict
import dicom

import numpy as np
import scipy
#import theano.tensor as T

from keras.models import Model, Sequential
from keras.layers import Input, Convolution3D, Convolution2D, MaxPooling3D, MaxPooling2D, Dense
from keras.layers import Dropout, Reshape, Lambda, Flatten
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.callbacks import ModelCheckpoint

from preprocess import normalize, zero_center
import models_lib as ml

val_slice = 100 # First 100 patients for validation

directory = '/home/data/henry/Databowl-2017/'

print('Load Targets')

targets_train = {}
targets_val = {}

with open(directory + 'stage1_labels.csv') as f:
    for i, line in enumerate(f):
        if i > 0:
            if i < (val_slice + 1):
                patient_id, cancer = line.split(',')
                targets_val[str(patient_id)] = int(cancer)
            else:
                patient_id, cancer = line.split(',')
                targets_train[str(patient_id)] = int(cancer)

#input_shape = (1, 530, 490, 490) # use the maximum with batchsize 2
input_shape = (1, 128, 128, 128) # Resize everything to 128 x 128 x 128

class generator(object):
    def __init__(self, folder, targets, shape):
        assert type(targets) is dict or type(targets) is OrderedDict, 'Targets must be a dictionary or ordered dictionary'

        self.folder = folder
        self.targets = targets
        self.shape = shape

        random.seed(1234)

    def __next__(self):
        x = np.zeros(self.shape, dtype = np.float32)
        y = np.zeros((self.shape[0], 1), dtype = np.uint8)

        for i in range(self.shape[0]):
            index = random.randint(0, len(self.targets.keys()) - 1)

            data = np.load(self.folder + list(self.targets.keys())[index] + '/img_segment_fill.npy').astype(np.float32)
            data = zero_center(normalize(data))
            data = scipy.ndimage.interpolation.zoom(data, self.shape[1:] / np.array(data.shape, dtype = np.float32), order = 1)

            #x[i, :data.shape[0], :data.shape[1], :data.shape[2]] = data
            x[i] = data

            y[i] = self.targets[list(self.targets.keys())[index]]

        x = np.expand_dims(x, axis = 1)

        return x, y

    # For python2 compatibility
    def next(self):
        return self.__next__()

print('Build Models')

model = ml.convnet3d(input_shape)

print(model.summary())

print('Setup Generators')

train_generator = generator(directory + 'stage1/', targets_train, input_shape)
val_generator = generator(directory + 'stage1/', targets_val, input_shape)

t0 = time.time()
d = next(train_generator)
t1 = time.time()
print(t1 - t0)

checkpointer = ModelCheckpoint(
    filepath="weights.hdf5",
    verbose=1,
    save_best_only=True
)

print('Training')

model.fit_generator(train_generator, samples_per_epoch = 100, nb_epoch = 200, callbacks = [checkpointer],
                    validation_data = val_generator, nb_val_samples = 100, max_q_size = 10)
