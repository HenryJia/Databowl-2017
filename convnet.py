from __future__ import print_function
import os
import random
import time

import numpy as np
import scipy
#import theano.tensor as T

from keras.callbacks import ModelCheckpoint

import models_lib as ml
import data_utils as du

random.seed(1234)

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

#input_shape = (530, 490, 490) # use the maximum with batchsize 2
input_shape = (None, 1, 128, 128) # Resize everything to 128 x 128 x 128, first dimension is batchsize

print('Build Models')

model = ml.slicewise_convnet(input_shape)

model.summary()

print('Setup Generators')

#train_generator = du.generator(directory + 'stage1/', targets_train, input_shape, batch_size = 1)
#val_generator = du.generator(directory + 'stage1/', targets_val, input_shape, batch_size = 1)

train_generator = du.slicewise_generator(directory + 'stage1_processed/', targets_train, input_shape[2:])
val_generator = du.slicewise_generator(directory + 'stage1_processed/', targets_val, input_shape[2:])

t0 = time.time()
d = next(val_generator)
t1 = time.time()
print(t1 - t0)

checkpointer = ModelCheckpoint(
    filepath="weights.hdf5",
    verbose=1,
    save_best_only=True
)

print('Training')

model.fit_generator(train_generator, samples_per_epoch = 128, nb_epoch = 30, callbacks = [checkpointer],
                    validation_data = val_generator, nb_val_samples = 128, max_q_size = 20, nb_worker = 6, pickle_safe = True)
