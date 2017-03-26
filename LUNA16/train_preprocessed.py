from __future__ import print_function
import SimpleITK as sitk
import numpy as np
np.random.seed(1234)

from keras.callbacks import ModelCheckpoint

import csv
from glob import glob
import pandas as pd
import time

import pylab as plt
plt.ion()

import data_utils as du
import models_lib as ml

val_split = 50
min_value = -1000
max_value = 1000

print('Load Data')
data = np.load('/home/data/henry/Databowl-2017/LUNA16/data.npy')
targets = np.load('/home/data/henry/Databowl-2017/LUNA16/targets.npy')
binary_imgs = np.load('/home/data/henry/Databowl-2017/LUNA16/binary_imgs.npy')

#for i in range(2):
    #plt.figure(2 * i)
    #plt.imshow(data[i], cmap = 'gray')
    #plt.figure(2 * i + 1)
    #plt.imshow((targets * data)[i], cmap = 'gray')

shuffle = np.random.permutation(data.shape[0])
data = data[shuffle]
targets = targets[shuffle]

data = np.expand_dims(data, axis = 1)
targets = np.expand_dims(targets, axis = 1)

data_train = data[:-val_split]
targets_train = targets[:-val_split]
binary_imgs_train = binary_imgs[:-val_split]

data_val = data[-val_split:]
targets_val = targets[-val_split:]
binary_imgs_val = binary_imgs[-val_split:]

#raw_input('Press enter to continue')

print('Building model')

model = ml.slicewise_convnet()

checkpointer = ModelCheckpoint(filepath = "luna16.hdf5", verbose = 1, save_best_only = True)

print('Done, initialising training')
#model.load_weights('luna16.hdf5')
model.fit(data_train, targets_train, validation_data = (data_val, targets_val), nb_epoch = 100, batch_size = 2, callbacks = [checkpointer])

print('Done training, evaluate on validation set')
print(model.evaluate(data_val, targets_val))

out = model.predict(data_val)

plt.figure(1)
plt.imshow(data_val[0, 0], cmap = 'gray')
plt.figure(2)
plt.imshow(targets_val[0, 0], cmap = 'gray')
plt.figure(3)
plt.imshow(out[0, 0], cmap = 'gray')

# Postprocessing
out_masked = np.round(np.expand_dims(binary_imgs_val, axis = 1) * out)

print('Postprocess by masking using original image mask and rounding')
print('New dice coefficient ', np.mean(ml.np_dice_coef(targets_val, out_masked)))

plt.figure(4)
plt.imshow(out_masked[0, 0], cmap = 'gray')
