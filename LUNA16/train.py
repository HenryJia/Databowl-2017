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

print('Setting up generators')

directory = '/home/data/henry/Databowl-2017/LUNA16/'

file_list = glob(directory + '**/*.mhd')

df_node = pd.read_csv(directory + "CSVFILES/annotations.csv")
df_node = df_node.dropna()

train_generator = du.generator(directory, file_list, df_node, (512, 512), 4, 50, False)
val_generator = du.generator(directory, file_list, df_node, (512, 512), 4, 50, True)

d = val_generator.next()

img, mask = d
for i in range(min(img.shape[1], 2)):
    plt.figure(2 * i)
    plt.imshow(img[i, 0], cmap = 'gray')
    plt.figure(2 * i + 1)
    plt.imshow((mask * img)[i, 0], cmap = 'gray')

raw_input('Press enter to continue')

print('Building model')

model = ml.slicewise_convnet()

checkpointer = ModelCheckpoint(filepath = "luna16.hdf5", verbose = 1, save_best_only = True)

print('Done, initialising training')
model.load_weights('luna16.hdf5')
model.fit_generator(train_generator, steps_per_epoch = 800 / 4, epochs = 100,
                    validation_data = val_generator, validation_steps = 16,
                    max_q_size = 100, workers = 6, pickle_safe = True,
                    callbacks = [checkpointer])
#out = model.predict(d[0])

#plt.figure(1)
#plt.imshow(d[1][0, 0, 1])
#plt.figure(2)
#plt.imshow(out[0, 0, 1])
