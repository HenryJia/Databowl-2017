from __future__ import print_function
import SimpleITK as sitk
import numpy as np
np.random.seed(1234)

from keras.callbacks import ModelCheckpoint

import csv
from glob import glob
import pandas as pd

import pylab as plt
plt.ion()

import data_utils as du
import models_lib as ml

directory = '/home/data/henry/Databowl-2017/LUNA16/'

file_list = glob(directory + '**/*.mhd')

df_node = pd.read_csv(directory + "CSVFILES/annotations.csv")
df_node = df_node.dropna()

train_generator = du.generator(directory, file_list, df_node, (None, 512, 512), True, 1, 50, False)
val_generator = du.generator(directory, file_list, df_node, (None, 512, 512), True, 1, 50, True)

d = val_generator.next()
#print(d[2])

#plt.figure(1)
#plt.imshow(d[0][0, 257], cmap='gray')
#plt.imshow(d[0][0, 3], cmap='gray')

#plt.figure(2)
#plt.imshow(d[0][0, 257] * d[1][0, 257])
#plt.imshow(d[0][0, 3] * d[1][0, 3])

#plt.figure(3)
#plt.imshow(d[0][0, 241] * d[1][0, 241])

#plt.figure(4)
#plt.imshow(d[0][0, 230] * d[1][0, 230])

print('Building model')

model = ml.slicewise_convnet()

checkpointer = ModelCheckpoint(filepath = "luna16.hdf5", verbose = 1, save_best_only = True)

print('Done, initialising training')

model.fit_generator(train_generator, samples_per_epoch = 800, nb_epoch = 1,
                    validation_data = val_generator, nb_val_samples = 50,
                    max_q_size = 100,# nb_worker = 6, pickle_safe = True,
                    callbacks = [checkpointer])

#out = model.predict(d[0])

#plt.figure(1)
#plt.imshow(d[1][0, 0, 1])
#plt.figure(2)
#plt.imshow(out[0, 0, 1])
