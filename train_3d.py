from __future__ import print_function
import os
import time
from multiprocessing import Pool
p = Pool(6)
import dicom

import pandas as pd
import numpy as np
from scipy.ndimage.interpolation import zoom
from skimage import measure
np.random.seed(1234)

from keras.callbacks import ModelCheckpoint
from keras.utils.generic_utils import Progbar

import LUNA16.models_lib as lunaML
import LUNA16.data_utils as lunaDU
import data_utils as du
import models_lib as ml

import pylab as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.ion()

val_split = 50
batch_size = 4

print('Load Targets')

directory = '/home/data/henry/Databowl-2017/'
mask_directory = './'
df = pd.read_csv(directory + 'stage1_labels.csv')
df = df.sample(frac = 1)

targets_train = df.iloc[:-val_split]
targets_val = df.iloc[-val_split:]

print('Total data ', len(df))

print('Build Models')

model = ml.convnet3d()

def preprocess_loadmask(img, df, k):
    mask = np.round(np.load(mask_directory + 'stage1_nodules/' + df['id'].iloc[k] + '.npy'))
    #img_nodules = mask * img

    # Now use label to crop out each nodule and resize them to 32 x 32 x 32
    mask_labels = label(mask)
    mask_nodules, mask_counts = np.unique(mask_labels, return_counts = True)
    mask_nodules = mask_nodules[mask_counts >= (3 ** 3)]

    img_cropped = np.zeros(img.shape[0], 32, 32, 32)
    for i in xrange(1, mask_nodules_new.shape[0]):
        selection = (mask_nodules_new[i] == mask_labels).astype(bool)
        # Crop to only the parts we need
        zs, xs, ys = np.indices(selection.shape)
        zi, xi, yi = zs[selection], xs[selection], ys[selection]
        cropped = img[np.min(zi):np.max(zi) + 1, np.min(xi):np.max(xi) + 1, np.min(yi):np.min(yi) + 1]
        img_cropped[i] = zoom(cropped, np.array(cropped.shape) / 32.0, order = 1)

    # Sample only batch_size
    np.random.shuffle(img_cropped)
    img_cropped = img_cropped[:batch_size]

    return img_cropped

print('Setup Generators')

train_generator = du.custom_generator(directory, targets_train, preprocess_loadmask, softmax = True)
val_generator = du.custom_generator(directory, targets_val, preprocess_loadmask)

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

model.fit_generator(train_generator, steps_per_epoch = 256, epochs = 30, callbacks = [checkpointer],
                    validation_data = val_generator, validation_steps = 32, max_q_size = 20, workers = 6, pickle_safe = True)

#for k in xrange(20):
    #pb = Progbar(len(targets_train) + 1)
    #val_loss = 0.0
    #for i in xrange(len(targets_train)):
        #data = np.expand_dims(preprocess(targets_train, i), axis = 0)
        #loss = model.train_on_batch(data, np.array(targets_train['cancer'].iloc[i]).reshape((1, 1)))
        #pb.update(i, values = [('Train loss', loss), ('Validation loss', val_loss)])

    #for i in xrange(val_split):
        #val_loss += model.test_on_batch(np.expand_dims(preprocess(targets_val, i), axis = 0))
    #val_loss /= val_split

    #pb.update(i + 1, values = [('Train loss', loss), ('Validation loss', val_loss)])
