from __future__ import print_function
import os
import time
from multiprocessing import Pool
p = Pool(6)
import dicom

import pandas as pd
import numpy as np
import scipy
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

#model_luna = lunaML.slicewise_convnet()
#model_luna.load_weights('LUNA16/luna16.hdf5')

model = ml.slicewise_convnet()

#def preprocess(img, df, k):
    ##img = du.get_pixels_hu(du.load_scan(directory + 'stage1/' + df['id'].iloc[k]))
    #img_segment = np.zeros_like(img)
    #mask = np.zeros_like(img)

    ## Parallelrise this bit
    #segment_out = p.map(lunaDU.get_segmented_lungs, [img[i] for i in range(img.shape[0])])

    #for i in range(img.shape[0]):
        #img_segment[i] = segment_out[i][0]
        #mask[i] = mask[i][1]

    ##for i in range(img.shape[0]):
        ##img_segment[i], mask[i] = lunaDU.get_segmented_lungs(img[i])

    #out = np.squeeze(model_luna.predict(np.expand_dims(img_segment, axis = 1)))
    #out = np.round(out) * mask * img

    #return out

def preprocess_loadmask(img, df, k):
    mask = np.load(mask_directory + 'stage1_nodules/' + df['id'].iloc[k] + '.npy')
    out = np.round(mask) * img

    # Compress by removing all 0 slices to save memory
    out = out[np.sum(out, axis = (1, 2)) != 0]

    # Sample only batch_size
    np.random.shuffle(out)
    out = out[:batch_size]

    return out

print('Setup Generators')

train_generator = du.custom_generator(directory, targets_train, preprocess_loadmask)
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
