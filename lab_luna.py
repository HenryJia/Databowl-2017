from __future__ import print_function
import os
import random
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

import LUNA16.models_lib as lunaML
import LUNA16.data_utils as lunaDU
import data_utils as du

import pylab as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.ion()

model_conv = lunaML.slicewise_convnet()
model_conv.load_weights('LUNA16/luna16.hdf5')

directory = '/home/data/henry/Databowl-2017/'
df = pd.read_csv(directory + 'stage1_labels.csv')


print('Iterate through data')

def cycle(df, k):
    img = du.get_pixels_hu(du.load_scan(directory + 'stage1/' + df['id'].iloc[k]))

    img_segment = np.zeros_like(img)
    mask = np.zeros_like(img)

    # Parallelrise this bit
    segment_out = p.map(lunaDU.get_segmented_lungs, [img[i] for i in range(img.shape[0])])

    for i in range(img.shape[0]):
        img_segment[i] = segment_out[i][0]
        mask[i] = mask[i][1]

    #for i in range(img.shape[0]):
        #img_segment[i], mask[i] = lunaDU.get_segmented_lungs(img[i])

    out = np.squeeze(model_conv.predict(np.expand_dims(img_segment, axis = 1)))

    return out

healthy_df = df[df['cancer'] == 0].sample(50)
cancer_df = df[df['cancer'] == 1].sample(50)

healthy_stats = np.zeros((len(healthy_df), 2))
cancer_stats = np.zeros((len(cancer_df), 2))

healthy_slice = np.zeros((len(healthy_df), 512, 512))
cancer_slice = np.zeros((len(healthy_df), 512, 512))

print('Healthy patients')

for k in xrange(len(healthy_df)):
    if k % 10 == 0:
        print(k)
    out_healthy = np.round(cycle(healthy_df, k))
    healthy_slice[k] = out_healthy[np.argmax(np.sum(out_healthy, axis = (1, 2)))]
    healthy_stats[k] = np.unique(out_healthy, return_counts = True)[1] / np.prod(out_healthy.shape).astype(np.float32)

print('Cancerous patients')

for k in xrange(len(cancer_df)):
    if k % 10 == 0:
        print(k)
    out_cancer = np.round(cycle(cancer_df, k))
    cancer_slice[k] = out_cancer[np.argmax(np.sum(out_cancer, axis = (1, 2)))]
    cancer_stats[k] = np.unique(out_cancer, return_counts = True)[1] / np.prod(out_cancer.shape).astype(np.float32)

print('Done, saving')

np.save('healthy_stats50.npy', healthy_stats)
np.save('cancer_stats50.npy', cancer_stats)

#du.plot_3d(out, 0)
plt.figure(1)
plt.hist(healthy_stats[:, 1], 25, normed = True, facecolor = 'green', alpha = 0.75)
plt.hist(cancer_stats[:, 1], 25, normed = True, facecolor = 'red', alpha = 0.75)


