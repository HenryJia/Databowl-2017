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

import pylab as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.ion()

model_conv = lunaML.slicewise_convnet()
model_conv.load_weights('LUNA16/luna16.hdf5')

directory = '/home/data/henry/Databowl-2017/'
out_directory = './'
df = pd.read_csv(directory + 'stage1_labels.csv')


print('Iterate through data ', len(df))

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

if not os.path.exists(out_directory + 'stage1_nodules/'):
    os.makedirs(out_directory + 'stage1_nodules/')

pb = Progbar(len(df))
for k in xrange(len(df)):

    path = out_directory + 'stage1_nodules/' + df['id'].iloc[k] + '.npy'

    if not os.path.isfile(path):
        out = np.round(cycle(df, k)).astype(np.bool)
        np.save(path, out)

    pb.update(k, force = True)

print('All done')
