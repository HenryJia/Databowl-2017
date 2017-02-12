from __future__ import print_function
import os
import random
import time
from collections import OrderedDict
import dicom

import numpy as np
import scipy
#import theano.tensor as T

from preprocess import normalize, zero_center
import models_lib as ml

directory = '/home/data/henry/Databowl-2017/'

filenames = []

with open(directory + 'stage1_sample_submission.csv') as f:
    for i, line in enumerate(f):
        if i > 0:
            patient_id, _ = line.split(',')
            filenames += [patient_id]

print('Loading model')

model = ml.convnet3d()
model.load_weights('weights.hdf5')

print('Read data')

x = []
for fn in filenames:
    data = np.load(directory + 'stage1/' + fn + '/img_segment_fill.npy').astype(np.float32)
    data = zero_center(normalize(data))
    data = scipy.ndimage.interpolation.zoom(data, np.array((64.0, 64.0, 64.0)) / np.array(data.shape), order = 1)

    x += [np.expand_dims(data, axis = 0)]

x = np.expand_dims(np.concatenate(x, axis = 0), axis = 1)

print('Run model')

out = model.predict(x)

print('Save data')

f_out = open('submission.csv', 'w')
f_round = open('submission_round.csv', 'w')

f_out.write('id,cancer\n')
f_round.write('id,cancer\n')

for i, fn in enumerate(filenames):
    f_out.write(fn + ',' + str(float(out[i])) + '\n')
    f_round.write(fn + ',' + str(float(np.round(out[i]))) + '\n')

f_out.close()
f_round.close()

print('All done :)')
