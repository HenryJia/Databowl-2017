from __future__ import print_function
import os
import random
import time
from collections import OrderedDict
import dicom

import numpy as np
import scipy

import preprocess as pr
from preprocess import normalize, zero_center

def load_from_dicom(directory):
    scan = pr.load_scan(directory)
    img = pr.get_pixels_hu(scan)
    #img, spacing = pr.resample(img, scan)
    mask = pr.segment_lung_mask(img, True)
    img = mask * img
    return img

class generator(object):
    def __init__(self, folder, targets, shape, batch_size):
        assert type(targets) is dict or type(targets) is OrderedDict, 'Targets must be a dictionary or ordered dictionary'

        self.folder = folder
        self.targets = targets
        self.shape = shape
        self.batch_size = batch_size

        self.log = []

        random.seed(1234)

    def __next__(self):
        x = np.zeros((self.batch_size, ) + self.shape, dtype = np.float32)
        y = np.zeros((self.batch_size, 2), dtype = np.uint8)

        for i in range(self.batch_size):
            index = random.randint(0, len(self.targets.keys()) - 1)
            self.log += [index]

            data = np.load(self.folder + list(self.targets.keys())[index] + '_img.npy').astype(np.float32)
            #data = load_from_dicom(self.folder + list(self.targets.keys())[index]).astype(np.float32)
            data = scipy.ndimage.interpolation.zoom(data, self.shape / np.array(data.shape, dtype = np.float32), order = 1)
            #data = zero_center(normalize(data))
            data = normalize(data)

            #x[i, :data.shape[0], :data.shape[1], :data.shape[2]] = data
            x[i] = data

            y[i, self.targets[list(self.targets.keys())[index]]] = 1

        x = np.expand_dims(x, axis = 1)

        return x, y

    # For python2 compatibility
    def next(self):
        return self.__next__()

class slicewise_generator(object):
    def __init__(self, folder, targets, shape):
        assert type(targets) is dict or type(targets) is OrderedDict, 'Targets must be a dictionary or ordered dictionary'

        self.folder = folder
        self.targets = targets
        self.shape = shape

        self.log = []

    def __next__(self):
        y = np.zeros((1, 2), dtype = np.uint8)

        index = random.randint(0, len(self.targets.keys()) - 1)
        self.log += [index]

        x = np.load(self.folder + list(self.targets.keys())[index] + '_img.npy').astype(np.float32)
        #x = load_from_dicom(self.folder + list(self.targets.keys())[index]).astype(np.float32)
        x = scipy.ndimage.interpolation.zoom(x, (1, ) + tuple(self.shape / np.array(x.shape[1:], dtype = np.float32)), order = 1)
        #x = zero_center(normalize(x))
        x = zero_center(x)

        x = np.reshape(x, (1, x.shape[0], 1) + x.shape[1:])
        #print(x.shape)

        y[0, self.targets[list(self.targets.keys())[index]]] = 1

        return x, y

    # For python2 compatibility
    def next(self):
        return self.__next__()
