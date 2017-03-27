from __future__ import print_function
import os
import time
from collections import OrderedDict
import dicom

import numpy as np
import scipy
from skimage import measure

import preprocess as pr
from preprocess import normalize, zero_center

import pylab as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#plt.ion()

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    for i in range(len(scans)):
        # Convert to Hounsfield units (HU)
        intercept = scans[i].RescaleIntercept
        slope = scans[i].RescaleSlope

        if slope != 1:
            image[i] = slope * image[i].astype(np.float64)
            image[i] = image[i].astype(np.int16)

        image[i] = image[i] + np.int16(intercept)

    return np.array(image, dtype = np.int16)

def plot_3d(image, threshold = -300):

    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, projection = '3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha = 0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show(block = False)

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

class custom_generator(object):
    def __init__(self, folder, targets, preprocess_func):

        self.folder = folder
        self.targets = targets
        self.preprocess_func = preprocess_func

        self.log = []

    def __next__(self):
        y = np.zeros((1, 1), dtype = np.uint8)

        index = np.random.randint(0, len(self.targets))
        self.log += [index]

        x = get_pixels_hu(load_scan(self.folder + 'stage1/' + self.targets['id'].iloc[index]))
        x = self.preprocess_func(x, self.targets, index)
        x = np.expand_dims(x, axis = 0)

        y = np.array(self.targets['cancer'].iloc[index]).reshape((1, 1))

        return x, y

    # For python2 compatibility
    def next(self):
        return self.__next__()
