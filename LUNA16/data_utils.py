from __future__ import print_function
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd

import scipy as sp
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi

import pylab as plt


def mk_mask(shape, center, radius):#, height):
    z, x, y = np.ogrid[:shape[0], :shape[1], :shape[2]]

    cz, cx, cy = center.astype(np.int16)

    r2 = (x - cx) ** 2 + (y - cy) ** 2

    mask_slice = (r2 <= radius ** 2)
    mask = np.zeros(shape)
    mask[np.clip(cz - 1, 0, shape[0]):np.clip(cz + 2, 0, shape[0])] = mask_slice
    #mask = np.tile(mask, (shape[0], 1, 1))

    #mask[:int(cz - height / 2.0)] = False
    #mask[int(cz + height / 2.0) + 1:] = False

    return mask

# WIP
#def image_histogram_equalization(image):
    #image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    #cdf = image_histogram.cumsum() # cumulative distribution function

    ## use linear interpolation of cdf to find new pixel values
    #image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    #return image_equalized.reshape(image.shape)

def get_segmented_lungs(im, plot=False):

    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < -400
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone) 
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone) 
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone) 

    return im, binary

class generator:

    # Note if batch_size means the number of CT scans to load at an iteration, actual slice numebr may be more

    def __init__(self, directory, paths, targets, shape, batch_size, split, validation):
        self.directory = directory
        self.paths = paths
        self.targets = targets
        self.shape = shape
        self.batch_size = batch_size
        self.split = split
        self.validation = validation

        self.log = []
        self.outofbounds = 0

    def get_rand_df(self, paths, df):
        mini_df = []
        # some files may not have a nodule--skipping those
        # Also, some have erroneous data, 289th has a nodule outside of the image
        while(not len(mini_df)):
            if self.validation:
                k = np.random.randint(len(self.paths) - self.split, len(self.paths))
            else:
                k = np.random.randint(0, len(self.paths) - self.split)

            img_id = paths[k].split('subset')[1][2:-4]
            mini_df = df[self.targets['seriesuid'] == img_id] # get all nodules associate with file

        return k, mini_df

    def load_img(self, path):
        itk_img = sitk.ReadImage(path)
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z, y, x (notice the ordering)

        return img_array, itk_img

    def get_mask(self, mini_df, itk_img, img_array):

        mask = np.zeros_like(img_array)
        for node_idx, cur_row in mini_df.iterrows():
            node_x = cur_row['coordX']
            node_y = cur_row['coordY']
            node_z = cur_row['coordZ']
            diam = cur_row['diameter_mm']

            origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
            spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)

            center = np.array([node_x, node_y, node_z])   # nodule center
            v_center = np.rint((center - origin) / spacing)  # nodule center in voxel space (still x,y,z ordering)
            v_center = v_center[::-1] # Now z, y, x ordering

            v_diam = np.sqrt(np.sum((float(diam) / spacing[:2]) ** 2)) # Approximate the diameter of the nodule

            # Check if the data makes sense, sometimes we get nodules outside of the image
            if np.sum(v_center >= 0) != 3 or np.sum(v_center < img_array.shape) != 3:
                self.outofbounds += 1
                continue

            mask = np.logical_or(mask, mk_mask(img_array.shape, v_center, v_diam / 2.0))

        return mask

    def resize(self, img_array, mask, shape):
        factor = [1] + [float(self.shape[0]) / img_array.shape[1], float(self.shape[1]) / img_array.shape[2]]

        if np.sum(np.array(factor)) != 3: # Not changing anything, don't bother with the zoom call
            img_array_resized = ndi.interpolation.zoom(img_array, factor, order = 1)
            mask_resized = ndi.interpolation.zoom(mask, factor, order = 1)

            return img_array_resized, mask_resized

        else:
            return img_array, mask


    def compress(self, img_array, mask):
        mask_compressed = []
        img_array_compressed = []

        for i in xrange(mask.shape[0]):
            if np.sum(mask[i]) > 0:
                img_array_compressed += [img_array[i:i + 1]]
                mask_compressed += [mask[i:i + 1]]

        #self.log += [(k, len(mini_df), np.sum(mask), v_center, v_diam / 2.0, float(diam) / spacing[2])]

        return np.concatenate(img_array_compressed, axis = 0), np.concatenate(mask_compressed, axis = 0)

    def __next__(self, k = None):

        samples = 0
        if k is not None: # We're doing offline preprocessing
            img_id = self.paths[k].split('subset')[1][2:-4]
            mini_df = self.targets[self.targets['seriesuid'] == img_id]
            if not len(mini_df):
                return None, None, None, None

            img_array, itk_img = self.load_img(self.paths[k])
            mask = self.get_mask(mini_df, itk_img, img_array)

            if np.sum(mask) == 0:
                return None, None, None, None

            img_array, mask = self.resize(img_array, mask, self.shape)
            img_array, mask = self.compress(img_array, mask)

            img_array_original = img_array
            binary_image = np.zeros_like(img_array).astype(np.int8)
            for i in xrange(img_array.shape[0]):
                img_array[i], binary_image[i] = get_segmented_lungs(img_array[i])

            return img_array, mask, binary_image, img_array_original

        else:
            data = np.zeros((self.batch_size, 1) + self.shape)
            targets = np.zeros((self.batch_size, 1) + self.shape)
            while(samples < self.batch_size):

                k, mini_df = self.get_rand_df(self.paths, self.targets)
                img_array, itk_img = self.load_img(self.paths[k])
                mask = self.get_mask(mini_df, itk_img, img_array)

                if np.sum(mask) == 0:
                    continue

                img_array, mask = self.resize(img_array, mask, self.shape)
                img_array, mask = self.compress(img_array, mask)

                for i in xrange(img_array.shape[0]):
                    img_array[i], _ = get_segmented_lungs(img_array[i])

                remaining = min(self.batch_size - samples, mask.shape[0])

                shuffle = np.random.permutation(img_array.shape[0])
                data[samples:samples + remaining] = np.expand_dims(img_array[shuffle][:remaining], axis = 1)
                targets[samples:samples + remaining] = np.expand_dims(mask[shuffle][:remaining], axis = 1)

                samples += remaining

        return data, targets

    def next(self, k = None):
        return self.__next__(k)
