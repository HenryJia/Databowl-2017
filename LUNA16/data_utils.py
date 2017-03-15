from __future__ import print_function
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd

import scipy as sp
from scipy.ndimage.interpolation import zoom


def mk_mask(shape, center, radius, height):
    z, x, y = np.ogrid[:shape[0], :shape[1], :shape[2]]

    cz, cx, cy = center

    r2 = (x - cx) ** 2 + (y - cy) ** 2

    mask = (r2 <= radius ** 2)
    mask = np.tile(mask, (shape[0], 1, 1))

    mask[:int(cz - height / 2.0)] = False
    mask[int(cz + height / 2.0) + 1:] = False

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
    binary = im < 604
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

    return im

class generator:
    def __init__(self, directory, paths, targets, shape, compress, batch_size, split, validation):
        self.directory = directory
        self.paths = paths
        self.targets = targets
        self.shape = shape
        self.compress = compress
        self.batch_size = batch_size
        self.split = split
        self.validation = validation

        self.log = []
        self.outofbounds = 0
    def __next__(self):

        for s in self.shape:
            if s is None:
                assert self.batch_size == 1, 'batch_size must be 1 if any dimensions are variable/None'

        data = []
        tagets = []
        nodes = []
        for j in xrange(self.batch_size):
            data = []
            targets = []

            node_df = []
            k = 0
            # some files may not have a nodule--skipping those
            # Also, some have erroneous data, 289th has a nodule outside of the image
            while(not len(node_df)):
                if self.validation:
                    k = np.random.randint(len(self.paths) - self.split, len(self.paths))
                else:
                    k = np.random.randint(0, len(self.paths) - self.split)

                img_id = self.paths[k].split('subset')[1][2:-4]
                node_df = self.targets[self.targets['seriesuid'] == img_id] #get all nodules associate with file

            itk_img = sitk.ReadImage(self.paths[k])
            img_array = sitk.GetArrayFromImage(itk_img) # indexes are z, y, x (notice the ordering)

            factor = [(float(s1) / s0) if s1 else 1 for s1, s0 in zip(self.shape, img_array.shape)]

            mask = np.zeros_like(img_array)

            for i in xrange(node_df['diameter_mm'].values.shape[0]):
                node_x = node_df['coordX'].values[i]
                node_y = node_df['coordY'].values[i]
                node_z = node_df['coordZ'].values[i]
                diam = node_df['diameter_mm'].values[i]

                center = np.array([node_x, node_y, node_z])   # nodule center
                origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
                spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
                v_center = np.rint((center - origin) / spacing)  # nodule center in voxel space (still x,y,z ordering)
                v_center = v_center[::-1] # Now z, y, x ordering
                v_diam = np.sqrt(np.sum((float(diam) / spacing[:2]) ** 2)) # Approximate the diameter of the nodule

                if np.sum(v_center >= 0) != 3: # Check if the data makes sense, sometimes we get nodules outside of the image
                    self.outofbounds += 1
                    return next(self)

                mask = np.logical_or(mask, mk_mask(img_array.shape, v_center, v_diam / 2.0, float(diam) / spacing[2]))


            if np.sum(np.array(factor)) != 3: # Not changing anything, don't bother with the zoom call
                img_array = zoom(img_array, factor, order = 1)
                mask = zoom(mask, factor, order = 1)

            if self.compress:
                mask_compressed = []
                img_array_compressed = []
                for i in xrange(mask.shape[0]):
                    if np.sum(mask[i]) > 0:
                        img_array_compressed += [img_array[i:i + 1]]
                        mask_compressed += [mask[i:i + 1]]
                self.log += [(k, len(node_df), np.sum(mask), v_center, v_diam / 2.0, float(diam) / spacing[2])]
                img_array = np.concatenate(img_array_compressed, axis = 0)
                mask = np.concatenate(mask_compressed, axis = 0)

            data += [np.expand_dims(img_array, axis = 0)]
            targets += [np.expand_dims(mask, axis = 0)]

        data = np.concatenate(data, axis = 0)
        targets = np.expand_dims(np.concatenate(targets, axis = 0), axis = 2)
        targets = np.concatenate([1 - targets, targets], axis = 2)
        return data, targets

    def next(self):
        return self.__next__()
