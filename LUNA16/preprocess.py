from __future__ import print_function
import SimpleITK as sitk
import numpy as np
np.random.seed(1234)

import csv
from glob import glob
import pandas as pd
import time
from multiprocessing import Pool

import pylab as plt
plt.ion()

#from keras.utils.generic_utils import Progbar

import data_utils as du

print('Setup generators')

# Code to preprocess is same as generator
directory = '/home/data/henry/Databowl-2017/LUNA16/'

file_list = glob(directory + '**/*.mhd')

df_node = pd.read_csv(directory + "CSVFILES/annotations.csv")
df_node = df_node.dropna()

gen = du.generator(directory, file_list, df_node, (512, 512), 4, 50, False)

print('Run generator through all data')

#data = []
#targets = []
#pb = Progbar(len(file_list))

#for i in range(len(file_list)):
    #x, y = gen.next(i)
    #if x is not None and y is not None:
        #data += [x]
        #targets += [y]
    #pb.update(i)

def run(k):
    x, y, binary_x, x_original = gen.next(k)
    return x, y, binary_x, x_original

p = Pool(6)
data_all = p.map(run, range(len(file_list)))

data = []
targets = []
binary_imgs = []
imgs = []

for i in range(len(data_all)):
    assert type(data_all[i][0]) == type(data_all[i][1]), 'data type and target type mismatch'
    if data_all[i][0] is not None:
        data += [data_all[i][0]]
        targets += [data_all[i][1]]
        binary_imgs += [data_all[i][2]]
        imgs += [data_all[i][3]]

data = np.concatenate(data, axis = 0)
targets = np.concatenate(targets, axis = 0)
binary_imgs = np.concatenate(binary_imgs, axis = 0)
imgs = np.concatenate(imgs, axis = 0)

print('Done, total patients with nodules ', data.shape[0])
print('Saving')

np.save('/home/data/henry/Databowl-2017/LUNA16/data.npy', data)
np.save('/home/data/henry/Databowl-2017/LUNA16/targets.npy', targets)
np.save('/home/data/henry/Databowl-2017/LUNA16/binary_imgs.npy', binary_imgs)
np.save('/home/data/henry/Databowl-2017/LUNA16/imgs.npy', imgs)

print('Done :)')
