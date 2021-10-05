import numpy as np

import os 
from PIL import Image
import numpy as np

import skimage.measure

import matplotlib.pyplot as plt

import zarr

def load_from_dir(dirname):
    final = []
    for fname in os.listdir(dirname):
        im = Image.open(os.path.join(dirname, fname))
        imarray = np.array(im)
        final.append(imarray)
    return np.asarray(final) # shape = (60000,28,28)

def save_to_container(arr, container, name):
	for i in range(arr.shape[0]):
		dataset = container.create_dataset(f"{name}/{i}", shape=arr[i].shape)
		dataset[:] = arr[i]

raw = load_from_dir('raw')
gt = load_from_dir('gt')

zarr_container = zarr.open("training.zarr", "w")
save_to_container(raw, zarr_container, 'raw')
save_to_container(gt, zarr_container, 'gt')


zarr_container = zarr.open("predict1.zarr", "w")
dat = load_from_dir('1')
save_to_container(dat, zarr_container, 'raw')
zarr_container = zarr.open("predict2.zarr", "w")
dat = load_from_dir('2')
save_to_container(dat, zarr_container, 'raw')
zarr_container = zarr.open("predict3.zarr", "w")
dat = load_from_dir('3')
save_to_container(dat, zarr_container, 'raw')
zarr_container = zarr.open("predict4.zarr", "w")
dat = load_from_dir('4')
save_to_container(dat, zarr_container, 'raw')
zarr_container = zarr.open("predict5.zarr", "w")
dat = load_from_dir('5')
save_to_container(dat, zarr_container, 'raw')
