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

num_samples = 5

raw = load_from_dir('0')
gt = np.zeros((num_samples, raw.shape[0], raw.shape[1], raw.shape[2]))
raw = np.zeros((num_samples, raw.shape[0], raw.shape[1], raw.shape[2]))

print(np.shape(gt))

for i in range(num_samples):
	raw[i] = load_from_dir(f'{i}')
	gt_container = zarr.open(f"gt{i}.zarr", "r")
	for j in range(len(gt_container['predict'])):
		gt[i, j] = gt_container['predict'][j]

print(np.shape(raw))
print(type(raw))

gt = gt / np.max(gt)
gt[gt < 0.5] = 0
gt[gt != 0] = 1

zarr_container = zarr.open("training.zarr", "w")
save_to_container(raw, zarr_container, 'raw')
save_to_container(gt, zarr_container, 'gt')
