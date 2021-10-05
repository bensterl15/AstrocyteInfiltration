import matplotlib.pyplot as plt
import numpy as np
import random
import zarr
from skimage import data
from skimage import filters

# make sure we all see the same
np.random.seed(19623)
random.seed(19623)

# open a sample image (channels first)
raw_data = data.astronaut().transpose(2, 0, 1)

# create some dummy "ground-truth" to train on
gt_data = filters.gaussian(raw_data[0], sigma=3.0) > 0.75
gt_data = gt_data[np.newaxis,:].astype(np.float32)

# store image in zarr container
f = zarr.open('sample_data.zarr', 'w')
f['raw'] = raw_data
f['raw'].attrs['resolution'] = (1, 1)
f['ground_truth'] = gt_data
f['ground_truth'].attrs['resolution'] = (1, 1)

# helper function to show image(s), channels first
def imshow(raw, ground_truth=None, prediction=None):
  rows = 1
  if ground_truth is not None:
    rows += 1
  if prediction is not None:
    rows += 1
  cols = raw.shape[0] if len(raw.shape) > 3 else 1
  fig, axes = plt.subplots(rows, cols, figsize=(10, 4), sharex=True, sharey=True, squeeze=False)
  if len(raw.shape) == 3:
    axes[0][0].imshow(raw.transpose(1, 2, 0))
  else:
    for i, im in enumerate(raw):
      axes[0][i].imshow(im.transpose(1, 2, 0))
  row = 1
  if ground_truth is not None:
    if len(ground_truth.shape) == 3:
      axes[row][0].imshow(ground_truth[0])
    else:
      for i, gt in enumerate(ground_truth):
        axes[row][i].imshow(gt[0])
    row += 1
  if prediction is not None:
    if len(prediction.shape) == 3:
      axes[row][0].imshow(prediction[0])
    else:
      for i, gt in enumerate(prediction):
        axes[row][i].imshow(gt[0])
  plt.show()

#imshow(zarr.open('sample_data.zarr')['raw'][:])

import gunpowder as gp

raw = gp.ArrayKey('RAW')
gt = gp.ArrayKey('GROUND_TRUTH')
prediction = gp.ArrayKey('PREDICTION')

source = gp.ZarrSource(
	'sample_data.zarr',	# The zarr container
	{raw: 'raw',
	 gt: 'ground_truth'},
	{raw: gp.ArraySpec(interpolatable=True),
	 gt: gp.ArraySpec(interpolatable=False)}
)

import torch
from funlib.learn.torch.models import UNet, ConvPass

torch.manual_seed(18)

kernel_dsize = [[[3,3],[3,3]]]*3

unet = UNet(
  in_channels=3,
  num_fmaps=4,
  fmap_inc_factor=2,
  downsample_factors=[[2, 2], [2, 2]],
  kernel_size_down=[[[3, 3], [3, 3]]]*3,
  kernel_size_up=[[[3, 3], [3, 3]]]*2,
  padding='same')

model = torch.nn.Sequential(unet, ConvPass(4,1,[(1,1)], activation=None), torch.nn.Sigmoid())
loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

predict = gp.torch.Predict(model, inputs = {'input': raw}, outputs = {0:prediction})
stack = gp.Stack(1)

train = gp.torch.Train(model,
			loss,
			optimizer,
			inputs={'input':raw},
			loss_inputs = {0: prediction, 1:gt},
			outputs = {0: prediction})

pipeline = source

# Formulate a request for "raw"
request = gp.BatchRequest()
request[raw] = gp.Roi((0, 0), (512, 512))
request[gt] = gp.Roi((0, 0), (512, 512))
request[prediction] = gp.Roi((0, 0), (512, 512))

normalize = gp.Normalize(raw)

pipeline = source + normalize + gp.RandomLocation() + stack + train

# Build the pipeline:
with gp.build(pipeline):
	# Request a batch:
	batch = pipeline.request_batch(request)

#print(f'batch returned:{batch}')
#imshow(batch[raw].data, batch[gt].data, batch[prediction].data)
#print(request)
