from funlib.learn.torch.models import UNet, ConvPass
from gunpowder.torch import Train
import gunpowder as gp
import math
import numpy as np
import torch
import logging
import os
import sys
from tqdm import tqdm
from PIL import Image
from skimage import filters
import matplotlib.pyplot as plt
import zarr

logging.basicConfig(level=logging.INFO)

n_samples = 28

data_dir = "/mnt/efs/woods_hole/segmeNationData/Astro_data/"

zarr_name = "Astro.zarr"
zarr_path = os.path.join(data_dir, zarr_name)
log_dir = "logs"

# network parameters
num_fmaps = 32
input_shape = gp.Coordinate((196, 196))
output_shape = gp.Coordinate((156, 156))

batch_size = 32  # TODO: increase later

voxel_size = gp.Coordinate((1, 1))  # TODO: change later
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

checkpoint_every = 5000
train_until = 20000
snapshot_every = 1000
zarr_snapshot = False
num_workers = 11

# class WeightedMSELoss(torch.nn.MSELoss):

#     def __init__(self):
#         super(WeightedMSELoss, self).__init__()

#     def forward(self, prediction, gt, weights):

#         loss = super(WeightedMSELoss, self).forward(
#             prediction*weights,
#             gt*weights)

#         return loss


def test():

    unet = UNet(
        in_channels=1,
        num_fmaps=num_fmaps,
        fmap_inc_factor=2,
        downsample_factors=[
            [2, 2],
            [2, 2],
        ],
        kernel_size_down=[[[3, 3], [3, 3]]]*3,
        kernel_size_up=[[[3, 3], [3, 3]]]*2,
        )
    model = torch.nn.Sequential(
        unet,
        ConvPass(num_fmaps, 1, [[1, 1]], activation='Sigmoid'),
        )

    model.eval()

    raw = gp.ArrayKey('raw')
    predict = gp.ArrayKey('predict')

    torch_predict = gp.torch.Predict(model, inputs={'input' : raw}, outputs={0: predict})

    # Scanning request:
    stack = gp.Stack(1)
    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(predict, output_size)
    scan = gp.Scan(scan_request)
    
    # Request matching the model input and output sizes:
    sources = tuple(
        gp.ZarrSource(
            zarr_path,
            {
                raw: f'raw/{i}',
            },
            {
                predict: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size)
            })
    )

    # raw:      (h, w)
    # gt:   (h, w)

    pipeline = sources
    pipeline += gp.Normalize(predict, factor=1)
    pipeline += gp.Stack(batch_size)
    pipeline += gp.PreCache(num_workers=num_workers)
    pipeline += torch_predict
    pipeline += scan

    pipeline += gp.Snapshot({
            raw: 'raw',
            predict: 'predict'
        },
        every=snapshot_every,
        # output_filename='batch_{iteration}.hdf',
        output_filename='test_batch.zarr' if zarr_snapshot else 'test_batch.hdf',
        additional_request=snapshot_request)

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(predict, output_size)

    with gp.build(pipeline):
        pipeline.request_batch(request)

if __name__ == '__main__':
    test()
