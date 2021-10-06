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
from numcodecs import Blosc
import glob


logging.basicConfig(level=logging.INFO)

n_samples = 29

data_dir = "/mnt/efs/woods_hole/segmeNationData/Astro_data/"

zarr_name = "Astro_raw_gt_train_0.zarr"
zarr_path = os.path.join(data_dir, zarr_name)

log_dir = "logs"

# network parameters
num_fmaps = 32
input_size = (30, 196, 196)
output_size = (3, 156, 156)
input_shape = gp.Coordinate(input_size)
output_shape = gp.Coordinate(output_size)


batch_size = 32  # TODO: increase later

voxel_size = gp.Coordinate((1, 1))  # TODO: change later
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

checkpoint_every = 1000
train_until = 50000
snapshot_every = 1000
zarr_snapshot = False
num_workers = 11


def mknet():
    unet = UNet(
        in_channels=1,
        num_fmaps=num_fmaps,
        fmap_inc_factor=2,
        downsample_factors=[
            [2, 2],
            [2, 2],
            [2, 2],
        ],
        kernel_size_down=[[[3, 3], [3, 3]]]*4,
        kernel_size_up=[[[3, 3], [3, 3]]]*3,
        )
    model = torch.nn.Sequential(
        unet,
        ConvPass(num_fmaps, 1, [[1, 1]], activation='Sigmoid'),
        )

    print(model)

    return(model)



def train(iterations):

    model = mknet()

    #model = unet
    # loss = WeightedMSELoss()
    # loss = torch.nn.L1Loss()
    loss = torch.nn.BCELoss()
    # optimizer = torch.optim.Adam(lr=1e-5, params=model.parameters())
    optimizer = torch.optim.Adam(lr=5e-5, params=model.parameters())
    
    raw = gp.ArrayKey('raw')
    gt = gp.ArrayKey('gt')
    predict = gp.ArrayKey('predict')
    # gradients = gp.ArrayKey('GRADIENTS')

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(gt, output_size)

    snapshot_request = gp.BatchRequest()
    snapshot_request[predict] = request[gt].copy()

    sources = tuple(
        gp.ZarrSource(
            zarr_path,
            {
                raw: f'raw/{i}',
                gt: f'gt/{i}',
            },
            {
                raw: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
                gt: gp.ArraySpec(interpolatable=False, voxel_size=voxel_size),
            }) +
        # gp.RandomLocation(min_masked=0.01, mask=fg_mask)
        gp.RandomLocation() +
        gp.Normalize(raw)
        # gp.Normalize(gt)
        for i in range(n_samples)
    )

    # raw:      (h, w)
    # gt:   (h, w)

    pipeline = sources
    pipeline += gp.RandomProvider()

    pipeline += gp.SimpleAugment()
    pipeline += gp.ElasticAugment(
        # control_point_spacing=(64, 64),
        control_point_spacing=(48, 48),
        jitter_sigma=(5.0, 5.0),
        rotation_interval=(0, math.pi/2),
        subsample=4,
        )
    pipeline += gp.IntensityAugment(
        raw,
        scale_min=0.8,
        scale_max=1.2,
        shift_min=-0.2,
        shift_max=0.2)
    # pipeline += gp.NoiseAugment(raw, var=0.01)
    # pipeline += gp.NoiseAugment(raw, var=0.001)
    pipeline += gp.NoiseAugment(raw, var=0.01)

    # raw:          (h, w)
    # affinities:   (2, h, w)
    # affs weights: (2, h, w)

    # add "channel" dimensions
    pipeline += gp.Unsqueeze([raw, gt])
    # pipeline += gp.Unsqueeze([gt])

    # raw:          (1, h, w)
    # affinities:   (2, h, w)
    # affs weights: (2, h, w)

    # pipeline += gp.Squeeze([predict], axis=1)
    pipeline += gp.Stack(batch_size)

    # raw:          (b, 1, h, w)
    # affinities:   (b, 2, h, w)
    # affs weights: (b, 2, h, w)

    pipeline += gp.PreCache(num_workers=num_workers)

    pipeline += gp.Normalize(gt, factor=1)

    pipeline += Train(
        model,
        loss,
        optimizer,
        inputs={
            'input': raw
        },
        outputs={
            0: predict,
        },
        #gradients={
        #    0: gradients
        #},
        loss_inputs={
            0: predict,
            1: gt,
        },
        log_dir = log_dir,
        save_every=1000)

    #pipeline += gp.Squeeze([raw, gt], axis=1)
    pipeline += gp.Squeeze([raw, gt, predict], axis=1)

    pipeline += gp.Snapshot({
            gt: 'gt',
            predict: 'predict',
            raw: 'raw',
        },
        every=snapshot_every,
        # output_filename='batch_{iteration}.hdf',
        output_filename= 'batch_{iteration}.zarr' if zarr_snapshot else 'batch_{iteration}.hdf',
        additional_request=snapshot_request)

    with gp.build(pipeline):
        for i in tqdm(range(iterations)):
            pipeline.request_batch(request)



if __name__ == '__main__':

    if 'test' in sys.argv:
        # global train_until
        train_until = 10
        snapshot_every = 1
        #zarr_snapshot = True
        num_workers = 1
    
    train(train_until)
    
