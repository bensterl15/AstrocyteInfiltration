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

n_samples = 43

data_dir = "/mnt/efs/woods_hole/segmeNationData/Astro_data/"
val_data_dir = "/mnt/efs/woods_hole/segmeNationData/Astro_data/"
test_data_dir = "/mnt/efs/woods_hole/segmeNationData/Astro_data/"
predict_data_dir = "./" # save the prediction in the model folder
# result_dir = "/mnt/efs/woods_hole/segmeNationData/Astro_models/model00/"

zarr_name = "Astro_raw_gt_train_0.zarr"
zarr_path = os.path.join(data_dir, zarr_name)

val_zarr_name = "Astro_raw_gt_validate_0.zarr"
val_zarr_path = os.path.join(val_data_dir, zarr_name)

test_zarr_name = "Astro_raw_gt_test_0.zarr"
test_zarr_path = os.path.join(test_data_dir, zarr_name)

predict_zarr_name = "Astro_predict_0.zarr"
predict_zarr_path = os.path.join(predict_data_dir, zarr_name)

log_dir = "logs"

# network parameters
num_fmaps = 32
input_size = (196, 196)
output_size = (156, 156)
input_shape = gp.Coordinate((196, 196))
output_shape = gp.Coordinate((156, 156))

image_size = (1200, 1200)
loss_fn = torch.nn.BCELoss()
metric_fn = lambda x, y: np.sum(np.logical_and(x, y)) / np.sum(np.logical_or(x, y))


batch_size = 32  # TODO: increase later

voxel_size = gp.Coordinate((1, 1))  # TODO: change later
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

checkpoint_every = 5000
train_until = 20000
snapshot_every = 1000
zarr_snapshot = False
num_workers = 11



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

print(model)

#model = unet
# loss = WeightedMSELoss()
# loss = torch.nn.L1Loss()
loss = torch.nn.BCELoss()
# optimizer = torch.optim.Adam(lr=1e-5, params=model.parameters())
optimizer = torch.optim.Adam(lr=5e-5, params=model.parameters())



def train(iterations):

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



def predict_single_image(val_zarr_path, file_ind, input_size, output_size, image_size, checkpoint=None):
    raw = gp.ArrayKey('val_raw')
    gt = gp.ArrayKey('val_gt')
    prediction = gp.ArrayKey('val_predict')
    # gradients = gp.ArrayKey('GRADIENTS')

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(gt, output_size)

    snapshot_request = gp.BatchRequest()
    snapshot_request[prediction] = request[gt].copy()

    source = gp.ZarrSource(
            val_zarr_path,
            {
                raw: f'raw/{file_ind}',
                gt: f'gt/{file_ind}',
            },
            {
                raw: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
                gt: gp.ArraySpec(interpolatable=False, voxel_size=voxel_size),
            })

    # normalize
    normalize = gp.Normalize(raw) 

    # unsqueeze
    unsqueeze = gp.Unsqueeze([raw])

    # set model into evaluation mode
    model.eval()

    predict = gp.torch.Predict(
      model,
      inputs = {
        'input': raw
      },
      outputs = {
        0: prediction
      },
     checkpoint = checkpoint
    )

    stack = gp.Stack(1)

    # request matching the model input and output sizes
    scan_request = gp.BatchRequest()
    scan_request[raw] = gp.Roi((0, 0), input_size)
    scan_request[prediction] = gp.Roi((0, 0), output_size)

    scan = gp.Scan(scan_request)

    pipeline = (
      source +
      normalize +
      unsqueeze + 
      stack +
      predict +
      scan)

    # request for raw and prediction for the whole image
    request = gp.BatchRequest()
    request[raw] = gp.Roi((0, 0), image_size)
    # request[gt] = gp.Roi((0, 0), image_size)
    request[prediction] = gp.Roi((0, 0), image_size)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    # imshow(batch[raw].data, None, batch[prediction].data)
    return batch[prediction].data



# prediction on the whole dataset
def predict(test_zarr_path, predict_zarr_path, input_size, output_size, image_size, checkpoint=None):
    z = zarr.open(test_zarr_path)
    n_samples = len(z['raw'])

    print("Prediction and Writing to zarr...")
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

    f = zarr.open(predict_zarr_path, 'a')

    for i in range(n_samples):
        file_ind = i
        prediction_i = predict_single_image(test_zarr_path, file_ind, input_size, output_size, image_size, checkpoint=checkpoint)
        f[f'predict/{i}'] = zarr.array(np.squeeze(prediction_i), compressor=compressor)

    print("Done!")



def validate(val_zarr_path, input_size, output_size, image_size, loss_fn, metric_fn, checkpoint=None):
    z = zarr.open(val_zarr_path)
    n_samples = len(z['raw'])
    
    # print("Validate...")
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    
    val_loss = 0
    val_acc = 0
    n_skip_samples = 0
    d_size = (input_size[0] - output_size[0], input_size[1] - output_size[1])
    hd_size = (int((input_size[0] - output_size[0]) / 2), int((input_size[1] - output_size[1]) / 2))
    
    for i in range(n_samples):
        file_ind = i
        gt_i = z['gt'][f'{i}'][:]
        gt_i = gt_i[hd_size[0] : -hd_size[0], hd_size[1] : -hd_size[1]]
        
        prediction_i = predict_single_image(val_zarr_path, file_ind, input_size, output_size, image_size, checkpoint=checkpoint)
        prediction_i = np.squeeze(prediction_i)
        prediction_i = prediction_i[:-d_size[0], :-d_size[1]]
        
        val_loss += loss_fn(torch.tensor(prediction_i.astype(np.float64)), torch.tensor(gt_i.astype(np.float64)))
        val_acc_i = metric_fn(prediction_i > 0.5, gt_i > 0.)
        if np.isnan(val_acc_i):
            n_skip_samples += 1
        else:
            val_acc += val_acc_i
    
    val_loss /= n_samples
    val_acc /= (n_samples - n_skip_samples)
    
    return np.asscalar(val_loss.numpy()), val_acc



if __name__ == '__main__':

    if 'test' in sys.argv:
        # global train_until
        train_until = 10
        snapshot_every = 1
        #zarr_snapshot = True
        num_workers = 1
    
    train(train_until)
    
    # validate on the checkpoints
    print('Validate the model with the checkpoints...')
    checkpoint_files = sorted(glob.glob('model_checkpoint_*'))
    n_models = len(checkpoint_files)
    
    for i, ckpt_file in enumerate(checkpoint_files):
        val_loss, val_acc = validate(val_zarr_path, input_size, output_size, image_size, loss_fn, metric_fn, checkpoint=ckpt_file)
        step = float(ckpt_file.split('_')[-1])
        val_arr[i] = np.array([step, val_loss, val_acc])
    
    val_arr = val_arr[val_arr[:, 0].argsort()]
    print('Done!')

    # pick the best model and predict on test images
    best_ind = np.argmax(val_arr[:, 2])
    best_ckpt_file = 'model_checkpoint_{}'.format(int(val_arr[best_ind, 0]))
    print('The best model is on step {}'.format(val_arr[best_ind, 0]))
    predict(test_zarr_path, predict_zarr_path, input_size, output_size, image_size, best_ckpt_file)


