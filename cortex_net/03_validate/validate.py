import importlib
import sys
import os
import zarr
import gunpowder as gp
from numcodecs import Blosc
import numpy as np
import torch
import argparse

torch.backends.cudnn.benchmark = True

def run(model, input_size, output_size,
		image_size,
		input_zarr,
		raw_dataset,
        voxel_size,
		checkpoint=None,
		):

    # pipeline = ZarrSource(
	    # input_zarr,
	    # datasets={raw: raw_dataset},
	    # array_specs={raw: ArraySpec(interpolatable=True)})

    raw = gp.ArrayKey('val_raw')
    gt = gp.ArrayKey('val_gt')
    prediction = gp.ArrayKey('val_predict')
    # gradients = gp.ArrayKey('GRADIENTS')

    # request = gp.BatchRequest()
    # request.add(raw, input_size)
    # request.add(gt, output_size)
    # snapshot_request = gp.BatchRequest()
    # snapshot_request[prediction] = request[gt].copy()

    pipeline = gp.ZarrSource(
            input_zarr,
            {
                raw: raw_dataset,
            },
            {
                raw: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
            })

    # normalize
    pipeline += gp.Normalize(raw) 

    # unsqueeze
    pipeline += gp.Unsqueeze([raw])

    # set model into evaluation mode
    model.eval()

    pipeline += gp.torch.Predict(
      model,
      inputs = {
        'input': raw
      },
      outputs = {
        0: prediction
      },
     checkpoint = checkpoint
    )

    pipeline += gp.IntensityScaleShift(prediction, 255, 0)

    # request matching the model input and output sizes
    scan_request = gp.BatchRequest()
    scan_request[raw] = gp.Roi((0, 0), input_size)
    scan_request[prediction] = gp.Roi((0, 0), output_size)

    pipeline += gp.Scan(scan_request)

    # request for raw and prediction for the whole image
    request = gp.BatchRequest()
    request[raw] = gp.Roi((0, 0), image_size)
    # request[gt] = gp.Roi((0, 0), image_size)
    request[prediction] = gp.Roi((0, 0), image_size)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    # imshow(batch[raw].data, None, batch[prediction].data)
    return batch[prediction].data



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("zarr_f")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    checkpoint = os.path.abspath(checkpoint)
    checkpoint_name = os.path.split(checkpoint)[1]
    # train_script = os.path.dirname(checkpoint)
    model_dir = os.path.dirname(checkpoint)

    sys.path.insert(0, model_dir)
    torch_model = importlib.import_module('train')
    model = torch_model.mknet()
    input_size = torch_model.input_size
    output_size = torch_model.output_size
    # torch_model.eval()
    half_diff_size = (int((input_size[0] - output_size[0]) / 2), int((input_size[1] - output_size[1]) / 2))


    # read input to get dimension
    input_zarr_name = args.zarr_f
    input_zarr = zarr.open(input_zarr_name)
    n_samples = len(input_zarr['raw'])
    img_dim = input_zarr['raw']['0'].shape
    print(img_dim)

    # make output zarr
    in_zarr = os.path.basename(input_zarr_name)[:-5]
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    output_dir = model_dir + f'/test_{checkpoint_name}_{in_zarr}.zarr'
    f = zarr.open(output_dir, 'a')

    # loop over indices

    for i in range(n_samples):

        ret = run(model, input_size, output_size,
                image_size=img_dim[1:],
            checkpoint=checkpoint,
            raw_dataset=f'raw/{i}',
            input_zarr=input_zarr_name,
            voxel_size=(1, 1),
            )
        ret = np.roll(ret, half_diff_size, axis=(2, 3))

        f[f'predict/{i}'] = ret
        # f[f'raw/{i}'] = input_zarr[f'raw/{i}']
        # f[f'gt/{i}'] = input_zarr[f'gt/{i}']







