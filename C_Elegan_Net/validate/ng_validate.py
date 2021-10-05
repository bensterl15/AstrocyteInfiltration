import daisy
import neuroglancer
import sys
import glob
import numpy as np
import argparse

from funlib.show.neuroglancer import add_layer
# import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("f")
parser.add_argument("--raw_f", default=None)
parser.add_argument("--gt_f", default=None)
args = parser.parse_args()

for i in range(33400, 33500):
    try:
        # neuroglancer.set_server_bind_address('0.0.0.0', i)
        neuroglancer.set_server_bind_address('localhost', i)
        if len(sys.argv) > 1 and sys.argv[1] == "--unsynced":
            viewer = neuroglancer.UnsynchronizedViewer()
        else:
            viewer = neuroglancer.Viewer()
        break
    except:
        continue

#path to raw
# f = sys.argv[1]
# gt_f = '/mnt/efs/woods_hole/segmeNationData/Astro_data/Cortex_raw_gt_test_0.zarr'

f = args.f
gt_f = args.gt_f
raw_f = args.raw_f

if f[-1] == '/':
    f = f[:-1]

n_batch = len(glob.glob(f'{f}/predict/*'))
# n_batch_gt = len(glob.glob(f'{f}/gt/*'))
# assert n_batch == n_batch_gt
# print(n_batch)

def stack_images(
        f,
        ds_name,
        dtype,
        voxel_size=(1, 1, 1),
        scale_factor=1,
        ):
    ds = daisy.open_ds(f, f'{ds_name}/0')
    multi_channel = False
    if len(ds.shape) == 3:
        multi_channel = True

    if multi_channel:
        shape = (ds.shape[-3], n_batch, ds.shape[-2], ds.shape[-1])
        roi = daisy.Roi((0, 0, 0), shape[1:])
    else:
        shape = (n_batch, ds.shape[-2], ds.shape[-1])
        roi = daisy.Roi((0, 0, 0), shape)
    print(roi)
    np_array = np.empty(shape=shape, dtype=dtype)

    for i in range(n_batch):
        ds_array = daisy.open_ds(f, f'{ds_name}/{i}').to_ndarray()*scale_factor
        if multi_channel:
            for c in range(len(ds_array)):
                np_array[c, i, :, :] = ds_array[c]
        else:
            np_array[i] = ds_array
    ret = daisy.Array(data=np_array,
                       roi=roi,
                       voxel_size=voxel_size)
    return ret

predict = stack_images(f, 'predict', np.uint8)
if raw_f is not None:
    raw = stack_images(gt_f, 'raw', np.uint8, scale_factor=1/(64*1024)*255)
if gt_f is not None:
    gt = stack_images(gt_f, 'gt', np.uint16)

viewer = neuroglancer.Viewer()

with viewer.txn() as s:
    if raw_f is not None:
        add_layer(s, [raw], 'raw')
        s.layers[0].shader = "\nvoid main() {\n    emitRGB(\n        vec3(\n            toNormalized(getDataValue(0))*0.0,\n            toNormalized(getDataValue(1))*3.0,\n            toNormalized(getDataValue(0))*3.0)\n        );\n}"
    if gt_f is not None:
        add_layer(s, [gt], 'gt')
    add_layer(s, [predict], 'predict')
    s.layout = 'yz'
    # s.projectionScale = 2048
    # s.crossSectionScale = 4

print(viewer)
link = str(viewer)
