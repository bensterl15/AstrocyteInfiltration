import daisy
import neuroglancer
import sys
import glob
import numpy as np
import argparse

from funlib.show.neuroglancer import add_layer
# import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("net1_f")
parser.add_argument("net2_f")
parser.add_argument("raw1_f")
parser.add_argument("raw2_f")
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
# net1_f = sys.argv[1]
# gt_f = '/mnt/efs/woods_hole/segmeNationData/Astro_data/Cortex_raw_gt_test_0.zarr'

net1_f = args.net1_f
net2_f = args.net2_f
raw1_f = args.raw1_f
raw2_f = args.raw2_f

if net1_f[-1] == '/':
    net1_f = net1_f[:-1]

n_batch = len(glob.glob(f'{net1_f}/predict/*'))
# n_batch_gt = len(glob.glob(f'{net1_f}/gt/*'))
# assert n_batch == n_batch_gt
# print(n_batch)

def stack_images(
        net1_f,
        ds_name,
        dtype,
        voxel_size=(1, 1, 1),
        scale_factor=1,
        ):
    ds = daisy.open_ds(net1_f, f'{ds_name}/0')
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
        ds_array = daisy.open_ds(net1_f, f'{ds_name}/{i}').to_ndarray()*scale_factor
        if multi_channel:
            for c in range(len(ds_array)):
                np_array[c, i, :, :] = ds_array[c]
        else:
            np_array[i] = ds_array
    ret = daisy.Array(data=np_array,
                       roi=roi,
                       voxel_size=voxel_size)
    return ret

net1 = stack_images(net1_f, 'predict', np.uint8)
net2 = stack_images(net2_f, 'predict', np.uint8)
raw1 = stack_images(raw1_f, 'raw', np.uint8, scale_factor=1/(64*1024)*255)
raw2 = stack_images(raw2_f, 'raw', np.uint8, scale_factor=1/(64*1024)*255)

combined_raw_shape = (3,) + raw1.shape
combined_raw_array = np.empty(shape=combined_raw_shape, dtype=np.uint8)
combined_raw_array[0,:,:,:] = raw1.to_ndarray()
combined_raw_array[1:,:,:,:] = raw2.to_ndarray()
combined_raw_roi = daisy.Roi((0, 0, 0), raw1.shape)
combined_raw = daisy.Array(data=combined_raw_array,
                           roi=combined_raw_roi,
                           voxel_size=(1, 1, 1))

combined_net_shape = (2,) + net1.shape
combined_net_array = np.empty(shape=combined_net_shape, dtype=np.uint8)
combined_net_array[0,:,:,:] = net1.to_ndarray()
combined_net_array[1,:,:,:] = net2.to_ndarray()
combined_net_roi = daisy.Roi((0, 0, 0), net1.shape)
combined_net = daisy.Array(data=combined_net_array,
                           roi=combined_net_roi,
                           voxel_size=(1, 1, 1))

viewer = neuroglancer.Viewer()

with viewer.txn() as s:
    # add_layer(s, [raw1], 'raw1')
    # add_layer(s, [raw2], 'raw2')
    add_layer(s, [combined_raw], 'combined_raw')
    s.layers[0].shader = "\nvoid main() {\n    emitRGB(\n        vec3(\n            toNormalized(getDataValue(0))*3.0,\n            toNormalized(getDataValue(1))*3.0,\n            toNormalized(getDataValue(2))*3.0)\n        );\n}"
    # add_layer(s, [net1], 'net1')
    # add_layer(s, [net2], 'net2')
    add_layer(s, [combined_net], 'combined_net')
    s.layers[1].shader = "\nvoid main() {\n    emitRGB(\n        vec3(\n            toNormalized(getDataValue(0))*1.0,\n            toNormalized(getDataValue(1))*1.0,\n            toNormalized(getDataValue(0))*0.0)\n        );\n}"
    s.layout = 'yz'
    # s.projectionScale = 2048
    # s.crossSectionScale = 4

print(viewer)
link = str(viewer)
