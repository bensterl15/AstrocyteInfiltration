import daisy
import neuroglancer
import sys
import glob
import numpy as np

from funlib.show.neuroglancer import add_layer
# import numpy as np

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
f = sys.argv[1]
# f = '/mnt/efs/woods_hole/segmeNationData/Astro_data/Cortex_raw_gt_0.zarr'

n_batch = len(glob.glob(f'{f}/raw/*'))
n_batch_gt = len(glob.glob(f'{f}/gt/*'))
assert n_batch == n_batch_gt

print(n_batch)
n_batch = 1
n_batch_gt = 1

# n_batch = 43
raw = daisy.open_ds(f, 'raw/1')
print(raw.shape)
roi = (raw.shape[-3], n_batch, raw.shape[-1], raw.shape[-2])  # channel (2) last
gt_roi = (n_batch, raw.shape[-1], raw.shape[-2])
print(f'gt_roi: {gt_roi}')

#raw key
raw_np_array = np.empty(shape=roi, dtype=np.uint8)
for i in range(n_batch):
    raw = daisy.open_ds(f, f'raw/{i}')
    raw_ndarray = raw.to_ndarray().astype(np.float32)/(64*1024)*255
    # raw_ndarray = raw_ndarray[0,:,:]
    raw_np_array[0, i, :, :] = raw_ndarray[0]
    raw_np_array[1, i, :, :] = raw_ndarray[1]
raws = daisy.Array(data=raw_np_array,
                   roi=daisy.Roi((0, 0, 0), (roi[1], roi[2], roi[3])),
                   voxel_size=(1, 1, 1))

print(f'raw_np_array.shape: {raw_np_array.shape}')
# exit()
# asdf

#gt key
gt = daisy.open_ds(f, 'gt/1')
gts_np_array = np.empty(shape=gt_roi, dtype=np.uint16)
for i in range(n_batch):
    gt = daisy.open_ds(f, f'gt/{i}')
    gts_np_array[i] = gt.to_ndarray().astype(np.uint16)
gts = daisy.Array(data=gts_np_array,
                   roi=daisy.Roi((0, 0, 0), gt_roi),
                   voxel_size=(1, 1, 1))

print(f'gts_np_array.shape: {gts_np_array.shape}')

viewer = neuroglancer.Viewer()

with viewer.txn() as s:

    add_layer(s, [raws], 'raw')
    add_layer(s, [gts], 'gt')

    s.layout = 'yz'
    s.projectionScale = 2048

    print(s.layers[0].shader)
    s.layers[0].shader = "\nvoid main() {\n    emitRGB(\n        vec3(\n            toNormalized(getDataValue(0))*3.0,\n            toNormalized(getDataValue(1))*3.0,\n            toNormalized(getDataValue(1))*3.0)\n        );\n}"

print(viewer)
link = str(viewer)
