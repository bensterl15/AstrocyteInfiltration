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
# f = '/mnt/efs/woods_hole/segmeNationData/Astro_data/Astro_raw_gt_0.zarr'

n_batch = len(glob.glob(f'{f}/raw/*'))
n_batch_gt = len(glob.glob(f'{f}/gt/*'))
assert n_batch == n_batch_gt

print(n_batch)

# n_batch = 43
raw = daisy.open_ds(f, 'raw/1')
roi = (n_batch, raw.shape[1], raw.shape[0])

#raw key
raw_np_array = np.empty(shape=roi, dtype=np.uint8)
for i in range(n_batch):
    raw = daisy.open_ds(f, f'raw/{i}')
    raw_np_array[i] = raw.to_ndarray().astype(np.float32)/(64*1024)*255
raws = daisy.Array(data=raw_np_array,
                   roi=daisy.Roi((0, 0, 0), roi),
                   voxel_size=(1, 1, 1))

#gt key
gt = daisy.open_ds(f, 'gt/1')
gts_np_array = np.empty(shape=roi, dtype=np.uint16)
for i in range(n_batch):
    gt = daisy.open_ds(f, f'gt/{i}')
    gts_np_array[i] = gt.to_ndarray().astype(np.float32)
gts = daisy.Array(data=gts_np_array,
                   roi=daisy.Roi((0, 0, 0), roi),
                   voxel_size=(1, 1, 1))

viewer = neuroglancer.Viewer()

with viewer.txn() as s:

    add_layer(s, [raws], 'raw')
    add_layer(s, [gts], 'gt')

    s.layout = 'yz'
    s.projectionScale = 2048

print(viewer)
link = str(viewer)
