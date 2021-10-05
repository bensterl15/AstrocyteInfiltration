import glob
import numpy as np
import random
import skimage.io
import zarr
from numcodecs import Blosc

normalize_raw = 4000.0

print("Loading data...")

raw_files = sorted(glob.glob('/n/groups/htem/Segmentation/networks/tmn7/cortex2_gt_00/source/*'))
labels_files = sorted(glob.glob('/n/groups/htem/Segmentation/networks/tmn7/cortex2_gt_00/target/*'))
# train_masks_files = sorted(glob.glob('/n/groups/htem/users/ras9540/learned_lsds/experiments/worms/data/embedseg_original/bbbc010-2012/train/masks/*.tif'))
# test_raw_files = sorted(glob.glob('/n/groups/htem/users/ras9540/learned_lsds/experiments/worms/data/embedseg_original/bbbc010-2012/test/images/*.tif'))
# test_masks_files = sorted(glob.glob('/n/groups/htem/users/ras9540/learned_lsds/experiments/worms/data/embedseg_original/bbbc010-2012/test/masks/*.tif'))

# source = [skimage.io.imread(f).astype(np.float32)/normalize_raw for f in train_raw_files]
source = [skimage.io.imread(f) for f in source_files]
target = [skimage.io.imread(f) for f in target_files]
# print(source[0].shape)

# strip rgb -> bw
#source = [s[:, :, 0] for s in source]
#target = [s[:, :, 0] for s in target]


# train_masks = [skimage.io.imread(f).astype(np.bool) for f in train_masks_files]
# test_raw = [skimage.io.imread(f).astype(np.float32)/normalize_raw for f in test_raw_files]
# test_masks = [skimage.io.imread(f).astype(np.bool) for f in test_masks_files]

# split training images 85/15 into train and validate
# n = len(train_raw)
# n_validate = int(0.15*n)
# indices = list(range(n))
# random.seed(1912)
# random.shuffle(indices)
# validate_indices = indices[:n_validate]
# train_indices = indices[n_validate:]
# print(f"Using {validate_indices} for validation")
# print(f"Using {train_indices} for training")
# validate_raw = [train_raw[i] for i in validate_indices]
# validate_masks = [train_masks[i] for i in validate_indices]
# train_raw = [train_raw[i] for i in train_indices]
# train_masks = [train_masks[i] for i in train_indices]

# # create unique label segmentation from binary masks
# def to_labels(maskss):
#     segs = []
#     ignores = []
#     for masks in maskss:
#         ignore = np.sum(masks, axis=0) <= 1
#         seg = np.zeros(shape=masks.shape[1:], dtype=np.uint64)
#         for i, mask in enumerate(masks):
#             seg[mask] = i + 1
#         segs.append(seg)
#         ignores.append(ignore)
#     return segs, ignores
# train_seg, train_ignore = to_labels(train_masks)
# validate_seg, validate_ignore = to_labels(validate_masks)
# test_seg, test_ignore = to_labels(test_masks)

print("Writing to zarr...")
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

f = zarr.open('test_gt.zarr', 'a')

for i, (s, t) in enumerate(zip(source, target)):
    f[f'source/{i}'] = zarr.array(s, compressor=compressor)
    f[f'target/{i}'] = zarr.array(t, compressor=compressor)
    # f[f'train/labels/{i}'] = seg
    # f[f'train/ignore/{i}'] = ignore.astype(np.uint8)
    # f[f'train/fg_mask/{i}'] = (seg != 0).astype(np.uint8)
    # f[f'train/masks/{i}'] = masks.astype(np.uint8)
    # f[f'train/labels_mask/{i}'] = np.ones_like(raw)

# for i, (raw, seg, ignore, masks) in enumerate(zip(train_raw, train_seg, train_ignore, train_masks)):
#     f[f'train/raw/{i}'] = raw
#     f[f'train/labels/{i}'] = seg
#     f[f'train/ignore/{i}'] = ignore.astype(np.uint8)
#     f[f'train/fg_mask/{i}'] = (seg != 0).astype(np.uint8)
#     f[f'train/masks/{i}'] = masks.astype(np.uint8)
#     f[f'train/labels_mask/{i}'] = np.ones_like(raw)

# for i, (raw, seg, ignore, masks) in enumerate(zip(validate_raw, validate_seg, validate_ignore, validate_masks)):
#     #raw = np.expand_dims(raw, axis=0)
#     #seg = np.expand_dims(seg, axis=0)
#     f[f'validate/raw/{i}'] = raw
#     f[f'validate/labels/{i}'] = seg
#     f[f'validate/ignore/{i}'] = ignore.astype(np.uint8)
#     f[f'validate/fg_mask/{i}'] = (seg != 0).astype(np.uint8)
#     f[f'validate/masks/{i}'] = masks.astype(np.uint8)
#     f[f'validate/labels_mask/{i}'] = np.ones_like(raw)

# for i, (raw, seg, ingore, masks) in enumerate(zip(test_raw, test_seg, test_ignore, test_masks)):
#     f[f'test/raw/{i}'] = raw
#     f[f'test/labels/{i}'] = seg
#     f[f'test/ignore/{i}'] = ignore.astype(np.uint8)
#     f[f'test/fg_mask/{i}'] = (seg != 0).astype(np.uint8)
#     f[f'test/masks/{i}'] = masks.astype(np.uint8)
#     f[f'test/labels_mask/{i}'] = np.ones_like(raw)
