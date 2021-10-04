# !/bin/bash

raw_gt_f=/mnt/efs/woods_hole/segmeNationData/Astro_data/Cortex_raw_gt_test_0.zarr

python -i ng_validate.py $1 --raw_f $raw_gt_f --gt_f $raw_gt_f
