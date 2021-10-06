# !/bin/bash

raw_gt_f=/mnt/efs/woods_hole/segmeNationData/Astro_data/Cortex_raw_gt_validate_0.zarr

for ckpt_file in $i/model_checkpoint_*;
do 
	echo $ckpt_file
	python -i validate.py $ckpt_file $raw_gt_f; 
	echo 'Done!'
done
