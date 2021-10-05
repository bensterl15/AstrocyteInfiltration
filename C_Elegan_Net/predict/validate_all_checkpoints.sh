# !/bin/bash

# 1: model dir
# 2: dataset zarr file

model_dir=$1
raw_gt_f=$2

for ckpt_file in $model_dir/model_checkpoint_20000;
do 
	echo $ckpt_file
	python validate.py $ckpt_file $raw_gt_f; 
	echo 'Done!'
done
