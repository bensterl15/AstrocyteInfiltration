#!/bin/bash


data_str=$1
data_ind=${data_str::1}


cortex_rt=/mnt/efs/woods_hole/segmeNationData/cortex_net/02_train/setup01
astro_rt=/mnt/efs/woods_hole/segmeNationData/astrocytes_net/02_train/setup04

cortex_raw_rt=/mnt/efs/woods_hole/segmeNationData/Astro_data/3D/$data_str

for file in $cortex_rt/test_model_checkpoint_30000_${data_ind}L* ; 
do 
	file_str=$(echo $file | grep -oE test_model.*$ )
	file_str=${file_str:28:-7}
	
	astro_file=$(ls -d $astro_rt/test_model_checkpoint_44000_$file_str*)
	cortex_raw_file=$(ls -d $cortex_raw_rt/${file_str}12.zarr)
	echo $file_str	
	echo $file 
	echo $astro_file
	echo $cortex_raw_file
	python InvasionEntropy.py $astro_file $file $cortex_raw_file


done
