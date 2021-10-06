
cd /mnt/efs/woods_hole/segmeNationData

datapath=$1

for fn in $datapath/*C0.zarr
do 
	fullbasename=${fn::-8}
	basename=$( echo $fullbasename | grep -oE [6-8]L[^/]*$ )

	raw1=${basename}_C0.zarr
	raw2=${basename}_C12.zarr
	fullraw1=${fullbasename}_C0.zarr
	fullraw2=${fullbasename}_C12.zarr
	
	echo $fullraw1 $fullraw2
	# run astro network
        cd /mnt/efs/woods_hole/segmeNationData/astrocytes_net/03_validate
	python validate.py ../02_train/setup04/model_checkpoint_44000 $fullraw1
	# run cortex network
	# cd /mnt/efs/woods_hole/segmeNationData/cortex_net/03_validate/
	# python validate.py ../02_train/setup01/model_checkpoint_30000 $fullraw2

	# visualize results
	net1=/mnt/efs/woods_hole/segmeNationData/segmeNationRepo/astrocyte_net/02_train/setup00/test_model_checkpoint_20000_${raw1}
	net2=/mnt/efs/woods_hole/segmeNationData/cortex_net/02_train/setup01/test_model_checkpoint_30000_${raw2}
	# python -i cortex_net/03_validate/ng_validate_both.py $net1 $net2 $fullraw1 $fullraw2


done
