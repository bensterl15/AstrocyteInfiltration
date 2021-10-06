from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
import numpy as np
import zarr
import os
import argparse


#cortex_path = '/mnt/efs/woods_hole/segmeNationData/cortex_net/02_train/setup01/7L.63X.RNAiSpz3_5.30_488Ctx_Cy3ast_Cy5elav_1_XY1578699281_Z00_T0_C12.zarr'
#astro_path = '/mnt/efs/woods_hole/segmeNationData/astrocytes_net/02_train/setup04/test_model_checkpoint_44000_7L.63X.RNAiSpz3_5.30_488Ctx_Cy3ast_Cy5elav_1_XY1578699281_Z00_T0_C0.zarr'
#green_path = '/mnt/efs/woods_hole/segmeNationData/Astro_data/3D/collection/7L.63X.RNAiSpz3_5.30_488Ctx_Cy3ast_Cy5elav_1_XY1578699281_Z00_T0_C12.zarr'

#cortex_root_dir = os.path.dirname(cortex_path)
#basename = os.path.basename(cortex_path)
#basename_str = basename[28:-9]


def saveIE(cortex_path, astro_path, green_path):
	cortex_root_dir = os.path.dirname(cortex_path)
	basename = os.path.basename(cortex_path)
	basename_str = basename[28:-9]

	cortex = zarr.open(cortex_path, 'r')
	astro = zarr.open(astro_path, 'r')
	green = zarr.open(green_path, 'r')

	cortex = cortex['predict']
	astro = astro['predict']
	green = green['raw']

	print(np.shape(green))

	M = np.shape(cortex[0])[2]
	N = np.shape(cortex[0])[3]
	cortex_arr = np.zeros((len(cortex), M, N))

	for i in range(len(cortex)):
		cortex_arr[i] = np.squeeze(cortex[i])

	M = np.shape(astro[0])[2]
	N = np.shape(astro[0])[3]
	astro_arr = np.zeros((len(astro), M, N))

	for i in range(len(astro)):
		astro_arr[i] = np.squeeze(astro[i])


	M = np.shape(green[0])[1]
	N = np.shape(green[0])[2]
	green_arr = np.zeros((len(green), M, N))

	for i in range(len(green)):
		green_arr[i] = green[i][0]

	cortex_arr[cortex_arr < np.max(cortex_arr)/2] = 0
	cortex_arr[cortex_arr != 0] = 1

	astro_arr[astro_arr < np.max(astro_arr)/2] = 0
	astro_arr[astro_arr != 0] = 1

	IS = np.sum( np.logical_and(astro_arr, cortex_arr) ) / np.sum(cortex_arr)
	ent = shannon_entropy(green_arr)

	out_file = cortex_root_dir + '/' + basename_str
	print(out_file)
	np.save(out_file, [IS, ent])

	print('IS')
	print(IS)
	print('entropy')
	print(ent)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("astro_path")
	parser.add_argument("cortex_path")
	parser.add_argument("cortex_raw")
	args = parser.parse_args()

	
	astro_path = args.astro_path
	cortex_path = args.cortex_path
	cortex_raw = args.cortex_raw
	saveIE(cortex_path, astro_path, cortex_raw)
