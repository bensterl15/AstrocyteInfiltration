import importlib
import sys
import os
import zarr
import gunpowder as gp
from numcodecs import Blosc
import numpy as np
import torch
import argparse

torch.backends.cudnn.benchmark = True



def val_loss_acc(predict_zarr, gt_zarr, input_size, output_size, loss_fn, metric_fun):
    z_gt = zarr.open(gt_zarr)
    n_samples = len(z_gt['gt'])
    z_pred = zarr.open(predict_zarr)

    # print("Validate...")
    val_loss = 0
    val_acc = 0
    n_skip_samples = 0
    half_diff_size = (int((input_size[0] - output_size[0]) / 2), int((input_size[1] - output_size[1]) / 2))

    for i in range(n_samples):
        file_ind = i
        gt_i = z_gt['gt'][f'{i}'][:]
        gt_i = gt_i[half_diff_size[0] : -half_diff_size[0], half_diff_size[1] : -half_diff_size[1]]

        pred_i = np.squeeze(z_pred['predict'][f'{i}'][:].astype(np.float64))
        pred_i /= 255
        pred_i = pred_i[half_diff_size[0] : -half_diff_size[0], half_diff_size[1] : -half_diff_size[1]]
        
        val_loss += loss_fn(torch.tensor(pred_i.astype(np.float64)), torch.tensor(gt_i.astype(np.float64)))
        val_acc_i = metric_fn(pred_i > 0.5, gt_i > 0.)
        if np.isnan(val_acc_i):
            n_skip_samples += 1
        else:
            val_acc += val_acc_i

        val_loss = val_loss / n_samples
        val_acc = val_acc / (n_samples - n_skip_samples)

    return(val_loss, val_acc)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("zarr_f")
    args = parser.parse_args()

    # train_script = os.path.dirname(checkpoint)
    model_dir = args.model_dir
    model_dir = os.path.abspath(model_dir)

    sys.path.insert(0, model_dir)
    torch_model = importlib.import_module('train')
    model = torch_model.mknet()
    input_size = torch_model.input_size
    output_size = torch_model.output_size
    # torch_model.eval()
    half_diff_size = (int((input_size[0] - output_size[0]) / 2), int((input_size[1] - output_size[1]) / 2))
    loss_fn = torch.nn.BCELoss()
    metric_fn = lambda x, y: np.sum(np.logical_and(x, y)) / np.sum(np.logical_or(x, y))


    # number of steps and checkpoint interval
    n_iter = torch_model.train_until
    checkpoint_every = torch_model.checkpoint_every
    n_ckpt = int(np.floor(n_iter / checkpoint_every))
    val_arr = np.zeros((n_ckpt, 3))


    # parse groundtruth folder
    zarr_f = args.zarr_f
    gt_zarr = os.path.abspath(zarr_f)
    gt_zarr_file = gt_zarr.split('/')[-1]

    # loop over checkpoints
    for i in range(n_ckpt):
        ckpt_step = (i + 1) * checkpoint_every
        checkpoint_name = f'model_checkpoint_{ckpt_step}' 
        predict_zarr = model_dir + f'/test_{checkpoint_name}_{gt_zarr_file}'
        print(predict_zarr)
        val_loss, val_acc = val_loss_acc(predict_zarr, gt_zarr, input_size, output_size, loss_fn, metric_fn)
        val_arr[i] = [checkpoint_every * (i + 1), val_loss, val_acc]

    val_arr_file = model_dir + '/validation.npz'
    np.save(val_arr_file, val_arr)


