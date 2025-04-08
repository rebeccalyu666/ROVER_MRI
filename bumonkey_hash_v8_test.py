#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
import argparse
import pdb
import numpy as np 
import torch 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # 第1步
import nibabel as nib

import sys
sys.path.append("fda")
from data_objects.data_objects_bumonkey import ObservationPoints3D
from models.nodf import NODF
import glob
import matplotlib.pyplot as plt
from utils import get_config, normalization, prepare_sub_folder
from util_args_bumk_v8_lr1e4 import get_args
import time

start_time = time.time()

# Load experiment setting
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/bumonkeynew_lev11_r192_d2_nl2_log25_bs18_0_150.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
opts = parser.parse_args()
args = get_args()
config = get_config(opts.config)

def flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr

nx, ny, nk = args.img_size
nv = args.view_num

print('Load image: {}'.format(args.img_path))
img_path = args.img_path
batch_size = args.batch_size

device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
files_path = sorted(glob.glob(img_path))

X, Y, Z = np.mgrid[0:nx:1, 0:ny:1, 0:nk:1]
x, y, z = X.ravel(), Y.ravel(), Z.ravel()
V_xyk = np.stack([x, y, z], axis=-1).reshape(-1, 3)

N = nx*ny*nk

coods = [np.zeros((N * nv, 3)) for _ in range(5)]  # 对应 cood1 ~ cood5

def trans(x, y, z, Affine):
    P = np.array([x, y, z, [1] * x.size])
    x_, y_, z_, _ = np.dot(Affine, P)
    base = np.stack([x_, y_, z_], axis=-1).reshape(-1, 3)

    axis_vec = Affine[0:3, 2]
    norm_axis = axis_vec / np.linalg.norm(axis_vec)

    offsets = [-2, -1, 0, 1, 2]
    result = [base + spa_res * offset * norm_axis for offset in offsets]

    print(np.linalg.norm(axis_vec))
    return result

for view_num in range(nv):
    Affine = np.load(f'../DWI_SR/BU_monkey_V8/Affine_nii_{view_num + 1}.npy')
    results = trans(x, y, z, Affine)

    for i in range(5):
        coods[i][view_num * N:(view_num + 1) * N, :] = results[i]

    print(view_num)

cood1, cood2, cood3, cood4, cood5 = coods

all_coords = np.concatenate([cood1, cood2, cood3, cood4, cood5], axis=0)

axi_x_min = np.min(all_coords[:, 0])
axi_x_max = np.max(all_coords[:, 0])

axi_y_min = np.min(all_coords[:, 1])
axi_y_max = np.max(all_coords[:, 1])

axi_z_min = np.min(all_coords[:, 2])
axi_z_max = np.max(all_coords[:, 2])
coords = [cood1, cood2, cood3, cood4, cood5]
for i in range(len(coords)):
    coords[i] = normalization(coords[i], axi_x_min, axi_x_max, axi_y_min, axi_y_max, axi_z_min, axi_z_max)
cood1, cood2, cood3, cood4, cood5 = coords


Y_flat = np.zeros([N*nv,1])
for view_num in range(nv):
    Ydata = np.load(files_path[view_num])
    Ydata = np.transpose(Ydata,[1,0,2])
    Y_flat[view_num*N:(view_num+1)*N,:] = Ydata.reshape(-1)[:, np.newaxis]
    print(view_num)
Y_flat = Y_flat/Y_flat.max()

O = ObservationPoints3D(cood1, cood2, cood3, cood4, cood5, Y_flat, batch_size)
dataloader = DataLoader(O, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)


# Setup output folder

output_folder = config['output_folder'] 

model_name = os.path.join(output_folder + '/v81003noastype_lr1e4_{}_lev{}_r{}_d{}_log{}_nl_{}_br{}_bs{}_lambda{}' \
    .format(config['experiment_name'], config['n_levels'], config['r'], config['depth'], config['log2_hashmap_size'],config['n_features_per_level'], config['base_resolution'],args.batch_size, config['lambda_c']))

print(model_name)
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory, log_directory = prepare_sub_folder(output_directory)

# Setup input encoder:
field_model = NODF(args,config)

field_model = field_model.to(device)
field_model.eval()

# Load pretrain model
model_path = os.path.join(checkpoint_directory, "model_{:06d}.pt".format(config['iter']))
state_dict = torch.load(model_path,map_location=device)
field_model.load_state_dict(state_dict['net'])
print('Load pretrain model: {}'.format(model_path))


## plot fits vs. real data 
coordmap, data = dataloader.dataset.getfulldata()
f_list = [np.zeros((nx * ny * nk, 1)) for _ in range(5)]

for slc in range(nk):
    for i in range(5):
        key = f"coord{i+1}"
        coords = coordmap[key][slc * nx * ny : (slc + 1) * nx * ny, :]
        f_hat = field_model(coords)["model_out"].cpu().detach().numpy()
        f_list[i][slc * nx * ny : (slc + 1) * nx * ny] = f_hat

# reshape
f_list = [f.reshape((nx, ny, nk)) for f in f_list]

# stack into final result: shape (nx, ny, 5 * nk)
result = np.zeros((nx, ny, 5 * nk))
for i in range(5):
    for slc in range(nk):
        result[:, :, 5 * slc + i] = f_list[i][:, :, slc]

bumksrr = nib.load('./Results_imgs/GT/hr_monkey_centered.nii.gz')
recon_new = np.zeros(result.shape)
recon_new[:,:,0:488] = result[:,:,2:490]
recon = nib.Nifti1Image(recon_new, affine=bumksrr.affine, header=bumksrr.header)
nib.save(recon, 'Results_imgs/T23Thash_predict/v81003noastype_lr1e4_'+
         str(config['r'])+'w_'+str(config['depth'])+'d_'+str(config['n_levels'])+'lev_'+str(config['n_features_per_level'])+'nplev_'+str(config['log2_hashmap_size'])+'hasize_'+str(config['base_resolution'])+'bs_'+str(config['lambda_c'])+'lambda_'+str(config['iter'])+'iter.nii')

print('done')
end_time = time.time()
execution_time = end_time - start_time
print("Cost：", execution_time, "seconds")

# nohup python -u bumonkey_hash_v8_test.py > bumonkey_hash_test.out 2>&1 &
