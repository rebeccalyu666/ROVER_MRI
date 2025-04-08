#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import argparse
import pdb
import numpy as np 
import torch 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # 第1步

import sys
sys.path.append("fda")
from data_objects.data_objects_bumonkey import ObservationPoints3D
from optimization.training3D_bumk_hash_v8 import train_emb
from models.nodf import NODF
import glob
import matplotlib.pyplot as plt
from utils import get_config, normalization, prepare_sub_folder
from util_args_bumk_v8_lr1e4 import get_args

# Load experiment setting
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/bumonkeynew_lev11_r192_d2_nl2_log25_bs18_0_150.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
opts = parser.parse_args()
args = get_args()
config = get_config(opts.config)

max_iter = args.num_epochs

nx, ny, nk = args.img_size
nv = args.view_num

print('Load image: {}'.format(args.img_path))
img_path = args.img_path
batch_size = args.batch_size

## fixed hyper parameters
hyper_params = {"lambda_c":  config['lambda_c']}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
files_path = sorted(glob.glob(img_path))
print(files_path)
spa_res = 0.1250

X, Y, Z = np.mgrid[0:nx:1, 0:ny:1, 0:nk:1]
x, y, z = X.ravel(), Y.ravel(), Z.ravel()
N = nx * ny * nk
Y_flat = np.zeros([N*nv,1])
for view_num in range(nv):
    Ydata = np.load(files_path[view_num])
    Ydata = np.transpose(Ydata,[1,0,2])
    Y_flat[view_num*N:(view_num+1)*N,:] = Ydata.reshape(-1)[:, np.newaxis]
    print(view_num)
Y_flat = Y_flat/Y_flat.max()

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


O = ObservationPoints3D(cood1, cood2, cood3, cood4, cood5, Y_flat, batch_size)
dataloader = DataLoader(O, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

# define estimator
field_model = NODF(args,config)
field_model = field_model.to(device)

# optimization parameters
optim = torch.optim.Adam(params=field_model.parameters(),
                         lr=args.learning_rate)

# Setup output folder
output_folder = config['output_folder'] 
model_name = os.path.join(output_folder + '/v81003noastype_lr1e4_{}_lev{}_r{}_d{}_log{}_nl_{}_br{}_bs{}_lambda{}' \
    .format(config['experiment_name'], config['n_levels'], config['r'], config['depth'], config['log2_hashmap_size'],config['n_features_per_level'], config['base_resolution'],args.batch_size, config['lambda_c']))

print(model_name)

output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory, log_directory = prepare_sub_folder(output_directory)

# Load pretrain model
# model_path = os.path.join(checkpoint_directory, "model_001900.pt")
# state_dict = torch.load(model_path,map_location=device)
# field_model.load_state_dict(state_dict['net'])
# print('Load pretrain model: {}'.format(model_path))

writer = SummaryWriter(log_directory)

## run optimization algorithm
train_emb(args, checkpoint_directory, writer, device, field_model, optim, hyper_params, dataloader, max_iter, verbose=True)

## plot fits vs. real data 
coordmap, data = dataloader.dataset.getfulldata()


