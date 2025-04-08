from models.wire import RealGaborLayer, Wire2DLayer, Wire3DLayer, WireLayer
# from optimization.functions import neg_log_likelihood, integrated_roughness
from models.siren import SineLayer
from models.relu import ReluLayer
from torch import optim
import pytorch_lightning as pl
from torch import nn
import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from math import cos, pi
from models.hash_embeddings import HashEmbedder
from math import log2


class INR(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=True,
        first_omega_0=30,
        hidden_omega_0=30.0,
        sigma0=10.0,
        inr="wire",
        skip_conn=False,
        batchnorm=False,
    ):
        super().__init__()

        self.net = []
        self.domain_dim = in_features
        self.range_dim = out_features
        self.inr = inr
        self.skip_conn = skip_conn
        self.batchnorm = batchnorm

        if inr == "wire":
            self.nn = WireLayer
            dtype = torch.cfloat
            bias = True
            trainable = False
        elif inr == "siren":
            self.nn = SineLayer
            dtype = torch.float
            bias = False
            trainable = True
        elif inr == "relu":
            self.nn = ReluLayer
            dtype = torch.float
            bias = False
            trainable = True
        else:
            raise Exception(
                f"Invalid inr selected: {inr}. Valid options are siren and wire."
            )

        self.net.append(
            self.nn(
                in_features,
                hidden_features,
                is_first=True,
                trainable=trainable,
                omega_0=first_omega_0,
                sigma0=sigma0,
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                self.nn(
                    hidden_features,
                    hidden_features,
                    omega_0=hidden_omega_0,
                    sigma0=sigma0,
                    batchnorm=self.batchnorm,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(
                hidden_features, out_features, bias=bias, dtype=dtype
            )
            if inr == "siren":
                with torch.no_grad():
                    final_linear.weight.uniform_(
                        -np.sqrt(6 / hidden_features) / hidden_omega_0,
                        np.sqrt(6 / hidden_features) / hidden_omega_0,
                    )

            self.net.append(final_linear)
        else:
            self.net.append(
                self.nn(hidden_features, out_features, is_first=False, dtype=dtype)
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.to(torch.float)

        if self.skip_conn:
            first_output = self.net[:1](coords)
            second_output = self.net[1:-2](first_output)
            second_input = first_output + second_output  # skip connection
            output = self.net[-2:](second_input)
        else:
            output = self.net(coords)

        if self.inr == "wire":
            output = output.real
        if not self.training:
            output = output.to(torch.float)
        return {"model_embedding": coords, "model_out": output}


# define the LightningModule
class NODF(nn.Module):
    def __init__(
        self,
        args,
        config,
        outermost_linear=True,
    ):
        super().__init__()
        self.args = args
        self.config = config

        # define harominc function space
        K = 1 #int((args.sh_order + 1) * (args.sh_order + 2) / 2)

        self.inr = INR(
            in_features=(self.config['n_levels'] * self.config['n_features_per_level']) + 3,
            out_features=K,
            hidden_features=config['r'],
            hidden_layers=config['depth'],
            outermost_linear=outermost_linear,
            first_omega_0=args.omega0,
            hidden_omega_0=args.omega0_hidden,
            sigma0=args.sigma0,
            inr=self.args.inr,
            skip_conn=self.args.skip_conn,
            batchnorm=self.args.batchnorm,
        )

        self.hash_embedder = HashEmbedder(
            n_levels=self.config['n_levels'],
            n_features_per_level=self.config['n_features_per_level'],
            log2_hashmap_size=self.config['log2_hashmap_size'],
            base_resolution=self.config['base_resolution'],
            per_level_scale=self.args.per_level_scale,
        )

        # self.save_hyperparameters()

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        coords = coords.squeeze(0)
        embeddings = self.hash_embedder(coords)
        model_input = torch.cat([embeddings, coords], dim=-1)
        model_output = self.inr(model_input)
        model_output["model_in"] = coords

        return model_output

class Encoder3DCNN(nn.Module):
    def __init__(self):
        super(Encoder3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=27, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        return x

class NODF_CNN(nn.Module):
    def __init__(
        self,
        args,
        config,
        outermost_linear=True,
    ):
        super().__init__()
        self.args = args
        self.config = config

        # define harominc function space
        K = 1 #int((args.sh_order + 1) * (args.sh_order + 2) / 2)

        self.inr = INR(
            in_features=(self.config['n_levels'] * self.config['n_features_per_level']) + 3 + 27,
            out_features=K,
            hidden_features=config['r'],
            hidden_layers=config['depth'],
            outermost_linear=outermost_linear,
            first_omega_0=args.omega0,
            hidden_omega_0=args.omega0_hidden,
            sigma0=args.sigma0,
            inr=self.args.inr,
            skip_conn=self.args.skip_conn,
            batchnorm=self.args.batchnorm,
        )

        self.hash_embedder = HashEmbedder(
            n_levels=self.config['n_levels'],
            n_features_per_level=self.config['n_features_per_level'],
            log2_hashmap_size=self.config['log2_hashmap_size'],
            base_resolution=self.config['base_resolution'],
            per_level_scale=self.args.per_level_scale,
        )

        # self.save_hyperparameters()
        self.Encoder3DCNN = Encoder3DCNN()

    def forward(self, coords, Y_cnn):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        coords = coords.squeeze(0)
        embeddings = self.hash_embedder(coords)
        cnn_feature = self.Encoder3DCNN(Y_cnn.squeeze().permute(0,4,1,2,3)) #[1,80000,8,8,8,8]
        model_input = torch.cat([embeddings, coords, cnn_feature], dim=-1)
        model_output = self.inr(model_input)
        model_output["model_in"] = coords

        return model_output