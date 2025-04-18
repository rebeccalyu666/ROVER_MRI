from typing import Tuple
import torch
import torch.nn as nn


class HashEmbedder(nn.Module):
    def __init__(
        self,
        n_input_dims: int = 3,
        otype: str = "HashGrid",
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 1.39,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super(HashEmbedder, self).__init__()
        assert n_input_dims == 3 and otype == "HashGrid"

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.b = per_level_scale

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(2**self.log2_hashmap_size, self.n_features_per_level)
                for _ in range(n_levels)
            ]
        )
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

        self.register_buffer(
            "box_offsets",
            torch.tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]]),
        )

    def trilinear_interp(
        self,
        x: torch.Tensor,
        voxel_min_vertex: torch.Tensor,
        voxel_embedds: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        """
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = x - voxel_min_vertex

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = (
            voxel_embedds[:, 0] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 4] * weights[:, 0][:, None]
        )
        c01 = (
            voxel_embedds[:, 1] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 5] * weights[:, 0][:, None]
        )
        c10 = (
            voxel_embedds[:, 2] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 6] * weights[:, 0][:, None]
        )
        c11 = (
            voxel_embedds[:, 3] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 7] * weights[:, 0][:, None]
        )

        # step 2
        c0 = c00 * (1 - weights[:, 1][:, None]) + c10 * weights[:, 1][:, None]
        c1 = c01 * (1 - weights[:, 1][:, None]) + c11 * weights[:, 1][:, None]

        # step 3
        c = c0 * (1 - weights[:, 2][:, None]) + c1 * weights[:, 2][:, None]

        return c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = int(self.base_resolution * self.b**i)
            (
                voxel_min_vertex,
                hashed_voxel_indices,
                xi,
            ) = self.get_voxel_vertices(x, resolution)
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)
            x_embedded = self.trilinear_interp(xi, voxel_min_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)
        return torch.cat(x_embedded_all, dim=-1)

    def get_voxel_vertices(
        self, xyz: torch.Tensor, resolution: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz = xyz * resolution
        voxel_min_vertex = torch.floor(xyz).int()

        voxel_indices = voxel_min_vertex.unsqueeze(1) + self.box_offsets
        hashed_voxel_indices = _hash(voxel_indices, self.log2_hashmap_size)

        return voxel_min_vertex, hashed_voxel_indices, xyz


def _hash(coords: torch.Tensor, log2_hashmap_size: int) -> torch.Tensor:
    """
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    """
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    return (
        torch.tensor((1 << log2_hashmap_size) - 1, device=xor_result.device)
        & xor_result
    )
