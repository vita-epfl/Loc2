import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import DeepResBlockDesc

_DEFAULT_BLOCK_DIMS = (512, 256, 128, 64)

class CrossViewMatcher(nn.Module):

    def __init__(
        self,
        device,
        sat_bev_res,
        temperature=0.1,
        embed_dim=1024,
        desc_dim=128,
    ):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.embed_dim = embed_dim
        self.sat_bev_res = sat_bev_res

        self.dustbin_score = nn.Parameter(torch.tensor(1.0))
        self.grd_projector = DeepResBlockDesc(
            True,
            last_dim=desc_dim,
            in_channels=embed_dim,
            block_dims=_DEFAULT_BLOCK_DIMS,
            add_posEnc=True,
            norm_desc=True,
        )
        self.sat_projector = DeepResBlockDesc(
            True,
            last_dim=desc_dim,
            in_channels=embed_dim,
            block_dims=_DEFAULT_BLOCK_DIMS,
            add_posEnc=False,
            norm_desc=True,
        )

    def forward(self, grd_feature, sat_feature, mask):
        grd_desc = self.grd_projector(grd_feature).flatten(2)

        sat_feature_bev = F.interpolate(
            sat_feature,
            (self.sat_bev_res, self.sat_bev_res),
            mode="bilinear",
            align_corners=False,
        )
        sat_desc = self.sat_projector(sat_feature_bev).flatten(2)

        matching_score_original = (
            torch.matmul(sat_desc.transpose(1, 2).contiguous(), grd_desc)
            / self.temperature
        )
        matching_score_original = matching_score_original.masked_fill(
            ~mask.unsqueeze(1), float("-inf")
        )

        batch_size, num_sat, num_grd = matching_score_original.shape
        sat_dustbin = self.dustbin_score.expand(batch_size, num_sat, 1)
        grd_dustbin = self.dustbin_score.expand(batch_size, 1, num_grd)
        dustbin_corner = self.dustbin_score.expand(batch_size, 1, 1)

        couplings = torch.cat(
            [
                torch.cat([matching_score_original, sat_dustbin], dim=-1),
                torch.cat([grd_dustbin, dustbin_corner], dim=-1),
            ],
            dim=1,
        )
        couplings = F.softmax(couplings, dim=1) * F.softmax(couplings, dim=2)
        matching_score = couplings[:, :-1, :-1]

        return matching_score, matching_score_original
