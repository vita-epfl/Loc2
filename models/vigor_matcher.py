from models.cross_view_matcher import CrossViewMatcher

class VigorCrossViewMatcher(CrossViewMatcher):
    def __init__(self, device, sat_bev_res, temperature=0.1, embed_dim=1024, desc_dim=128):
        super().__init__(
            device=device,
            sat_bev_res=sat_bev_res,
            temperature=temperature,
            embed_dim=embed_dim,
            desc_dim=desc_dim,
        )
