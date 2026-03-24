from pathlib import Path
import configparser

from models.cross_view_matcher import CrossViewMatcher

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.ini"

def _get_sat_bev_res():
    config = configparser.ConfigParser()
    config.read(_CONFIG_PATH)
    return config.getint("Model", "sat_bev_res")

class KittiCrossViewMatcher(CrossViewMatcher):
    def __init__(self, device, temperature=0.1, embed_dim=1024, desc_dim=128):
        super().__init__(
            device=device,
            sat_bev_res=_get_sat_bev_res(),
            temperature=temperature,
            embed_dim=embed_dim,
            desc_dim=desc_dim,
        )
