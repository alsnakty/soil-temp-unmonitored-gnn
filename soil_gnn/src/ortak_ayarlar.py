from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Tuple

DEGISKENLER = ["T", "RH", "WS", "WD_sin", "WD_cos", "Z"]
N_VAR = len(DEGISKENLER)
IDX_RH = 1

DOGRULA_ALANLARI = [
    "pressure_levels",
    "split_ratios",
    "year_aligned_split",
    "bearing_scale",
    "window",
    "attn_heads",
    "hidden_dim",
    "dropout",
    "rh_exclude_inclusive",
]


@dataclass
class Config:
    pressure_levels: Tuple[float, ...] = (1000, 850, 700, 500, 400, 300, 250, 200,
                                          150, 100, 70, 50, 30, 10)
    level_elev_thresholds: Dict[float, float] = field(
        default_factory=lambda: {1000: 100.0, 850: 1450.0, 700: 2900.0})
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    year_aligned_split: bool = False
    step_hours: float = 12.0

    horiz_radius_km: float = 600.0
    horiz_knn: int = 4
    bearing_scale: str = "deg360"

    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    window: int = 3
    attn_heads: int = 4

    lr: float = 1e-3
    batch_size: int = 16
    max_epochs: int = 200
    patience: int = 10
    mask_ratio: float = 0.15
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    realistic_masking: bool = False

    rh_exclude_pressure: float = 200.0
    rh_exclude_inclusive: bool = True
    dedup: str = "first"               # tekrarli pencere tahmininden hangisi tutulur

    device: str = "cpu"

    def __post_init__(self):
        if self.hidden_dim % 4 != 0:
            raise ValueError("hidden_dim 4'e bolunmeli (konumsal kodlama d/4 x 4).")
        if self.hidden_dim % self.attn_heads != 0:
            raise ValueError("hidden_dim, attn_heads'e bolunmeli.")
        if abs(sum(self.split_ratios) - 1.0) > 1e-9:
            raise ValueError("split_ratios toplami 1 olmali.")
        if self.realistic_masking:
            raise NotImplementedError(
                "Gercekci maskeleme (balon patlamasi, ustten kesme) dugum tutarli degil; kapali.")
        if self.bearing_scale not in ("deg360", "rad", "rad_signed"):
            raise ValueError(f"Bilinmeyen bearing_scale: {self.bearing_scale}")
        if self.dedup not in ("first", "last", "mean"):
            raise ValueError(f"Bilinmeyen dedup: {self.dedup}")

    def dogrula_raporu(self) -> List[str]:
        degerler = {f.name: getattr(self, f.name) for f in fields(self)}
        return [f"{ad} = {degerler[ad]!r}" for ad in DOGRULA_ALANLARI]


VARIANTS = {
    "gnn": {},
    "gnn_no_vertical": {"drop": ("V",)},
    "gnn_no_horizontal": {"drop": ("H",)},
    "gnn_fixed_fusion": {"fixed_fusion": True},
    "gnn_global_norm": {"norm": "global"},
    "flat_graphsage": {"flat": True},
}


def variant_opts(name: str) -> dict:
    if name not in VARIANTS:
        raise KeyError(f"Bilinmeyen varyant: {name}")
    o = {"drop": (), "fixed_fusion": False, "norm": "level", "flat": False}
    o.update(VARIANTS[name])
    return o
