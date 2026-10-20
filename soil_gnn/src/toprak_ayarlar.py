from dataclasses import dataclass, field
from typing import Dict, Tuple

DERINLIKLER = (0, 5, 10, 20, 50, 100)
HEDEF_KOLON = {5: "t5cmsicaklik", 10: "t10cmsicaklik", 20: "t20cmsicaklik",
               50: "t50cmsicaklik", 100: "t100cmsicaklik"}
UYDU = ["LST_Day_C", "LST_Night_C", "NDVI", "soil_moisture"]
SOIL_VARS = ["sand", "silt", "clay", "bdod", "soc", "cfvo", "wv0033", "wv1500"]
# derinlik dugumu -> SoilGrids katmani (yuzey dugumu ust katmani alir)
SOIL_KATMAN = {0: "0-5cm", 5: "0-5cm", 10: "5-15cm", 20: "15-30cm", 50: "30-60cm", 100: "60-100cm"}

# oznitelik sutunlari (tum dugumlerde ayni vektor; uygulanamayan = NaN)
OZNITELIK = ["T"] + UYDU + SOIL_VARS + ["rakim"]
IDX_T = 0
N_FEAT = len(OZNITELIK)
MASKE_VARSAYILAN = ("T", "LST_Day_C", "LST_Night_C")   # eksik olabilen girdiler (NDVI bos degil, SoilGrids/rakim sabit)

SPATIAL_HOLDOUT = (17020, 17130, 17095, 17310, 17282, 17090, 17196, 17238)

EDGE_DIM = {"V": 1, "H": 2, "T": 1}


def _md5(yol) -> str:
    import hashlib
    h = hashlib.md5()
    with open(yol, "rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


def ortam_bilgisi(out_dir=None, master_path=None) -> str:
    import glob
    import hashlib
    import os
    import platform
    import numpy
    import pandas
    import torch
    surum = {"python": platform.python_version(), "torch": torch.__version__, "cuda": str(torch.version.cuda),
             "numpy": numpy.__version__, "pandas": pandas.__version__}
    for ad in ("torch_geometric", "sklearn"):
        try:
            surum[ad] = __import__(ad).__version__
        except Exception:
            surum[ad] = "yok"
    surum["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "yok"
    satir = " | ".join(f"{k} {v}" for k, v in surum.items())
    if out_dir:
        kod = os.path.dirname(os.path.abspath(__file__))
        md5 = [f"{_md5(f)}  {os.path.basename(f)}" for f in sorted(glob.glob(os.path.join(kod, "*.py")))]
        veri = [f"{_md5(master_path)}  {os.path.basename(master_path)}"] if master_path else []
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "ortam.txt"), "w", encoding="utf-8") as f:
            f.write(satir + "\n\nkod md5:\n" + "\n".join(md5) + "\n\nveri md5:\n" + "\n".join(veri) + "\n")
    return satir


def edge_dims(cfg) -> Dict[str, int]:
    """Kenar oznitelik boyutlari; yatay_rakim acikken H = [d/1000, yon, (rakim_kaynak - rakim_hedef)/1000 km]."""
    d = dict(EDGE_DIM)
    if cfg.yatay_rakim:
        d["H"] = 3
    return d


@dataclass
class SoilConfig:
    master_path: str = "DATASET/master_dataset_qc.csv"
    train_years: Tuple[int, ...] = tuple(range(2015, 2023))
    val_years: Tuple[int, ...] = (2023,)
    test_years: Tuple[int, ...] = (2024, 2025)
    holdout: Tuple[int, ...] = SPATIAL_HOLDOUT
    ic_dogrulama: bool = False

    horiz_radius_km: float = 200.0
    horiz_knn: int = 4
    bearing_scale: str = "deg360"
    dikey_tam: bool = False          # dusey kenar: True = istasyonda her derinlik tum derinliklere; False = yalniz komsu derinlik
    yatay_rakim: bool = False
    step_hours: float = 24.0         # zamansal kenar: gun farki
    uydu_derinlik: str = "yuzey"     # "yuzey": uydu yalniz 0 cm dugumunde; "tum": her derinlik dugumunde
    konum_pe: bool = True            # False: enlem/boylam konumsal kodlamadan cikarilir (ablasyon; derinlik ve yilin gunu kalir)
    cikar: Tuple[str, ...] = ("soil_moisture",)
    maske: Tuple[str, ...] = ()      # eksik-deger gostergesi eklenen oznitelikler (bos = yok; ara deger gercek olcum sayilir)

    idw_artik: bool = False          # True: her dugume IDW_rakim tahmini girdi; model ciktisi = IDW_rakim + duzeltme
    idw_p: float = 2.0
    idw_k: int = 6
    derin_maske_orani: float = 0.0
    derin_gizli: Tuple[int, ...] = (50, 100)   # yalniz bu derinlikler gizlenir (5-20 cm gorunur kalir)
    taban: str = "idw"               # hibrit tabani: "idw" = rakim-IDW; "a3" / "a5" = rakim-IDW + gamma * (istasyonun kendi
                                     # 20 cm sapmasinin son L gun ortalamasi / erfc cekirdekli agirlikli gecmisi)
    taban_param: Tuple[Tuple[int, float, float], ...] = ()   # (derinlik, L ya da kappa [cm2/gun], gamma); ic egitimde secilir
    derin_test: bool = False         # True: son test bolumu derin_spatial_test (kat disi istasyonlar x test yillari) de yazilir
    epoch_sifir: bool = False        # True: egitimden once dogrulama kaybi kaydedilir (egitilen_epoch = 0); en iyi durum
                                     # egitilmemis model olabilir; sabit epoch 0 = egitim yok (hibritte model = taban)
    yatay: str = "kapi"
    ceza: float = 0.0                # hibrit: kayip + ceza * ortalama((cikti - taban)^2), yalniz egitimde (aday: 0,1 / 1 / 10)
    blok_maske: bool = False         # egitim: rastgele istasyon maskesi yerine cografi blok (merkez + en yakinlari)
    blok_orani: float = 0.20         # blok buyuklugu = ceil(blok_orani * istasyon sayisi)
    epoch_kayit: str = ""            # bos degilse her egitilen epoch'un agirliklari bu klasore (epoch_<e>.pt; e = 0 dahil)
    sadece_ic: bool = False

    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    window: int = 3
    attn_heads: int = 4

    lr: float = 1e-3
    batch_size: int = 16
    max_epochs: int = 200
    patience: int = 10
    station_mask_ratio: float = 0.15  # her pencerede maskelenen istasyon orani
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)

    eval_groups: int = 5              # zamansal testte istasyonlar 5 grupta sirayla maskelenir
    dedup: str = "first"
    device: str = "cpu"

    def __post_init__(self):
        if self.hidden_dim % 4 or self.hidden_dim % self.attn_heads:
            raise ValueError("hidden_dim 4'e ve attn_heads'e bolunmeli.")
        if set(self.cikar) - set(OZNITELIK[1:]):
            raise ValueError(f"Cikarilamaz oznitelik: {sorted(set(self.cikar) - set(OZNITELIK[1:]))}")
        if set(self.maske) - set(OZNITELIK) or set(self.maske) & set(self.cikar):
            raise ValueError(f"Gecersiz maske ozniteligi (bilinmeyen ya da cikarilmis): {sorted(self.maske)}")
        if self.uydu_derinlik not in ("yuzey", "tum"):
            raise ValueError(self.uydu_derinlik)
        if not 0.0 <= self.derin_maske_orani < 1.0:
            raise ValueError(f"derin_maske_orani [0, 1) olmali: {self.derin_maske_orani}")
        if set(self.derin_gizli) - {50, 100}:
            raise ValueError(f"derin_gizli yalniz 50/100 icerebilir: {self.derin_gizli}")
        if self.taban not in ("idw", "a3", "a5"):
            raise ValueError(self.taban)
        if self.taban != "idw":
            if not self.idw_artik:
                raise ValueError("taban a3/a5 yalniz hibritte (idw_artik)")
            if {int(d) for d, _, _ in self.taban_param} != set(self.derin_gizli):
                raise ValueError(f"taban_param derinlikleri derin_gizli ile ayni olmali: {self.taban_param}")
        if self.yatay not in ("kapi", "softmax"):
            raise ValueError(f"yatay 'kapi' ya da 'softmax' olmali: {self.yatay}")
        if self.ceza < 0 or (self.ceza > 0 and not self.idw_artik):
            raise ValueError(f"ceza >= 0 olmali ve yalniz hibritte (idw_artik): {self.ceza}")
        if not 0.0 < self.blok_orani < 1.0:
            raise ValueError(f"blok_orani (0, 1) olmali: {self.blok_orani}")
        if self.idw_p <= 0 or self.idw_k < 1:
            raise ValueError(f"idw_p > 0 ve idw_k >= 1 olmali: {self.idw_p}, {self.idw_k}")
        if self.dedup not in ("first", "mean"):
            raise ValueError(self.dedup)
        if set(self.train_years) & (set(self.val_years) | set(self.test_years)) or \
                set(self.val_years) & set(self.test_years):
            raise ValueError("Yil kumeleri ortusuyor.")
