import os
import sys

import numpy as np
import pandas as pd
import torch

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 220)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOK = os.path.dirname(ANA_DIZIN)
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))
from toprak_ayarlar import SoilConfig
from toprak_katlar import kat_atamasi
from ortak_veri import station_pairs

KATLI = os.path.join(ANA_DIZIN, "results", "folds")
OUT = os.path.join(KATLI, "analysis_edge_gate")
IDW_P = (1, 1, 2, 2, 1)
sig = lambda x: 1.0 / (1.0 + np.exp(-x))


def kapi():
    satir = []
    for m in ("gnn", "hibrit"):
        for k in range(5):
            for s in range(3):
                sd = torch.load(os.path.join(KATLI, f"{m}_fold{k}_s{s}", f"model_gnn_s{s}.pt"), map_location="cpu",
                                weights_only=True)
                L = len([x for x in sd if x.endswith(".H.lin_edge.weight")])
                for l in range(L):
                    W = sd[f"convs.{l}.H.lin_edge.weight"].numpy().astype(float)        # (d, 2)
                    b = sd[f"convs.{l}.H.lin_edge.bias"].numpy().astype(float)
                    if W.shape[1] != 2:
                        raise RuntimeError("H kenar ozniteligi 2 boyutlu degil")
                    r = sig(W @ [0.025, 0.5] + b) / sig(W @ [0.100, 0.5] + b)
                    fw = torch.softmax(sd[f"fusion_w.{l}"], 0).numpy()
                    satir.append(dict(model=m, kat=k, tohum=s, katman=l, r_medyan=np.median(r), r_p90=np.percentile(r, 90),
                                      r_max=r.max(), r_min=r.min(), w_uzaklik_max=np.abs(W[:, 0]).max(),
                                      w_uzaklik_medyan=np.median(np.abs(W[:, 0])), w_yon_max=np.abs(W[:, 1]).max(),
                                      w_yon_medyan=np.median(np.abs(W[:, 1])), alfa_V=fw[0], alfa_H=fw[1], alfa_T=fw[2],
                                      idw_ref=4.0 ** IDW_P[k]))
    return pd.DataFrame(satir)


def uzaklik():
    st = pd.read_csv(os.path.join(KOK, "DATASET", "MGM_istasyonlar_final.csv"), encoding="latin-1")
    kat = kat_atamasi()
    st = st[st.istasyon_no.isin(list(kat))].reset_index(drop=True)
    assert len(st) == 82, len(st)
    cfg = SoilConfig()
    satir = []

    def ozet(ad, k, s_):
        src, dst, attr = station_pairs(s_.reset_index(drop=True), cfg)
        d = attr[:, 0] * 1000.0
        satir.append(dict(graf=ad, kat=k, n_istasyon=len(s_), n_yonlu_cift=len(d), medyan=np.median(d),
                          p10=np.percentile(d, 10), p90=np.percentile(d, 90), en_buyuk=d.max(),
                          n_200km_disi=int((d > cfg.horiz_radius_km).sum()), ort_derece=len(d) / len(s_)))
    ozet("g_all", -1, st)
    for k in range(5):
        ozet("g_tr (2. asama)", k, st[st.istasyon_no.map(kat) != k])
    return pd.DataFrame(satir)


def main():
    os.makedirs(OUT, exist_ok=True)
    t = kapi()
    t.to_csv(os.path.join(OUT, "kapi_orani.csv"), index=False)
    g = t.groupby(["model", "katman"]).agg(r_medyan=("r_medyan", "median"), r_p90=("r_p90", "median"), r_max=("r_max", "max"),
                                           r_min=("r_min", "min"), w_uzaklik_max=("w_uzaklik_max", "max"),
                                           w_uzaklik_medyan=("w_uzaklik_medyan", "median"), w_yon_max=("w_yon_max", "max"),
                                           w_yon_medyan=("w_yon_medyan", "median"), alfa_H=("alfa_H", "median"))
    print("a) Kapi orani r = sigmoid(25 km) / sigmoid(100 km), 15 model (5 kat x 3 tohum) uzerinden:")
    print("   r_medyan, r_p90: model basina degerlerin medyani; r_max/r_min: 15 modelin en buyugu/en kucugu;")
    print("   w: lin_edge agirliklari (uzaklik sutunu birimi 1/1000 km, yon sutunu birimi 1/360 derece)")
    print(g.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n   Referans rakim-IDW (100/25)^p: p=1 -> 4, p=2 -> 16 (kat 0,1,4: 4; kat 2,3: 16)")
    print("\n   Kat x tohum (r_medyan / r_max):")
    print(t.pivot_table(index=["model", "katman"], columns=["kat", "tohum"], values="r_max").round(3).to_string())
    u = uzaklik()
    u.to_csv(os.path.join(OUT, "kenar_uzaklik.csv"), index=False)
    print("\nb) Yatay kenar uzakligi (km), yonlu ciftler (200 km yaricap U kNN4, simetrik):")
    print(u.to_string(index=False, float_format=lambda v: f"{v:.1f}"))


if __name__ == "__main__":
    main()
