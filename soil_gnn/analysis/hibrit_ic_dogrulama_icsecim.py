import os
import sys

import numpy as np
import pandas as pd
import torch

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 220)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))

from ortak_ayarlar import variant_opts
from gnn_model import build_model
from gnn_egitim import run_epoch
from toprak_ayarlar import DERINLIKLER, IDX_T, MASKE_VARSAYILAN, N_FEAT, OZNITELIK, SoilConfig, edge_dims
from toprak_veri_ve_graf import SoilWindows, norm_stats, normalize
from toprak_gnn_calistir import hazirla
from toprak_katlar import kat_istasyonlari
import idw_karsilastirma_modeli as IDWM

IDW_P = (1, 1, 2, 2, 1)
IDW_K = (6, 6, 8, 8, 8)
SECIM = os.path.join(ANA_DIZIN, "results", "fold_selection")
OUT = os.path.join(ANA_DIZIN, "results", "folds", "analysis_inner_validation")


def ic_secim(P):
    X, dates, stations, vi = P["X"], P["dates"], P["stations"], P["idx"]["val"]
    n_st = len(stations)
    Tobs = X[..., IDX_T].reshape(len(dates), n_st, len(DERINLIKLER)).astype(float)
    st_no = stations["istasyon_no"].to_numpy()
    lat, lon = stations["lat"].to_numpy(float), stations["lon"].to_numpy(float)
    rak = stations["rakim"].to_numpy(float) / 1000.0
    tr = P["g_tr"].st_global
    if len(set(tr) & set(P["ic_global"])):
        raise RuntimeError("ic dogrulama istasyonu secim kumesinde")
    tr_gruplar = [tr[g] for g in P["groups"]]
    secim = []
    for p in IDWM.P_ADAY:
        for k in IDWM.K_ADAY:
            df = IDWM.satirlar(Tobs[vi], dates[vi], st_no, lat, lon, tr, tr_gruplar, p, k, rak)
            secim.append(dict(p=p, k="hepsi" if k is None else k, rmse_2023=IDWM.rmse(df.dropna()),
                              tahminsiz=int(df.y_pred.isna().sum())))
    secim = pd.DataFrame(secim).sort_values("rmse_2023").reset_index(drop=True)
    return secim


def egitimsiz_kayip(c, P):
    X, idx, g_tr, g_val = P["X"], P["idx"], P["g_tr"], P["g_val"]
    sub = lambda ti, g: X[ti][:, g.slot_global]
    mu, sd = norm_stats(sub(idx["train"], g_tr), g_tr.slot_depth, "level")
    val_w = SoilWindows(normalize(sub(idx["val"], g_val), g_val, mu, sd), P["dates"][idx["val"]],
                        g_val, c, "eval", 0, masked_stations=P["val_set"])
    torch.manual_seed(0)
    m = build_model(c, variant_opts("gnn"), in_dim=N_FEAT, target_cols=[IDX_T], edge_dims=edge_dims(c),
                    maske_cols=[OZNITELIK.index(x) for x in c.maske], idw_artik=True)
    assert float(m.out[-1].weight.abs().sum()) == 0.0 and float(m.out[-1].bias.abs().sum()) == 0.0
    return run_epoch(m, val_w, c)


def main():
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    eski = pd.read_csv(os.path.join(OUT, "hibrit_ic_dogrulama_egrisi.csv"))
    df = None
    satir = []
    for k in range(5):
        c = SoilConfig()
        c.uydu_derinlik, c.n_layers, c.maske = "tum", 2, MASKE_VARSAYILAN
        c.holdout, c.ic_dogrulama = kat_istasyonlari(k), True
        if df is None:
            df = pd.read_csv(c.master_path, parse_dates=["tarih"])
        P = hazirla(c, df)
        sec = ic_secim(P)
        sec.to_csv(os.path.join(OUT, f"ic_secim_2023_kat{k}.csv"), index=False)
        p_yeni = float(sec.p[0]); k_yeni = None if sec.k[0] == "hepsi" else int(sec.k[0])
        c.idw_artik, c.idw_p, c.idw_k = True, p_yeni, (10 ** 6 if k_yeni is None else k_yeni)
        c.__post_init__()
        l_yeni = egitimsiz_kayip(c, P)
        c2 = SoilConfig()
        c2.uydu_derinlik, c2.n_layers, c2.maske = "tum", 2, MASKE_VARSAYILAN
        c2.holdout, c2.ic_dogrulama = kat_istasyonlari(k), True
        c2.idw_artik, c2.idw_p, c2.idw_k = True, float(IDW_P[k]), IDW_K[k]
        c2.__post_init__()
        l_eski = egitimsiz_kayip(c2, P)
        for ad in ("hibrit", "hibrit_konumyok"):
            e = eski[(eski.model == ad) & (eski.kat == k)].iloc[0]
            satir.append(dict(model=ad, kat=k, n_ic=int(e.n_ic), pk_eski=f"{IDW_P[k]},{IDW_K[k]}",
                              pk_ic=f"{p_yeni:g},{'hepsi' if k_yeni is None else k_yeni}",
                              kayip_0_eski=l_eski, kayip_0_ic=l_yeni, en_iyi_epoch=int(e.en_iyi_epoch),
                              kayip_en_iyi=e.kayip_en_iyi, goreli_iyilesme_ic=(l_yeni - e.kayip_en_iyi) / l_yeni,
                              egitimsiz_iyi=bool(l_yeni < e.kayip_en_iyi),
                              sinama_eski_fark=l_eski - e.kayip_0_rakimIDW))
        print(satir[-2], flush=True)
    t = pd.DataFrame(satir)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    t.to_csv(os.path.join(OUT, "hibrit_ic_dogrulama_icsecim.csv"), index=False)
    print("\nEgitimsiz taban en iyi epoch'tan iyi (hibrit):", t[(t.model == "hibrit") & t.egitimsiz_iyi].kat.tolist())


if __name__ == "__main__":
    main()
