import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

pd.set_option("compute.use_numexpr", False)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))

from ortak_ayarlar import variant_opts
from gnn_model import build_model
from toprak_ayarlar import DERINLIKLER, IDX_T, MASKE_VARSAYILAN, N_FEAT, OZNITELIK, SoilConfig, edge_dims
from toprak_veri_ve_graf import SoilWindows, collate, load_master, norm_stats, normalize
from toprak_gnn_calistir import hazirla
from toprak_katlar import kat_istasyonlari

IDW_P = (1, 1, 2, 2, 1)
IDW_K = (6, 6, 8, 8, 8)
KATLI = os.path.join(ANA_DIZIN, "results", "folds")


def ayar(k, hibrit):
    c = SoilConfig()
    c.uydu_derinlik, c.n_layers, c.maske = "tum", 2, MASKE_VARSAYILAN
    c.holdout = kat_istasyonlari(k)
    if hibrit:
        c.idw_artik, c.idw_p, c.idw_k = True, float(IDW_P[k]), IDW_K[k]
    c.__post_init__()
    return c


def ileri(model, wins, starts, bs=16):
    """Secilen pencerelerin orta gunu: model ciktisi ve (varsa) taban, (n_pencere, S) normalize."""
    out, base = [], []
    with torch.no_grad():
        for a in range(0, len(starts), bs):
            b = collate([wins.window(i) for i in starts[a:a + bs]], wins.W, wins.S, wins.g.templates)
            y = model(b)[:, 0].view(b.B, b.W, b.S)[:, 1].numpy()
            out.append(y)
            if b.base is not None:
                base.append(b.base[:, 0].view(b.B, b.W, b.S)[:, 1].numpy())
    return np.concatenate(out), (np.concatenate(base) if base else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adim", type=int, default=10, help="2023 icinde pencere baslangic araligi (gun)")
    ap.add_argument("--kat", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tohum", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    out_dir = os.path.join(KATLI, "analysis_nearest_neighbour")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    c0 = SoilConfig(); c0.uydu_derinlik = "tum"
    soil = [f"{v}_{k}" for v in ("sand", "silt", "clay", "bdod", "soc", "cfvo", "wv0033", "wv1500")
            for k in ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm")]
    cols = ["istasyon_no", "tarih", "lat", "lon", "rakim", "t5cmsicaklik", "t10cmsicaklik", "t20cmsicaklik",
            "t50cmsicaklik", "t100cmsicaklik", "LST_Day_C", "LST_Night_C", "NDVI", "soil_moisture", *soil]
    df = pd.read_csv(c0.master_path, usecols=cols, parse_dates=["tarih"])
    print(f"veri okundu [{time.time() - t0:.0f} s]", flush=True)

    satir, top = [], []
    for k in a.kat:
        cg, ch = ayar(k, False), ayar(k, True)
        P = hazirla(cg, df)
        g, g_tr, X, idx = P["g_all"], P["g_tr"], P["X"], P["idx"]
        sub = lambda ti, gg: X[ti][:, gg.slot_global]
        mu, sd = norm_stats(sub(idx["train"], g_tr), g_tr.slot_depth, "level")
        Xn = normalize(sub(idx["val"], g), g, mu, sd)
        hold = P["hold_local"]
        gor = np.setdiff1d(np.arange(g.n_st), hold)
        no = g.stations["istasyon_no"].to_numpy()
        sdT = sd[g.slot_depth, IDX_T]                                           # (S,)
        dep_slot = g.slot_depth
        hedef_slot = np.isin(g.slot_station, hold) & (np.asarray(DERINLIKLER)[dep_slot] > 0)
        hs = np.nonzero(hedef_slot)[0]
        sira = np.argsort(np.argsort(np.where(np.isin(np.arange(g.n_st), gor)[None, :], g.st_dist, np.inf), axis=1), axis=1)

        def kaydir(js):
            Xp = Xn.copy()
            sl = np.nonzero(np.isin(g.slot_station, js))[0]
            Xp[:, sl, IDX_T] += (1.0 / sd[dep_slot[sl], IDX_T]).astype(np.float32)
            return Xp

        for hib in (False, True):
            ad = "hibrit" if hib else "gnn"
            cfg = ch if hib else cg
            for s in a.tohum:
                yol = os.path.join(KATLI, f"{ad}_fold{k}_s{s}", f"model_gnn_s{s}.pt")
                model = build_model(cfg, variant_opts("gnn"), in_dim=N_FEAT, target_cols=[IDX_T],
                                    edge_dims=edge_dims(cfg), maske_cols=[OZNITELIK.index(c) for c in cfg.maske],
                                    idw_artik=cfg.idw_artik)
                model.load_state_dict(torch.load(yol, map_location="cpu", weights_only=True))
                model.eval()
                wins = SoilWindows(Xn, P["dates"][idx["val"]], g, cfg, "eval", s, masked_stations=hold)
                starts = np.arange(0, len(wins), a.adim)
                y0, b0 = ileri(model, wins, starts)
                idw_yap = hib and s == a.tohum[0]                               # IDW tohumdan bagimsiz: bir kez
                for j in list(gor) + ["hepsi"]:
                    wins.Xn = kaydir(gor if j == "hepsi" else [j])
                    y1, b1 = ileri(model, wins, starts)
                    dm = np.nanmean((y1 - y0)[:, hs], 0) * sdT[hs]
                    db = np.nanmean((b1 - b0)[:, hs], 0) * sdT[hs] if idw_yap else None
                    for ad2, d in [(ad, dm)] + ([("idw_rakim", db)] if idw_yap else []):
                        tb = pd.DataFrame({"kat": k, "model": ad2, "tohum": s if ad2 != "idw_rakim" else -1,
                                           "hedef": no[g.slot_station[hs]],
                                           "derinlik": np.asarray(DERINLIKLER)[dep_slot[hs]], "agirlik": d})
                        if j == "hepsi":
                            top.append(tb)
                        else:
                            h_st = g.slot_station[hs]
                            satir.append(tb.assign(kaynak=no[j], km=g.st_dist[h_st, j], sira=sira[h_st, j] + 1))
                wins.Xn = Xn
                print(f"kat {k} {ad} s{s}: {len(gor)} kaynak, {len(starts)} pencere [{(time.time() - t0) / 60:.1f} dk]",
                      flush=True)
        pd.concat(satir).to_csv(os.path.join(out_dir, "agirlik.csv.gz"), index=False)
        pd.concat(top).to_csv(os.path.join(out_dir, "toplam.csv"), index=False)
    print("bitti", flush=True)


if __name__ == "__main__":
    main()
