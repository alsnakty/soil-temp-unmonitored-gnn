import glob
import os
import sys

import numpy as np
import pandas as pd

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 220)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))

from toprak_ayarlar import DERINLIKLER, IDX_T, SoilConfig
from toprak_gnn_calistir import hazirla
from toprak_katlar import kat_istasyonlari
from ortak_veri import haversine_km
import idw_karsilastirma_modeli as IDWM
import kriging_karsilastirma_modeli as KRM

SON = os.path.join(ANA_DIZIN, "results")
KATLI = os.path.join(SON, "folds")
IDW_P = (1, 1, 2, 2, 1)
IDW_K = (6, 6, 8, 8, 8)
ANAHTAR = ["istasyon_no", "derinlik", "tarih"]


def ayristir(df):
    """df: istasyon_no, derinlik, tarih, y_true, y_pred -> havuz MSE ve uc bilesen (C^2)."""
    d = df.assign(e=df.y_pred - df.y_true, ay=pd.to_datetime(df.tarih).dt.month)
    b = d.groupby(["istasyon_no", "derinlik"]).e.transform("mean")
    m = d.groupby(["istasyon_no", "derinlik", "ay"]).e.transform("mean")
    mse, sb, sm, sg = (d.e ** 2).mean(), (b ** 2).mean(), ((m - b) ** 2).mean(), ((d.e - m) ** 2).mean()
    if not np.isclose(mse, sb + sm + sg, rtol=1e-9):
        raise RuntimeError("ayrisma tutmadi")
    return dict(n=len(d), rmse=np.sqrt(mse), mse=mse, sabit=sb, mevsimsel=sm, gunluk=sg,
                pay_sabit=sb / mse, pay_mevsimsel=sm / mse, pay_gunluk=sg / mse,
                rms_sabit=np.sqrt(sb), rms_mevsimsel=np.sqrt(sm), rms_gunluk=np.sqrt(sg))


def vg_sozluk(yol):
    v = pd.read_csv(yol)
    out = {}
    for (dep, model), g in v.groupby(["derinlik", "model"]):
        out[(int(dep), model)] = g.sort_values("ay")[["c0", "c1", "a_km"]].to_numpy(float)
    return out


def ic_dogrulama():
    df_m = None
    parca = []
    for k in range(5):
        c = SoilConfig()
        c.uydu_derinlik = "tum"
        c.holdout, c.ic_dogrulama = kat_istasyonlari(k), True
        if df_m is None:
            df_m = pd.read_csv(c.master_path, parse_dates=["tarih"])
        P = hazirla(c, df_m)
        X, dates, stations, vi = P["X"], P["dates"], P["stations"], P["idx"]["val"]
        n_st = len(stations)
        Tobs = X[..., IDX_T].reshape(len(dates), n_st, len(DERINLIKLER)).astype(float)
        st_no = stations["istasyon_no"].to_numpy()
        lat, lon = stations["lat"].to_numpy(float), stations["lon"].to_numpy(float)
        rak = stations["rakim"].to_numpy(float) / 1000.0
        hold = P["g_all"].st_global[P["hold_local"]]
        gor = np.setdiff1d(np.arange(n_st), hold)
        ic = P["ic_global"]
        idw = IDWM.satirlar(Tobs[vi], dates[vi], st_no, lat, lon, gor, [ic], float(IDW_P[k]), IDW_K[k], rak)
        s = pd.read_csv(os.path.join(SON, f"kriging_elevation_fold{k}", "secim_2023.csv"))
        km = None if s.k[0] == "hepsi" else int(s.k[0])
        D_all = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        kr = KRM.satirlar(Tobs[vi], dates[vi], dates[vi].month.to_numpy(), st_no, D_all, gor, [ic],
                          vg_sozluk(os.path.join(SON, f"kriging_elevation_fold{k}", "variogram.csv")), s.model[0], km, rak)
        tah = {"idw_rakim": idw, "kriging_rakim": kr}
        for m in ("hibrit", "gnn"):
            tah[m] = pd.read_csv(os.path.join(SON, "fold_selection", f"{m}_fold{k}", "pred_ic_val2023_gnn_s0.csv.gz"))
        for m, t in tah.items():
            t = t.copy()
            t["tarih"] = pd.to_datetime(t.tarih)
            tah[m] = t.dropna(subset=["y_pred"])
        ortak = tah["hibrit"][ANAHTAR]
        for m in tah:
            ortak = ortak.merge(tah[m][ANAHTAR], on=ANAHTAR)
        for m, t in tah.items():
            u = t.merge(ortak, on=ANAHTAR)
            parca.append(u.assign(model=m, kat=k))
        print(f"kat {k}: ic {len(ic)} istasyon, ortak satir {len(ortak)} "
              f"(idw {len(tah['idw_rakim'])}, kriging {len(tah['kriging_rakim'])}, hibrit {len(tah['hibrit'])}, "
              f"gnn {len(tah['gnn'])})", flush=True)
    t = pd.concat(parca)
    p = t.pivot_table(index=ANAHTAR, columns="model", values="y_true")
    print("y_true yontemler arasi en buyuk fark:", float((p.max(1) - p.min(1)).max()))
    return pd.DataFrame([dict(model=m, **ayristir(g)) for m, g in t.groupby("model")])


def spatial_test():
    satir = []
    for m, kl, ek, tohum in (("idw_rakim", "idw_elevation_fold{k}", "idw", [None]),
                             ("kriging_rakim", "kriging_elevation_fold{k}", "kriging", [None]),
                             ("hibrit", "folds/hibrit_fold{k}_s{s}", "gnn_s{s}", [0, 1, 2]),
                             ("gnn", "folds/gnn_fold{k}_s{s}", "gnn_s{s}", [0, 1, 2])):
        r = []
        for s in tohum:
            df = pd.concat([pd.read_csv(os.path.join(SON, kl.format(k=k, s=s), f"pred_spatial_test_{ek.format(s=s)}.csv.gz"))
                            for k in range(5)])
            r.append(ayristir(df))
        r = pd.DataFrame(r)
        o = r[["n", "mse", "sabit", "mevsimsel", "gunluk"]].mean().to_dict()
        satir.append(dict(model=m, tohum=len(tohum), n=int(o["n"]), rmse=np.sqrt(o["mse"]), mse=o["mse"],
                          sabit=o["sabit"], mevsimsel=o["mevsimsel"], gunluk=o["gunluk"],
                          pay_sabit=o["sabit"] / o["mse"], pay_mevsimsel=o["mevsimsel"] / o["mse"],
                          pay_gunluk=o["gunluk"] / o["mse"], rms_sabit=np.sqrt(o["sabit"]),
                          rms_mevsimsel=np.sqrt(o["mevsimsel"]), rms_gunluk=np.sqrt(o["gunluk"])))
    return pd.DataFrame(satir)


def main():
    out = os.path.join(KATLI, "analysis_error_threeway")
    os.makedirs(out, exist_ok=True)
    a = ic_dogrulama()
    print("\nA) Ic dogrulama x 2023 (1. asama, tohum 0; ortak satirlar):")
    print(a.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    a.to_csv(os.path.join(out, "ic_dogrulama_2023.csv"), index=False)
    b = spatial_test()
    print("\nB) spatial_test 2024-25 (betimleyici; ogrenenlerde 3 tohum ortalamasi):")
    print(b.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    b.to_csv(os.path.join(out, "spatial_test.csv"), index=False)


if __name__ == "__main__":
    main()
