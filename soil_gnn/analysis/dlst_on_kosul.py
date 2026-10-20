import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 220)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))

from toprak_ayarlar import DERINLIKLER, IDX_T, OZNITELIK, SoilConfig
from toprak_gnn_calistir import hazirla
from toprak_katlar import kat_istasyonlari
from ortak_veri import haversine_km
import idw_karsilastirma_modeli as IDWM

IDW_P = (1, 1, 2, 2, 1)
IDW_K = (6, 6, 8, 8, 8)
OUT = os.path.join(ANA_DIZIN, "results", "folds", "analysis_dlst")
DER = (5, 10, 20, 50, 100)


def loo_r2(y, X):
    """Istasyon birakmali dogrusal fit (sabit + X). Donus (R2_LOO, R2_orneklem_ici)."""
    A = np.column_stack([np.ones(len(y)), X])
    tah = np.empty(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        c, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        tah[i] = A[i] @ c
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - tah) ** 2).sum() / ss, 1 - ((y - A @ c) ** 2).sum() / ss


def main():
    os.makedirs(OUT, exist_ok=True)
    df = None
    satir, ist_tablo = [], []
    for k in range(5):
        c = SoilConfig()
        c.uydu_derinlik = "tum"                       # LST her derinlik dugumunde; yuzey dugumunden okunur
        c.holdout = kat_istasyonlari(k)
        if df is None:
            df = pd.read_csv(c.master_path, parse_dates=["tarih"])
        P = hazirla(c, df)
        X, dates, stations, ti = P["X"], P["dates"], P["stations"], P["idx"]["train"]
        if dates[ti].year.min() != 2015 or dates[ti].year.max() != 2022:
            raise RuntimeError("egitim yillari 2015-2022 degil")
        n_st = len(stations)
        X4 = X.reshape(len(dates), n_st, len(DERINLIKLER), -1)
        Tobs = X4[..., IDX_T].astype(float)
        st_no = stations["istasyon_no"].to_numpy()
        lat, lon = stations["lat"].to_numpy(float), stations["lon"].to_numpy(float)
        rak = stations["rakim"].to_numpy(float) / 1000.0
        tr = P["g_tr"].st_global                                 # katin egitim istasyonlari (test kati yok)
        if set(st_no[tr]) & set(c.holdout):
            raise RuntimeError("test kati egitim istasyonlarinda")
        # LST: yuzey dugumu (derinlik 0); 2015-2022 olculmus gunlerin ortalamasi
        lst = {}
        for ad in ("LST_Day_C", "LST_Night_C"):
            v = X4[ti][:, :, 0, OZNITELIK.index(ad)].astype(float)
            lst[ad] = np.nanmean(v[:, tr], axis=0)
        # dLST: duz IDW, i haric katin diger egitim istasyonlari
        D = haversine_km(lat[tr][:, None], lon[tr][:, None], lat[tr][None, :], lon[tr][None, :])
        dl = {}
        for ad, L in lst.items():
            idw_v = np.empty(len(tr))
            for i in range(len(tr)):
                m = np.arange(len(tr)) != i
                idw_v[i] = IDWM.idw(L[m][None, :], D[i:i + 1, m], float(IDW_P[k]), IDW_K[k])[0, 0]
            dl[ad] = L - idw_v
        # kayma: rakim-IDW, i gizli, digerleri gorunur, 2015-2022
        r = IDWM.satirlar(Tobs[ti], dates[ti], st_no, lat, lon, tr, [np.array([i]) for i in tr],
                          float(IDW_P[k]), IDW_K[k], rak)
        r = r.dropna(subset=["y_pred"])
        kay = (r.y_pred - r.y_true).groupby([r.istasyon_no, r.derinlik]).mean().unstack("derinlik")
        t = pd.DataFrame({"istasyon_no": st_no[tr], "dLST_gunduz": dl["LST_Day_C"], "dLST_gece": dl["LST_Night_C"]}
                         ).set_index("istasyon_no").join(kay)
        ist_tablo.append(t.assign(kat=k))
        for d in DER:
            x = t[["dLST_gunduz", "dLST_gece", d]].dropna()
            y = x[d].to_numpy()
            for ad, cols in (("gece", ["dLST_gece"]), ("gunduz", ["dLST_gunduz"]), ("gece+gunduz", ["dLST_gece", "dLST_gunduz"])):
                r2l, r2 = loo_r2(y, x[cols].to_numpy())
                rho = spearmanr(x[cols[0]], y)[0] if len(cols) == 1 else np.nan
                satir.append(dict(kat=k, derinlik=d, ozellik=ad, n=len(y), R2_LOO=r2l, R2_ici=r2, spearman=rho))
        print(f"kat {k}: {len(tr)} egitim istasyonu, kayma satiri {len(r)}", flush=True)
    s = pd.DataFrame(satir)
    s.to_csv(os.path.join(OUT, "dlst_r2.csv"), index=False)
    pd.concat(ist_tablo).to_csv(os.path.join(OUT, "dlst_istasyon.csv"))
    for oz in ("gece", "gunduz", "gece+gunduz"):
        x = s[s.ozellik == oz]
        print(f"\n{oz}: R2_LOO (kat x derinlik)")
        print(x.pivot(index="kat", columns="derinlik", values="R2_LOO").round(3).to_string())
        print("  kat ortalamasi: R2_LOO", x.groupby("derinlik").R2_LOO.mean().round(3).to_dict(),
              "| R2_ici", x.groupby("derinlik").R2_ici.mean().round(3).to_dict(),
              "| Spearman", x.groupby("derinlik").spearman.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
