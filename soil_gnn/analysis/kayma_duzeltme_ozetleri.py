import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 230)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOK = os.path.dirname(ANA_DIZIN)
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))

from toprak_ayarlar import DERINLIKLER, IDX_T, SoilConfig
from toprak_gnn_calistir import hazirla
from toprak_katlar import kat_istasyonlari
import idw_karsilastirma_modeli as IDWM

SON = os.path.join(ANA_DIZIN, "results")
KATLI = os.path.join(SON, "folds")
OUT = os.path.join(KATLI, "analysis_offset_correction_summary")
IDW_P = (1, 1, 2, 2, 1)
IDW_K = (6, 6, 8, 8, 8)
ANAHTAR = ["istasyon_no", "derinlik", "tarih"]
OMEGA, T0 = 2 * np.pi / 365.25, pd.Timestamp("2015-01-01")


def bolum1():
    g = pd.read_csv(os.path.join(KATLI, "analysis_spatial_full", "genlik_faz.csv"))
    k = pd.read_csv(os.path.join(KATLI, "analysis_spatial_full", "gorunur_difuzivite.csv"))
    g, k = g[g.kume == "82 istasyon"], k[k.kume == "82 istasyon"]
    yaz = lambda r, c: f"{r[c + '_medyan']:.3f} ({r[c + '_q25']:.3f}-{r[c + '_q75']:.3f})"
    out = {}
    for c in ("genlik_orani", "faz_farki_gun", "sonum_tahmin", "gecikme_tahmin_gun"):
        out[c] = g.assign(v=g.apply(lambda r: yaz(r, c), axis=1)).pivot(index="model", columns="derinlik", values="v")
    for c in ("sonum_olcum", "gecikme_olcum_gun"):
        if g.groupby("derinlik")[c + "_medyan"].agg(lambda v: v.max() - v.min()).max() > 1e-9:
            raise RuntimeError(f"{c}: olcum tarafi modeller arasinda farkli")
        out[c] = g[g.model == "idw_rakim"].set_index("derinlik").apply(lambda r: yaz(r, c), axis=1)
    out["sayilar"] = g[["model", "derinlik", "n_istasyon_derinlik", "n_olcum_yok", "n_kapsam_disi", "n_cift_olcum_yok",
                        "n_cift_kapsam_disi"]]
    for c in ("kA_olcum", "kA_tahmin", "kA_oran", "kphi_olcum", "kphi_tahmin", "kphi_oran"):
        k[c] = k.apply(lambda r: f"{r[c + '_medyan']:.3g} ({r[c + '_q25']:.3g}-{r[c + '_q75']:.3g})", axis=1)
    out["kappa"] = k[["model", "n_istasyon", "n_olcum_yok_ortak", "n_kapsam_disi_ortak", "kA_tahmin_tanimli",
                      "kA_tahmin_tanimsiz", "kphi_tahmin_tanimsiz", "kA_olcum", "kA_tahmin", "kA_oran", "kphi_olcum",
                      "kphi_tahmin", "kphi_oran"]]
    return out


def duzeltme(cal, test, harmonik):
    cal = cal.assign(tarih=pd.to_datetime(cal.tarih), e=cal.y_pred - cal.y_true).reset_index(drop=True)
    test = test.assign(tarih=pd.to_datetime(test.tarih), e=test.y_pred - test.y_true).reset_index(drop=True)
    if cal.tarih.dt.year.unique().tolist() != [2023] or test.tarih.dt.year.min() < 2024:
        raise RuntimeError("kalibrasyon/test yillari")
    test["duz"] = 0.0
    sayi = {"harmonik": 0, "sabit": 0, "yok": 0}
    gc = dict(tuple(cal.dropna(subset=["e"]).groupby(["istasyon_no", "derinlik"])))
    for key, idx in test.groupby(["istasyon_no", "derinlik"]).groups.items():
        c = gc.get(key)
        if c is None:
            sayi["yok"] += 1
            continue
        if harmonik and len(c) >= 90 and c.tarih.dt.month.nunique() >= 6:
            tt = (c.tarih - T0).dt.days.to_numpy(float)
            A = np.column_stack([np.ones(len(c)), np.sin(OMEGA * tt), np.cos(OMEGA * tt)])
            cf, *_ = np.linalg.lstsq(A, c.e.to_numpy(), rcond=None)
            t2 = (test.loc[idx, "tarih"] - T0).dt.days.to_numpy(float)
            test.loc[idx, "duz"] = cf[0] + cf[1] * np.sin(OMEGA * t2) + cf[2] * np.cos(OMEGA * t2)
            sayi["harmonik"] += 1
        else:
            test.loc[idx, "duz"] = c.e.mean()
            sayi["sabit"] += 1
    return test.assign(e_duz=test.e - test.duz), sayi


def bolum2(df_m):
    satir, sayilar, ist = [], [], {"yok": [], "sabit": [], "harmonik": []}
    for k in range(5):
        kl = os.path.join(SON, f"idw_elevation_fold{k}")
        cal = pd.read_csv(os.path.join(kl, "pred_val2023_idw.csv.gz"))
        test = pd.read_csv(os.path.join(kl, "pred_temporal_idw.csv.gz"))
        if k == 0:
            c = SoilConfig(); c.holdout = kat_istasyonlari(0)
            P = hazirla(c, df_m)
            X, dates, st, vi = P["X"], P["dates"], P["stations"], P["idx"]["val"]
            T = X[..., IDX_T].reshape(len(dates), len(st), len(DERINLIKLER)).astype(float)
            tr = P["g_tr"].st_global
            r = IDWM.satirlar(T[vi], dates[vi], st.istasyon_no.to_numpy(), st.lat.to_numpy(float), st.lon.to_numpy(float),
                              tr, [tr[g] for g in P["groups"]], float(IDW_P[0]), IDW_K[0], st.rakim.to_numpy(float) / 1000.0)
            a = cal.assign(tarih=pd.to_datetime(cal.tarih)).merge(r, on=ANAHTAR, suffixes=("", "_y"))
            if len(a) != len(cal) or len(a) != len(r):
                raise RuntimeError("sinama: satir sayisi")
            print(f"sinama kat 0: 2023 temporal rakim-IDW yeniden hesap, en buyuk fark "
                  f"{float(np.nanmax(np.abs(a.y_pred - a.y_pred_y))):.2e} ({len(a)} satir)", flush=True)
        for ad, harm in (("duzeltme yok", None), ("sabit", False), ("sabit + harmonik", True)):
            if harm is None:
                t = test.assign(e_duz=test.y_pred - test.y_true)
                s = {}
            else:
                t, s = duzeltme(cal, test, harm)
            r_ = (t.e_duz ** 2).groupby(t.istasyon_no).mean() ** 0.5
            satir.append(pd.DataFrame({"istasyon_no": r_.index, "rmse": r_.to_numpy(), "kat": k, "duzeltme": ad}))
            if s:
                sayilar.append(dict(kat=k, duzeltme=ad, **s))
    t = pd.concat(satir)
    ist_ort = t.groupby(["duzeltme", "istasyon_no"]).rmse.mean().reset_index()
    nk = t.groupby(["duzeltme", "istasyon_no"]).kat.nunique()
    if nk.min() != 4 or nk.max() != 4:
        raise RuntimeError(f"istasyon basina kat sayisi {nk.min()}-{nk.max()}")
    oz = ist_ort.groupby("duzeltme").rmse.agg(["mean", "median", "count"])
    return oz, pd.DataFrame(sayilar).groupby("duzeltme")[["harmonik", "sabit", "yok"]].sum(), ist_ort


def bolum3():
    ad = pd.read_csv(os.path.join(KOK, "DATASET", "MGM_istasyonlar_final.csv"), encoding="cp1254"
                     ).set_index("istasyon_no").istasyon_adi
    dl = pd.read_csv(os.path.join(KATLI, "analysis_offset_correction", "dlst_test_istasyon.csv")).set_index("istasyon_no")
    out = {}
    for m, kl, ek in (("idw_rakim", "idw_elevation_fold{k}", "idw"), ("kriging_rakim", "kriging_elevation_fold{k}", "kriging")):
        rs = []
        for k in range(5):
            b = pd.read_csv(os.path.join(SON, kl.format(k=k), f"pred_spatial_test_{ek}.csv.gz"))
            d = pd.read_csv(os.path.join(KATLI, "analysis_offset_correction", f"pred_spatial_test_{m}_dlst_kat{k}.csv.gz"))
            j = b.merge(d, on=ANAHTAR, suffixes=("_b", "_d"))
            if len(j) != len(b) or len(j) != len(d) or not np.allclose(j.y_true_b, j.y_true_d):
                raise RuntimeError(f"{m} kat {k}: satirlar")
            g = j.assign(eb=(j.y_pred_b - j.y_true_b) ** 2, ed=(j.y_pred_d - j.y_true_d) ** 2,
                         mb=j.y_pred_b - j.y_true_b).groupby("istasyon_no")
            rs.append(pd.DataFrame({"rmse_taban": g.eb.mean() ** 0.5, "rmse_dlst": g.ed.mean() ** 0.5,
                                    "mbe_taban": g.mb.mean(), "kat": k}))
        r = pd.concat(rs)
        r["fark"] = r.rmse_dlst - r.rmse_taban
        r = r.join(dl[["dLST_gece"]])
        r["isaret"] = np.sign(r.dLST_gece)
        r["ad"] = ad.reindex(r.index).to_numpy()
        out[m] = r
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    b1 = bolum1()
    print("1) Harmonik ozet (spatial_full, 82 istasyon kumesi; katli_birincil_karsilastirma.py ciktisi)")
    for c, t in b1.items():
        print(f"\n{c}:")
        print(t.to_string())
        (t if isinstance(t, pd.DataFrame) else t.to_frame()).to_csv(os.path.join(OUT, f"b1_{c}.csv"))
    df_m = pd.read_csv(SoilConfig().master_path, parse_dates=["tarih"])
    oz, say, ist = bolum2(df_m)
    print("\n2) Temporal duzende kalibrasyon (rakim-IDW; 2024-25 istasyon RMSE, 4 kat ortalamasi):")
    print(oz.round(3).to_string())
    print(say.to_string())
    oz.to_csv(os.path.join(OUT, "b2_temporal_kalibrasyon.csv"))
    ist.to_csv(os.path.join(OUT, "b2_istasyon.csv"), index=False)
    b3 = bolum3()
    print("\n3) dLST kazancinin dagilimi (spatial_test; fark = dLST'li - taban, eksi = kazanc):")
    for m, r in b3.items():
        r.to_csv(os.path.join(OUT, f"b3_{m}.csv"))
        rho, p = spearmanr(r.fark, r.dLST_gece.abs())
        rho2, p2 = spearmanr(r.fark, r.dLST_gece)
        print(f"\n{m}: n {len(r)}, ort fark {r.fark.mean():+.3f}, medyan {r.fark.median():+.3f}, kazanan {int((r.fark < 0).sum())}, "
              f"kaybeden {int((r.fark > 0).sum())}; Spearman(fark, |dLST|) {rho:+.2f} (p {p:.2g}); "
              f"Spearman(fark, dLST) {rho2:+.2f} (p {p2:.2g})")
        cols = ["ad", "kat", "dLST_gece", "isaret", "mbe_taban", "rmse_taban", "rmse_dlst", "fark"]
        print("  en cok kazanan 10:")
        print(r.sort_values("fark").head(10)[cols].round(3).to_string())
        print("  en cok kaybeden 10:")
        print(r.sort_values("fark", ascending=False).head(10)[cols].round(3).to_string())
        print("  isarete gore:", r.groupby("isaret").fark.agg(["count", "mean", "median"]).round(3).to_dict("index"))


if __name__ == "__main__":
    main()
