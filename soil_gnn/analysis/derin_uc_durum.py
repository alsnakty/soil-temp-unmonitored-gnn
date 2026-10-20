import os
import sys

import numpy as np
import pandas as pd

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 220)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))
sys.path.insert(0, os.path.join(ANA_DIZIN, "analysis"))

from toprak_ayarlar import SPATIAL_HOLDOUT
import katli_birincil_karsilastirma as KB
from kayma_duzeltme_karsilastirmasi import ANAHTAR, OMEGA, T0, TEST0

SON = os.path.join(ANA_DIZIN, "results")
OUT = os.path.join(SON, "folds", "analysis_deep_final_test")
N_AY = 12


def harmonik_satir(df, N):
    df = df.reset_index(drop=True).assign(tarih=lambda x: pd.to_datetime(x.tarih))
    df["e"] = df.y_pred - df.y_true
    if df.duplicated(ANAHTAR).any():
        raise RuntimeError("tekrarli satir")
    bas = TEST0 - pd.DateOffset(months=N)
    w = df[(df.tarih >= bas) & (df.tarih < TEST0)]
    test = df[df.tarih >= TEST0].copy()
    test["duz"], test["kal_turu"] = 0.0, "yok"
    gw = dict(tuple(w.groupby(["istasyon_no", "derinlik"])))
    for key, idx in test.groupby(["istasyon_no", "derinlik"]).groups.items():
        g = gw.get(key)
        if g is None or len(g) == 0:
            continue
        if len(g) >= 90 and g.tarih.dt.month.nunique() >= 6:
            tt = (g.tarih - T0).dt.days.to_numpy(float)
            A = np.column_stack([np.ones(len(g)), np.sin(OMEGA * tt), np.cos(OMEGA * tt)])
            cf, *_ = np.linalg.lstsq(A, g.e.to_numpy(), rcond=None)
            t2 = (test.loc[idx, "tarih"] - T0).dt.days.to_numpy(float)
            test.loc[idx, "duz"] = cf[0] + cf[1] * np.sin(OMEGA * t2) + cf[2] * np.cos(OMEGA * t2)
            test.loc[idx, "kal_turu"] = "harmonik"
        else:
            test.loc[idx, "duz"] = g.e.mean()
            test.loc[idx, "kal_turu"] = "sabit"
    return test


def main():
    t = pd.read_csv(os.path.join(OUT, "tahminler.csv.gz"), parse_dates=["tarih"])
    parca = []
    for k in range(5):
        full = pd.read_csv(os.path.join(SON, f"idw_elevation_fold{k}", "pred_spatial_full_idw.csv.gz"))
        full = full[full.derinlik.isin((50, 100))]
        h = harmonik_satir(full, N_AY)
        x = t[t.kat == k].merge(h[ANAHTAR + ["y_pred", "duz", "kal_turu"]], on=ANAHTAR, how="left")
        if x.y_pred.isna().any() or not np.allclose(x.y_pred, x.A1, atol=1e-6):
            raise RuntimeError(f"kat {k}: spatial_full A1 ile son test A1 ayni satirda ayni degil")
        x["A1_kal12"] = x.A1 - x.duz
        parca.append(x)
    t = pd.concat(parca, ignore_index=True)
    yon = {"1 olcum yok: A1": "A1", "2 12 ay kampanya: A1 + sabit/harmonik": "A1_kal12",
           "3 kalici sig sensor: A5": "A5"}
    ist82 = sorted(t.istasyon_no.unique())
    kume = {"82": ist82, "74": [s for s in ist82 if s not in SPATIAL_HOLDOUT]}
    satir = []
    for ad, kol in yon.items():
        kl = [t[t.kat == k][ANAHTAR + ["y_true"]].assign(y_pred=t[t.kat == k][kol].to_numpy()) for k in range(5)]
        r = KB.kat_ortalama_rmse(kl)
        for kad, ist in kume.items():
            u = t[t.istasyon_no.isin(ist)]
            e = u[kol] - u.y_true
            d = {f"havuz_{dz}": float(np.sqrt((e[u.derinlik == dz] ** 2).mean())) for dz in (50, 100)}
            satir.append(dict(durum=ad, kume=kad, n_istasyon=len(ist), istasyon_rmse_ort=float(r.loc[ist].mean()),
                              havuz_rmse=float(np.sqrt((e ** 2).mean())), **d))
    tab = pd.DataFrame(satir)
    tab.to_csv(os.path.join(OUT, "uc_durum.csv"), index=False)
    tur = t.groupby(["istasyon_no", "derinlik"]).kal_turu.first().value_counts()
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nKalibrasyon turu (istasyon x derinlik):", tur.to_dict())
    mut, der = [], []
    for kol in ("A1", "A3", "A5", "A6"):
        for dz in (None, 50, 100):
            u = t if dz is None else t[t.derinlik == dz]
            kl = [u[u.kat == k][ANAHTAR + ["y_true"]].assign(y_pred=u[u.kat == k][kol].to_numpy()) for k in range(5)]
            r = KB.kat_ortalama_rmse(kl)
            for kad, ist in kume.items():
                e = u[u.istasyon_no.isin(ist)][kol] - u[u.istasyon_no.isin(ist)].y_true
                mut.append(dict(yontem=kol, derinlik="50+100" if dz is None else dz, kume=kad,
                                istasyon_rmse_ort=float(r.loc[[s for s in ist if s in r.index]].mean()),
                                havuz_rmse=float(np.sqrt((e ** 2).mean()))))
    mut = pd.DataFrame(mut)
    for dz in (50, 100):
        u = t[t.derinlik == dz]
        r = {kol: KB.kat_ortalama_rmse([u[u.kat == k][ANAHTAR + ["y_true"]].assign(y_pred=u[u.kat == k][kol].to_numpy())
                                         for k in range(5)]) for kol in ("A1", "A3", "A5", "A6")}
        for m, ref in (("A5", "A1"), ("A5", "A6"), ("A5", "A3")):
            for kad, ist in kume.items():
                ist = [s for s in ist if s in r[m].index and s in r[ref].index]
                o, _ = KB.karsilastir(r[m], r[ref], ist)
                der.append(dict(derinlik=dz, model=m, referans=ref, kume=kad, **o))
    der = pd.DataFrame(der)
    mut.to_csv(os.path.join(OUT, "mutlak_rmse.csv"), index=False)
    der.to_csv(os.path.join(OUT, "karsilastirma_derinlik.csv"), index=False)
    print("\nMutlak RMSE:")
    print(mut.pivot_table(index=["kume", "derinlik"], columns="yontem", values=["istasyon_rmse_ort", "havuz_rmse"])
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nDerinlik basina (ikincil, karar yok):")
    print(der.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
