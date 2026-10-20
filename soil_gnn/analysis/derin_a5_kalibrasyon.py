import os
import sys

import numpy as np
import pandas as pd

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 220)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))
sys.path.insert(0, os.path.join(ANA_DIZIN, "analysis"))

import derin_son_test as D
from derin_uc_durum import harmonik_satir
import katli_birincil_karsilastirma as KB

E, E2 = D.E, D.E2
OUT = os.path.join(ANA_DIZIN, "results", "folds", "analysis_deep_final_test")
ANAHTAR, SIG, HEDEF = D.ANAHTAR, D.SIG, D.HEDEF


def a5_kat(c, df_m, p, k):
    P = E.hazirla(c, df_m)
    T, dates, st_no, lat, lon, rak = E.veri(P)
    tr = P["g_tr"].st_global
    hold = P["g_all"].st_global[P["hold_local"]]
    ti_tr = P["idx"]["train"]
    ti_ev = np.concatenate([P["idx"]["val"], P["idx"]["test"]])
    gecmis = np.nonzero(dates <= dates[ti_tr].max())[0]
    tum = np.arange(len(dates))
    iz = [list(E.DERINLIKLER).index(d) for d in HEDEF]
    T_g = T.copy()
    T_g[np.ix_(np.arange(T.shape[0]), hold, iz)] = np.nan
    rs_tr = D.sapma(T_g, dates, st_no, lat, lon, rak, tr, tr, p, k, gecmis, SIG + HEDEF)
    rs = {dz: rs_tr[rs_tr.derinlik == dz][["istasyon_no", "tarih", "r"]] for dz in SIG}
    a1 = E.IDWM.satirlar(T[ti_ev], dates[ti_ev], st_no, lat, lon, tr, [hold], p, k, rak)
    a1 = a1[a1.derinlik.isin(HEDEF)].dropna(subset=["y_true"])
    rs_te = D.sapma(T_g, dates, st_no, lat, lon, rak, tr, hold, p, k, tum, SIG)
    rh = {dz: rs_te[rs_te.derinlik == dz][["istasyon_no", "tarih", "r"]] for dz in SIG}
    out = []
    for d in HEDEF:
        t = E.artik_tablosu(rs[20], rs_tr[rs_tr.derinlik == d].copy())
        t5 = E2.a5_ozellik(rs[20], t, E2.DZ[d])
        en = None
        for kp in E2.KAPPA:
            S = E.toplamlar(t5, f"s_{kp}").sum()
            g = S.smr / S.smm
            rss = S.srr - 2 * g * S.smr + g ** 2 * S.smm
            if en is None or rss < en[0]:
                en = (rss, kp, g)
        x = a1[a1.derinlik == d].copy()
        x["r"] = np.nan
        x = E2.a5_ozellik(rh[20], x, E2.DZ[d])
        x["A5"] = x.y_pred + en[2] * x[f"s_{en[1]}"].fillna(0.0)
        out.append(x[ANAHTAR + ["y_true", "A5"]])
    return pd.concat(out, ignore_index=True)


def main():
    t = pd.read_csv(os.path.join(OUT, "tahminler.csv.gz"), parse_dates=["tarih"])
    df_m = pd.read_csv(D.SoilConfig().master_path, parse_dates=["tarih"])
    IDW_P, IDW_K = (1, 1, 2, 2, 1), (6, 6, 8, 8, 8)
    parca = []
    for kn in range(5):
        c = D.SoilConfig(holdout=D.kat_istasyonlari(kn))
        c.__post_init__()
        a = a5_kat(c, df_m, float(IDW_P[kn]), IDW_K[kn])
        ref = t[t.kat == kn]
        j = ref[ANAHTAR + ["A5"]].merge(a[ANAHTAR + ["A5"]], on=ANAHTAR, how="left", suffixes=("", "_yeni"))
        fark = float(np.abs(j.A5 - j.A5_yeni).max())
        if j.A5_yeni.isna().any() or fark > 1e-9:
            raise RuntimeError(f"kat {kn}: 2024-25 A5 son testle ayni degil (fark {fark})")
        h = harmonik_satir(a.rename(columns={"A5": "y_pred"}), 12)
        x = ref.merge(h[ANAHTAR + ["duz", "kal_turu"]], on=ANAHTAR, how="left")
        if x.duz.isna().any():
            raise RuntimeError("kalibrasyon satiri eksik")
        x["A5_kal12"] = x.A5 - x.duz
        parca.append(x)
        print(f"kat {kn}: 2024-25 A5 son testle ayni (en buyuk fark {fark:.1e}), {len(x)} satir", flush=True)
    t = pd.concat(parca, ignore_index=True)
    ist82 = sorted(t.istasyon_no.unique())
    kume = {"82": ist82, "74": [s for s in ist82 if s not in D.SPATIAL_HOLDOUT]}
    satir = []
    for ad, kol in (("3 kalici sig sensor: A5", "A5"),
                    ("4 12 ay derin olcum + kalici sig sensor: A5 + sabit/harmonik", "A5_kal12")):
        kl = [t[t.kat == k][ANAHTAR + ["y_true"]].assign(y_pred=t[t.kat == k][kol].to_numpy()) for k in range(5)]
        r = KB.kat_ortalama_rmse(kl)
        for kad, ist in kume.items():
            u = t[t.istasyon_no.isin(ist)]
            e = u[kol] - u.y_true
            satir.append(dict(durum=ad, kume=kad, n_istasyon=len(ist), istasyon_rmse_ort=float(r.loc[ist].mean()),
                              havuz_rmse=float(np.sqrt((e ** 2).mean())),
                              **{f"havuz_{dz}": float(np.sqrt((e[u.derinlik == dz] ** 2).mean())) for dz in HEDEF}))
    tab = pd.DataFrame(satir)
    tab.to_csv(os.path.join(OUT, "uc_durum_4satir.csv"), index=False)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("Kalibrasyon turu (istasyon x derinlik):", t.groupby(["istasyon_no", "derinlik"]).kal_turu.first().value_counts().to_dict())


if __name__ == "__main__":
    main()
