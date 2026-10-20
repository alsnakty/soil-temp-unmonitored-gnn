import os
import sys

import numpy as np
import pandas as pd

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 220)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))

from toprak_ayarlar import DERINLIKLER, IDX_T, OZNITELIK, SoilConfig
from toprak_gnn_calistir import hazirla
from toprak_katlar import kat_istasyonlari
from ortak_veri import haversine_km
import idw_karsilastirma_modeli as IDWM
import kriging_karsilastirma_modeli as KRM

SON = os.path.join(ANA_DIZIN, "results")
OUT = os.path.join(SON, "folds", "analysis_structural_correction")
IDW_P = (1, 1, 2, 2, 1)
IDW_K = (6, 6, 8, 8, 8)
DER = (5, 10, 20, 50, 100)
OMEGA, T0 = 2 * np.pi / 365.25, pd.Timestamp("2015-01-01")
ALFA = np.logspace(-3, 4, 29)
OZ = ["dLSTn_ort", "dLSTd_ort", "dLSTn_sin", "dLSTn_cos", "dNDVI_ort", "dNDVI_sin", "dNDVI_cos", "drakim"]


def harmonik(tarih, y):
    ok = np.isfinite(y)
    t = (tarih[ok] - T0).days.to_numpy(float) if hasattr(tarih, "days") else (pd.DatetimeIndex(tarih[ok]) - T0).days.to_numpy(float)
    A = np.column_stack([np.ones(ok.sum()), np.sin(OMEGA * t), np.cos(OMEGA * t)])
    cf, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    return cf


def vg_sozluk(yol):
    v = pd.read_csv(yol)
    return {(int(d), m): g.sort_values("ay")[["c0", "c1", "a_km"]].to_numpy(float) for (d, m), g in v.groupby(["derinlik", "model"])}


def ridge_loo_kapali(X, y, alfa):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    yc = y - y.mean()
    n = len(y)
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    rss = []
    for a in alfa:
        d = s ** 2 / (s ** 2 + a)
        H = (U * d) @ U.T + 1.0 / n                          # sabit dahil sapka matrisi
        yhat = (U * d) @ (U.T @ yc) + y.mean()
        h = np.diag(H)
        rss.append(np.sum(((y - yhat) / (1 - h)) ** 2))
    return np.array(rss)


def ridge_fit_tahmin(X, y, a, x0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + a * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return y.mean() + ((x0 - mu) / sd) @ w


def ic_ice_loo(X, y):
    """Dis LOO tahminleri; her dis adimda ceza kalan istasyonlarda kapali bicim LOO ile secilir."""
    n = len(y)
    tah, alfalar = np.empty(n), np.empty(n)
    for i in range(n):
        m = np.arange(n) != i
        a = ALFA[np.argmin(ridge_loo_kapali(X[m], y[m], ALFA))]
        tah[i], alfalar[i] = ridge_fit_tahmin(X[m], y[m], a, X[i]), a
    return tah, alfalar


def kat_hesap(k, df_m):
    c = SoilConfig(); c.holdout = kat_istasyonlari(k)
    P = hazirla(c, df_m)
    X, dates, st = P["X"], P["dates"], P["stations"]
    n_st = len(st)
    X4 = X.reshape(len(dates), n_st, len(DERINLIKLER), -1)
    T = X4[..., IDX_T].astype(float)
    st_no, lat, lon = st.istasyon_no.to_numpy(), st.lat.to_numpy(float), st.lon.to_numpy(float)
    rak = st.rakim.to_numpy(float) / 1000.0
    tr = P["g_tr"].st_global
    if set(st_no[tr]) & set(c.holdout):
        raise RuntimeError("test kati egitimde")
    p, kk = float(IDW_P[k]), IDW_K[k]
    ti = np.nonzero(dates.year <= 2022)[0]
    t15 = np.nonzero(dates.year <= 2023)[0]
    # istasyon duzeyi uydu nicelikleri, 2015-2022 (yuzey dugumu)
    q = {}
    Ln = X4[ti][:, :, 0, OZNITELIK.index("LST_Night_C")].astype(float)
    Ld = X4[ti][:, :, 0, OZNITELIK.index("LST_Day_C")].astype(float)
    Nd = X4[ti][:, :, 0, OZNITELIK.index("NDVI")].astype(float)
    q["LSTn_ort"], q["LSTd_ort"], q["NDVI_ort"] = np.nanmean(Ln, 0), np.nanmean(Ld, 0), np.nanmean(Nd, 0)
    hn = np.array([harmonik(dates[ti], Ln[:, j]) for j in range(n_st)])
    hv = np.array([harmonik(dates[ti], Nd[:, j]) for j in range(n_st)])
    q["LSTn_sin"], q["LSTn_cos"], q["NDVI_sin"], q["NDVI_cos"] = hn[:, 1], hn[:, 2], hv[:, 1], hv[:, 2]
    q["rakim"] = st.rakim.to_numpy(float)
    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    F = {}
    for ad, v in q.items():
        F["d" + ad] = np.array([v[i] - IDWM.idw(v[np.setdiff1d(tr, [i])][None, :], D[i:i + 1, np.setdiff1d(tr, [i])], p, kk)[0, 0]
                                for i in tr])
    F = pd.DataFrame(F, index=st_no[tr])[OZ]
    # kriging LOO 2015-2023
    kl = os.path.join(SON, f"kriging_elevation_fold{k}")
    s = pd.read_csv(os.path.join(kl, "secim_2023.csv"))
    km = None if s.k[0] == "hepsi" else int(s.k[0])
    r = KRM.satirlar(T[t15], dates[t15], dates[t15].month.to_numpy(), st_no, D, tr, [np.array([i]) for i in tr],
                     vg_sozluk(os.path.join(kl, "variogram.csv")), s.model[0], km, rak).dropna(subset=["y_pred"])
    r["e"] = r.y_pred - r.y_true
    r["tarih"] = pd.to_datetime(r.tarih)
    gecmis, d23 = r[r.tarih.dt.year <= 2022], r[r.tarih.dt.year == 2023]
    hedef = []
    for (sno, dz), g in gecmis.groupby(["istasyon_no", "derinlik"]):
        if len(g) >= 90 and g.tarih.dt.month.nunique() >= 6:
            cf = harmonik(pd.DatetimeIndex(g.tarih), g.e.to_numpy())
            hedef.append(dict(istasyon_no=sno, derinlik=dz, c=cf[0], a=cf[1], b=cf[2], kayma=g.e.mean(), n=len(g)))
    hedef = pd.DataFrame(hedef)
    return F, hedef, d23


def main():
    os.makedirs(OUT, exist_ok=True)
    df_m = pd.read_csv(SoilConfig().master_path, parse_dates=["tarih"])
    r2, deg, oz_tab = [], [], []
    for k in range(5):
        F, hedef, d23 = kat_hesap(k, df_m)
        oz_tab.append(F.assign(kat=k))
        for dz in DER:
            h = hedef[hedef.derinlik == dz].set_index("istasyon_no")
            ist = [s for s in F.index if s in h.index]
            Xm = F.loc[ist].to_numpy(float)
            tah = {}
            for par in ("c", "a", "b"):
                y = h.loc[ist, par].to_numpy(float)
                yh, al = ic_ice_loo(Xm, y)
                r2.append(dict(kat=k, derinlik=dz, param=par, n=len(y),
                               R2_LOO=1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum(), alfa_medyan=float(np.median(al))))
                tah[par] = pd.Series(yh, index=ist)
            # gece dLST'li kriging: kayma ~ alfa + beta * dLSTn_ort, istasyon birakmali
            yk = h.loc[ist, "kayma"].to_numpy(float); xk = F.loc[ist, "dLSTn_ort"].to_numpy(float)
            dl = {}
            for j, sno in enumerate(ist):
                m = np.arange(len(ist)) != j
                A = np.column_stack([np.ones(m.sum()), xk[m]])
                (a0, b0), *_ = np.linalg.lstsq(A, yk[m], rcond=None)
                dl[sno] = a0 + b0 * xk[j]
            g = d23[(d23.derinlik == dz) & d23.istasyon_no.isin(ist)].copy()
            tt = (g.tarih - T0).dt.days.to_numpy(float)
            cs = g.istasyon_no.map(tah["c"]).to_numpy(); as_ = g.istasyon_no.map(tah["a"]).to_numpy(); bs = g.istasyon_no.map(tah["b"]).to_numpy()
            g["kriging"] = g.y_pred
            g["kriging_yapisal"] = g.y_pred - (cs + as_ * np.sin(OMEGA * tt) + bs * np.cos(OMEGA * tt))
            g["kriging_dLST"] = g.y_pred - g.istasyon_no.map(dl).to_numpy()
            deg.append(g.assign(kat=k))
        print(f"kat {k} tamam ({len(F)} istasyon)", flush=True)
    r2 = pd.DataFrame(r2)
    r2.to_csv(os.path.join(OUT, "r2_loo.csv"), index=False)
    pd.concat(oz_tab).to_csv(os.path.join(OUT, "oznitelikler.csv"))
    print("\nR2_LOO (kat ortalamasi; parantez icinde katlarin min-max):")
    o = r2.groupby(["param", "derinlik"]).R2_LOO.agg(["mean", "min", "max"]).round(3)
    print(o.to_string())
    print("\nR2_LOO kat x derinlik:")
    print(r2.pivot_table(index=["param", "kat"], columns="derinlik", values="R2_LOO").round(3).to_string())
    g = pd.concat(deg, ignore_index=True)
    satir = []
    for dz, x in g.groupby("derinlik"):
        for kol in ("kriging", "kriging_dLST", "kriging_yapisal"):
            e = x[kol] - x.y_true
            ist = ((e ** 2).groupby([x.kat, x.istasyon_no]).mean() ** 0.5).groupby(level=1).mean()
            satir.append(dict(derinlik=dz, yontem=kol, n_satir=len(x), n_istasyon=len(ist),
                              havuz_rmse=float(np.sqrt((e ** 2).mean())), istasyon_rmse_ort=float(ist.mean())))
    e_all = pd.concat([pd.DataFrame(dict(derinlik="hepsi", yontem=kol, n_satir=len(g), n_istasyon=g.istasyon_no.nunique(),
                                         havuz_rmse=float(np.sqrt(((g[kol] - g.y_true) ** 2).mean())),
                                         istasyon_rmse_ort=float(((((g[kol] - g.y_true) ** 2).groupby([g.kat, g.istasyon_no]).mean() ** 0.5)
                                                                  .groupby(level=1).mean()).mean())), index=[0])
                       for kol in ("kriging", "kriging_dLST", "kriging_yapisal")])
    t = pd.concat([pd.DataFrame(satir), e_all], ignore_index=True)
    print("\n2023, birakilan egitim istasyonlari (ayni satirlar):")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    t.to_csv(os.path.join(OUT, "rmse_2023.csv"), index=False)


if __name__ == "__main__":
    main()
