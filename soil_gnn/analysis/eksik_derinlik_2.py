import os
import sys

import numpy as np
import pandas as pd
from scipy.special import erfc

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 220)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))
sys.path.insert(0, os.path.join(ANA_DIZIN, "analysis"))

import eksik_derinlik as E
from hata_ayristirma_uclu import ayristir

OUT = os.path.join(ANA_DIZIN, "results", "folds", "analysis_missing_depth_2")
KAPPA = (150, 250, 350, 450, 600, 800, 1000)
DZ = {50: 30.0, 100: 80.0}
L6 = (1, 3, 7, 15, 30, 60, 120)
SIG = (5, 10, 20)
ALFA_GORELI = np.logspace(-7, 0, 15)          # ceza = alfa_goreli * egitim satir sayisi (standart oznitelikte)


def agirlik(kappa, dz, K=365):
    k = np.arange(K, dtype=float)
    ust = erfc(dz / (2 * np.sqrt(kappa * (k + 1))))
    alt = np.where(k > 0, erfc(dz / (2 * np.sqrt(kappa * np.maximum(k, 1e-12)))), 0.0)
    w = ust - alt
    return w / w.sum()


def gunluk_seri(r, s, gun):
    a = r[r.istasyon_no == s].set_index("tarih").r
    return a.reindex(gun).to_numpy(float)


def a5_ozellik(r20, rd, dz):
    """rd'ye her kappa icin s_kappa sutunu ekler."""
    W = {kp: agirlik(kp, dz) for kp in KAPPA}
    out = []
    for s, g in rd.groupby("istasyon_no"):
        gun = pd.date_range(min(r20.tarih.min(), g.tarih.min()), max(r20.tarih.max(), g.tarih.max()))
        x = gunluk_seri(r20, s, gun)
        var = np.isfinite(x).astype(float)
        x0 = np.where(var > 0, x, 0.0)
        g = g.copy()
        pos = gun.get_indexer(g.tarih)
        for kp, w in W.items():
            pay = np.convolve(x0, w)[:len(gun)]
            payda = np.convolve(var, w)[:len(gun)]
            s_ = np.where(payda > 1e-12, pay / np.where(payda > 1e-12, payda, 1), np.nan)
            g[f"s_{kp}"] = s_[pos]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def a5_tahmin(t):
    rss, gam = {}, {}
    for kp in KAPPA:
        S = E.toplamlar(t, f"s_{kp}")
        tot = S.sum()
        g = (tot.smr - S.smr) / (tot.smm - S.smm)
        gam[kp] = g
        rss[kp] = (tot.srr - S.srr) - 2 * g * (tot.smr - S.smr) + g ** 2 * (tot.smm - S.smm)
    sec = pd.DataFrame(rss).idxmin(axis=1)
    t = t.copy()
    t["kappa_A5"] = sec.reindex(t.istasyon_no).to_numpy()
    t["gamma_A5"] = [gam[kp][s] for s, kp in zip(t.istasyon_no, t.kappa_A5)]
    sk = np.choose(np.searchsorted(KAPPA, t.kappa_A5.to_numpy()), [t[f"s_{kp}"].fillna(0.0).to_numpy() for kp in KAPPA])
    t["A5"] = t.y_pred + t.gamma_A5 * sk
    return t


def a6_ozellik(rs, rd):
    """rs: {derinlik: 5/10/20 sapma satirlari}; rd'ye 21 oznitelik ekler (olcum yoksa 0)."""
    out = []
    for s, g in rd.groupby("istasyon_no"):
        g = g.copy()
        bas = min(min(x.tarih.min() for x in rs.values()), g.tarih.min())
        son = max(max(x.tarih.max() for x in rs.values()), g.tarih.max())
        gun = pd.date_range(bas, son)
        for dz, r in rs.items():
            a = pd.Series(gunluk_seri(r, s, gun), index=gun)
            for L in L6:
                g[f"f{dz}_{L}"] = a.rolling(L, min_periods=1).mean().reindex(g.tarih).fillna(0.0).to_numpy()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def ridge_sumdan(G, b, yy, a_gor):
    """G = [1 X]'[1 X], b = [1 X]'y (toplamlar). Sabit cezasiz, oznitelikler bu kumede standart. Donus beta = [c, w]."""
    n = G[0, 0]
    sx = G[0, 1:]
    mu = sx / n
    var = np.diag(G[1:, 1:]) / n - mu ** 2
    sd = np.sqrt(np.maximum(var, 1e-12))
    ybar = b[0] / n
    Sxx = G[1:, 1:] - n * np.outer(mu, mu)
    Sxy = b[1:] - n * mu * ybar
    Ss = Sxx / np.outer(sd, sd)
    ws = np.linalg.solve(Ss + a_gor * n * np.eye(len(mu)), Sxy / sd)
    w = ws / sd
    return np.concatenate([[ybar - mu @ w], w])


def sse(beta, G, b, yy):
    return yy - 2 * beta @ b + beta @ G @ beta


def a6_tahmin(t, kol):
    ist = t.istasyon_no.unique()
    Gs, bs, yys, Xs = {}, {}, {}, {}
    for s, g in t.groupby("istasyon_no"):
        X = np.column_stack([np.ones(len(g)), g[kol].to_numpy(float)])
        y = g.r.to_numpy(float)
        Gs[s], bs[s], yys[s], Xs[s] = X.T @ X, X.T @ y, y @ y, (g.index, X)
    Gt, bt, yyt = sum(Gs.values()), sum(bs.values()), sum(yys.values())
    tah = pd.Series(np.nan, index=t.index)
    secim = {}
    for i in ist:
        Go, bo = Gt - Gs[i], bt - bs[i]
        diger = [j for j in ist if j != i]
        cv = np.zeros(len(ALFA_GORELI))
        for j in diger:                                              # ic: diger istasyonlar sirayla disarida
            Gi, bi = Go - Gs[j], bo - bs[j]
            for ai, a in enumerate(ALFA_GORELI):
                cv[ai] += sse(ridge_sumdan(Gi, bi, None, a), Gs[j], bs[j], yys[j])
        a = ALFA_GORELI[np.argmin(cv)]
        beta = ridge_sumdan(Go, bo, None, a)
        idx, X = Xs[i]
        tah.loc[idx] = X @ beta
        secim[i] = a
    t = t.copy()
    t["A6"] = t.y_pred + tah
    t["alfa_A6"] = t.istasyon_no.map(secim)
    return t


def main():
    os.makedirs(OUT, exist_ok=True)
    df_m = pd.read_csv(E.SoilConfig().master_path, parse_dates=["tarih"])
    kol6 = [f"f{dz}_{L}" for dz in SIG for L in L6]
    parca = []
    for k in range(5):
        c = E.SoilConfig(); c.holdout = E.kat_istasyonlari(k)
        P = E.hazirla(c, df_m)
        T, dates, st_no, lat, lon, rak = E.veri(P)
        ti = np.nonzero(dates.year <= 2023)[0]
        if dates[ti].year.max() != 2023:
            raise RuntimeError("yil")
        tr = P["g_tr"].st_global
        r = E.IDWM.satirlar(T[ti], dates[ti], st_no, lat, lon, tr, [np.array([i]) for i in tr],
                            float(E.IDW_P[k]), E.IDW_K[k], rak).dropna(subset=["y_pred"])
        r["r"] = r.y_true - r.y_pred
        rs = {dz: r[r.derinlik == dz][["istasyon_no", "tarih", "r"]] for dz in SIG}
        for d in E.HEDEF:
            t = E.artik_tablosu(rs[20], r[r.derinlik == d].copy())
            t = E.loo_tahmin(t)
            t = a5_tahmin(a5_ozellik(rs[20], t, DZ[d]))
            t = a6_tahmin(a6_ozellik(rs, t), kol6)
            parca.append(t[["istasyon_no", "derinlik", "tarih", "y_true", "y_pred", "A1", "A2", "A3", "A5", "A6",
                            "L_A3", "gamma_A3", "kappa_A5", "gamma_A5", "alfa_A6"]].assign(kat=k))
        print(f"kat {k} tamam ({len(tr)} istasyon)", flush=True)
    t = pd.concat(parca, ignore_index=True)
    if t[["A1", "A2", "A3", "A5", "A6"]].isna().any().any():
        raise RuntimeError("tahminsiz satir")
    tab = E.rmse_tablo(t, ["A1", "A2", "A3", "A5", "A6"], "egitim istasyonlari 2015-2023 (istasyon birakmali)")
    print("\nRMSE:")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    tab.to_csv(os.path.join(OUT, "rmse.csv"), index=False)
    sec = t.groupby(["kat", "derinlik", "istasyon_no"])[["kappa_A5", "gamma_A5", "alfa_A6", "L_A3", "gamma_A3"]].first().reset_index()
    sec.to_csv(os.path.join(OUT, "secimler.csv"), index=False)
    print("\nSecilen kappa (A5), hedef x kat sayisi:")
    print(sec.groupby(["derinlik", "kappa_A5"]).size().unstack(fill_value=0).to_string())
    print("gamma_A5 ortalamasi:", sec.groupby("derinlik").gamma_A5.mean().round(3).to_dict())
    print("Secilen alfa (A6, goreli):")
    print(sec.groupby(["derinlik", "alfa_A6"]).size().unstack(fill_value=0).to_string())
    # hata ayrismasi: grup = (kat, istasyon, derinlik)
    ay = []
    for d, g in t.groupby("derinlik"):
        for kol in ("A1", "A2", "A3", "A5", "A6"):
            x = pd.DataFrame({"istasyon_no": g.kat * 10 ** 6 + g.istasyon_no, "derinlik": d, "tarih": g.tarih,
                              "y_true": g.y_true, "y_pred": g[kol]})
            ay.append(dict(derinlik=d, yontem=kol, **ayristir(x)))
    ay = pd.DataFrame(ay)
    print("\nHata ayrismasi (C^2; grup = kat x istasyon x derinlik):")
    print(ay[["derinlik", "yontem", "rmse", "mse", "sabit", "mevsimsel", "gunluk", "pay_sabit", "pay_mevsimsel", "pay_gunluk",
              "rms_sabit", "rms_mevsimsel", "rms_gunluk"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    ay.to_csv(os.path.join(OUT, "hata_ayristirmasi.csv"), index=False)


if __name__ == "__main__":
    main()
