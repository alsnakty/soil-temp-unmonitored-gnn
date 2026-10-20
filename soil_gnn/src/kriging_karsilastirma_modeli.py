import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

pd.set_option("compute.use_numexpr", False)

from ortak_veri import haversine_km
from toprak_ayarlar import DERINLIKLER, IDX_T, SoilConfig, ortam_bilgisi
from toprak_degerlendirme import metrics, station_metrics
from toprak_katlar import en_yakin_gorunur, kat_istasyonlari
from toprak_gnn_calistir import hazirla
from idw_karsilastirma_modeli import rakim_dogrusu, rmse

BURADA = os.path.dirname(os.path.abspath(__file__))
ANA_DIZIN = os.path.dirname(BURADA)
MODELLER = ("ustel", "kuresel", "gauss")
K_ADAY = (3, 4, 6, 8, 12, None)                   # IDW ile ayni adaylar
KUTU_KM = np.arange(0.0, 800.0 + 1e-9, 50.0)
NUGGET_MIN = 1e-4
PARCA_GUN = 512                                   # bellek icin gunler parca parca cozulur


def gamma(model, h, c0, c1, a):
    """Variogram; argumanlar yayinlanabilir diziler. gamma(0) = 0."""
    r = h / a
    if model == "ustel":
        f = 1.0 - np.exp(-3.0 * r)
    elif model == "kuresel":
        f = np.where(r < 1.0, 1.5 * r - 0.5 * r ** 3, 1.0)
    elif model == "gauss":
        f = 1.0 - np.exp(-3.0 * r ** 2)
    else:
        raise ValueError(model)
    return np.where(h > 0, c0 + c1 * f, 0.0)


def artik(V, z):
    """V: (gun, n) olcum, z: (n,) rakim km. Rakim dogrusundan artik + dogru katsayilari (<3 olcumde a = b = 0)."""
    a, b = rakim_dogrusu(V, z)
    yok = ~np.isfinite(b)
    a, b = np.where(yok, 0.0, a), np.where(yok, 0.0, b)
    return V - (a[:, None] + b[:, None] * z[None, :]), a, b


def deneysel(R, D):
    """R: (gun, n) (NaN = yok); D: (n, n) km. Kutu basina ortalama uzaklik, ortalama yari kare fark, cift-gun sayisi."""
    i, j = np.triu_indices(R.shape[1], 1)
    fark = R[:, i] - R[:, j]
    ok = np.isfinite(fark)
    top = np.where(ok, 0.5 * fark ** 2, 0.0).sum(0)
    n = ok.sum(0).astype(float)
    h = D[i, j]
    kutu = np.digitize(h, KUTU_KM) - 1
    sec = (kutu >= 0) & (kutu < len(KUTU_KM) - 1) & (n > 0)
    nk = len(KUTU_KM) - 1
    say = np.bincount(kutu[sec], n[sec], nk)
    gk = np.bincount(kutu[sec], top[sec], nk)
    hk = np.bincount(kutu[sec], h[sec] * n[sec], nk)
    dolu = say > 0
    return hk[dolu] / say[dolu], gk[dolu] / say[dolu], say[dolu]


def uydur(model, h, g, n):
    """Agirlikli en kucuk kareler (agirlik = cift-gun sayisi). Donus c0, c1, a (km)."""
    s = float(g.max())
    par, _ = curve_fit(lambda hh, c0, c1, a: gamma(model, hh, c0, c1, a), h, g, p0=(0.1 * s, 0.9 * s, 300.0),
                       sigma=1.0 / np.sqrt(n), bounds=([0.0, 0.0, 10.0], [np.inf, np.inf, 5000.0]), maxfev=20000)
    c0, c1, a = (float(v) for v in par)
    return max(c0, NUGGET_MIN * (c0 + c1)), c1, a


def standartlastir(R):
    """Her gun istasyonlar arasi sd'ye bol; en az 3 olcum ve sd > 0 olmayan gun NaN."""
    n = np.isfinite(R).sum(1)
    ort = np.where(n > 0, np.where(np.isfinite(R), R, 0.0).sum(1) / np.maximum(n, 1), np.nan)
    sd = np.sqrt(np.where(np.isfinite(R), (R - ort[:, None]) ** 2, 0.0).sum(1) / np.maximum(n, 1))
    gecerli = (n >= 3) & (sd > 0)
    return np.where(gecerli[:, None], R / np.where(gecerli, sd, 1.0)[:, None], np.nan)


def variogram_tablosu(Tobs, aylar, gorunur, D_all, rak):
    """Tobs: egitim yillari (gun, istasyon, derinlik). Donus: {(derinlik, model): (12, 3) [c0, c1, a]} ve uzun tablo."""
    Dvv = D_all[np.ix_(gorunur, gorunur)]
    vg, kayit = {}, []
    for d, dep in enumerate(DERINLIKLER):
        if dep == 0:
            continue
        V = Tobs[:, gorunur, d]
        R = standartlastir(V if rak is None else artik(V, rak[gorunur])[0])
        for model in MODELLER:
            vg[(dep, model)] = np.zeros((12, 3))
        for ay in range(1, 13):
            h, g, n = deneysel(R[aylar == ay], Dvv)
            for model in MODELLER:
                try:
                    c0, c1, a = uydur(model, h, g, n)
                except (RuntimeError, ValueError) as e:
                    raise RuntimeError(f"variogram uydurulamadi: derinlik {dep}, ay {ay}, {model}: {e}") from e
                vg[(dep, model)][ay - 1] = (c0, c1, a)
                kayit.append(dict(derinlik=dep, ay=ay, model=model, c0=c0, c1=c1, a_km=a,
                                  n_kutu=len(h), n_cift_gun=int(n.sum())))
    return vg, pd.DataFrame(kayit)


def kriging(A, Dtv, Dvv, model, c0, c1, a, k):
    sira = np.argsort(Dtv, kind="stable")
    V = A[:, sira]
    ok = np.isfinite(V)
    K = V.shape[1] if k is None else min(k, V.shape[1])
    secim = np.argsort(~ok, axis=1, kind="stable")[:, :K]          # olcumu olanlar once, uzaklik sirasi korunur
    gercek = np.take_along_axis(ok, secim, 1)
    nb = sira[secim]
    deger = np.where(gercek, np.take_along_axis(V, secim, 1), 0.0)
    out = np.full(A.shape[0], np.nan)
    ii = np.arange(K)
    for s in range(0, A.shape[0], PARCA_GUN):
        sl = slice(s, s + PARCA_GUN)
        g_, n_ = gercek[sl], nb[sl]
        p3 = [v[sl][:, None, None] for v in (c0, c1, a)]
        G = gamma(model, Dvv[n_[:, :, None], n_[:, None, :]], *p3)
        g0 = gamma(model, Dtv[n_], *[v[:, :, 0] for v in p3])
        pad = ~g_
        M = np.zeros((len(g_), K + 1, K + 1))
        M[:, :K, :K] = np.where(pad[:, :, None] | pad[:, None, :], 0.0, G)
        M[:, ii, ii] = np.where(pad, 1.0, 0.0)
        M[:, :K, K] = M[:, K, :K] = np.where(pad, 0.0, 1.0)
        b = np.zeros((len(g_), K + 1))
        b[:, :K] = np.where(pad, 0.0, g0)
        b[:, K] = 1.0
        hic = ~g_.any(1)
        M[hic, K, K], b[hic, K] = 1.0, 0.0                          # tekil olmasin; sonuc zaten NaN
        lam = np.linalg.solve(M, b[..., None])[..., 0][:, :K]
        out[sl] = np.where(hic, np.nan, (lam * deger[sl]).sum(1))
    return out


def satirlar(Tobs, dates, aylar, st_no, D_all, gorunur, hedef_gruplari, vg, model, k, rak=None):
    """Tobs: (gun, istasyon, derinlik) ham T. Her hedef grubu icin gorunur = gorunur - grup (IDW ile ayni)."""
    parca = []
    for grp in hedef_gruplari:
        vis = np.setdiff1d(gorunur, grp)
        if not (D_all[np.ix_(grp, vis)] > 0).all():
            raise RuntimeError("hedef ile gorunur istasyon arasi mesafe 0")
        Dvv = D_all[np.ix_(vis, vis)]
        for d, dep in enumerate(DERINLIKLER):
            if dep == 0:
                continue
            V = Tobs[:, vis, d]
            if rak is None:
                A = V
            else:
                A, a_, b_ = artik(V, rak[vis])
            c0, c1, a = (vg[(dep, model)][aylar - 1, j] for j in range(3))
            yhat = np.column_stack([kriging(A, D_all[g, vis], Dvv, model, c0, c1, a, k) for g in grp])
            if rak is not None:
                yhat = yhat + a_[:, None] + b_[:, None] * rak[grp][None, :]
            y = Tobs[:, grp, d]
            tt, jj = np.nonzero(np.isfinite(y))
            parca.append(pd.DataFrame({"istasyon_no": st_no[grp][jj], "derinlik": dep, "tarih": dates[tt],
                                       "y_true": y[tt, jj].astype(float), "y_pred": yhat[tt, jj]}))
    return pd.concat(parca, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="Kriging / regresyon krigingi karsilastirma modeli")
    ap.add_argument("--rakim", action="store_true", help="regresyon krigingi (rakim dogrusu + artiklarin krigingi)")
    ap.add_argument("--kat", type=int, help="uzaysal katli dogrulama: 0-4 (toprak_katlar.py)")
    ap.add_argument("--master", help="baska bir master_dataset; cikti ayri klasore (_sample) gider")
    arg = ap.parse_args()
    RAKIM = arg.rakim
    ek = ("" if arg.kat is None else f"_fold{arg.kat}") + ("_sample" if arg.master else "")
    CIKTI = os.path.join(ANA_DIZIN, "results", ("kriging_elevation" if RAKIM else "kriging") + ek)
    IDW_KLASOR = os.path.join(ANA_DIZIN, "results", ("idw_elevation" if RAKIM else "idw") + ek)
    cfg = SoilConfig()
    if arg.master:
        cfg.master_path = arg.master
    if arg.kat is not None:
        cfg.holdout = kat_istasyonlari(arg.kat)
    print("Veri:", cfg.master_path, "| cikti:", CIKTI, flush=True)
    t0 = time.time()
    P = hazirla(cfg)
    X, dates, stations, idx = P["X"], P["dates"], P["stations"], P["idx"]
    n_st = len(stations)
    Tobs = X[..., IDX_T].reshape(len(dates), n_st, len(DERINLIKLER)).astype(float)
    st_no = stations["istasyon_no"].to_numpy()
    lat, lon = stations["lat"].to_numpy(float), stations["lon"].to_numpy(float)
    D_all = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    aylar = dates.month.to_numpy()
    tr = P["g_tr"].st_global
    tr_gruplar = [tr[g] for g in P["groups"]]
    hold = P["g_all"].st_global[P["hold_local"]]
    rak = stations["rakim"].to_numpy(float) / 1000.0 if RAKIM else None
    if RAKIM and not np.isfinite(rak).all():
        raise RuntimeError(f"rakim eksik: {st_no[~np.isfinite(rak)].tolist()}")
    print("Yontem:", "regresyon krigingi (rakim)" if RAKIM else "siradan kriging", flush=True)
    os.makedirs(CIKTI, exist_ok=True)
    print("Ortam:", ortam_bilgisi(CIKTI, cfg.master_path), flush=True)

    ti_tr = idx["train"]
    vg, vg_tablo = variogram_tablosu(Tobs[ti_tr], aylar[ti_tr], tr, D_all, rak)
    vg_tablo.to_csv(os.path.join(CIKTI, "variogram.csv"), index=False)
    print(f"Variogram uyduruldu ({len(vg_tablo)} satir, {time.time() - t0:.0f} sn)", flush=True)

    vi = idx["val"]
    secim = []
    for model in MODELLER:
        for k in K_ADAY:
            df = satirlar(Tobs[vi], dates[vi], aylar[vi], st_no, D_all, tr, tr_gruplar, vg, model, k, rak)
            secim.append(dict(model=model, k="hepsi" if k is None else k, rmse_2023=rmse(df.dropna()),
                              tahminsiz=int(df.y_pred.isna().sum())))
    secim = pd.DataFrame(secim).sort_values("rmse_2023", kind="mergesort").reset_index(drop=True)
    secim.to_csv(os.path.join(CIKTI, "secim_2023.csv"), index=False)
    print("model, k secimi (2023, egitim istasyonlari):")
    print(secim.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    m_en = secim.model[0]
    k_en = None if secim.k[0] == "hepsi" else int(secim.k[0])
    print(f"\nSecilen: model={m_en}, k={'hepsi' if k_en is None else k_en}  ({time.time() - t0:.0f} sn)\n", flush=True)

    tum = np.arange(len(dates))
    bolumler = {"temporal": (idx["test"], tr_gruplar),
                "spatial_test": (idx["test"], [hold]),
                "spatial_full": (tum, [hold]),
                "val2023": (idx["val"], tr_gruplar),
                "spatial_val2023": (idx["val"], [hold])}
    en_yakin_gorunur(stations, st_no[hold], st_no[tr]).assign(kat=-1 if arg.kat is None else arg.kat).to_csv(
        os.path.join(CIKTI, "istasyon_mesafe.csv"), index=False)
    tum_m = []
    for ad, (ti, grp) in bolumler.items():
        df = satirlar(Tobs[ti], dates[ti], aylar[ti], st_no, D_all, tr, grp, vg, m_en, k_en, rak)
        bos = df.y_pred.isna()
        if bos.any():
            print(f"UYARI {ad}: {int(bos.sum())} satirda tahmin yok (gorunur olcum yok); puanlanmadi.")
        df = df[~bos].reset_index(drop=True)
        yol_idw = os.path.join(IDW_KLASOR, f"pred_{ad}_idw.csv.gz")
        if os.path.exists(yol_idw):
            g = pd.read_csv(yol_idw, parse_dates=["tarih"])
            m = df.merge(g, on=["istasyon_no", "derinlik", "tarih"], how="outer", suffixes=("", "_idw"), indicator=True)
            say = m["_merge"].value_counts()
            if say.get("left_only", 0) or say.get("right_only", 0):
                raise RuntimeError(f"{ad}: IDW ile satir kumeleri farkli: {say.to_dict()}")
            if not np.allclose(m.y_true, m.y_true_idw, atol=1e-4):
                raise RuntimeError(f"{ad}: IDW ile y_true farkli")
            denetim = "IDW ile satir kumesi ve y_true ayni"
        else:
            denetim = "IDW klasoru yok, satir denetimi yapilmadi"
        df.to_csv(os.path.join(CIKTI, f"pred_{ad}_kriging.csv.gz"), index=False)
        tum_m.append(metrics(df).assign(split=ad, model="kriging_rakim" if RAKIM else "kriging", vg_model=m_en, k=k_en))
        station_metrics(df).to_csv(os.path.join(CIKTI, f"station_{ad}_kriging.csv"), index=False)
        print(f"{ad}: {len(df)} satir yazildi; {denetim}", flush=True)
    pd.concat(tum_m).to_csv(os.path.join(CIKTI, "metrics.csv"), index=False)
    print(f"Bitti ({time.time() - t0:.0f} sn)")


if __name__ == "__main__":
    main()
