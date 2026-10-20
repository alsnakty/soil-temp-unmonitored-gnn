import argparse
import os

import numpy as np
import pandas as pd

pd.set_option("compute.use_numexpr", False)

from ortak_veri import haversine_km
from toprak_ayarlar import DERINLIKLER, IDX_T, SoilConfig, ortam_bilgisi
from toprak_degerlendirme import metrics, station_metrics
from toprak_katlar import en_yakin_gorunur, kat_istasyonlari
from toprak_gnn_calistir import hazirla

BURADA = os.path.dirname(os.path.abspath(__file__))
ANA_DIZIN = os.path.dirname(BURADA)
P_ADAY = (1.0, 2.0, 3.0)
K_ADAY = (3, 4, 6, 8, 12, None)


def idw(T, D, p, k):
    """T: (gun, n_gorunur) olcum (NaN = yok); D: (n_hedef, n_gorunur) km. Donus (gun, n_hedef)."""
    out = np.full((T.shape[0], D.shape[0]), np.nan)
    for i in range(D.shape[0]):
        sira = np.argsort(D[i])
        V = T[:, sira]
        ok = np.isfinite(V)
        if k is not None:
            ok &= np.cumsum(ok, axis=1) <= k
        w = np.where(ok, D[i, sira][None, :] ** -p, 0.0)
        pay = (w * np.where(ok, V, 0.0)).sum(1)
        payda = w.sum(1)
        out[:, i] = np.where(payda > 0, pay / np.where(payda > 0, payda, 1), np.nan)
    return out


def rakim_dogrusu(V, z):
    """V: (gun, n) olcum, z: (n,) rakim (km). Her gun icin T = a + b*z en kucuk kareler; <3 olcumde NaN."""
    ok = np.isfinite(V)
    n = ok.sum(1)
    zb = (ok * z).sum(1) / np.maximum(n, 1)
    yb = np.where(ok, V, 0.0).sum(1) / np.maximum(n, 1)
    dz = np.where(ok, z[None, :] - zb[:, None], 0.0)
    pay = (dz * np.where(ok, V - yb[:, None], 0.0)).sum(1)
    payda = (dz ** 2).sum(1)
    b = np.where((n >= 3) & (payda > 0), pay / np.where(payda > 0, payda, 1), np.nan)
    return yb - b * zb, b


def satirlar(Tobs, dates, st_no, lat, lon, gorunur, hedef_gruplari, p, k, rak=None, egimler=None):
    parca = []
    for grp in hedef_gruplari:
        vis = np.setdiff1d(gorunur, grp)
        D = haversine_km(lat[grp][:, None], lon[grp][:, None], lat[vis][None, :], lon[vis][None, :])
        if not (D > 0).all():
            raise RuntimeError("hedef ile gorunur istasyon arasi mesafe 0")
        for d, dep in enumerate(DERINLIKLER):
            if dep == 0:
                continue
            if rak is None:
                yhat = idw(Tobs[:, vis, d], D, p, k)
            else:
                a, b = rakim_dogrusu(Tobs[:, vis, d], rak[vis])
                if egimler is not None:                    # egim istatistigi geri donusten once (NaN = kurulamadi)
                    egimler.append(pd.DataFrame({"derinlik": dep, "egim": b}))
                yok = ~np.isfinite(b)                      # <3 olcum: a = b = 0 -> sonuc duz IDW ile ayni
                a, b = np.where(yok, 0.0, a), np.where(yok, 0.0, b)
                artik = Tobs[:, vis, d] - (a[:, None] + b[:, None] * rak[vis][None, :])
                yhat = idw(artik, D, p, k) + a[:, None] + b[:, None] * rak[grp][None, :]
            y = Tobs[:, grp, d]
            tt, jj = np.nonzero(np.isfinite(y))
            parca.append(pd.DataFrame({"istasyon_no": st_no[grp][jj], "derinlik": dep, "tarih": dates[tt],
                                       "y_true": y[tt, jj].astype(float), "y_pred": yhat[tt, jj]}))
    return pd.concat(parca, ignore_index=True)


def rmse(df):
    return float(np.sqrt(np.mean((df.y_pred.to_numpy(float) - df.y_true.to_numpy(float)) ** 2)))


def main():
    ap = argparse.ArgumentParser(description="IDW / rakim duzeltmeli IDW karsilastirma modeli")
    ap.add_argument("karsi", nargs="?", default="final_B_s0", help="results/ altinda GNN klasoru")
    ap.add_argument("--rakim", action="store_true", help="rakim duzeltmeli IDW")
    ap.add_argument("--master", help="baska bir master_dataset; cikti ayri klasore (_sample) gider")
    ap.add_argument("--karsilastirma-yok", action="store_true", help="GNN ile satir satir karsilastirmayi atla")
    ap.add_argument("--kat", type=int, help="uzaysal katli dogrulama: 0-4 (toprak_katlar.py)")
    arg = ap.parse_args()
    RAKIM, karsi = arg.rakim, arg.karsi
    CIKTI = os.path.join(ANA_DIZIN, "results", ("idw_elevation" if RAKIM else "idw")
                         + ("" if arg.kat is None else f"_fold{arg.kat}") + ("_sample" if arg.master else ""))
    cfg = SoilConfig()
    if arg.master:
        cfg.master_path = arg.master
    if arg.kat is not None:
        cfg.holdout = kat_istasyonlari(arg.kat)
    print("Veri:", cfg.master_path, "| cikti:", CIKTI, flush=True)
    P = hazirla(cfg)
    X, dates, stations, idx = P["X"], P["dates"], P["stations"], P["idx"]
    n_st = len(stations)
    Tobs = X[..., IDX_T].reshape(len(dates), n_st, len(DERINLIKLER)).astype(float)
    st_no = stations["istasyon_no"].to_numpy()
    lat, lon = stations["lat"].to_numpy(float), stations["lon"].to_numpy(float)
    tr = P["g_tr"].st_global                       # egitim istasyonlarinin global indisleri
    tr_gruplar = [tr[g] for g in P["groups"]]      # temporal gruplar (global indis)
    hold = P["g_all"].st_global[P["hold_local"]]
    rak = stations["rakim"].to_numpy(float) / 1000.0 if RAKIM else None
    if RAKIM and not np.isfinite(rak).all():
        raise RuntimeError(f"rakim eksik: {st_no[~np.isfinite(rak)].tolist()}")
    print("Yontem:", "rakim duzeltmeli IDW" if RAKIM else "IDW", flush=True)

    vi = idx["val"]
    secim = []
    for p in P_ADAY:
        for k in K_ADAY:
            df = satirlar(Tobs[vi], dates[vi], st_no, lat, lon, tr, tr_gruplar, p, k, rak)
            secim.append(dict(p=p, k="hepsi" if k is None else k, rmse_2023=rmse(df.dropna()),
                              tahminsiz=int(df.y_pred.isna().sum())))
    secim = pd.DataFrame(secim).sort_values("rmse_2023").reset_index(drop=True)
    os.makedirs(CIKTI, exist_ok=True)
    print("Ortam:", ortam_bilgisi(CIKTI, cfg.master_path), flush=True)
    secim.to_csv(os.path.join(CIKTI, "secim_2023.csv"), index=False)
    print("p, k secimi (2023, egitim istasyonlari):")
    print(secim.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    p_en = float(secim.p[0])
    k_en = None if secim.k[0] == "hepsi" else int(secim.k[0])
    print(f"\nSecilen: p={p_en}, k={'hepsi' if k_en is None else k_en}\n")

    tum = np.arange(len(dates))
    bolumler = {"temporal": (idx["test"], tr, tr_gruplar),
                "spatial_test": (idx["test"], tr, [hold]),
                "spatial_full": (tum, tr, [hold]),
                "val2023": (idx["val"], tr, tr_gruplar),
                "spatial_val2023": (idx["val"], tr, [hold])}
    en_yakin_gorunur(stations, st_no[hold], st_no[tr]).assign(kat=-1 if arg.kat is None else arg.kat).to_csv(
        os.path.join(CIKTI, "istasyon_mesafe.csv"), index=False)
    tum_m, ozet, egimler = [], [], []
    for ad, (ti, gor, grp) in bolumler.items():
        df = satirlar(Tobs[ti], dates[ti], st_no, lat, lon, gor, grp, p_en, k_en, rak,
                      egimler if (RAKIM and ad == "spatial_full") else None)
        yol_g = os.path.join(ANA_DIZIN, "results", karsi, f"pred_{ad}_gnn_s0.csv.gz")
        kars = not arg.karsilastirma_yok
        if kars and not os.path.exists(yol_g):
            if not ad.endswith("val2023"):
                raise FileNotFoundError(yol_g)
            print(f"{ad}: {karsi} klasorunde GNN tahmini yok; bu bolum karsilastirmasiz yazildi.")
            kars = False
        if not kars:
            m = df.rename(columns={"y_true": "y_true_idw", "y_pred": "y_pred_idw"})
        else:
            g = pd.read_csv(yol_g, parse_dates=["tarih"])
            m = df.merge(g, on=["istasyon_no", "derinlik", "tarih"], how="outer", suffixes=("_idw", "_gnn"),
                         indicator=True)
            yalniz = m["_merge"].value_counts()
            if yalniz.get("left_only", 0) or yalniz.get("right_only", 0):
                raise RuntimeError(f"{ad}: satir kumeleri farkli: {yalniz.to_dict()}")
            if not np.allclose(m.y_true_idw, m.y_true_gnn, atol=1e-4):
                raise RuntimeError(f"{ad}: y_true farkli")
        bos = m.y_pred_idw.isna()
        if bos.any():
            print(f"UYARI {ad}: {int(bos.sum())} satirda IDW tahmini yok (gorunur olcum yok); "
                  + ("bu satirlar puanlanmadi." if not kars else "iki model de bu satirlarsiz puanlandi."))
        m = m[~bos]
        idw_df = m[["istasyon_no", "derinlik", "tarih", "y_true_idw", "y_pred_idw"]].set_axis(
            ["istasyon_no", "derinlik", "tarih", "y_true", "y_pred"], axis=1)
        idw_df.to_csv(os.path.join(CIKTI, f"pred_{ad}_idw.csv.gz"), index=False)
        mi = metrics(idw_df).assign(split=ad, model="idw_rakim" if RAKIM else "idw", p=p_en, k=k_en)
        tum_m.append(mi)
        station_metrics(idw_df).to_csv(os.path.join(CIKTI, f"station_{ad}_idw.csv"), index=False)
        if not kars:
            print(f"{ad}: IDW RMSE {mi.iloc[0].rmse:.3f} (n={int(mi.iloc[0].n)})")
            continue
        gnn_df = m[["istasyon_no", "derinlik", "tarih", "y_true_gnn", "y_pred_gnn"]].set_axis(
            ["istasyon_no", "derinlik", "tarih", "y_true", "y_pred"], axis=1)
        mg = metrics(gnn_df)
        for (_, a), (_, b) in zip(mi.iterrows(), mg.iterrows()):
            ozet.append(dict(split=ad, derinlik=a.derinlik, n=a.n, rmse_idw=a.rmse, rmse_gnn=b.rmse,
                             fark=b.rmse - a.rmse, mbe_idw=a.mbe, mbe_gnn=b.mbe))
        if ad.startswith("spatial"):
            si, sg = station_metrics(idw_df), station_metrics(gnn_df)
            ist = si.merge(sg, on="istasyon_no", suffixes=("_idw", "_gnn"))
            ist["fark"] = ist.rmse_gnn - ist.rmse_idw
            print(f"{ad} istasyon RMSE:")
            print(ist[["istasyon_no", "rmse_idw", "mbe_idw", "rmse_gnn", "mbe_gnn", "fark"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
            print()
    pd.concat(tum_m).to_csv(os.path.join(CIKTI, "metrics.csv"), index=False)
    if egimler:
        e = pd.concat(egimler).dropna()
        print(f"Gunluk rakim egimi (spatial_full, gorunur {len(tr)} istasyon), C/km: derinlik -> medyan [%10, %90]")
        for dep, g in e.groupby("derinlik"):
            q = g.egim.quantile([0.1, 0.5, 0.9]).to_numpy()
            print(f"  {dep:>3} cm: {q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}]  (gun {len(g)})")
        print()
    if not ozet:
        return
    ozet = pd.DataFrame(ozet)
    ozet.to_csv(os.path.join(CIKTI, f"karsilastir_idw_{karsi}.csv"), index=False)
    print(f"IDW ile GNN ({karsi}) ayni satirlarda; fark = GNN - IDW (eksi = GNN iyi):")
    print(ozet.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
