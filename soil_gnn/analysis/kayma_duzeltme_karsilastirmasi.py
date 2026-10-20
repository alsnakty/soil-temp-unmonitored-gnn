import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 230)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))
sys.path.insert(0, os.path.join(ANA_DIZIN, "analysis"))

from toprak_ayarlar import DERINLIKLER, IDX_T, OZNITELIK, SPATIAL_HOLDOUT, SoilConfig
from toprak_gnn_calistir import hazirla
from toprak_katlar import kat_istasyonlari
from ortak_veri import haversine_km
import idw_karsilastirma_modeli as IDWM
import kriging_karsilastirma_modeli as KRM
import katli_birincil_karsilastirma as KB

SON = os.path.join(ANA_DIZIN, "results")
KATLI = os.path.join(SON, "folds")
OUT = os.path.join(KATLI, "analysis_offset_correction")
IDW_P = (1, 1, 2, 2, 1)
IDW_K = (6, 6, 8, 8, 8)
DER = (5, 10, 20, 50, 100)
ANAHTAR = ["istasyon_no", "derinlik", "tarih"]
TEST0 = pd.Timestamp("2024-01-01")
OMEGA, T0 = 2 * np.pi / 365.25, pd.Timestamp("2015-01-01")


def oku(klasor, dosya, k):
    return pd.read_csv(os.path.join(SON, klasor.format(k=k), dosya))


def vg_sozluk(yol):
    v = pd.read_csv(yol)
    return {(int(dep), m): g.sort_values("ay")[["c0", "c1", "a_km"]].to_numpy(float) for (dep, m), g in v.groupby(["derinlik", "model"])}


def dlst_kat(k, df_m, sinama):
    """Bir kat: dLST'li rakim-IDW ve kriging test tahminleri + fit tablosu."""
    c = SoilConfig()
    c.uydu_derinlik = "tum"
    c.holdout = kat_istasyonlari(k)
    P = hazirla(c, df_m)
    X, dates, stations, ti = P["X"], P["dates"], P["stations"], P["idx"]["train"]
    if dates[ti].year.min() != 2015 or dates[ti].year.max() != 2022:
        raise RuntimeError("egitim yillari 2015-2022 degil")
    n_st = len(stations)
    X4 = X.reshape(len(dates), n_st, len(DERINLIKLER), -1)
    Tobs = X4[..., IDX_T].astype(float)
    st_no = stations["istasyon_no"].to_numpy()
    lat, lon = stations["lat"].to_numpy(float), stations["lon"].to_numpy(float)
    rak = stations["rakim"].to_numpy(float) / 1000.0
    tr = P["g_tr"].st_global
    hold = P["g_all"].st_global[P["hold_local"]]
    if set(st_no[tr]) & set(c.holdout) or set(st_no[hold]) != set(c.holdout):
        raise RuntimeError("kat bolmesi hatali")
    p, kk = float(IDW_P[k]), IDW_K[k]
    # gece LST ortalamasi 2015-2022 (yuzey dugumu; yalniz uydu)
    L = np.nanmean(X4[ti][:, :, 0, OZNITELIK.index("LST_Night_C")].astype(float), axis=0)
    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    dl_tr = np.array([L[i] - IDWM.idw(L[np.setdiff1d(tr, [i])][None, :], D[i:i + 1, np.setdiff1d(tr, [i])], p, kk)[0, 0]
                      for i in tr])
    dl_h = np.array([L[h] - IDWM.idw(L[tr][None, :], D[h:h + 1, tr], p, kk)[0, 0] for h in hold])
    # egitim kaymalari: rakim-IDW ve kriging, istasyon gizli, katin diger egitim istasyonlari gorunur, 2015-2022
    gr = [np.array([i]) for i in tr]
    r_idw = IDWM.satirlar(Tobs[ti], dates[ti], st_no, lat, lon, tr, gr, p, kk, rak)
    kl = os.path.join(SON, f"kriging_elevation_fold{k}")
    s = pd.read_csv(os.path.join(kl, "secim_2023.csv"))
    km = None if s.k[0] == "hepsi" else int(s.k[0])
    r_kr = KRM.satirlar(Tobs[ti], dates[ti], dates[ti].month.to_numpy(), st_no, D, tr, gr,
                        vg_sozluk(os.path.join(kl, "variogram.csv")), s.model[0], km, rak)
    if sinama is not None:
        s4 = sinama[sinama.kat == k].set_index("istasyon_no")
        ky = r_idw.dropna(subset=["y_pred"]).assign(e=lambda x: x.y_pred - x.y_true).groupby(["istasyon_no", "derinlik"]).e.mean().unstack()
        A_, B_ = ky.loc[s4.index, list(DER)].to_numpy(float), s4[[str(d) for d in DER]].to_numpy(float)
        if not np.array_equal(np.isnan(A_), np.isnan(B_)):
            raise RuntimeError("8.4 ile kayma NaN deseni farkli")
        f1 = float(np.nanmax(np.abs(A_ - B_)))
        f2 = float(np.abs(pd.Series(dl_tr, index=st_no[tr]).loc[s4.index].to_numpy() - s4.dLST_gece.to_numpy()).max())
        print(f"  kat {k} sinama: 8.4 ile kayma farki {f1:.2e}, dLST farki {f2:.2e}", flush=True)
    fit, tah = [], {}
    for ad, r, taban_kl, ek in (("idw_rakim", r_idw, "idw_elevation_fold{k}", "idw"),
                                ("kriging_rakim", r_kr, "kriging_elevation_fold{k}", "kriging")):
        kay = r.dropna(subset=["y_pred"]).assign(e=lambda x: x.y_pred - x.y_true).groupby(["istasyon_no", "derinlik"]).e.mean().unstack()
        dls = pd.Series(dl_tr, index=st_no[tr])
        duz = {}
        for d in DER:
            x = pd.concat([kay[d].rename("y"), dls.rename("x")], axis=1).dropna()
            A = np.column_stack([np.ones(len(x)), x.x.to_numpy()])
            (a_, b_), *_ = np.linalg.lstsq(A, x.y.to_numpy(), rcond=None)
            duz[d] = (a_, b_)
            fit.append(dict(kat=k, yontem=ad, derinlik=d, n=len(x), alfa=a_, beta=b_))
        corr = pd.DataFrame([(st_no[h], d, duz[d][0] + duz[d][1] * dl_h[j]) for j, h in enumerate(hold) for d in DER],
                            columns=["istasyon_no", "derinlik", "duz"])
        for bolum in ("spatial_test", "spatial_full"):
            t = oku(taban_kl, f"pred_{bolum}_{ek}.csv.gz", k).merge(corr, on=["istasyon_no", "derinlik"], how="left")
            if t.duz.isna().any():
                raise RuntimeError(f"{ad} kat {k}: duzeltmesi olmayan satir {int(t.duz.isna().sum())}")
            tah[(ad, bolum)] = t.assign(y_pred=t.y_pred - t.duz).drop(columns="duz")
    dl_tab = pd.DataFrame({"istasyon_no": st_no[hold], "kat": k, "dLST_gece": dl_h})
    return tah, pd.DataFrame(fit), dl_tab


def rm(katlar_by_seed):
    """{tohum: [kat df]} -> tohum basina istasyon RMSE'sinin ortalamasi (birincil olcu, KB.model_rmse ile ayni)."""
    return KB.model_rmse(katlar_by_seed)[0]


def harmonik_duzeltme(df, N):
    df = df.reset_index(drop=True)
    df = df.assign(tarih=pd.to_datetime(df.tarih), e=df.y_pred - df.y_true)
    if df.index.has_duplicates or df.duplicated(ANAHTAR).any():
        raise RuntimeError("tekrarli satir/indis")
    bas = TEST0 - pd.DateOffset(months=N)
    w = df[(df.tarih >= bas) & (df.tarih < TEST0)]
    if w.tarih.max() >= TEST0:
        raise RuntimeError("kalibrasyonda 2024-25")
    test = df[df.tarih >= TEST0].copy()
    test["duz"] = 0.0
    tur = {"harmonik": 0, "sabit": 0, "yok": 0}
    gw = dict(tuple(w.groupby(["istasyon_no", "derinlik"])))
    for key, idx in test.groupby(["istasyon_no", "derinlik"]).groups.items():
        g = gw.get(key)
        if g is None or len(g) == 0:
            tur["yok"] += 1
            continue
        tt = ((g.tarih - T0).dt.days.to_numpy(float))
        if len(g) >= 90 and g.tarih.dt.month.nunique() >= 6:
            A = np.column_stack([np.ones(len(g)), np.sin(OMEGA * tt), np.cos(OMEGA * tt)])
            cf, *_ = np.linalg.lstsq(A, g.e.to_numpy(), rcond=None)
            t2 = ((test.loc[idx, "tarih"] - T0).dt.days.to_numpy(float))
            test.loc[idx, "duz"] = cf[0] + cf[1] * np.sin(OMEGA * t2) + cf[2] * np.cos(OMEGA * t2)
            tur["harmonik"] += 1
        else:
            test.loc[idx, "duz"] = g.e.mean()
            tur["sabit"] += 1
    e = test.e - test.duz
    ist = (e ** 2).groupby(test.istasyon_no).mean() ** 0.5
    return dict(havuz_rmse=float(np.sqrt((e ** 2).mean())), istasyon_rmse_ort=float(ist.mean()), **tur)


def main():
    os.makedirs(OUT, exist_ok=True)
    tum82 = sorted(set().union(*[set(kat_istasyonlari(k)) for k in range(5)]))
    sec = pd.read_csv(os.path.join(KATLI, "analysis_inner_validation", "hibrit_ic_dogrulama_icsecim.csv"))
    idw_kat = sec[(sec.model == "hibrit") & sec.egitimsiz_iyi].kat.astype(int).tolist()
    print("a) egitimsiz taban secilen katlar (rakim-IDW):", idw_kat, flush=True)
    st = {"idw_rakim": {-1: [oku("idw_elevation_fold{k}", "pred_spatial_test_idw.csv.gz", k) for k in range(5)]},
          "kriging_rakim": {-1: [oku("kriging_elevation_fold{k}", "pred_spatial_test_kriging.csv.gz", k) for k in range(5)]},
          "hibrit": {s: [oku(f"folds/hibrit_fold{{k}}_s{s}", f"pred_spatial_test_gnn_s{s}.csv.gz", k) for k in range(5)]
                     for s in range(3)}}
    st["hibrit_0epoch"] = {s: [st["idw_rakim"][-1][k] if k in idw_kat else st["hibrit"][s][k] for k in range(5)]
                           for s in range(3)}
    df_m = pd.read_csv(SoilConfig().master_path, parse_dates=["tarih"])
    sinama = pd.read_csv(os.path.join(KATLI, "analysis_dlst", "dlst_istasyon.csv"))
    fits, dls, tah = [], [], {}
    for k in range(5):
        t, f, d = dlst_kat(k, df_m, sinama)
        fits.append(f); dls.append(d)
        for key, v in t.items():
            tah.setdefault(key, []).append(v)
        print(f"  kat {k} dLST tamam", flush=True)
    fits = pd.concat(fits); dls = pd.concat(dls)
    fits.to_csv(os.path.join(OUT, "dlst_fit.csv"), index=False)
    dls.to_csv(os.path.join(OUT, "dlst_test_istasyon.csv"), index=False)
    st["idw_rakim_dlst"] = {-1: tah[("idw_rakim", "spatial_test")]}
    st["kriging_rakim_dlst"] = {-1: tah[("kriging_rakim", "spatial_test")]}
    for ad in ("idw_rakim_dlst", "kriging_rakim_dlst"):
        for k, x in enumerate(st[ad][-1]):
            pd.DataFrame(x).to_csv(os.path.join(OUT, f"pred_spatial_test_{ad}_kat{k}.csv.gz"), index=False)
    ref = pd.concat(st["idw_rakim"][-1])[ANAHTAR + ["y_true"]].sort_values(ANAHTAR).reset_index(drop=True)
    for ad, d in st.items():
        for s, kl in d.items():
            x = pd.concat(kl)[ANAHTAR + ["y_true"]].sort_values(ANAHTAR).reset_index(drop=True)
            if not (x[ANAHTAR].equals(ref[ANAHTAR]) and np.allclose(x.y_true, ref.y_true)):
                raise RuntimeError(f"{ad} s{s}: satir kumesi farkli")
    print(f"satir kumesi butun yontemlerde ayni ({len(ref)})", flush=True)
    r = {ad: rm(d) for ad, d in st.items()}
    kume = {"82 istasyon": tum82, "74 istasyon (gelistirme 8 haric)": [s for s in tum82 if s not in SPATIAL_HOLDOUT]}
    satir = []
    for (m, ref_m, rol) in (("idw_rakim_dlst", "idw_rakim", "ana"), ("kriging_rakim_dlst", "kriging_rakim", "ikincil"),
                            ("hibrit_0epoch", "idw_rakim", "ikincil"), ("hibrit", "idw_rakim", "bilgi")):
        for kad, kis in kume.items():
            ist = [s for s in kis if s in r[m].index and s in r[ref_m].index]
            o, karar = KB.karsilastir(r[m], r[ref_m], ist, sifir_esik=1e-12)
            satir.append(dict(model=m, referans=ref_m, rol=rol, kume=kad, **o, karar=karar if rol == "ana" else ""))
    kars = pd.DataFrame(satir)
    kars.to_csv(os.path.join(OUT, "karsilastirma.csv"), index=False)
    print("\nc) Karsilastirmalar (spatial_test, istasyon RMSE farki; eksi = model iyi):")
    print(kars.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    a = kars[kars.rol == "ana"]
    print("Birlesik karar (ana):", "ayni: " + a.karar.iloc[0] if a.karar.nunique() == 1 else "duyarli: " + " / ".join(a.karar))
    hv = []
    for ad, d in st.items():
        for dz in ["all"] + list(DER):
            v = []
            for kl in d.values():
                x = pd.concat(kl)
                if dz != "all":
                    x = x[x.derinlik == dz]
                v.append(np.sqrt(((x.y_pred - x.y_true) ** 2).mean()))
            hv.append(dict(model=ad, derinlik=dz, havuz_rmse=float(np.mean(v))))
    hv = pd.DataFrame(hv).pivot(index="derinlik", columns="model", values="havuz_rmse")
    print("\nHavuz RMSE (spatial_test):")
    print(hv.round(3).to_string())
    hv.to_csv(os.path.join(OUT, "havuz_rmse.csv"))
    print("\ndLST fit (kat ortalamasi alfa, beta):")
    print(fits.groupby(["yontem", "derinlik"])[["alfa", "beta"]].mean().round(3).to_string())
    full = {"idw_rakim": pd.concat([oku("idw_elevation_fold{k}", "pred_spatial_full_idw.csv.gz", k) for k in range(5)],
                                   ignore_index=True),
            "kriging_rakim": pd.concat([oku("kriging_elevation_fold{k}", "pred_spatial_full_kriging.csv.gz", k) for k in range(5)],
                                       ignore_index=True)}
    kal, sp = [], []
    for ad, df in full.items():
        for N in (12, 24):
            kal.append(dict(yontem=ad, N_ay=N, duzeltme="sabit+harmonik", **harmonik_duzeltme(df, N)))
        d2 = df.assign(tarih=pd.to_datetime(df.tarih), e=df.y_pred - df.y_true)
        b23 = d2[d2.tarih.dt.year == 2023].groupby(["istasyon_no", "derinlik"]).e.mean()
        b45 = d2[d2.tarih >= TEST0].groupby(["istasyon_no", "derinlik"]).e.mean()
        j = pd.concat([b23.rename("b23"), b45.rename("b2425")], axis=1).dropna()
        for dz in DER:
            x = j.xs(dz, level="derinlik")
            rr, pp = spearmanr(x.b23, x.b2425)
            sp.append(dict(yontem=ad, derinlik=dz, n=len(x), spearman=rr, p=pp))
    kal, sp = pd.DataFrame(kal), pd.DataFrame(sp)
    kal.to_csv(os.path.join(OUT, "kalibrasyon_harmonik.csv"), index=False)
    sp.to_csv(os.path.join(OUT, "kayma_2023_2425_spearman.csv"), index=False)
    print("\nd) Harmonik kalibrasyon (2024-25):")
    print(kal.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nd) 2023 ve 2024-25 kaymalari, Spearman:")
    print(sp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
