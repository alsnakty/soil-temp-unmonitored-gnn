import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("compute.use_numexpr", False)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))
from toprak_ayarlar import SPATIAL_HOLDOUT

N_KAT = 5
ANAHTAR = ["istasyon_no", "derinlik", "tarih"]
OGRENEN = {"hibrit": (0, 1, 2), "gnn": (0, 1, 2), "mlp": (0, 1, 2), "ft_transformer": (0, 1, 2), "rf": (0, 1, 2),
           "rfsi": (0, 1, 2), "gnn_konumyok": (0,), "hibrit_konumyok": (0,), "ignnk": (0, 1, 2),
           "hibrit_ceza1": (0, 1, 2)}
EGITIMSIZ = {"idw_rakim": ("idw_elevation_fold{k}", "idw"), "idw": ("idw_fold{k}", "idw"),
             "kriging_rakim": ("kriging_elevation_fold{k}", "kriging")}      # (klasor, pred dosya eki)
BIRINCIL = ("hibrit", "idw_rakim")
KARARLI = (BIRINCIL, ("ignnk", "idw_rakim"), ("hibrit_ceza1", "idw_rakim"))
BIRINCIL_BOLUM = "spatial_test"
REFERANS = "idw_rakim"
OZET_MODELLER = ("hibrit", "gnn", "idw_rakim")
BOOT_N, BOOT_TOHUM = 10000, 2026
ESDEGER = 0.10
EK_CIFTLER = (("hibrit", "kriging_rakim"),             # referansi rakim-IDW olmayan ikincil karsilastirmalar
              ("hibrit_ceza1", "kriging_rakim"))
OMEGA, DZ, T0 = 2 * np.pi / 365.25, 95.0, pd.Timestamp("2015-01-01")   # 1/gun; 5-100 cm araligi; harmonik zaman sifiri
EN_AZ_GUN = 365                                         # harmonik uydurma icin istasyon-derinlik basina en az olcum
EN_AZ_AY_GUN = 15                                       # ve 12 takvim ayinin her birinde en az bu kadar olcum gunu
DERIN_ALT = 100                                         # gorunur difuzivite: 5 cm ile bu derinlik


def _oku(f):
    return pd.read_csv(f)[ANAHTAR + ["y_true", "y_pred"]].sort_values(ANAHTAR).reset_index(drop=True)


def yukle(kok, model, bolum):
    """{tohum: [kat 0..4 DataFrame]}; ogrenen ya da egitimsiz (IDW, kriging). Eksik dosya varsa None."""
    tum = {}
    if model in EGITIMSIZ:
        kl, ek = EGITIMSIZ[model]
        tohumlar, yol = (0,), lambda k, s: [os.path.join(kok, kl.format(k=k), f"pred_{bolum}_{ek}.csv.gz")]
    else:
        tohumlar = OGRENEN[model]
        yol = lambda k, s: glob.glob(os.path.join(kok, "folds", f"{model}_fold{k}_s{s}", f"pred_{bolum}_*.csv.gz"))
    for s in tohumlar:
        katlar = []
        for k in range(N_KAT):
            f = [x for x in yol(k, s) if os.path.exists(x)]
            if len(f) != 1:
                return None
            katlar.append(_oku(f[0]).assign(kat=k))
        tum[s] = katlar
    ilk = next(iter(tum.values()))
    for s, katlar in tum.items():
        for k in range(N_KAT):
            if not katlar[k][ANAHTAR + ["y_true"]].equals(ilk[k][ANAHTAR + ["y_true"]]):
                raise RuntimeError(f"{model}: tohum {s}, kat {k} satirlari farkli")
    return tum


def kat_butunlugu(katlar, model, bolum):
    """Her katta tekrarli satir yok; spatial bolumlerde her istasyon tek katta ve katlar arasi tekrarli satir yok."""
    for k, df in enumerate(katlar):
        if df.duplicated(ANAHTAR).any():
            raise RuntimeError(f"{model} kat {k}: tekrarli satir")
    if bolum.startswith("spatial"):
        hepsi = pd.concat(katlar, ignore_index=True)
        if hepsi.duplicated(ANAHTAR).any():
            raise RuntimeError(f"{model}: katlar arasi tekrarli satir")
        kat_sayisi = hepsi.groupby("istasyon_no").kat.nunique()
        if (kat_sayisi > 1).any():
            raise RuntimeError(f"{model}: birden fazla katta istasyon: {kat_sayisi[kat_sayisi > 1].index.tolist()}")


def istasyon_rmse(df):
    e = df.y_pred.to_numpy(float) - df.y_true.to_numpy(float)
    g = pd.DataFrame({"istasyon_no": df.istasyon_no.to_numpy(), "e2": e ** 2, "e": e})
    o = g.groupby("istasyon_no").agg(n=("e2", "size"), mse=("e2", "mean"), mbe=("e", "mean"))
    o["rmse"] = np.sqrt(o.mse)
    return o.drop(columns="mse")


def kat_ortalama_rmse(katlar):
    """Istasyon RMSE'si her katta ayri, istasyonun bulundugu katlar uzerinden ortalama."""
    return pd.concat([istasyon_rmse(df).rmse for df in katlar]).groupby(level=0).mean()


def model_rmse(tum):
    """(birincil: tohum basina istasyon RMSE'sinin tohum ortalamasi, topluluk istasyon RMSE'si, tohum sd)."""
    per = pd.concat([kat_ortalama_rmse(katlar).rename(s) for s, katlar in tum.items()], axis=1)
    toplu = []
    for k in range(N_KAT):
        df = tum[next(iter(tum))][k][ANAHTAR + ["y_true"]].copy()
        df["y_pred"] = np.mean([tum[s][k].y_pred.to_numpy(float) for s in tum], axis=0)
        toplu.append(df)
    return per.mean(1), kat_ortalama_rmse(toplu), (per.std(1) if per.shape[1] > 1 else None)


def havuz(tum, derinlik=None):
    """Havuzlanmis RMSE (tum katlarin satirlari), tohumlar uzerinden ortalama."""
    r = []
    for katlar in tum.values():
        df = pd.concat(katlar, ignore_index=True)
        if derinlik is not None:
            df = df[df.derinlik == derinlik]
        r.append(float(np.sqrt(np.mean((df.y_pred - df.y_true) ** 2))))
    return float(np.mean(r))


def karsilastir(ra, rb, istasyonlar, sifir_esik=0.0):
    d = (ra.loc[istasyonlar] - rb.loc[istasyonlar]).to_numpy(float)
    if sifir_esik > 0:
        d = np.where(np.abs(d) < sifir_esik, 0.0, d)
    rng = np.random.RandomState(BOOT_TOHUM)
    boot = d[rng.randint(0, len(d), size=(BOOT_N, len(d)))].mean(1)
    alt, ust = np.percentile(boot, [2.5, 97.5])
    alt90, ust90 = np.percentile(boot, [5, 95])
    w = stats.wilcoxon(d, alternative="two-sided")
    karar = "model iyi" if (d.mean() < 0 and ust < 0) else ("referans iyi" if (d.mean() > 0 and alt > 0) else "ayirt edilemedi")
    icinde = -ESDEGER < alt90 and ust90 < ESDEGER
    if karar == "ayirt edilemedi":
        etiket = "esdeger (+/-0,10 C)" if icinde else "belirsiz"
    else:
        etiket = "anlamli, +/-0,10 C icinde" if icinde else ""
    return dict(n_istasyon=len(d), ort_fark=d.mean(), medyan_fark=float(np.median(d)), ga_alt=alt, ga_ust=ust,
                ga90_alt=alt90, ga90_ust=ust90, wilcoxon_p=float(w.pvalue), model_iyi_istasyon=int((d < 0).sum()),
                etiket=etiket), karar


def gorulmus_gorulmemis(kok):
    satir = []
    for m in OZET_MODELLER:
        for bolum, ad in (("temporal", "gorulmus istasyon"), ("spatial_test", "gorulmemis istasyon")):
            t = yukle(kok, m, bolum)
            if t is None:
                continue
            r = model_rmse(t)[0]
            satir.append(dict(model=m, durum=ad, bolum=bolum, n_istasyon=len(r), istasyon_rmse_ort=r.mean(),
                              istasyon_rmse_medyan=r.median(), havuz_rmse=havuz(t)))
    return pd.DataFrame(satir)


def model_rmse_derinlik(tum, dz):
    """Tek derinlikte istasyon RMSE'si (katlar uzerinden ortalama), tohumlar uzerinden ortalama."""
    return pd.concat([kat_ortalama_rmse([df[df.derinlik == dz] for df in katlar]).rename(s)
                      for s, katlar in tum.items()], axis=1).mean(1)


def ayristir(katlar):
    """Istasyon-derinlik basina n, b (ortalama hata = sabit kayma), mse. Spatial bolumlerde her istasyon tek katta."""
    df = pd.concat(katlar, ignore_index=True)
    e = df.y_pred.to_numpy(float) - df.y_true.to_numpy(float)
    g = pd.DataFrame({"istasyon_no": df.istasyon_no.to_numpy(), "derinlik": df.derinlik.to_numpy(), "e": e, "e2": e ** 2})
    return g.groupby(["istasyon_no", "derinlik"]).agg(n=("e", "size"), b=("e", "mean"), mse=("e2", "mean")).reset_index()


def ayristirma_ozet(sd, istasyonlar):
    """MSE = sabit kayma (Sum n*b^2 / Sum n) + zamanla degisen kisim; havuz ve istasyon basina."""
    x = sd[sd.istasyon_no.isin(istasyonlar)]
    w = x.n.to_numpy(float)
    havuz = ((w * x.mse).sum() / w.sum(), (w * x.b ** 2).sum() / w.sum())
    g = x.assign(wm=w * x.mse, wb=w * x.b ** 2).groupby("istasyon_no")[["n", "wm", "wb"]].sum()
    return havuz, pd.DataFrame({"mse": g.wm / g.n, "sabit": g.wb / g.n})


def sar(x):
    """Aciyi (-pi, pi] araligina sarar."""
    return np.pi - np.mod(np.pi - np.asarray(x, float), 2 * np.pi)


def kapsam(tarih):
    """Mevsim kapsami: en az EN_AZ_GUN olcum gunu ve 12 takvim ayinin her birinde en az EN_AZ_AY_GUN gun."""
    ay = np.bincount(pd.DatetimeIndex(tarih).month, minlength=13)[1:]
    return bool(len(tarih) >= EN_AZ_GUN and ay.min() >= EN_AZ_AY_GUN)


def uydur_harmonik(tarih, y):
    """T = m + A cos(OMEGA t - phi) en kucuk kareler; donus (A, phi)."""
    t = (pd.DatetimeIndex(tarih) - T0).days.to_numpy(float)
    X = np.column_stack([np.ones_like(t), np.cos(OMEGA * t), np.sin(OMEGA * t)])
    c = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)[0]
    return float(np.hypot(c[1], c[2])), float(np.arctan2(c[2], c[1]))


def harmonik(df):
    tarih = pd.to_datetime(df.tarih).to_numpy()
    yt, yp = df.y_true.to_numpy(float), df.y_pred.to_numpy(float)
    gruplar = df.groupby(["istasyon_no", "derinlik"]).indices
    out = []
    for s in np.unique(df.istasyon_no):
        for d in np.unique(df.derinlik):
            ix = gruplar.get((s, d), np.array([], int))
            r = dict(istasyon_no=s, derinlik=d, n=len(ix), kapsam=kapsam(tarih[ix]))
            for ad, y in (("olcum", yt), ("tahmin", yp)):
                r[f"A_{ad}"], r[f"phi_{ad}"] = uydur_harmonik(tarih[ix], y[ix]) if r["kapsam"] else (np.nan, np.nan)
            out.append(r)
    return pd.DataFrame(out).set_index(["istasyon_no", "derinlik"])


def cift_harmonik(df, d):
    sut = ["istasyon_no", "tarih", "y_true", "y_pred"]
    o = df.loc[df.derinlik == 5, sut].merge(df.loc[df.derinlik == d, sut], on=["istasyon_no", "tarih"],
                                             suffixes=("_5", "_d"))
    o["tarih"] = pd.to_datetime(o.tarih)
    gruplar = o.groupby("istasyon_no").indices
    out = []
    for s in np.unique(df.istasyon_no):
        g = o.iloc[gruplar.get(s, np.array([], int))]
        r = dict(istasyon_no=s, derinlik=d, n_ortak=len(g), kapsam_ortak=kapsam(g.tarih))
        for ad, kol in (("olcum", "y_true"), ("tahmin", "y_pred")):
            for b in ("5", "d"):
                r[f"A{b}_{ad}"], r[f"phi{b}_{ad}"] = (uydur_harmonik(g.tarih, g[f"{kol}_{b}"]) if r["kapsam_ortak"]
                                                      else (np.nan, np.nan))
        out.append(r)
    return pd.DataFrame(out).set_index(["istasyon_no", "derinlik"])


def harmonik_turet(df):
    H = harmonik(df)
    C = pd.concat([cift_harmonik(df, d) for d in np.unique(df.derinlik)])
    R = pd.DataFrame({"n": H.n, "kapsam": H.kapsam.astype(float),
                      "genlik_orani": H.A_tahmin / H.A_olcum,
                      "faz_farki_gun": sar(H.phi_tahmin - H.phi_olcum) / OMEGA}, index=H.index)
    R = R.join(C[["n_ortak", "kapsam_ortak"]].astype(float))
    for ad in ("olcum", "tahmin"):
        R[f"sonum_{ad}"] = C[f"Ad_{ad}"] / C[f"A5_{ad}"]
        R[f"gecikme_{ad}_gun"] = pd.Series(sar(C[f"phid_{ad}"] - C[f"phi5_{ad}"]) / OMEGA, index=C.index)
    return R, C.xs(DERIN_ALT, level="derinlik")


def difuzivite(C):
    """Istasyon basina 5-100 cm gorunur difuzivite (cm2/gun), olcum ve tahmin; C = (5, 100) cifti, ortak gunler."""
    K = pd.DataFrame({"n_ortak": C.n_ortak, "kapsam_ortak": C.kapsam_ortak.astype(bool)}, index=C.index)
    for ad in ("olcum", "tahmin"):
        lnA = np.log(C[f"A5_{ad}"] / C[f"Ad_{ad}"]).to_numpy(float)
        dphi = sar(C[f"phid_{ad}"] - C[f"phi5_{ad}"])
        with np.errstate(invalid="ignore", divide="ignore"):
            K[f"kA_{ad}"] = np.where(lnA > 0, OMEGA * DZ ** 2 / (2 * lnA ** 2), np.nan)
            K[f"kphi_{ad}"] = np.where(dphi > 0, OMEGA * DZ ** 2 / (2 * dphi ** 2), np.nan)
        K[f"veri_{ad}"] = np.isfinite(lnA) & np.isfinite(dphi)
    return K


def tohum_birlestir_kappa(Ks):
    return pd.concat(Ks).groupby(level=0).agg(lambda v: v.all() if v.dtype == bool else v.median(skipna=False))


def _ozet(x, ad):
    x = pd.Series(np.asarray(x, float)).dropna()
    return {f"{ad}_medyan": x.median(), f"{ad}_q25": x.quantile(0.25), f"{ad}_q75": x.quantile(0.75)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bolum", default="spatial_test", choices=["spatial_test", "spatial_full", "temporal"])
    ap.add_argument("--kok", default=os.path.join(ANA_DIZIN, "results"))
    a = ap.parse_args()
    out = os.path.join(a.kok, "folds", f"analysis_{a.bolum}")
    os.makedirs(out, exist_ok=True)
    birincil_bolum = a.bolum == BIRINCIL_BOLUM

    tum = {}
    for m in list(OGRENEN) + list(EGITIMSIZ):
        t = yukle(a.kok, m, a.bolum)
        if t is None:
            print(f"(yok) {m}: bes katin / tohumlarin hepsi bulunamadi, atlandi")
            continue
        for katlar in t.values():
            kat_butunlugu(katlar, m, a.bolum)
        tum[m] = t
    if BIRINCIL[0] not in tum or BIRINCIL[1] not in tum:
        raise SystemExit(f"Birincil karsilastirma icin {BIRINCIL} eksik.")

    ref = tum[REFERANS][0]
    for m, t in tum.items():
        for katlar in t.values():
            for k in range(N_KAT):
                c = ANAHTAR + ["kat"]
                if not katlar[k][c].equals(ref[k][c]) or not np.allclose(katlar[k].y_true, ref[k].y_true, atol=1e-4):
                    raise RuntimeError(f"{m} kat {k}: satir kumesi {REFERANS} ile ayni degil")
    istasyonlar = np.sort(pd.concat(ref).istasyon_no.unique())
    gelistirme_haric = np.setdiff1d(istasyonlar, SPATIAL_HOLDOUT)

    rm, rm_top, oyn = {}, {}, []
    for m, t in tum.items():
        rm[m], top, sd = model_rmse(t)
        if m in OGRENEN and len(OGRENEN[m]) > 1:
            rm_top[m] = top
            oyn.append(pd.DataFrame({"model": m, "istasyon_no": sd.index, "tohum_sd": sd.to_numpy()}))

    mesafe = None
    if a.bolum.startswith("spatial"):
        ms = [os.path.join(a.kok, EGITIMSIZ[REFERANS][0].format(k=k), "istasyon_mesafe.csv") for k in range(N_KAT)]
        if all(os.path.exists(x) for x in ms):
            mesafe = pd.concat([pd.read_csv(x) for x in ms]).set_index("istasyon_no")

    tablo = pd.DataFrame({f"rmse_{m}": r for m, r in rm.items()})
    tablo = tablo.join(pd.DataFrame({f"rmse_topluluk_{m}": r for m, r in rm_top.items()}))
    if mesafe is not None:
        tablo = tablo.join(mesafe[["en_yakin_gorunur_km", "rakim_farki_m", "bolge", "kat"]])
    tablo["gelistirme_8"] = tablo.index.isin(SPATIAL_HOLDOUT)
    tablo.to_csv(os.path.join(out, "istasyon_tablosu.csv"))

    satir, kararlar, uyum = [], {}, {}
    kumeler = ((f"{len(istasyonlar)} istasyon", istasyonlar),
               (f"{len(gelistirme_haric)} istasyon (gelistirme 8 haric)", gelistirme_haric))
    ciftler = [(m, REFERANS) for m in rm if m != REFERANS] + [c for c in EK_CIFTLER if c[0] in rm and c[1] in rm]
    for kume_ad, kume in kumeler:
        for m, ref_m in ciftler:
            bir = birincil_bolum and (m, ref_m) == BIRINCIL
            kararli = birincil_bolum and (m, ref_m) in KARARLI
            r, karar = karsilastir(rm[m], rm[ref_m], kume)
            satir.append(dict(kume=kume_ad, model=m, referans=ref_m,
                              olcu="tohum basina RMSE ortalamasi" if m in OGRENEN else "tek",
                              birincil=bir, **r, karar=karar if kararli else ""))
            uyum.setdefault((m, ref_m), []).append((karar, r["etiket"]))
            if kararli:
                kararlar.setdefault((m, ref_m), {})[kume_ad] = (karar, r["etiket"])
        for m, rt in rm_top.items():
            r, _ = karsilastir(rt, rm[REFERANS], kume)
            satir.append(dict(kume=kume_ad, model=m, referans=REFERANS, olcu="topluluk (tahmin ortalamasi), ikincil",
                              birincil=False, **r, karar=""))
    for cift in KARARLI:
        if cift not in kararlar:
            continue
        (k82, e82), (k74, e74) = kararlar[cift][kumeler[0][0]], kararlar[cift][kumeler[1][0]]
        birlesik = f"{k82} (iki kumede ayni)" if k82 == k74 else f"duyarli: {kumeler[0][0]} '{k82}', {kumeler[1][0]} '{k74}'"
        etiket = (f"{e82} (iki kumede ayni)" if e82 else "") if e82 == e74 else f"duyarli: {kumeler[0][0]} '{e82}', {kumeler[1][0]} '{e74}'"
        satir.append(dict(kume="birlesik", model=cift[0], referans=cift[1], olcu="82/74 birlesik karar",
                          birincil=cift == BIRINCIL, karar=birlesik, etiket=etiket))
    for (m, ref_m), v in uyum.items():
        if len(v) != 2 or (birincil_bolum and (m, ref_m) in KARARLI):
            continue
        yaz = lambda kv: kv[0] + (f" / {kv[1]}" if kv[1] else "")
        satir.append(dict(kume="birlesik", model=m, referans=ref_m, olcu="82/74 uyumu (betimleyici)", birincil=False,
                          karar="", kume_uyumu=f"ayni: {yaz(v[0])}" if v[0] == v[1] else
                          f"duyarli: {kumeler[0][0]} '{yaz(v[0])}', {kumeler[1][0]} '{yaz(v[1])}'"))
    kars = pd.DataFrame(satir)
    for c in ("n_istasyon", "model_iyi_istasyon"):
        kars[c] = kars[c].astype("Int64")
    kars.to_csv(os.path.join(out, "karsilastirma.csv"), index=False)

    derinlikler = sorted(pd.concat(ref).derinlik.unique())
    derin = pd.DataFrame({m: [havuz(t)] + [havuz(t, dz) for dz in derinlikler] for m, t in tum.items()},
                         index=["all"] + list(derinlikler))
    derin.index.name = "derinlik"
    derin.to_csv(os.path.join(out, "derinlik.csv"))
    dfark = []
    for dz in derinlikler:
        rmd = {m: model_rmse_derinlik(t, dz) for m, t in tum.items()}
        for kume_ad, kume in kumeler:
            for m, ref_m in ciftler:
                ortak = np.array([s for s in kume if np.isfinite(rmd[m].get(s, np.nan)) and np.isfinite(rmd[ref_m].get(s, np.nan))])
                r, karar = karsilastir(rmd[m], rmd[ref_m], ortak)
                dfark.append(dict(derinlik=dz, kume=kume_ad, model=m, referans=ref_m, **r, karar_betimleyici=karar))
    pd.DataFrame(dfark).to_csv(os.path.join(out, "derinlik_fark.csv"), index=False)

    if mesafe is not None:
        sp = []
        for m in rm:
            r = stats.spearmanr(tablo["en_yakin_gorunur_km"], tablo[f"rmse_{m}"], nan_policy="omit")
            satir_sp = dict(model=m, rho_rmse_mesafe=float(r.statistic), p_rmse_mesafe=float(r.pvalue))
            if m != REFERANS:
                d = tablo[f"rmse_{m}"] - tablo[f"rmse_{REFERANS}"]
                r2 = stats.spearmanr(tablo["en_yakin_gorunur_km"], d, nan_policy="omit")
                satir_sp.update(rho_fark_mesafe=float(r2.statistic), p_fark_mesafe=float(r2.pvalue))
            sp.append(satir_sp)
        pd.DataFrame(sp).to_csv(os.path.join(out, "mesafe_spearman.csv"), index=False)
        ud = []
        for kume_ad, kume in kumeler:
            km = tablo.loc[kume, "en_yakin_gorunur_km"].dropna()
            dilim = pd.qcut(km, 3, labels=["yakin", "orta", "uzak"])
            for dl in ("yakin", "orta", "uzak"):
                ist = km.index[dilim == dl].to_numpy()
                for m, ref_m in ciftler:
                    r, karar = karsilastir(rm[m], rm[ref_m], ist)
                    ud.append(dict(kume=kume_ad, dilim=dl, km_min=km[ist].min(), km_max=km[ist].max(), model=m,
                                   referans=ref_m, **r, karar_betimleyici=karar))
        pd.DataFrame(ud).to_csv(os.path.join(out, "uzaklik_dilimi.csv"), index=False)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        renk = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
        fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor="#fcfcfb")
        for i, m in enumerate([REFERANS] + [x for x in rm if x != REFERANS]):
            ax.scatter(tablo["en_yakin_gorunur_km"], tablo[f"rmse_{m}"], s=22, color=renk[i % len(renk)],
                       edgecolor="#fcfcfb", linewidth=0.8, label=m)
        ax.set_xlabel("En yakin gorunur istasyona uzaklik (km)")
        ax.set_ylabel(f"Istasyon RMSE ({a.bolum}, C)")
        ax.legend(fontsize=8, frameon=False, ncol=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "mesafe_hata.png"), dpi=150, facecolor="#fcfcfb")
    if oyn:
        pd.concat(oyn).to_csv(os.path.join(out, "tohum_oynakligi.csv"), index=False)
    hao = None
    if a.bolum.startswith("spatial"):
        ha, hai = [], []
        for m, t in tum.items():
            sds = [ayristir(katlar) for katlar in t.values()]
            for kume_ad, kume in kumeler:
                oz = [ayristirma_ozet(sd, kume) for sd in sds]
                mse, sabit = np.mean([o[0][0] for o in oz]), np.mean([o[0][1] for o in oz])
                ist = pd.concat([o[1] for o in oz]).groupby(level=0).mean()
                pay = ist.sabit / ist.mse
                ha.append(dict(model=m, kume=kume_ad, havuz_mse=mse, havuz_sabit_kayma=sabit,
                               havuz_zamanla_degisen=mse - sabit, havuz_sabit_payi=sabit / mse,
                               **_ozet(pay, "istasyon_sabit_payi")))
                if kume is istasyonlar:
                    hai.append(ist.assign(model=m, zamanla_degisen=ist.mse - ist.sabit, sabit_payi=pay))
        hao = pd.DataFrame(ha)
        hao.to_csv(os.path.join(out, "hata_ayristirmasi.csv"), index=False)
        pd.concat(hai).rename_axis("istasyon_no").to_csv(os.path.join(out, "hata_ayristirmasi_istasyon.csv"))
    kap = None
    if a.bolum == "spatial_full":
        gf, kp, hi, ki = [], [], [], []
        for m, t in tum.items():
            Rs, Ks = [], []
            for katlar in t.values():
                Rt, C100 = harmonik_turet(pd.concat(katlar, ignore_index=True))
                Rs.append(Rt)
                Ks.append(difuzivite(C100))
            R = pd.concat(Rs).groupby(level=[0, 1]).mean()
            K = tohum_birlestir_kappa(Ks)
            hi.append(R.assign(model=m))
            ki.append(K.assign(model=m))
            for kume_ad, kume in kumeler:
                Rk = R[R.index.get_level_values(0).isin(kume)]
                for dz, g in Rk.groupby(level=1):
                    sat = dict(model=m, kume=kume_ad, derinlik=dz, n_istasyon=int(g.genlik_orani.notna().sum()),
                               n_istasyon_derinlik=len(g), n_olcum_yok=int((g.n == 0).sum()),
                               n_kapsam_disi=int(((g.kapsam < 1) & (g.n > 0)).sum()),
                               n_cift_olcum_yok=int((g.n_ortak == 0).sum()),
                               n_cift_kapsam_disi=int(((g.kapsam_ortak < 1) & (g.n_ortak > 0)).sum()))
                    for c in ("genlik_orani", "faz_farki_gun", "sonum_olcum", "sonum_tahmin", "gecikme_olcum_gun",
                              "gecikme_tahmin_gun"):
                        sat.update(_ozet(g[c], c))
                    gf.append(sat)
                Kk = K[K.index.isin(kume)]
                sat = dict(model=m, kume=kume_ad, n_istasyon=len(Kk), n_olcum_yok_ortak=int((Kk.n_ortak == 0).sum()),
                           n_kapsam_disi_ortak=int((~Kk.kapsam_ortak.astype(bool) & (Kk.n_ortak > 0)).sum()))
                for kk in ("kA", "kphi"):
                    for ad in ("olcum", "tahmin"):
                        sat.update(_ozet(Kk[f"{kk}_{ad}"], f"{kk}_{ad}"))
                        sat[f"{kk}_{ad}_tanimli"] = int(np.isfinite(Kk[f"{kk}_{ad}"]).sum())
                        sat[f"{kk}_{ad}_tanimsiz"] = int((Kk[f"veri_{ad}"].astype(bool) & ~np.isfinite(Kk[f"{kk}_{ad}"])).sum())
                    sat.update(_ozet(Kk[f"{kk}_tahmin"] / Kk[f"{kk}_olcum"], f"{kk}_oran"))
                kp.append(sat)
        pd.DataFrame(gf).to_csv(os.path.join(out, "genlik_faz.csv"), index=False)
        kap = pd.DataFrame(kp)
        kap.to_csv(os.path.join(out, "gorunur_difuzivite.csv"), index=False)
        pd.concat(hi).to_csv(os.path.join(out, "harmonik_istasyon.csv"))
        pd.concat(ki).rename_axis("istasyon_no").to_csv(os.path.join(out, "gorunur_difuzivite_istasyon.csv"))
    gg = gorulmus_gorulmemis(a.kok)
    if len(gg):
        gg.to_csv(os.path.join(a.kok, "folds", "gorulmus_gorulmemis.csv"), index=False)

    print(f"Bolum: {a.bolum} | istasyon {len(istasyonlar)} | modeller: {', '.join(rm)}"
          + ("" if birincil_bolum else "  (birincil bolum degil: karar sutunu bos)"))
    print("\nHavuzlanmis RMSE (C; ogrenenlerde tohumlarin ortalamasi):")
    print(derin.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nIstasyon RMSE farki = model - {REFERANS} (eksi = model iyi):")
    print(kars.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if oyn:
        print("\nTohum oynakligi (istasyon RMSE'sinin tohumlar arasi sd), medyan:")
        print(pd.concat(oyn).groupby("model").tohum_sd.median().to_string(float_format=lambda v: f"{v:.3f}"))
    if len(gg):
        print("\nGorulmus / gorulmemis istasyon (2024-25):")
        print(gg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if hao is not None:
        print("\nHata ayristirmasi (havuz, C^2; sabit kayma + zamanla degisen kisim):")
        print(hao[["model", "kume", "havuz_mse", "havuz_sabit_kayma", "havuz_zamanla_degisen", "havuz_sabit_payi",
                   "istasyon_sabit_payi_medyan"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if kap is not None:
        gfr = pd.DataFrame(gf)
        gfr = gfr[gfr.model == REFERANS]
        print(f"\nHarmonik kapsam (en az {EN_AZ_GUN} gun ve her ayda en az {EN_AZ_AY_GUN} gun; {REFERANS} satirlari): "
              "derinligin kendi gunleri ve (5 cm, d) ortak gunleri; 'olcum yok' kapsam disindan ayri:")
        print(gfr[["kume", "derinlik", "n_istasyon_derinlik", "n_olcum_yok", "n_kapsam_disi", "n_cift_olcum_yok",
                   "n_cift_kapsam_disi"]].to_string(index=False))
        print("\nGorunur difuzivite, 5-100 cm (cm2/gun; ortak gunler; tohumlar uzerinden medyan, bir tohumda tanimsizsa "
              "tanimsiz; istasyonlar uzerinden medyan):")
        print(kap[["model", "kume", "kA_olcum_medyan", "kA_tahmin_medyan", "kA_oran_medyan", "kphi_olcum_medyan",
                   "kphi_tahmin_medyan", "kphi_oran_medyan", "kA_tahmin_tanimsiz", "kphi_tahmin_tanimsiz",
                   "n_olcum_yok_ortak", "n_kapsam_disi_ortak"]]
              .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nCikti: {out}")


if __name__ == "__main__":
    main()
