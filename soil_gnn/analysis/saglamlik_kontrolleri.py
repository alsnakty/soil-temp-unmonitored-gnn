import argparse
import glob
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 220)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOK = os.path.dirname(ANA_DIZIN)
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))
sys.path.insert(0, os.path.join(ANA_DIZIN, "analysis"))

import katli_birincil_karsilastirma as KB
import idw_karsilastirma_modeli as IW
import kayma_kalicilik_kalibrasyon as KK
import derin_a5_kalibrasyon as DA
from derin_uc_durum import harmonik_satir
from toprak_ayarlar import SPATIAL_HOLDOUT, SoilConfig, IDX_T, DERINLIKLER
from toprak_katlar import kat_istasyonlari
from toprak_gnn_calistir import hazirla
from toprak_veri_ve_graf import SoilWindows, norm_stats, normalize

SON = os.path.join(ANA_DIZIN, "results")
OUT = os.path.join(SON, "folds", "analysis_robustness")
AST = os.path.join(SON, "folds", "analysis_spatial_test")
MASTER = SoilConfig().master_path
GEC = os.path.join(ANA_DIZIN, "data_quality", "homojen_testsiz_duzeltilmeyen_gec_kirilmalar.csv")
QC_ISL = os.path.join(KOK, "DATASET", "hedef_qc_islemleri.csv")
BIRLESIK = os.path.join(KOK, "DATASET", "kirilma_istasyon_birlesik.csv")
LST_YP = os.path.join(KOK, "data_preparation", "modis_lst", "MODIS_LST_YP")
LST_KIYI5 = os.path.join(KOK, "MODIS_Turkiye", "MODIS_LST_KIYI5.csv")
ANAHTAR = KB.ANAHTAR
DER = (5, 10, 20, 50, 100)
IDW_P, IDW_K = (1.0, 1.0, 2.0, 2.0, 1.0), (6, 6, 8, 8, 8)
TEST0 = pd.Timestamp("2024-01-01")
B, TOHUM = KB.BOOT_N, KB.BOOT_TOHUM
TOL = 1e-9
SILME_ILGILI = {(17262, 5), (17310, 100), (17302, 100), (17302, 50)}
A2_KUME = {"a": (17240, 17262, 17190), "b": (17240, 17262, 17190, 17310, 17302, 17046)}
A9_KUME = {"a": (17320,), "b": (17320, 17287, 17034, 17270, 17046)}
KIYI5_SECILEN = (17026, 17040, 17320)
N_KUME = 16


def dur(kosul, mesaj):
    if not kosul:
        raise SystemExit(f"SAGLAMA BASARISIZ: {mesaj}")


def kumeler(ist):
    ist = np.sort(np.asarray(ist))
    return {"82": ist, "74": np.setdiff1d(ist, SPATIAL_HOLDOUT)}


def yukle_hepsi():
    tum = {}
    for m in list(KB.OGRENEN) + list(KB.EGITIMSIZ):
        t = KB.yukle(SON, m, "spatial_test")
        dur(t is not None, f"{m}: tahmin dosyalari eksik")
        for katlar in t.values():
            KB.kat_butunlugu(katlar, m, "spatial_test")
        tum[m] = t
    ref = tum[KB.REFERANS][0]
    for m, t in tum.items():
        for katlar in t.values():
            for k in range(KB.N_KAT):
                c = ANAHTAR + ["kat"]
                dur(katlar[k][c].equals(ref[k][c]) and np.allclose(katlar[k].y_true, ref[k].y_true, atol=1e-4),
                    f"{m} kat {k}: satir kumesi rakim-IDW ile ayni degil")
    return tum


def ciftler(rm):
    return [(m, KB.REFERANS) for m in rm if m != KB.REFERANS] + [c for c in KB.EK_CIFTLER if c[0] in rm and c[1] in rm]


def tablo2(rm, kume_d, cikar=()):
    satir = []
    for kad, kume in kume_d.items():
        ist = np.array([s for s in kume if s not in cikar and s in rm[KB.REFERANS].index])
        for m, r in ciftler(rm):
            o, karar = KB.karsilastir(rm[m], rm[r], ist)
            satir.append(dict(kume=kad, model=m, referans=r, **o, karar=karar))
    return pd.DataFrame(satir)


def rmse_hepsi(tum):
    return {m: KB.model_rmse(t)[0] for m, t in tum.items()}


def ayristirma(tum, kume_d):
    satir = []
    for m, t in tum.items():
        sds = [KB.ayristir(katlar) for katlar in t.values()]
        for kad, kume in kume_d.items():
            oz = [KB.ayristirma_ozet(sd, kume) for sd in sds]
            mse, sabit = np.mean([o[0][0] for o in oz]), np.mean([o[0][1] for o in oz])
            satir.append(dict(model=m, kume=kad, havuz_mse=mse, havuz_sabit_kayma=sabit, havuz_sabit_payi=sabit / mse))
    return pd.DataFrame(satir)


def saglama_tablo2(tum, rm, ist):
    kars = pd.read_csv(os.path.join(AST, "karsilastirma.csv"))
    kars = kars[kars.olcu.isin(["tohum basina RMSE ortalamasi", "tek"])]
    ad = {"82": f"{len(ist)} istasyon", "74": f"{len(kumeler(ist)['74'])} istasyon (gelistirme 8 haric)"}
    t2 = tablo2(rm, kumeler(ist))
    n = 0
    for _, r in t2.iterrows():
        e = kars[(kars.kume == ad[r.kume]) & (kars.model == r.model) & (kars.referans == r.referans)]
        dur(len(e) == 1, f"karsilastirma.csv satiri yok: {r.kume} {r.model} - {r.referans}")
        e = e.iloc[0]
        for c in ("ort_fark", "ga_alt", "ga_ust", "ga90_alt", "ga90_ust", "wilcoxon_p"):
            dur(abs(float(e[c]) - float(r[c])) <= TOL, f"{r.model} - {r.referans} {r.kume} {c}")
        dur(int(e.n_istasyon) == r.n_istasyon and int(e.model_iyi_istasyon) == r.model_iyi_istasyon,
            f"{r.model} - {r.referans} {r.kume} sayilar")
        n += 1
    ha = pd.read_csv(os.path.join(AST, "hata_ayristirmasi.csv"))
    ay = ayristirma(tum, kumeler(ist))
    for _, r in ay.iterrows():
        e = ha[(ha.model == r.model) & (ha.kume == ad[r.kume])]
        dur(len(e) == 1 and abs(float(e.havuz_mse.iloc[0]) - r.havuz_mse) <= TOL
            and abs(float(e.havuz_sabit_kayma.iloc[0]) - r.havuz_sabit_kayma) <= TOL, f"hata ayristirmasi {r.model} {r.kume}")
    print(f"S1 gecti: {n} cift ve {len(ay)} hata ayristirmasi satiri kayitli ciktilarla ayni", flush=True)
    return ay


def derinlik_listesi(x):
    return list(DER) if str(x) == "hepsi" else [int(v) for v in str(x).split(";")]


def a1_eslestir(df_m):
    kir = pd.read_csv(GEC)
    isl = pd.read_csv(QC_ISL, encoding="utf-8-sig")
    sil = isl[(isl.islem == "sil") & (isl.onay == "evet")]
    silme = [(int(r.istasyon_no), d, pd.Timestamp(r.baslangic), pd.Timestamp(r.bitis))
             for _, r in sil.iterrows() for d in derinlik_listesi(r.derinlik)]
    test = df_m[df_m.tarih >= TEST0]
    n_test = {(s, d): int(test.loc[test.istasyon_no == s, f"t{d}cmsicaklik"].notna().sum())
              for s, d in set(zip(kir.istasyon_no, kir.derinlik))}
    satir = []
    for _, r in kir.iterrows():
        s, d = int(r.istasyon_no), int(r.derinlik)
        ay = pd.Period(r.ay, "M")
        tam = n_test[(s, d)] == 0
        sinir = any(ss == s and dd == d and bas <= ay.end_time and ay.start_time <= bit + pd.Timedelta(days=92)
                    for ss, dd, bas, bit in silme)
        satir.append(dict(istasyon_no=s, istasyon_ad=r.istasyon_ad, derinlik=d, ay=r.ay, adim=r.adim_tarama,
                          test_olcum_gunu=n_test[(s, d)], tum_seri_silinmis=tam, silme_siniri=sinir,
                          silme_ile_ilgili=tam or sinir))
    e = pd.DataFrame(satir)
    ilgili = set(zip(e[e.silme_ile_ilgili].istasyon_no, e[e.silme_ile_ilgili].derinlik))
    kal = e[~e.silme_ile_ilgili]
    ciftler_ = set(zip(kal.istasyon_no, kal.derinlik))
    dur(len(e) == 53 and ilgili == SILME_ILGILI and len(kal) == 49 and kal.istasyon_no.nunique() == 14
        and len(ciftler_) == 49, f"A1 eslestirme: {len(e)}, {sorted(ilgili)}, {len(kal)}, {kal.istasyon_no.nunique()}")
    print("S2 gecti: silmeyle ilgili 4 kirilma; kalan 49 kirilma, 14 istasyon, 49 istasyon-derinlik cifti", flush=True)
    return e, ciftler_, sorted(kal.istasyon_no.unique())


def filtrele(tum, ciftler_):
    out = {}
    for m, t in tum.items():
        out[m] = {}
        for s, katlar in t.items():
            out[m][s] = []
            for df in katlar:
                anah = pd.MultiIndex.from_arrays([df.istasyon_no, df.derinlik])
                out[m][s].append(df[~anah.isin(list(ciftler_))].reset_index(drop=True))
    return out


def tohum_rmse(t):
    return pd.concat([KB.kat_ortalama_rmse(katlar).rename(s) for s, katlar in t.items()], axis=1)


def kume_etiketi(df_m):
    from sklearn.cluster import KMeans
    st = df_m.groupby("istasyon_no")[["lat", "lon"]].first().sort_index()
    dur(len(st) == 82 and (df_m.groupby("istasyon_no")[["lat", "lon"]].nunique() == 1).all().all(), "koordinat tekil degil")
    R = 6371.0
    phi, lam = np.radians(st.lat.to_numpy()), np.radians(st.lon.to_numpy())
    x = R * (lam - lam.mean()) * np.cos(phi.mean())
    y = R * (phi - phi.mean())
    km = KMeans(n_clusters=N_KUME, n_init=100, random_state=2026).fit(np.column_stack([x, y]))
    return pd.Series(km.labels_, index=st.index, name="kume")


def etiket(ort, alt, ust, alt90, ust90):
    karar = "model iyi" if (ort < 0 and ust < 0) else ("referans iyi" if (ort > 0 and alt > 0) else "ayirt edilemedi")
    icinde = -KB.ESDEGER < alt90 and ust90 < KB.ESDEGER
    if karar == "ayirt edilemedi":
        return karar, ("esdeger (+/-0,10 C)" if icinde else "belirsiz")
    return karar, ("anlamli, +/-0,10 C icinde" if icinde else "")


def a3_bootstrap(per_m, r_idw, lab, ist):
    """per_m: (istasyon x tohum) model RMSE; r_idw: istasyon IDW RMSE; lab: istasyon kumesi."""
    ist = np.asarray(ist)
    R_ = per_m.loc[ist].to_numpy(float)
    I_ = r_idw.loc[ist].to_numpy(float)
    D = R_ - I_[:, None]
    dbar = D.mean(1)
    L = lab.loc[ist].to_numpy()
    kids = np.arange(N_KUME)
    uyeler = [np.nonzero(L == c)[0] for c in kids]
    sonuc = {}
    rng = np.random.RandomState(TOHUM)
    n = len(ist)
    ii = rng.randint(0, n, size=(B, n))
    ss = rng.randint(0, D.shape[1], size=(B, n))
    sonuc["i tohumlu"] = D[ii, ss].mean(1)
    rng = np.random.RandomState(TOHUM)
    cc = rng.randint(0, N_KUME, size=(B, N_KUME))
    S = np.array([dbar[u].sum() for u in uyeler])
    N = np.array([len(u) for u in uyeler], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        sonuc["ii bloklu"] = S[cc].sum(1) / N[cc].sum(1)
    rng = np.random.RandomState(TOHUM)
    v = np.empty(B)
    for b in range(B):
        sec = np.concatenate([uyeler[c] for c in rng.randint(0, N_KUME, size=N_KUME)])
        v[b] = D[sec, rng.randint(0, D.shape[1], size=len(sec))].mean() if len(sec) else np.nan
    sonuc["iii bloklu+tohumlu"] = v
    satir = []
    for ad, boot in sonuc.items():
        alt, ust = np.nanpercentile(boot, [2.5, 97.5])
        alt90, ust90 = np.nanpercentile(boot, [5, 95])
        karar, et = etiket(dbar.mean(), alt, ust, alt90, ust90)
        satir.append(dict(yontem=ad, n_istasyon=n, ort_fark=dbar.mean(), ga_alt=alt, ga_ust=ust, ga90_alt=alt90,
                          ga90_ust=ust90, bos_ornek=int(np.isnan(boot).sum()), karar_betimleyici=karar, etiket=et))
    return satir


def cfg_kat(k):
    c = SoilConfig(holdout=kat_istasyonlari(k))
    c.__post_init__()
    c.idw_p, c.idw_k = IDW_P[k], IDW_K[k]
    return c


def a4_taban(df_m):
    """Hibritin test istasyonunda kullandigi IDW tabani, modelsiz (SoilWindows.idw_taban), C."""
    parca = []
    for k in range(KB.N_KAT):
        c = cfg_kat(k)
        P = hazirla(c, df_m)
        X, dates, idx, g_tr, g_all = P["X"], P["dates"], P["idx"], P["g_tr"], P["g_all"]
        sub = lambda ti, g: X[ti][:, g.slot_global]
        mu, sd = norm_stats(sub(idx["train"], g_tr), g_tr.slot_depth, "level")
        ti = idx["test"]
        Xn = normalize(sub(ti, g_all), g_all, mu, sd)
        gizli = np.isin(g_all.slot_station, P["hold_local"])
        Xn[:, gizli, IDX_T] = np.nan
        taban = np.full((len(ti), g_all.S), np.nan)
        for a in range(0, len(ti), 60):
            x = Xn[a:a + 60]
            ns = SimpleNamespace(g=g_all, W=len(x), cfg=c)
            taban[a:a + 60] = SoilWindows.idw_taban(ns, x)[..., 0]
        tabanC = taban * sd[g_all.slot_depth, IDX_T] + mu[g_all.slot_depth, IDX_T]
        sec = gizli & (np.asarray(DERINLIKLER)[g_all.slot_depth] > 0)
        tt, ss = np.nonzero(np.ones((len(ti), g_all.S), bool) & sec[None, :])
        parca.append(pd.DataFrame({"istasyon_no": g_all.stations["istasyon_no"].to_numpy()[g_all.slot_station[ss]],
                                   "derinlik": np.asarray(DERINLIKLER)[g_all.slot_depth[ss]],
                                   "tarih": dates[ti][tt].strftime("%Y-%m-%d"), "taban": tabanC[tt, ss], "kat": k}))
    return pd.concat(parca, ignore_index=True)


def saglama_a4(tum, taban):
    en = 0.0
    for k in range(KB.N_KAT):
        idw = tum[KB.REFERANS][0][k]
        j = idw.merge(taban[taban.kat == k], on=ANAHTAR, how="left")
        dur(j.taban.notna().all(), f"A4 kat {k}: IDW satirlarinda taban tanimsiz ({int(j.taban.isna().sum())})")
        f = float(np.abs(j.taban - j.y_pred).max())
        dur(f <= 1e-4, f"A4 kat {k}: taban ile kayitli IDW farki {f}")
        en = max(en, f)
    print(f"S4 gecti: hibrit IDW tabani kayitli rakim-IDW ile ayni (en buyuk fark {en:.1e} C)", flush=True)


def a4_olcu(c, o):
    r = stats.spearmanr(c, o)
    pr = np.corrcoef(c, o)[0, 1]
    r2oz = 1 - np.sum((o - c) ** 2) / np.sum((o - o.mean()) ** 2)
    egim = np.polyfit(o, c, 1)[0]
    return dict(n=len(c), spearman=float(r.statistic), p=float(r.pvalue), pearson_r=float(pr), pearson_r2=float(pr ** 2),
                r2_ozdes=float(r2oz), egim_c_o=float(egim))


def a4(tum):
    idw = pd.concat(tum[KB.REFERANS][0], ignore_index=True)
    ist_tab, ozet, ist_der = [], [], []
    for m in ("hibrit_ceza1", "hibrit"):
        t = tum[m]
        yh = np.mean([pd.concat(t[s], ignore_index=True).y_pred.to_numpy(float) for s in t], axis=0)
        df = idw[ANAHTAR + ["y_true"]].assign(c=yh - idw.y_pred.to_numpy(float), o=idw.y_true - idw.y_pred)
        st = df.groupby("istasyon_no")[["c", "o"]].mean()
        sd = df.groupby(["istasyon_no", "derinlik"])[["c", "o"]].mean().reset_index()
        ist_tab.append(st.assign(model=m).reset_index())
        for kad, kume in kumeler(st.index).items():
            x = st.loc[kume]
            ozet.append(dict(model=m, kume=kad, duzey="istasyon", **a4_olcu(x.c.to_numpy(), x.o.to_numpy())))
            y = sd[sd.istasyon_no.isin(kume)]
            ozet.append(dict(model=m, kume=kad, duzey="istasyon-derinlik", **a4_olcu(y.c.to_numpy(), y.o.to_numpy())))
        ist_der.append(sd.assign(model=m))
    return pd.DataFrame(ozet), pd.concat(ist_tab, ignore_index=True), pd.concat(ist_der, ignore_index=True)


def degismemis_istasyonlar(df_m):
    degisen = set()
    for d in DER:
        q, h = df_m[f"t{d}cmsicaklik"].to_numpy(float), df_m[f"t{d}cmsicaklik_ham"].to_numpy(float)
        b = np.isfinite(q) & np.isfinite(h) & (np.abs(h - q) > 1e-6)
        degisen |= set(df_m.istasyon_no.to_numpy()[b])
    tum = set(df_m.istasyon_no.unique())
    bir = set(pd.read_csv(BIRLESIK).istasyon_no)
    dur(len(tum - degisen) == 44 and degisen == bir, f"degismemis {len(tum - degisen)}, birlesik ile ayni {degisen == bir}")
    return sorted(tum - degisen)


def hedef_dizisi(df_m, dates, st_no, ham):
    T = np.full((len(dates), len(st_no), len(DERINLIKLER)), np.nan)
    for j, d in enumerate(DERINLIKLER):
        if d == 0:
            continue
        v = df_m[f"t{d}cmsicaklik"]
        if ham:
            v = df_m[f"t{d}cmsicaklik_ham"].where(df_m[f"t{d}cmsicaklik"].notna())
        p = pd.DataFrame({"tarih": df_m.tarih, "istasyon_no": df_m.istasyon_no, "v": v}) \
            .pivot(index="tarih", columns="istasyon_no", values="v").reindex(index=dates, columns=st_no)
        T[:, :, j] = p.to_numpy(float)
    return T


def a6_idw(df_m):
    qc, ham = [], []
    en = 0.0
    for k in range(KB.N_KAT):
        c = cfg_kat(k)
        P = hazirla(c, df_m)
        X, dates, stations = P["X"], P["dates"], P["stations"]
        st_no = stations["istasyon_no"].to_numpy()
        Tobs = X[..., IDX_T].reshape(len(dates), len(st_no), len(DERINLIKLER)).astype(float)
        Tq = hedef_dizisi(df_m, dates, st_no, False)
        dur(np.array_equal(np.isnan(Tq), np.isnan(Tobs)) and np.nanmax(np.abs(Tq - Tobs)) <= 1e-4,
            f"A6 kat {k}: master pivotu hazirla dizisiyle ayni degil")
        Th = hedef_dizisi(df_m, dates, st_no, True)
        lat, lon = stations["lat"].to_numpy(float), stations["lon"].to_numpy(float)
        rak = stations["rakim"].to_numpy(float) / 1000.0
        tr = P["g_tr"].st_global
        hold = P["g_all"].st_global[P["hold_local"]]
        tum_g = np.arange(len(dates))
        dq = IW.satirlar(Tobs[tum_g], dates[tum_g], st_no, lat, lon, tr, [hold], IDW_P[k], IDW_K[k], rak)
        dq = dq.dropna(subset=["y_pred"]).sort_values(ANAHTAR).reset_index(drop=True)
        kay = pd.read_csv(os.path.join(SON, f"idw_elevation_fold{k}", "pred_spatial_full_idw.csv.gz"), parse_dates=["tarih"])
        kay = kay.sort_values(ANAHTAR).reset_index(drop=True)
        dur(len(dq) == len(kay) and (dq[ANAHTAR].to_numpy() == kay[ANAHTAR].to_numpy()).all(),
            f"A6 kat {k}: satir kumesi kayitli spatial_full IDW ile ayni degil ({len(dq)} / {len(kay)})")
        f = float(np.abs(dq.y_pred - kay.y_pred).max())
        dur(f <= 1e-6 and np.allclose(dq.y_true, kay.y_true, atol=1e-4), f"A6 kat {k}: IDW farki {f}")
        en = max(en, f)
        dh = IW.satirlar(Th[tum_g], dates[tum_g], st_no, lat, lon, tr, [hold], IDW_P[k], IDW_K[k], rak)
        dh = dh.dropna(subset=["y_pred"])
        qc.append(dq.assign(e=dq.y_pred - dq.y_true))
        ham.append(dh.assign(e=dh.y_pred - dh.y_true))
    print(f"S5a gecti: QC hedefleriyle IDW kayitli spatial_full ile ayni (en buyuk fark {en:.1e} C)", flush=True)
    return pd.concat(qc, ignore_index=True), pd.concat(ham, ignore_index=True)


def saglama_a6(qc):
    k, _ = KK.kalicilik(qc)
    kay = pd.read_csv(os.path.join(SON, "folds", "analysis_offset_calibration", "a_kalicilik.csv"))
    dur(len(k) == len(kay) and np.allclose(k.spearman, kay.spearman, atol=1e-9) and (k.n_istasyon == kay.n_istasyon).all(),
        "A6 kalicilik a_kalicilik.csv ile ayni degil")
    print("S5b gecti: QC kaliciligi a_kalicilik.csv ile ayni", flush=True)


def a7_veri(df_m):
    t = pd.read_csv(os.path.join(SON, "folds", "analysis_deep_final_test", "tahminler.csv.gz"), parse_dates=["tarih"])
    parca = []
    for k in range(KB.N_KAT):
        full = pd.read_csv(os.path.join(SON, f"idw_elevation_fold{k}", "pred_spatial_full_idw.csv.gz"))
        full = full[full.derinlik.isin((50, 100))]
        h = harmonik_satir(full, 12)
        x = t[t.kat == k].merge(h[ANAHTAR + ["y_pred", "duz", "kal_turu"]], on=ANAHTAR, how="left")
        dur(x.y_pred.notna().all() and np.allclose(x.y_pred, x.A1, atol=1e-6), f"A7 kat {k}: A1 ayni degil")
        x = x.rename(columns={"duz": "duz_ii", "kal_turu": "kal_ii"}).drop(columns="y_pred")
        x["A1_kal12"] = x.A1 - x.duz_ii
        c = SoilConfig(holdout=kat_istasyonlari(k))
        c.__post_init__()
        a = DA.a5_kat(c, df_m, IDW_P[k], IDW_K[k])
        j = x[ANAHTAR + ["A5"]].merge(a[ANAHTAR + ["A5"]], on=ANAHTAR, how="left", suffixes=("", "_yeni"))
        dur(j.A5_yeni.notna().all() and float(np.abs(j.A5 - j.A5_yeni).max()) <= 1e-9, f"A7 kat {k}: A5 ayni degil")
        h4 = harmonik_satir(a.rename(columns={"A5": "y_pred"}), 12)
        x = x.merge(h4[ANAHTAR + ["duz", "kal_turu"]], on=ANAHTAR, how="left")
        dur(x.duz.notna().all(), f"A7 kat {k}: 4. durum kalibrasyon satiri eksik")
        x = x.rename(columns={"duz": "duz_iv", "kal_turu": "kal_iv"})
        x["A5_kal12"] = x.A5 - x.duz_iv
        parca.append(x)
    return pd.concat(parca, ignore_index=True)


A7_DURUM = (("1 olcum yok", "A1"), ("2 12 ay kampanya", "A1_kal12"), ("3 kalici sig sensor", "A5"),
            ("4 ikisi", "A5_kal12"))


def a7_tablo(t, ist_d):
    satir = []
    for ad, kol in A7_DURUM:
        kl = [t[t.kat == k][ANAHTAR + ["y_true"]].assign(y_pred=t[t.kat == k][kol].to_numpy()) for k in range(KB.N_KAT)]
        r = KB.kat_ortalama_rmse(kl)
        for kad, ist in ist_d.items():
            u = t[t.istasyon_no.isin(ist)]
            e = u[kol] - u.y_true
            satir.append(dict(durum=ad, kume=kad, n_istasyon=len(ist), istasyon_rmse_ort=float(r.loc[list(ist)].mean()),
                              havuz_rmse=float(np.sqrt((e ** 2).mean())),
                              **{f"havuz_{dz}": float(np.sqrt((e[u.derinlik == dz] ** 2).mean())) for dz in (50, 100)}))
    return pd.DataFrame(satir)


def saglama_a7(t):
    ist = sorted(t.istasyon_no.unique())
    tab = a7_tablo(t, {"82": ist, "74": [s for s in ist if s not in SPATIAL_HOLDOUT]})
    d = os.path.join(SON, "folds", "analysis_deep_final_test")
    u3 = pd.read_csv(os.path.join(d, "uc_durum.csv"))
    u4 = pd.read_csv(os.path.join(d, "uc_durum_4satir.csv"))
    kay = pd.concat([u3, u4[u4.durum.str.startswith("4")]], ignore_index=True)
    for i, (ad, _) in enumerate(A7_DURUM):
        for kad in ("82", "74"):
            a = tab[(tab.durum == ad) & (tab.kume == kad)].iloc[0]
            b = kay[kay.durum.str.startswith(str(i + 1)) & (kay.kume.astype(str) == kad)]
            dur(len(b) == 1, f"A7 kayitli satir yok: {ad} {kad}")
            for c in ("istasyon_rmse_ort", "havuz_rmse", "havuz_50", "havuz_100"):
                dur(abs(float(b[c].iloc[0]) - a[c]) <= TOL, f"A7 {ad} {kad} {c}")
    print("S6 gecti: dort durum kisitsiz kumede uc_durum.csv ve uc_durum_4satir.csv ile ayni", flush=True)


def a7_ortak(t, df_m):
    test = df_m[df_m.tarih >= TEST0]
    t20 = test.groupby("istasyon_no").t20cmsicaklik.apply(lambda v: int(v.notna().sum()))
    satir = []
    for s, g in t.groupby("istasyon_no"):
        der = set(g.derinlik)
        kal = g.groupby("derinlik").kal_ii.first()
        neden = []
        if der != {50, 100}:
            neden.append("50 ve 100 cm'nin ikisinde test olcumu yok")
        if (kal == "yok").any():
            neden.append("12 ay kampanyada duzeltme yok (" + ",".join(str(d) for d in kal[kal == "yok"].index) + " cm)")
        if t20.get(s, 0) == 0:
            neden.append("2024-2025'te 20 cm olcumu yok")
        satir.append(dict(istasyon_no=s, ortak=not neden, neden="; ".join(neden),
                          a5_esit_a1_payi=float((np.abs(g.A5 - g.A1) <= 1e-6).mean()),
                          gelistirme_8=s in SPATIAL_HOLDOUT))
    return pd.DataFrame(satir)


def a8_veri(df_m):
    yp = pd.concat([pd.read_csv(f, parse_dates=["tarih"]) for f in sorted(glob.glob(os.path.join(LST_YP, "MODIS_LST_YP_*.csv")))],
                   ignore_index=True)
    k5 = pd.read_csv(LST_KIYI5, parse_dates=["tarih"])
    kara = set(yp[yp.nokta == "kara_komsu"].istasyon_no)
    dur(len(kara) == 7 and not kara & set(KIYI5_SECILEN), f"kara_komsu istasyonlari {sorted(kara)}")
    dur({"QC_Day", "QC_Night"} <= set(k5.columns), "KIYI5 dosyasinda QC sutunu yok")
    a = yp[(yp.nokta == "kara_komsu") & yp.istasyon_no.isin(kara)]
    b = yp[(yp.nokta == "istasyon") & ~yp.istasyon_no.isin(kara | set(KIYI5_SECILEN))]
    c = k5[k5.istasyon_no.isin(KIYI5_SECILEN)]
    sec = pd.concat([a, b, c], ignore_index=True)[["istasyon_no", "tarih", "LST_Day_C", "LST_Night_C", "QC_Day", "QC_Night"]]
    dur(not sec.duplicated(["istasyon_no", "tarih"]).any() and sec.istasyon_no.nunique() == 82, "A8 piksel secimi")
    m = df_m[["istasyon_no", "tarih", "LST_Day_C", "LST_Night_C"]]
    j = m.merge(sec, on=["istasyon_no", "tarih"], how="left", suffixes=("", "_px"))
    for band in ("Day", "Night"):
        v, w = j[f"LST_{band}_C"].to_numpy(float), j[f"LST_{band}_C_px"].to_numpy(float)
        ayni_bos = np.array_equal(np.isnan(v), np.isnan(w))
        f = float(np.nanmax(np.abs(v - w))) if np.isfinite(v).any() else 0.0
        dur(ayni_bos and f <= 1e-6, f"A8 {band}: master ile piksel degeri farkli (bos deseni {ayni_bos}, fark {f})")
    print("S7 gecti: secilen piksel LST degerleri master_dataset_qc.csv ile ayni", flush=True)
    return j


def a8_say(j):
    toplam, yil, ist = [], [], []
    for band in ("Day", "Night"):
        x = j[j[f"LST_{band}_C"].notna()].copy()
        q = x[f"QC_{band}"].fillna(-1).astype(int).to_numpy()
        x["maskeli"] = q < 0
        q = np.where(q < 0, 0, q)
        x["zorunlu"] = q & 3
        x["lst_hata"] = (q >> 6) & 3
        x["yayinim"] = (q >> 4) & 3
        x["kotu"] = (x.lst_hata >= 2) & ~x.maskeli
        x["yil"] = x.tarih.dt.year
        ek = dict(band=band, n=len(x), kotu_payi=float(x.kotu.mean()), zorunlu_1_payi=float((x.zorunlu == 1).mean()),
                  **{f"lst_hata_{v}_payi": float((x.lst_hata == v).mean()) for v in range(4)},
                  yayinim_3_payi=float((x.yayinim == 3).mean()),
                  zorunlu_2_3_sayisi=int(x.zorunlu.isin([2, 3]).sum()), maskeli_sayisi=int(x.maskeli.sum()))
        toplam.append(ek)
        g = x.groupby("yil").agg(n=("kotu", "size"), kotu_payi=("kotu", "mean"))
        yil.append(g.reset_index().assign(band=band))
        g = x.groupby("istasyon_no").agg(n=("kotu", "size"), kotu_payi=("kotu", "mean"))
        ist.append(g.reset_index().assign(band=band))
    return pd.DataFrame(toplam), pd.concat(yil, ignore_index=True), pd.concat(ist, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sadece-saglama", action="store_true", help="yalniz saglamalar; sonuc yazilmaz")
    a = ap.parse_args()
    print("Veri:", MASTER, flush=True)
    df_m = pd.read_csv(MASTER, parse_dates=["tarih"])
    tum = yukle_hepsi()
    rm = rmse_hepsi(tum)
    ist = np.sort(pd.concat(tum[KB.REFERANS][0]).istasyon_no.unique())
    dur(len(ist) == 82, f"istasyon sayisi {len(ist)}")
    ay0 = saglama_tablo2(tum, rm, ist)
    esl, ciftler_, ist14 = a1_eslestir(df_m)
    itab = pd.read_csv(os.path.join(AST, "istasyon_tablosu.csv")).set_index("istasyon_no")
    per = {m: tohum_rmse(tum[m]) for m in ("hibrit", "hibrit_ceza1")}
    for m, p in per.items():
        dur(float(np.abs(p.mean(1) - itab[f"rmse_{m}"].reindex(p.index)).max()) <= TOL, f"A3 {m} tohum ortalamasi")
    print("S3 gecti: tohum basina istasyon RMSE'sinin ortalamasi istasyon_tablosu.csv ile ayni", flush=True)
    taban = a4_taban(df_m)
    saglama_a4(tum, taban)
    ist44 = degismemis_istasyonlar(df_m)
    print("S5c gecti: hic duzeltme gormeyen 44 istasyon; degisen 38 = kirilma_istasyon_birlesik.csv", flush=True)
    qc6, ham6 = a6_idw(df_m)
    saglama_a6(qc6)
    t7 = a7_veri(df_m)
    saglama_a7(t7)
    j8 = a8_veri(df_m)
    if a.sadece_saglama:
        print("Butun saglamalar gecti; --sadece-saglama: sonuc hesaplanmadi, dosya yazilmadi.")
        return

    os.makedirs(OUT, exist_ok=True)
    kd = kumeler(ist)
    esl.to_csv(os.path.join(OUT, "a1_eslestirme.csv"), index=False)
    tf = filtrele(tum, ciftler_)
    rmf = rmse_hepsi(tf)
    istf = np.sort(pd.concat(tf[KB.REFERANS][0]).istasyon_no.unique())
    a1 = pd.concat([tablo2(rm, kd).assign(durum="kisitsiz"),
                    tablo2(rmf, kumeler(istf)).assign(durum="a 49 cift cikarildi"),
                    tablo2(rm, kd, cikar=ist14).assign(durum="b 14 istasyon cikarildi")], ignore_index=True)
    a1.to_csv(os.path.join(OUT, "a1_tablo2.csv"), index=False)
    sk = pd.concat([ay0.assign(durum="kisitsiz"),
                    ayristirma(tf, kumeler(istf)).assign(durum="a 49 cift cikarildi"),
                    ayristirma(tum, {k: np.setdiff1d(v, ist14) for k, v in kd.items()}).assign(durum="b 14 istasyon cikarildi")],
                   ignore_index=True)
    sk.to_csv(os.path.join(OUT, "a1_sabit_kayma.csv"), index=False)
    a2 = pd.concat([tablo2(rm, kd, cikar=v).assign(durum=f"{k} cikarilan {len(v)}: {','.join(map(str, v))}")
                    for k, v in A2_KUME.items()], ignore_index=True)
    a2.to_csv(os.path.join(OUT, "a2_tablo2.csv"), index=False)
    lab = kume_etiketi(df_m)
    lab.to_frame().assign(boyut=lab.map(lab.value_counts())).to_csv(os.path.join(OUT, "a3_kumeler.csv"))
    a3 = []
    for m in ("hibrit", "hibrit_ceza1"):
        for kad, kume in kd.items():
            a3 += [dict(model=m, referans=KB.REFERANS, kume=kad, **r) for r in a3_bootstrap(per[m], rm[KB.REFERANS], lab, kume)]
    pd.DataFrame(a3).to_csv(os.path.join(OUT, "a3_bootstrap.csv"), index=False)
    oz4, st4, sd4 = a4(tum)
    oz4.to_csv(os.path.join(OUT, "a4_ozet.csv"), index=False)
    st4.to_csv(os.path.join(OUT, "a4_istasyon.csv"), index=False)
    sd4.to_csv(os.path.join(OUT, "a4_istasyon_derinlik.csv"), index=False)
    a5 = []
    for m, ref in (("gnn_konumyok", "gnn"), ("hibrit_konumyok", "hibrit")):
        r1, r0 = KB.kat_ortalama_rmse(tum[m][0]), KB.kat_ortalama_rmse(tum[ref][0])
        for kad, kume in kd.items():
            o, karar = KB.karsilastir(r1, r0, kume)
            a5.append(dict(model=m, referans=f"{ref} (tohum 0)", kume=kad, **o, karar_betimleyici=karar))
    pd.DataFrame(a5).to_csv(os.path.join(OUT, "a5_konumsuz_tohum0.csv"), index=False)
    a6 = []
    for hedef, df in (("QC", qc6), ("ham", ham6)):
        for kad, kume in (("82", ist), ("44 duzeltme gormeyen", ist44)):
            k6, _ = KK.kalicilik(df[df.istasyon_no.isin(kume)])
            a6.append(k6.assign(hedef=hedef, kume=kad))
    pd.concat(a6, ignore_index=True).to_csv(os.path.join(OUT, "a6_kalicilik.csv"), index=False)
    pd.Series(ist44, name="istasyon_no").to_csv(os.path.join(OUT, "a6_44_istasyon.csv"), index=False)
    oi = a7_ortak(t7, df_m)
    oi.to_csv(os.path.join(OUT, "a7_istasyonlar.csv"), index=False)
    ortak = oi[oi.ortak].istasyon_no.tolist()
    t7a = a7_tablo(t7, {"82": sorted(t7.istasyon_no.unique()),
                        "74": [s for s in sorted(t7.istasyon_no.unique()) if s not in SPATIAL_HOLDOUT]}).assign(kume_turu="kisitsiz")
    t7b = a7_tablo(t7, {"82 ortak": ortak, "74 ortak": [s for s in ortak if s not in SPATIAL_HOLDOUT]}).assign(kume_turu="ortak")
    pd.concat([t7a, t7b], ignore_index=True).to_csv(os.path.join(OUT, "a7_tablo3.csv"), index=False)
    t8, y8, i8 = a8_say(j8)
    t8.to_csv(os.path.join(OUT, "a8_toplam.csv"), index=False)
    y8.to_csv(os.path.join(OUT, "a8_yil.csv"), index=False)
    i8.to_csv(os.path.join(OUT, "a8_istasyon.csv"), index=False)
    a9 = []
    for k, v in A9_KUME.items():
        for kad, kume in kd.items():
            o, karar = KB.karsilastir(rm["hibrit_ceza1"], rm[KB.REFERANS], np.setdiff1d(kume, v))
            a9.append(dict(durum=f"{k} cikarilan {len(v)}: {','.join(map(str, v))}", kume=kad, **o, karar_betimleyici=karar))
    pd.DataFrame(a9).to_csv(os.path.join(OUT, "a9_cezali_hibrit.csv"), index=False)
    print("Ciktilar:", OUT)


if __name__ == "__main__":
    main()
