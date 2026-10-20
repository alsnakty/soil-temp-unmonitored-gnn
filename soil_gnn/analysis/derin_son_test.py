import argparse
import os
import sys

import numpy as np
import pandas as pd

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 220)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOK = os.path.dirname(ANA_DIZIN)
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))
sys.path.insert(0, os.path.join(ANA_DIZIN, "analysis"))

from toprak_ayarlar import SPATIAL_HOLDOUT, SoilConfig
from toprak_katlar import kat_istasyonlari
import eksik_derinlik as E
import eksik_derinlik_2 as E2
import katli_birincil_karsilastirma as KB

ANAHTAR = ["istasyon_no", "derinlik", "tarih"]
SIG = (5, 10, 20)
HEDEF = (50, 100)


def sapma(T, dates, st_no, lat, lon, rak, gorunur, hedefler, p, k, ti, derinlikler):
    """Her hedef istasyon tek basina gizli; kaynak = gorunur - hedef. r = olcum - rakim-IDW."""
    r = E.IDWM.satirlar(T[ti], dates[ti], st_no, lat, lon, gorunur, [np.array([i]) for i in hedefler], p, k, rak)
    r = r[r.derinlik.isin(derinlikler)].dropna(subset=["y_pred"]).copy()
    r["r"] = r.y_true - r.y_pred
    return r


def a6_matris(rs, rd):
    return E2.a6_ozellik({dz: rs[dz] for dz in SIG}, rd)


def ridge_sec(t, kol):
    """Istasyon birakmali ceza secimi (egitim istasyonlari), sonra butun istasyonlarla katsayi."""
    Gs, bs, yys = {}, {}, {}
    for s, g in t.groupby("istasyon_no"):
        X = np.column_stack([np.ones(len(g)), g[kol].to_numpy(float)])
        y = g.r.to_numpy(float)
        Gs[s], bs[s], yys[s] = X.T @ X, X.T @ y, y @ y
    Gt, bt = sum(Gs.values()), sum(bs.values())
    cv = np.zeros(len(E2.ALFA_GORELI))
    for j in Gs:
        for ai, a in enumerate(E2.ALFA_GORELI):
            cv[ai] += E2.sse(E2.ridge_sumdan(Gt - Gs[j], bt - bs[j], None, a), Gs[j], bs[j], yys[j])
    a = E2.ALFA_GORELI[int(np.argmin(cv))]
    return E2.ridge_sumdan(Gt, bt, None, a), a


def dur(kosul, mesaj):
    if not kosul:
        raise RuntimeError("DURDURUCU KONTROL: " + mesaj)


def kontrol_e():
    dur(tuple(E.LAR) == (3, 7, 15, 30, 60, 120), "e: A3 L izgarasi")
    dur(tuple(E2.KAPPA) == (150, 250, 350, 450, 600, 800, 1000), "e: A5 kappa izgarasi")
    dur(E2.DZ == {50: 30.0, 100: 80.0}, "e: A5 dz")
    dur(tuple(E2.L6) == (1, 3, 7, 15, 30, 60, 120) and tuple(E2.SIG) == SIG, "e: A6 oznitelikleri")
    dur(np.allclose(E2.ALFA_GORELI, np.logspace(-7, 0, 15)), "e: A6 ceza izgarasi")


def kontrol_f(a1, kat_no, p, k):
    yol = os.path.join(ANA_DIZIN, "results", f"idw_elevation_fold{kat_no}")
    sec = pd.read_csv(os.path.join(yol, "secim_2023.csv"))
    kayit = pd.read_csv(os.path.join(yol, "pred_spatial_test_idw.csv.gz"), usecols=ANAHTAR + ["y_pred"], parse_dates=["tarih"])
    kayit = kayit[kayit.derinlik.isin(HEDEF)]
    j = a1[ANAHTAR + ["y_pred"]].merge(kayit, on=ANAHTAR, how="outer", suffixes=("", "_kayit"), indicator=True)
    dur((j._merge == "both").all(), f"f: kat {kat_no} satir kumesi farkli ({(j._merge != 'both').sum()} satir)")
    fark = float(np.nanmax(np.abs(j.y_pred - j.y_pred_kayit)))
    dur(np.isfinite(j.y_pred).all() and np.isfinite(j.y_pred_kayit).all() and fark < 1e-6, f"f: kat {kat_no} A1 farki {fark}")
    return dict(kat=kat_no, f_satir=len(j), f_en_buyuk_fark=fark, f_secim_dosyasi=sec.to_dict("records")[:1])


def kat(c, df_m, p, k, kat_no, gercek=True):
    """Bir kat: parametre secimi (egitim istasyonlari, egitim yillari) ve test satirlarinda A1, A3, A5, A6."""
    P = E.hazirla(c, df_m)
    T, dates, st_no, lat, lon, rak = E.veri(P)
    tr = P["g_tr"].st_global
    hold = P["g_all"].st_global[P["hold_local"]]
    if set(st_no[tr]) & set(c.holdout):
        raise RuntimeError("test istasyonu egitimde")
    ti_tr, ti_te = P["idx"]["train"], P["idx"]["test"]
    gecmis = np.nonzero(dates <= dates[ti_tr].max())[0]          # secim: yalniz egitim yillari
    tum = np.arange(len(dates))
    son_egitim = pd.Timestamp(f"{max(c.train_years)}-12-31")
    dur(len(np.intersect1d(tr, hold)) == 0, "a/b: egitim ve test istasyonlari kesisiyor")
    dur(set(st_no[hold]) == set(c.holdout) and len(tr) + len(hold) == len(st_no), "a/b: istasyon bolmesi")
    dur(not P["ic_global"].size, "a: ic dogrulama acik (katin butun egitim istasyonlari kullanilmali)")
    dur(dates[gecmis].max() <= son_egitim and (son_egitim == pd.Timestamp("2022-12-31") or not gercek), "a: secim tarihi > 2022-12-31")
    T_g = T.copy()
    iz = [list(E.DERINLIKLER).index(d) for d in HEDEF]
    T_g[np.ix_(np.arange(T.shape[0]), hold, iz)] = np.nan
    rs_tr = sapma(T_g, dates, st_no, lat, lon, rak, tr, tr, p, k, gecmis, SIG + HEDEF)
    dur(set(rs_tr.istasyon_no) <= set(st_no[tr]) and rs_tr.tarih.max() <= son_egitim, "a: secim satirlari")
    rs = {dz: rs_tr[rs_tr.derinlik == dz][["istasyon_no", "tarih", "r"]] for dz in SIG}
    par = {}
    for d in HEDEF:
        t = E.artik_tablosu(rs[20], rs_tr[rs_tr.derinlik == d].copy())
        dur(set(t.istasyon_no) <= set(st_no[tr]) and t.tarih.max() <= son_egitim and t.r.notna().all(), "a: secim tablosu")
        s3 = E.tam_secim(t)
        t5 = E2.a5_ozellik(rs[20], t, E2.DZ[d])
        en = None
        for kp in E2.KAPPA:
            S = E.toplamlar(t5, f"s_{kp}").sum()
            g = S.smr / S.smm
            rss = S.srr - 2 * g * S.smr + g ** 2 * S.smm
            if en is None or rss < en[0]:
                en = (rss, kp, g)
        kol6 = [f"f{dz}_{L}" for dz in SIG for L in E2.L6]
        beta, alfa = ridge_sec(a6_matris(rs, t), kol6)
        par[d] = dict(L=int(s3["L_A3"]), gamma_A3=float(s3["gamma_A3"]), kappa=int(en[1]), gamma_A5=float(en[2]),
                      alfa_A6=float(alfa), beta_A6=beta)
    a1 = E.IDWM.satirlar(T[ti_te], dates[ti_te], st_no, lat, lon, tr, [hold], p, k, rak)
    a1 = a1[a1.derinlik.isin(HEDEF)].dropna(subset=["y_true"])
    T_s = T.copy()
    T_s[np.ix_(np.arange(T.shape[0]), hold, iz)] = 1e4
    a1g = E.IDWM.satirlar(T_s[ti_te], dates[ti_te], st_no, lat, lon, tr, [hold], p, k, rak)
    a1g = a1g[a1g.derinlik.isin(HEDEF)]
    dur((a1g.y_true == 1e4).all(), "c: sahte kopya")
    j = a1[ANAHTAR + ["y_pred"]].merge(a1g[ANAHTAR + ["y_pred"]], on=ANAHTAR, how="left", suffixes=("", "_g"))
    dur(np.array_equal(j.y_pred.to_numpy(), j.y_pred_g.to_numpy(), equal_nan=True), "c: A1 tahmini test olcumune bagli")
    rs_te = sapma(T_g, dates, st_no, lat, lon, rak, tr, hold, p, k, tum, SIG)
    dur(set(rs_te.istasyon_no) <= set(st_no[hold]), "b: sapma hedefleri test istasyonu degil")
    rh = {dz: rs_te[rs_te.derinlik == dz][["istasyon_no", "tarih", "r"]] for dz in SIG}
    kf = kontrol_f(a1, kat_no, p, k) if gercek else dict(f_satir=0, f_en_buyuk_fark=float("nan"))
    out = []
    for d in HEDEF:
        x = a1[a1.derinlik == d].copy()
        x["r"] = np.nan                                              # test olcumu secim/ozniteliklere girmez
        x = E.artik_tablosu(rh[20], x)
        x = E2.a5_ozellik(rh[20], x, E2.DZ[d])
        x = a6_matris(rh, x)
        q = par[d]
        x["A1"] = x.y_pred
        x["A3"] = x.y_pred + q["gamma_A3"] * x[f"m_{q['L']}"].fillna(0.0)
        x["A5"] = x.y_pred + q["gamma_A5"] * x[f"s_{q['kappa']}"].fillna(0.0)
        X6 = np.column_stack([np.ones(len(x)), x[[f"f{dz}_{L}" for dz in SIG for L in E2.L6]].to_numpy(float)])
        x["A6"] = x.y_pred + X6 @ q["beta_A6"]
        if x[["A1", "A3", "A5", "A6"]].isna().any().any():
            raise RuntimeError("tahminsiz satir")
        if dates[ti_te].min() > x.tarih.min() or x.tarih.max() > dates[ti_te].max():
            raise RuntimeError("test disi tarih")
        out.append(x[ANAHTAR + ["y_true", "A1", "A3", "A5", "A6"]].assign(kat=kat_no))
    o = pd.concat(out, ignore_index=True)
    dur(set(o.derinlik) == set(HEDEF), "d: derinlik 50/100 disinda")
    dur(not o.duplicated(ANAHTAR).any() and len(o) == len(a1), "d: satir kumesi")
    dur(set(o.istasyon_no) <= set(st_no[hold]), "d: test istasyonu disinda satir")
    secim = pd.DataFrame([dict(kat=kat_no, derinlik=d, **{a: b for a, b in v.items() if a != "beta_A6"}) for d, v in par.items()])
    print(f"kat {kat_no}: kontroller a-f tamam (f: {kf['f_satir']} satir, en buyuk fark {kf['f_en_buyuk_fark']:.2e})", flush=True)
    return o, secim


def degerlendir(tah, istasyonlar_82, hibrit=None):
    """tah: {yontem: {tohum: [kat df]}}; istasyon RMSE 50+100 birlikte; karsilastirmalar."""
    # KB.model_rmse(...)[0] ile ayni tanim (tohum basina istasyon RMSE'si, katlarda ortalama, sonra tohum ortalamasi);
    # model_rmse kat sayisini 5 sabitledigi icin dogrudan kat_ortalama_rmse kullanilir.
    r = {m: pd.concat([KB.kat_ortalama_rmse(kl).rename(s) for s, kl in d.items()], axis=1).mean(1) for m, d in tah.items()}
    kume = {"82": istasyonlar_82, "74": [s for s in istasyonlar_82 if s not in SPATIAL_HOLDOUT]}
    cift = [("A5", "A1", "ana: sig sensorun degeri"), ("A5", "A6", "esdegerlik"), ("A5", "A3", "ikincil")]
    if "hibrit" in tah:
        cift.insert(0, ("hibrit", "A5", "ana: hibrit - A5"))
    satir = []
    for m, ref, rol in cift:
        for kad, ist in kume.items():
            ist = [s for s in ist if s in r[m].index and s in r[ref].index]
            if len(ist) < 2:
                continue
            o, karar = KB.karsilastir(r[m], r[ref], ist)
            satir.append(dict(model=m, referans=ref, rol=rol, kume=kad, **o, karar=karar if rol != "ikincil" else ""))
    return pd.DataFrame(satir)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--kos", action="store_true", help="GERCEK son test (bir kez)")
    g.add_argument("--sinama", action="store_true", help="ornek veri, test disi yillar (2015-2019 secim, 2021-22 'test')")
    ap.add_argument("--hibrit", help="hibrit 2. asama koku: <kok>/hibrit_a5_fold<k>_s<s>/pred_derin_spatial_test_gnn_s<s>.csv.gz")
    a = ap.parse_args()
    IDW_P, IDW_K = (1, 1, 2, 2, 1), (6, 6, 8, 8, 8)
    if a.sinama:
        df_m = pd.read_csv(os.path.join(KOK, "DATASET", "master_dataset_qc_ornek.csv"), parse_dates=["tarih"])
        df_m = df_m[df_m.tarih.dt.year <= 2022]                    # 2023-25 okunmaz
        katlar = [tuple(SPATIAL_HOLDOUT)]
        yil = dict(train_years=tuple(range(2015, 2020)), val_years=(2020,), test_years=(2021, 2022))
        out = os.path.join(ANA_DIZIN, "results", "folds", "analysis_deep_final_test_check")
    else:
        df_m = pd.read_csv(SoilConfig().master_path, parse_dates=["tarih"])
        katlar = [kat_istasyonlari(k) for k in range(5)]
        yil = {}
        out = os.path.join(ANA_DIZIN, "results", "folds", "analysis_deep_final_test")
    os.makedirs(out, exist_ok=True)
    kontrol_e()
    parca, sec = [], []
    for kn, h in enumerate(katlar):
        c = SoilConfig(holdout=h, **yil)
        c.__post_init__()
        x, s = kat(c, df_m, float(IDW_P[kn]), IDW_K[kn], kn, gercek=a.kos)
        parca.append(x); sec.append(s)
        print(f"kat {kn}: {x.istasyon_no.nunique()} istasyon, {len(x)} satir", flush=True)
    t = pd.concat(parca, ignore_index=True)
    sec = pd.concat(sec, ignore_index=True)
    sec.to_csv(os.path.join(out, "secimler.csv"), index=False)
    t.to_csv(os.path.join(out, "tahminler.csv.gz"), index=False)
    tah = {m: {-1: [t[t.kat == kn][ANAHTAR + ["y_true"]].assign(y_pred=t[t.kat == kn][m].to_numpy()) for kn in range(len(katlar))]}
           for m in ("A1", "A3", "A5", "A6")}
    if a.hibrit:
        tah["hibrit"] = {}
        for s in range(3):
            kl = []
            for kn in range(len(katlar)):
                hb = pd.read_csv(os.path.join(a.hibrit, f"hibrit_a5_fold{kn}_s{s}", f"pred_derin_spatial_test_gnn_s{s}.csv.gz"),
                                 parse_dates=["tarih"])
                ref = t[t.kat == kn][ANAHTAR + ["y_true"]]
                j = ref.merge(hb[ANAHTAR + ["y_pred"]], on=ANAHTAR, how="left")
                if j.y_pred.isna().any():
                    raise RuntimeError(f"hibrit kat {kn} s{s}: satir eksik")
                kl.append(j)
            tah["hibrit"][s] = kl
    ist = sorted(t.istasyon_no.unique())
    kars = degerlendir(tah, ist)
    kars.to_csv(os.path.join(out, "karsilastirma.csv"), index=False)
    print(sec.to_string(index=False))
    print(kars.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
