import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 220)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))
sys.path.insert(0, os.path.join(ANA_DIZIN, "analysis"))

import saglamlik_kontrolleri as DR
import katli_birincil_karsilastirma as KB
import kayma_kalicilik_kalibrasyon as KK
import idw_karsilastirma_modeli as IW
import kriging_karsilastirma_modeli as KR
from toprak_ayarlar import DERINLIKLER, IDX_T, SPATIAL_HOLDOUT
from toprak_gnn_calistir import hazirla
from ortak_veri import haversine_km

SON = DR.SON
OUT = os.path.join(SON, "folds", "analysis_robustness_extra")
DRV = os.path.join(SON, "folds", "analysis_robustness")
ANAHTAR = KB.ANAHTAR
B, TOHUM = KB.BOOT_N, KB.BOOT_TOHUM
TOL = 1e-9
K_LISTE, KM_TOHUM = (10, 16, 25), (2026, 2027, 2028)


def dur(kosul, mesaj):
    if not kosul:
        raise SystemExit(f"SAGLAMA BASARISIZ: {mesaj}")


def kismi(x, y, z):
    r = np.corrcoef(np.vstack([x, y, z]))
    rxy, rxz, ryz = r[0, 1], r[0, 2], r[1, 2]
    k = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    n = len(x)
    t = k * np.sqrt((n - 3) / (1 - k ** 2))
    return float(k), float(2 * stats.t.sf(abs(t), n - 3)), float(rxz), float(ryz)


def k1(tum):
    idw = pd.concat(tum[KB.REFERANS][0], ignore_index=True)
    a4 = pd.read_csv(os.path.join(DRV, "a4_istasyon.csv"))
    satir = []
    for m in ("hibrit_ceza1", "hibrit"):
        t = tum[m]
        yh = np.mean([pd.concat(t[s], ignore_index=True).y_pred.to_numpy(float) for s in t], axis=0)
        df = idw[ANAHTAR].assign(c=yh - idw.y_pred.to_numpy(float), o=idw.y_true - idw.y_pred, z=idw.y_pred)
        st = df.groupby("istasyon_no")[["c", "o", "z"]].mean()
        ref = a4[a4.model == m].set_index("istasyon_no").reindex(st.index)
        dur(float(np.abs(ref.c - st.c).max()) <= TOL and float(np.abs(ref.o - st.o).max()) <= TOL, f"K1 {m} c, o")
        for kad, kume in DR.kumeler(st.index).items():
            x = st.loc[kume]
            kp, pp, rcz, roz = kismi(x.c.to_numpy(), x.o.to_numpy(), x.z.to_numpy())
            rc, ro, rz = (stats.rankdata(x[c]) for c in ("c", "o", "z"))
            ks, ps, scz, soz = kismi(rc, ro, rz)
            satir.append(dict(model=m, kume=kad, n=len(x), spearman=float(stats.spearmanr(x.c, x.o).statistic),
                              kismi_spearman=ks, p_kismi_spearman=ps, pearson=float(np.corrcoef(x.c, x.o)[0, 1]),
                              kismi_pearson=kp, p_kismi_pearson=pp, spearman_c_z=scz, spearman_o_z=soz,
                              pearson_c_z=rcz, pearson_o_z=roz))
    print("T1 gecti: c ve o a4_istasyon.csv ile ayni", flush=True)
    return pd.DataFrame(satir)


def k2(ciftler49):
    df = pd.concat([pd.read_csv(os.path.join(SON, f"idw_elevation_fold{k}", "pred_spatial_full_idw.csv.gz"),
                                parse_dates=["tarih"]) for k in range(KB.N_KAT)], ignore_index=True)
    dur(not df.duplicated(ANAHTAR).any() and df.istasyon_no.nunique() == 82, "K2 spatial_full satirlari")
    df = df.assign(e=df.y_pred - df.y_true)
    kk, _ = KK.kalicilik(df)
    kay = pd.read_csv(os.path.join(SON, "folds", "analysis_offset_calibration", "a_kalicilik.csv"))
    dur(np.allclose(kk.spearman, kay.spearman, atol=TOL) and (kk.n_istasyon == kay.n_istasyon).all(), "K2 a_kalicilik")
    print("T2 gecti: 2015-2019 / 2020-2023 kaliciligi a_kalicilik.csv ile ayni", flush=True)
    y = df.tarih.dt.year
    a = df[(y >= 2020) & (y <= 2023)].groupby(["istasyon_no", "derinlik"]).e.mean().rename("k2023")
    b = df[y >= 2024].groupby(["istasyon_no", "derinlik"]).e.mean().rename("k2425")
    j = pd.concat([a, b], axis=1).dropna().reset_index()
    satir = []
    for ad, x in (("butun ciftler", j),
                  ("49 gec kirilmali cift cikarildi",
                   j[~pd.MultiIndex.from_arrays([j.istasyon_no, j.derinlik]).isin(list(ciftler49))])):
        for d in DR.DER:
            u = x[x.derinlik == d]
            r = stats.spearmanr(u.k2023, u.k2425)
            satir.append(dict(kume=ad, derinlik=d, n_istasyon=len(u), spearman=float(r.statistic), p=float(r.pvalue),
                              pearson=float(np.corrcoef(u.k2023, u.k2425)[0, 1]),
                              ayni_isaret=float((np.sign(u.k2023) == np.sign(u.k2425)).mean())))
    return pd.DataFrame(satir), j


def k3():
    e = pd.read_csv(os.path.join(SON, "variant_selection", "ic_dogrulama_egrileri.csv"))
    satir = []
    for k in range(KB.N_KAT):
        kol = str(k)
        taban = float(e[(e.ayar == "a_hibrit") & (e.egitilen_epoch == 0)][kol].iloc[0])   # egitimsiz hibrit = IDW
        for ayar, g in e.groupby("ayar"):
            g = g.sort_values("egitilen_epoch")
            i = int(np.argmin(g[kol].to_numpy()))
            satir.append(dict(kat=k, ayar=ayar, e_kat=int(g.egitilen_epoch.iloc[i]), kayip_kat=float(g[kol].iloc[i]),
                              oran_egitimsiz_hibrit=float(g[kol].iloc[i]) / taban,
                              oran_28=float(g[g.egitilen_epoch == 28][kol].iloc[0]) / taban))
    t = pd.DataFrame(satir)
    ozet = []
    for k, g in t.groupby("kat"):
        en = g.loc[g.kayip_kat.idxmin()]
        cz = g[g.ayar.str.startswith("ceza")]
        cz_en = cz.loc[cz.kayip_kat.idxmin()]
        ozet.append(dict(kat=k, secilen_ayar=en.ayar, secilen_epoch=int(en.e_kat), secilen_oran=float(en.oran_egitimsiz_hibrit),
                         ceza_secimi=cz_en.ayar, ceza_epoch=int(cz_en.e_kat), ceza_oran=float(cz_en.oran_egitimsiz_hibrit),
                         ceza1_28_oran=float(g[g.ayar == "ceza1"].oran_28.iloc[0])))
    return t, pd.DataFrame(ozet)


def etiketler(df_m, k, tohum):
    from sklearn.cluster import KMeans
    st = df_m.groupby("istasyon_no")[["lat", "lon"]].first().sort_index()
    R = 6371.0
    phi, lam = np.radians(st.lat.to_numpy()), np.radians(st.lon.to_numpy())
    xy = np.column_stack([R * (lam - lam.mean()) * np.cos(phi.mean()), R * (phi - phi.mean())])
    return pd.Series(KMeans(n_clusters=k, n_init=100, random_state=tohum).fit(xy).labels_, index=st.index)


def bootstrap(D, L, nk):
    dbar = D.mean(1)
    uyeler = [np.nonzero(L == c)[0] for c in range(nk)]
    n, ns = D.shape
    rng = np.random.RandomState(TOHUM)
    ii = rng.randint(0, n, size=(B, n))
    ss = rng.randint(0, ns, size=(B, n))
    sonuc = {"i tohumlu": D[ii, ss].mean(1)}
    rng = np.random.RandomState(TOHUM)
    cc = rng.randint(0, nk, size=(B, nk))
    S = np.array([dbar[u].sum() for u in uyeler])
    N = np.array([len(u) for u in uyeler], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        sonuc["ii bloklu"] = S[cc].sum(1) / N[cc].sum(1)
    rng = np.random.RandomState(TOHUM)
    v = np.empty(B)
    for b in range(B):
        sec = np.concatenate([uyeler[c] for c in rng.randint(0, nk, size=nk)])
        v[b] = D[sec, rng.randint(0, ns, size=len(sec))].mean() if len(sec) else np.nan
    sonuc["iii bloklu+tohumlu"] = v
    out = []
    for ad, boot in sonuc.items():
        alt, ust = np.nanpercentile(boot, [2.5, 97.5])
        alt90, ust90 = np.nanpercentile(boot, [5, 95])
        karar, et = DR.etiket(dbar.mean(), alt, ust, alt90, ust90)
        out.append(dict(yontem=ad, n_istasyon=n, ort_fark=dbar.mean(), ga_alt=alt, ga_ust=ust, ga90_alt=alt90,
                        ga90_ust=ust90, karar_betimleyici=karar, etiket=et))
    return out


def fark_matrisi(per, ref, ist):
    A = per.loc[ist].to_numpy(float)
    R = ref.loc[ist].to_numpy(float)
    if R.ndim == 1:
        R = R[:, None]
    return A - R if R.shape[1] == 1 else A - R


def k4(tum, rm, df_m, ist):
    per = {m: DR.tohum_rmse(tum[m]) for m in ("hibrit", "hibrit_ceza1", "ignnk")}
    kr = rm["kriging_rakim"]
    idw = rm[KB.REFERANS]
    eski = pd.read_csv(os.path.join(DRV, "a3_bootstrap.csv"))
    eski["kume"] = eski.kume.astype(str)
    duy, ek = [], []
    for k in K_LISTE:
        for kt in KM_TOHUM:
            lab = etiketler(df_m, k, kt)
            boy = lab.value_counts()
            for m in ("hibrit", "hibrit_ceza1"):
                for kad, kume in DR.kumeler(ist).items():
                    D = fark_matrisi(per[m], idw, kume)
                    r = bootstrap(D, lab.loc[kume].to_numpy(), k)
                    for x in r:
                        duy.append(dict(k=k, kmeans_tohum=kt, kume_min=int(boy.min()), kume_max=int(boy.max()), model=m,
                                        kume=kad, **x))
                        if k == 16 and kt == 2026:
                            e = eski[(eski.model == m) & (eski.kume == kad) & (eski.yontem == x["yontem"])].iloc[0]
                            dur(abs(e.ga_alt - x["ga_alt"]) <= TOL and abs(e.ga_ust - x["ga_ust"]) <= TOL,
                                f"K4 {m} {kad} {x['yontem']}")
            if k == 16 and kt == 2026:
                for ad, D_f in (("kriging_rakim - idw_rakim", lambda s: (kr.loc[s] - idw.loc[s]).to_numpy(float)[:, None]),
                                ("hibrit_ceza1 - kriging_rakim", lambda s: fark_matrisi(per["hibrit_ceza1"], kr, s)),
                                ("ignnk - idw_rakim", lambda s: fark_matrisi(per["ignnk"], idw, s))):
                    for kad, kume in DR.kumeler(ist).items():
                        for x in bootstrap(D_f(kume), lab.loc[kume].to_numpy(), k):
                            ek.append(dict(cift=ad, kume=kad, **x))
    print("T3 gecti: k = 16, tohum 2026 bloklu bootstrap a3_bootstrap.csv ile ayni", flush=True)
    return pd.DataFrame(duy), pd.DataFrame(ek)


def k5():
    gg = pd.read_csv(os.path.join(SON, "folds", "gorulmus_gorulmemis.csv"))
    satir = []
    for m in ("hibrit", "idw_rakim"):
        t = KB.yukle(SON, m, "temporal")
        dur(t is not None, f"K5 {m} temporal tahminleri yok")
        mse_s, sab_s, rm_s = [], [], []
        for katlar in t.values():
            parca = []
            for k, df in enumerate(katlar):
                e = df.y_pred.to_numpy(float) - df.y_true.to_numpy(float)
                g = pd.DataFrame({"istasyon_no": df.istasyon_no, "derinlik": df.derinlik, "e": e, "e2": e ** 2})
                parca.append(g.groupby(["istasyon_no", "derinlik"]).agg(n=("e", "size"), b=("e", "mean"), mse=("e2", "mean")))
            x = pd.concat(parca)
            w = x.n.to_numpy(float)
            mse, sab = (w * x.mse).sum() / w.sum(), (w * x.b ** 2).sum() / w.sum()
            mse_s.append(mse)
            sab_s.append(sab)
            rm_s.append(np.sqrt(mse))
        ref = gg[(gg.model == m) & (gg.bolum == "temporal")].havuz_rmse.iloc[0]
        dur(abs(np.mean(rm_s) - ref) <= 1e-9, f"K5 {m} havuz RMSE {np.mean(rm_s)} / {ref}")
        satir.append(dict(model=m, durum="gorulmus (temporal)", havuz_mse=float(np.mean(mse_s)),
                          havuz_sabit_kayma=float(np.mean(sab_s)), havuz_zamanla_degisen=float(np.mean(mse_s) - np.mean(sab_s)),
                          havuz_sabit_payi=float(np.mean(sab_s) / np.mean(mse_s))))
    print("T4 gecti: gorulmus istasyon havuz RMSE gorulmus_gorulmemis.csv ile ayni", flush=True)
    return pd.DataFrame(satir)


def vg_oku(k):
    t = pd.read_csv(os.path.join(SON, f"kriging_elevation_fold{k}", "variogram.csv"))
    vg = {}
    for (dep, model), g in t.groupby(["derinlik", "model"]):
        a = np.zeros((12, 3))
        a[g.ay.to_numpy() - 1] = g[["c0", "c1", "a_km"]].to_numpy(float)
        vg[(int(dep), model)] = a
    s = pd.read_csv(os.path.join(SON, f"kriging_elevation_fold{k}", "secim_2023.csv"))
    return vg, s.model[0], (None if str(s.k[0]) == "hepsi" else int(s.k[0]))


def k6(df_m, ciftler49):
    eski_i, yeni_i, eski_k, yeni_k = [], [], [], []
    en_i, en_k, kayip = 0.0, 0.0, 0
    for k in range(KB.N_KAT):
        c = DR.cfg_kat(k)
        P = hazirla(c, df_m)
        X, dates, stations, idx = P["X"], P["dates"], P["stations"], P["idx"]
        st_no = stations["istasyon_no"].to_numpy()
        Tobs = X[..., IDX_T].reshape(len(dates), len(st_no), len(DERINLIKLER)).astype(float)
        lat, lon = stations["lat"].to_numpy(float), stations["lon"].to_numpy(float)
        rak = stations["rakim"].to_numpy(float) / 1000.0
        D_all = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        aylar = dates.month.to_numpy()
        tr = P["g_tr"].st_global
        hold = P["g_all"].st_global[P["hold_local"]]
        ti = idx["test"]
        vg, vm, vk = vg_oku(k)
        Tm = Tobs.copy()
        konum = {s: i for i, s in enumerate(st_no)}
        for s, d in ciftler49:
            j = konum[s]
            if j not in set(hold):
                Tm[np.ix_(ti, [j], [list(DERINLIKLER).index(d)])] = np.nan
        for ad, T_, idw_l, kr_l in (("eski", Tobs, eski_i, eski_k), ("yeni", Tm, yeni_i, yeni_k)):
            di = IW.satirlar(T_[ti], dates[ti], st_no, lat, lon, tr, [hold], DR.IDW_P[k], DR.IDW_K[k], rak)
            dk = KR.satirlar(T_[ti], dates[ti], aylar[ti], st_no, D_all, tr, [hold], vg, vm, vk, rak)
            for d_, l_ in ((di, idw_l), (dk, kr_l)):
                d_ = d_.dropna(subset=["y_pred"]).assign(tarih=lambda x: pd.to_datetime(x.tarih).dt.strftime("%Y-%m-%d"))
                l_.append(d_.sort_values(ANAHTAR).reset_index(drop=True).assign(kat=k))
        for kl, uz, l_ in (("idw_elevation_fold{k}", "idw", eski_i), ("kriging_elevation_fold{k}", "kriging", eski_k)):
            kay = pd.read_csv(os.path.join(SON, kl.format(k=k), f"pred_spatial_test_{uz}.csv.gz")).sort_values(ANAHTAR)
            kay = kay.reset_index(drop=True)
            x = l_[-1]
            dur(len(x) == len(kay) and (x[ANAHTAR].to_numpy() == kay[ANAHTAR].to_numpy()).all(), f"K6 {uz} kat {k} satirlar")
            f = float(np.abs(x.y_pred - kay.y_pred).max())
            dur(f <= 1e-6, f"K6 {uz} kat {k} fark {f}")
            if uz == "idw":
                en_i = max(en_i, f)
            else:
                en_k = max(en_k, f)
        kayip += len(eski_i[-1]) - len(yeni_i[-1]) + len(eski_k[-1]) - len(yeni_k[-1])
    print(f"T5 gecti: maskesiz IDW ve kriging kayitli spatial_test ile ayni (en buyuk fark {en_i:.1e}, {en_k:.1e} C)",
          flush=True)
    return eski_i, yeni_i, eski_k, yeni_k, kayip


def k6_ozet(eski_i, yeni_i, eski_k, yeni_k, kayip):
    r = {ad: KB.kat_ortalama_rmse([x[ANAHTAR + ["y_true", "y_pred"]] for x in l_])
         for ad, l_ in (("idw_eski", eski_i), ("idw_yeni", yeni_i), ("kriging_eski", eski_k), ("kriging_yeni", yeni_k))}
    ist = np.sort(r["idw_eski"].index)
    satir = []
    for yon in ("idw", "kriging"):
        d = (r[f"{yon}_yeni"].loc[ist] - r[f"{yon}_eski"].loc[ist])
        satir.append(dict(olcu=f"{yon} istasyon RMSE degisimi (yeni - eski)", ort=float(d.mean()), medyan=float(d.median()),
                          en_buyuk_mutlak=float(d.abs().max()), degisen_istasyon=int((d.abs() > 1e-9).sum()),
                          ort_rmse_eski=float(r[f"{yon}_eski"].loc[ist].mean()), ort_rmse_yeni=float(r[f"{yon}_yeni"].loc[ist].mean())))
    tab = pd.DataFrame(satir)
    kars = []
    for durum in ("eski", "yeni"):
        for kad, kume in DR.kumeler(ist).items():
            o, karar = KB.karsilastir(r[f"kriging_{durum}"], r[f"idw_{durum}"], kume)
            kars.append(dict(durum=durum, kume=kad, cift="kriging_rakim - idw_rakim", **o, karar=karar))
    ist_t = pd.DataFrame({k: v.loc[ist] for k, v in r.items()}).rename_axis("istasyon_no")
    return tab.assign(kayip_satir=kayip), pd.DataFrame(kars), ist_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sadece-saglama", action="store_true")
    a = ap.parse_args()
    df_m = pd.read_csv(DR.MASTER, parse_dates=["tarih"])
    tum = DR.yukle_hepsi()
    rm = DR.rmse_hepsi(tum)
    ist = np.sort(pd.concat(tum[KB.REFERANS][0]).istasyon_no.unique())
    _, ciftler49, _ = DR.a1_eslestir(df_m)
    r1 = k1(tum)
    r2, r2j = k2(ciftler49)
    r3, r3o = k3()
    r4, r4e = k4(tum, rm, df_m, ist)
    r5 = k5()
    r6 = k6(df_m, ciftler49)
    if a.sadece_saglama:
        print("Butun saglamalar gecti; --sadece-saglama: dosya yazilmadi.")
        return
    os.makedirs(OUT, exist_ok=True)
    r1.to_csv(os.path.join(OUT, "k1_kismi_korelasyon.csv"), index=False)
    r2.to_csv(os.path.join(OUT, "k2_kalicilik_test.csv"), index=False)
    r2j.to_csv(os.path.join(OUT, "k2_kayma_donem.csv"), index=False)
    r3.to_csv(os.path.join(OUT, "k3_kat_ici_secim.csv"), index=False)
    r3o.to_csv(os.path.join(OUT, "k3_ozet.csv"), index=False)
    r4.to_csv(os.path.join(OUT, "k4_kume_duyarliligi.csv"), index=False)
    r4e.to_csv(os.path.join(OUT, "k4_s4_ekleri.csv"), index=False)
    r5.to_csv(os.path.join(OUT, "k5_gorulmus_ayrisim.csv"), index=False)
    t6, k6k, k6i = k6_ozet(*r6)
    t6.to_csv(os.path.join(OUT, "k6_komsu_kirilma.csv"), index=False)
    k6k.to_csv(os.path.join(OUT, "k6_kriging_idw.csv"), index=False)
    k6i.to_csv(os.path.join(OUT, "k6_istasyon.csv"))
    print("Ciktilar:", OUT)


if __name__ == "__main__":
    main()
