import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

pd.set_option("compute.use_numexpr", False)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))

from toprak_ayarlar import DERINLIKLER, IDX_T, SoilConfig
import idw_karsilastirma_modeli as IDWM
from toprak_katlar import kat_istasyonlari
from toprak_gnn_calistir import hazirla
from rf_mlp_transformer_karsilastirma_modelleri import RF_PARAMS, Olcek, tablo
from rfsi_karsilastirma_modeli import birlestir, ozellik_kubu, satir_ozellik

N_IZGARA = (10, 25, 35)
MF_IZGARA = ("sqrt", 1 / 3, 1.0)
YAPRAK_IZGARA = (5, 20)
AGAC = 60
KAYITLI_N = (3, 10, 10, 3, 3)
ANAHTAR = ["istasyon_no", "derinlik", "tarih"]


def olcu(S, y):
    e = y - S.y_true.to_numpy(float)
    ist = pd.Series(e ** 2).groupby(S.istasyon_no.to_numpy()).mean() ** 0.5
    return float(ist.mean()), float(np.sqrt(np.mean(e ** 2))), int(len(ist))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kat", type=int, required=True)
    ap.add_argument("--master")
    ap.add_argument("--out", default=os.path.join(ANA_DIZIN, "results", "folds", "analysis_rfsi_tuning"))
    ap.add_argument("--sadece-taban", action="store_true")
    ap.add_argument("--kucuk", action="store_true", help="sinama: kucuk alt kume, 2 agac, kucuk izgara")
    a = ap.parse_args()
    t0 = time.time()
    k = a.kat
    sec = pd.read_csv(os.path.join(ANA_DIZIN, "results", "folds", "analysis_inner_validation", f"ic_secim_2023_kat{k}.csv"))
    p, kk = float(sec.p.iloc[0]), int(sec.k.iloc[0])
    cfg = SoilConfig(uydu_derinlik="tum", cikar=("soil_moisture",))
    if a.master:
        cfg.master_path = a.master
    cfg.holdout, cfg.ic_dogrulama = kat_istasyonlari(k), True
    df = None
    if a.kucuk:
        var = set(pd.read_csv(cfg.master_path, usecols=["istasyon_no"]).istasyon_no)
        h = tuple(s for s in cfg.holdout if s in var)
        from toprak_ayarlar import SPATIAL_HOLDOUT
        cfg.holdout = h if h else tuple(s for s in SPATIAL_HOLDOUT if s in var)
    if not a.kucuk:
        df = pd.read_csv(cfg.master_path, parse_dates=["tarih"])
        df = df[df.tarih.dt.year <= 2023]
        cfg.test_years = (2023,)                           # hazirla bos olmayan test bolumu ister; kullanilmaz
    P = hazirla(cfg, df)
    if P["dates"].max() > pd.Timestamp("2023-12-31") and not a.kucuk:
        raise RuntimeError("2024-25 okunmus")
    idx = P["idx"]
    tr, ic = P["g_tr"].st_global, P["ic_global"]
    hold = P["g_all"].st_global[P["hold_local"]]
    if len(np.intersect1d(tr, hold)) or len(np.intersect1d(ic, hold)) or len(np.intersect1d(tr, ic)) or not len(ic):
        raise RuntimeError("istasyon bolmesi")
    n_max = max(max(N_IZGARA), max(KAYITLI_N))
    bol = {"ic_val2023": (idx["val"], ic, [(ic, tr)])}
    if not a.sadece_taban:
        bol["train"] = (idx["train"], tr, [(tr, tr)])
    G, S, F = {}, {}, {}
    for ad, (ti, st, gruplar) in bol.items():
        G[ad], S[ad] = tablo(P, ti, st)
        if a.kucuk:
            s_ = np.sort(np.random.RandomState(0).choice(len(S[ad]), min(len(S[ad]), 4000), replace=False))
            G[ad], S[ad] = G[ad][s_], S[ad].iloc[s_].reset_index(drop=True)
        kup = ozellik_kubu(P, ti, gruplar, n_max, p, kk)
        F[ad] = {n: satir_ozellik(P, ti, S[ad], kup, n, n_max) for n in sorted(set(N_IZGARA) | {KAYITLI_N[k]})}
        del kup
        print(f"kat {k} {ad}: {len(S[ad])} satir  [{time.time() - t0:.0f} s]", flush=True)
    # taban: rakim-IDW (oznitelik sutunu; her n'de ayni)
    taban = F["ic_val2023"][N_IZGARA[0]][:, -1]
    if not np.isfinite(taban).all():
        raise RuntimeError("taban tahminsiz satir")
    ist_t, hav_t, n_ist = olcu(S["ic_val2023"], taban)
    X, dates, st = P["X"], P["dates"], P["stations"]
    Tobs = X[..., IDX_T].reshape(len(dates), len(st), len(DERINLIKLER)).astype(float)
    ref = IDWM.satirlar(Tobs[idx["val"]], dates[idx["val"]], st.istasyon_no.to_numpy(), st.lat.to_numpy(float),
                        st.lon.to_numpy(float), tr, [ic], p, kk, st.rakim.to_numpy(float) / 1000.0)
    j = S["ic_val2023"][ANAHTAR].assign(tab=taban).merge(ref[ANAHTAR + ["y_pred"]], on=ANAHTAR, how="left")
    fark = float(np.abs(j.tab - j.y_pred).max())
    if j.y_pred.isna().any() or not fark < 1e-4:
        raise RuntimeError(f"taban saglamasi tutmadi: en buyuk fark {fark}, eslesmeyen {int(j.y_pred.isna().sum())}")
    satir = [dict(kat=k, ayar="taban_rakim_idw", n=np.nan, max_features="", yaprak=np.nan, idw_ozellik=False, kayitli=False,
                  istasyon_rmse_ort=ist_t, havuz_rmse=hav_t, n_istasyon=n_ist, p=p, k=kk, sn=0.0)]
    print(f"kat {k}: taban saglamasi tamam (p={p}, k={kk}, {n_ist} ic dogrulama istasyonu, {len(j)} satir, fark {fark:.1e})", flush=True)
    if a.sadece_taban:
        return
    olc0 = Olcek(G["train"], S["train"])
    ytr = olc0.y(S["train"])
    ayarlar = [(n, mf, y, True, False) for n in N_IZGARA for mf in MF_IZGARA for y in YAPRAK_IZGARA]
    ayarlar += [(n, mf, y, False, False) for n in N_IZGARA for mf in MF_IZGARA for y in YAPRAK_IZGARA]
    ayarlar += [(KAYITLI_N[k], 1.0, 5, True, True)]
    if a.kucuk:
        ayarlar = ayarlar[:2] + ayarlar[18:19] + ayarlar[-1:]
    for n, mf, y, idw_oz, kayitli in ayarlar:
        t1 = time.time()
        Ftr, Fva = F["train"][n], F["ic_val2023"][n]
        if not idw_oz:
            Ftr, Fva = Ftr[:, :-1], Fva[:, :-1]
        Xtr, Xva = birlestir(G["train"], Ftr), birlestir(G["ic_val2023"], Fva)
        olc = Olcek(Xtr, S["train"])
        rf = RandomForestRegressor(n_estimators=2 if a.kucuk else AGAC, max_depth=RF_PARAMS["max_depth"], max_features=mf,
                                   min_samples_leaf=y, n_jobs=-1, random_state=0)
        rf.fit(olc.x(Xtr), ytr)
        yv = olc.geri(rf.predict(olc.x(Xva)), S["ic_val2023"])
        ist, hav, _ = olcu(S["ic_val2023"], yv)
        mf_ad = mf if isinstance(mf, str) else ("1/3" if abs(mf - 1 / 3) < 1e-9 else str(mf))
        satir.append(dict(kat=k, ayar="kayitli_ayar" if kayitli else "izgara", n=n, max_features=mf_ad, yaprak=y,
                          idw_ozellik=idw_oz, kayitli=kayitli, istasyon_rmse_ort=ist, havuz_rmse=hav, n_istasyon=n_ist,
                          p=p, k=kk, sn=time.time() - t1))
        print(f"kat {k}: n={n} mf={mf_ad} yaprak={y} idw={idw_oz} kayitli={kayitli}  [{time.time() - t1:.0f} s]", flush=True)
        del rf
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(satir).to_csv(os.path.join(a.out, f"rfsi_ayar_kat{k}{'_kucuk' if a.kucuk else ''}.csv"), index=False)
    print(f"Bitti kat {k}  [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
