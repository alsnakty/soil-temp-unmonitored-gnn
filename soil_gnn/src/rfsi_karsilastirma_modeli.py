import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from toprak_ayarlar import DERINLIKLER, IDX_T, SoilConfig, ortam_bilgisi
from toprak_degerlendirme import metrics, station_metrics
from toprak_katlar import en_yakin_gorunur, kat_istasyonlari
from toprak_veri_ve_graf import idw_rakim
from toprak_gnn_calistir import hazirla
from rf_mlp_transformer_karsilastirma_modelleri import RF_PARAMS, Olcek, egit_rf, tablo

pd.set_option("compute.use_numexpr", False)   # pyg ortamindaki numexpr 2.11.0 buyuk dizilerde ara sira yanlis sonuc veriyor

ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_ADAY = (3, 6, 10)
SECIM_AGAC = 60
GUN_PARCA = 120
ND = len(DERINLIKLER) - 1


def komsu_ozellik(T, kaynak, g, n_kom, p, k):
    """T, kaynak: (K, n) gun-derinlik dilimleri. Her istasyon i icin kendisi haric, uzaklik sirasinda ilk n_kom kaynak:
    olcum (K, n, n_kom), uzaklik km (K, n, n_kom); ve rakim egimli IDW (K, n). Yetmeyen sira NaN."""
    K, n = T.shape
    M = kaynak[:, None, :] & ~np.eye(n, dtype=bool)[None]                 # (K, i, j)
    sirali = np.take_along_axis(M, np.broadcast_to(g.st_order, M.shape), axis=2)
    sayi = np.cumsum(sirali, axis=2)
    Tk = np.where(kaynak, T, np.nan)
    ii = np.arange(n)[None, :]
    olc = np.full((K, n, n_kom), np.nan)
    uz = np.full((K, n, n_kom), np.nan)
    for r in range(n_kom):
        sec = sirali & (sayi == r + 1)
        var = sec.any(2)
        j = g.st_order[ii, sec.argmax(2)]                                     # (K, n) global komsu indisi
        olc[..., r] = np.where(var, np.take_along_axis(Tk, j, axis=1), np.nan)
        uz[..., r] = np.where(var, g.st_dist[ii, j], np.nan)
    return olc, uz, idw_rakim(T, kaynak, g, p, k)


def ozellik_kubu(P, ti, gruplar, n_kom, p, k):
    """ti gunleri icin (gun, istasyon, derinlik>0, 2*n_kom+1) ozellik kubu: [olcum x n_kom, uzaklik x n_kom, IDW_rakim].
    gruplar: [(hedef global indisleri, gorunur global indisleri)]; her hedef yalniz kendi grubunun gorunurleriyle."""
    g = P["g_all"]
    n = g.n_st
    T = P["X"][ti][..., IDX_T].reshape(len(ti), n, ND + 1)[:, :, 1:].astype(float)
    kup = np.full((len(ti), n, ND, 2 * n_kom + 1), np.nan, np.float32)
    for hedef, gor in gruplar:
        vis = np.zeros(n, bool)
        vis[gor] = True
        for a in range(0, len(ti), GUN_PARCA):
            Tg = T[a:a + GUN_PARCA]
            G = len(Tg)
            Kd = Tg.transpose(0, 2, 1).reshape(-1, n)                          # (G*ND, n)
            olc, uz, gids = komsu_ozellik(Kd, np.isfinite(Kd) & vis[None, :], g, n_kom, p, k)
            f = np.concatenate([olc, uz, gids[..., None]], -1).reshape(G, ND, n, -1).transpose(0, 2, 1, 3)
            kup[a:a + G, hedef] = f[:, hedef]
    return kup


def satir_ozellik(P, ti, satir, kup, n_kom, n_max):
    """tablo() satirlarina hizali [olcum x n_kom, uzaklik x n_kom, IDW_rakim] (kup n_max ile kuruldu)."""
    di = pd.Index(P["dates"][ti]).get_indexer(satir.tarih)
    si = pd.Index(P["stations"]["istasyon_no"]).get_indexer(satir.istasyon_no)
    dd = np.searchsorted(np.asarray(DERINLIKLER[1:]), satir.derinlik.to_numpy())
    if (di < 0).any() or (si < 0).any():
        raise RuntimeError("satir - kup eslesmesi bozuk")
    F = kup[di, si, dd]
    return np.column_stack([F[:, :n_kom], F[:, n_max:n_max + n_kom], F[:, -1]])


def birlestir(G, F):
    """Kovaryatlar + komsu ozellikleri; LST eksik isaretleri sonda kalir (Olcek son 2 sutunu olceklemez)."""
    return np.column_stack([G[:, :-2], F, G[:, -2:]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kat", type=int, help="uzaysal katli dogrulama: 0-4 (toprak_katlar.py); ic dogrulamayi da acar")
    ap.add_argument("--ic-dogrulama", action="store_true", help="katli 1. asama: n secimi ic dogrulama istasyonlariyla")
    ap.add_argument("--sadece-secim", action="store_true", help="n secilir, secim_n_2023.csv yazilir, son model egitilmez")
    ap.add_argument("--n-kaynak", metavar="SECIM_CSV", help="katli 2. asama: n = 1. asama secim_n_2023.csv'nin ilk satiri")
    ap.add_argument("--idw-p", type=float, default=2.0, help="IDW_rakim ozniteligi uzaklik ussu (katli kosuda o katin secimi)")
    ap.add_argument("--idw-k", type=int, default=6, help="IDW_rakim ozniteligi en yakin k (katli kosuda o katin secimi)")
    ap.add_argument("--master", help="master_dataset_qc.csv yolu (varsayilan SoilConfig.master_path)")
    ap.add_argument("--out", help="cikti klasoru (varsayilan results/rfsi_s<seed>[_fold<k>])")
    ap.add_argument("--kucuk", action="store_true", help="duman testi: kucuk alt kume, az agac")
    a = ap.parse_args()
    np.random.seed(a.seed)

    cfg = SoilConfig(uydu_derinlik="tum", cikar=("soil_moisture",))
    if a.master:
        cfg.master_path = a.master
    n_sabit = int(pd.read_csv(a.n_kaynak).n.iloc[0]) if a.n_kaynak else None
    if n_sabit is not None and (a.ic_dogrulama or a.sadece_secim):
        raise ValueError("--n-kaynak (2. asama) ile --ic-dogrulama / --sadece-secim (1. asama) birlikte kullanilmaz.")
    if a.kat is not None:
        cfg.holdout = kat_istasyonlari(a.kat)
        if not a.ic_dogrulama and n_sabit is None:
            raise ValueError("Katli kosuda asama belirtilmeli: --ic-dogrulama (1. asama) ya da --n-kaynak (2. asama).")
    if a.ic_dogrulama:
        cfg.ic_dogrulama = True
    etiket = f"rfsi_s{a.seed}"
    out = (a.out or os.path.join(ANA_DIZIN, "results", f"{etiket}" + ("" if a.kat is None else f"_fold{a.kat}"))) \
        + ("_small" if a.kucuk else "")
    t0 = time.time()
    P = hazirla(cfg)
    idx = P["idx"]
    tr = P["g_tr"].st_global
    hold = P["g_all"].st_global[P["hold_local"]]
    tr_tum = np.setdiff1d(np.arange(len(P["stations"])), hold)
    ic = P["ic_global"]
    gr = [tr[g_] for g_ in P["groups"]]
    gruplu = [(h, np.setdiff1d(tr, h)) for h in gr]
    bol = {"train": (idx["train"], tr, [(tr, tr)]),
           "val2023": (idx["val"], tr, gruplu),
           "temporal": (idx["test"], tr, gruplu),
           "spatial_test": (idx["test"], hold, [(hold, tr_tum)]),
           "spatial_full": (np.arange(len(P["dates"])), hold, [(hold, tr_tum)]),
           "spatial_val2023": (idx["val"], hold, [(hold, tr_tum)])}
    if cfg.ic_dogrulama:
        bol["ic_val2023"] = (idx["val"], ic, [(ic, tr)])
    sec_ad = "ic_val2023" if cfg.ic_dogrulama else "spatial_val2023"
    n_max = max(N_ADAY)
    G, S, Fk = {}, {}, {}
    for ad, (ti, st, gruplar) in bol.items():
        G[ad], S[ad] = tablo(P, ti, st)
        if a.kucuk:
            sec = np.sort(np.random.RandomState(0).choice(len(S[ad]), min(len(S[ad]), 4000), replace=False))
            G[ad], S[ad] = G[ad][sec], S[ad].iloc[sec].reset_index(drop=True)
        kup = ozellik_kubu(P, ti, gruplar, n_max, a.idw_p, a.idw_k)
        Fk[ad] = {n: satir_ozellik(P, ti, S[ad], kup, n, n_max) for n in N_ADAY}
        del kup
        print(f"  {ad}: {len(S[ad])} satir  [{time.time() - t0:.0f} s]", flush=True)

    olc0 = Olcek(G["train"], S["train"])
    ytr = olc0.y(S["train"])
    secim = []
    for n in (N_ADAY if n_sabit is None else ()):
        olc = Olcek(birlestir(G["train"], Fk["train"][n]), S["train"])
        rf = RandomForestRegressor(n_estimators=2 if a.kucuk else SECIM_AGAC, max_depth=RF_PARAMS["max_depth"],
                                   min_samples_leaf=RF_PARAMS["min_samples_leaf"], n_jobs=-1, random_state=a.seed)
        rf.fit(olc.x(birlestir(G["train"], Fk["train"][n])), ytr)
        yv = olc.geri(rf.predict(olc.x(birlestir(G[sec_ad], Fk[sec_ad][n]))), S[sec_ad])
        r = float(np.sqrt(np.mean((yv - S[sec_ad].y_true.to_numpy(float)) ** 2)))
        secim.append(dict(n=n, bolum=sec_ad, rmse=r))
        print(f"  secim n={n}: {sec_ad} RMSE {r:.3f}  [{time.time() - t0:.0f} s]", flush=True)
        del rf
    if n_sabit is None:
        secim = pd.DataFrame(secim).sort_values("rmse").reset_index(drop=True)
        n_en = int(secim.n[0])
    else:
        if n_sabit not in N_ADAY:
            raise ValueError(f"n={n_sabit} aday kumesinde degil: {N_ADAY}")
        secim = pd.DataFrame([dict(n=n_sabit, bolum=f"1. asamadan ({a.n_kaynak})", rmse=np.nan)])
        n_en = n_sabit
    print(f"Secilen n={n_en}", flush=True)
    os.makedirs(out, exist_ok=True)
    print("Ortam:", ortam_bilgisi(out, cfg.master_path), flush=True)
    secim.to_csv(os.path.join(out, "secim_n_2023.csv"), index=False)
    if a.sadece_secim:
        print(f"Bitti (yalniz secim): {out}  [{time.time() - t0:.0f} s]", flush=True)
        return

    olc = Olcek(birlestir(G["train"], Fk["train"][n_en]), S["train"])
    Xtr = olc.x(birlestir(G["train"], Fk["train"][n_en]))
    degerlendir = {ad: olc.x(birlestir(G[ad], Fk[ad][n_en]))
                   for ad in ("temporal", "spatial_test", "spatial_full", "val2023", "spatial_val2023", "ic_val2023") if ad in G}
    tah, _ = egit_rf(Xtr, ytr, degerlendir, a.seed, a.kucuk)

    st_no = P["stations"]["istasyon_no"].to_numpy()
    en_yakin_gorunur(P["stations"], st_no[hold], st_no[tr_tum]).assign(kat=-1 if a.kat is None else a.kat).to_csv(
        os.path.join(out, "istasyon_mesafe.csv"), index=False)
    ozet = []
    for ad, yz in tah.items():
        df = S[ad].copy()
        df["y_pred"] = olc.geri(np.asarray(yz, float), df)
        df.to_csv(os.path.join(out, f"pred_{ad}_{etiket}.csv.gz"), index=False)
        station_metrics(df).to_csv(os.path.join(out, f"station_{ad}_{etiket}.csv"), index=False)
        mt = metrics(df).assign(split=ad, model="rfsi", seed=a.seed, n_komsu=n_en)
        ozet.append(mt)
        print(f"  {ad:13s} RMSE {mt.iloc[0].rmse:.3f}  MBE {mt.iloc[0].mbe:+.3f}  n={int(mt.iloc[0].n)}", flush=True)
    pd.concat(ozet, ignore_index=True).to_csv(os.path.join(out, "metrics.csv"), index=False)
    print(f"Bitti: {out}  [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
