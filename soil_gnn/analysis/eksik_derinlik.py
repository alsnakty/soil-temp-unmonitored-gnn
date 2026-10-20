import os
import sys

import numpy as np
import pandas as pd
import torch

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 220)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))

from ortak_ayarlar import variant_opts
from gnn_model import build_model
from toprak_ayarlar import DERINLIKLER, IDX_T, MASKE_VARSAYILAN, N_FEAT, OZNITELIK, SoilConfig, edge_dims
from toprak_veri_ve_graf import SoilWindows, denorm_T, norm_stats, normalize
from toprak_degerlendirme import predict_masked
from toprak_gnn_calistir import hazirla
from toprak_katlar import kat_istasyonlari
import idw_karsilastirma_modeli as IDWM

SON = os.path.join(ANA_DIZIN, "results")
OUT = os.path.join(SON, "folds", "analysis_missing_depth")
IDW_P = (1, 1, 2, 2, 1)
IDW_K = (6, 6, 8, 8, 8)
LAR = (3, 7, 15, 30, 60, 120)
HEDEF = (50, 100)
ANAHTAR = ["istasyon_no", "derinlik", "tarih"]


def veri(P):
    X, dates, st = P["X"], P["dates"], P["stations"]
    T = X[..., IDX_T].reshape(len(dates), len(st), len(DERINLIKLER)).astype(float)
    return (T, dates, st.istasyon_no.to_numpy(), st.lat.to_numpy(float), st.lon.to_numpy(float),
            st.rakim.to_numpy(float) / 1000.0)


def artik_tablosu(r20, rd):
    out = []
    for s, g in rd.groupby("istasyon_no"):
        a = r20[r20.istasyon_no == s].set_index("tarih").r.sort_index()
        gun = pd.date_range(min(a.index.min() if len(a) else g.tarih.min(), g.tarih.min()),
                            max(a.index.max() if len(a) else g.tarih.max(), g.tarih.max()))
        a = a.reindex(gun)
        g = g.copy()
        yil = a.groupby(a.index.year).mean()
        g["m_yil"] = yil.reindex(g.tarih.dt.year).to_numpy()
        for L in LAR:
            g[f"m_{L}"] = a.rolling(L, min_periods=1).mean().reindex(g.tarih).to_numpy()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def toplamlar(t, kol):
    """Istasyon basina sum m*r, sum m^2, sum r^2 (m NaN -> 0)."""
    m = t[kol].fillna(0.0)
    return pd.DataFrame({"smr": (m * t.r).groupby(t.istasyon_no).sum(), "smm": (m * m).groupby(t.istasyon_no).sum(),
                         "srr": (t.r ** 2).groupby(t.istasyon_no).sum(), "n": t.r.groupby(t.istasyon_no).size()})


def loo_tahmin(t):
    """A1, A2, A3 tahminleri (istasyon birakmali gamma ve L). t: tek derinlik, butun istasyonlar."""
    t = t.copy()
    t["A1"] = t.y_pred
    S = toplamlar(t, "m_yil")
    tot = S.sum()
    g2 = ((tot.smr - S.smr) / (tot.smm - S.smm)).rename("g")
    t["gamma_A2"] = g2.reindex(t.istasyon_no).to_numpy()
    t["A2"] = t.y_pred + t.gamma_A2 * t.m_yil.fillna(0.0)
    # A3: her L icin gamma ve diger istasyonlardaki RSS; L = en kucuk RSS
    rss, gam = {}, {}
    for L in LAR:
        S = toplamlar(t, f"m_{L}")
        tot = S.sum()
        g = (tot.smr - S.smr) / (tot.smm - S.smm)
        gam[L] = g
        rss[L] = (tot.srr - S.srr) - 2 * g * (tot.smr - S.smr) + g ** 2 * (tot.smm - S.smm)
    R = pd.DataFrame(rss)
    Lsec = R.idxmin(axis=1)
    t["L_A3"] = Lsec.reindex(t.istasyon_no).to_numpy()
    t["gamma_A3"] = [gam[L][s] for s, L in zip(t.istasyon_no, t.L_A3)]
    mL = np.choose(np.searchsorted(LAR, t.L_A3.to_numpy()), [t[f"m_{L}"].fillna(0.0).to_numpy() for L in LAR])
    t["A3"] = t.y_pred + t.gamma_A3 * mL
    return t


def tam_secim(t):
    """Butun istasyonlarla gamma ve L (A4: ic egitim istasyonlarinda secim, ic hedeflere uygulanir)."""
    g2 = toplamlar(t, "m_yil").sum()
    sec = {"gamma_A2": g2.smr / g2.smm}
    best = None
    for L in LAR:
        s = toplamlar(t, f"m_{L}").sum()
        g = s.smr / s.smm
        r = s.srr - 2 * g * s.smr + g ** 2 * s.smm
        if best is None or r < best[0]:
            best = (r, L, g)
    sec.update(L_A3=best[1], gamma_A3=best[2])
    return sec


def rmse_tablo(t, kollar, etiket):
    satir = []
    for d, g in t.groupby("derinlik"):
        for k in kollar:
            e = g[k] - g.y_true
            ist = (e ** 2).groupby([g.kat, g.istasyon_no]).mean() ** 0.5
            ist = ist.groupby(level=1).mean()                                  # istasyon: bulundugu katlarin ortalamasi
            satir.append(dict(kume=etiket, derinlik=d, yontem=k, n_satir=len(g), n_istasyon=len(ist),
                              havuz_rmse=float(np.sqrt((e ** 2).mean())), istasyon_rmse_ort=float(ist.mean())))
    return pd.DataFrame(satir)


def a1_a3(df_m):
    parca, secim = [], []
    for k in range(5):
        c = SoilConfig(); c.holdout = kat_istasyonlari(k)
        P = hazirla(c, df_m)
        T, dates, st_no, lat, lon, rak = veri(P)
        ti = np.nonzero(dates.year <= 2023)[0]
        tr = P["g_tr"].st_global
        r = IDWM.satirlar(T[ti], dates[ti], st_no, lat, lon, tr, [np.array([i]) for i in tr],
                          float(IDW_P[k]), IDW_K[k], rak).dropna(subset=["y_pred"])
        r["r"] = r.y_true - r.y_pred
        r20 = r[r.derinlik == 20][["istasyon_no", "tarih", "r"]]
        for d in HEDEF:
            t = artik_tablosu(r20, r[r.derinlik == d].copy())
            t = loo_tahmin(t).assign(kat=k)
            parca.append(t)
            secim.append(t.groupby("istasyon_no")[["gamma_A2", "L_A3", "gamma_A3"]].first().assign(kat=k, derinlik=d))
        print(f"kat {k}: {len(tr)} egitim istasyonu, satir {len(r)}", flush=True)
    t = pd.concat(parca, ignore_index=True)
    return t, pd.concat(secim)


class DerinMaske(SoilWindows):
    """Degerlendirme penceresi: dugum duzeyi sabit maske (gizli_slot: S boyutlu bool)."""

    def __init__(self, Xn, dates, g, cfg, seed, gizli_slot):
        super().__init__(Xn, dates, g, cfg, "eval", seed, masked_stations=np.array([], dtype=int))
        self.gizli = np.asarray(gizli_slot, bool)

    def window(self, i):
        W, S = self.W, self.S
        tt = np.arange(self.starts[i], self.starts[i] + W)
        valid = tt < self.T
        full = np.full((W, S, N_FEAT), np.nan, dtype=np.float32)
        full[valid] = self.Xn[tt[valid]]
        node_m = self.gizli
        tgt = full[..., IDX_T:IDX_T + 1].copy()
        art = node_m[None, :, None] & np.isfinite(tgt)
        x = full.copy()
        x[:, node_m, IDX_T] = np.nan
        pos = np.zeros((W, S, 5), dtype=np.float32)
        pos[:, :, :3] = self.g.slot_pos[None]
        if not self.cfg.konum_pe:
            pos[:, :, 1:3] = 0.0
        pos[valid, :, 3:] = self.doy[tt[valid]][:, None, :]
        dh = np.full(W - 1, np.nan)
        both = valid[:-1] & valid[1:]
        dh[both] = (self.hours[tt[1:][both]] - self.hours[tt[:-1][both]]) / self.cfg.step_hours
        out = dict(x=x, target=tgt, mask=art, pos=pos, pad=~valid,
                   t=np.where(valid, tt, -1), dh=dh, st_mask=np.zeros(self.g.n_st, bool))
        if self.cfg.idw_artik:
            out["base"] = self.idw_taban(x)
        return out


def a4(df_m):
    parca, secler = [], []
    for k in range(5):
        tah = {}
        for ad in ("hibrit", "gnn"):
            c = SoilConfig()
            c.uydu_derinlik, c.n_layers, c.maske = "tum", 2, MASKE_VARSAYILAN
            c.holdout, c.ic_dogrulama = kat_istasyonlari(k), True
            if ad == "hibrit":
                c.idw_artik, c.idw_p, c.idw_k = True, float(IDW_P[k]), IDW_K[k]
            c.__post_init__()
            P = hazirla(c, df_m)
            X, idx, g_tr, g_val = P["X"], P["idx"], P["g_tr"], P["g_val"]
            sub = lambda ti, g: X[ti][:, g.slot_global]
            mu, sd = norm_stats(sub(idx["train"], g_tr), g_tr.slot_depth, "level")
            raw = sub(idx["val"], g_val)
            dep = np.asarray(DERINLIKLER)[g_val.slot_depth]
            gizli = np.isin(g_val.slot_station, P["val_set"]) & np.isin(dep, HEDEF)
            wins = DerinMaske(normalize(raw, g_val, mu, sd), P["dates"][idx["val"]], g_val, c, 0, gizli)
            m = build_model(c, variant_opts("gnn"), in_dim=N_FEAT, target_cols=[IDX_T], edge_dims=edge_dims(c),
                            maske_cols=[OZNITELIK.index(x) for x in c.maske], idw_artik=c.idw_artik)
            m.load_state_dict(torch.load(os.path.join(SON, "fold_selection", f"{ad}_fold{k}", "model_gnn_s0.pt"),
                                         map_location="cpu", weights_only=True))
            yhat = denorm_T(predict_masked(m, wins, c), g_val, mu, sd)
            rT = raw[..., IDX_T]
            tt, ss = np.nonzero(np.isfinite(rT) & gizli[None, :])
            tah[ad] = pd.DataFrame({"istasyon_no": g_val.stations.istasyon_no.to_numpy()[g_val.slot_station[ss]],
                                    "derinlik": dep[ss], "tarih": P["dates"][idx["val"]][tt],
                                    "y_true": rT[tt, ss].astype(float), ad: yhat[tt, ss].astype(float)})
        T, dates, st_no, lat, lon, rak = veri(P)
        ic = P["ic_global"]
        hold = P["g_all"].st_global[P["hold_local"]]
        ictr = P["g_tr"].st_global
        tum = np.setdiff1d(np.arange(len(st_no)), hold)
        vi = P["idx"]["val"]
        p, kk = float(IDW_P[k]), IDW_K[k]
        # derin: kaynak ic egitim; 20 cm: kaynak hedef disindaki butun gorunurler
        rd = IDWM.satirlar(T[vi], dates[vi], st_no, lat, lon, np.concatenate([ictr, ic]), [ic], p, kk, rak)
        rd = rd[rd.derinlik.isin(HEDEF)]
        r20 = IDWM.satirlar(T[vi], dates[vi], st_no, lat, lon, tum, [np.array([i]) for i in ic], p, kk, rak)
        r20 = r20[r20.derinlik == 20].assign(r=lambda x: x.y_true - x.y_pred)[["istasyon_no", "tarih", "r"]]
        ti = np.nonzero(dates.year <= 2023)[0]
        rr = IDWM.satirlar(T[ti], dates[ti], st_no, lat, lon, ictr, [np.array([i]) for i in ictr], p, kk, rak).dropna(subset=["y_pred"])
        rr["r"] = rr.y_true - rr.y_pred
        for d in HEDEF:
            sec = tam_secim(artik_tablosu(rr[rr.derinlik == 20][["istasyon_no", "tarih", "r"]], rr[rr.derinlik == d].copy()))
            secler.append(dict(kat=k, derinlik=d, **sec))
            t = artik_tablosu(r20, rd[rd.derinlik == d].assign(r=lambda x: x.y_true - x.y_pred).copy())
            t["A1"] = t.y_pred
            t["A2"] = t.y_pred + sec["gamma_A2"] * t.m_yil.fillna(0.0)
            t["A3"] = t.y_pred + sec["gamma_A3"] * t[f"m_{sec['L_A3']}"].fillna(0.0)
            u = t[ANAHTAR + ["y_true", "A1", "A2", "A3"]]
            for ad in ("hibrit", "gnn"):
                v = tah[ad][tah[ad].derinlik == d]
                u = u.merge(v[ANAHTAR + ["y_true", ad]], on=ANAHTAR, suffixes=("", "_" + ad))
                if not np.allclose(u.y_true, u["y_true_" + ad]):
                    raise RuntimeError("y_true farkli")
                u = u.drop(columns="y_true_" + ad)
            u = u.dropna(subset=["A1", "hibrit", "gnn"])
            parca.append(u.assign(kat=k))
        print(f"kat {k} A4: ic {len(ic)}, satir {sum(len(x) for x in parca if x.kat.iloc[0] == k)}", flush=True)
    return pd.concat(parca, ignore_index=True), pd.DataFrame(secler)


def main():
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    os.makedirs(OUT, exist_ok=True)
    df_m = pd.read_csv(SoilConfig().master_path, parse_dates=["tarih"])
    t, sec = a1_a3(df_m)
    tab = rmse_tablo(t, ["A1", "A2", "A3"], "egitim istasyonlari 2015-2023 (istasyon birakmali)")
    print("\nA1-A3:")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nSecilen L (A3), hedef sayisi:")
    print(sec.groupby(["derinlik", "L_A3"]).size().unstack(fill_value=0).to_string())
    print("gamma ortalamasi:", sec.groupby("derinlik")[["gamma_A2", "gamma_A3"]].mean().round(3).to_dict())
    tab.to_csv(os.path.join(OUT, "a1_a3.csv"), index=False)
    sec.to_csv(os.path.join(OUT, "a1_a3_secim.csv"))
    u, s4 = a4(df_m)
    tab4 = rmse_tablo(u, ["hibrit", "gnn", "A1", "A2", "A3"], "ic dogrulama x 2023")
    print("\nA4 (ayni satirlar):")
    print(tab4.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(s4.to_string(index=False))
    tab4.to_csv(os.path.join(OUT, "a4.csv"), index=False)
    s4.to_csv(os.path.join(OUT, "a4_secim.csv"), index=False)


if __name__ == "__main__":
    main()
