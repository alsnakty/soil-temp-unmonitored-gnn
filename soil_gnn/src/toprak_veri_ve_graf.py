import os
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from ortak_veri import Batch, haversine_km, station_pairs
from toprak_ayarlar import (DERINLIKLER, HEDEF_KOLON, IDX_T, N_FEAT, OZNITELIK, SOIL_KATMAN,
                         SOIL_VARS, UYDU, SoilConfig)

N_DEP = len(DERINLIKLER)


def load_master(cfg: SoilConfig, df: Optional[pd.DataFrame] = None):
    """master_dataset.csv -> X (T, n_st*N_DEP, F) float32, tarihler, istasyon tablosu.
    Slot sirasi: istasyon-major, derinlik DERINLIKLER sirasinda."""
    if df is None:
        if not os.path.isfile(cfg.master_path):
            raise FileNotFoundError(f"master_dataset bulunamadi: {cfg.master_path}")
        soil = [f"{v}_{k}" for v in SOIL_VARS for k in sorted(set(SOIL_KATMAN.values()))]
        cols = ["istasyon_no", "tarih", "lat", "lon", "rakim", *HEDEF_KOLON.values(), *UYDU, *soil]
        df = pd.read_csv(cfg.master_path, usecols=cols, parse_dates=["tarih"])
    if df.duplicated(["istasyon_no", "tarih"]).any():
        raise ValueError("Ayni (istasyon, tarih) icin birden fazla satir var.")

    stations = (df.groupby("istasyon_no")[["lat", "lon", "rakim"]].first()
                .reset_index().sort_values("istasyon_no").reset_index(drop=True))
    if stations[["lat", "lon", "rakim"]].isna().any().any():
        raise ValueError("Istasyon meta verisinde (lat/lon/rakim) NaN var.")
    eksik = set(cfg.holdout) - set(stations["istasyon_no"])
    if eksik:
        raise ValueError(f"Holdout istasyonlari veride yok: {sorted(eksik)}")

    dates = np.sort(df["tarih"].unique())
    t = np.searchsorted(dates, df["tarih"].to_numpy())
    s = pd.Index(stations["istasyon_no"]).get_indexer(df["istasyon_no"])

    X = np.full((len(dates), len(stations), N_DEP, N_FEAT), np.nan, dtype=np.float32)
    for d, dep in enumerate(DERINLIKLER):
        if dep > 0:
            X[t, s, d, IDX_T] = df[HEDEF_KOLON[dep]].to_numpy()
        if dep == 0 or cfg.uydu_derinlik == "tum":
            X[t, s, d, 1:1 + len(UYDU)] = df[UYDU].to_numpy()
        k = SOIL_KATMAN[dep]
        X[t, s, d, 1 + len(UYDU):1 + len(UYDU) + len(SOIL_VARS)] = \
            df[[f"{v}_{k}" for v in SOIL_VARS]].to_numpy()
        X[t, s, d, -1] = df["rakim"].to_numpy()
    for c in cfg.cikar:
        X[..., OZNITELIK.index(c)] = np.nan
    return X.reshape(len(dates), len(stations) * N_DEP, N_FEAT), pd.DatetimeIndex(dates), stations


class SoilGraph:

    def __init__(self, stations: pd.DataFrame, st_global: np.ndarray, cfg: SoilConfig):
        self.stations = stations.reset_index(drop=True)
        self.st_global = np.asarray(st_global)
        n = len(self.stations)
        self.n_st = n
        self.S = n * N_DEP
        self.slot_station = np.repeat(np.arange(n), N_DEP)
        self.slot_depth = np.tile(np.arange(N_DEP), n)
        self.slot_global = (self.st_global[:, None] * N_DEP + np.arange(N_DEP)).ravel()
        z = np.asarray(DERINLIKLER, float)[self.slot_depth]

        # dusey: ayni istasyonda komsu derinlikler, oznitelik |dz| (m)
        if cfg.dikey_tam:   # her derinlik ayni istasyonun tum derinliklerine
            di, dj = np.nonzero(~np.eye(N_DEP, dtype=bool))
            vs = (np.arange(n)[:, None] * N_DEP + di).ravel()
            vd = (np.arange(n)[:, None] * N_DEP + dj).ravel()
        else:
            a = (np.arange(n)[:, None] * N_DEP + np.arange(N_DEP - 1)).ravel()
            b = a + 1
            vs, vd = np.concatenate([a, b]), np.concatenate([b, a])
        v_attr = (np.abs(z[vs] - z[vd]) / 100.0)[:, None]

        # yatay: ayni derinlik; istasyon cifti sablonu her derinlige kopyalanir
        ps, pd_, pattr = station_pairs(self.stations, cfg)
        if cfg.yatay_rakim:   # mesaj kaynaktan hedefe: + = komsu daha yuksekte
            rk = self.stations["rakim"].to_numpy(float)
            pattr = np.column_stack([pattr, (rk[ps] - rk[pd_]) / 1000.0])
        self.n_pairs = len(ps)
        hs = np.concatenate([ps * N_DEP + d for d in range(N_DEP)])
        hd = np.concatenate([pd_ * N_DEP + d for d in range(N_DEP)])
        h_attr = np.tile(pattr, (N_DEP, 1))
        self.templates = {"V": (np.stack([vs, vd]), v_attr), "H": (np.stack([hs, hd]), h_attr)}

        lat = self.stations["lat"].to_numpy(float)[self.slot_station]
        lon = self.stations["lon"].to_numpy(float)[self.slot_station]
        self.slot_pos = np.column_stack([z / 100.0, lat / 90.0, lon / 180.0])

        # IDW_rakim icin: istasyonlar arasi uzaklik (km; kendisi inf), uzakliga gore sira, rakim (km)
        la, lo = self.stations["lat"].to_numpy(float), self.stations["lon"].to_numpy(float)
        self.st_dist = haversine_km(la[:, None], lo[:, None], la[None, :], lo[None, :])
        np.fill_diagonal(self.st_dist, np.inf)
        self.st_order = np.argsort(self.st_dist, axis=1, kind="stable")
        self.st_rakim_km = self.stations["rakim"].to_numpy(float) / 1000.0


def idw_rakim(T, kaynak, g: SoilGraph, p: float, k: int):
    n, z = g.n_st, g.st_rakim_km
    M = kaynak[:, None, :] & ~np.eye(n, dtype=bool)[None]            # (K, i, j): i'nin kaynaklari
    y = np.where(kaynak, T, 0.0).astype(np.float64)                    # (K, j)
    Mf = M.astype(np.float64)
    m = Mf.sum(-1)
    sz, szz = Mf @ z, Mf @ (z * z)
    sy, szy = np.einsum("kij,kj->ki", Mf, y), np.einsum("kij,kj->ki", Mf, y * z)
    with np.errstate(invalid="ignore", divide="ignore"):
        zb, yb = sz / m, sy / m
        sxx = szz - m * zb * zb
        b = np.where((m >= 3) & (sxx > 1e-12), (szy - m * zb * yb) / sxx, 0.0)   # <3 kaynak: duz IDW
    sira = np.broadcast_to(g.st_order, M.shape)
    sirali = np.take_along_axis(M, sira, axis=2)
    sec = np.take_along_axis(sirali & (np.cumsum(sirali, axis=2) <= k),
                             np.broadcast_to(np.argsort(g.st_order, axis=1), M.shape), axis=2)
    w = np.where(sec, g.st_dist[None] ** -p, 0.0)
    sw = w.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tah = (np.einsum("kij,kj->ki", w, y) + b * (z[None, :] * sw - w @ z)) / sw
    return np.where(sw > 0, tah, np.nan)


def derin_aktarim_ek(Traw, g: SoilGraph, cfg: SoilConfig, sd_T, gizli: Optional[np.ndarray] = None):
    from scipy.special import erfc
    n, T = g.n_st, Traw.shape[0]
    R = Traw.reshape(T, n, N_DEP)
    d20 = DERINLIKLER.index(20)
    t20 = R[:, :, d20].astype(np.float64)
    kaynak = np.isfinite(t20)
    if gizli is not None:
        kaynak &= ~np.asarray(gizli, bool)[None, :]
    idw20 = idw_rakim(t20, kaynak, g, cfg.idw_p, cfg.idw_k)
    d = t20 - idw20                                                   # (T, n), NaN = yok
    var = np.isfinite(d).astype(float)
    d0 = np.where(var > 0, d, 0.0)
    ek = np.zeros((T, n, N_DEP), dtype=np.float32)
    for dep, par, gamma in cfg.taban_param:
        j = DERINLIKLER.index(int(dep))
        if cfg.taban == "a3":
            w = np.ones(int(par))
        else:
            k = np.arange(365, dtype=float)
            dz = float(dep) - 20.0
            w = erfc(dz / (2 * np.sqrt(par * (k + 1)))) - np.where(k > 0, erfc(dz / (2 * np.sqrt(par * np.maximum(k, 1e-12)))), 0.0)
        pay = np.stack([np.convolve(d0[:, i], w)[:T] for i in range(n)], axis=1)
        payda = np.stack([np.convolve(var[:, i], w)[:T] for i in range(n)], axis=1)
        m = np.where(payda > 1e-12, pay / np.where(payda > 1e-12, payda, 1.0), 0.0)
        ek[:, :, j] = (gamma * m / sd_T[j]).astype(np.float32)
    return ek.reshape(T, n * N_DEP)


def norm_stats(X_train: np.ndarray, slot_depth: np.ndarray, mode: str):
    """mode='level': derinlik x oznitelik basina mu/sigma; 'global': oznitelik basina.
    Yalniz egitim istasyonlari + egitim yillarindan. NaN dislanir, sigma~0 -> 1."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if mode == "level":
            mu = np.zeros((N_DEP, N_FEAT)); sd = np.ones((N_DEP, N_FEAT))
            for d in range(N_DEP):
                v = X_train[:, slot_depth == d, :].reshape(-1, N_FEAT).astype(np.float64)
                mu[d], sd[d] = np.nanmean(v, axis=0), np.nanstd(v, axis=0)
        elif mode == "global":
            v = X_train.reshape(-1, N_FEAT).astype(np.float64)
            mu = np.tile(np.nanmean(v, axis=0), (N_DEP, 1))
            sd = np.tile(np.nanstd(v, axis=0), (N_DEP, 1))
        else:
            raise ValueError(mode)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    return mu, sd


def normalize(X, g: SoilGraph, mu, sd):
    return ((X - mu[g.slot_depth]) / sd[g.slot_depth]).astype(np.float32)


def denorm_T(Tn, g: SoilGraph, mu, sd):
    return Tn * sd[g.slot_depth, IDX_T] + mu[g.slot_depth, IDX_T]


class SoilWindows:

    def __init__(self, Xn, dates: pd.DatetimeIndex, g: SoilGraph, cfg: SoilConfig,
                 mode: str, seed: int, masked_stations: Optional[np.ndarray] = None,
                 derin_gizli_istasyon: Optional[np.ndarray] = None, taban_ek: Optional[np.ndarray] = None):
        # derin_gizli_istasyon: degerlendirmede bu istasyonlarin cfg.derin_gizli derinlikleri gizli (5-20 cm gorunur).
        # taban_ek: (T, S) normalize ek taban (derin_aktarim_ek); hibrit tabanina eklenir, tamamen gizli istasyonda 0.
        if len(Xn) == 0:
            raise ValueError(f"{mode} bolumu bos.")
        if mode != "train" and masked_stations is None:
            raise ValueError("Degerlendirmede masked_stations verilmeli.")
        self.Xn, self.g, self.cfg, self.mode = Xn, g, cfg, mode
        self.T, self.S, self.W = Xn.shape[0], g.S, cfg.window
        self.starts = np.arange(max(1, self.T - self.W + 1))
        self.rng = np.random.RandomState(seed)
        self.fixed = None
        if masked_stations is not None:
            self.fixed = np.zeros(g.n_st, bool)
            self.fixed[np.asarray(masked_stations, int)] = True
        self.derin_slot = np.isin(np.asarray(DERINLIKLER)[g.slot_depth], cfg.derin_gizli)
        self.fixed_derin = None
        if derin_gizli_istasyon is not None:
            self.fixed_derin = np.zeros(g.n_st, bool)
            self.fixed_derin[np.asarray(derin_gizli_istasyon, int)] = True
        self.taban_ek = taban_ek
        days = ((dates - dates[0]) / pd.Timedelta(hours=1)).to_numpy(float)
        self.hours = days
        doy = dates.dayofyear.to_numpy(float)
        self.doy = np.column_stack([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)])

    def __len__(self):
        return len(self.starts)

    def station_mask(self):
        if self.fixed is not None:
            return self.fixed
        if getattr(self.cfg, "blok_maske", False):
            n = self.g.n_st
            c = self.rng.randint(n)
            k = int(np.ceil(self.cfg.blok_orani * n))
            m = np.zeros(n, bool)
            m[c] = True
            m[np.argsort(self.g.st_dist[c], kind="stable")[:k - 1]] = True     # st_dist kosegeni inf: merkez sirada yok
            return m
        m = self.rng.rand(self.g.n_st) < self.cfg.station_mask_ratio
        tam_gizleme_yok = self.cfg.derin_maske_orani > 0 and self.cfg.station_mask_ratio == 0   # derin gorev: istasyon gizlenmez
        if not m.any() and not tam_gizleme_yok:
            m[self.rng.randint(self.g.n_st)] = True
        return m

    def derin_maske(self, st_m):
        """Derinlik maskesi alan istasyonlar (tamamen gizlilerin disindan); kapaliysa None (RNG cekilmez)."""
        if self.fixed_derin is not None:
            return self.fixed_derin & ~st_m
        if self.mode == "train" and self.cfg.derin_maske_orani > 0:
            return ~st_m & (self.rng.rand(self.g.n_st) < self.cfg.derin_maske_orani)
        return None

    def window(self, i):
        W, S = self.W, self.S
        tt = np.arange(self.starts[i], self.starts[i] + W)
        valid = tt < self.T
        full = np.full((W, S, N_FEAT), np.nan, dtype=np.float32)
        full[valid] = self.Xn[tt[valid]]
        st_m = self.station_mask()
        node_m = st_m[self.g.slot_station]
        dm = self.derin_maske(st_m)
        if dm is not None:
            node_m = node_m | (dm[self.g.slot_station] & self.derin_slot)
        tgt = full[..., IDX_T:IDX_T + 1].copy()
        art = node_m[None, :, None] & np.isfinite(tgt)
        x = full.copy()
        x[:, node_m, IDX_T] = np.nan
        pos = np.zeros((W, S, 5), dtype=np.float32)
        pos[:, :, :3] = self.g.slot_pos[None]
        if not self.cfg.konum_pe:                  # ablasyon: enlem/boylam yok (derinlik ve yilin gunu kalir)
            pos[:, :, 1:3] = 0.0
        pos[valid, :, 3:] = self.doy[tt[valid]][:, None, :]
        dh = np.full(W - 1, np.nan)
        both = valid[:-1] & valid[1:]
        dh[both] = (self.hours[tt[1:][both]] - self.hours[tt[:-1][both]]) / self.cfg.step_hours
        out = dict(x=x, target=tgt, mask=art, pos=pos, pad=~valid,
                   t=np.where(valid, tt, -1), dh=dh, st_mask=st_m)
        if self.cfg.idw_artik:
            out["base"] = self.idw_taban(x)
            if self.taban_ek is not None:
                ek = np.zeros((W, S), dtype=np.float32)
                ek[valid] = self.taban_ek[tt[valid]]
                ek[:, st_m[self.g.slot_station]] = 0.0        # kendi 20 cm'si gizli istasyona aktarim yok
                out["base"] = out["base"] + ek[..., None]
        return out

    def idw_taban(self, x):
        g, W = self.g, self.W
        Td = x[..., IDX_T].reshape(W, g.n_st, N_DEP)[:, :, 1:]          # (W, n, derinlik>0)
        K = Td.transpose(0, 2, 1).reshape(-1, g.n_st)                     # (W * 5, n)
        tah = idw_rakim(K, np.isfinite(K), g, self.cfg.idw_p, self.cfg.idw_k)
        base = np.full((W, g.n_st, N_DEP), np.nan, dtype=np.float32)
        base[:, :, 1:] = tah.reshape(W, N_DEP - 1, g.n_st).transpose(0, 2, 1)
        return base.reshape(W, g.S, 1)

    def batches(self, shuffle: bool):
        order = self.rng.permutation(len(self.starts)) if shuffle else np.arange(len(self.starts))
        bs = self.cfg.batch_size
        for k in range(0, len(order), bs):
            yield collate([self.window(i) for i in order[k:k + bs]], self.W, self.S, self.g.templates)


def collate(ws: List[dict], W: int, S: int, templates) -> Batch:
    B = len(ws)
    N = B * W * S
    cat = lambda key: np.stack([w[key] for w in ws])
    ei, ea = {}, {}
    off = (np.arange(B * W) * S)[:, None, None]
    for k in ("V", "H"):
        e, a = templates[k]
        ei[k] = (e[None] + off).transpose(1, 0, 2).reshape(2, -1)
        ea[k] = np.tile(a, (B * W, 1))
    src, dst, att = [], [], []
    s_ar = np.arange(S)
    for b, w_ in enumerate(ws):
        for w in range(W - 1):
            if not np.isfinite(w_["dh"][w]):
                continue
            a_ = b * W * S + w * S + s_ar
            src += [a_, a_ + S]; dst += [a_ + S, a_]
            att.append(np.full(2 * S, w_["dh"][w]))
    if src:
        ei["T"] = np.stack([np.concatenate(src), np.concatenate(dst)])
        ea["T"] = np.concatenate(att)[:, None]
    else:
        ei["T"] = np.zeros((2, 0), dtype=np.int64); ea["T"] = np.zeros((0, 1))
    f = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)
    base = f(cat("base").reshape(N, 1)) if "base" in ws[0] else None
    return Batch(
        x=f(cat("x").reshape(N, -1)), target=f(cat("target").reshape(N, 1)),
        art_mask=torch.as_tensor(cat("mask").reshape(N, 1)), pos=f(cat("pos").reshape(N, 5)),
        edge_index={k: torch.as_tensor(v, dtype=torch.long) for k, v in ei.items()},
        edge_attr={k: f(v) for k, v in ea.items()},
        time_pad=torch.as_tensor(cat("pad")),
        node_t=torch.as_tensor(np.broadcast_to(cat("t")[:, :, None], (B, W, S)).reshape(N).copy()),
        node_slot=torch.as_tensor(np.broadcast_to(np.arange(S), (B, W, S)).reshape(N).copy()),
        B=B, W=W, S=S, base=base)


class Splits:

    def __init__(self, train, val):
        self.splits = {"train": train, "val": val}


def synthetic_master(n_st=8, seed=0, holdout=(3, 7)):
    rng = np.random.RandomState(seed)
    ids = np.arange(17000, 17000 + n_st)
    lat, lon = rng.uniform(36, 42, n_st), rng.uniform(26, 44, n_st)
    rakim = rng.uniform(0, 2000, n_st)
    dates = (pd.date_range("2021-06-01", periods=21, freq="D").delete(10)
             .append(pd.date_range("2022-06-01", periods=8, freq="D"))
             .append(pd.date_range("2023-06-01", periods=8, freq="D")))
    rows = []
    for i in range(n_st):
        for j, t in enumerate(dates):
            yuzey = 10 + 8 * np.sin(2 * np.pi * t.dayofyear / 365) - rakim[i] / 300 + rng.randn()
            r = dict(istasyon_no=ids[i], tarih=t, lat=lat[i], lon=lon[i], rakim=rakim[i],
                     LST_Day_C=yuzey + 5 + rng.randn(), LST_Night_C=yuzey - 5 + rng.randn(),
                     NDVI=rng.uniform(0.1, 0.6), soil_moisture=rng.uniform(0.1, 0.4))
            for dep, col in HEDEF_KOLON.items():
                r[col] = yuzey * np.exp(-dep / 60) + 12 * (1 - np.exp(-dep / 60)) + 0.3 * rng.randn()
            for v in SOIL_VARS:
                for k in set(SOIL_KATMAN.values()):
                    r[f"{v}_{k}"] = rng.uniform(1, 50)
            rows.append(r)
    df = pd.DataFrame(rows)
    df.loc[rng.rand(len(df)) < 0.5, "LST_Day_C"] = np.nan
    df.loc[rng.rand(len(df)) < 0.05, "t100cmsicaklik"] = np.nan
    return df, tuple(int(ids[h]) for h in holdout)
