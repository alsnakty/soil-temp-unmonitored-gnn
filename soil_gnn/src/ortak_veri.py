import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ortak_ayarlar import Config, N_VAR

HAM_SUTUNLAR = ["T", "RH", "WS", "WD", "Z"]
EARTH_R_KM = 6371.0088


def load_stations(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Istasyon listesi bulunamadi: {path}")
    df = pd.read_csv(path)
    eksik = {"station_id", "lat", "lon", "elevation"} - set(df.columns)
    if eksik:
        raise ValueError(f"Istasyon listesinde eksik sutun: {sorted(eksik)}")
    if df["station_id"].duplicated().any():
        raise ValueError("Istasyon listesinde tekrarlanan station_id var.")
    return df.reset_index(drop=True)


def load_observations(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Gozlem dosyasi bulunamadi: {path}")
    df = pd.read_csv(path, parse_dates=["datetime"])
    eksik = {"station_id", "datetime", "pressure", *HAM_SUTUNLAR} - set(df.columns)
    if eksik:
        raise ValueError(f"Gozlem dosyasinda eksik sutun: {sorted(eksik)}")
    return df


def encode_features(df: pd.DataFrame) -> np.ndarray:
    wd = df["WD"].to_numpy(dtype=float)
    return np.column_stack([
        df["T"].to_numpy(dtype=float),
        df["RH"].to_numpy(dtype=float),
        df["WS"].to_numpy(dtype=float),
        np.sin(np.pi * wd / 180.0),
        np.cos(np.pi * wd / 180.0),
        df["Z"].to_numpy(dtype=float),
    ])


class NodeIndexer:

    def __init__(self, stations: pd.DataFrame, cfg: Config):
        self.stations = stations
        self.levels = np.asarray(cfg.pressure_levels, dtype=float)
        n_st, n_lev = len(stations), len(self.levels)
        elev = stations["elevation"].to_numpy(dtype=float)

        valid = np.ones((n_st, n_lev), dtype=bool)
        for p, esik in cfg.level_elev_thresholds.items():
            hit = np.where(self.levels == p)[0]
            if len(hit):
                valid[elev > esik, hit[0]] = False
        self.valid = valid

        st_idx, lev_idx = np.nonzero(valid)
        self.slot_station = st_idx
        self.slot_level = lev_idx
        self.slot_pressure = self.levels[lev_idx]
        self.S = len(st_idx)
        self.lookup = np.full((n_st, n_lev), -1, dtype=np.int64)
        self.lookup[st_idx, lev_idx] = np.arange(self.S)

        ids = stations["station_id"].to_numpy()
        self.station_valid_levels: Dict = {
            ids[i]: [float(p) for p in self.levels[valid[i]]] for i in range(n_st)}
        self._st_index = pd.Index(ids)
        self._lev_index = pd.Index(self.levels)

    def build_tensor(self, obs: pd.DataFrame):
        times = np.sort(obs["datetime"].unique())
        t = np.searchsorted(times, obs["datetime"].to_numpy())
        st = self._st_index.get_indexer(obs["station_id"])
        lev = self._lev_index.get_indexer(obs["pressure"].astype(float))

        bilgi = {
            "bilinmeyen_istasyon_satiri": int((st < 0).sum()),
            "standart_disi_seviye_satiri": int((lev < 0).sum()),
        }
        ok = (st >= 0) & (lev >= 0)
        slot = np.full(len(obs), -1, dtype=np.int64)
        slot[ok] = self.lookup[st[ok], lev[ok]]
        bilgi["yer_alti_seviye_satiri"] = int((ok & (slot < 0)).sum())
        keep = slot >= 0

        key = t[keep] * self.S + slot[keep]
        if len(np.unique(key)) != len(key):
            raise ValueError("Ayni (istasyon, seviye, zaman) icin birden fazla satir var.")

        X = np.full((len(times), self.S, N_VAR), np.nan, dtype=np.float64)
        X[t[keep], slot[keep]] = encode_features(obs.loc[keep])
        return X, times, bilgi


def chrono_split(times: np.ndarray, cfg: Config):
    times = np.sort(np.asarray(times))
    n = len(times)
    r0, r1, _ = cfg.split_ratios
    b1, b2 = int(round(n * r0)), int(round(n * (r0 + r1)))
    if cfg.year_aligned_split:
        def yil_basi(i):
            if i >= n:
                return n
            yil = pd.Timestamp(times[i]).year
            return int(np.searchsorted(times, np.datetime64(f"{yil}-01-01")))
        b1, b2 = yil_basi(b1), yil_basi(b2)
    parcalar = {"train": slice(0, b1), "val": slice(b1, b2), "test": slice(b2, n)}
    for ad, s in parcalar.items():
        if s.stop - s.start <= 0:
            raise ValueError(f"'{ad}' bolumu bos (n_zaman={n}, oranlar={cfg.split_ratios}, "
                             f"year_aligned={cfg.year_aligned_split}).")
    return parcalar


class Normalizer:

    def __init__(self, mode: str = "level"):
        if mode not in ("level", "global"):
            raise ValueError(mode)
        self.mode = mode

    def fit(self, X_train: np.ndarray, slot_level: np.ndarray, n_levels: int):
        nv = X_train.shape[-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if self.mode == "level":
                mu = np.zeros((n_levels, nv))
                sd = np.ones((n_levels, nv))
                for l in range(n_levels):
                    v = X_train[:, slot_level == l, :].reshape(-1, nv)
                    if len(v):
                        mu[l], sd[l] = np.nanmean(v, axis=0), np.nanstd(v, axis=0)
                self.mu_slot, self.sd_slot = mu[slot_level], sd[slot_level]
            else:
                v = X_train.reshape(-1, nv)
                mu, sd = np.nanmean(v, axis=0), np.nanstd(v, axis=0)
                self.mu_slot = np.broadcast_to(mu, (len(slot_level), nv)).copy()
                self.sd_slot = np.broadcast_to(sd, (len(slot_level), nv)).copy()
        self.mu_slot = np.where(np.isfinite(self.mu_slot), self.mu_slot, 0.0)
        self.sd_slot = np.where(np.isfinite(self.sd_slot) & (self.sd_slot > 1e-6),
                                self.sd_slot, 1.0)
        return self

    def transform(self, X):
        return (X - self.mu_slot) / self.sd_slot

    def inverse(self, X):
        return X * self.sd_slot + self.mu_slot


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """i'den j'ye baslangic yon acisi, [0, 360)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def station_pairs(stations: pd.DataFrame, cfg: Config):
    """Yonlu istasyon ciftleri: (mesafe <= R) U kNN, simetrik; ozellik [d/1000, bearing]."""
    lat = stations["lat"].to_numpy(dtype=float)
    lon = stations["lon"].to_numpy(dtype=float)
    n = len(lat)
    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(D, np.inf)
    A = D <= cfg.horiz_radius_km
    k = min(cfg.horiz_knn, n - 1)
    if k > 0:
        nn = np.argsort(D, axis=1)[:, :k]
        A[np.repeat(np.arange(n), k), nn.ravel()] = True
    A = A | A.T
    np.fill_diagonal(A, False)
    src, dst = np.nonzero(A)
    brg = bearing_deg(lat[src], lon[src], lat[dst], lon[dst])
    if cfg.bearing_scale == "deg360":
        brg = brg / 360.0
    elif cfg.bearing_scale == "rad":
        brg = np.radians(brg)
    else:  # rad_signed: arctan2 ciktisi, [-pi, pi]
        brg = np.radians(np.where(brg > 180.0, brg - 360.0, brg))
    attr = np.column_stack([D[src, dst] / 1000.0, brg])
    return src, dst, attr


def step_templates(idx: NodeIndexer, stations: pd.DataFrame, cfg: Config):
    same = idx.slot_station[:-1] == idx.slot_station[1:]
    a = np.nonzero(same)[0]
    b = a + 1
    lnp = np.log(idx.slot_pressure)
    v_src = np.concatenate([a, b])
    v_dst = np.concatenate([b, a])
    v_attr = np.abs(lnp[v_src] - lnp[v_dst])[:, None]

    ps, pd_, pattr = station_pairs(stations, cfg)
    h_src, h_dst, h_attr = [], [], []
    for l in range(len(idx.levels)):
        sa, sb = idx.lookup[ps, l], idx.lookup[pd_, l]
        ok = (sa >= 0) & (sb >= 0)
        h_src.append(sa[ok]); h_dst.append(sb[ok]); h_attr.append(pattr[ok])
    return {
        "V": (np.stack([v_src, v_dst]), v_attr),
        "H": (np.stack([np.concatenate(h_src), np.concatenate(h_dst)]), np.concatenate(h_attr)),
    }


@dataclass
class Batch:
    x: torch.Tensor
    target: torch.Tensor
    art_mask: torch.Tensor
    pos: torch.Tensor
    edge_index: Dict[str, torch.Tensor]
    edge_attr: Dict[str, torch.Tensor]
    time_pad: torch.Tensor
    node_t: torch.Tensor
    node_slot: torch.Tensor
    B: int
    W: int
    S: int
    base: Optional[torch.Tensor] = None

    def to(self, device):
        for f in ("x", "target", "art_mask", "pos", "time_pad", "node_t", "node_slot"):
            setattr(self, f, getattr(self, f).to(device))
        if self.base is not None:
            self.base = self.base.to(device)
        self.edge_index = {k: v.to(device) for k, v in self.edge_index.items()}
        self.edge_attr = {k: v.to(device) for k, v in self.edge_attr.items()}
        return self


class WindowData:

    def __init__(self, Xn: np.ndarray, times: np.ndarray, idx: NodeIndexer,
                 stations: pd.DataFrame, templates, cfg: Config, mode: str, seed: int):
        if len(Xn) == 0:
            raise ValueError(f"{mode} bolumu bos.")
        self.Xn, self.times, self.cfg, self.mode = Xn, times, cfg, mode
        self.T, self.S = Xn.shape[0], Xn.shape[1]
        self.W = cfg.window
        self.starts = np.arange(max(1, self.T - self.W + 1))
        self.rng = np.random.RandomState(seed)
        self.templates = templates
        self.observed = np.isfinite(Xn)
        self.global_mask = None
        if mode != "train":
            self.global_mask = global_random_mask(self.observed, cfg.mask_ratio,
                                                  np.random.RandomState(seed))

        hrs = (times - times[0]) / np.timedelta64(1, "h")
        self.hours = hrs.astype(float)
        hod = pd.DatetimeIndex(times).hour.to_numpy() + pd.DatetimeIndex(times).minute.to_numpy() / 60
        self.hod = np.column_stack([np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)])

        lat = stations["lat"].to_numpy(dtype=float)[idx.slot_station]
        lon = stations["lon"].to_numpy(dtype=float)[idx.slot_station]
        self.slot_pos = np.column_stack([idx.slot_pressure / 1000.0, lat / 90.0, lon / 180.0])

    def __len__(self):
        return len(self.starts)

    def _mask_for(self, t0, tt, valid):
        W, S = self.W, self.S
        m = np.zeros((W, S, N_VAR), dtype=bool)
        if self.mode == "train":
            obs = self.observed[tt[valid]]
            m[valid] = (self.rng.rand(*obs.shape) < self.cfg.mask_ratio) & obs
        else:
            m[valid] = self.global_mask[tt[valid]]
        return m

    def window(self, i):
        t0 = int(self.starts[i])
        tt = np.arange(t0, t0 + self.W)
        valid = tt < self.T
        tgt = np.full((self.W, self.S, N_VAR), np.nan)
        tgt[valid] = self.Xn[tt[valid]]
        m = self._mask_for(t0, tt, valid)
        x = np.where(m, np.nan, tgt)
        pos = np.zeros((self.W, self.S, 5))
        pos[:, :, :3] = self.slot_pos[None]
        pos[valid, :, 3:] = self.hod[tt[valid]][:, None, :]
        dh = np.full(self.W - 1, np.nan)
        both = valid[:-1] & valid[1:]
        dh[both] = (self.hours[tt[1:][both]] - self.hours[tt[:-1][both]]) / self.cfg.step_hours
        return dict(x=x, target=tgt, mask=m, pos=pos, pad=~valid,
                    t=np.where(valid, tt, -1), dh=dh)

    def batches(self, shuffle: bool):
        order = np.arange(len(self.starts))
        if shuffle:
            order = self.rng.permutation(order)
        bs = self.cfg.batch_size
        for k in range(0, len(order), bs):
            yield self.collate([self.window(i) for i in order[k:k + bs]])

    def collate(self, ws: List[dict]) -> Batch:
        B, W, S = len(ws), self.W, self.S
        cat = lambda key: np.stack([w[key] for w in ws])
        x, tgt, m, pos = cat("x"), cat("target"), cat("mask"), cat("pos")
        pad = cat("pad")
        N = B * W * S

        ei, ea = {}, {}
        for k in ("V", "H"):
            e, a = self.templates[k]
            off = (np.arange(B * W) * S)[:, None, None]
            ei[k] = (e[None] + off).transpose(1, 0, 2).reshape(2, -1)
            ea[k] = np.tile(a, (B * W, 1))
        src, dst, att = [], [], []
        s_ar = np.arange(S)
        for b, w_ in enumerate(ws):
            for w in range(W - 1):
                if not np.isfinite(w_["dh"][w]):
                    continue
                a_ = b * W * S + w * S + s_ar
                c_ = a_ + S
                src += [a_, c_]; dst += [c_, a_]
                att += [np.full(2 * S, w_["dh"][w])]
        if src:
            ei["T"] = np.stack([np.concatenate(src), np.concatenate(dst)])
            ea["T"] = np.concatenate(att)[:, None]
        else:
            ei["T"] = np.zeros((2, 0), dtype=np.int64)
            ea["T"] = np.zeros((0, 1))

        f = lambda a: torch.as_tensor(a, dtype=torch.float32)
        return Batch(
            x=f(x.reshape(N, N_VAR)), target=f(tgt.reshape(N, N_VAR)),
            art_mask=torch.as_tensor(m.reshape(N, N_VAR)),
            pos=f(pos.reshape(N, 5)),
            edge_index={k: torch.as_tensor(v, dtype=torch.long) for k, v in ei.items()},
            edge_attr={k: f(v) for k, v in ea.items()},
            time_pad=torch.as_tensor(pad),
            node_t=torch.as_tensor(np.broadcast_to(cat("t")[:, :, None], (B, W, S)).reshape(N).copy()),
            node_slot=torch.as_tensor(np.broadcast_to(np.arange(S), (B, W, S)).reshape(N).copy()),
            B=B, W=W, S=S,
        )


def global_random_mask(observed: np.ndarray, ratio: float, rng: np.random.RandomState):
    return (rng.rand(*observed.shape) < ratio) & observed


@dataclass
class Prepared:
    idx: NodeIndexer
    stations: pd.DataFrame
    norm: Normalizer
    splits: Dict[str, WindowData]
    raw: Dict[str, np.ndarray]
    templates: dict
    bilgi: dict


def prepare(stations: pd.DataFrame, obs: pd.DataFrame, cfg: Config,
            norm_mode: str = "level", seed: int = 0) -> Prepared:
    idx = NodeIndexer(stations, cfg)
    X, times, bilgi = idx.build_tensor(obs)
    parts = chrono_split(times, cfg)
    norm = Normalizer(norm_mode).fit(X[parts["train"]], idx.slot_level, len(idx.levels))
    tmpl = step_templates(idx, stations, cfg)
    splits, raw = {}, {}
    for k, (ad, s) in enumerate(parts.items()):
        raw[ad] = X[s]
        splits[ad] = WindowData(norm.transform(X[s]), times[s], idx, stations, tmpl, cfg,
                                mode=ad, seed=seed * 1000 + k)
    return Prepared(idx, stations, norm, splits, raw, tmpl, bilgi)


def synthetic(n_st=5, levels=(1000, 850, 500, 200), n_t=20, seed=0, drop_times=(7,)):
    rng = np.random.RandomState(seed)
    stations = pd.DataFrame({
        "station_id": [f"S{i:02d}" for i in range(n_st)],
        "lat": rng.uniform(40, 55, n_st),
        "lon": rng.uniform(0, 25, n_st),
        "elevation": np.array([50, 300, 1600, 20, 700][:n_st] + [10] * max(0, n_st - 5), float),
    })
    times = pd.date_range("2020-01-01", periods=n_t, freq="12h")
    times = times.delete(list(drop_times))
    rows = []
    for _, s in stations.iterrows():
        for t in times:
            for p in levels:
                z = 44330.0 * (1 - (p / 1013.25) ** 0.1903)
                rows.append(dict(station_id=s.station_id, datetime=t, pressure=float(p),
                                 T=15 - 0.0065 * z + rng.randn(), RH=np.clip(70 - z / 300 + 5 * rng.randn(), 1, 100),
                                 WS=abs(5 + z / 1000 + rng.randn()), WD=rng.uniform(0, 360),
                                 Z=z + 10 * rng.randn()))
    obs = pd.DataFrame(rows)
    nan = rng.rand(len(obs), 5) < 0.1
    for j, c in enumerate(HAM_SUTUNLAR):
        obs.loc[nan[:, j], c] = np.nan
    return stations, obs
