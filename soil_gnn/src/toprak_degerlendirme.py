import numpy as np
import pandas as pd
import torch

from toprak_ayarlar import DERINLIKLER, IDX_T, SoilConfig
from toprak_veri_ve_graf import SoilGraph, SoilWindows, denorm_T


@torch.no_grad()
def predict_masked(model, wins: SoilWindows, cfg: SoilConfig):
    """Ortusen pencere tahminlerini dugum (t, slot) bazinda tekillestirir (normalize T).
    Ayni dugum farkli pencerelerde farkli maskeliyse hata verir."""
    model.eval()
    T, S = wins.T, wins.S
    toplam = np.zeros((T, S)); sayi = np.zeros((T, S))
    pred = np.full((T, S), np.nan)
    goruldu = np.full((T, S), -1, dtype=np.int8)
    for batch in wins.batches(shuffle=False):
        batch.to(cfg.device)
        out = model(batch)[:, 0].cpu().numpy()
        t = batch.node_t.cpu().numpy(); s = batch.node_slot.cpu().numpy()
        ok = t >= 0
        t, s, out = t[ok], s[ok], out[ok]
        m_art = batch.art_mask[:, 0].cpu().numpy()[ok]
        onceki = goruldu[t, s]
        if ((onceki >= 0) & (onceki != m_art)).any():
            raise RuntimeError("Maske dugum tutarli degil.")
        goruldu[t, s] = m_art
        if cfg.dedup == "mean":
            np.add.at(toplam, (t, s), out); np.add.at(sayi, (t, s), 1)
        else:
            _, ilk = np.unique(t * S + s, return_index=True)
            sec = ilk[sayi[t[ilk], s[ilk]] == 0]
            pred[t[sec], s[sec]] = out[sec]; sayi[t[sec], s[sec]] += 1
    if cfg.dedup == "mean":
        pred = np.where(sayi > 0, toplam / np.maximum(sayi, 1), np.nan)
    return pred


def eval_rows(model, Xn, raw_T, dates, g: SoilGraph, groups, cfg: SoilConfig, mu, sd, seed=0, derin=False, taban_ek=None):
    """Her grup ayri bir geciste tamamen maskelenir (derin=True: yalniz cfg.derin_gizli derinlikleri); o grubun
    gizli ve olculmus dugumleri icin (istasyon, derinlik, tarih, y_true, y_pred) satirlari."""
    parca = []
    for grp in groups:
        te = taban_ek(grp) if callable(taban_ek) else taban_ek
        if derin:
            wins = SoilWindows(Xn, dates, g, cfg, "eval", seed, masked_stations=np.array([], dtype=int),
                               derin_gizli_istasyon=grp, taban_ek=te)
            sec_slot = np.isin(g.slot_station, grp) & np.isin(np.asarray(DERINLIKLER)[g.slot_depth], cfg.derin_gizli)
        else:
            wins = SoilWindows(Xn, dates, g, cfg, "eval", seed, masked_stations=grp, taban_ek=te)
            sec_slot = np.isin(g.slot_station, grp) & (np.asarray(DERINLIKLER)[g.slot_depth] > 0)
        yhat = denorm_T(predict_masked(model, wins, cfg), g, mu, sd)
        tt, ss = np.nonzero(np.isfinite(raw_T) & sec_slot[None, :])
        parca.append(pd.DataFrame({
            "istasyon_no": g.stations["istasyon_no"].to_numpy()[g.slot_station[ss]],
            "derinlik": np.asarray(DERINLIKLER)[g.slot_depth[ss]],
            "tarih": dates[tt],
            "y_true": raw_T[tt, ss].astype(float),
            "y_pred": yhat[tt, ss].astype(float),
        }))
    df = pd.concat(parca, ignore_index=True)
    if df["y_pred"].isna().any():
        raise RuntimeError(f"{int(df['y_pred'].isna().sum())} satirda tahmin yok.")
    return df


def _m(y, yh):
    e = yh - y
    ss = ((y - y.mean()) ** 2).sum()
    return dict(n=len(y), rmse=float(np.sqrt((e ** 2).mean())), mae=float(np.abs(e).mean()),
                r2=float(1 - (e ** 2).sum() / ss) if ss > 0 else np.nan, mbe=float(e.mean()))


def metrics(df: pd.DataFrame) -> pd.DataFrame:
    satir = [dict(derinlik="all", **_m(df.y_true.values, df.y_pred.values))]
    for d, g in df.groupby("derinlik"):
        satir.append(dict(derinlik=int(d), **_m(g.y_true.values, g.y_pred.values)))
    return pd.DataFrame(satir)


def station_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([dict(istasyon_no=k, **_m(g.y_true.values, g.y_pred.values))
                         for k, g in df.groupby("istasyon_no")])
