import argparse
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
from gnn_egitim import masked_mse, run_epoch
from toprak_ayarlar import IDX_T, MASKE_VARSAYILAN, N_FEAT, OZNITELIK, SoilConfig, edge_dims
from toprak_veri_ve_graf import SoilWindows, derin_aktarim_ek, norm_stats, normalize
from toprak_gnn_calistir import hazirla
from toprak_katlar import kat_istasyonlari

KOK = os.path.join(ANA_DIZIN, "results", "deep_selection")
KOK_HIBRIT = os.path.join(ANA_DIZIN, "results", "fold_selection")
IDW_P = (1, 1, 2, 2, 1)
IDW_K = (6, 6, 8, 8, 8)
TABAN = ("50:350:0.9280,100:450:0.9239", "50:450:0.9622,100:600:0.9324", "50:350:0.9266,100:600:0.9503",
         "50:350:0.9484,100:600:0.9117", "50:350:0.9810,100:600:0.8938")
SEED = 0


def ayar_hibrit(k):
    c = SoilConfig()
    c.uydu_derinlik, c.n_layers, c.maske = "tum", 2, MASKE_VARSAYILAN
    c.idw_artik, c.idw_p, c.idw_k = True, float(IDW_P[k]), IDW_K[k]
    c.holdout, c.ic_dogrulama = kat_istasyonlari(k), True
    c.device = "cpu"
    c.__post_init__()
    return c


def ayar(k):
    c = SoilConfig()
    c.uydu_derinlik, c.n_layers, c.maske = "tum", 2, MASKE_VARSAYILAN
    c.window, c.derin_maske_orani, c.station_mask_ratio = 7, 0.3, 0.0
    c.patience, c.epoch_sifir, c.max_epochs = 1000, True, 40
    c.taban = "a5"
    c.taban_param = tuple((int(x.split(":")[0]), float(x.split(":")[1]), float(x.split(":")[2])) for x in TABAN[k].split(","))
    c.idw_artik, c.idw_p, c.idw_k = True, float(IDW_P[k]), IDW_K[k]
    c.holdout, c.ic_dogrulama = kat_istasyonlari(k), True
    c.device = "cpu"
    c.__post_init__()
    return c


def pencereler_hibrit(c, P):
    o = variant_opts("gnn")
    X, dates, idx, g_tr, g_val = P["X"], P["dates"], P["idx"], P["g_tr"], P["g_val"]
    sub = lambda ti, g: X[ti][:, g.slot_global]
    mu, sd = norm_stats(sub(idx["train"], g_tr), g_tr.slot_depth, o["norm"])
    train_w = SoilWindows(normalize(sub(idx["train"], g_tr), g_tr, mu, sd), dates[idx["train"]], g_tr, c, "train", SEED)
    val_w = SoilWindows(normalize(sub(idx["val"], g_val), g_val, mu, sd), dates[idx["val"]], g_val, c, "eval", SEED,
                        masked_stations=P["val_set"])
    return o, train_w, val_w


def pencereler(c, P):
    o = variant_opts("gnn")
    X, dates, idx, g_tr, g_val = P["X"], P["dates"], P["idx"], P["g_tr"], P["g_val"]
    sub = lambda ti, g: X[ti][:, g.slot_global]
    mu, sd = norm_stats(sub(idx["train"], g_tr), g_tr.slot_depth, o["norm"])
    ek = lambda ti, g: derin_aktarim_ek(X[:, g.slot_global][..., IDX_T], g, c, sd[:, IDX_T], None)[ti]
    train_w = SoilWindows(normalize(sub(idx["train"], g_tr), g_tr, mu, sd), dates[idx["train"]], g_tr, c, "train", SEED,
                          taban_ek=ek(idx["train"], g_tr))
    val_w = SoilWindows(normalize(sub(idx["val"], g_val), g_val, mu, sd), dates[idx["val"]], g_val, c, "eval", SEED,
                        masked_stations=np.array([], dtype=int), derin_gizli_istasyon=P["val_set"],
                        taban_ek=ek(idx["val"], g_val))
    return o, train_w, val_w


def kayip_karistirmali(model, wd, c, n_kontrol=3):
    """run_epoch'un egitim dongusu (shuffle=True), adim yok, eval kipi. Ilk n_kontrol yigında model ciktisi = taban sinanir."""
    model.eval()
    top, n, i = 0.0, 0, 0
    with torch.no_grad():
        for batch in wd.batches(shuffle=True):
            batch.to(c.device)
            xt = batch.x[:, [IDX_T]]
            pred = torch.where(torch.isnan(xt), torch.nan_to_num(batch.base, nan=0.0), xt)
            if i < n_kontrol:
                pm = model(batch)
                if pm.shape != pred.shape or not torch.allclose(pm, pred, atol=1e-6, equal_nan=True):
                    raise RuntimeError(f"epoch 0 ciktisi tabana esit degil (yigin {i})")
            loss = masked_mse(pred, batch)
            i += 1
            if loss is None:
                continue
            k = int(batch.art_mask.sum())
            top += float(loss) * k
            n += k
    return top / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gorev", choices=["derin", "hibrit"], default="derin")
    gorev = ap.parse_args().gorev
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    df_m = pd.read_csv(SoilConfig().master_path, parse_dates=["tarih"])
    df_m = df_m[df_m.tarih.dt.year <= 2023]                     # test yillari okunmaz (egitim + dogrulama yeter)
    satir = []
    for k in range(5):
        c = ayar(k) if gorev == "derin" else ayar_hibrit(k)
        c.test_years = (2023,)                                  # hazirla bos olmayan test bolumu ister; kullanilmaz
        P = hazirla(c, df_m)
        o, train_w, val_w = (pencereler if gorev == "derin" else pencereler_hibrit)(c, P)
        torch.manual_seed(SEED)
        model = build_model(c, o, in_dim=N_FEAT, target_cols=[IDX_T], edge_dims=edge_dims(c),
                            maske_cols=[OZNITELIK.index(x) for x in c.maske], idw_artik=c.idw_artik)
        va0 = run_epoch(model, val_w, c)
        tr0 = kayip_karistirmali(model, train_w, c)
        if gorev == "derin":
            h = pd.read_csv(os.path.join(KOK, f"hibrit_a5_fold{k}", "history_gnn_s0.csv")).set_index("egitilen_epoch")
            v0 = h.val_loss[0]
        else:
            h = pd.read_csv(os.path.join(KOK_HIBRIT, f"hibrit_fold{k}", "history_gnn_s0.csv"))
            h.index = h.epoch + 1
            ic = pd.read_csv(os.path.join(ANA_DIZIN, "results", "folds", "analysis_inner_validation", "hibrit_ic_dogrulama_icsecim.csv"))
            v0 = float(ic[(ic.model == "hibrit") & (ic.kat == k)].kayip_0_eski.iloc[0])
        tr = h.train_loss.dropna()
        e_min, e_son = int(tr.idxmin()), int(tr.index.max())
        e_val = int(h.val_loss.loc[1:].idxmin())
        satir.append(dict(kat=k, val0_yeniden=va0, val0_kayit=v0, taban_egitim=tr0,
                          egitim_ep1=tr[1], egitim_min=tr[e_min], ep_min=e_min, egitim_son=tr[e_son], ep_son=e_son,
                          val_ep1=h.val_loss[1], val_epmin=h.val_loss[e_min], val_son=h.val_loss[e_son],
                          ep_en_iyi_val=e_val, egitim_en_iyi_val=tr[e_val], val_en_iyi=h.val_loss[e_val],
                          egitim_min_altinda=bool(tr[e_min] < tr0)))
        print(f"kat {k}: taban egitim {tr0:.5f} | history egitim ep1 {tr[1]:.5f} min {tr[e_min]:.5f} (ep {e_min}) "
              f"son {tr[e_son]:.5f} (ep {e_son}) | val0 yeniden {va0:.6f} kayit {v0:.6f}", flush=True)
    t = pd.DataFrame(satir)
    t["val0_fark"] = t.val0_yeniden - t.val0_kayit
    if (t.val0_fark.abs() > 1e-6).any():
        raise RuntimeError("epoch 0 dogrulama kaybi kayitla tutmadi")
    t.to_csv(os.path.join(KOK, "egitim_kaybi_a5.csv") if gorev == "derin"
             else os.path.join(ANA_DIZIN, "results", "folds", "analysis_inner_validation", "egitim_kaybi_hibrit.csv"), index=False)
    print()
    print(t.to_string(index=False, float_format=lambda v: f"{v:.5f}"))


if __name__ == "__main__":
    main()
