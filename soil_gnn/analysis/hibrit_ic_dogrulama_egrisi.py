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
from gnn_egitim import run_epoch
from toprak_ayarlar import IDX_T, MASKE_VARSAYILAN, N_FEAT, OZNITELIK, SoilConfig, edge_dims
from toprak_veri_ve_graf import SoilWindows, norm_stats, normalize
from toprak_gnn_calistir import hazirla
from toprak_katlar import kat_istasyonlari

IDW_P = (1, 1, 2, 2, 1)
IDW_K = (6, 6, 8, 8, 8)
SECIM = os.path.join(ANA_DIZIN, "results", "fold_selection")


def main():
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    df = None
    satir = []
    for k in range(5):
        for ad, konum in (("hibrit", True), ("hibrit_konumyok", False)):
            c = SoilConfig()
            c.uydu_derinlik, c.n_layers, c.maske = "tum", 2, MASKE_VARSAYILAN
            c.holdout, c.ic_dogrulama = kat_istasyonlari(k), True
            c.idw_artik, c.idw_p, c.idw_k, c.konum_pe = True, float(IDW_P[k]), IDW_K[k], konum
            c.__post_init__()
            if df is None:
                df = pd.read_csv(c.master_path, parse_dates=["tarih"])
            P = hazirla(c, df)
            X, idx, g_tr, g_val = P["X"], P["idx"], P["g_tr"], P["g_val"]
            sub = lambda ti, g: X[ti][:, g.slot_global]
            mu, sd = norm_stats(sub(idx["train"], g_tr), g_tr.slot_depth, "level")
            val_w = SoilWindows(normalize(sub(idx["val"], g_val), g_val, mu, sd), P["dates"][idx["val"]],
                                g_val, c, "eval", 0, masked_stations=P["val_set"])
            yap = lambda: build_model(c, variant_opts("gnn"), in_dim=N_FEAT, target_cols=[IDX_T],
                                      edge_dims=edge_dims(c), maske_cols=[OZNITELIK.index(x) for x in c.maske],
                                      idw_artik=True)
            torch.manual_seed(0)
            taze = yap()
            assert float(taze.out[-1].weight.abs().sum()) == 0.0
            l0 = run_epoch(taze, val_w, c)
            kayitli = yap()
            kayitli.load_state_dict(torch.load(os.path.join(SECIM, f"{ad}_fold{k}", "model_gnn_s0.pt"),
                                               map_location="cpu", weights_only=True))
            l_kayitli = run_epoch(kayitli, val_w, c)
            h = pd.read_csv(os.path.join(SECIM, f"{ad}_fold{k}", "history_gnn_s0.csv"))
            i = int(h.val_loss.idxmin())
            satir.append(dict(model=ad, kat=k, n_ic=len(P["val_set"]), kayip_0_rakimIDW=l0,
                              en_iyi_epoch=i, sabit_epoch=i + 1, kayip_en_iyi=h.val_loss[i],
                              goreli_iyilesme=(l0 - h.val_loss[i]) / l0, kayip_ep0=h.val_loss[0],
                              son_epoch=int(h.epoch.iloc[-1]), kayip_son=h.val_loss.iloc[-1],
                              kayitli_model_kaybi=l_kayitli, sinama_fark=l_kayitli - h.val_loss[i]))
            print(satir[-1], flush=True)
    t = pd.DataFrame(satir)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out = os.path.join(ANA_DIZIN, "results", "folds", "analysis_inner_validation")
    os.makedirs(out, exist_ok=True)
    t.to_csv(os.path.join(out, "hibrit_ic_dogrulama_egrisi.csv"), index=False)


if __name__ == "__main__":
    main()
