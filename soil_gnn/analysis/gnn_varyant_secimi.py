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
from toprak_ayarlar import DERINLIKLER, IDX_T, MASKE_VARSAYILAN, N_FEAT, OZNITELIK, SoilConfig, edge_dims
from toprak_veri_ve_graf import norm_stats, normalize
from toprak_degerlendirme import eval_rows
from toprak_gnn_calistir import hazirla
from toprak_katlar import kat_istasyonlari
import idw_karsilastirma_modeli as IDWM

IC_P, IC_K = (1, 1, 2, 2, 2), (4, 4, 8, 6, 8)
AYARLAR = ("a_gnn", "a_hibrit", "ceza01", "ceza1", "ceza10", "blok")
TOLERANS = 1e-4
ANAHTAR = ["istasyon_no", "derinlik", "tarih"]


def ayar_cfg(ayar, k):
    c = SoilConfig()
    c.uydu_derinlik, c.n_layers, c.maske = "tum", 2, MASKE_VARSAYILAN
    c.holdout, c.ic_dogrulama = kat_istasyonlari(k), True
    if ayar != "a_gnn":
        c.idw_artik, c.idw_p, c.idw_k = True, float(IC_P[k]), IC_K[k]
    if ayar in ("a_gnn", "a_hibrit"):
        c.yatay = "softmax"
    c.ceza = {"ceza01": 0.1, "ceza1": 1.0, "ceza10": 10.0}.get(ayar, 0.0)
    c.blok_maske = ayar == "blok"
    c.__post_init__()
    c.test_years = (2023,)
    return c


def ist_rmse(df, kol="y_pred"):
    e = df[kol].to_numpy(float) - df.y_true.to_numpy(float)
    return pd.Series(e ** 2).groupby(df.istasyon_no.to_numpy()).mean() ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kok", default=os.path.join(ANA_DIZIN, "results", "variant_selection"))
    ap.add_argument("--katlar", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--ayarlar", nargs="+", default=list(AYARLAR))
    ap.add_argument("--master", default=SoilConfig().master_path)
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    df_m = pd.read_csv(a.master, parse_dates=["tarih"])
    df_m = df_m[df_m.tarih.dt.year <= 2023]
    estar, egri = {}, []
    for ay in a.ayarlar:
        h = []
        for k in a.katlar:
            x = pd.read_csv(os.path.join(a.kok, f"{ay}_fold{k}", "history_gnn_s0.csv"))
            if len(x) != 41 or list(x.egitilen_epoch) != list(range(41)) or x.val_loss.isna().any():
                raise RuntimeError(f"{ay} kat {k}: gecmis 41 satir / egitilen_epoch 0..40 / NaN'siz degil "
                                   f"({len(x)} satir, NaN {int(x.val_loss.isna().sum())})")
            h.append(x.set_index("egitilen_epoch").val_loss.rename(k))
        t = pd.concat(h, axis=1)
        t["ortalama"] = t.mean(axis=1)
        estar[ay] = int(t.ortalama.idxmin())
        egri.append(t.assign(ayar=ay))
    satir = []
    for k in a.katlar:
        ref = None
        for ay in a.ayarlar:
            c = ayar_cfg(ay, k)
            P = hazirla(c, df_m)
            X, idx, g_tr, g_val = P["X"], P["idx"], P["g_tr"], P["g_val"]
            sub = lambda ti, g: X[ti][:, g.slot_global]
            mu, sd = norm_stats(sub(idx["train"], g_tr), g_tr.slot_depth, variant_opts("gnn")["norm"])
            m = build_model(c, variant_opts("gnn"), in_dim=N_FEAT, target_cols=[IDX_T], edge_dims=edge_dims(c),
                            maske_cols=[OZNITELIK.index(x) for x in c.maske], idw_artik=c.idw_artik)
            m.load_state_dict(torch.load(os.path.join(a.kok, f"{ay}_fold{k}", "epochs", f"epoch_{estar[ay]:03d}.pt"),
                                         map_location="cpu", weights_only=True))
            raw = sub(idx["val"], g_val)
            df = eval_rows(m, normalize(raw, g_val, mu, sd), raw[..., IDX_T], P["dates"][idx["val"]], g_val,
                           [P["val_set"]], c, mu, sd, 0)
            if ref is None:                         # rakim-IDW, ic p, k, ayni satirlar
                st = P["stations"]
                T = X[..., IDX_T].reshape(len(P["dates"]), len(st), len(DERINLIKLER)).astype(float)
                vi = idx["val"]
                r = IDWM.satirlar(T[vi], P["dates"][vi], st.istasyon_no.to_numpy(), st.lat.to_numpy(float),
                                  st.lon.to_numpy(float), g_tr.st_global, [P["ic_global"]], float(IC_P[k]), IC_K[k],
                                  st.rakim.to_numpy(float) / 1000.0)
                ref = r.rename(columns={"y_pred": "idw"})[ANAHTAR + ["y_true", "idw"]]
                satir.append(dict(kat=k, ayar="rakim_idw (ic p, k)", estar=np.nan,
                                  istasyon_rmse_ort=float(ist_rmse(ref, "idw").mean()), n_satir=len(ref)))
            j = df[ANAHTAR + ["y_true", "y_pred"]].merge(ref, on=ANAHTAR, how="outer", suffixes=("", "_r"), indicator=True)
            if not (j._merge == "both").all() or not np.allclose(j.y_true, j.y_true_r) or j.idw.isna().any():
                raise RuntimeError(f"kat {k} {ay}: satirlar rakim-IDW ile ayni degil")
            satir.append(dict(kat=k, ayar=ay, estar=estar[ay], istasyon_rmse_ort=float(ist_rmse(df).mean()), n_satir=len(df)))
            print(f"kat {k} {ay}: E*={estar[ay]}, {len(df)} satir", flush=True)
    t = pd.DataFrame(satir)
    ozet = t.pivot_table(index="ayar", columns="kat", values="istasyon_rmse_ort")
    ozet["ortalama"] = ozet.mean(axis=1)
    ozet["E*"] = pd.Series(estar)
    os.makedirs(a.kok, exist_ok=True)
    t.to_csv(os.path.join(a.kok, "kapi_kat.csv"), index=False)
    ozet.to_csv(os.path.join(a.kok, "kapi_ozet.csv"))
    pd.concat(egri).to_csv(os.path.join(a.kok, "ic_dogrulama_egrileri.csv"))
    print(ozet.to_string(float_format=lambda v: f"{v:.3f}"))
    adaylar = ozet.drop(index="rakim_idw (ic p, k)")
    en = adaylar.ortalama.idxmin()
    referans = ozet.loc["rakim_idw (ic p, k)", "ortalama"]
    print(f"\nEn iyi ayar: {en} ({adaylar.ortalama[en]:.3f}); rakim-IDW {referans:.3f}")
    gecti = adaylar.ortalama[en] < referans - TOLERANS
    print(f"Fark (en iyi - rakim-IDW): {adaylar.ortalama[en] - referans:+.6f} C (gecme kosulu < -{TOLERANS})")
    print("KAPI:", "GECTI (2. asama ayrica onaya)" if gecti else "GECMEDI -> dur")


if __name__ == "__main__":
    main()
