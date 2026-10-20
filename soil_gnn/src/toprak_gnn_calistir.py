import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from ortak_ayarlar import VARIANTS, variant_opts
from gnn_model import build_model
from toprak_ayarlar import IDX_T, MASKE_VARSAYILAN, N_FEAT, OZNITELIK, SoilConfig, edge_dims, ortam_bilgisi
from toprak_veri_ve_graf import SoilGraph, SoilWindows, Splits, derin_aktarim_ek, load_master, norm_stats, normalize
from toprak_degerlendirme import eval_rows, metrics, station_metrics
from toprak_katlar import en_yakin_gorunur, ic_dogrulama_istasyonlari, kat_istasyonlari
from gnn_egitim import fit


def hazirla(cfg: SoilConfig, df=None):
    X, dates, stations = load_master(cfg, df)
    yil = dates.year
    idx = {}
    for ad in ("train", "val", "test"):
        idx[ad] = np.nonzero(np.isin(yil, getattr(cfg, f"{ad}_years")))[0]
        if len(idx[ad]) == 0:
            raise ValueError(f"'{ad}' bolumu bos: {getattr(cfg, f'{ad}_years')}")
    hold = np.isin(stations["istasyon_no"], cfg.holdout)
    ic = np.zeros(len(stations), bool)
    if cfg.ic_dogrulama:
        ic = np.isin(stations["istasyon_no"], ic_dogrulama_istasyonlari(stations["istasyon_no"].to_numpy()[~hold]))
    tr_st, all_st = np.nonzero(~hold & ~ic)[0], np.arange(len(stations))
    g_tr = SoilGraph(stations.iloc[tr_st], tr_st, cfg)
    g_all = SoilGraph(stations, all_st, cfg)
    rng = np.random.RandomState(2026)
    groups = np.array_split(rng.permutation(g_tr.n_st), cfg.eval_groups)
    if cfg.ic_dogrulama:      # erken durdurma: ic dogrulama istasyonlari egitim grafinda yok, dogrulamada tamamen maskeli
        val_st = np.nonzero(~hold)[0]
        g_val = SoilGraph(stations.iloc[val_st], val_st, cfg)
        val_set = np.nonzero(ic[val_st])[0]
    else:
        g_val = g_tr
        val_set = np.random.RandomState(7).choice(
            g_tr.n_st, max(1, int(round(cfg.station_mask_ratio * g_tr.n_st))), replace=False)
    return dict(X=X, dates=dates, stations=stations, idx=idx, g_tr=g_tr, g_all=g_all, g_val=g_val,
                hold_local=np.nonzero(hold)[0], ic_global=np.nonzero(ic)[0], groups=groups, val_set=val_set)


def epoch_kaynaktan(yol):
    """1. asama gecmisi (history_*.csv): en dusuk dogrulama kaybinin epoch'u + 1 = 2. asamanin sabit epoch sayisi."""
    h = pd.read_csv(yol)
    if h.val_loss.notna().sum() == 0:
        raise ValueError(f"{yol}: dogrulama kaybi yok (1. asama gecmisi degil)")
    if "egitilen_epoch" in h.columns:             # epoch_sifir: 0 (egitim yok) da secilebilir
        return int(h.egitilen_epoch[h.val_loss.idxmin()])
    return int(h.val_loss.idxmin()) + 1


def kos(cfg: SoilConfig, P: dict, variant: str, seed: int, verbose=False, model_yolu=None, sabit_epoch=None):
    o = variant_opts(variant)
    X, dates, idx, g_tr, g_all = P["X"], P["dates"], P["idx"], P["g_tr"], P["g_all"]
    sub = lambda ti, g: X[ti][:, g.slot_global]
    mu, sd = norm_stats(sub(idx["train"], g_tr), g_tr.slot_depth, o["norm"])
    derin_mod = cfg.derin_maske_orani > 0
    ek_on = {}

    def ek(ti, g, gizli=None):
        """taban_ek: butun donem icin (graf, kaynak disi gizli istasyonlar) basina bir kez, sonra ti ile kesilir."""
        if cfg.taban == "idw":
            return None
        anahtar = (id(g), None if gizli is None else tuple(sorted(np.asarray(gizli, int).tolist())))
        if anahtar not in ek_on:
            gz = None
            if gizli is not None:
                gz = np.zeros(g.n_st, bool)
                gz[np.asarray(gizli, int)] = True
            ek_on[anahtar] = derin_aktarim_ek(X[:, g.slot_global][..., IDX_T], g, cfg, sd[:, IDX_T], gz)
        return ek_on[anahtar][ti]

    train_w = SoilWindows(normalize(sub(idx["train"], g_tr), g_tr, mu, sd), dates[idx["train"]],
                          g_tr, cfg, "train", seed, taban_ek=ek(idx["train"], g_tr))
    g_val = P["g_val"]
    if derin_mod:                 # dogrulama da derin maskeli: ic dogrulama istasyonlarinin yalniz derin_gizli'si gizli
        val_w = SoilWindows(normalize(sub(idx["val"], g_val), g_val, mu, sd), dates[idx["val"]], g_val, cfg, "eval", seed,
                            masked_stations=np.array([], dtype=int), derin_gizli_istasyon=P["val_set"],
                            taban_ek=ek(idx["val"], g_val))
    else:
        val_w = SoilWindows(normalize(sub(idx["val"], g_val), g_val, mu, sd), dates[idx["val"]],
                            g_val, cfg, "eval", seed, masked_stations=P["val_set"],
                            taban_ek=ek(idx["val"], g_val, P["val_set"]))
    torch.manual_seed(seed)
    model = build_model(cfg, o, in_dim=N_FEAT, target_cols=[IDX_T], edge_dims=edge_dims(cfg),
                        maske_cols=[OZNITELIK.index(c) for c in cfg.maske], idw_artik=cfg.idw_artik)
    if model_yolu:     # yalniz degerlendirme: kayitli agirliklar, egitim yok
        model.load_state_dict(torch.load(model_yolu, map_location=cfg.device, weights_only=True))
        model.to(cfg.device)
        hist = []
    else:
        hist = fit(model, Splits(train_w, val_w), cfg, seed=seed, verbose=verbose, sabit_epoch=sabit_epoch)

    tum = np.arange(len(dates))
    sonuc = {}
    bolumler = [("temporal", idx["test"], g_tr, P["groups"]),
                ("spatial_test", idx["test"], g_all, [P["hold_local"]]),
                ("spatial_full", tum, g_all, [P["hold_local"]]),
                ("val2023", idx["val"], g_tr, P["groups"]),
                ("spatial_val2023", idx["val"], g_all, [P["hold_local"]])]
    if cfg.ic_dogrulama:
        bolumler.append(("ic_val2023", idx["val"], g_val, [P["val_set"]]))
    if cfg.derin_maske_orani > 0 or cfg.taban != "idw":   # eksik derinlik gorevi: 5-20 cm gorunur, derinler gizli
        bolumler += [("derin_val2023", idx["val"], g_tr, P["groups"]),
                     ("derin_spatial_val2023", idx["val"], g_all, [P["hold_local"]])]
        if cfg.ic_dogrulama:
            bolumler.append(("derin_ic_val2023", idx["val"], g_val, [P["val_set"]]))
        if cfg.derin_test:        # son test (bayrakla; varsayilan kapali)
            bolumler.append(("derin_spatial_test", idx["test"], g_all, [P["hold_local"]]))
    if getattr(cfg, "sadece_ic", False):
        bolumler = [b for b in bolumler if b[0] in ("ic_val2023", "derin_ic_val2023")]
    for ad, ti, g, grp in bolumler:
        raw = sub(ti, g)
        derin = ad.startswith("derin_")
        sonuc[ad] = eval_rows(model, normalize(raw, g, mu, sd), raw[..., IDX_T], dates[ti], g,
                              grp, cfg, mu, sd, seed, derin=derin,
                              taban_ek=lambda grp_, ti=ti, g=g, derin=derin: ek(ti, g, None if derin else grp_))
    return model, hist, sonuc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--uydu", choices=["yuzey", "tum"], help="uydu oznitelikleri hangi dugumlerde")
    ap.add_argument("--n-layers", type=int)
    ap.add_argument("--pencere", type=int, help="pencere uzunlugu W (gun; varsayilan 3)")
    ap.add_argument("--derin-maske-orani", type=float,
                    help="egitim: gorunur istasyonlarin bu oraninda 50/100 cm gizli (eksik derinlik gorevi; varsayilan 0)")
    ap.add_argument("--istasyon-maske-orani", type=float, help="egitimde tamamen gizlenen istasyon orani (varsayilan 0.15)")
    ap.add_argument("--patience", type=int, help="erken durdurma sabri (varsayilan 10; buyuk deger = erken durdurma yok)")
    ap.add_argument("--epoch-sifir", action="store_true", help="egitimden once dogrulama kaybi (egitilen_epoch 0) kaydedilir")
    ap.add_argument("--derin-test", action="store_true", help="son test bolumu derin_spatial_test'i de yaz (varsayilan kapali)")
    ap.add_argument("--taban", choices=["idw", "a3", "a5"], help="hibrit tabani (varsayilan idw)")
    ap.add_argument("--taban-param", help="a3/a5: derinlik:L_ya_da_kappa:gamma virgulle, or. 50:7:0.88,100:30:0.86")
    ap.add_argument("--cikar", nargs="*", default=None,
                    help="girdiden cikarilan oznitelikler, listenin tamami (varsayilan soil_moisture; tek basina --cikar = hicbiri)")
    ap.add_argument("--maske", nargs="*", default=None,
                    help="eksik-deger gostergesi; degersiz: T LST_Day_C LST_Night_C, ya da oznitelik adlari")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "soil"))
    ap.add_argument("--dikey-tam", action="store_true", help="dusey kenar: her derinlik tum derinliklere")
    ap.add_argument("--yatay-rakim", action="store_true", help="yatay kenara rakim farki (3. oznitelik)")
    ap.add_argument("--idw-artik", action="store_true", help="hibrit: IDW_rakim tahmini girdi + cikti IDW_rakim'e eklenir")
    ap.add_argument("--idw-p", type=float, help="hibrit IDW uzaklik ussu (varsayilan 2; katli kosuda o katin IDW secimi)")
    ap.add_argument("--idw-k", type=int, help="hibrit IDW en yakin k (varsayilan 6; katli kosuda o katin IDW secimi)")
    ap.add_argument("--device", help="cpu (varsayilan) ya da cuda")
    ap.add_argument("--master", help="master_dataset_qc.csv yolu (varsayilan SoilConfig.master_path)")
    ap.add_argument("--kat", type=int, help="uzaysal katli dogrulama: 0-4; bu katin istasyonlari holdout (toprak_katlar.py)")
    ap.add_argument("--konum-yok", action="store_true", help="ablasyon: enlem/boylam konumsal kodlamadan cikarilir")
    ap.add_argument("--ic-dogrulama", action="store_true",
                    help="katli 1. asama: ic egitim istasyonlariyla egitim, erken durdurma ic dogrulama istasyonlarinin 2023'uyle")
    ap.add_argument("--sabit-epoch", type=int, help="katli 2. asama: erken durdurma yok, tam bu kadar epoch")
    ap.add_argument("--epoch-kaynak", metavar="HISTORY_CSV",
                    help="katli 2. asama: sabit epoch = 1. asama gecmisinde en dusuk dogrulama kaybinin epoch'u + 1")
    ap.add_argument("--sadece-degerlendir", metavar="MODEL_PT", help="egitim yok: kayitli agirliklarla dort degerlendirme")
    ap.add_argument("--yatay", choices=["kapi", "softmax"], help="yatay katman (varsayilan kapi; softmax = ogrenilen uzaklik agirligi)")
    ap.add_argument("--ceza", type=float, help="hibrit: duzeltme cezasi agirligi (varsayilan 0)")
    ap.add_argument("--blok-maske", action="store_true", help="egitimde cografi blok maskesi (~%%20)")
    ap.add_argument("--epoch-kayit", action="store_true", help="her epoch'un agirliklari <out>/epochs/ altina")
    ap.add_argument("--sadece-ic", action="store_true", help="1. asama: yalniz ic dogrulama bolumu degerlendirilir")
    ap.add_argument("--save-pred", choices=["none", "gnn", "all"], default="gnn")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    cfg = SoilConfig()
    if a.max_epochs:
        cfg.max_epochs = a.max_epochs
    if a.uydu:
        cfg.uydu_derinlik = a.uydu
    if a.n_layers:
        cfg.n_layers = a.n_layers
    if a.pencere:
        cfg.window = a.pencere
    if a.derin_maske_orani is not None:
        cfg.derin_maske_orani = a.derin_maske_orani
    if a.istasyon_maske_orani is not None:
        cfg.station_mask_ratio = a.istasyon_maske_orani
    if a.patience is not None:
        cfg.patience = a.patience
    if a.epoch_sifir:
        cfg.epoch_sifir = True
    if a.derin_test:
        cfg.derin_test = True
    if a.taban:
        cfg.taban = a.taban
    if a.taban_param:
        cfg.taban_param = tuple((int(x.split(":")[0]), float(x.split(":")[1]), float(x.split(":")[2]))
                                for x in a.taban_param.split(","))
    if a.cikar is not None:
        cfg.cikar = tuple(a.cikar)
    if a.maske is not None:
        cfg.maske = tuple(a.maske) if a.maske else MASKE_VARSAYILAN
    if a.dikey_tam:
        cfg.dikey_tam = True
    if a.yatay_rakim:
        cfg.yatay_rakim = True
    if a.idw_artik:
        cfg.idw_artik = True
    if a.idw_p is not None:
        cfg.idw_p = a.idw_p
    if a.idw_k is not None:
        cfg.idw_k = a.idw_k
    if a.device:
        cfg.device = a.device
    if a.master:
        cfg.master_path = a.master
    sabit = epoch_kaynaktan(a.epoch_kaynak) if a.epoch_kaynak else a.sabit_epoch
    if a.ic_dogrulama and sabit is not None:
        raise ValueError("--ic-dogrulama (1. asama) ile --sabit-epoch/--epoch-kaynak (2. asama) birlikte kullanilmaz.")
    if a.kat is not None:
        cfg.holdout = kat_istasyonlari(a.kat)
        if not a.ic_dogrulama and sabit is None:
            raise ValueError("Katli kosuda asama belirtilmeli: --ic-dogrulama (1. asama, tohum 0) ya da "
                             "--sabit-epoch/--epoch-kaynak (2. asama, 65-66 istasyonun tamami).")
    if a.ic_dogrulama:
        cfg.ic_dogrulama = True
    if a.konum_yok:
        cfg.konum_pe = False
    if a.yatay:
        cfg.yatay = a.yatay
    if a.ceza is not None:
        cfg.ceza = a.ceza
    if a.blok_maske:
        cfg.blok_maske = True
    if a.epoch_kayit:
        cfg.epoch_kayit = os.path.join(a.out, "epochs")
    if a.sadece_ic:
        if not a.ic_dogrulama:
            raise ValueError("--sadece-ic yalniz --ic-dogrulama ile")
        cfg.sadece_ic = True
    cfg.__post_init__()
    varyant_kosusu = (cfg.yatay != "kapi" or cfg.ceza > 0 or cfg.blok_maske or set(cfg.cikar) != {"soil_moisture"})
    if varyant_kosusu and cfg.ic_dogrulama and not cfg.sadece_ic:
        raise ValueError("varyant ve uydu ablasyonu kosularinin 1. asamasi yalniz --sadece-ic ile kosulur (test bolumleri hesaplanmaz).")
    if a.epoch_kayit and not a.epoch_sifir:
        raise ValueError("--epoch-kayit yalniz --epoch-sifir ile (epoch_000.pt = egitimsiz model; E* 0 olabilir).")
    seeds = a.seeds if a.seeds is not None else list(cfg.seeds)
    if a.epoch_kayit and (len(a.variants) != 1 or len(seeds) != 1):
        raise ValueError("--epoch-kayit tek varyant ve tek tohumla kullanilir.")
    if a.sadece_degerlendir and (len(a.variants) != 1 or len(seeds) != 1):
        raise ValueError("--sadece-degerlendir tek varyant ve tek tohumla kullanilir.")
    os.makedirs(a.out, exist_ok=True)
    print("Ortam:", ortam_bilgisi(a.out, cfg.master_path), flush=True)

    t0 = time.time()
    P = hazirla(cfg)
    g = P["g_tr"]
    print(f"Veri: {len(P['dates'])} gun, {len(P['stations'])} istasyon "
          f"(egitim grafi {g.n_st}, holdout {len(P['hold_local'])}), S_egitim={g.S}/adim, "
          f"yatay cift={g.n_pairs} (ort. derece {g.n_pairs / g.n_st:.1f}), uydu={cfg.uydu_derinlik}, "
          f"katman={cfg.n_layers}, cikar={list(cfg.cikar)}, maske={list(cfg.maske)}, dikey_tam={cfg.dikey_tam}, yatay_rakim={cfg.yatay_rakim}, "
          f"idw_artik={cfg.idw_artik} (p={cfg.idw_p}, k={cfg.idw_k}), konum_pe={cfg.konum_pe}, kat={a.kat}, "
          f"ic_dogrulama={cfg.ic_dogrulama} ({len(P['ic_global'])} ist.), sabit_epoch={sabit}, "
          f"device={cfg.device}  [{time.time() - t0:.0f} s]", flush=True)
    hold_no = P["stations"]["istasyon_no"].to_numpy()[P["hold_local"]]
    gorunur_no = P["stations"]["istasyon_no"].to_numpy()[np.setdiff1d(np.arange(len(P["stations"])), P["hold_local"])]
    en_yakin_gorunur(P["stations"], hold_no, gorunur_no).assign(
        kat=-1 if a.kat is None else a.kat).to_csv(os.path.join(a.out, "istasyon_mesafe.csv"), index=False)

    ozet = []
    for v in a.variants:
        for s in seeds:
            t1 = time.time()
            model, hist, sonuc = kos(cfg, P, v, s, a.verbose, model_yolu=a.sadece_degerlendir, sabit_epoch=sabit)
            etiket = f"{v}_s{s}"
            pd.DataFrame(hist).to_csv(os.path.join(a.out, f"history_{etiket}.csv"), index=False)
            torch.save(model.state_dict(), os.path.join(a.out, f"model_{etiket}.pt"))
            for ad, df in sonuc.items():
                m = metrics(df).assign(split=ad, model=v, seed=s)
                ozet.append(m)
                station_metrics(df).to_csv(os.path.join(a.out, f"station_{ad}_{etiket}.csv"), index=False)
                if a.save_pred == "all" or (a.save_pred == "gnn" and v == "gnn"):
                    df.to_csv(os.path.join(a.out, f"pred_{ad}_{etiket}.csv.gz"), index=False)
            with open(os.path.join(a.out, f"edge_weights_{etiket}.json"), "w", encoding="utf-8") as f:
                json.dump(model.get_edge_weights(), f, indent=2)
            satir = " | ".join(f"{ad} RMSE {metrics(df).iloc[0].rmse:.3f}" for ad, df in sonuc.items())
            print(f"{etiket:28s} epoch={len(hist):3d}  {satir}  [{(time.time() - t1) / 60:.1f} dk]", flush=True)
            pd.concat(ozet, ignore_index=True).to_csv(os.path.join(a.out, "metrics.csv"), index=False)


if __name__ == "__main__":
    main()
