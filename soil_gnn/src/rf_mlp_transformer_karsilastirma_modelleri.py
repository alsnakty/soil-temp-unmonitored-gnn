import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from toprak_ayarlar import DERINLIKLER, IDX_T, OZNITELIK, SOIL_VARS, SoilConfig, ortam_bilgisi
from toprak_degerlendirme import metrics, station_metrics
from toprak_katlar import en_yakin_gorunur, kat_istasyonlari
from toprak_gnn_calistir import hazirla

pd.set_option("compute.use_numexpr", False)   # pyg ortamindaki numexpr 2.11.0 buyuk dizilerde ara sira yanlis sonuc veriyor

ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RF_PARAMS = dict(n_estimators=300, max_depth=20, min_samples_leaf=5)
RF_PARCA = 10
MLP_PARAMS = dict(hidden_dims=[128, 128, 64], dropout=0.1, lr=1e-3, weight_decay=1e-4,
                  batch_size=256, n_epochs=200, patience=15)
TR_PARAMS = dict(d_token=16, n_heads=2, n_layers=2, ffn_factor=1.33, dropout=0.1, lr=1e-3,
                 weight_decay=1e-4, batch_size=1024, n_epochs=100, patience=10)

DIN = ["LST_Day_C", "LST_Night_C", "NDVI"]
GIRDI_ADLARI = DIN + ["gun_sin", "gun_cos"] + SOIL_VARS + ["derinlik", "lat", "lon", "rakim", "LST_Day_eksik", "LST_Night_eksik"]
GIRDI_ADLARI_UYDUSUZ = ["gun_sin", "gun_cos"] + SOIL_VARS + ["derinlik", "lat", "lon", "rakim"]


def tablo(P, ti, st, uydu=True):
    """X (gun, istasyon*derinlik, oznitelik) -> satir tablosu: yalniz derinlik > 0 ve T olculmus hucreler.
    uydu=False (ablasyon): LST/NDVI sutunlari ve LST eksik isaretleri yok (GIRDI_ADLARI_UYDUSUZ); satirlar ayni."""
    X, dates, stations = P["X"], P["dates"], P["stations"]
    nd = len(DERINLIKLER)
    Xr = X[ti].reshape(len(ti), len(stations), nd, -1)[:, st]           # (gun, ist, derinlik, F)
    tt, ss, dd = np.nonzero(np.isfinite(Xr[..., IDX_T]) & (np.asarray(DERINLIKLER) > 0)[None, None, :])
    F = Xr[tt, ss, dd]
    doy = dates[ti][tt].dayofyear.to_numpy(float)
    din = F[:, [OZNITELIK.index(c) for c in DIN]].astype(float)
    soil = F[:, [OZNITELIK.index(v) for v in SOIL_VARS]].astype(float)
    st_g = np.asarray(st)[ss]
    orta = [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), soil,
            np.asarray(DERINLIKLER, float)[dd] / 100.0,
            stations["lat"].to_numpy(float)[st_g], stations["lon"].to_numpy(float)[st_g], stations["rakim"].to_numpy(float)[st_g]]
    if uydu:
        G = np.column_stack([din] + orta + [np.isnan(din[:, 0]).astype(float), np.isnan(din[:, 1]).astype(float)])
    else:
        G = np.column_stack(orta)
    satir = pd.DataFrame({"istasyon_no": stations["istasyon_no"].to_numpy()[st_g],
                          "derinlik": np.asarray(DERINLIKLER)[dd], "tarih": dates[ti][tt],
                          "y_true": F[:, IDX_T].astype(float)})
    return G, satir


class Olcek:
    """Girdi z-skor (egitimle), eksik -> 0; isaret sutunlari olceklenmez. Hedef derinlige gore z-skor."""

    def __init__(self, G, satir, n_isaret=2):
        """n_isaret: sondaki olceklenmeyen isaret sutunu sayisi (uydulu 2; uydusuz ablasyonda 0)."""
        self.mu = np.nanmean(G, 0); self.sd = np.nanstd(G, 0)
        if n_isaret:
            self.mu[-n_isaret:], self.sd[-n_isaret:] = 0.0, 1.0
        self.sd[self.sd == 0] = 1.0
        y = satir.y_true.to_numpy(float); d = satir.derinlik.to_numpy()
        self.ymu = {k: y[d == k].mean() for k in np.unique(d)}
        self.ysd = {k: y[d == k].std() for k in np.unique(d)}

    def x(self, G):
        Z = (G - self.mu) / self.sd
        return np.nan_to_num(Z, nan=0.0).astype(np.float32)

    def y(self, satir):
        d = satir.derinlik.to_numpy()
        return ((satir.y_true.to_numpy(float) - np.vectorize(self.ymu.get)(d)) / np.vectorize(self.ysd.get)(d)).astype(np.float32)

    def geri(self, yz, satir):
        d = satir.derinlik.to_numpy()
        return yz * np.vectorize(self.ysd.get)(d) + np.vectorize(self.ymu.get)(d)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TransformerBlock(nn.Module):
    def __init__(self, d_token, n_heads, ffn_hidden, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(nn.Linear(d_token, ffn_hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_hidden, d_token), nn.Dropout(dropout))

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        return x + self.ffn(self.norm2(x))


class FTTransformer(nn.Module):
    def __init__(self, n_features, d_token, n_heads, n_layers, ffn_factor, dropout):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.blocks = nn.ModuleList([TransformerBlock(d_token, n_heads, int(d_token * ffn_factor), dropout)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(d_token, 1))

    def forward(self, x):
        tokens = torch.cat([self.cls_token.expand(x.size(0), -1, -1), self.tokenizer(x)], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens[:, 0])).squeeze(-1)


def egit_rf(Xtr, ytr, degerlendir, seed, kucuk):
    from sklearn.ensemble import RandomForestRegressor
    parca, agac = (2, 1) if kucuk else (RF_PARCA, RF_PARAMS["n_estimators"] // RF_PARCA)
    toplam = {ad: np.zeros(len(X)) for ad, X in degerlendir.items()}
    for i in range(parca):
        t0 = time.time()
        rf = RandomForestRegressor(n_estimators=agac, max_depth=RF_PARAMS["max_depth"],
                                   min_samples_leaf=RF_PARAMS["min_samples_leaf"], n_jobs=-1,
                                   random_state=seed * 1000 + i)
        rf.fit(Xtr, ytr)
        for ad, X in degerlendir.items():
            toplam[ad] += rf.predict(X)
        del rf
        print(f"  RF parca {i + 1}/{parca} ({agac} agac) {time.time() - t0:.0f} s", flush=True)
    return {ad: v / parca for ad, v in toplam.items()}, []


def egit_ag(model, prm, Xtr, ytr, Xva, yva, degerlendir, seed, kucuk, adam_w, device="cpu", sabit_lr=None):
    """Erken durdurmali egitim; gecmise her epoch'un ogrenme hizi (lr) yazilir. sabit_lr verilirse (katli 2. asama):
    dogrulama ve erken durdurma yok, epoch sayisi = len(sabit_lr), her epoch'un lr'si 1. asamadaki siradan tekrar oynatilir."""
    g = torch.Generator().manual_seed(seed)
    model.to(device)
    Xtr_t, ytr_t = torch.from_numpy(Xtr).to(device), torch.from_numpy(ytr).to(device)

    opt = (torch.optim.AdamW if adam_w else torch.optim.Adam)(model.parameters(), lr=prm["lr"], weight_decay=prm["weight_decay"])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=7, factor=0.5, min_lr=1e-5)
    kayip = nn.MSELoss()
    en_iyi, sabir, durum, hist = float("inf"), 0, None, []
    n_ep = 2 if kucuk else prm["n_epochs"]
    bs = prm["batch_size"]

    def tahmin(X):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(torch.from_numpy(X[i:i + 65536]).to(device)).cpu()
                              for i in range(0, len(X), 65536)]).numpy()

    def bir_epoch():
        model.train()
        sira = torch.randperm(len(Xtr_t), generator=g).to(device)   # uretec CPU'da: cihazdan bagimsiz ayni sira
        top, nb = 0.0, 0
        for i in range(0, len(sira), bs):
            j = sira[i:i + bs]
            opt.zero_grad()
            l = kayip(model(Xtr_t[j]), ytr_t[j])
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            top += l.item(); nb += 1
        return top / nb

    if sabit_lr is not None:
        for ep, lr in enumerate(sabit_lr):
            t0 = time.time()
            for grp in opt.param_groups:
                grp["lr"] = float(lr)
            tr = bir_epoch()
            hist.append(dict(epoch=ep, train=tr, val=np.nan, lr=float(lr), sn=time.time() - t0))
            print(f"  ep {ep:3d}  train {tr:.4f}  lr {lr:.2e}  (sabit {len(sabit_lr)} epoch)  {time.time() - t0:.0f} s", flush=True)
        return {ad: tahmin(X) for ad, X in degerlendir.items()}, hist

    for ep in range(n_ep):
        t0 = time.time()
        lr = opt.param_groups[0]["lr"]
        tr = bir_epoch()
        vl = float(np.mean((tahmin(Xva) - yva) ** 2))
        sch.step(vl)
        hist.append(dict(epoch=ep, train=tr, val=vl, lr=lr, sn=time.time() - t0))
        print(f"  ep {ep:3d}  train {tr:.4f}  val {vl:.4f}  {time.time() - t0:.0f} s", flush=True)
        if vl < en_iyi:
            en_iyi, sabir, durum = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            sabir += 1
            if sabir >= prm["patience"]:
                print(f"  erken durdurma: epoch {ep}", flush=True)
                break
    if durum is None:
        raise RuntimeError("dogrulama kaybi hic sonlu olmadi; en iyi durum yok")
    model.load_state_dict(durum)
    return {ad: tahmin(X) for ad, X in degerlendir.items()}, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["rf", "mlp", "ft_transformer"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kucuk", action="store_true", help="duman testi: kucuk alt kume")
    ap.add_argument("--device", default="cpu", help="MLP / FT-Transformer icin: cpu (varsayilan) ya da cuda")
    ap.add_argument("--master", help="master_dataset_qc.csv yolu (varsayilan SoilConfig.master_path)")
    ap.add_argument("--out", help="cikti klasoru (varsayilan results/tabular_<model>_s<seed>[_fold<k>])")
    ap.add_argument("--kat", type=int, help="uzaysal katli dogrulama: 0-4 (toprak_katlar.py); ic dogrulamayi da acar")
    ap.add_argument("--ic-dogrulama", action="store_true",
                    help="katli 1. asama (MLP/FT): ic egitim istasyonlari, erken durdurma ic dogrulama istasyonlarinin 2023'uyle")
    ap.add_argument("--epoch-kaynak", metavar="HISTORY_CSV",
                    help="katli 2. asama (MLP/FT): 1. asama gecmisinden epoch sayisi (en dusuk val + 1) ve lr sirasi")
    ap.add_argument("--uydusuz", action="store_true", help="uydu ablasyonu: LST/NDVI ve LST eksik isaretleri girdi disi")
    ap.add_argument("--sadece-ic", action="store_true",
                    help="1. asama: yalniz ic_val2023 degerlendirilir (test yok); --ic-dogrulama gerekir")
    a = ap.parse_args()
    if a.sadece_ic and not a.ic_dogrulama:
        raise ValueError("--sadece-ic yalniz --ic-dogrulama ile")
    if a.uydusuz and a.ic_dogrulama and not a.sadece_ic:
        raise ValueError("uydusuz 1. asama yalniz --sadece-ic ile kosulur")
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    cfg = SoilConfig(uydu_derinlik="tum", cikar=("soil_moisture",))
    if a.master:
        cfg.master_path = a.master
    sabit_lr = None
    if a.epoch_kaynak:
        if a.model == "rf" or a.ic_dogrulama:
            raise ValueError("--epoch-kaynak yalniz MLP/FT 2. asamasi icin; --ic-dogrulama ile birlikte kullanilmaz.")
        h = pd.read_csv(a.epoch_kaynak)
        if "lr" not in h or h.val.notna().sum() == 0:
            raise ValueError(f"{a.epoch_kaynak}: 1. asama gecmisi degil (val ya da lr sutunu yok)")
        sabit_lr = h.lr.to_numpy(float)[:int(h.val.idxmin()) + 1]
    if a.model == "rf" and a.ic_dogrulama:
        raise ValueError("RF'de 1. asama yok (erken durdurma yok): katli kosuda dogrudan --kat k ile egitilir.")
    if a.kat is not None:
        cfg.holdout = kat_istasyonlari(a.kat)
        if a.model != "rf" and not a.ic_dogrulama and sabit_lr is None:
            raise ValueError("Katli kosuda asama belirtilmeli: --ic-dogrulama (1. asama) ya da --epoch-kaynak (2. asama).")
    if a.ic_dogrulama:
        cfg.ic_dogrulama = True
    t0 = time.time()
    P = hazirla(cfg)
    idx = P["idx"]
    tr = P["g_tr"].st_global
    hold = P["g_all"].st_global[P["hold_local"]]
    ic = P["ic_global"]
    bol = {"train": (idx["train"], tr), "val": (idx["val"], ic if cfg.ic_dogrulama else tr), "temporal": (idx["test"], tr),
           "spatial_test": (idx["test"], hold), "spatial_full": (np.arange(len(P["dates"])), hold),
           "val2023": (idx["val"], tr), "spatial_val2023": (idx["val"], hold)}
    if cfg.ic_dogrulama:
        bol["ic_val2023"] = (idx["val"], ic)
    if a.sadece_ic:
        bol = {k: v for k, v in bol.items() if k in ("train", "val", "ic_val2023")}
    G, S = {}, {}
    for ad, (ti, st) in bol.items():
        G[ad], S[ad] = tablo(P, ti, st, uydu=not a.uydusuz)
        if a.kucuk:
            sec = np.random.RandomState(0).choice(len(S[ad]), min(len(S[ad]), 4000), replace=False)
            sec.sort()
            G[ad], S[ad] = G[ad][sec], S[ad].iloc[sec].reset_index(drop=True)
    print(f"Tablo: " + ", ".join(f"{ad} {len(S[ad])}" for ad in bol) + f" satir, {G['train'].shape[1]} girdi  [{time.time() - t0:.0f} s]",
          flush=True)
    assert G["train"].shape[1] == len(GIRDI_ADLARI_UYDUSUZ if a.uydusuz else GIRDI_ADLARI)

    olc = Olcek(G["train"], S["train"], n_isaret=0 if a.uydusuz else 2)
    Xtr, ytr = olc.x(G["train"]), olc.y(S["train"])
    Xva, yva = olc.x(G["val"]), olc.y(S["val"])
    adlar = ("ic_val2023",) if a.sadece_ic else ("temporal", "spatial_test", "spatial_full", "val2023",
                                                 "spatial_val2023", "ic_val2023")
    degerlendir = {ad: olc.x(G[ad]) for ad in adlar if ad in G}

    if a.model == "rf":
        tah, hist = egit_rf(Xtr, ytr, degerlendir, a.seed, a.kucuk)
    elif a.model == "mlp":
        m = MLP(Xtr.shape[1], MLP_PARAMS["hidden_dims"], MLP_PARAMS["dropout"])
        tah, hist = egit_ag(m, MLP_PARAMS, Xtr, ytr, Xva, yva, degerlendir, a.seed, a.kucuk, adam_w=False, device=a.device,
                            sabit_lr=sabit_lr)
    else:
        m = FTTransformer(Xtr.shape[1], TR_PARAMS["d_token"], TR_PARAMS["n_heads"], TR_PARAMS["n_layers"],
                          TR_PARAMS["ffn_factor"], TR_PARAMS["dropout"])
        tah, hist = egit_ag(m, TR_PARAMS, Xtr, ytr, Xva, yva, degerlendir, a.seed, a.kucuk, adam_w=True, device=a.device,
                            sabit_lr=sabit_lr)

    etiket = f"{a.model}_s{a.seed}"
    out = (a.out or os.path.join(ANA_DIZIN, "results", f"tabular_{etiket}" + ("" if a.kat is None else f"_fold{a.kat}")
                                  + ("_uydusuz" if a.uydusuz else ""))) + ("_small" if a.kucuk else "")
    os.makedirs(out, exist_ok=True)
    print("Ortam:", ortam_bilgisi(out, cfg.master_path), flush=True)
    st_no = P["stations"]["istasyon_no"].to_numpy()
    en_yakin_gorunur(P["stations"], st_no[hold], np.setdiff1d(st_no, st_no[hold])).assign(kat=-1 if a.kat is None else a.kat).to_csv(
        os.path.join(out, "istasyon_mesafe.csv"), index=False)
    ozet = []
    for ad, yz in tah.items():
        df = S[ad].copy()
        df["y_pred"] = olc.geri(np.asarray(yz, float), df)
        df.to_csv(os.path.join(out, f"pred_{ad}_{etiket}.csv.gz"), index=False)
        station_metrics(df).to_csv(os.path.join(out, f"station_{ad}_{etiket}.csv"), index=False)
        mt = metrics(df).assign(split=ad, model=a.model, seed=a.seed)
        ozet.append(mt)
        print(f"  {ad:13s} RMSE {mt.iloc[0].rmse:.3f}  MBE {mt.iloc[0].mbe:+.3f}  n={int(mt.iloc[0].n)}", flush=True)
    pd.concat(ozet, ignore_index=True).to_csv(os.path.join(out, "metrics.csv"), index=False)
    if hist:
        pd.DataFrame(hist).to_csv(os.path.join(out, f"history_{etiket}.csv"), index=False)
    print(f"Bitti: {out}  [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
