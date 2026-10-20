import copy
import os
import time

import numpy as np
import torch

from ortak_ayarlar import Config
from ortak_veri import Prepared


def masked_mse(pred, batch):
    """Yalniz yapay maskelenen hucrelerde MSE (normalize uzay). Orijinal NaN kayba girmez."""
    m = batch.art_mask
    if not bool(m.any()):
        return None
    return ((pred[m] - batch.target[m]) ** 2).mean()


def duzeltme_cezasi(pred, batch):
    """Hibrit: ortalama((cikti - taban)^2) yapay maskeli hucrelerde (normalize birim; taban NaN -> 0, modeldeki gibi)."""
    m = batch.art_mask
    taban = torch.nan_to_num(batch.base, nan=0.0)
    return ((pred[m] - taban[m]) ** 2).mean()


def run_epoch(model, wd, cfg: Config, opt=None):
    """Egitimde kayip = maskeli MSE (+ cfg.ceza * duzeltme cezasi, varsa); dogrulamada yalniz maskeli MSE."""
    train = opt is not None
    ceza = float(getattr(cfg, "ceza", 0.0))
    model.train(train)
    top, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in wd.batches(shuffle=train):
            batch.to(cfg.device)
            pred = model(batch)
            loss = masked_mse(pred, batch)
            if loss is None:
                continue
            if train and ceza > 0:
                loss = loss + ceza * duzeltme_cezasi(pred, batch)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            k = int(batch.art_mask.sum())
            top += float(loss) * k
            n += k
    return top / max(n, 1)


def fit(model, prep: Prepared, cfg: Config, seed: int, verbose=False, sabit_epoch=None):
    """Erken durdurmali egitim (en iyi dogrulama durumu geri yuklenir). sabit_epoch verilirse erken durdurma ve dogrulama
    kumesi yok: tam o kadar epoch egitilir, son durum kalir (katli 2. asama; epoch sayisi 1. asamada secilir)."""
    torch.manual_seed(seed)
    model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    if sabit_epoch is not None:
        hist = []
        for ep in range(int(sabit_epoch)):
            t0 = time.time()
            tr = run_epoch(model, prep.splits["train"], cfg, opt)
            hist.append({"epoch": ep, "train_loss": tr, "val_loss": float("nan"), "sec": time.time() - t0})
            if verbose:
                print(f"  ep {ep:3d}  train {tr:.4f}  (sabit {sabit_epoch} epoch)  {time.time() - t0:.0f} s", flush=True)
        return hist
    best, best_state, bekle, hist = np.inf, None, 0, []
    sifir = bool(getattr(cfg, "epoch_sifir", False))
    kayit = getattr(cfg, "epoch_kayit", "")
    if kayit:
        if not sifir:
            raise ValueError("epoch_kayit yalniz epoch_sifir ile")
        os.makedirs(kayit, exist_ok=True)
        kaydet = lambda e: torch.save(model.state_dict(), os.path.join(kayit, f"epoch_{e:03d}.pt"))
    if sifir:                    # egitimden once ayni dogrulama dongusu: egitilen_epoch = 0 (hibritte taban)
        t0 = time.time()
        va0 = run_epoch(model, prep.splits["val"], cfg)
        hist.append({"epoch": -1, "train_loss": float("nan"), "val_loss": va0, "sec": time.time() - t0, "egitilen_epoch": 0})
        best, best_state = va0, copy.deepcopy(model.state_dict())
        if kayit:
            kaydet(0)
        if verbose:
            print(f"  ep  -1  (egitimsiz)  val {va0:.4f}", flush=True)
    for ep in range(cfg.max_epochs):
        t0 = time.time()
        tr = run_epoch(model, prep.splits["train"], cfg, opt)
        va = run_epoch(model, prep.splits["val"], cfg)
        if kayit:
            kaydet(ep + 1)
        hist.append({"epoch": ep, "train_loss": tr, "val_loss": va, "sec": time.time() - t0,
                     **({"egitilen_epoch": ep + 1} if sifir else {})})
        if verbose:
            print(f"  ep {ep:3d}  train {tr:.4f}  val {va:.4f}  {time.time() - t0:.0f} s", flush=True)
        if va < best - 1e-12:
            best, best_state, bekle = va, copy.deepcopy(model.state_dict()), 0
        else:
            bekle += 1
            if bekle >= cfg.patience:
                break
    if best_state is None:
        raise RuntimeError("dogrulama kaybi hic sonlu olmadi; en iyi durum yok")
    model.load_state_dict(best_state)
    return hist
