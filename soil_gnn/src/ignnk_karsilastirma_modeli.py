import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from idw_karsilastirma_modeli import rakim_dogrusu
from toprak_ayarlar import DERINLIKLER, IDX_T, SoilConfig, ortam_bilgisi
from toprak_degerlendirme import metrics, station_metrics
from toprak_katlar import en_yakin_gorunur, kat_istasyonlari
from toprak_gnn_calistir import hazirla
from ortak_veri import haversine_km

pd.set_option("compute.use_numexpr", False)

ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = 3                       # pencere (gun)
Z = 64
K_DIF = 1
YIGIN = 16
MASKE_ORANI = 0.2
SIGMA_ADAY = (50.0, 100.0, 200.0, 400.0, 800.0)
LR, MAX_EPOCH, SABIR = 1e-3, 200, 10
ANAHTAR = ["istasyon_no", "derinlik", "tarih"]


class DGCN(nn.Module):

    def __init__(self, din, dout, K):
        super().__init__()
        self.K = K
        self.th_b = nn.ModuleList([nn.Linear(din, dout, bias=False) for _ in range(K)])
        self.th_f = nn.ModuleList([nn.Linear(din, dout, bias=False) for _ in range(K)])

    @staticmethod
    def cheb(Wn, Hx, K):
        """[T_1(Wn) H, ..., T_K(Wn) H]; Wn (B, n, n), H (B, n, d)."""
        out, onceki, simdi = [], Hx, torch.bmm(Wn, Hx)
        out.append(simdi)
        for _ in range(2, K + 1):
            onceki, simdi = simdi, 2 * torch.bmm(Wn, simdi) - onceki
            out.append(simdi)
        return out

    def forward(self, Hx, Wf, Wb):
        tf, tb = self.cheb(Wf, Hx, self.K), self.cheb(Wb, Hx, self.K)
        return sum(self.th_b[k](tf[k]) + self.th_f[k](tb[k]) for k in range(self.K))


class IGNNK(nn.Module):
    def __init__(self, h=H, z=Z, K=K_DIF):
        super().__init__()
        self.l0, self.l1, self.l2 = DGCN(h, z, K), DGCN(z, z, K), DGCN(z, h, K)

    @staticmethod
    def gecis(W):
        Wt = W.transpose(1, 2)
        return W / W.sum(-1, keepdim=True), Wt / Wt.sum(-1, keepdim=True)

    def forward(self, H0, W):
        Wf, Wb = self.gecis(W)
        H1 = self.l0(H0, Wf, Wb)
        H2 = torch.relu(self.l1(H1, Wf, Wb)) + H1
        return self.l2(H2, Wf, Wb)


def komsuluk(D, sigma):
    return np.exp(-(D / sigma) ** 2)


def dogru(Tw, rak):
    """Tw (gun, n) olcum, rak (n,) km -> a, b (gun,); < 3 olcumde a = b = 0 (IDW betigiyle ayni geri donus)."""
    a, b = rakim_dogrusu(Tw, rak)
    yok = ~np.isfinite(b)
    return np.where(yok, 0.0, a), np.where(yok, 0.0, b)


class Veri:
    """Bir derinligin gunluk T'si (gun, istasyon), rakim (km) ve uzaklik matrisi (km)."""

    def __init__(self, P, di):
        st = P["stations"]
        n_gun, n_st = len(P["dates"]), len(st)
        self.T = P["X"][..., IDX_T].reshape(n_gun, n_st, len(DERINLIKLER))[:, :, di].astype(float)
        self.rak = st.rakim.to_numpy(float) / 1000.0
        la, lo = st.lat.to_numpy(float), st.lon.to_numpy(float)
        self.D = haversine_km(la[:, None], lo[:, None], la[None, :], lo[None, :])
        np.fill_diagonal(self.D, 0.0)


def ornek_yigini(V, gunler, baslar, egitim, rng, sigma):
    n = len(egitim)
    n_s = rng.randint(int(math.ceil(n / 2)), n + 1)
    n_m = int(round(MASKE_ORANI * n_s))
    n_o = n_s - n_m
    H0 = np.zeros((len(baslar), n_s, H), np.float32)
    R = np.full((len(baslar), n_s, H), np.nan, np.float32)
    W = np.zeros((len(baslar), n_s, n_s), np.float32)
    for i, j in enumerate(baslar):
        sec = egitim[rng.permutation(n)[:n_s]]                  # ilk n_o gozlenen, kalan n_m maskeli
        g = gunler[j:j + H]
        Tw = V.T[np.ix_(g, sec)]                                # (h, n_s)
        a, b = dogru(Tw[:, :n_o], V.rak[sec[:n_o]])
        r = (Tw - (a[:, None] + b[:, None] * V.rak[sec][None, :])).T          # (n_s, h)
        R[i] = r
        H0[i, :n_o] = np.nan_to_num(r[:n_o], nan=0.0)
        W[i] = komsuluk(V.D[np.ix_(sec, sec)], sigma)
    return H0, R, W


def pencere_baslari(n_gun):
    """Ortusmeyen 3 gunluk pencereler; son pencere donem sonuna hizali."""
    if n_gun < H:
        raise ValueError(f"donem {n_gun} gun, pencere {H}")
    b = list(range(0, n_gun - H + 1, H))
    if b[-1] + H < n_gun:
        b.append(n_gun - H)
    return np.array(b)


@torch.no_grad()
def tahmin(model, V, gunler, gorunur, hedef, sigma, device):
    """Hedef istasyonlarin (gun, hedef) tahmini; gorunur istasyonlar girdide, hedefler 0. Her gun bir kez doldurulur."""
    dugum = np.concatenate([gorunur, hedef])
    ng = len(gorunur)
    W = torch.as_tensor(komsuluk(V.D[np.ix_(dugum, dugum)], sigma), dtype=torch.float32, device=device)
    baslar = pencere_baslari(len(gunler))
    Tg = V.T[np.ix_(gunler, gorunur)]
    a, b = dogru(Tg, V.rak[gorunur])
    R = Tg - (a[:, None] + b[:, None] * V.rak[gorunur][None, :])
    out = np.full((len(gunler), len(hedef)), np.nan)
    model.eval()
    for p0 in range(0, len(baslar), 512):
        bb = baslar[p0:p0 + 512]
        H0 = np.zeros((len(bb), len(dugum), H), np.float32)
        for i, j in enumerate(bb):
            H0[i, :ng] = np.nan_to_num(R[j:j + H].T, nan=0.0)
        Xh = model(torch.as_tensor(H0, device=device), W.expand(len(bb), -1, -1)).cpu().numpy()   # (B, n, h)
        for i, j in enumerate(bb):
            yaz = np.isnan(out[j:j + H, 0])
            out[j:j + H][yaz] = Xh[i, ng:, :].T[yaz]
    if np.isnan(out).any():
        raise RuntimeError("tahminsiz gun")
    return out + a[:, None] + b[:, None] * V.rak[hedef][None, :]


def satirlar(V, P, gunler, hedef, Y, derinlik):
    """Olculmus hedef hucreleri (diger modellerle ayni satirlar)."""
    yt = V.T[np.ix_(gunler, hedef)]
    tt, jj = np.nonzero(np.isfinite(yt))
    return pd.DataFrame({"istasyon_no": P["stations"].istasyon_no.to_numpy()[hedef][jj], "derinlik": derinlik,
                         "tarih": P["dates"][gunler][tt], "y_true": yt[tt, jj], "y_pred": Y[tt, jj]})


def egit(V, gunler, egitim, sigma, seed, device, n_epoch, dogrulama=None, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = IGNNK().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    baslar = np.arange(len(gunler) - H + 1)
    gecmis, en_iyi, en_iyi_durum, bekle = [], np.inf, None, 0
    for ep in range(n_epoch):
        t0 = time.time()
        model.train()
        sira = rng.permutation(baslar)
        top, say = 0.0, 0
        for p0 in range(0, len(sira), YIGIN):
            H0, R, W = ornek_yigini(V, gunler, sira[p0:p0 + YIGIN], egitim, rng, sigma)
            R = torch.as_tensor(R, device=device)
            m = torch.isfinite(R)
            if not bool(m.any()):
                continue
            Xh = model(torch.as_tensor(H0, device=device), torch.as_tensor(W, device=device))
            loss = ((Xh[m] - R[m]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            top += float(loss) * int(m.sum())
            say += int(m.sum())
        g = dict(epoch=ep, train_mse=top / max(say, 1), val_mse=np.nan, sn=0.0)
        if dogrulama is not None:
            dg, gor, hed = dogrulama
            Y = tahmin(model, V, dg, gor, hed, sigma, device)
            yt = V.T[np.ix_(dg, hed)]
            ok = np.isfinite(yt)
            g["val_mse"] = float(np.mean((Y[ok] - yt[ok]) ** 2))
            if g["val_mse"] < en_iyi - 1e-12:
                en_iyi, en_iyi_durum, bekle = g["val_mse"], {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                bekle += 1
        g["sn"] = time.time() - t0
        gecmis.append(g)
        if verbose:
            print(f"    ep {ep:3d} train {g['train_mse']:.4f} val {g['val_mse']:.4f} {g['sn']:.1f} s", flush=True)
        if dogrulama is not None and bekle >= SABIR:
            break
    if dogrulama is not None:
        if en_iyi_durum is None:
            raise RuntimeError("dogrulama MSE'si hic sonlu olmadi")
        model.load_state_dict(en_iyi_durum)
    return model, gecmis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kat", type=int, required=True, help="0-4 (toprak_katlar.py)")
    ap.add_argument("--ic-dogrulama", action="store_true", help="1. asama: sigma ve epoch secimi")
    ap.add_argument("--secim-kaynak", metavar="SECIM_CSV", help="2. asama: 1. asamanin secim_ignnk.csv'si")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--master")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kucuk", action="store_true", help="duman testi: 2 epoch, yalniz 5 cm, 1. asamada bir sigma (donem ayni)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.ic_dogrulama == bool(a.secim_kaynak):
        raise ValueError("Asama belirtilmeli: --ic-dogrulama (1. asama) ya da --secim-kaynak (2. asama), biri.")
    cfg = SoilConfig()
    if a.master:
        cfg.master_path = a.master
    cfg.holdout = kat_istasyonlari(a.kat)
    cfg.ic_dogrulama = a.ic_dogrulama
    cfg.__post_init__()
    out = a.out or os.path.join(ANA_DIZIN, "results", ("fold_selection/ignnk_fold%d" % a.kat) if a.ic_dogrulama
                                else ("folds/ignnk_fold%d_s%d" % (a.kat, a.seed)))
    if a.kucuk:
        out += "_small"
    os.makedirs(out, exist_ok=True)
    print("Ortam:", ortam_bilgisi(out, cfg.master_path), flush=True)
    t0 = time.time()
    P = hazirla(cfg)
    idx = P["idx"]
    tr = P["g_tr"].st_global
    hold = P["g_all"].st_global[P["hold_local"]]
    ic = P["ic_global"]
    if len(np.intersect1d(tr, hold)) or len(np.intersect1d(tr, ic)):
        raise RuntimeError("istasyon bolmesi")
    derinlikler = [d for d in DERINLIKLER if d > 0]
    if a.kucuk:
        derinlikler = derinlikler[:1]
    print(f"kat {a.kat}: egitim {len(tr)} ist., hedef {len(hold)}, ic {len(ic)}  [{time.time() - t0:.0f} s]", flush=True)
    gecmis, tah = [], {}
    if a.ic_dogrulama:
        secim = []
        sigmalar = SIGMA_ADAY[:1] if a.kucuk else SIGMA_ADAY
        for d in derinlikler:
            V = Veri(P, DERINLIKLER.index(d))
            en = None
            for sg in sigmalar:
                m, g = egit(V, idx["train"], tr, sg, a.seed, a.device, 2 if a.kucuk else MAX_EPOCH,
                            dogrulama=(idx["val"], tr, ic), verbose=a.verbose)
                gecmis += [dict(derinlik=d, sigma=sg, **x) for x in g]
                h = pd.DataFrame(g)
                e = int(h.val_mse.idxmin())
                print(f"  {d} cm sigma {sg:.0f} km: en iyi epoch {e} val MSE {h.val_mse[e]:.4f} ({len(h)} epoch)  "
                      f"[{time.time() - t0:.0f} s]", flush=True)
                if en is None or h.val_mse[e] < en[0]:
                    en = (h.val_mse[e], sg, e, m)
            secim.append(dict(kat=a.kat, derinlik=d, sigma=en[1], en_iyi_epoch=en[2], epoch_sayisi=en[2] + 1, val_mse=en[0]))
            Y = tahmin(en[3], V, idx["val"], tr, ic, en[1], a.device)
            tah.setdefault("ic_val2023", []).append(satirlar(V, P, idx["val"], ic, Y, d))
        pd.DataFrame(secim).to_csv(os.path.join(out, "secim_ignnk.csv"), index=False)
    else:
        sec = pd.read_csv(a.secim_kaynak).set_index("derinlik")
        if int(sec.kat.iloc[0]) != a.kat:
            raise ValueError("secim dosyasi baska kata ait")
        gr = [tr[g_] for g_ in P["groups"]]
        tum = np.arange(len(P["dates"]))
        for d in derinlikler:
            V = Veri(P, DERINLIKLER.index(d))
            sg, ne = float(sec.loc[d, "sigma"]), (2 if a.kucuk else int(sec.loc[d, "epoch_sayisi"]))
            m, g = egit(V, idx["train"], tr, sg, a.seed, a.device, ne, verbose=a.verbose)
            gecmis += [dict(derinlik=d, sigma=sg, **x) for x in g]
            for bol, gunler in (("spatial_test", idx["test"]), ("spatial_full", tum), ("spatial_val2023", idx["val"])):
                tah.setdefault(bol, []).append(satirlar(V, P, gunler, hold, tahmin(m, V, gunler, tr, hold, sg, a.device), d))
            for bol, gunler in (("temporal", idx["test"]), ("val2023", idx["val"])):
                for h_ in gr:
                    gor = np.setdiff1d(tr, h_)
                    tah.setdefault(bol, []).append(satirlar(V, P, gunler, h_, tahmin(m, V, gunler, gor, h_, sg, a.device), d))
            print(f"  {d} cm sigma {sg:.0f} km, {ne} epoch  [{time.time() - t0:.0f} s]", flush=True)
        st_no = P["stations"].istasyon_no.to_numpy()
        en_yakin_gorunur(P["stations"], st_no[hold], st_no[tr]).assign(kat=a.kat).to_csv(
            os.path.join(out, "istasyon_mesafe.csv"), index=False)
    etiket = f"ignnk_s{a.seed}"
    pd.DataFrame(gecmis).to_csv(os.path.join(out, f"history_{etiket}.csv"), index=False)
    ozet = []
    for bol, parca in tah.items():
        df = pd.concat(parca, ignore_index=True).sort_values(ANAHTAR).reset_index(drop=True)
        if df.duplicated(ANAHTAR).any() or df.y_pred.isna().any():
            raise RuntimeError(f"{bol}: tekrarli ya da tahminsiz satir")
        df.to_csv(os.path.join(out, f"pred_{bol}_{etiket}.csv.gz"), index=False)
        station_metrics(df).to_csv(os.path.join(out, f"station_{bol}_{etiket}.csv"), index=False)
        ozet.append(metrics(df).assign(split=bol, model="ignnk", seed=a.seed))
    pd.concat(ozet, ignore_index=True).to_csv(os.path.join(out, "metrics.csv"), index=False)
    print(f"Bitti: {out}  [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
