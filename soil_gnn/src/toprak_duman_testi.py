import os

import numpy as np
import pandas as pd
import torch

from ortak_ayarlar import VARIANTS, variant_opts
from gnn_model import RELATIONS, build_model
from toprak_ayarlar import (DERINLIKLER, EDGE_DIM, IDX_T, N_FEAT, OZNITELIK, SoilConfig, edge_dims)
from toprak_veri_ve_graf import Splits, SoilWindows, norm_stats, normalize, synthetic_master
from toprak_degerlendirme import metrics
from toprak_gnn_calistir import hazirla, kos
from gnn_egitim import masked_mse

SONUC = []


def kontrol(ad, kosul, detay=""):
    SONUC.append((ad, bool(kosul)))
    print(f"[{'OK ' if kosul else 'HATA'}] {ad}" + (f"  ({detay})" if detay else ""))


def nan_esit(a, b):
    return np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)])


df, hold = synthetic_master(n_st=8, holdout=(3, 7))
cfg = SoilConfig(train_years=(2021,), val_years=(2022,), test_years=(2023,), holdout=hold,
                 horiz_radius_km=300, max_epochs=2, batch_size=4, eval_groups=3,
                 cikar=())
P = hazirla(cfg, df)
X, dates, st, g_tr, g_all = P["X"], P["dates"], P["stations"], P["g_tr"], P["g_all"]
print(f"X {X.shape}, egitim grafi {g_tr.n_st} ist, tum graf {g_all.n_st} ist, holdout {hold}\n")

print("== Veri ==")
Xr = X.reshape(len(dates), len(st), len(DERINLIKLER), N_FEAT)
kontrol("yuzey dugumunde T yok", np.isnan(Xr[:, :, 0, IDX_T]).all())
kontrol("derinlik dugumlerinde uydu girdisi yok", np.isnan(Xr[:, :, 1:, 1:5]).all())
r = df[(df.istasyon_no == st.istasyon_no[0]) & (df.tarih == dates[0])].iloc[0]
kontrol("T derinlik eslemesi (20 cm)", np.isclose(Xr[0, 0, 3, IDX_T], r["t20cmsicaklik"]))
kontrol("SoilGrids katman eslemesi (50 cm -> 30-60cm clay)",
        np.isclose(Xr[0, 0, 4, OZNITELIK.index("clay")], r["clay_30-60cm"]))
Xu = hazirla(SoilConfig(**{**cfg.__dict__, "uydu_derinlik": "tum"}), df)["X"].reshape(Xr.shape)
kontrol("uydu=tum: her derinlik yuzeyle ayni uydu girdisi",
        all(nan_esit(Xu[:, :, d, 1:5], Xr[:, :, 0, 1:5]) for d in range(len(DERINLIKLER))))
kontrol("uydu=tum: T/SoilGrids/rakim degismiyor", nan_esit(np.delete(Xu, [1, 2, 3, 4], -1), np.delete(Xr, [1, 2, 3, 4], -1)))
Xc = hazirla(SoilConfig(**{**cfg.__dict__, "cikar": ("soil_moisture",)}), df)["X"].reshape(Xr.shape)
i_sm = OZNITELIK.index("soil_moisture")
kontrol("cikar=soil_moisture: sutun tamamen NaN, digerleri ayni",
        np.isnan(Xc[..., i_sm]).all() and nan_esit(np.delete(Xc, i_sm, -1), np.delete(Xr, i_sm, -1)))
kontrol("satir sirasi tensoru degistirmiyor",
        nan_esit(X, hazirla(cfg, df.sample(frac=1, random_state=1))["X"]))

print("\n== Graf ==")
kontrol("holdout egitim grafinda yok", not set(hold) & set(g_tr.stations.istasyon_no))
kontrol("holdout tum grafta var", set(hold) <= set(g_all.stations.istasyon_no))
(v, va), (h, ha) = g_tr.templates["V"], g_tr.templates["H"]
z = np.asarray(DERINLIKLER, float)
kontrol("dusey: ayni istasyon, komsu derinlik, |dz|/100",
        (g_tr.slot_station[v[0]] == g_tr.slot_station[v[1]]).all()
        and (np.abs(g_tr.slot_depth[v[0]] - g_tr.slot_depth[v[1]]) == 1).all()
        and np.allclose(va[:, 0], np.abs(z[g_tr.slot_depth[v[0]]] - z[g_tr.slot_depth[v[1]]]) / 100))
kontrol("yatay: ayni derinlik, farkli istasyon",
        (g_tr.slot_depth[h[0]] == g_tr.slot_depth[h[1]]).all() and (g_tr.slot_station[h[0]] != g_tr.slot_station[h[1]]).all())
for k, (e, _) in g_tr.templates.items():
    c = set(map(tuple, e.T))
    kontrol(f"{k} cift yonlu", all((b, a) in c for a, b in c))

kontrol("yatay_rakim kapali: H 2 oznitelik (eski davranis)", ha.shape[1] == 2 and edge_dims(cfg)["H"] == 2)
from dataclasses import replace as _rep
from toprak_veri_ve_graf import SoilGraph
cfg_r = _rep(cfg, yatay_rakim=True)
g_r = SoilGraph(g_tr.stations, g_tr.st_global, cfg_r)
hr, har = g_r.templates["H"]
_rk = g_tr.stations["rakim"].to_numpy(float)
kontrol("yatay_rakim acik: ayni kenarlar, ilk 2 oznitelik ayni",
        np.array_equal(hr, h) and np.allclose(har[:, :2], ha))
kontrol("yatay_rakim acik: 3. oznitelik = (rakim kaynak - hedef)/1000 km",
        har.shape[1] == 3 and edge_dims(cfg_r)["H"] == 3
        and np.allclose(har[:, 2], (_rk[g_tr.slot_station[h[0]]] - _rk[g_tr.slot_station[h[1]]]) / 1000.0))

g_d = SoilGraph(g_tr.stations, g_tr.st_global, _rep(cfg, dikey_tam=True))
vd_, vda = g_d.templates["V"]
kontrol("dikey_tam: ayni istasyon, her derinlik tum diger derinliklere, |dz|/100",
        (g_tr.slot_station[vd_[0]] == g_tr.slot_station[vd_[1]]).all()
        and vd_.shape[1] == g_tr.n_st * len(DERINLIKLER) * (len(DERINLIKLER) - 1)
        and len(set(map(tuple, vd_.T))) == vd_.shape[1]
        and np.allclose(vda[:, 0], np.abs(z[g_tr.slot_depth[vd_[0]]] - z[g_tr.slot_depth[vd_[1]]]) / 100))
kontrol("dikey_tam: yatay kenarlar degismiyor", np.array_equal(g_d.templates["H"][0], h))

print("\n== Normalizasyon ==")
mu, sd = norm_stats(X[P["idx"]["train"]][:, g_tr.slot_global], g_tr.slot_depth, "level")
tr_ist = set(g_tr.stations.istasyon_no)
ref = df[df.istasyon_no.isin(tr_ist) & (df.tarih.dt.year == 2021)]["t50cmsicaklik"]
kontrol("mu(50 cm, T) yalniz egitim istasyonu + egitim yili", np.isclose(mu[4, IDX_T], ref.mean(), atol=1e-4),
        f"{mu[4, IDX_T]:.4f} vs {ref.mean():.4f}")
kontrol("derinlige duyarli: mu(T) derinlige gore farkli", len(set(np.round(mu[1:, IDX_T], 3))) == 5,
        str(np.round(mu[:, IDX_T], 2)))

print("\n== Degismezler ==")
sub = lambda ti, g: X[ti][:, g.slot_global]
tr_w = SoilWindows(normalize(sub(P["idx"]["train"], g_tr), g_tr, mu, sd), dates[P["idx"]["train"]], g_tr, cfg, "train", 0)
ev_w = SoilWindows(normalize(sub(P["idx"]["test"], g_all), g_all, mu, sd), dates[P["idx"]["test"]], g_all, cfg,
                   "eval", 0, masked_stations=P["hold_local"])
for ad, w in (("train", tr_w), ("eval", ev_w)):
    wl = [w.window(i) for i in range(min(4, len(w)))]
    from toprak_veri_ve_graf import collate
    b = collate(wl, w.W, w.S, w.g.templates)
    x, tg, m = b.x.numpy(), b.target.numpy(), b.art_mask.numpy()
    B, W, S = b.B, b.W, b.S
    kontrol(f"{ad}: 1) B*W*S == N", B * W * S == x.shape[0], f"B={B} W={W} S={S}")
    kontrol(f"{ad}: 2) maskeli hucrede target gercek", np.isfinite(tg[m]).all() and m.sum() > 0, f"{int(m.sum())} hucre")
    xt = x[:, IDX_T:IDX_T + 1]
    kontrol(f"{ad}: 3) maskesiz T girdisi == target", nan_esit(xt[~m & np.isfinite(tg)], tg[~m & np.isfinite(tg)]))
    tt = b.node_t.numpy().reshape(B, W, S)
    kontrol(f"{ad}: 5) duzen (B, W, S)", (tt == tt[:, :, :1]).all() and (b.node_slot.numpy().reshape(B, W, S) == np.arange(S)).all())
    ok6 = all(nan_esit(tg.reshape(B, W, S)[i, j], w.Xn[tt[i, j, 0], :, IDX_T]) for i in range(B) for j in range(W))
    kontrol(f"{ad}: 6) target kaynakla birebir", ok6)
    sizinti = True; cov = True
    for i, wd in enumerate(wl):
        node_m = wd["st_mask"][w.g.slot_station]
        xi = x.reshape(B, W, S, N_FEAT)[i]
        sizinti &= np.isnan(xi[:, node_m, IDX_T]).all()
        cov &= nan_esit(xi[:, :, 1:], wd["x"][:, :, 1:]) and nan_esit(wd["x"][:, :, 1:], w.Xn[tt[i, :, 0]][:, :, 1:])
    kontrol(f"{ad}: maskeli istasyonun T'si hic gorunmuyor", sizinti)
    kontrol(f"{ad}: ortak degiskenler (uydu, toprak, rakim) maskelenmiyor", cov)
kontrol("eval: 4) dugum tutarli maske (sabit istasyon kumesi)",
        all(np.array_equal(ev_w.window(i)["st_mask"], ev_w.window(0)["st_mask"]) for i in range(len(ev_w))))
kontrol("eval: maskeli istasyonlar = holdout",
        set(g_all.stations.istasyon_no[ev_w.window(0)["st_mask"]]) == set(hold))
kontrol("train: maske her cagrida degisir",
        any(not np.array_equal(tr_w.window(0)["st_mask"], tr_w.window(0)["st_mask"]) for _ in range(5)))
s0 = np.random.get_state()[1][:3].copy(); tr_w.window(1)
kontrol("train maskesi global RNG'ye dokunmuyor", np.array_equal(s0, np.random.get_state()[1][:3]))
dh = np.concatenate([bb.edge_attr["T"][:, 0].numpy() for bb in tr_w.batches(False)])
kontrol("zamansal kenar gun farki (boslukta 2.0)", set(np.unique(dh)) == {1.0, 2.0}, str(np.unique(dh)))

print("\n== Gradyanlar ==")
bb = next(tr_w.batches(False))
for vad in ("gnn", "gnn_no_vertical", "gnn_no_horizontal", "gnn_fixed_fusion", "flat_graphsage"):
    o = variant_opts(vad)
    torch.manual_seed(0)
    mdl = build_model(cfg, o, in_dim=N_FEAT, target_cols=[IDX_T], edge_dims=EDGE_DIM)
    loss = masked_mse(mdl(bb), bb); loss.backward()
    kontrol(f"{vad}: kayip sonlu, NaN gradyan yok",
            torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in mdl.parameters()),
            f"loss={float(loss):.3f}")
    if o["flat"]:
        continue
    for l in range(cfg.n_layers):
        a = mdl.effective_edge_weights(l).detach(); g = mdl.fusion_w[l].grad
        ok = torch.isclose(a.sum(), torch.tensor(1.0)) and torch.isfinite(a).all()
        if o["fixed_fusion"]:
            ok = ok and g is None
        for k in o["drop"]:
            i = RELATIONS.index(k)
            ok = ok and float(a[i]) == 0 and float(g[i]) == 0 and \
                all(p.grad is None for p in mdl.convs[l][k].parameters())
        kontrol(f"{vad} L{l}: alpha={[round(float(q), 3) for q in a]}", ok)

print("\n== Eksik-deger gostergesi ==")
from toprak_ayarlar import MASKE_VARSAYILAN
mc = [OZNITELIK.index(c) for c in MASKE_VARSAYILAN]
o = variant_opts("gnn")
torch.manual_seed(0); m0 = build_model(cfg, o, in_dim=N_FEAT, target_cols=[IDX_T], edge_dims=EDGE_DIM)
kontrol("maske kapali: giris boyutu degismedi", m0.inp.in_features == N_FEAT)
torch.manual_seed(0); m1 = build_model(cfg, o, in_dim=N_FEAT, target_cols=[IDX_T], edge_dims=EDGE_DIM, maske_cols=mc)
kontrol("maske acik: giris boyutu + 3", m1.inp.in_features == N_FEAT + 3)
m1.eval(); m0.eval()
with torch.no_grad():
    b1 = next(tr_w.batches(False))
    lst = OZNITELIK.index("LST_Day_C")
    i = int(torch.nonzero(~torch.isnan(b1.x[:, lst]))[0])
    xa = b1.x.clone(); xa[i, lst] = float("nan")
    xb = b1.x.clone(); xb[i, lst] = 0.0
    b1.x = xa; ya = m1(b1); b1.x = xb; yb = m1(b1)
    kontrol("maske acik: eksik LST ile ortalama LST farkli cikti verir", not torch.allclose(ya, yb))
    kontrol("maske acik: cikti sonlu", torch.isfinite(ya).all())
    b1.x = xa; za = m0(b1); b1.x = xb; zb = m0(b1)
    kontrol("maske kapali: eksik LST ile ortalama LST ayni (eski davranis)", torch.allclose(za, zb))
m1.train()
loss = masked_mse(m1(bb), bb); loss.backward()
torch.manual_seed(0)
P_r = hazirla(cfg_r, df)
_w = SoilWindows(normalize(P_r["X"][P_r["idx"]["train"]][:, P_r["g_tr"].slot_global], P_r["g_tr"], mu, sd),
                 P_r["dates"][P_r["idx"]["train"]], P_r["g_tr"], cfg_r, "train", 0)
_bb = next(_w.batches(False))
m_r = build_model(cfg_r, variant_opts("gnn"), in_dim=N_FEAT, target_cols=[IDX_T], edge_dims=edge_dims(cfg_r))
_l = masked_mse(m_r(_bb), _bb); _l.backward()
kontrol("yatay_rakim acik: batch H ozniteligi 3 sutun, kayip ve gradyan sonlu",
        _bb.edge_attr["H"].shape[1] == 3 and torch.isfinite(_l)
        and all(p.grad is None or torch.isfinite(p.grad).all() for p in m_r.parameters()), f"loss={float(_l):.3f}")
P_d = hazirla(_rep(cfg, dikey_tam=True), df)
_wd = SoilWindows(normalize(P_d["X"][P_d["idx"]["train"]][:, P_d["g_tr"].slot_global], P_d["g_tr"], mu, sd),
                  P_d["dates"][P_d["idx"]["train"]], P_d["g_tr"], _rep(cfg, dikey_tam=True), "train", 0)
_bd = next(_wd.batches(False))
torch.manual_seed(0)
m_d = build_model(cfg, variant_opts("gnn"), in_dim=N_FEAT, target_cols=[IDX_T], edge_dims=edge_dims(cfg))
_ld = masked_mse(m_d(_bd), _bd); _ld.backward()
kontrol("dikey_tam: kayip ve gradyan sonlu",
        torch.isfinite(_ld) and all(p.grad is None or torch.isfinite(p.grad).all() for p in m_d.parameters()),
        f"loss={float(_ld):.3f}, V kenar {_bd.edge_index['V'].shape[1]}")
kontrol("maske acik: kayip ve gradyan sonlu",
        torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in m1.parameters()))
try:
    SoilConfig(maske=("NDVI_yok",)); kontrol("gecersiz maske adi reddedilir", False)
except ValueError:
    kontrol("gecersiz maske adi reddedilir", True)
try:
    SoilConfig(maske=("LST_Day_C",), cikar=("LST_Day_C",)); kontrol("cikarilan oznitelige maske reddedilir", False)
except ValueError:
    kontrol("cikarilan oznitelige maske reddedilir", True)

print("\n== Hibrit: IDW_rakim tabani ==")
from idw_karsilastirma_modeli import idw as _idw, rakim_dogrusu as _rd
from ortak_veri import haversine_km as _hv
cfg_h = _rep(cfg, idw_artik=True)
kontrol("idw_artik kapali: pencerede base yok, Batch.base None",
        "base" not in tr_w.window(0) and next(tr_w.batches(False)).base is None)
Xn_ev = normalize(sub(P["idx"]["test"], g_all), g_all, mu, sd)
_ev = lambda Xn_: SoilWindows(Xn_, dates[P["idx"]["test"]], g_all, cfg_h, "eval", 0, masked_stations=P["hold_local"])
ev_h = _ev(Xn_ev)
tr_h = SoilWindows(normalize(sub(P["idx"]["train"], g_tr), g_tr, mu, sd), dates[P["idx"]["train"]], g_tr, cfg_h, "train", 0)


def _referans(wd, g):
    W = wd["x"].shape[0]
    Tn = wd["x"][..., IDX_T].reshape(W, g.n_st, len(DERINLIKLER)).astype(float)
    z, la, lo = g.st_rakim_km, g.stations.lat.to_numpy(float), g.stations.lon.to_numpy(float)
    ref = np.full((W, g.n_st, len(DERINLIKLER)), np.nan)
    for i in range(g.n_st):
        vis = np.setdiff1d(np.arange(g.n_st), [i])
        D = _hv(la[[i]][:, None], lo[[i]][:, None], la[vis][None, :], lo[vis][None, :])
        for d in range(1, len(DERINLIKLER)):
            V = Tn[:, vis, d]
            a, b = _rd(V, z[vis])
            yok = ~np.isfinite(b)                       # IDW betigiyle ayni geri donus: <3 kaynak -> duz IDW
            a, b = np.where(yok, 0.0, a), np.where(yok, 0.0, b)
            ref[:, i, d] = _idw(V - (a[:, None] + b[:, None] * z[vis][None, :]), D, cfg_h.idw_p, cfg_h.idw_k)[:, 0] \
                + a + b * z[i]
    return ref.reshape(W, g.S)


for ad, w in (("eval", ev_h), ("train", tr_h)):
    ok, fark, n_ = True, 0.0, 0
    for i in range(min(6, len(w))):
        wd = w.window(i)
        ref, bs = _referans(wd, w.g), wd["base"][..., 0]
        ok &= np.array_equal(np.isnan(ref), np.isnan(bs))
        m = np.isfinite(ref)
        n_ += int(m.sum())
        if m.any():
            fark = max(fark, float(np.abs(ref[m] - bs[m]).max()))
    kontrol(f"{ad}: base = eski IDW_rakim fonksiyonlari (kendisi haric, maskeliler kaynak degil)",
            ok and n_ > 0 and fark < 1e-4, f"{n_} deger, en buyuk fark {fark:.1e}")
hm = np.isin(g_all.slot_station, P["hold_local"])
Xn2 = Xn_ev.copy(); Xn2[:, hm, IDX_T] += 50.0
ev_h2 = _ev(Xn2)
kontrol("sizinti yok: maskeli istasyonun T'si degisince hicbir dugumun base'i degismiyor",
        all(nan_esit(ev_h.window(i)["base"], ev_h2.window(i)["base"]) for i in range(len(ev_h))))
Xn3 = Xn_ev.copy(); Xn3[:, ~hm, IDX_T] += 5.0 * np.random.RandomState(3).randn(int((~hm).sum()))
kontrol("ters kontrol: gorunur istasyonlarin T'si degisince base degisiyor (test bos degil)",
        not nan_esit(ev_h.window(0)["base"], _ev(Xn3).window(0)["base"]))
wd = ev_h.window(0); bs = wd["base"][..., 0]; xt0 = wd["x"][..., IDX_T]
gor = np.isfinite(xt0) & np.isfinite(bs)
kontrol("gorunur dugumun base'i kendi olcumune esit degil (kendisi haric)",
        gor.sum() > 0 and not np.isclose(bs[gor], xt0[gor]).any(), f"{int(gor.sum())} dugum")
kontrol("yuzey dugumunde base NaN", np.isnan(bs[:, g_all.slot_depth == 0]).all())
kontrol("maskeli derinlik dugumlerinde base hesaplandi",
        np.isfinite(bs[:, hm & (g_all.slot_depth > 0)]).all())

torch.manual_seed(0)
mh = build_model(cfg_h, variant_opts("gnn"), in_dim=N_FEAT, target_cols=[IDX_T], edge_dims=EDGE_DIM, idw_artik=True)
kontrol("hibrit: giris boyutu + 1", mh.inp.in_features == N_FEAT + 1)
bh = collate([ev_h.window(i) for i in range(min(4, len(ev_h)))], ev_h.W, ev_h.S, g_all.templates)
mh.eval()
with torch.no_grad():
    yh = mh(bh)
miss = torch.isnan(bh.x[:, IDX_T:IDX_T + 1])
kontrol("hibrit: baslangicta cikti = IDW_rakim (duzeltme sifir)",
        torch.equal(yh[miss], torch.nan_to_num(bh.base, nan=0.0)[miss]))
kontrol("hibrit: gorunur dugumde cikti = olcum", torch.equal(yh[~miss], bh.x[:, IDX_T:IDX_T + 1][~miss]))
mh.train()
bt = next(tr_h.batches(False))
opt = torch.optim.Adam(mh.parameters(), lr=1e-3)
lh = masked_mse(mh(bt), bt); opt.zero_grad(); lh.backward()
kontrol("hibrit: kayip/gradyan sonlu, cikis katmani gradyan aliyor",
        torch.isfinite(lh) and all(p.grad is None or torch.isfinite(p.grad).all() for p in mh.parameters())
        and float(mh.out[-1].weight.grad.abs().sum()) > 0, f"loss={float(lh):.3f}")
opt.step()
kontrol("hibrit: bir adimdan sonra duzeltme katmani sifir degil", float(mh.out[-1].weight.abs().sum()) > 0)
try:
    mh(next(tr_w.batches(False))); kontrol("hibrit: base'siz batch reddedilir", False)
except ValueError:
    kontrol("hibrit: base'siz batch reddedilir", True)
try:
    SoilConfig(idw_k=0); kontrol("gecersiz idw_k reddedilir", False)
except ValueError:
    kontrol("gecersiz idw_k reddedilir", True)

from toprak_veri_ve_graf import idw_rakim
_g = P["g_all"]
_T = np.random.RandomState(5).randn(4, _g.n_st) * 3 + 10
_k = np.zeros((4, _g.n_st), bool); _k[:, [0, 1]] = True          # yalniz 2 kaynak: egim kurulamaz
_D = _hv(_g.stations.lat.to_numpy(float)[:, None], _g.stations.lon.to_numpy(float)[:, None],
         _g.stations.lat.to_numpy(float)[[0, 1]][None, :], _g.stations.lon.to_numpy(float)[[0, 1]][None, :])
_hed = np.arange(2, _g.n_st)
kontrol("hibrit: <3 kaynakta taban = duz IDW (IDW betigiyle ayni geri donus)",
        np.allclose(idw_rakim(_T, _k, _g, 2.0, 6)[:, _hed], _idw(_T[:, [0, 1]], _D[_hed], 2.0, 6)))

print("\n== Secenekler: nem varsayilani, konum, katlar, val2023, kayitli model, RFSI ==")
kontrol("varsayilan SoilConfig: nem girdi disi", SoilConfig().cikar == ("soil_moisture",))
cfg_k = _rep(cfg, konum_pe=False)
_wk = SoilWindows(tr_w.Xn, dates[P["idx"]["train"]], g_tr, cfg_k, "eval", 0, masked_stations=np.array([0]))
_wa = SoilWindows(tr_w.Xn, dates[P["idx"]["train"]], g_tr, cfg, "eval", 0, masked_stations=np.array([0]))
_pk, _pa = _wk.window(0)["pos"], _wa.window(0)["pos"]
kontrol("konum_pe=False: enlem/boylam 0, derinlik ve yilin gunu ayni",
        (_pk[..., 1:3] == 0).all() and np.array_equal(_pk[..., [0, 3, 4]], _pa[..., [0, 3, 4]]) and (_pa[..., 1:3] != 0).any())

from toprak_katlar import ISTASYON_BOLGE, en_yakin_gorunur, kat_atamasi, kat_istasyonlari
_ka = kat_atamasi()
_katlar = [kat_istasyonlari(k) for k in range(5)]
kontrol("katlar: 82 istasyon, her biri tek katta, buyukluk 16-17",
        sorted(sum(map(list, _katlar), [])) == sorted(ISTASYON_BOLGE) and {len(k) for k in _katlar} <= {16, 17})
kontrol("katlar: sabit tohum ayni atama, farkli tohum farkli", kat_atamasi() == _ka and kat_atamasi(tohum=1) != _ka)
_bs = {}
for s_, k_ in _ka.items():
    _bs.setdefault(ISTASYON_BOLGE[s_], []).append(k_)
kontrol("katlar: her bolge katlara dengeli yayiliyor (bolge icinde kat sayilari en fazla 1 farkli)",
        all(max(np.bincount(v, minlength=5)) - min(np.bincount(v, minlength=5)) <= 1 for v in _bs.values()))
_m = en_yakin_gorunur(st, st.istasyon_no.to_numpy()[[3, 7]], st.istasyon_no.to_numpy()[[0, 1, 2, 4]])
_Db = _hv(st.lat.to_numpy()[[3, 7]][:, None], st.lon.to_numpy()[[3, 7]][:, None],
          st.lat.to_numpy()[[0, 1, 2, 4]][None, :], st.lon.to_numpy()[[0, 1, 2, 4]][None, :])
kontrol("en_yakin_gorunur: uzaklik ve istasyon kaba kuvvetle ayni",
        np.allclose(_m.en_yakin_gorunur_km, _Db.min(1))
        and np.array_equal(_m.en_yakin_gorunur_no, st.istasyon_no.to_numpy()[[0, 1, 2, 4]][_Db.argmin(1)]))

import gnn_egitim as _E
_orj = _E.run_epoch
_E.run_epoch = lambda *a_, **k_: float("nan")
try:
    _E.fit(build_model(cfg, variant_opts("gnn"), in_dim=N_FEAT, target_cols=[IDX_T], edge_dims=EDGE_DIM),
           Splits(tr_w, ev_w), _rep(cfg, max_epochs=1), seed=0)
    kontrol("dogrulama kaybi hic sonlu degilse anlasilir hata", False)
except RuntimeError as e_:
    kontrol("dogrulama kaybi hic sonlu degilse anlasilir hata", "sonlu" in str(e_))
finally:
    _E.run_epoch = _orj

import rfsi_karsilastirma_modeli as R
_rng = np.random.RandomState(11)
_T = _rng.randn(6, _g.n_st) * 3 + 10
_kay = _rng.rand(6, _g.n_st) < 0.7
_o, _u, _gi = R.komsu_ozellik(_T, _kay, _g, 3, 2.0, 6)
ok_ = True
for kk in range(6):
    for i in range(_g.n_st):
        aday = [j for j in np.argsort(_g.st_dist[i], kind="stable") if j != i and _kay[kk, j]][:3]
        bek_o = np.full(3, np.nan); bek_u = np.full(3, np.nan)
        bek_o[:len(aday)] = _T[kk, aday]; bek_u[:len(aday)] = _g.st_dist[i, aday]
        ok_ &= nan_esit(_o[kk, i], bek_o) and nan_esit(_u[kk, i], bek_u)
kontrol("RFSI: en yakin n gorunur olcum/uzaklik kaba kuvvetle ayni (kendisi haric)", ok_)
kontrol("RFSI: IDW_rakim ozniteligi hibrit tabaniyla ayni fonksiyon", nan_esit(_gi, idw_rakim(_T, _kay, _g, 2.0, 6)))
_tr = P["g_tr"].st_global
_grp = [_tr[g_] for g_ in P["groups"]]
_kup = R.ozellik_kubu(P, P["idx"]["test"], [(h_, np.setdiff1d(_tr, h_)) for h_ in _grp], 3, 2.0, 6)
_Tt = X[P["idx"]["test"]][..., IDX_T].reshape(len(P["idx"]["test"]), len(st), len(DERINLIKLER))[:, :, 1:]
ok_ = True
for h_ in _grp:
    gor = np.setdiff1d(_tr, h_)
    for i in h_:
        for d_ in range(len(DERINLIKLER) - 1):
            for t_ in range(len(_Tt)):
                aday = [j for j in np.argsort(_g.st_dist[i], kind="stable") if j in gor and np.isfinite(_Tt[t_, j, d_])][:3]
                bek = np.full(3, np.nan); bek[:len(aday)] = _Tt[t_, aday, d_]
                ok_ &= nan_esit(_kup[t_, i, d_, :3].astype(float), bek)
kontrol("RFSI: gruplu degerlendirmede komsular yalniz diger gruplardan (sizinti yok)", ok_)

cfg_ic = _rep(cfg, ic_dogrulama=True)
P_ic = hazirla(cfg_ic, df)
_ic_no = set(P_ic["stations"].istasyon_no.to_numpy()[P_ic["ic_global"]])
kontrol("ic dogrulama: istasyonlar egitim grafinda yok, holdout'tan ayri, bos degil",
        len(_ic_no) > 0 and not _ic_no & set(P_ic["g_tr"].stations.istasyon_no) and not _ic_no & set(hold),
        f"{len(_ic_no)} ic dogrulama, {P_ic['g_tr'].n_st} ic egitim")
kontrol("ic dogrulama: erken durdurmada maskeli istasyonlar tam olarak ic dogrulama istasyonlari",
        set(P_ic["g_val"].stations.istasyon_no.to_numpy()[P_ic["val_set"]]) == _ic_no
        and set(P_ic["g_val"].stations.istasyon_no) == set(P_ic["g_tr"].stations.istasyon_no) | _ic_no)
_, _, sonuc_ic = kos(cfg_ic, P_ic, "gnn", 0)
kontrol("ic dogrulama: ic_val2023 bolumu ic dogrulama istasyonlarinda; holdout satirlari degismiyor",
        set(sonuc_ic["ic_val2023"].istasyon_no) == _ic_no
        and all(sonuc_ic[a_][["istasyon_no", "derinlik", "tarih"]].equals(
            kos(cfg, P, "gnn", 0)[2][a_][["istasyon_no", "derinlik", "tarih"]]) for a_ in ("spatial_test", "spatial_full")))

print("\n== Iki asamali katli egitim ==")
import tempfile
import toprak_gnn_calistir as _C
cfg_a = _rep(cfg, max_epochs=4, patience=10)
torch.manual_seed(0)
m1, h1, s1 = kos(cfg_a, P, "gnn", 0)
_hy = os.path.join(tempfile.mkdtemp(), "history.csv")
pd.DataFrame(h1).to_csv(_hy, index=False)
E = _C.epoch_kaynaktan(_hy)
kontrol("1. asama: epoch_kaynaktan = en dusuk dogrulama kaybinin epoch'u + 1",
        E == int(pd.DataFrame(h1).val_loss.idxmin()) + 1, f"{len(h1)} epoch, secilen {E}")
m2, h2, s2 = kos(cfg_a, P, "gnn", 0, sabit_epoch=E)
kontrol("2. asama (ayni veride): sabit epoch, dogrulama yok, 1. asamanin en iyi modelini birebir uretir",
        len(h2) == E and pd.DataFrame(h2).val_loss.isna().all()
        and all(np.allclose(s1[a_].y_pred, s2[a_].y_pred, atol=1e-5) for a_ in s1))
import sys as _sys
_eski = _sys.argv
for _arg, _ad in ((["--kat", "0"], "asamasiz"), (["--kat", "0", "--ic-dogrulama", "--sabit-epoch", "3"], "iki asama birden")):
    _sys.argv = ["x"] + _arg
    try:
        _C.main(); kontrol(f"katli kosu {_ad} reddedilir", False)
    except ValueError:
        kontrol(f"katli kosu {_ad} reddedilir", True)
_sys.argv = _eski

import rf_mlp_transformer_karsilastirma_modelleri as _T
_r = np.random.RandomState(0)
_Xa = _r.randn(600, 6).astype(np.float32); _ya = (_Xa[:, 0] * 2 + _r.randn(600) * 0.1).astype(np.float32)
_Xv = _r.randn(200, 6).astype(np.float32); _yv = (_Xv[:, 0] * 2).astype(np.float32)
_p = dict(_T.MLP_PARAMS, n_epochs=12, patience=3, batch_size=64)
torch.manual_seed(0)
_t1, _g1 = _T.egit_ag(_T.MLP(6, [16], 0.1), _p, _Xa, _ya, _Xv, _yv, {"v": _Xv}, 0, False, False)
_hh = pd.DataFrame(_g1)
_lr = _hh.lr.to_numpy()[:int(_hh.val.idxmin()) + 1]
torch.manual_seed(0)
_t2, _g2 = _T.egit_ag(_T.MLP(6, [16], 0.1), _p, _Xa, _ya, _Xv, _yv, {"v": _Xv}, 0, False, False, sabit_lr=_lr)
kontrol("MLP/FT 2. asama (ayni veride): lr sirasi tekrar, 1. asamanin en iyi modelini birebir uretir",
        len(_g2) == len(_lr) and np.allclose([g_["lr"] for g_ in _g2], _lr) and np.allclose(_t1["v"], _t2["v"], atol=1e-6),
        f"{len(_g1)} epoch, secilen {len(_lr)}, lr {sorted(set(np.round(_lr, 6)))}")

_lr2 = np.array([1e-3, 1e-3, 5e-4, 2.5e-4])
torch.manual_seed(0)
_m3 = _T.MLP(6, [16], 0.1)
_, _g3 = _T.egit_ag(_m3, _p, _Xa, _ya, _Xv, _yv, {"v": _Xv}, 0, False, False, sabit_lr=_lr2)
torch.manual_seed(0)
_, _g4 = _T.egit_ag(_T.MLP(6, [16], 0.1), _p, _Xa, _ya, _Xv, _yv, {"v": _Xv}, 0, False, False, sabit_lr=np.full(4, 1e-3))
kontrol("MLP/FT 2. asama: degisen lr sirasi epoch epoch uygulanir (sabit lr'den farkli egitim)",
        np.allclose([g_["lr"] for g_ in _g3], _lr2) and not np.isclose(_g3[-1]["train"], _g4[-1]["train"]))

print("\n== Uctan uca (6 varyant, tohum 0, 2 epoch) ==")
for vad in VARIANTS:
    mdl, hist, sonuc = kos(cfg, P, vad, 0)
    s = "  ".join(f"{ad}: RMSE {metrics(d).iloc[0].rmse:.3f} n={len(d)}" for ad, d in sonuc.items())
    print(f"  {vad:24s} {s}")
    if vad == "gnn":
        kontrol("temporal: her egitim istasyonu tahmin edildi",
                set(sonuc["temporal"].istasyon_no) == set(g_tr.stations.istasyon_no))
        kontrol("spatial: yalniz holdout istasyonlari", set(sonuc["spatial_full"].istasyon_no) == set(hold))
        kontrol("spatial_full tum donemi kapsiyor", sonuc["spatial_full"].tarih.nunique() == len(dates))
        n_obs = int(np.isfinite(Xr[:, P["hold_local"], 1:, IDX_T]).sum())
        kontrol("spatial_full satir sayisi = holdout gozlem sayisi", len(sonuc["spatial_full"]) == n_obs, f"{n_obs}")
        kontrol("derinlik 0 metriklerde yok", 0 not in set(sonuc["temporal"].derinlik))
        sonuc_gnn = sonuc
        kontrol("val2023: egitim istasyonlari, dogrulama yili",
                set(sonuc["val2023"].istasyon_no) == set(g_tr.stations.istasyon_no)
                and set(sonuc["val2023"].tarih.dt.year) == set(cfg.val_years))
        import tempfile
        _y = os.path.join(tempfile.mkdtemp(), "m.pt")
        torch.save(mdl.state_dict(), _y)
        _, _h0, sonuc_y = kos(cfg, P, vad, 0, model_yolu=_y)
        kontrol("--sadece-degerlendir: kayitli modelle tahminler birebir ayni, egitim yok",
                len(_h0) == 0 and all(np.allclose(sonuc_y[a_].y_pred, sonuc[a_].y_pred) for a_ in sonuc))
mdl, hist, sonuc_h = kos(cfg_h, P, "gnn", 0)
print("  " + f"{'hibrit (idw_artik)':24s} " + "  ".join(f"{ad}: RMSE {metrics(d).iloc[0].rmse:.3f} n={len(d)}" for ad, d in sonuc_h.items()))
kontrol("hibrit uctan uca: tahminler sonlu, satir kumeleri hibritsizle ayni",
        all(np.isfinite(sonuc_h[ad].y_pred).all() and
            sonuc_h[ad][["istasyon_no", "derinlik", "tarih"]].equals(sonuc_gnn[ad][["istasyon_no", "derinlik", "tarih"]])
            for ad in sonuc_h))

print("\n== Kriging (regresyon krigingi) ==")
import kriging_karsilastirma_modeli as KR
import idw_karsilastirma_modeli as IW
_r = np.random.RandomState(3)
_xy = _r.rand(13, 2) * 600
_Dk = np.sqrt(((_xy[:, None] - _xy[None]) ** 2).sum(-1))
_A = _r.randn(40, 12); _A[_r.rand(40, 12) < 0.4] = np.nan
_A[0] = np.nan; _A[1] = np.nan; _A[1, 4] = 2.5
_c0, _c1, _aa = _r.rand(40) * 0.3 + 0.01, _r.rand(40) + 0.5, _r.rand(40) * 400 + 50
ok_ = True
for _mo in KR.MODELLER:
    for _kk in (3, None):
        _top = KR.kriging(_A, _Dk[0, 1:], _Dk[1:, 1:], _mo, _c0, _c1, _aa, _kk)
        for t_ in range(40):
            nb = [j for j in np.argsort(_Dk[0, 1:], kind="stable") if np.isfinite(_A[t_, j])][:_kk]
            if not nb:
                ok_ &= bool(np.isnan(_top[t_])); continue
            M_ = np.zeros((len(nb) + 1,) * 2); b_ = np.zeros(len(nb) + 1)
            M_[:-1, :-1] = KR.gamma(_mo, _Dk[1:, 1:][np.ix_(nb, nb)], _c0[t_], _c1[t_], _aa[t_]); M_[:-1, -1] = M_[-1, :-1] = 1
            b_[:-1] = KR.gamma(_mo, _Dk[0, 1:][nb], _c0[t_], _c1[t_], _aa[t_]); b_[-1] = 1
            ok_ &= bool(np.isclose(_top[t_], np.linalg.solve(M_, b_)[:-1] @ _A[t_, nb]))
kontrol("kriging: toplu cozum gun gun elle cozumle ayni (3 model, k=3 ve hepsi; eksik olcum, hic/tek komsu)", ok_)
_t5 = KR.kriging(np.where(np.isfinite(_A), 5.0, np.nan), _Dk[0, 1:], _Dk[1:, 1:], "gauss", _c0, _c1, _aa, None)
kontrol("kriging: sabit alan -> ayni sabit (agirliklar toplami 1); tek komsu -> o deger; komsusuz -> NaN",
        np.allclose(_t5[2:], 5.0) and np.isclose(_t5[1], 5.0) and np.isnan(_t5[0])
        and np.isclose(KR.kriging(_A, _Dk[0, 1:], _Dk[1:, 1:], "ustel", _c0, _c1, _aa, 6)[1], 2.5))
kontrol("kriging: variogram olcegi agirliklari degistirmez",
        nan_esit(KR.kriging(_A, _Dk[0, 1:], _Dk[1:, 1:], "kuresel", 7 * _c0, 7 * _c1, _aa, 6),
                 KR.kriging(_A, _Dk[0, 1:], _Dk[1:, 1:], "kuresel", _c0, _c1, _aa, 6)))
_S = KR.standartlastir(np.vstack([_r.randn(3, 12) * 4 + 9, np.r_[1.0, 2.0, np.full(10, np.nan)][None]]))
kontrol("kriging: gunluk standartlastirma gun ici sd = 1; 3'ten az olcumlu gun disarida",
        np.allclose(np.nanstd(_S[:3], 1), 1.0) and np.isnan(_S[3]).all())
_C = 0.2 * np.eye(12) + np.exp(-3 * _Dk[1:, 1:] / 250.0)
_h, _gv, _nv = KR.deneysel(_r.multivariate_normal(np.zeros(12), _C, size=4000), _Dk[1:, 1:])
_p = KR.uydur("ustel", _h, _gv, _nv)
kontrol("kriging: variogram uydurma bilinen ustel alanin menzilini buluyor (250 km, %25 icinde)",
        abs(_p[2] - 250) < 62, f"c0={_p[0]:.2f} c1={_p[1]:.2f} a={_p[2]:.0f}")
_st = P["stations"]
_lat, _lon = _st.lat.to_numpy(float), _st.lon.to_numpy(float)
_Dall = _hv(_lat[:, None], _lon[:, None], _lat[None, :], _lon[None, :])
_Tob = X[..., IDX_T].reshape(len(dates), len(_st), len(DERINLIKLER)).astype(float)
_rak = _st.rakim.to_numpy(float) / 1000
_vg = {(d_, m_): np.tile([0.2, 1.0, 300.0], (12, 1)) for d_ in DERINLIKLER[1:] for m_ in KR.MODELLER}
_hl = P["g_all"].st_global[P["hold_local"]]
_ti = P["idx"]["test"]
_kr = KR.satirlar(_Tob[_ti], dates[_ti], dates[_ti].month.to_numpy(), _st.istasyon_no.to_numpy(), _Dall, _tr, [_hl],
                  _vg, "gauss", 6, _rak)
_iw = IW.satirlar(_Tob[_ti], dates[_ti], _st.istasyon_no.to_numpy(), _lat, _lon, _tr, [_hl], 2.0, 6, _rak)
kontrol("kriging: satir kumesi, y_true ve tahminsiz satirlar rakim-IDW ile ayni (spatial_test)",
        _kr[["istasyon_no", "derinlik", "tarih", "y_true"]].equals(_iw[["istasyon_no", "derinlik", "tarih", "y_true"]])
        and np.array_equal(_kr.y_pred.isna(), _iw.y_pred.isna()))
_kr1 = KR.satirlar(_Tob[_ti], dates[_ti], dates[_ti].month.to_numpy(), _st.istasyon_no.to_numpy(), _Dall, _tr, [_hl],
                   _vg, "gauss", 1, _rak)
_iw1 = IW.satirlar(_Tob[_ti], dates[_ti], _st.istasyon_no.to_numpy(), _lat, _lon, _tr, [_hl], 2.0, 1, _rak)
kontrol("kriging: k=1 iken rakim-IDW k=1 ile ayni (tek komsunun artigi + dogru)", nan_esit(_kr1.y_pred.to_numpy(), _iw1.y_pred.to_numpy()))
_Tz = _Tob.copy(); _Tz[:, _hl] = np.where(np.isfinite(_Tob[:, _hl]), 99.0, np.nan)
_krz = KR.satirlar(_Tz[_ti], dates[_ti], dates[_ti].month.to_numpy(), _st.istasyon_no.to_numpy(), _Dall, _tr, [_hl],
                   _vg, "gauss", 6, _rak)
kontrol("kriging: hedefin olcumu tahmini etkilemez (sizinti yok)", nan_esit(_krz.y_pred.to_numpy(), _kr.y_pred.to_numpy()))

print("\n== softmax yatay katman, duzeltme cezasi, blok maskesi, epoch kaydi ==")
import tempfile
from gnn_model import SoftmaxUzaklikConv
from gnn_egitim import duzeltme_cezasi, fit, run_epoch
from toprak_veri_ve_graf import collate as _col
kontrol("yatay varsayilan 'kapi': H iliskisi GatedEdgeConv, iki varsayilan modelin anahtarlari ayni (lin_edge var)",
        SoilConfig().yatay == "kapi" and set(mh.state_dict()) == set(build_model(
            cfg_h, variant_opts("gnn"), in_dim=N_FEAT, target_cols=[IDX_T], edge_dims=EDGE_DIM,
            idw_artik=True).state_dict()) and "convs.0.H.lin_edge.weight" in mh.state_dict())
for _k, _v in (("yatay", "baska"), ("ceza", -1.0), ("blok_orani", 1.5)):
    try:
        SoilConfig(**{_k: _v}); kontrol(f"gecersiz {_k} reddedilir", False)
    except ValueError:
        kontrol(f"gecersiz {_k} reddedilir", True)
try:
    SoilConfig(ceza=1.0); kontrol("ceza yalniz hibritte (idw_artik kapaliyken reddedilir)", False)
except ValueError:
    kontrol("ceza yalniz hibritte (idw_artik kapaliyken reddedilir)", True)
_sc = SoftmaxUzaklikConv(8)
_ei = torch.tensor([[1, 2, 3, 1, 4], [0, 0, 0, 2, 3]])                 # 0 <- {1,2,3}; 2 <- 1; 3 <- 4
_ea = torch.tensor([[0.02, 0.0], [0.08, 0.0], [0.15, 0.0], [0.05, 0.0], [0.10, 0.0]])
_gor = torch.tensor([False, True, True, False, False])                 # 3 ve 4 gorunmez
_e, _a = _sc.agirlik(_ei, _ea, _gor, 5)
kontrol("softmax: yalniz gorunur kaynakli kenarlar, hedef basina agirlik toplami 1",
        _e.shape[1] == 3 and set(map(tuple, _e.t().tolist())) == {(1, 0), (2, 0), (1, 2)}
        and torch.allclose(torch.zeros(5).index_add_(0, _e[1], _a)[[0, 2]], torch.ones(2)))
_x8 = torch.randn(5, 8)
_o = _sc(_x8, _ei, _ea, _gor)
kontrol("softmax: gorunur komsusu olmayan dugumde mesaj 0 (cikti = ReLU(W_self h))",
        torch.allclose(_o[3], torch.relu(_sc.lin_self(_x8[3]))) and torch.allclose(_o[1], torch.relu(_sc.lin_self(_x8[1]))))
_x8b = _x8.clone(); _x8b[3] += 100.0; _x8b[4] -= 100.0
kontrol("softmax: gorunmez kaynagin h'si degisince diger dugumlerin ciktisi degismiyor",
        torch.allclose(_sc(_x8b, _ei, _ea, _gor)[[0, 2]], _o[[0, 2]]))
with torch.no_grad():                                                   # skor uzaklikla azalsin: agirlik yakina buyuk
    _sc.skor[0].weight.copy_(torch.eye(16, 9)); _sc.skor[0].bias.zero_()
    _sc.skor[2].weight.copy_(torch.tensor([[8.0, 4, 2, 1, 0, -1, -2, -4, -8] + [0.0] * 7])); _sc.skor[2].bias.zero_()
_e, _a = _sc.agirlik(_ei, _ea, _gor, 5)
_w = {tuple(p): float(q) for p, q in zip(_e.t().tolist(), _a)}
kontrol("softmax: uzaklik ozniteligi agirligi etkileyebiliyor (20 km komsu 80 km komsudan agir)",
        _w[(1, 0)] > 2 * _w[(2, 0)], f"{_w[(1, 0)]:.3f} / {_w[(2, 0)]:.3f}")
from ortak_veri import station_pairs as _sp
from toprak_katlar import kat_atamasi as _kat_at
_stg = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "DATASET", "MGM_istasyonlar_final.csv"), encoding="latin-1")
_stg = _stg[_stg.istasyon_no.isin(list(_kat_at()))].reset_index(drop=True)
_src, _dst, _att = _sp(_stg, SoilConfig())
_rg = np.random.RandomState(11)
_Tg = _rg.normal(12.0, 5.0, len(_stg))
_vis = _rg.rand(len(_stg)) < 0.6
_ok_id = len(_stg) == 82
for _p in (1.0, 2.0):
    _ci = SoftmaxUzaklikConv(1).double()                                           # float64: esitlik sinamasi
    _ci.skor_d = lambda d, p=_p: -p * torch.log(d[:, 0])
    with torch.no_grad():
        _ci.lin_neigh.weight.fill_(1.0); _ci.lin_neigh.bias.zero_()
        _ci.lin_self.weight.zero_(); _ci.lin_self.bias.zero_()
        _oi = _ci(torch.tensor(_Tg + 100.0, dtype=torch.float64)[:, None], torch.as_tensor(np.stack([_src, _dst])),
                  torch.as_tensor(_att, dtype=torch.float64), torch.as_tensor(_vis))[:, 0].numpy() - 100.0
    _fark, _say = 0.0, 0
    for _i in range(len(_stg)):
        _kom = _src[(_dst == _i) & _vis[_src]]
        if len(_kom) == 0:
            continue
        _ref = IW.idw(_Tg[_kom][None, :], (_att[(_dst == _i) & _vis[_src], 0] * 1000.0)[None, :], _p, None)[0, 0]
        _fark = max(_fark, abs(_oi[_i] - _ref)); _say += 1
    _ok_id &= _fark < 1e-5 and _say > 40
kontrol("softmax katmani kimligi: skor=-p ln d, lin_neigh=birim, lin_self=0 -> cikti = IDW (ayni gorunur H komsulari, "
        "k=hepsi), p=1 ve 2, gercek 82 istasyon", _ok_id, f"en buyuk fark {_fark:.2e}, {_say} hedef")
cfg_s = _rep(cfg_h, yatay="softmax")
torch.manual_seed(0)
ms = build_model(cfg_s, variant_opts("gnn"), in_dim=N_FEAT, target_cols=[IDX_T], edge_dims=EDGE_DIM, idw_artik=True)
kontrol("softmax modeli: H iliskisi SoftmaxUzaklikConv, V ve T GatedEdgeConv",
        isinstance(ms.convs[0]["H"], SoftmaxUzaklikConv) and not isinstance(ms.convs[0]["V"], SoftmaxUzaklikConv))
ms.eval()
with torch.no_grad():
    ys = ms(bh)
kontrol("softmax hibrit: baslangicta cikti = taban, gorunur dugumde cikti = olcum",
        torch.equal(ys[miss], torch.nan_to_num(bh.base, nan=0.0)[miss]) and torch.equal(ys[~miss], bh.x[:, IDX_T:IDX_T + 1][~miss]))
ms.train()
_ls = masked_mse(ms(bt), bt); _ls.backward()
kontrol("softmax hibrit: kayip/gradyan sonlu, skor MLP'si gradyan aliyor (bir adimdan sonra)",
        torch.isfinite(_ls) and all(p.grad is None or torch.isfinite(p.grad).all() for p in ms.parameters()))
_opt = torch.optim.Adam(ms.parameters(), lr=1e-3); _opt.step(); _opt.zero_grad()
_ls2 = masked_mse(ms(bt), bt); _ls2.backward()
kontrol("softmax hibrit: 2. adimda skor MLP'sine gradyan geliyor",
        float(ms.convs[0]["H"].skor[0].weight.grad.abs().sum()) > 0)
torch.manual_seed(0)
mg = build_model(_rep(cfg, yatay="softmax"), variant_opts("gnn"), in_dim=N_FEAT, target_cols=[IDX_T],
                 edge_dims=EDGE_DIM)
_bg = next(SoilWindows(normalize(X[P["idx"]["train"]][:, g_tr.slot_global], g_tr, mu, sd), dates[P["idx"]["train"]],
                       g_tr, cfg, "train", 0).batches(False))
_lg = masked_mse(mg(_bg), _bg); _lg.backward()
kontrol("softmax GNN (hibrit degil): kayip ve gradyan sonlu",
        torch.isfinite(_lg) and all(p.grad is None or torch.isfinite(p.grad).all() for p in mg.parameters()))
mh0 = build_model(cfg_h, variant_opts("gnn"), in_dim=N_FEAT, target_cols=[IDX_T], edge_dims=EDGE_DIM, idw_artik=True)
with torch.no_grad():
    kontrol("ceza: baslangicta (cikti = taban) ceza 0", float(duzeltme_cezasi(mh0(bt), bt)) == 0.0)
mh.eval()
_c0 = _rep(cfg_h, ceza=0.0); _c10 = _rep(cfg_h, ceza=10.0)
kontrol("ceza: dogrulama kaybi cezadan bagimsiz",
        np.isclose(run_epoch(mh, ev_h, _c0), run_epoch(mh, ev_h, _c10)))
_m1 = build_model(cfg_h, variant_opts("gnn"), in_dim=N_FEAT, target_cols=[IDX_T], edge_dims=EDGE_DIM, idw_artik=True)
_m2 = build_model(cfg_h, variant_opts("gnn"), in_dim=N_FEAT, target_cols=[IDX_T], edge_dims=EDGE_DIM, idw_artik=True)
_m2.load_state_dict(mh.state_dict()); _m1.load_state_dict(mh.state_dict())
_tw = lambda: SoilWindows(normalize(X[P["idx"]["train"]][:, g_tr.slot_global], g_tr, mu, sd), dates[P["idx"]["train"]],
                          g_tr, cfg_h, "train", 3)
torch.manual_seed(1); _l0 = run_epoch(_m1, _tw(), _c0, torch.optim.SGD(_m1.parameters(), lr=0.0))
torch.manual_seed(1); _l10 = run_epoch(_m2, _tw(), _c10, torch.optim.SGD(_m2.parameters(), lr=0.0))
kontrol("ceza: egitim kaybina eklenir (duzeltme sifir degilken ceza > 0)", _l10 > _l0, f"{_l0:.4f} -> {_l10:.4f}")
_wb = SoilWindows(normalize(X[P["idx"]["train"]][:, g_tr.slot_global], g_tr, mu, sd), dates[P["idx"]["train"]],
                  g_tr, _rep(cfg, blok_maske=True, blok_orani=0.5), "train", 5)
_ok, _kk = True, int(np.ceil(0.5 * g_tr.n_st))
for _ in range(20):
    _mb = _wb.station_mask()
    _dmax = [g_tr.st_dist[c][_mb & (np.arange(g_tr.n_st) != c)].max(initial=0) for c in np.nonzero(_mb)[0]]
    _c = np.nonzero(_mb)[0][int(np.argmin(_dmax))]
    _ic = g_tr.st_dist[_c][_mb & (np.arange(g_tr.n_st) != _c)]
    _dis = g_tr.st_dist[_c][~_mb]
    _ok &= _mb.sum() == _kk and (len(_dis) == 0 or _ic.max(initial=0) <= _dis.min())
kontrol("blok maskesi: ceil(oran x n) istasyon, merkez + en yakinlari (20 cekim)", _ok, f"n={g_tr.n_st}, k={_kk}")
kontrol("blok maskesi kapali: eski maske (ayni tohumda ayni cekimler)",
        np.array_equal(SoilWindows(normalize(X[P["idx"]["train"]][:, g_tr.slot_global], g_tr, mu, sd),
                                   dates[P["idx"]["train"]], g_tr, cfg, "train", 5).station_mask(),
                       SoilWindows(normalize(X[P["idx"]["train"]][:, g_tr.slot_global], g_tr, mu, sd),
                                   dates[P["idx"]["train"]], g_tr, _rep(cfg, blok_maske=False), "train", 5).station_mask()))
with tempfile.TemporaryDirectory() as _td:
    _ce = _rep(cfg_h, epoch_sifir=True, max_epochs=2, patience=100, epoch_kayit=os.path.join(_td, "ep"))
    torch.manual_seed(0)
    _me = build_model(_ce, variant_opts("gnn"), in_dim=N_FEAT, target_cols=[IDX_T], edge_dims=EDGE_DIM, idw_artik=True)
    _ilk = {k: v.clone() for k, v in _me.state_dict().items()}
    fit(_me, Splits(_tw(), ev_h), _ce, seed=0)
    _dos = sorted(os.listdir(os.path.join(_td, "ep")))
    _s0 = torch.load(os.path.join(_td, "ep", "epoch_000.pt"), weights_only=True)
    kontrol("epoch kaydi: epoch 0 (egitimsiz) dahil her epoch, epoch 0 = baslangic agirliklari",
            _dos == ["epoch_000.pt", "epoch_001.pt", "epoch_002.pt"] and all(torch.equal(_s0[k], _ilk[k]) for k in _ilk), str(_dos))

print("\n== Uydu ablasyonu: tablo modelleri (--uydusuz) ==")
import rf_mlp_transformer_karsilastirma_modelleri as _TM
_ti = P["idx"]["train"]
_Gu, _Su = _TM.tablo(P, _ti, g_tr.st_global)
_Gs, _Ss = _TM.tablo(P, _ti, g_tr.st_global, uydu=False)
kontrol("uydusuz tablo: 14 girdi, satirlar ayni, sutunlar uydulu tablonun uydu disi sutunlariyla ayni",
        _Gs.shape[1] == len(_TM.GIRDI_ADLARI_UYDUSUZ) == 14 and _Su.equals(_Ss)
        and nan_esit(_Gs, _Gu[:, 3:-2]) and _Gu.shape[1] == len(_TM.GIRDI_ADLARI) == 19)
_o2, _o0 = _TM.Olcek(_Gu, _Su), _TM.Olcek(_Gs, _Ss, n_isaret=0)
kontrol("Olcek: varsayilanda son 2 sutun (isaret) olceklenmez; n_isaret=0'da hepsi olceklenir",
        np.allclose(_o2.mu[-2:], 0) and np.allclose(_o2.sd[-2:], 1) and np.allclose(_o0.mu, np.nanmean(_Gs, 0)))

print("\n== IGNNK ==")
import math as _math
import ignnk_karsilastirma_modeli as IG
torch.manual_seed(0)
_dg = IG.DGCN(3, 5, 1)
_W = torch.rand(2, 4, 4) + 0.1
_Wf, _Wb = IG.IGNNK.gecis(_W)
_H = torch.randn(2, 4, 3)
kontrol("IGNNK: gecis matrisleri satir toplami 1; Wb = W^T / rowsum(W^T)",
        torch.allclose(_Wf.sum(-1), torch.ones(2, 4)) and torch.allclose(_Wb, _W.transpose(1, 2) / _W.transpose(1, 2).sum(-1, keepdim=True)))
kontrol("IGNNK: DGCN K=1 = Wf H Th_b + Wb H Th_f (Denk. 1, bias yok)",
        torch.allclose(_dg(_H, _Wf, _Wb), _dg.th_b[0](torch.bmm(_Wf, _H)) + _dg.th_f[0](torch.bmm(_Wb, _H)), atol=1e-6)
        and _dg.th_b[0].bias is None)
_dg3 = IG.DGCN(3, 5, 3)
_t = IG.DGCN.cheb(_Wf, _H, 3)
kontrol("IGNNK: Chebyshev T_2 = 2 W T_1 - T_0", torch.allclose(_t[1], 2 * torch.bmm(_Wf, _t[0]) - _H, atol=1e-6) and len(_t) == 3)
_Dk = np.array([[0, 50, 100], [50, 0, 80], [100, 80, 0.0]])
_Wk = IG.komsuluk(_Dk, 100.0)
kontrol("IGNNK: komsuluk exp(-(d/sigma)^2), kosegen 1, uzaklikla azalir",
        np.allclose(np.diag(_Wk), 1) and np.isclose(_Wk[0, 2], np.exp(-1.0)) and _Wk[0, 1] > _Wk[0, 2])
kontrol("IGNNK: pencere baslari her gunu kapsar, son pencere sona hizali",
        list(IG.pencere_baslari(10)) == [0, 3, 6, 7] and list(IG.pencere_baslari(9)) == [0, 3, 6])
_VI = IG.Veri(P, DERINLIKLER.index(20))
_egt = P["g_tr"].st_global
_gtr = P["idx"]["train"]
_rng = np.random.RandomState(4)
_H0, _R, _Wy = IG.ornek_yigini(_VI, _gtr, np.array([0]), _egt, _rng, 100.0)
_rng = np.random.RandomState(4)
_ns = _rng.randint(int(_math.ceil(len(_egt) / 2)), len(_egt) + 1)
_sec = _egt[_rng.permutation(len(_egt))[:_ns]]
_nm = int(round(0.2 * _ns)); _no = _ns - _nm
_Tw = _VI.T[np.ix_(_gtr[:3], _sec)]
_a, _b = IG.dogru(_Tw[:, :_no], _VI.rak[_sec[:_no]])
_r = (_Tw - (_a[:, None] + _b[:, None] * _VI.rak[_sec][None, :])).T
kontrol("IGNNK ornek: n_s araligi, n_m = round(0,2 n_s), maskeli girdi 0, dogru yalniz gozlenenlerden, hedef = artik",
        _H0.shape[1] == _ns and np.all(_H0[0, _no:] == 0) and nan_esit(_R[0], _r.astype(np.float32))
        and np.allclose(_H0[0, :_no], np.nan_to_num(_r[:_no], nan=0.0), atol=1e-6)
        and np.allclose(_Wy[0], IG.komsuluk(_VI.D[np.ix_(_sec, _sec)], 100.0)), f"n_s={_ns}, n_m={_nm}")
_hdf = P["g_all"].st_global[P["hold_local"]]
_gt = P["idx"]["test"]
_m0 = IG.IGNNK()
for _p in _m0.parameters():
    torch.nn.init.zeros_(_p)
_Y0 = IG.tahmin(_m0, _VI, _gt, _egt, _hdf, 100.0, "cpu")
_a0, _b0 = IG.dogru(_VI.T[np.ix_(_gt, _egt)], _VI.rak[_egt])
kontrol("IGNNK: sifir model -> tahmin = gorunurlerden gunluk rakim dogrusu",
        np.allclose(_Y0, _a0[:, None] + _b0[:, None] * _VI.rak[_hdf][None, :]))
torch.manual_seed(1)
_mi, _gi = IG.egit(_VI, _gtr, _egt, 100.0, 0, "cpu", 2, dogrulama=(P["idx"]["val"], _egt, _hdf))
kontrol("IGNNK egitim: 2 epoch, kayip ve dogrulama MSE'si sonlu",
        len(_gi) == 2 and all(np.isfinite(g["train_mse"]) and np.isfinite(g["val_mse"]) for g in _gi))
_Y1 = IG.tahmin(_mi, _VI, _gt, _egt, _hdf, 100.0, "cpu")
_VI2 = IG.Veri(P, DERINLIKLER.index(20)); _VI2.T = _VI.T.copy(); _VI2.T[:, _hdf] += 50.0
kontrol("IGNNK: hedef istasyonun olcumu tahmini etkilemez (sizinti yok)",
        np.allclose(IG.tahmin(_mi, _VI2, _gt, _egt, _hdf, 100.0, "cpu"), _Y1))
_VI3 = IG.Veri(P, DERINLIKLER.index(20)); _VI3.T = _VI.T.copy(); _VI3.T[:, _egt[:2]] += 5.0
kontrol("IGNNK ters kontrol: gorunur istasyon degisince tahmin degisir",
        not np.allclose(IG.tahmin(_mi, _VI3, _gt, _egt, _hdf, 100.0, "cpu"), _Y1))
_sat = IG.satirlar(_VI, P, _gt, _hdf, _Y1, 20)
_iws = IW.satirlar(_Tob[_gt], dates[_gt], _st.istasyon_no.to_numpy(), _lat, _lon, _egt, [_hdf], 2.0, 6, _rak)
_iws = _iws[_iws.derinlik == 20].sort_values(["istasyon_no", "tarih"]).reset_index(drop=True)
_sat = _sat.sort_values(["istasyon_no", "tarih"]).reset_index(drop=True)
kontrol("IGNNK: satir kumesi ve y_true rakim-IDW ile ayni (spatial_test, 20 cm), tahminsiz satir yok",
        _sat[["istasyon_no", "tarih", "y_true"]].equals(_iws[["istasyon_no", "tarih", "y_true"]]) and _sat.y_pred.notna().all(),
        f"{len(_sat)} satir")

hata = [a for a, ok in SONUC if not ok]
print(f"\n{len(SONUC) - len(hata)}/{len(SONUC)} kontrol gecti." + (f" HATALI: {hata}" if hata else ""))
