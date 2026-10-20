import importlib.util
import os
import sys

import numpy as np
import pandas as pd

pd.set_option("compute.use_numexpr", False)
BURADA = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("katli", os.path.join(BURADA, "katli_birincil_karsilastirma.py"))
K = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(K)

SONUC = []
W = 2 * np.pi / 365.25
KAPPA = 450.0
D = np.sqrt(2 * KAPPA / W)
DER = (5, 10, 20, 50, 100)
TARIH = pd.date_range("2015-01-01", "2025-12-31", freq="D")


def kontrol(ad, ok, ek=""):
    SONUC.append((ad, bool(ok)))
    print(f"[{'OK ' if ok else 'HATA'}] {ad}" + (f"  ({ek})" if ek else ""))


def alan(istasyon, rng=None, genlik_oynak=0.0, eksik=None, kappa_tahmin=None):
    """Isi denklemi cozumu; eksik: {derinlik: maske(tarih) -> True = sil}. y_pred = y_true; kappa_tahmin verilirse
    y_pred ayni bicimde o kappa ile (d_t = sqrt(2 kappa_tahmin / w))."""
    t = (TARIH - K.T0).days.to_numpy(float)
    f = np.ones(len(TARIH))
    if genlik_oynak:
        yillar = TARIH.year.to_numpy()
        fy = {y: 1 + genlik_oynak * rng.randn() for y in np.unique(yillar)}
        f = np.array([fy[y] for y in yillar])
    Dt = D if kappa_tahmin is None else np.sqrt(2 * kappa_tahmin / W)
    par = []
    for z in DER:
        T = 15 + 12 * f * np.exp(-z / D) * np.cos(W * t - z / D)
        P = 15 + 12 * f * np.exp(-z / Dt) * np.cos(W * t - z / Dt)
        sec = np.ones(len(TARIH), bool) if not eksik or z not in eksik else ~eksik[z](TARIH)
        par.append(pd.DataFrame({"istasyon_no": istasyon, "derinlik": z, "tarih": TARIH[sec].strftime("%Y-%m-%d"),
                                 "y_true": T[sec], "y_pred": P[sec]}))
    return pd.concat(par, ignore_index=True)


def hesap(df):
    R, C100 = K.harmonik_turet(df)
    return R, K.difuzivite(C100)


def yakin(x, hedef, oran):
    return np.isfinite(x) and abs(x / hedef - 1) <= oran


R, Kp = hesap(alan(1))
s100, g100 = R.loc[(1, 100), "sonum_olcum"], R.loc[(1, 100), "gecikme_olcum_gun"]
kontrol("isi denklemi, tam kapsam: kappa_A ve kappa_phi %3 icinde 450",
        yakin(Kp.loc[1, "kA_olcum"], KAPPA, 0.03) and yakin(Kp.loc[1, "kphi_olcum"], KAPPA, 0.03),
        f"kA={Kp.loc[1, 'kA_olcum']:.1f} kphi={Kp.loc[1, 'kphi_olcum']:.1f}")
kontrol("isi denklemi, tam kapsam: 100 cm sonumu exp(-95/d) %1 icinde, gecikme 24,1 gun +/- 1",
        yakin(s100, np.exp(-95 / D), 0.01) and abs(g100 - 24.1) <= 1,
        f"sonum={s100:.4f} beklenen={np.exp(-95 / D):.4f} gecikme={g100:.2f} gun")
kontrol("isi denklemi, tam kapsam: tahmin = olcum iken genlik orani 1, faz farki 0, kappa orani 1",
        np.allclose(R.genlik_orani, 1) and np.allclose(R.faz_farki_gun, 0, atol=1e-6)
        and np.isclose(Kp.loc[1, "kA_tahmin"], Kp.loc[1, "kA_olcum"]))

KT = 280.0
Dt = np.sqrt(2 * KT / W)
R, Kp = hesap(alan(5, kappa_tahmin=KT))
ff = (100 / Dt - 100 / D) / W
kontrol("tahmin tarafi (olcum kappa 450, tahmin 280): kA/kphi_tahmin %3 icinde 280, kA_olcum %3 icinde 450",
        yakin(Kp.loc[5, "kA_tahmin"], KT, 0.03) and yakin(Kp.loc[5, "kphi_tahmin"], KT, 0.03)
        and yakin(Kp.loc[5, "kA_olcum"], KAPPA, 0.03),
        f"kA_t={Kp.loc[5, 'kA_tahmin']:.1f} kphi_t={Kp.loc[5, 'kphi_tahmin']:.1f} kA_o={Kp.loc[5, 'kA_olcum']:.1f}")
kontrol("tahmin tarafi: 100 cm sonum_tahmin %1 icinde exp(-95/d_t); genlik_orani %1 icinde exp(-100/d_t)/exp(-100/d_o); "
        "faz_farki_gun 0,5 gun icinde (100/d_t - 100/d_o)/w",
        yakin(R.loc[(5, 100), "sonum_tahmin"], np.exp(-95 / Dt), 0.01)
        and yakin(R.loc[(5, 100), "genlik_orani"], np.exp(-100 / Dt) / np.exp(-100 / D), 0.01)
        and abs(R.loc[(5, 100), "faz_farki_gun"] - ff) <= 0.5,
        f"sonum_t={R.loc[(5, 100), 'sonum_tahmin']:.4f}/{np.exp(-95 / Dt):.4f} "
        f"genlik_orani={R.loc[(5, 100), 'genlik_orani']:.4f}/{np.exp(-100 / Dt) / np.exp(-100 / D):.4f} "
        f"faz_farki={R.loc[(5, 100), 'faz_farki_gun']:+.2f}/{ff:+.2f} gun")

eksik = {5: lambda tr: tr.year.isin([2020, 2021, 2022]), 100: lambda tr: tr.year.isin([2016, 2017, 2018])}
ok_k, ok_s, bilgi = True, True, []
for tohum in range(5):
    df = alan(2, np.random.RandomState(tohum), 0.15, eksik)
    R, Kp = hesap(df)
    ok_k &= yakin(Kp.loc[2, "kA_olcum"], KAPPA, 0.03) and yakin(Kp.loc[2, "kphi_olcum"], KAPPA, 0.03)
    ok_s &= yakin(R.loc[(2, 100), "sonum_olcum"], np.exp(-95 / D), 0.01)
    H = K.harmonik(df)
    kA_eski = W * 95 ** 2 / (2 * np.log(H.loc[(2, 5), "A_olcum"] / H.loc[(2, 100), "A_olcum"]) ** 2)
    bilgi.append(f"{Kp.loc[2, 'kA_olcum']:.0f}/{kA_eski:.0f}")
kontrol("eksik yillar + genlik %15 (5 alan): ortak gunlerle kappa_A ve kappa_phi %3 icinde 450", ok_k,
        "kA ortak/kendi gunleri: " + " ".join(bilgi))
kontrol("eksik yillar + genlik %15 (5 alan): 100 cm sonumu (ortak gunler) %1 icinde exp(-95/d)", ok_s)

yaz = {z: (lambda tr: tr.month.isin([6, 7, 8])) for z in DER}
df3 = pd.concat([alan(3, eksik=yaz), alan(4, eksik={100: lambda tr: np.ones(len(tr), bool)})], ignore_index=True)
R, Kp = hesap(df3)
kontrol("her yil Haziran-Agustos eksik: kapsam disi (olcum var), kappa ve sonum NaN",
        (R.loc[3].kapsam == 0).all() and (R.loc[3].n > 0).all() and not Kp.loc[3, "kapsam_ortak"]
        and Kp.loc[3, "n_ortak"] > 0 and np.isnan(Kp.loc[3, ["kA_olcum", "kphi_olcum"]].astype(float)).all()
        and R.loc[3].sonum_olcum.isna().all())
kontrol("100 cm olcumu hic yok: 'olcum yok' (n = 0, n_ortak = 0), kappa NaN; diger derinlikler uydurulur",
        R.loc[(4, 100), "n"] == 0 and Kp.loc[4, "n_ortak"] == 0 and np.isnan(Kp.loc[4, "kA_olcum"])
        and R.loc[(4, 50), "kapsam"] == 1 and np.isfinite(R.loc[(4, 50), "sonum_olcum"]))

rng = np.random.RandomState(11)
ist = np.arange(100, 140)
b = dict(zip(ist, rng.randn(len(ist))))
SIGMA = 0.5
gun = pd.date_range("2024-01-01", "2025-12-31", freq="D").strftime("%Y-%m-%d")
p4 = []
for s in ist:
    for z in DER:
        y = 10 + rng.randn(len(gun))
        p4.append(pd.DataFrame({"istasyon_no": s, "derinlik": z, "tarih": gun, "y_true": y,
                                "y_pred": y + b[s] + SIGMA * rng.randn(len(gun))}))
(mse, sabit), _ = K.ayristirma_ozet(K.ayristir([pd.concat(p4, ignore_index=True)]), ist)
bb = np.mean([v ** 2 for v in b.values()])
beklenen = bb / (bb + SIGMA ** 2)
kontrol("sabit kayma payi beklenen degerin 0,01 yakininda", abs(sabit / mse - beklenen) < 0.01,
        f"pay={sabit / mse:.4f} beklenen={beklenen:.4f}")

ist5 = np.arange(60)
z = np.random.RandomState(5).randn(60)
z = (z - z.mean()) / z.std()
rb = pd.Series(2.0 + 0.3 * np.random.RandomState(6).rand(60), index=ist5)
durumlar = [("kucuk tutarli kazanc", -0.03 + 0.01 * z, "model iyi", "anlamli, +/-0,10 C icinde"),
            ("sifir civari dar", 0.02 * z, "ayirt edilemedi", "esdeger (+/-0,10 C)"),
            ("sifir civari genis", 0.8 * z, "ayirt edilemedi", "belirsiz"),
            ("referans lehine kucuk fark", 0.03 + 0.01 * z, "referans iyi", "anlamli, +/-0,10 C icinde"),
            ("buyuk kazanc", -0.5 + 0.1 * z, "model iyi", "")]
for ad, d, k_bek, e_bek in durumlar:
    r, karar = K.karsilastir(rb + d, rb, ist5)
    kontrol(f"karsilastir, {ad}: '{k_bek}' + '{e_bek}'", karar == k_bek and r["etiket"] == e_bek,
            f"karar='{karar}' etiket='{r['etiket']}' %90 GA=({r['ga90_alt']:.3f}, {r['ga90_ust']:.3f})")

def kt(kA):
    return pd.DataFrame({"n_ortak": [4000, 4000], "kapsam_ortak": [True, True], "kA_olcum": [450.0, 450.0],
                         "kA_tahmin": kA, "veri_olcum": [True, True], "veri_tahmin": [True, True]},
                        index=pd.Index([1, 2], name="istasyon_no"))
T = K.tohum_birlestir_kappa([kt([400.0, 400.0]), kt([np.nan, 450.0]), kt([500.0, 40000.0])])
tanimsiz = int((T.veri_tahmin & ~np.isfinite(T.kA_tahmin)).sum())
kontrol("tohum toplamasi: bir tohumda tanimsiz kappa -> istasyon tanimsiz (sayilir)",
        np.isnan(T.loc[1, "kA_tahmin"]) and tanimsiz == 1 and bool(T.loc[1, "veri_tahmin"]))
kontrol("tohum toplamasi: asiri tek tohum (400/450/40000) medyani 450", np.isclose(T.loc[2, "kA_tahmin"], 450.0),
        f"{T.loc[2, 'kA_tahmin']:.1f}")

hata = [a for a, ok in SONUC if not ok]
print(f"\n{len(SONUC) - len(hata)}/{len(SONUC)} kontrol gecti." + (f" HATALI: {hata}" if hata else ""))
sys.exit(1 if hata else 0)
