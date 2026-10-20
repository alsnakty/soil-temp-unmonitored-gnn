import os
import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.abspath(__file__))
GIRDI = os.path.join(KOK, "DATASET", "master_dataset.csv")
KIRILMA = os.path.join(KOK, "DATASET", os.environ.get("QC_KIRILMA", "kirilma_istasyon_orta.csv"))
TABLO = os.path.join(KOK, "DATASET", "hedef_qc_islemleri.csv")
CIKTI = os.path.join(KOK, "DATASET", "master_dataset_qc.csv")
CIKTI_K = os.path.join(KOK, "DATASET", "hedef_qc_duzeltmeler.csv")
CIKTI_A = os.path.join(KOK, "DATASET", "hedef_qc_atlanan.csv")
META = os.path.join(KOK, "DATASET", "MGM_istasyonlar_final.csv")

DER = [5, 10, 20, 50, 100]
SUT = {d: f"t{d}cmsicaklik" for d in DER}
SON_TAHMIN = pd.Timestamp("2023-12-31")   # adim tahmininde kullanilan en gec gun (test yillari disarida)
SON_KIRILMA = pd.Timestamp("2023-06-30")  # bundan sonraki kirilmalarda sonrasi icin yeterli test disi veri yok
SON_TEST = pd.Timestamp("2025-12-31")
TEST_KIRILMA = os.environ.get("QC_TEST_KIRILMA", "0") == "1"
SON_TARIH = pd.Timestamp(os.environ["QC_SON_TARIH"]) if os.environ.get("QC_SON_TARIH") else None
N_KOMSU = 5
MIN_GUN = 200                             # harmonik uyum icin pencere basina en az gun
ATLA = 30                                 # kirilma sonrasi tahminde atlanan gun (derin katman oturmasi)
INCELT = 60
PENCERE_INCE = 120


def derinlikler(s):
    return DER if str(s).strip() == "hepsi" else [int(v) for v in str(s).split(";")]


def harmonik(t, y):
    w = 2 * np.pi * t.dayofyear.values / 365.25
    A = np.column_stack([np.ones(len(w)), np.cos(w), np.sin(w)])
    k, *_ = np.linalg.lstsq(A, y, rcond=None)
    return k


def egri(k, t):
    w = 2 * np.pi * pd.DatetimeIndex(t).dayofyear.values / 365.25
    return k[0] + k[1] * np.cos(w) + k[2] * np.sin(w)


tablo = pd.read_csv(TABLO, dtype=str).fillna("")
krl = pd.read_csv(KIRILMA)
m = pd.read_csv(GIRDI, low_memory=False)
m["tarih"] = pd.to_datetime(m["tarih"])
meta = pd.read_csv(META, encoding="cp1254").set_index("istasyon_no")
for c in SUT.values():
    m[c + "_ham"] = m[c]

ref = {d: m.pivot_table(index="tarih", columns="istasyon_no", values=SUT[d]) for d in DER}
for _, r in tablo[tablo.islem == "sil"].iterrows():
    s = int(r.istasyon_no)
    for d in derinlikler(r.derinlik):
        if s in ref[d]:
            ref[d].loc[r.baslangic:r.bitis, s] = np.nan
KOMSU = {}


def komsular(s):
    if s not in KOMSU:
        la, lo = np.radians(meta.lat), np.radians(meta.lon)
        a = np.sin((la - la[s]) / 2) ** 2 + np.cos(la) * np.cos(la[s]) * np.sin((lo - lo[s]) / 2) ** 2
        km = (6371 * 2 * np.arcsin(np.sqrt(a))).drop(s).sort_values()
        KOMSU[s] = [j for j in km.index if j in ref[DER[0]].columns][:N_KOMSU]
    return KOMSU[s]


def incelt(s, d, ay, isaret):
    """Ay basindaki kirilmayi gune indir: +-INCELT gun icinde iki yan ortalama farki en buyuk gun."""
    f = (ref[d][s] - ref[d][komsular(s)].median(axis=1))
    if SON_TARIH is not None:
        f = f.loc[:SON_TARIH]
    t0 = pd.Period(ay, "M").to_timestamp()
    w = f.loc[t0 - pd.Timedelta(days=INCELT + PENCERE_INCE):t0 + pd.Timedelta(days=INCELT + PENCERE_INCE)]
    deg = w.fillna(0).values
    say = w.notna().astype(int).values
    cd, cs = np.concatenate([[0], np.cumsum(deg)]), np.concatenate([[0], np.cumsum(say)])
    en, en_t = -np.inf, t0
    for i, t in enumerate(w.index):
        if abs((t - t0).days) > INCELT or i < PENCERE_INCE or i + PENCERE_INCE > len(w):
            continue
        na, nb = cs[i] - cs[i - PENCERE_INCE], cs[i + PENCERE_INCE] - cs[i]
        if na < 60 or nb < 60:
            continue
        v = ((cd[i + PENCERE_INCE] - cd[i]) / nb - (cd[i] - cd[i - PENCERE_INCE]) / na) * isaret
        if v > en:
            en, en_t = v, t
    return en_t


kirilmalar, atlanan = {}, []
for _, r in krl.iterrows():
    s, d = int(r.istasyon_no), int(r.derinlik)
    t = incelt(s, d, r.ay, np.sign(r.adim_medyan))
    if t > SON_KIRILMA and not TEST_KIRILMA:
        atlanan.append({"istasyon_no": s, "istasyon_ad": r.istasyon_ad, "derinlik": d, "ay": r.ay,
                        "tarih": t.date(), "adim_tarama": r.adim_medyan,
                        "neden": "test donemine yakin/icinde; adim test verisi olmadan tahmin edilemez"})
        continue
    kirilmalar.setdefault((s, d), []).append(t)
for k in kirilmalar:
    kirilmalar[k] = sorted(kirilmalar[k])


def pencereler(s, d, i):
    """i. kirilma icin once/sonra pencere sinirlari (komsu parcalar icinde, en fazla 3 yil)."""
    b = kirilmalar[(s, d)]
    B = b[i]
    once_bas = max(b[i - 1] if i > 0 else pd.Timestamp("1900-01-01"), B - pd.DateOffset(years=3))
    sonra_bas = B + pd.Timedelta(days=ATLA)
    sinir = SON_TAHMIN if (B <= SON_KIRILMA and (SON_TAHMIN - sonra_bas).days >= MIN_GUN) or not TEST_KIRILMA else SON_TEST
    sonra_son = min(b[i + 1] - pd.Timedelta(days=1) if i + 1 < len(b) else pd.Timestamp("2100-01-01"),
                    sonra_bas + pd.DateOffset(years=3), sinir)
    return once_bas, B, sonra_bas, sonra_son


def duzeltme_serisi(s, d, adimlar, idx):
    """Her gune, kendisinden sonraki kirilmalarin adimlarinin toplami (en son parca = 0)."""
    out = np.zeros(len(idx))
    for B, k in zip(kirilmalar[(s, d)], adimlar):
        if k is None:
            continue
        sec = idx < B
        out[sec] += egri(k, idx[sec])
    return out


def fark(a, b):
    v = []
    for x, y in zip(a, b):
        if x is None and y is None:
            v.append(0.0)
        elif x is None or y is None:
            v.append(np.inf)
        else:
            v.append(abs(x[0] - y[0]))
    return max(v)


adim_tum, kayit = {}, []
for tur in range(1, 16):
    R = {d: ref[d].copy() for d in DER}
    for (s, d), adimlar in adim_tum.items():
        R[d][s] = R[d][s] + duzeltme_serisi(s, d, adimlar, R[d].index)
    yeni, kayit = {}, []
    for (s, d), b in kirilmalar.items():
        x = ref[d][s]
        adimlar = []
        for i, B in enumerate(b):
            ob, B_, sb, ss = pencereler(s, d, i)
            farklar, kom = [], []
            for j in komsular(s):
                dj = (x - R[d][j]).dropna()
                once = dj.loc[ob:B_ - pd.Timedelta(days=1)]
                sonra = dj.loc[sb:ss]
                if len(once) < MIN_GUN or len(sonra) < MIN_GUN:
                    continue
                farklar.append(harmonik(sonra.index, sonra.values) - harmonik(once.index, once.values))
                kom.append(j)
            if len(farklar) < 2:
                adimlar.append(None)
                kayit.append({"istasyon_no": s, "istasyon_ad": meta.loc[s, "istasyon_adi"], "derinlik": d,
                              "kirilma": B.date(), "durum": "yetersiz komsu/veri", "n_komsu": len(farklar)})
                continue
            k = np.median(np.vstack(farklar), axis=0)
            adimlar.append(k)
            kayit.append({"istasyon_no": s, "istasyon_ad": meta.loc[s, "istasyon_adi"], "derinlik": d,
                          "kirilma": B.date(), "durum": "duzeltildi", "n_komsu": len(farklar),
                          "test_verisi": bool(ss > SON_TAHMIN),
                          "adim_ort": round(k[0], 3), "cos": round(k[1], 3), "sin": round(k[2], 3),
                          "komsu_adimlari": ";".join(f"{j}:{f[0]:+.2f}" for j, f in zip(kom, farklar)),
                          "komsu_yayilim": round(float(np.ptp([f[0] for f in farklar])), 2)})
        yeni[(s, d)] = adimlar
    degisim = max((fark(yeni[k], adim_tum[k]) if k in adim_tum else np.inf) for k in yeni)
    adim_tum = yeni
    print(f"tur {tur}: en buyuk adim degisimi {degisim:.3f} C", flush=True)
    if degisim < 0.01:
        break

for (s, d), adimlar in adim_tum.items():
    sec = m.istasyon_no == s
    m.loc[sec, SUT[d]] = m.loc[sec, SUT[d]] + duzeltme_serisi(s, d, adimlar, pd.DatetimeIndex(m.loc[sec, "tarih"]))

uygula = tablo[tablo.onay.str.strip() == "evet"]
for _, r in uygula.iterrows():
    s = int(r.istasyon_no)
    if r.islem == "sil":
        for d in derinlikler(r.derinlik):
            sec = (m.istasyon_no == s) & (m.tarih >= r.baslangic) & (m.tarih <= r.bitis)
            m.loc[sec, SUT[d]] = np.nan
            print(f"{s} {d:>3} cm sil {r.baslangic}..{r.bitis}: {int(sec.sum())} satir")
    elif r.islem == "cikar":
        n = int((m.istasyon_no == s).sum())
        m = m[m.istasyon_no != s]
        print(f"{s} cikarildi: {n} satir")

kay = pd.DataFrame(kayit)
print(f"\nDuzeltilen kirilma: {(kay.durum == 'duzeltildi').sum()} | yetersiz veri: {(kay.durum != 'duzeltildi').sum()} | "
      f"test donemi nedeniyle atlanan: {len(atlanan)} | test verisiyle duzeltilen: "
      f"{int(kay.get('test_verisi', pd.Series(dtype=bool)).fillna(False).astype(bool).sum())} | etkilenen istasyon: {kay[kay.durum == 'duzeltildi'].istasyon_no.nunique()}")
m["tarih"] = m["tarih"].dt.strftime("%Y-%m-%d")
m.to_csv(CIKTI, index=False, encoding="utf-8-sig")
kay.to_csv(CIKTI_K, index=False, encoding="utf-8-sig")
pd.DataFrame(atlanan, columns=["istasyon_no", "istasyon_ad", "derinlik", "ay", "tarih", "adim_tarama", "neden"]).to_csv(CIKTI_A, index=False, encoding="utf-8-sig")
print(f"Kaydedildi: {CIKTI} ({len(m):,} satir) | {os.path.basename(CIKTI_K)} | {os.path.basename(CIKTI_A)}")
