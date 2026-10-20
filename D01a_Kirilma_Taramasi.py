import os
import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.abspath(__file__))
DER = [5, 10, 20, 50, 100]
SUT = {d: f"t{d}cmsicaklik" for d in DER}
N_KOMSU = 7
MIN_GUN_AY = 20
MIN_PARCA = 12        # ay
# esikler ortam degiskeniyle ezilebilir (duyarlilik testi): KIRILMA_T, _ADIM, _ISABET, _ORAN, _ETIKET
T_ESIK = float(os.environ.get("KIRILMA_T", 15.0))       # SNHT; ozilisi yuzunden %95 tablo degerinden (~9) siki
MIN_ADIM = float(os.environ.get("KIRILMA_ADIM", 0.4))   # C
PENCERE = 4                                             # ay
MIN_ISABET = int(os.environ.get("KIRILMA_ISABET", 3))
MIN_ORAN = float(os.environ.get("KIRILMA_ORAN", 0.5))
ETIKET = os.environ.get("KIRILMA_ETIKET", "")
SON_TARIH = os.environ.get("QC_SON_TARIH", "")

m = pd.read_csv(os.path.join(KOK, "DATASET", os.environ.get("KIRILMA_MASTER", "master_dataset.csv")),
                usecols=["istasyon_no", "tarih"] + list(SUT.values()), parse_dates=["tarih"])
if SON_TARIH:
    m = m[m.tarih <= pd.Timestamp(SON_TARIH)]
meta = pd.read_csv(os.path.join(KOK, "DATASET", "MGM_istasyonlar_final.csv"), encoding="cp1254").set_index("istasyon_no")
ist = sorted(m.istasyon_no.unique())
m["ay"] = m.tarih.dt.to_period("M")
aylar = pd.period_range(m.ay.min(), m.ay.max(), freq="M")
A = {}
for d in DER:
    g = m.groupby(["ay", "istasyon_no"])[SUT[d]].agg(["mean", "count"])
    v = g["mean"].where(g["count"] >= MIN_GUN_AY).unstack()
    A[d] = v.reindex(index=aylar, columns=ist)

la, lo = np.radians(meta.loc[ist, "lat"].values), np.radians(meta.loc[ist, "lon"].values)
a = (np.sin((la[:, None] - la[None, :]) / 2) ** 2
     + np.cos(la[:, None]) * np.cos(la[None, :]) * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2)
km = 6371 * 2 * np.arcsin(np.sqrt(a))
np.fill_diagonal(km, np.inf)
komsu = {s: [ist[j] for j in np.argsort(km[i])[:N_KOMSU]] for i, s in enumerate(ist)}
ciftler = sorted({tuple(sorted((s, j))) for s in ist for j in komsu[s]})


def ikili_bolme(z, bas, son, sonuc, t_esik=T_ESIK, min_adim=MIN_ADIM):
    """z: numpy dizisi (NaN'siz), bas..son yari acik aralik; bulunan kirilma indekslerini ekler."""
    y = z[bas:son]
    n = len(y)
    if n < 2 * MIN_PARCA:
        return
    c = np.cumsum(y)
    k = np.arange(MIN_PARCA, n - MIN_PARCA + 1)
    m1 = c[k - 1] / k
    m2 = (c[-1] - c[k - 1]) / (n - k)
    tss = ((y - y.mean()) ** 2).sum()
    ess = tss - k * (n - k) / n * (m1 - m2) ** 2
    s2 = ess / (n - 2)
    T = k * (n - k) / n * (m1 - m2) ** 2 / np.maximum(s2, 1e-9)
    i = int(np.argmax(T))
    if T[i] < t_esik or abs(m2[i] - m1[i]) < min_adim:
        return
    kk = bas + int(k[i])
    sonuc.append((kk, float(m2[i] - m1[i]), float(T[i])))
    ikili_bolme(z, bas, kk, sonuc, t_esik, min_adim)
    ikili_bolme(z, kk, son, sonuc, t_esik, min_adim)


def cift_kirilmalari(A, t_esik=T_ESIK, min_adim=MIN_ADIM):
    satir, kapsam = [], {}
    for d in DER:
        M = A[d]
        for i, j in ciftler:
            f = (M[i] - M[j])
            f = f - f.groupby(f.index.month).transform("mean")
            z = f.dropna()
            if len(z) < 2 * MIN_PARCA:
                continue
            pos = np.array([aylar.get_loc(p) for p in z.index])
            kapsam[(d, i, j)] = pos
            bul = []
            ikili_bolme(z.values, 0, len(z), bul, t_esik, min_adim)
            for kk, adim, T in bul:
                satir.append({"derinlik": d, "i": i, "j": j, "ay": str(z.index[kk]), "ay_idx": int(pos[kk]),
                              "adim_i_eksi_j": round(adim, 3), "T": round(T, 1)})
    cols = ["derinlik", "i", "j", "ay", "ay_idx", "adim_i_eksi_j", "T"]
    return pd.DataFrame(satir, columns=cols), kapsam


def kapsiyor(pos, t):
    return (pos < t - MIN_PARCA // 2).sum() >= MIN_PARCA and (pos >= t + MIN_PARCA // 2).sum() >= MIN_PARCA


def ata(cift, kapsam, min_isabet=MIN_ISABET, min_oran=MIN_ORAN):
    atanan = []
    for d in DER:
        h = cift[cift.derinlik == d].reset_index(drop=True)
        if h.empty:
            continue
        I, J, T, AD = h.i.values, h.j.values, h.ay_idx.values, h.adim_i_eksi_j.values
        acik = np.ones(len(h), bool)
        kap_bellek = {}
        while True:
            en = None
            for s in ist:
                idx = np.where(acik & ((I == s) | (J == s)))[0]
                if len(idx) < min_isabet:
                    continue
                for t in np.unique(T[idx]):
                    yak = idx[np.abs(T[idx] - t) <= PENCERE]
                    dg = np.where(I[yak] == s, J[yak], I[yak])
                    _, ilk = np.unique(dg, return_index=True)          # ayni ciftten tek isabet
                    yak = yak[np.sort(ilk)]
                    isabet = len(yak)
                    if isabet < min_isabet:
                        continue
                    if (s, t) not in kap_bellek:
                        kap_bellek[(s, t)] = sum(1 for j in komsu[s] if (d, *sorted((s, j))) in kapsam
                                                 and kapsiyor(kapsam[(d, *sorted((s, j)))], t))
                    kap = max(kap_bellek[(s, t)], isabet)
                    oran = isabet / kap
                    if oran < min_oran:
                        continue
                    if en is None or (isabet, oran) > en[0]:
                        en = ((isabet, oran), s, t, yak)
            if en is None:
                break
            (isabet, oran), s, t, yak = en
            isaret = np.where(I[yak] == s, 1.0, -1.0)
            adimlar = AD[yak] * isaret
            tarih = aylar[int(np.median(T[yak]))]
            atanan.append({"istasyon_no": s, "istasyon_ad": meta.loc[s, "istasyon_adi"], "derinlik": d,
                           "ay": str(tarih), "isabet": isabet, "kapsam": int(round(isabet / oran)),
                           "oran": round(oran, 2), "adim_medyan": round(float(np.median(adimlar)), 2),
                           "adim_min": round(float(adimlar.min()), 2), "adim_max": round(float(adimlar.max()), 2),
                           "ciftler": ",".join(str(int(v)) for v in np.where(I[yak] == s, J[yak], I[yak]))})
            acik[yak] = False
    cols = ["istasyon_no", "istasyon_ad", "derinlik", "ay", "isabet", "kapsam", "oran",
            "adim_medyan", "adim_min", "adim_max", "ciftler"]
    atn = pd.DataFrame(atanan, columns=cols)
    return atn.sort_values(["istasyon_no", "ay", "derinlik"]).reset_index(drop=True)


def olaylar(atn):
    olay = []
    for s, g in atn.groupby("istasyon_no"):
        g = g.assign(idx=[aylar.get_loc(pd.Period(a, "M")) for a in g.ay]).sort_values("idx")
        kume, son = [], None
        for _, r in g.iterrows():
            if son is not None and r.idx - son > PENCERE:
                olay.append(kume)
                kume = []
            kume.append(r)
            son = r.idx
        olay.append(kume)
    cols = ["istasyon_no", "istasyon_ad", "ay", "derinlik_sayisi", "derinlikler", "adimlar", "tur"]
    return pd.DataFrame([{
        "istasyon_no": k[0].istasyon_no, "istasyon_ad": k[0].istasyon_ad,
        "ay": str(aylar[int(np.median([r.idx for r in k]))]),
        "derinlik_sayisi": len({r.derinlik for r in k}),
        "derinlikler": ";".join(str(r.derinlik) for r in sorted(k, key=lambda r: r.derinlik)),
        "adimlar": ";".join(f"{r.adim_medyan:+.2f}" for r in sorted(k, key=lambda r: r.derinlik)),
        "tur": "yer/parsel" if len({r.derinlik for r in k}) >= 3 else ("sensor" if len(k) == 1 else "2 derinlik")}
        for k in olay if k], columns=cols)


if __name__ == "__main__" and os.environ.get("D01A_SADECE_FONKSIYON") != "1":
    cift, kapsam = cift_kirilmalari(A)
    cift.to_csv(os.path.join(KOK, "DATASET", f"kirilma_ciftler{ETIKET}.csv"), index=False, encoding="utf-8-sig")
    print(f"Cift sayisi {len(ciftler)} | cift kirilmasi (5 derinlik toplam) {len(cift)}")
    atn = ata(cift, kapsam)
    atn.to_csv(os.path.join(KOK, "DATASET", f"kirilma_istasyon{ETIKET}.csv"), index=False, encoding="utf-8-sig")
    ol = olaylar(atn)
    ol.to_csv(os.path.join(KOK, "DATASET", f"kirilma_olay{ETIKET}.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(f"Atanan kirilma: {len(atn)} (istasyon x derinlik) | olay: {len(ol)} | "
          f"kirilmasi olan istasyon: {ol.istasyon_no.nunique()}/82")
    print(ol.tur.value_counts().to_dict())
    print(ol.sort_values(["derinlik_sayisi"], ascending=False).to_string(index=False))
