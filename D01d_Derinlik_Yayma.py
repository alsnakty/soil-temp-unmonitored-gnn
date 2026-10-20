import os
import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.abspath(__file__))
DER = [5, 10, 20, 50, 100]
S = {d: f"t{d}cmsicaklik" for d in DER}
N_KOMSU, MIN_KOMSU, MIN_GUN, MIN_ADIM = 5, 4, 200, 0.5
OLAY_GUN = 60
TEK_AY = 12   # bir derinlige bu kadar ay icinde en fazla bir yayma (en buyuk |adim|); mevcut kirilmaya da bu kadar uzak olmali
PENCERE = pd.DateOffset(years=2)
ONCE_BIT, SONRA_BAS = pd.Timedelta(days=30), pd.Timedelta(days=60)
SON_TARIH = pd.Timestamp(os.environ["QC_SON_TARIH"]) if os.environ.get("QC_SON_TARIH") else None


def komsu_tablosu(meta):
    la, lo = np.radians(meta.lat), np.radians(meta.lon)
    out = {}
    for s in meta.index:
        a = np.sin((la - la[s]) / 2) ** 2 + np.cos(la) * np.cos(la[s]) * np.sin((lo - lo[s]) / 2) ** 2
        out[s] = list((6371 * 2 * np.arcsin(np.sqrt(a))).drop(s).sort_values().index[:N_KOMSU])
    return out


def olaylar(kay):
    kay = kay[kay.durum == "duzeltildi"].copy()
    kay["kirilma"] = pd.to_datetime(kay.kirilma)
    out = []
    for s, g in kay.sort_values("kirilma").groupby("istasyon_no"):
        grup = []
        for _, r in g.iterrows():
            if grup and (r.kirilma - grup[-1]["t"][-1]).days <= OLAY_GUN:
                grup[-1]["t"].append(r.kirilma)
                grup[-1]["d"].add(int(r.derinlik))
            else:
                grup.append({"t": [r.kirilma], "d": {int(r.derinlik)}})
        out += [(s, pd.Series(x["t"]).min(), x["d"]) for x in grup]
    return out


def yay(master, kay, meta, mevcut=None):
    master = master.copy()
    master["tarih"] = pd.to_datetime(master.tarih)
    P = {d: master.pivot(index="tarih", columns="istasyon_no", values=S[d]) for d in DER}
    kom = komsu_tablosu(meta)
    satir = []
    for s, B, dset in olaylar(kay):
        for d in DER:
            if d in dset or s not in P[d]:
                continue
            if mevcut is not None:
                x = mevcut[(mevcut.istasyon_no == s) & (mevcut.derinlik == d)]
                if any(abs((pd.Period(str(a)[:7], "M") - B.to_period("M")).n) <= TEK_AY for a in x.ay):
                    continue
            adim = []
            for j in kom[s]:
                if j not in P[d]:
                    continue
                f = (P[d][s] - P[d][j]).dropna()
                a = f.loc[B - PENCERE:B - ONCE_BIT]
                b = f.loc[B + SONRA_BAS:(B + PENCERE if SON_TARIH is None else min(B + PENCERE, SON_TARIH))]
                if len(a) >= MIN_GUN and len(b) >= MIN_GUN:
                    adim.append(b.mean() - a.mean())
            if len(adim) < MIN_KOMSU:
                continue
            adim = np.array(adim)
            med = float(np.median(adim))
            if abs(med) >= MIN_ADIM and (np.all(adim > 0) or np.all(adim < 0)):
                satir.append({"istasyon_no": s, "istasyon_ad": meta.istasyon_adi[s], "derinlik": d,
                              "ay": B.strftime("%Y-%m"), "isabet": len(adim), "kapsam": len(adim), "oran": 1.0,
                              "adim_medyan": round(med, 2), "adim_min": round(float(adim.min()), 2),
                              "adim_max": round(float(adim.max()), 2), "ciftler": ",".join(map(str, kom[s])),
                              "kaynak": "yayma", "olay_derinlik": ";".join(map(str, sorted(dset)))})
    y = pd.DataFrame(satir)
    if y.empty:
        return y
    kalan = []
    for _, g in y.assign(b=y.adim_medyan.abs()).sort_values("b", ascending=False).groupby(["istasyon_no", "derinlik"], sort=False):
        secilen = []
        for _, r in g.iterrows():
            if all(abs((pd.Period(r.ay, "M") - pd.Period(k.ay, "M")).n) > TEK_AY for k in secilen):
                secilen.append(r)
        kalan += secilen
    return pd.DataFrame(kalan).drop(columns="b").reset_index(drop=True)


if __name__ == "__main__":
    meta = pd.read_csv(os.path.join(KOK, "DATASET", "MGM_istasyonlar_final.csv"), encoding="cp1254").set_index("istasyon_no")
    m = pd.read_csv(os.path.join(KOK, "DATASET", "master_dataset_qc.csv"), usecols=["istasyon_no", "tarih"] + list(S.values()))
    kay = pd.read_csv(os.path.join(KOK, "DATASET", "hedef_qc_duzeltmeler.csv"))
    L = pd.read_csv(os.path.join(KOK, "DATASET", "kirilma_istasyon_birlesik.csv"))
    y = yay(m, kay, meta, L)
    pd.set_option("display.width", 220)
    print(f"onerilen yayma: {len(y)}")
    print(y[["istasyon_no", "istasyon_ad", "derinlik", "ay", "adim_medyan", "adim_min", "adim_max", "olay_derinlik"]].to_string(index=False))
