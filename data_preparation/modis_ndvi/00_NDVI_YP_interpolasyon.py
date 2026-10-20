import glob
import os
import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.abspath(os.path.join(BURA, "..", ".."))
GIRDI = sorted(glob.glob(os.path.join(KOK, "MODIS_Turkiye", "MODIS_NDVI_YP_*.csv")))
CIKTI = os.path.join(BURA, "ndvi_gunluk.csv")
KARA_PIKSEL = {17030, 17033, 17034, 17112, 17119, 17310, 17340}   # modis_lst_yerel_piksel.js ile ayni
KIYI5 = os.path.join(KOK, "MODIS_Turkiye", "MODIS_NDVI_KIYI5.csv")
KIYI5_ISTASYON = {17026, 17040, 17056, 17302, 17320}            # modis_kiyi5_kara_pikseli.js
KARA_PIKSEL = KARA_PIKSEL | KIYI5_ISTASYON
BULUT_MUAF = {17320}
BAS, BIT = "2015-01-01", "2025-12-31"
UC_GUN = 48

k5 = pd.read_csv(KIYI5)
if set(k5.istasyon_no) != KIYI5_ISTASYON or set(k5.nokta) != {"kara_komsu"}:
    raise ValueError(f"KIYI5 NDVI dosyasi beklenen 5 kara pikselini icermiyor: {sorted(set(k5.istasyon_no))}")
y = pd.concat([pd.read_csv(f) for f in GIRDI] + [k5], ignore_index=True)
y["tarih"] = pd.to_datetime(y.tarih)
print(f"Girdi: {len(GIRDI)} dosya, {len(y):,} satir")

sec = y[((y.nokta == "istasyon") & ~y.istasyon_no.isin(KARA_PIKSEL)) |
        ((y.nokta == "kara_komsu") & y.istasyon_no.isin(KARA_PIKSEL))].copy()
bulut = (sec.SummaryQA == 3) & ~sec.istasyon_no.isin(BULUT_MUAF)
print(f"Secilen: {len(sec):,} satir, {sec.istasyon_no.nunique()} istasyon | NDVI dolu {sec.NDVI.notna().sum():,} | "
      f"bulut nedeniyle atilan {int((bulut & sec.NDVI.notna()).sum()):,} | Anamur'da muaf tutulan "
      f"{int(((sec.SummaryQA == 3) & sec.istasyon_no.isin(BULUT_MUAF) & sec.NDVI.notna()).sum())}")
sec.loc[bulut, "NDVI"] = np.nan
sec = sec.dropna(subset=["NDVI"])
sec = sec[sec.DayOfYear >= 1]

bas_doy = sec.tarih.dt.dayofyear
yil = sec.tarih.dt.year + (sec.DayOfYear < bas_doy).astype(int)
sec["gozlem"] = pd.to_datetime(yil.astype(str) + "-01-01") + pd.to_timedelta(sec.DayOfYear - 1, unit="D")
kayma = (sec.gozlem - sec.tarih).dt.days
assert kayma.between(0, 15).all(), "gozlem gunu donem disinda"
cakisma = int(sec.duplicated(["istasyon_no", "gozlem"]).sum())
gz = sec.groupby(["istasyon_no", "istasyon_ad", "nokta", "gozlem"], as_index=False).NDVI.mean()
print(f"Gecerli gozlem: {len(gz):,} (ayni gune dusen {cakisma} cift ortalandi)")

gun = pd.date_range(BAS, BIT, freq="D")
parca = []
for (s, ad, nk), g in gz.groupby(["istasyon_no", "istasyon_ad", "nokta"]):
    seri = g.set_index("gozlem").NDVI.reindex(gun)
    gercek = seri.notna()
    dolu = seri.interpolate(method="time", limit_area="inside")
    dolu = dolu.ffill(limit=UC_GUN).bfill(limit=UC_GUN)
    parca.append(pd.DataFrame({"istasyon_no": s, "istasyon_ad": ad, "tarih": gun, "NDVI": dolu.values,
                               "NDVI_interpolated": np.where(dolu.notna(), ~gercek.values, np.nan),
                               "ndvi_kaynak": nk}))
c = pd.concat(parca, ignore_index=True)
c["NDVI"] = c.NDVI.round(4)
c["NDVI_interpolated"] = c.NDVI_interpolated.map({1.0: True, 0.0: False})
c.to_csv(CIKTI, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

ara = gz.sort_values(["istasyon_no", "gozlem"]).groupby("istasyon_no").gozlem.diff().dt.days
print(f"\nKaydedildi: {CIKTI}")
print(f"  satir {len(c):,} | istasyon {c.istasyon_no.nunique()} | NDVI bos {int(c.NDVI.isna().sum())} | "
      f"gercek gozlem gunu {int((c.NDVI_interpolated == False).sum()):,} "
      f"(%{100 * (c.NDVI_interpolated == False).mean():.1f})")
print(f"  gozlemler arasi bosluk (gun): medyan {ara.median():.0f}, %99 {ara.quantile(.99):.0f}, en uzun {ara.max():.0f}")
print(f"  kaynak: {c.drop_duplicates('istasyon_no').ndvi_kaynak.value_counts().to_dict()}")
