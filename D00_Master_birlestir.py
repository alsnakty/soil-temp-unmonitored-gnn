import os
import pandas as pd
import numpy as np
import calendar
from pathlib import Path

KOK        = Path(__file__).resolve().parent
VERI_TEMIN = KOK / 'data_preparation'

KLASOR_LST       = VERI_TEMIN / 'modis_lst'
KLASOR_NDVI      = VERI_TEMIN / 'modis_ndvi'
KLASOR_SMAP      = VERI_TEMIN / 'smap_data'
KLASOR_SOILGRIDS = VERI_TEMIN / 'soilgrids'
KLASOR_MGM       = KOK / 'DATASET'

DOSYA_LST        = KLASOR_LST       / 'modis_lst_gunluk.csv'
DOSYA_NDVI       = KLASOR_NDVI      / 'ndvi_gunluk.csv'
DOSYA_SMAP       = KLASOR_SMAP      / 'soil_moisture_final.csv'
DOSYA_SOILGRIDS  = KLASOR_SOILGRIDS / 'soilgrids_istasyonlar_sonuc_wide.csv'
DOSYA_MGM        = KLASOR_MGM       / 'MGM_toprak_sicakligi_final.csv'

SOILGRIDS_DISLA = ['phh2o']

# D00_CIKTI ortam degiskeni ile farkli dosyaya yazilabilir (dogrulama icin)
CIKTI = Path(os.environ.get('D00_CIKTI', KOK / 'DATASET' / 'master_dataset.csv'))
CIKTI.parent.mkdir(exist_ok=True)

GECERLI_ISTASYONLAR = [
    17020, 17022, 17026, 17030, 17033, 17034, 17037, 17040, 17045, 17046,
    17050, 17052, 17056, 17064, 17066, 17069, 17070, 17072, 17074, 17077,
    17080, 17084, 17085, 17086, 17088, 17089, 17090, 17094, 17095, 17097,
    17099, 17100, 17112, 17116, 17119, 17120, 17126, 17130, 17135, 17140,
    17155, 17160, 17165, 17172, 17186, 17188, 17190, 17192, 17193, 17196,
    17199, 17201, 17203, 17204, 17208, 17210, 17220, 17234, 17237, 17238,
    17240, 17245, 17246, 17250, 17255, 17261, 17262, 17265, 17270, 17275,
    17282, 17285, 17287, 17292, 17302, 17310, 17320, 17340, 17351, 17355,
    17372, 17636,
]


print("=" * 60)
print("  Master Dataset Birlestirme")
print("=" * 60)

print("\n[1/6] MGM verisi okunuyor...")
mgm = pd.read_csv(DOSYA_MGM, sep=";", encoding="utf-8-sig")

mgm = mgm[mgm["yil"] <= 2025].copy()

mgm = mgm[mgm["istasyon_no"] != 17078].copy()

def gecerli_tarih(row):
    try:
        return int(row["gun"]) <= calendar.monthrange(int(row["yil"]), int(row["ay"]))[1]
    except:
        return False

mgm = mgm[mgm.apply(gecerli_tarih, axis=1)].copy()

mgm["tarih"] = pd.to_datetime(
    mgm[["yil", "ay", "gun"]].rename(columns={"yil": "year", "ay": "month", "gun": "day"})
)

mgm = mgm[mgm["istasyon_no"].isin(GECERLI_ISTASYONLAR)].copy()

mgm = mgm.drop(columns=["istasyon_adi", "yil", "ay", "gun"], errors="ignore")

mgm = mgm[mgm["tarih"] >= "2015-04-01"].copy()

print(f"  {len(mgm):,} satir | {mgm['istasyon_no'].nunique()} istasyon")
print(f"  Tarih: {mgm['tarih'].min().date()} -> {mgm['tarih'].max().date()}")

for col in ["t5cmsicaklik", "t10cmsicaklik", "t20cmsicaklik",
            "t50cmsicaklik", "t100cmsicaklik"]:
    dolu = mgm[col].notna().mean() * 100
    print(f"  {col}: {dolu:.1f}% dolu")

master = mgm.copy()

print("\n[2/6] MODIS LST okunuyor...")
lst = pd.read_csv(DOSYA_LST, usecols=["istasyon_no", "tarih", "LST_Day_C", "LST_Night_C"])
lst["tarih"] = pd.to_datetime(lst["tarih"])
lst = lst[lst["istasyon_no"].isin(GECERLI_ISTASYONLAR)]
print(f"  {len(lst):,} satir | LST_Day doluluk: {lst['LST_Day_C'].notna().mean()*100:.1f}%")

master = master.merge(lst, on=["istasyon_no", "tarih"], how="left")

print("\n[3/6] NDVI okunuyor...")
ndvi = pd.read_csv(DOSYA_NDVI, usecols=["istasyon_no", "tarih", "NDVI"])
ndvi["tarih"] = pd.to_datetime(ndvi["tarih"])
ndvi = ndvi[ndvi["istasyon_no"].isin(GECERLI_ISTASYONLAR)]
print(f"  {len(ndvi):,} satir | NDVI doluluk: {ndvi['NDVI'].notna().mean()*100:.1f}%")

master = master.merge(ndvi, on=["istasyon_no", "tarih"], how="left")

print("\n[4/6] Toprak nemi okunuyor...")
smap = pd.read_csv(DOSYA_SMAP, usecols=["istasyon_no", "tarih", "soil_moisture"])
smap["tarih"] = pd.to_datetime(smap["tarih"])
smap = smap[smap["istasyon_no"].isin(GECERLI_ISTASYONLAR)]
print(f"  {len(smap):,} satir | SM doluluk: {smap['soil_moisture'].notna().mean()*100:.1f}%")

master = master.merge(smap, on=["istasyon_no", "tarih"], how="left")

print("\n[5/6] SoilGrids okunuyor...")
soilgrids = pd.read_csv(DOSYA_SOILGRIDS)

if "Istasyon_No" in soilgrids.columns:
    soilgrids = soilgrids.rename(columns={"Istasyon_No": "istasyon_no"})

soilgrids = soilgrids.drop_duplicates(subset="istasyon_no")
soilgrids = soilgrids[soilgrids["istasyon_no"].isin(GECERLI_ISTASYONLAR)]
soilgrids = soilgrids.drop(columns=[c for c in soilgrids.columns
                                    if c.split('_')[0] in SOILGRIDS_DISLA])
print(f"  {soilgrids['istasyon_no'].nunique()} istasyon | {len(soilgrids.columns)-1} degisken")

eksik = sorted(set(GECERLI_ISTASYONLAR) - set(soilgrids["istasyon_no"]))
bos = soilgrids.loc[soilgrids.isna().any(axis=1), "istasyon_no"].tolist()
if eksik or bos:
    raise SystemExit(f"  HATA: SoilGrids eksik istasyon {eksik}, bos hucreli istasyon {bos}")

master = master.merge(soilgrids, on="istasyon_no", how="left")

print("\n[6/6] Zaman ozellikleri hesaplaniyor...")
master["gun_yili"] = master["tarih"].dt.dayofyear
master["gun_sin"]  = np.sin(2 * np.pi * master["gun_yili"] / 365.25)
master["gun_cos"]  = np.cos(2 * np.pi * master["gun_yili"] / 365.25)
master = master.drop(columns=["gun_yili"])

master = master.sort_values(["istasyon_no", "tarih"])
master["tarih"] = master["tarih"].dt.strftime("%Y-%m-%d")
master.to_csv(CIKTI, index=False, encoding="utf-8-sig")

print(f"\n{'=' * 60}")
print(f"  Kaydedildi     : {CIKTI}")
print(f"  Toplam satir   : {len(master):,}")
print(f"  Istasyon       : {master['istasyon_no'].nunique()}")
print(f"  Tarih araligi  : {master['tarih'].min()} -> {master['tarih'].max()}")
print(f"  Sutun sayisi   : {len(master.columns)}")
print(f"{'=' * 60}")
print("\nHedef degisken doluluk:")
for col in ["t5cmsicaklik", "t10cmsicaklik", "t20cmsicaklik",
            "t50cmsicaklik", "t100cmsicaklik"]:
    if col in master.columns:
        dolu = master[col].notna().sum()
        print(f"  {col}: {dolu:,} ({master[col].notna().mean()*100:.1f}%)")
