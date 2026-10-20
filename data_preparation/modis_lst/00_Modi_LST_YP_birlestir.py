import pandas as pd
from pathlib import Path

KLASOR = Path(__file__).resolve().parent
GIRDI  = KLASOR / 'MODIS_LST_YP'
CIKTI  = KLASOR / 'modis_lst_gunluk.csv'
KIYI5  = KLASOR.parents[1] / 'MODIS_Turkiye' / 'MODIS_LST_KIYI5.csv'
KIYI5_ISTASYON = [17026, 17040, 17056, 17302, 17320]
YILLAR = list(range(2015, 2026))
SUTUNLAR = ['istasyon_no', 'istasyon_ad', 'nokta', 'tarih', 'LST_Day_C', 'LST_Night_C',
            'QC_Day', 'QC_Night', 'Day_view_time', 'Night_view_time']

print("Dosyalar okunuyor...")
parcalar = []
for yil in YILLAR:
    dosya = GIRDI / f'MODIS_LST_YP_{yil}.csv'
    if not dosya.exists():
        raise FileNotFoundError(f"{dosya} yok")
    df = pd.read_csv(dosya, usecols=SUTUNLAR)
    parcalar.append(df)
    print(f"  {dosya.name} yuklendi -- {len(df):,} satir")

tum = pd.concat(parcalar, ignore_index=True)
del parcalar
if tum.duplicated(['istasyon_no', 'nokta', 'tarih']).any():
    raise ValueError("Ayni istasyon/nokta/tarih icin birden fazla satir var")

ist  = tum[tum['nokta'] == 'istasyon']
kara = tum[tum['nokta'] == 'kara_komsu']

# Istasyon pikseli su: tum donemde gunduz ve gece LST hic yok
doluluk = ist.groupby('istasyon_no')[['LST_Day_C', 'LST_Night_C']].count().sum(axis=1)
su_piksel = sorted(doluluk[doluluk == 0].index)
kara_var  = sorted(kara['istasyon_no'].unique())
print(f"\nIstasyon pikseli su: {su_piksel}")
print(f"Kara pikseli olan  : {kara_var}")
if su_piksel != kara_var:
    raise ValueError("Pikseli su olan istasyonlar ile kara pikseli olanlar ayni degil")

k5 = pd.read_csv(KIYI5, usecols=SUTUNLAR)
if sorted(k5['istasyon_no'].unique()) != KIYI5_ISTASYON or set(k5['nokta']) != {'kara_komsu'}:
    raise ValueError(f"KIYI5 dosyasi beklenen 5 istasyonun kara pikselini icermiyor: {sorted(k5['istasyon_no'].unique())}")
if k5.duplicated(['istasyon_no', 'tarih']).any():
    raise ValueError("KIYI5 dosyasinda tekrar eden istasyon/tarih var")
print(f"KIYI5 (deniz karisik, kara pikseliyle degisen): {KIYI5_ISTASYON} -- {len(k5):,} satir")
degisen = su_piksel + KIYI5_ISTASYON
birlesik = pd.concat([ist[~ist['istasyon_no'].isin(degisen)], kara, k5], ignore_index=True)
birlesik = birlesik.rename(columns={'nokta': 'lst_kaynak'})
if birlesik.duplicated(['istasyon_no', 'tarih']).any():
    raise ValueError("Birlesik dosyada tekrar eden istasyon/tarih var")

birlesik['tarih'] = pd.to_datetime(birlesik['tarih'])
birlesik = birlesik.sort_values(['istasyon_no', 'tarih'])
birlesik = birlesik[['istasyon_no', 'istasyon_ad', 'tarih', 'LST_Day_C', 'LST_Night_C',
                     'QC_Day', 'QC_Night', 'Day_view_time', 'Night_view_time', 'lst_kaynak']]
birlesik.to_csv(CIKTI, index=False, encoding='utf-8-sig', date_format='%Y-%m-%d')

print(f"\n{'=' * 50}")
print(f"  Kaydedildi       : {CIKTI.name}")
print(f"  Toplam satir     : {len(birlesik):,}")
print(f"  Istasyon         : {birlesik['istasyon_no'].nunique()}")
print(f"  Tarih araligi    : {birlesik['tarih'].min().date()} -> {birlesik['tarih'].max().date()}")
print(f"  LST_Day  doluluk : {birlesik['LST_Day_C'].notna().mean()*100:.1f}%")
print(f"  LST_Night doluluk: {birlesik['LST_Night_C'].notna().mean()*100:.1f}%")
print(f"  LST hic olmayan istasyon: "
      f"{(birlesik.groupby('istasyon_no')['LST_Day_C'].count() == 0).sum()}")
print(f"{'=' * 50}")
