import rasterio
import pandas as pd
import numpy as np
import os

ISTASYON_DOSYA = 'MGM_istasyonlar_final.csv'
TIF_KLASOR     = 'SoilGrids_Turkiye_250m'

TIF_DOSYALAR = {
    '0-5cm'   : f'{TIF_KLASOR}/SoilGrids_TR_250m_0_5cm.tif',
    '5-15cm'  : f'{TIF_KLASOR}/SoilGrids_TR_250m_5_15cm.tif',
    '15-30cm' : f'{TIF_KLASOR}/SoilGrids_TR_250m_15_30cm.tif',
    '30-60cm' : f'{TIF_KLASOR}/SoilGrids_TR_250m_30_60cm.tif',
    '60-100cm': f'{TIF_KLASOR}/SoilGrids_TR_250m_60_100cm.tif',
}

# GEE'den gelen 12 bandin icinden secilecekler (1-based index)
SECILI_BANDLAR = {
    'sand'  : 1,
    'silt'  : 2,
    'clay'  : 3,
    'bdod'  : 4,
    'soc'   : 5,
    'phh2o' : 6,
    'cfvo'  : 9,
    'wv0033': 11,
    'wv1500': 12,
}

# Birim donusum faktorleri (GEE Inspector dogrulamali)
DONUSUM = {
    'sand'  : 0.1,    # g/kg -> %
    'silt'  : 0.1,    # g/kg -> %
    'clay'  : 0.1,    # g/kg -> %
    'bdod'  : 0.01,   # cg/cm3 -> kg/dm3
    'soc'   : 0.1,    # dg/kg -> g/kg
    'phh2o' : 0.1,    # pH*10 -> pH
    'cfvo'  : 0.1,    # cm3/dm3 -> %
    'wv0033': 1.0,    # cm3/cm3
    'wv1500': 1.0,    # cm3/cm3
}

BAND_ISIMLER = list(SECILI_BANDLAR.keys())
CIKTI_DOSYA  = 'soilgrids_istasyonlar_sonuc.csv'

if not os.path.exists(ISTASYON_DOSYA):
    print(f"HATA: {ISTASYON_DOSYA} bulunamadi.")
    exit()

# MGM_istasyonlar_final.csv: virgul ayracli, sutunlar kucuk harf, lat/lon hazir
df_ist = pd.read_csv(ISTASYON_DOSYA, encoding='cp1254')
df_ist.columns = df_ist.columns.str.strip()
df_ist['lat'] = pd.to_numeric(df_ist['lat'], errors='coerce')
df_ist['lon'] = pd.to_numeric(df_ist['lon'], errors='coerce')
df_ist = df_ist.dropna(subset=['lat', 'lon'])

print(f"Istasyon sayisi: {len(df_ist)}")


def spatial_search(data_array, row, col):
    for yari_cap in [1, 2, 4, 8, 16, 32, 64]:
        r_min = max(0, row - yari_cap)
        r_max = row + yari_cap + 1
        c_min = max(0, col - yari_cap)
        c_max = col + yari_cap + 1
        pencere = data_array[r_min:r_max, c_min:c_max]
        gecerli = pencere[~np.isnan(pencere)]
        if gecerli.size > 0:
            return np.mean(gecerli), yari_cap
    return np.nan, -1


tum_satirlar = []

for derinlik, tif_yol in TIF_DOSYALAR.items():
    if not os.path.exists(tif_yol):
        print(f"Atlanıyor (dosya yok): {tif_yol}")
        continue

    print(f"Isleniyor: {derinlik}  ({os.path.basename(tif_yol)})")

    with rasterio.open(tif_yol) as src:
        band_verileri = {}
        for b_ad, b_idx in SECILI_BANDLAR.items():
            arr = src.read(b_idx).astype(float)
            arr[arr < -1000] = np.nan   # SoilGrids NoData (-32768 vb.)
            band_verileri[b_ad] = arr

        raster_h, raster_w = band_verileri[BAND_ISIMLER[0]].shape

        for _, ist in df_ist.iterrows():
            try:
                row, col = src.index(ist['lon'], ist['lat'])

                if row < 0 or col < 0 or row >= raster_h or col >= raster_w:
                    continue

                satir = {
                    'istasyon_no': ist['istasyon_no'],
                    'istasyon_ad': ist['istasyon_adi'],
                    'il'         : ist['il'],
                    'lat'        : ist['lat'],
                    'lon'        : ist['lon'],
                    'rakim'      : ist['rakim'],
                    'derinlik'   : derinlik,
                }

                # arama_adimi: bantlar icindeki en genis arama yaricapi (piksel)
                adim_maks = 0
                for b_ad in BAND_ISIMLER:
                    val  = band_verileri[b_ad][row, col]
                    adim = 0

                    if np.isnan(val):
                        val, adim = spatial_search(band_verileri[b_ad], row, col)

                    satir[b_ad] = val * DONUSUM[b_ad] if not np.isnan(val) else np.nan
                    adim_maks = -1 if (adim == -1 or adim_maks == -1) else max(adim_maks, adim)

                satir['arama_adimi'] = adim_maks

                tum_satirlar.append(satir)

            except Exception:
                continue


df_sonuc = pd.DataFrame(tum_satirlar)

sutun_sirasi = ['istasyon_no', 'istasyon_ad', 'il', 'lat', 'lon', 'rakim',
                'derinlik', 'sand', 'arama_adimi', 'silt', 'clay', 'bdod',
                'soc', 'phh2o', 'cfvo', 'wv0033', 'wv1500']
df_sonuc = df_sonuc[sutun_sirasi]

df_sonuc.to_csv(CIKTI_DOSYA, index=False, encoding='utf-8-sig')
df_sonuc.to_excel(CIKTI_DOSYA.replace('.csv', '.xlsx'), index=False)
print(f"Long format kaydedildi: {CIKTI_DOSYA}")

print("Wide format olusturuluyor...")
meta_cols = ['istasyon_no', 'istasyon_ad', 'il', 'lat', 'lon', 'rakim']
df_wide = df_sonuc.pivot(index=meta_cols, columns='derinlik', values=BAND_ISIMLER)
df_wide.columns = [f"{var}_{dep}" for var, dep in df_wide.columns]
df_wide = df_wide.reset_index()

wide_csv  = CIKTI_DOSYA.replace('.csv', '_wide.csv')
wide_xlsx = CIKTI_DOSYA.replace('.csv', '_wide.xlsx')
df_wide.to_csv(wide_csv, index=False, encoding='utf-8-sig')
df_wide.to_excel(wide_xlsx, index=False)
print(f"Wide format kaydedildi: {wide_csv}")

print("\n--- 0-5cm Katmani Ortalama Degerler ---")
print(df_sonuc[df_sonuc['derinlik'] == '0-5cm'][BAND_ISIMLER].mean().round(2))
