import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import contains_xy

ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOK = os.path.dirname(ANA_DIZIN)
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))
from toprak_katlar import N_KAT, kat_atamasi
from ortak_veri import haversine_km

ADIM_KM = 5.0
ESIK_KM = 133.0
SINIR = os.path.join(KOK, "data_preparation", "smap_data", "ne_10m_admin_0_countries_tur",
                     "ne_10m_admin_0_countries_tur.shp")
CIKTI = os.path.join(ANA_DIZIN, "analysis", "geometry_coverage")


def ozet(x):
    return {"n": len(x), "medyan": np.median(x), "p10": np.percentile(x, 10), "p90": np.percentile(x, 90),
            "en_buyuk": np.max(x), "ortalama": np.mean(x)}


def main():
    os.makedirs(CIKTI, exist_ok=True)
    st = pd.read_csv(os.path.join(KOK, "DATASET", "MGM_istasyonlar_final.csv"), encoding="latin-1")
    kat = kat_atamasi()
    st = st[st.istasyon_no.isin(list(kat))].reset_index(drop=True)
    assert len(st) == 82, len(st)
    st["kat"] = st.istasyon_no.map(kat)
    lat, lon = st.lat.to_numpy(float), st.lon.to_numpy(float)

    # test istasyonlari: en yakin gorunur (diger katlardaki) istasyon
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    ayni_kat = st.kat.to_numpy()[:, None] == st.kat.to_numpy()[None, :]
    test_nn = np.where(ayni_kat, np.inf, d).min(axis=1)
    st["en_yakin_gorunur_km"] = test_nn

    tr = gpd.read_file(SINIR)
    tr = tr[tr.ADM0_A3 == "TUR"].to_crs("EPSG:4326")      # dosya 250 ulke iceriyor
    assert len(tr) == 1, len(tr)
    poligon = tr.geometry.union_all() if hasattr(tr.geometry, "union_all") else tr.geometry.unary_union
    laea = "+proj=laea +lat_0=39 +lon_0=35 +datum=WGS84 +units=m"
    ileri = Transformer.from_crs("EPSG:4326", laea, always_xy=True)
    geri = Transformer.from_crs(laea, "EPSG:4326", always_xy=True)
    poli_m = gpd.GeoSeries([poligon], crs="EPSG:4326").to_crs(laea).iloc[0]
    x0, y0, x1, y1 = poli_m.bounds
    s = ADIM_KM * 1000.0
    gx, gy = np.meshgrid(np.arange(x0 + s / 2, x1, s), np.arange(y0 + s / 2, y1, s))
    gx, gy = gx.ravel(), gy.ravel()
    ic = contains_xy(poli_m, gx, gy)
    glon, glat = geri.transform(gx[ic], gy[ic])
    izgara_nn = np.full(len(glat), np.inf)
    for i in range(0, len(glat), 20000):
        dd = haversine_km(glat[i:i + 20000, None], glon[i:i + 20000, None], lat[None, :], lon[None, :])
        izgara_nn[i:i + 20000] = dd.min(axis=1)
    alan_km2 = poli_m.area / 1e6

    tab = pd.DataFrame({"test istasyonlari: en yakin gorunur komsu (km)": ozet(test_nn),
                        "kara izgarasi: en yakin istasyon (km)": ozet(izgara_nn)}).T
    print(f"Izgara: {ADIM_KM:.0f} km, {int(ic.sum())} kara noktasi (poligon alani {alan_km2:,.0f} km2, "
          f"nokta x {ADIM_KM:.0f}^2 = {ic.sum() * ADIM_KM ** 2:,.0f} km2)")
    print(tab.round(1).to_string())
    m = test_nn.max()
    print(f"\nKara alaninin en yakin istasyona uzakligi > {ESIK_KM:.0f} km olan payi: {np.mean(izgara_nn > ESIK_KM):.4f}")
    print(f"Test istasyonlarinin en buyuk en-yakin-komsu uzakligi: {m:.1f} km; kara alaninin bunu asan payi: "
          f"{np.mean(izgara_nn > m):.4f}")
    for e in (50, 83, 100):
        print(f"  pay(izgara > {e} km) = {np.mean(izgara_nn > e):.3f}   pay(test > {e} km) = {np.mean(test_nn > e):.3f}")
    print(f"\nIstasyon rakim araligi: {st.rakim.min():.0f}-{st.rakim.max():.0f} m (medyan {st.rakim.median():.0f}). "
          f"Kara alani rakimi icin DEM gerekiyor; projede DEM bulunamadi.")
    tab.to_csv(os.path.join(CIKTI, "uzaklik_ozet.csv"))
    st[["istasyon_no", "istasyon_adi", "kat", "lat", "lon", "rakim", "en_yakin_gorunur_km"]].to_csv(
        os.path.join(CIKTI, "test_en_yakin_gorunur.csv"), index=False)
    pd.DataFrame({"lat": glat, "lon": glon, "en_yakin_istasyon_km": izgara_nn}).to_csv(
        os.path.join(CIKTI, "izgara_5km.csv"), index=False)


if __name__ == "__main__":
    main()
