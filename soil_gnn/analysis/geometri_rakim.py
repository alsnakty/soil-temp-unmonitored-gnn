import os
import sys

import netCDF4
import numpy as np
import pandas as pd

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 200)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOK = os.path.dirname(ANA_DIZIN)
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))
from toprak_katlar import kat_atamasi

CIKTI = os.path.join(ANA_DIZIN, "analysis", "geometry_coverage")
DEM = os.path.join(KOK, "DATASET", "DEM", "etopo2022_60s_turkiye.nc")
DILIM = 500.0


def dem_oku(lat_q, lon_q, lat, lon, z):
    i = np.clip(np.round((lat_q - lat[0]) / (lat[1] - lat[0])).astype(int), 0, len(lat) - 1)
    j = np.clip(np.round((lon_q - lon[0]) / (lon[1] - lon[0])).astype(int), 0, len(lon) - 1)
    if np.abs(lat[i] - lat_q).max() > 0.6 / 60 or np.abs(lon[j] - lon_q).max() > 0.6 / 60:
        raise RuntimeError("en yakin hucre eslesmesi")
    return z[i, j]


def main():
    d = netCDF4.Dataset(DEM)
    lat, lon, z = d["lat"][:].data, d["lon"][:].data, np.asarray(d["z"][:], float)
    g = pd.read_csv(os.path.join(CIKTI, "izgara_5km.csv"))
    if len(g) != 31194:
        raise RuntimeError(f"izgara nokta sayisi {len(g)}")
    g["rakim_dem"] = dem_oku(g.lat.to_numpy(), g.lon.to_numpy(), lat, lon, z)
    st = pd.read_csv(os.path.join(KOK, "DATASET", "MGM_istasyonlar_final.csv"), encoding="latin-1")
    st = st[st.istasyon_no.isin(list(kat_atamasi()))].reset_index(drop=True)
    assert len(st) == 82, len(st)
    st["rakim_dem"] = dem_oku(st.lat.to_numpy(float), st.lon.to_numpy(float), lat, lon, z)
    fark = st.rakim_dem - st.rakim
    lo, hi = float(st.rakim.min()), float(st.rakim.max())
    r = g.rakim_dem.to_numpy()
    print(f"Saglama (istasyonda DEM - MGM rakimi, m): medyan {fark.median():.1f}, ort {fark.mean():.1f}, "
          f"|fark| en buyuk {fark.abs().max():.1f}, Pearson r {np.corrcoef(st.rakim_dem, st.rakim)[0, 1]:.4f}")
    print(f"Istasyon rakimi (MGM): {lo:.0f}-{hi:.0f} m, medyan {st.rakim.median():.0f}")
    print(f"Kara izgarasi rakimi (DEM): {r.min():.0f}-{r.max():.0f} m, medyan {np.median(r):.0f}, ortalama {r.mean():.0f}")
    ozet = {"pay_alt_sinir_alti (< %.0f m)" % lo: np.mean(r < lo), "pay_ust_sinir_ustu (> %.0f m)" % hi: np.mean(r > hi),
            "pay_aralik_disi": np.mean((r < lo) | (r > hi))}
    for k, v in ozet.items():
        print(f"  {k}: {v:.4f}")
    ust = int(np.ceil(max(r.max(), hi) / DILIM) * DILIM)
    kenar = np.concatenate([[-np.inf, 0.0], np.arange(DILIM, ust + DILIM, DILIM)])
    etk = ["< 0"] + [f"{int(a)}-{int(b)}" for a, b in zip(kenar[1:-1], kenar[2:])]
    gk = pd.cut(r, kenar, right=False, labels=etk)
    sk = pd.cut(st.rakim.to_numpy(float), kenar, right=False, labels=etk)
    tab = pd.DataFrame({"kara_pay": pd.Series(gk).value_counts(normalize=True).reindex(etk).fillna(0.0),
                        "kara_km2": pd.Series(gk).value_counts().reindex(etk).fillna(0).astype(int) * 25,
                        "istasyon": pd.Series(sk).value_counts().reindex(etk).fillna(0).astype(int)})
    tab["istasyon_pay"] = tab.istasyon / len(st)
    tab = tab[(tab.kara_km2 > 0) | (tab.istasyon > 0)]
    print("\n500 m dilimleri (alt sinir dahil):")
    print(tab.to_string(float_format=lambda v: f"{v:.3f}"))
    tab.to_csv(os.path.join(CIKTI, "rakim_dilim.csv"))
    pd.DataFrame([ozet]).to_csv(os.path.join(CIKTI, "rakim_aralik_disi.csv"), index=False)
    g.to_csv(os.path.join(CIKTI, "izgara_5km_rakim.csv"), index=False)
    st[["istasyon_no", "rakim", "rakim_dem"]].to_csv(os.path.join(CIKTI, "istasyon_rakim_dem.csv"), index=False)


if __name__ == "__main__":
    main()
