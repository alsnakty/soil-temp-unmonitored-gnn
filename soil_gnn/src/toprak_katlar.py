import numpy as np
import pandas as pd

from ortak_veri import haversine_km

N_KAT = 5
KAT_TOHUM = 2026
IC_TOHUM = 7          # ic dogrulama bolmesi (kat atamasindan bagimsiz tohum)

ISTASYON_BOLGE = {
    17020: "Karadeniz", 17022: "Karadeniz", 17026: "Karadeniz", 17030: "Karadeniz", 17033: "Karadeniz", 17034: "Karadeniz",
    17037: "Karadeniz", 17040: "Karadeniz", 17045: "Karadeniz", 17046: "Dogu Anadolu", 17050: "Marmara", 17052: "Marmara",
    17056: "Marmara", 17064: "Marmara", 17066: "Marmara", 17069: "Marmara", 17070: "Karadeniz", 17072: "Karadeniz",
    17074: "Karadeniz", 17077: "Karadeniz", 17080: "Ic Anadolu", 17084: "Karadeniz", 17085: "Karadeniz", 17086: "Karadeniz",
    17088: "Karadeniz", 17089: "Karadeniz", 17090: "Ic Anadolu", 17094: "Dogu Anadolu", 17095: "Dogu Anadolu", 17097: "Dogu Anadolu",
    17099: "Dogu Anadolu", 17100: "Dogu Anadolu", 17112: "Marmara", 17116: "Marmara", 17119: "Marmara", 17120: "Marmara",
    17126: "Ic Anadolu", 17130: "Ic Anadolu", 17135: "Ic Anadolu", 17140: "Ic Anadolu", 17155: "Ege", 17160: "Ic Anadolu",
    17165: "Dogu Anadolu", 17172: "Dogu Anadolu", 17186: "Ege", 17188: "Ege", 17190: "Ege", 17192: "Ic Anadolu",
    17193: "Ic Anadolu", 17196: "Ic Anadolu", 17199: "Dogu Anadolu", 17201: "Dogu Anadolu", 17203: "Dogu Anadolu", 17204: "Dogu Anadolu",
    17208: "Dogu Anadolu", 17210: "Guneydogu Anadolu", 17220: "Ege", 17234: "Ege", 17237: "Ege", 17238: "Akdeniz",
    17240: "Akdeniz", 17245: "Ic Anadolu", 17246: "Ic Anadolu", 17250: "Ic Anadolu", 17255: "Akdeniz", 17261: "Guneydogu Anadolu",
    17262: "Guneydogu Anadolu", 17265: "Guneydogu Anadolu", 17270: "Guneydogu Anadolu", 17275: "Guneydogu Anadolu", 17282: "Guneydogu Anadolu", 17285: "Dogu Anadolu",
    17287: "Guneydogu Anadolu", 17292: "Ege", 17302: "Akdeniz", 17310: "Akdeniz", 17320: "Akdeniz", 17340: "Akdeniz",
    17351: "Akdeniz", 17355: "Akdeniz", 17372: "Akdeniz", 17636: "Marmara",
}


def kat_atamasi(n_kat=N_KAT, tohum=KAT_TOHUM, bolge=None):
    """istasyon_no -> kat (0..n_kat-1). bolge verilmezse ISTASYON_BOLGE (82 istasyon)."""
    bolge = ISTASYON_BOLGE if bolge is None else bolge
    rng = np.random.RandomState(tohum)
    sira = []
    for b in sorted(set(bolge.values())):
        sira += list(rng.permutation(sorted(s for s, v in bolge.items() if v == b)))
    return {int(s): i % n_kat for i, s in enumerate(sira)}


def kat_istasyonlari(k, n_kat=N_KAT, tohum=KAT_TOHUM, bolge=None):
    if not 0 <= k < n_kat:
        raise ValueError(f"kat 0..{n_kat - 1} olmali: {k}")
    return tuple(sorted(s for s, kk in kat_atamasi(n_kat, tohum, bolge).items() if kk == k))


def ic_dogrulama_istasyonlari(egitim_no, n_grup=N_KAT, tohum=IC_TOHUM):
    b = {int(s): ISTASYON_BOLGE.get(int(s), "?") for s in egitim_no}
    return tuple(sorted(s for s, k in kat_atamasi(n_grup, tohum, b).items() if k == 0))


def en_yakin_gorunur(stations: pd.DataFrame, hedef_no, gorunur_no) -> pd.DataFrame:
    """Her hedef istasyon icin en yakin gorunur istasyon ve uzakligi (km); mesafe-hata analizi icin."""
    st = stations.set_index("istasyon_no")
    h, g = np.asarray(hedef_no), np.asarray(gorunur_no)
    D = haversine_km(st.loc[h, "lat"].to_numpy(float)[:, None], st.loc[h, "lon"].to_numpy(float)[:, None],
                     st.loc[g, "lat"].to_numpy(float)[None, :], st.loc[g, "lon"].to_numpy(float)[None, :])
    j = D.argmin(1)
    return pd.DataFrame({"istasyon_no": h, "en_yakin_gorunur_no": g[j], "en_yakin_gorunur_km": D[np.arange(len(h)), j],
                         "rakim_farki_m": st.loc[h, "rakim"].to_numpy(float) - st.loc[g[j], "rakim"].to_numpy(float),
                         "bolge": [ISTASYON_BOLGE.get(int(s), "") for s in h]})
