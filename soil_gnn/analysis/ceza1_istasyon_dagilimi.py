import os

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("compute.use_numexpr", False)

ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOK = os.path.dirname(ANA_DIZIN)
GIR = os.path.join(ANA_DIZIN, "results", "folds", "analysis_spatial_test")
OUT = os.path.join(ANA_DIZIN, "results", "folds", "analysis_penalty1_station")
ESIK = 0.10


def main():
    os.makedirs(OUT, exist_ok=True)
    t = pd.read_csv(os.path.join(GIR, "istasyon_tablosu.csv"), index_col="istasyon_no")
    ha = pd.read_csv(os.path.join(GIR, "hata_ayristirmasi_istasyon.csv"))
    sabit = ha[ha.model == "idw_rakim"].set_index("istasyon_no").sabit
    kars = pd.read_csv(os.path.join(GIR, "karsilastirma.csv"))
    ad = pd.read_csv(os.path.join(KOK, "DATASET", "MGM_istasyonlar_final.csv"), encoding="cp1254",
                     usecols=["istasyon_no", "istasyon_adi"]).set_index("istasyon_no").istasyon_adi

    df = pd.DataFrame({"istasyon_adi": ad.reindex(t.index), "bolge": t.bolge, "gelistirme_8": t.gelistirme_8,
                       "rmse_idw_rakim": t.rmse_idw_rakim, "rmse_hibrit_ceza1": t.rmse_hibrit_ceza1})
    df["d"] = df.rmse_hibrit_ceza1 - df.rmse_idw_rakim
    df["b_idw_rakim"] = np.sqrt(sabit.reindex(t.index))
    if df[["d", "b_idw_rakim"]].isna().any().any():
        raise SystemExit("eksik deger var")

    ozet = []
    for kume_ad, x in (("82 istasyon", df), ("74 istasyon (gelistirme 8 haric)", df[~df.gelistirme_8])):
        ref = kars[(kars.model == "hibrit_ceza1") & (kars.referans == "idw_rakim") & (kars.kume == kume_ad)
                   & kars.olcu.str.startswith("tohum")].ort_fark
        if len(ref) != 1 or abs(x.d.mean() - ref.iloc[0]) > 1e-9:
            raise SystemExit(f"{kume_ad}: d ortalamasi karsilastirma.csv ile ayni degil ({x.d.mean()} / {ref.tolist()})")
        s = x.sort_values("d")
        s.to_csv(os.path.join(OUT, f"siralama_{len(x)}.csv"))
        bes = s.d.iloc[:5].sum()
        rho, p = stats.spearmanr(x.d, x.b_idw_rakim)
        ozet.append(dict(kume=kume_ad, n=len(x), ort_d=x.d.mean(), toplam_d=x.d.sum(), en_iyi5_toplam=bes,
                         net_pay_en_iyi5=bes / x.d.sum(), brut_pay_en_iyi5=bes / x.d[x.d < 0].sum(),
                         n_d_kucuk_eksi010=int((x.d < -ESIK).sum()), n_d_bant=int((x.d.abs() <= ESIK).sum()),
                         n_d_buyuk_arti010=int((x.d > ESIK).sum()), spearman_d_b=rho, spearman_p=p))
    oz = pd.DataFrame(ozet)
    oz.to_csv(os.path.join(OUT, "ozet.csv"), index=False)
    pd.set_option("display.width", 250)
    print(oz.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nSiralama (82; d = ceza1 - rakim-IDW, eksi = ceza 1 iyi):")
    print(df.sort_values("d").to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nCikti: {OUT}")


if __name__ == "__main__":
    main()
