import os

import pandas as pd

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 220)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
KOK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ANA_DIZIN, "results", "folds", "analysis_rfsi_tuning")
AYAR = ["n", "max_features", "yaprak"]


def main():
    t = pd.concat([pd.read_csv(os.path.join(KOK, f"rfsi_ayar_kat{k}.csv")) for k in range(5)], ignore_index=True)
    t["max_features"] = t.max_features.fillna("").astype(str)
    izg = t[(t.ayar == "izgara") & t.idw_ozellik]
    say = izg.groupby(AYAR).kat.nunique()
    if len(say) != 18 or (say != 5).any():
        raise RuntimeError("izgara eksik (18 ayar x 5 kat bekleniyor)")
    ort = izg.groupby(AYAR).istasyon_rmse_ort.mean().sort_values()
    en = ort.index[0]
    sec = lambda m: t[m].set_index("kat").istasyon_rmse_ort.reindex(range(5))
    m_en = (t.ayar == "izgara") & t.idw_ozellik
    m_oz = (t.ayar == "izgara") & ~t.idw_ozellik
    for c, v in zip(AYAR, en):
        m_en &= t[c] == v
        m_oz &= t[c] == v
    tab = pd.DataFrame({"taban_rakim_idw": sec(t.ayar == "taban_rakim_idw"), "rfsi_en_iyi": sec(m_en),
                        "rfsi_kayitli_ayar (i)": sec(t.ayar == "kayitli_ayar"), "rfsi_en_iyi_idw_ozelliksiz (ii)": sec(m_oz)})
    tab.loc["ortalama"] = tab.mean()
    gecti = tab.loc["ortalama", "rfsi_en_iyi"] < tab.loc["ortalama", "taban_rakim_idw"]
    tab.to_csv(os.path.join(KOK, "ozet.csv"))
    ort.rename("5_kat_ortalama").to_csv(os.path.join(KOK, "izgara_5kat_ortalama.csv"))
    print(f"En iyi ayar (oznitelikli 18 ayar, 5 kat ortalamasi): n={en[0]}, max_features={en[1]}, yaprak={en[2]}")
    print("\nIc dogrulama istasyon RMSE ortalamasi (5 derinlik), kat basina:")
    print(tab.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nIzgara (oznitelikli), 5 kat ortalamasi:")
    print(ort.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nKAPI:", "GECTI (en iyi RFSI < rakim-IDW; 2. asama ayrica onaya)" if gecti
          else "GECMEDI -> dur: 'ayarlansa da fark kapanmadi'")


if __name__ == "__main__":
    main()
