import os
import sys

import pandas as pd

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 220)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))
sys.path.insert(0, os.path.join(ANA_DIZIN, "analysis"))
from toprak_ayarlar import SPATIAL_HOLDOUT
import katli_birincil_karsilastirma as KB

KATLI = os.path.join(ANA_DIZIN, "results", "folds")
EK = {"hibrit": "gnn", "gnn": "gnn", "rfsi": "rfsi"}


def main():
    b = pd.read_csv(os.path.join(KATLI, "analysis_offset_correction_summary", "b2_istasyon.csv"))
    ref = b[b.duzeltme == "sabit + harmonik"].set_index("istasyon_no").rmse
    satir = []
    for m, ek in EK.items():
        tum = {s: [pd.read_csv(os.path.join(KATLI, f"{m}_fold{k}_s{s}", f"pred_temporal_{ek}_s{s}.csv.gz")) for k in range(5)]
               for s in range(3)}
        r = KB.model_rmse(tum)[0]
        for kad, ist in (("82", list(ref.index)), ("74", [s for s in ref.index if s not in SPATIAL_HOLDOUT])):
            o, karar = KB.karsilastir(r, ref, ist)
            satir.append(dict(model=m, kume=kad, model_ort=r.loc[ist].mean(), ref_ort=ref.loc[ist].mean(), **o,
                              karar_ayni_kural=karar))
    t = pd.DataFrame(satir)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    t.to_csv(os.path.join(KATLI, "analysis_offset_correction_summary", "c_temporal_kalibrasyon_test.csv"), index=False)


if __name__ == "__main__":
    main()
