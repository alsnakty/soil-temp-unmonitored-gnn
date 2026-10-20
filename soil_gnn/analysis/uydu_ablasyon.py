import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 220)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))
sys.path.insert(0, os.path.join(ANA_DIZIN, "analysis"))
import katli_birincil_karsilastirma as KB
from toprak_ayarlar import SPATIAL_HOLDOUT

KATLI = os.path.join(ANA_DIZIN, "results", "folds")
OUT = os.path.join(KATLI, "analysis_satellite_ablation")
MODELLER = ("gnn", "hibrit", "mlp", "ft_transformer", "rf")


def yukle(model, bolum):
    kl = []
    for k in range(5):
        f = glob.glob(os.path.join(KATLI, f"{model}_fold{k}_s0", f"pred_{bolum}_*.csv.gz"))
        if len(f) != 1:
            return None
        kl.append(KB._oku(f[0]).assign(kat=k))
    return kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bolum", default="spatial_test")
    ap.add_argument("--uydusuz-ek", default="_uydusuz")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    satir, derin = [], []
    for m in MODELLER:
        u, z = yukle(m, a.bolum), yukle(m + a.uydusuz_ek, a.bolum)
        if u is None or z is None:
            print(f"{m}: dosya eksik, atlandi")
            continue
        for k in range(5):
            if not u[k][KB.ANAHTAR + ["y_true"]].equals(z[k][KB.ANAHTAR + ["y_true"]]):
                raise RuntimeError(f"{m} kat {k}: uydulu ve uydusuz satirlari farkli")
        ru, rz = KB.kat_ortalama_rmse(u), KB.kat_ortalama_rmse(z)
        for kad, ist in (("82", list(ru.index)), ("74", [i for i in ru.index if i not in SPATIAL_HOLDOUT])):
            o, _ = KB.karsilastir(ru, rz, ist)
            satir.append(dict(model=m, kume=kad, uydulu_ist_rmse=float(ru.loc[ist].mean()),
                              uydusuz_ist_rmse=float(rz.loc[ist].mean()),
                              uydulu_havuz=KB.havuz({0: [d[d.istasyon_no.isin(ist)] for d in u]}),
                              uydusuz_havuz=KB.havuz({0: [d[d.istasyon_no.isin(ist)] for d in z]}), **o))
        for dz in sorted(u[0].derinlik.unique()):
            derin.append(dict(model=m, derinlik=dz, uydulu_havuz=KB.havuz({0: u}, dz), uydusuz_havuz=KB.havuz({0: z}, dz)))
    t = pd.DataFrame(satir)
    t.to_csv(os.path.join(OUT, f"uydulu_eksi_uydusuz_{a.bolum}.csv"), index=False)
    pd.DataFrame(derin).to_csv(os.path.join(OUT, f"derinlik_{a.bolum}.csv"), index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(pd.DataFrame(derin).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
