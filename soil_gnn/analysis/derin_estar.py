import os

import numpy as np
import pandas as pd

ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOK = os.path.join(ANA_DIZIN, "results", "deep_selection")


def main():
    h = []
    for k in range(5):
        x = pd.read_csv(os.path.join(KOK, f"hibrit_a5_fold{k}", "history_gnn_s0.csv"))
        if "egitilen_epoch" not in x.columns or x.egitilen_epoch.iloc[0] != 0:
            raise RuntimeError(f"kat {k}: epoch 0 satiri yok")
        if x.egitilen_epoch.max() != 40:
            raise RuntimeError(f"kat {k}: 40 epoch tamamlanmamis ({x.egitilen_epoch.max()})")
        h.append(x.set_index("egitilen_epoch").val_loss.rename(f"kat{k}"))
    t = pd.concat(h, axis=1)
    t["ortalama"] = t.mean(axis=1)
    e = int(t.ortalama.idxmin())
    print(t.round(5).to_string())
    print(f"\nE* = {e} (ortalama kayip {t.ortalama[e]:.5f}; epoch 0 = {t.ortalama[0]:.5f})")
    print("katlarin kendi en iyi epoch'u:", {c: int(t[c].idxmin()) for c in t.columns if c.startswith("kat")})
    oz = pd.DataFrame({"kayip_epoch0": t.loc[0], f"kayip_E*={e}": t.loc[e]})
    oz["goreli_fark"] = (oz[f"kayip_E*={e}"] - oz.kayip_epoch0) / oz.kayip_epoch0      # eksi = egitim tabani iyilestirdi
    print("\nKat basina epoch 0 ve E* kaybi, goreli fark (E* - epoch 0) / epoch 0:")
    print(oz.round(5).to_string())
    t.to_csv(os.path.join(KOK, "estar.csv"))
    oz.to_csv(os.path.join(KOK, "estar_kat.csv"))


if __name__ == "__main__":
    main()
