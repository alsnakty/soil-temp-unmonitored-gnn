import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 220)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KATLI = os.path.join(ANA_DIZIN, "results", "folds")
D = os.path.join(KATLI, "analysis_nearest_neighbour")
ANAHTAR = ["kat", "hedef", "derinlik"]


def ist_rmse(desen):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(desen))])
    return df.assign(e2=(df.y_pred - df.y_true) ** 2).groupby("istasyon_no").e2.mean() ** 0.5


def hata_farki_2023(model):
    ref = pd.concat([ist_rmse(os.path.join(ANA_DIZIN, "results", f"idw_elevation_fold{k}", "pred_spatial_val2023_idw.csv.gz"))
                     for k in range(5)])
    m = pd.concat([pd.concat([ist_rmse(os.path.join(KATLI, f"{model}_fold{k}_s{s}",
                                                    f"pred_spatial_val2023_gnn_s{s}.csv.gz")) for k in range(5)])
                   for s in range(3)], axis=1).mean(1)
    return m - ref


def dagilim(x):
    return pd.Series({"ort": x.mean(), "medyan": x.median(), "p10": x.quantile(.1), "p90": x.quantile(.9),
                      "min": x.min(), "max": x.max()})


def main():
    a = pd.read_csv(os.path.join(D, "agirlik.csv.gz"))
    t = pd.read_csv(os.path.join(D, "toplam.csv"))
    idw = a[a.model == "idw_rakim"]
    nn = a[a.sira == 1]
    w1 = nn[nn.model != "idw_rakim"].merge(nn[nn.model == "idw_rakim"][ANAHTAR + ["agirlik"]].rename(
        columns={"agirlik": "w1_idw"}), on=ANAHTAR).rename(columns={"agirlik": "w1"})
    top = t[t.model != "idw_rakim"].rename(columns={"agirlik": "toplam"})
    tek = a.groupby(["model", "kat", "tohum", "hedef", "derinlik"]).agirlik.sum().rename("tek_toplam").reset_index()
    lin = t.merge(tek, on=["model", "kat", "tohum", "hedef", "derinlik"]).assign(fark=lambda x: x.tek_toplam - x.agirlik)

    print("1) En yakin gorunur istasyonun agirligi (hedef x derinlik birimleri, C/C)")
    print("   rakim-IDW (ayni komsu):", dagilim(w1.drop_duplicates(ANAHTAR).w1_idw).round(3).to_dict())
    for m in ("gnn", "hibrit"):
        x = w1[w1.model == m]
        print(f"   {m}: " + str(dagilim(x.w1).round(3).to_dict()))
        print(f"   {m} kat x tohum ortalamasi (model / rakim-IDW):")
        g = x.groupby(["kat", "tohum"]).agg(w1=("w1", "mean"), w1_idw=("w1_idw", "mean")).unstack("tohum")
        print(g.round(3).to_string())
        st = x.groupby("hedef")[["w1", "w1_idw"]].mean()
        d = st.w1 - st.w1_idw
        print(f"   {m} - rakim-IDW, istasyon duzeyi (n={len(d)}): ort {d.mean():+.3f}, medyan {d.median():+.3f}, "
              f"model kucuk {int((d < 0).sum())}/{len(d)}, Wilcoxon p {wilcoxon(d).pvalue:.2g}")

    print("\n2) Toplam agirlik (butun gorunurler birlikte +1 C)")
    print("   rakim-IDW:", dagilim(t[t.model == "idw_rakim"].agirlik).round(4).to_dict())
    for m in ("gnn", "hibrit"):
        x = top[top.model == m]
        print(f"   {m}: " + str(dagilim(x.toplam).round(3).to_dict()))
        print(x.groupby(["kat", "tohum"]).toplam.mean().unstack("tohum").round(3).to_string())
        print("   derinlik:", x.groupby("derinlik").toplam.mean().round(3).to_dict())

    print("\n3) Dogrusallik: tek tek agirliklarin toplami - toplu kaydirma (C/C)")
    for m in ("idw_rakim", "gnn", "hibrit"):
        x = lin[lin.model == m]
        print(f"   {m}: |fark| ort {x.fark.abs().mean():.4f}, max {x.fark.abs().max():.4f}, ort fark {x.fark.mean():+.4f}"
              f"  (toplam ort {x.agirlik.mean():.3f})")
        if m != "idw_rakim":
            print(x.groupby(["kat", "tohum"]).fark.apply(lambda v: v.abs().mean()).unstack("tohum").round(4).to_string())

    print("\n4) Sira profili (ortalama agirlik; tohum ve hedef x derinlik ortalamasi)")
    print(a[a.sira <= 8].groupby(["sira", "model"]).agirlik.mean().unstack().round(3).to_string())

    print("\n5) En yakin uzakliga gore (tersiller) ve 2023 hata farkiyla iliski (istasyon duzeyi)")
    st = w1.groupby(["model", "hedef"]).agg(w1=("w1", "mean"), w1_idw=("w1_idw", "mean"), km=("km", "first")).reset_index()
    km = st.drop_duplicates("hedef").set_index("hedef").km
    dil = pd.qcut(km, 3, labels=["yakin", "orta", "uzak"])
    for m in ("gnn", "hibrit"):
        s = st[st.model == m].set_index("hedef")
        acik = s.w1 - s.w1_idw
        print(f"   {m}: tersil ort w1 / w1_idw:",
              {str(k): (round(v.w1.mean(), 3), round(v.w1_idw.mean(), 3)) for k, v in s.groupby(dil, observed=True)})
        r, p = spearmanr(acik, km.loc[acik.index])
        f = hata_farki_2023(m).reindex(acik.index)
        r2, p2 = spearmanr(acik, f)
        tt = top[top.model == m].groupby("hedef").toplam.mean().reindex(acik.index)
        r3, p3 = spearmanr(tt, f)
        print(f"   {m}: acik (w1 - w1_idw) ~ uzaklik rho {r:+.2f} (p {p:.2g}); acik ~ 2023 hata farki ({m} - rakim-IDW) "
              f"rho {r2:+.2f} (p {p2:.2g}); toplam agirlik ~ 2023 hata farki rho {r3:+.2f} (p {p3:.2g})")
    w1.to_csv(os.path.join(D, "en_yakin_agirlik.csv"), index=False)
    lin.to_csv(os.path.join(D, "dogrusallik.csv"), index=False)


if __name__ == "__main__":
    main()
