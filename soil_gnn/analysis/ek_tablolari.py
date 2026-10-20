import os
import sys

import numpy as np
import pandas as pd

pd.set_option("compute.use_numexpr", False)

ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOK = os.path.dirname(ANA_DIZIN)
S = os.path.join(ANA_DIZIN, "results")
K = os.path.join(S, "folds")
OUT = os.path.join(KOK, "ARTICLE_GNN", "tables")
ESDEGER = 0.10

AD = {"rf": "Random forest", "mlp": "Multilayer perceptron", "ft_transformer": "FT-Transformer", "idw": "Plain IDW",
      "idw_rakim": "Elevation-detrended IDW", "kriging_rakim": "Regression kriging", "rfsi": "RFSI", "gnn": "Standalone GNN",
      "hibrit": "Hybrid GNN", "ignnk": "IGNNK", "hibrit_ceza1": "Penalised hybrid",
      "gnn_konumyok": "Standalone GNN, no coordinates", "hibrit_konumyok": "Hybrid GNN, no coordinates"}
GRUP = [("No neighbour measurements", ["rf", "mlp", "ft_transformer"]),
        ("Neighbour measurements, interpolation", ["idw", "idw_rakim", "kriging_rakim"]),
        ("Neighbour measurements, learned", ["rfsi", "gnn", "hibrit", "ignnk", "hibrit_ceza1"])]
ABLASYON = ("Location ablation, one seed", ["gnn_konumyok", "hibrit_konumyok"])
KARAR_CSV = {"model iyi": "better", "referans iyi": "worse", "ayirt edilemedi": "inconclusive"}
KISA = {"better": "b", "worse": "w", "equivalent": "e", "inconclusive": ""}
TOHUM = "tohum basina RMSE ortalamasi"


def dur(kosul, mesaj):
    if not kosul:
        sys.exit(f"DENETIM TUTMADI: {mesaj}")


def f2(x):
    if x is None or pd.isna(x):
        return "--"
    s = f"{x:.2f}"
    if s == "-0.00":
        s = "0.00"
    return "$-$" + s[1:] if s.startswith("-") else s


def fp(p):
    return "--" if pd.isna(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


def karar(r):
    if r.ort_fark < 0 and r.ga_ust < 0:
        k = "better"
    elif r.ort_fark > 0 and r.ga_alt > 0:
        k = "worse"
    else:
        k = "equivalent" if (-ESDEGER < r.ga90_alt and r.ga90_ust < ESDEGER) else "inconclusive"
    for sutun in ("karar", "karar_betimleyici"):
        v = r.get(sutun) if hasattr(r, "get") else None
        if isinstance(v, str) and v:
            beklenen = KARAR_CSV[v.replace(" (iki kumede ayni)", "")]
            dur(beklenen == ("inconclusive" if k == "equivalent" else k), f"karar {r.model}: {v} / {k}")
    e = r.get("etiket") if hasattr(r, "get") else None
    if isinstance(e, str) and e:
        if e.startswith("esdeger"):
            dur(k == "equivalent", f"etiket {r.model}: {e} / {k}")
        elif e.startswith("belirsiz"):
            dur(k == "inconclusive", f"etiket {r.model}: {e} / {k}")
    return k


def yaz(ad, satirlar):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{ad}.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"% uretildi: soil_gnn/analysis/ek_tablolari.py ({ad}); elle duzenlemeyin\n")
        # kapanis cizgisi govdede: tabular icinde \input'tan hemen sonra gelen \hline "Misplaced \noalign" veriyor
        f.write("\n".join(satirlar + ["\\hline"]) + "\n")
    print(f"{ad}: {len(satirlar)} satir")


def c2():
    k = pd.read_csv(os.path.join(S, "variant_selection", "kapi_ozet.csv")).set_index("ayar")
    sira = [("a_gnn", "Standalone GNN, learned distance weighting"), ("a_hibrit", "Hybrid, learned distance weighting"),
            ("ceza01", "Hybrid, penalty $\\lambda = 0.1$"), ("ceza1", "Hybrid, penalty $\\lambda = 1$"),
            ("ceza10", "Hybrid, penalty $\\lambda = 10$"), ("blok", "Hybrid, block hiding")]
    idw = [i for i in k.index if i.startswith("rakim_idw")]
    dur(len(idw) == 1, "C2 rakim_idw satiri")
    out = []
    for a, ad in sira + [(idw[0], "Elevation-detrended IDW")]:
        r = k.loc[a]
        e = "--" if pd.isna(r["E*"]) else str(int(r["E*"]))
        out.append(" & ".join([ad] + [f2(r[str(i)]) for i in range(5)] + [f2(r["ortalama"]), e]) + " \\\\")
    yaz("tabC2", out)


def c3():
    o = pd.read_csv(os.path.join(K, "analysis_rfsi_tuning", "ozet.csv")).set_index("kat")
    sira = [("taban_rakim_idw", "Elevation-detrended IDW"), ("rfsi_en_iyi", "RFSI, best of 18 settings"),
            ("rfsi_kayitli_ayar (i)", "RFSI, setting of Table~\\ref{M-tab:main}"),
            ("rfsi_en_iyi_idw_ozelliksiz (ii)", "RFSI, best setting without the IDW feature")]
    out = []
    for s, ad in sira:
        out.append(" & ".join([ad] + [f2(o.loc[str(i), s]) for i in range(5)] + [f2(o.loc["ortalama", s])]) + " \\\\")
    yaz("tabC3", out)


def c4():
    t = pd.read_csv(os.path.join(K, "analysis_offset_correction", "karsilastirma.csv"))
    sira = [("idw_rakim_dlst", "idw_rakim", "IDW with night-LST offset vs IDW"),
            ("kriging_rakim_dlst", "kriging_rakim", "Kriging with night-LST offset vs kriging"),
            ("hibrit_0epoch", "idw_rakim", "Hybrid allowed to stay untrained vs IDW")]
    out = []
    for m, ref, ad in sira:
        r = t[(t.model == m) & (t.referans == ref) & (t.kume == "82 istasyon")]
        dur(len(r) == 1, f"C4 {m}")
        r = r.iloc[0]
        out.append(" & ".join([ad, f2(r.ort_fark), f"{f2(r.ga_alt)}, {f2(r.ga_ust)}",
                               fp(r.wilcoxon_p), f"{int(r.model_iyi_istasyon)}", karar(r)]) + " \\\\")
    yaz("tabC4", out)


def d1():
    k = pd.read_csv(os.path.join(K, "analysis_spatial_test", "karsilastirma.csv"))
    it = pd.read_csv(os.path.join(K, "analysis_spatial_test", "istasyon_tablosu.csv"))
    it = it[~it.gelistirme_8.astype(bool)]
    dur(len(it) == 74, "D1 74 istasyon")
    kume = "74 istasyon (gelistirme 8 haric)"
    out = []
    for baslik, modeller in GRUP:
        out.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{baslik}}}}} \\\\")
        for m in modeller:
            rm = it[f"rmse_{m}"].mean()
            if m == "idw_rakim":
                out.append(" & ".join([AD[m], f2(rm), "--", "--", "--", "--", "reference"]) + " \\\\")
                continue
            r = k[(k.model == m) & (k.referans == "idw_rakim") & (k.kume == kume) & k.olcu.isin([TOHUM, "tek"])]
            dur(len(r) == 1, f"D1 {m}")
            r = r.iloc[0]
            dur(abs((it[f"rmse_{m}"] - it["rmse_idw_rakim"]).mean() - r.ort_fark) < 1e-9, f"D1 fark {m}")
            out.append(" & ".join([AD[m], f2(rm), f2(r.ort_fark), f"{f2(r.ga_alt)}, {f2(r.ga_ust)}", fp(r.wilcoxon_p),
                                   f"{int(r.model_iyi_istasyon)}", karar(r)]) + " \\\\")
    yaz("tabD1", out)


def d2():
    d = pd.read_csv(os.path.join(K, "analysis_spatial_test", "derinlik.csv")).set_index("derinlik")
    d = d[~d.index.duplicated()]
    out = []
    for baslik, modeller in GRUP:
        out.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{baslik}}}}} \\\\")
        for m in modeller:
            out.append(" & ".join([AD[m]] + [f2(d.loc[z, m]) for z in ("5", "10", "20", "50", "100", "all")]) + " \\\\")
    yaz("tabD2", out)


def d3():
    d = pd.read_csv(os.path.join(K, "analysis_spatial_test", "derinlik_fark.csv"))
    d = d[(d.kume == "82 istasyon") & (d.referans == "idw_rakim")]
    out, n = [], []
    for z in (5, 10, 20, 50, 100):
        n.append(str(int(d[d.derinlik == z].n_istasyon.iloc[0])))
    out.append(" & ".join(["Stations"] + n) + " \\\\")
    out.append("\\hline")
    for m in ("kriging_rakim", "gnn", "hibrit", "ignnk", "hibrit_ceza1"):
        hucre = []
        for z in (5, 10, 20, 50, 100):
            r = d[(d.model == m) & (d.derinlik == z)]
            dur(len(r) == 1, f"D3 {m} {z}")
            r = r.iloc[0]
            k = KISA[karar(r)]
            hucre.append(f2(r.ort_fark) + (f"$^{{\\mathrm{{{k}}}}}$" if k else ""))
        out.append(" & ".join([AD[m]] + hucre) + " \\\\")
    yaz("tabD3", out)


def d4():
    k = pd.read_csv(os.path.join(K, "analysis_spatial_test", "karsilastirma.csv"))
    f = pd.read_csv(os.path.join(K, "analysis_spatial_full", "karsilastirma.csv"))
    t = pd.read_csv(os.path.join(K, "analysis_spatial_test", "tohum_oynakligi.csv"))
    ms = pd.read_csv(os.path.join(K, "analysis_spatial_test", "mesafe_spearman.csv")).set_index("model")

    def hucre(tab, m, olcu_on):
        r = tab[(tab.model == m) & (tab.referans == "idw_rakim") & (tab.kume == "82 istasyon")
                & tab.olcu.str.startswith(olcu_on)]
        if m == "idw_rakim" or not len(r):
            return "--"
        dur(len(r) == 1, f"D4 {m} {olcu_on}")
        r = r.iloc[0]
        kk = KISA[karar(r)]
        return f2(r.ort_fark) + (f"$^{{\\mathrm{{{kk}}}}}$" if kk else "")

    out = []
    for baslik, modeller in GRUP + [ABLASYON]:
        out.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{baslik}}}}} \\\\")
        for m in modeller:
            ens = hucre(k, m, "topluluk")
            tum = hucre(f, m, TOHUM) if m not in ("idw", "kriging_rakim") else hucre(f, m, "tek")
            sd = t[t.model == m].tohum_sd
            sd = f2(sd.median()) if len(sd) else "--"
            rr = f2(ms.loc[m, "rho_rmse_mesafe"]) if m in ms.index else "--"
            rf = f2(ms.loc[m, "rho_fark_mesafe"]) if m in ms.index else "--"
            out.append(" & ".join([AD[m], ens, tum, sd, rr, rf]) + " \\\\")
    yaz("tabD4", out)


def d5():
    g = pd.read_csv(os.path.join(K, "gorulmus_gorulmemis.csv"))
    out = []
    for m in ("gnn", "hibrit", "idw_rakim"):
        h = []
        for b in ("temporal", "spatial_test"):
            r = g[(g.model == m) & (g.bolum == b)]
            dur(len(r) == 1, f"D5 {m} {b}")
            h += [f2(r.iloc[0].istasyon_rmse_ort), f2(r.iloc[0].havuz_rmse)]
        out.append(" & ".join([AD[m]] + h) + " \\\\")
    yaz("tabD5", out)


def d6():
    g = pd.read_csv(os.path.join(K, "analysis_spatial_full", "genlik_faz.csv"))
    d = pd.read_csv(os.path.join(K, "analysis_spatial_full", "gorunur_difuzivite.csv"))
    g, d = g[g.kume == "82 istasyon"], d[d.kume == "82 istasyon"].set_index("model")
    out = []
    for baslik, modeller in GRUP + [ABLASYON]:
        out.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{baslik}}}}} \\\\")
        for m in modeller:
            x = g[g.model == m]
            dur(len(x) == 5, f"D6 {m}")
            a = f"{f2(x.genlik_orani_medyan.min())}--{f2(x.genlik_orani_medyan.max())}"
            p = f"{f2(x.faz_farki_gun_medyan.min())} to {f2(x.faz_farki_gun_medyan.max())}"
            out.append(" & ".join([AD[m], a, p, f2(d.loc[m, "kA_oran_medyan"]), f2(d.loc[m, "kphi_oran_medyan"])]) + " \\\\")
    yaz("tabD6", out)


def main():
    for fn in (c2, c3, c4, d1, d2, d3, d4, d5, d6):
        fn()
    print("yazildi:", OUT)


if __name__ == "__main__":
    main()
