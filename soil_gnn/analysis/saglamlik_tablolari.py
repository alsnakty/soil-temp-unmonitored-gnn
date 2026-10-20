import os
import sys

import pandas as pd

pd.set_option("compute.use_numexpr", False)

ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOK = os.path.dirname(ANA_DIZIN)
DR = os.path.join(ANA_DIZIN, "results", "folds", "analysis_robustness")
AST = os.path.join(ANA_DIZIN, "results", "folds", "analysis_spatial_test")
OUT = os.path.join(KOK, "ARTICLE_GNN", "tables")
ESDEGER = 0.10
CIFTLER = [("hibrit", "idw_rakim", "Hybrid GNN vs IDW"), ("hibrit_ceza1", "idw_rakim", "Penalised hybrid vs IDW"),
           ("kriging_rakim", "idw_rakim", "Regression kriging vs IDW"),
           ("hibrit_ceza1", "kriging_rakim", "Penalised hybrid vs kriging"), ("ignnk", "idw_rakim", "IGNNK vs IDW")]
DR2 = os.path.join(ANA_DIZIN, "results", "folds", "analysis_robustness_extra")
EK_CIFT = {"kriging_rakim - idw_rakim": "Regression kriging vs IDW", "hibrit_ceza1 - kriging_rakim": "Penalised hybrid vs kriging",
           "ignnk - idw_rakim": "IGNNK vs IDW"}


def sinirda(r):
    return r.ort_fark < 0 and -0.0015 < r.ga_ust < 0


def dur(kosul, mesaj):
    if not kosul:
        sys.exit(f"DENETIM TUTMADI: {mesaj}")


def f2(x):
    if x is None or pd.isna(x):
        return "--"
    s = f"{x:.2f}"
    if s in ("0.00", "-0.00") and x != 0:          # sifira yakin aralik siniri: isaret kaybolmasin
        s = f"{x:.3f}"
    return "$-$" + s[1:] if s.startswith("-") else s


def fp(p):
    return "--" if pd.isna(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


def etiket(r):
    if r.ort_fark < 0 and r.ga_ust < 0:
        k = "better"
    elif r.ort_fark > 0 and r.ga_alt > 0:
        k = "worse"
    else:
        k = None
    icinde = -ESDEGER < r.ga90_alt and r.ga90_ust < ESDEGER
    if k is None:
        e = "equivalent" if icinde else "inconclusive"
    else:
        e = f"{k} within the margin" if icinde else k
    kc = r.get("karar", r.get("karar_betimleyici"))
    beklenen = {"model iyi": "better", "referans iyi": "worse", "ayirt edilemedi": None}[kc]
    dur(beklenen == k, f"karar {kc} / {e}")
    return e


def satir(ad, n, r, p=True):
    e = etiket(r)
    if sinirda(r):
        dur(e == "better", f"sinirda satir better degil: {ad}")
        e = "at the edge$^{a}$"
    return (f"{ad} & {n} & {f2(r.ort_fark)} & {f2(r.ga_alt)}, {f2(r.ga_ust)} & "
            f"{fp(r.wilcoxon_p) if p else '--'} & {e} \\\\")


def yaz(ad, satirlar):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{ad}.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"% uretildi: soil_gnn/analysis/saglamlik_tablolari.py ({ad}); elle duzenlemeyin\n")
        f.write("\n".join(satirlar + ["\\hline"]) + "\n")
    print(f"{ad}: {len(satirlar)} satir")


def c5():
    a1 = pd.read_csv(os.path.join(DR, "a1_tablo2.csv"))
    a2 = pd.read_csv(os.path.join(DR, "a2_tablo2.csv"))
    a3 = pd.read_csv(os.path.join(DR, "a3_bootstrap.csv"))
    a9 = pd.read_csv(os.path.join(DR, "a9_cezali_hibrit.csv"))
    for t in (a1, a2, a3, a9):
        t["kume"] = t.kume.astype(str)
    kars = pd.read_csv(os.path.join(AST, "karsilastirma.csv"))
    kars = kars[(kars.kume == "82 istasyon") & kars.olcu.isin(["tohum basina RMSE ortalamasi", "tek"])]
    out = []
    kosul = [("All stations (Table~\\ref{M-tab:main})", a1, "kisitsiz"),
             ("Station--depth pairs with late breaks removed", a1, "a 49 cift cikarildi"),
             ("Stations with late breaks removed", a1, "b 14 istasyon cikarildi"),
             ("Isparta, Kilis, Afyonkarahisar removed", a2, None),
             ("Also Alanya, Antalya, Ardahan removed", a2, None)]
    a2_durum = sorted(a2.durum.unique())
    for i, (baslik, tab, durum) in enumerate(kosul):
        d = durum if durum is not None else a2_durum[i - 3]
        x = tab[(tab.durum == d) & (tab.kume == "82")]
        out.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{baslik}}}}} \\\\")
        for m, ref, ad in CIFTLER:
            r = x[(x.model == m) & (x.referans == ref)]
            dur(len(r) == 1, f"{d} {m} {ref}")
            r = r.iloc[0]
            if durum == "kisitsiz":
                k = kars[(kars.model == m) & (kars.referans == ref)].iloc[0]
                dur(abs(k.ort_fark - r.ort_fark) < 1e-9 and abs(k.ga_ust - r.ga_ust) < 1e-9, f"kisitsiz {m} {ref}")
            out.append(satir(ad, int(r.n_istasyon), r))
    k4 = pd.read_csv(os.path.join(DR2, "k4_s4_ekleri.csv"))
    k4["kume"] = k4.kume.astype(str)
    for yontem, baslik in (("i tohumlu", "Seeds resampled"), ("ii bloklu", "16 spatial clusters resampled"),
                           ("iii bloklu+tohumlu", "Clusters and seeds resampled")):
        out.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{baslik}}}}} \\\\")
        for m, ad in (("hibrit", "Hybrid GNN vs IDW"), ("hibrit_ceza1", "Penalised hybrid vs IDW")):
            r = a3[(a3.model == m) & (a3.kume == "82") & (a3.yontem == yontem)].iloc[0].copy()
            r["karar"] = r.karar_betimleyici
            out.append(satir(ad, int(r.n_istasyon), r, p=False))
        for cift, ad in EK_CIFT.items():
            r = k4[(k4.cift == cift) & (k4.kume == "82") & (k4.yontem == yontem)]
            dur(len(r) == 1, f"k4 {cift} {yontem}")
            r = r.iloc[0].copy()
            r["karar"] = r.karar_betimleyici
            out.append(satir(ad, int(r.n_istasyon), r, p=False))
    for d, baslik in zip(sorted(a9.durum.unique()), ("Anamur removed", "Five stations with the largest gains removed")):
        r = a9[(a9.durum == d) & (a9.kume == "82")].iloc[0]
        out.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{baslik}}}}} \\\\")
        out.append(satir("Penalised hybrid vs IDW", int(r.n_istasyon), r))
    yaz("tabC5", out)


def c6():
    k = pd.read_csv(os.path.join(DR, "a6_kalicilik.csv"))
    out = []
    for hedef, kume, ad in (("QC", "82", "Adjusted, 82 stations"), ("ham", "82", "Unadjusted, 82 stations"),
                            ("QC", "44 duzeltme gormeyen", "Adjusted, 44 stations without corrections"),
                            ("ham", "44 duzeltme gormeyen", "Unadjusted, 44 stations without corrections")):
        x = k[(k.hedef == hedef) & (k.kume.astype(str) == kume)].set_index("derinlik")
        dur(len(x) == 5, f"{hedef} {kume}")
        out.append(ad + " & " + " & ".join(f"{x.spearman[d]:.2f}" for d in (5, 10, 20, 50, 100)) + " \\\\")
    yaz("tabC6", out)


def main():
    c5()
    c6()


if __name__ == "__main__":
    main()
