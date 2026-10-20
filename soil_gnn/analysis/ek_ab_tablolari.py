import os
import sys

import pandas as pd

pd.set_option("compute.use_numexpr", False)

ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOK = os.path.dirname(ANA_DIZIN)
S = os.path.join(ANA_DIZIN, "results")
OUT = os.path.join(KOK, "ARTICLE_GNN", "tables")
KATLAR = range(5)


def dur(kosul, mesaj):
    if not kosul:
        sys.exit("DENETIM: " + mesaj)


def secilen(yol, sira):
    t = pd.read_csv(yol)
    r = t.sort_values(sira).iloc[0]
    if "tahminsiz" in t.columns:
        dur(int(r.tahminsiz) == 0, f"{yol}: secilen satirda tahminsiz > 0")
    return r


def pk(r):
    return f"{int(r.p)}, {r.k}"


def epoch_say(model, k):
    """1. asama gecmisinden 2. asama epoch sayisi (en kucuk dogrulama kaybinin epoch'u + 1) ve 2. asama denetimi."""
    gnn_mi = not model.startswith(("mlp", "ft_transformer"))
    ad = "gnn" if gnn_mi else ("mlp" if model.startswith("mlp") else "ft_transformer")
    h = pd.read_csv(os.path.join(S, "fold_selection", f"{model}_fold{k}", f"history_{ad}_s0.csv"))
    kol = "val_loss" if gnn_mi else "val"
    e = int(h[kol].idxmin()) + 1
    yollar = [os.path.join(S, "folds", f"{model}_fold{k}_s{s}", f"history_{ad}_s{s}.csv") for s in range(3)]
    yollar = [y for y in yollar if os.path.exists(y)]
    dur(len(yollar) > 0, f"{model} kat{k}: 2. asama gecmisi yok")
    for y in yollar:
        dur(len(pd.read_csv(y)) == e, f"{y}: satir sayisi {e} degil")
    return e


def epoch_ceza1(k):
    for s in range(3):
        y = os.path.join(S, "folds", f"hibrit_ceza1_fold{k}_s{s}", f"history_gnn_s{s}.csv")
        dur(len(pd.read_csv(y)) == 28, f"{y}: satir sayisi 28 degil")
    return 28


def yaz(ad, satirlar):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{ad}.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"% uretildi: soil_gnn/analysis/ek_ab_tablolari.py ({ad}); elle duzenlemeyin\n")
        # kapanis cizgisi govdede: tabular icinde \input'tan hemen sonra gelen \hline "Misplaced \noalign" veriyor
        f.write("\n".join(satirlar + ["\\hline"]) + "\n")
    print(f"{ad}: {len(satirlar)} satir")


def satir(etiket, hucreler):
    return f"{etiket} & " + " & ".join(str(h) for h in hucreler) + " \\\\"


def tab_b1():
    idw = [secilen(os.path.join(S, f"idw_elevation_fold{k}", "secim_2023.csv"), "rmse_2023") for k in KATLAR]
    ic = [secilen(os.path.join(S, "folds", "analysis_inner_validation", f"ic_secim_2023_kat{k}.csv"), "rmse_2023") for k in KATLAR]
    duz = [secilen(os.path.join(S, f"idw_fold{k}", "secim_2023.csv"), "rmse_2023") for k in KATLAR]
    kr = [secilen(os.path.join(S, f"kriging_elevation_fold{k}", "secim_2023.csv"), "rmse_2023") for k in KATLAR]
    dur({r.model for r in kr} == {"gauss"}, f"kriging varyogrami her katta Gauss degil: {[r.model for r in kr]}")
    rfsi = [secilen(os.path.join(S, "fold_selection", f"rfsi_fold{k}", "secim_n_2023.csv"), "rmse") for k in KATLAR]
    dur(all(r.bolum == "ic_val2023" for r in rfsi), "RFSI secimi ic_val2023 degil")
    s = ["\\multicolumn{6}{l}{\\textit{Interpolators}} \\\\",
         satir("Elevation-detrended IDW, $p$, $k$", [pk(r) for r in idw]),
         satir("\\quad inner training stations$^{a}$", [pk(r) for r in ic]),
         satir("Plain IDW, $p$, $k$", [pk(r) for r in duz]),
         satir("Regression kriging, $k$$^{b}$", [r.k for r in kr]),
         "\\multicolumn{6}{l}{\\textit{Learned models: epochs in the second stage; RFSI neighbours}} \\\\"]
    modeller = [("gnn", "Standalone GNN"), ("hibrit", "Hybrid GNN"), ("mlp", "Multilayer perceptron"),
                ("ft_transformer", "FT-Transformer"), ("gnn_konumyok", "Standalone GNN, no coordinates"),
                ("hibrit_konumyok", "Hybrid GNN, no coordinates"), ("gnn_uydusuz", "Standalone GNN, no satellite inputs"),
                ("hibrit_uydusuz", "Hybrid GNN, no satellite inputs"),
                ("mlp_uydusuz", "Multilayer perceptron, no satellite inputs"),
                ("ft_transformer_uydusuz", "FT-Transformer, no satellite inputs")]
    for m, etiket in modeller:
        s.append(satir(etiket, [epoch_say(m, k) for k in KATLAR]))
        if m == "ft_transformer":
            s.append(satir("Penalised hybrid$^{c}$", [epoch_ceza1(k) for k in KATLAR]))
    s.append(satir("RFSI, $n$", [int(r.n) for r in rfsi]))
    s.append("\\multicolumn{6}{l}{\\textit{IGNNK: $\\sigma$ in km (epochs in the second stage)}} \\\\")
    ig = {k: pd.read_csv(os.path.join(S, "fold_selection", f"ignnk_fold{k}", "secim_ignnk.csv")).set_index("derinlik")
          for k in KATLAR}
    for d in (5, 10, 20, 50, 100):
        s.append(satir(f"{d}~cm", [f"{int(ig[k].loc[d, 'sigma'])} ({int(ig[k].loc[d, 'epoch_sayisi'])})" for k in KATLAR]))
    yaz("tabB1", s)


def tab_b2():
    t = pd.read_csv(os.path.join(S, "folds", "analysis_deep_final_test", "secimler.csv")).set_index(["kat", "derinlik"])
    dur(len(t) == 10, "secimler.csv 10 satir degil")
    s = []
    for d in (50, 100):
        r = [t.loc[(k, d)] for k in KATLAR]
        s += [f"\\multicolumn{{6}}{{l}}{{\\textit{{{d}~cm}}}} \\\\",
              satir("Running mean, $L$ (days)", [int(x.L) for x in r]),
              satir("Running mean, $\\gamma_z$", [f"{x.gamma_A3:.2f}" for x in r]),
              satir("Half-space kernel, $\\kappa$ (cm$^2$\\,d$^{-1}$)", [int(x.kappa) for x in r]),
              satir("Half-space kernel, $\\gamma_z$", [f"{x.gamma_A5:.2f}" for x in r]),
              satir("Ridge regression, $\\alpha$", [f"{x.alfa_A6:.2g}" if x.alfa_A6 >= 0.0995 else f"{x.alfa_A6:.3f}" for x in r])]
    yaz("tabB2", s)


if __name__ == "__main__":
    tab_b1()
    tab_b2()
