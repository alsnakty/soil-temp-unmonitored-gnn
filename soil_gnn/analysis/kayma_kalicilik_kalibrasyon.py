import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

pd.set_option("compute.use_numexpr", False)
pd.set_option("display.width", 220)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SON = os.path.join(ANA_DIZIN, "results")
OUT = os.path.join(SON, "folds", "analysis_offset_calibration")
DER = (5, 10, 20, 50, 100)
TEST0 = pd.Timestamp("2024-01-01")
KAYNAK = {"idw_rakim": [("idw_elevation_fold{k}", "pred_spatial_full_idw.csv.gz")],
          "kriging_rakim": [("kriging_elevation_fold{k}", "pred_spatial_full_kriging.csv.gz")],
          "hibrit": [(f"folds/hibrit_fold{{k}}_s{s}", f"pred_spatial_full_gnn_s{s}.csv.gz") for s in range(3)]}


def oku(klasor, dosya):
    df = pd.concat([pd.read_csv(os.path.join(SON, klasor.format(k=k), dosya), parse_dates=["tarih"]) for k in range(5)],
                   ignore_index=True)
    if df.duplicated(["istasyon_no", "derinlik", "tarih"]).any():
        raise RuntimeError(f"{klasor}: tekrarli satir")
    if df.istasyon_no.nunique() != 82:
        raise RuntimeError(f"{klasor}: istasyon sayisi {df.istasyon_no.nunique()}")
    return df.assign(e=df.y_pred - df.y_true)


def kalicilik(df):
    a = df[(df.tarih.dt.year >= 2015) & (df.tarih.dt.year <= 2019)].groupby(["istasyon_no", "derinlik"]).e.mean()
    b = df[(df.tarih.dt.year >= 2020) & (df.tarih.dt.year <= 2023)].groupby(["istasyon_no", "derinlik"]).e.mean()
    j = pd.concat([a.rename("k1519"), b.rename("k2023")], axis=1).dropna()
    satir = []
    for d in DER:
        x = j.xs(d, level="derinlik")
        r, p = spearmanr(x.k1519, x.k2023)
        satir.append(dict(derinlik=d, n_istasyon=len(x), spearman=r, p=p, pearson=np.corrcoef(x.k1519, x.k2023)[0, 1],
                          ayni_isaret=float((np.sign(x.k1519) == np.sign(x.k2023)).mean()),
                          ort_mutlak_degisim=float((x.k2023 - x.k1519).abs().mean()),
                          rms_kayma_1519=float(np.sqrt((x.k1519 ** 2).mean())), rms_kayma_2023=float(np.sqrt((x.k2023 ** 2).mean()))))
    return pd.DataFrame(satir), j


def derinlik_tutarliligi(df):
    k = df[df.tarih < TEST0].groupby(["istasyon_no", "derinlik"]).e.mean().unstack("derinlik")
    isaret = []
    for st, r in k.iterrows():
        v = r.dropna()
        if len(v) < 2:
            continue
        s = np.sign(v.to_numpy())
        ciftler = [(s[i] == s[j]) for i in range(len(s)) for j in range(i + 1, len(s))]
        isaret.append(dict(istasyon_no=st, n_derinlik=len(v), hepsi_ayni=bool((s == s[0]).all()), cift_ayni=np.mean(ciftler)))
    isaret = pd.DataFrame(isaret)
    kor = pd.DataFrame(index=DER, columns=DER, dtype=float)
    for i in DER:
        for j in DER:
            if i == j:
                kor.loc[i, j] = 1.0
                continue
            x = k[[i, j]].dropna()
            kor.loc[i, j] = spearmanr(x[i], x[j])[0]
    return isaret, kor, k


def pencere_kayma(df, bas, son):
    """[bas, son) arasindaki olcumlerin istasyon x derinlik ortalama e'si."""
    if son > TEST0:
        raise RuntimeError("kalibrasyon penceresi 2024'e tasiyor")
    w = df[(df.tarih >= bas) & (df.tarih < son)]
    if len(w) and w.tarih.max() >= TEST0:
        raise RuntimeError("kalibrasyonda 2024-25 satiri")
    return w.groupby(["istasyon_no", "derinlik"]).e.mean().rename("kal")


def test_rmse(df, kal):
    t = df[df.tarih >= TEST0].join(kal, on=["istasyon_no", "derinlik"]) if kal is not None else df[df.tarih >= TEST0].assign(kal=np.nan)
    duz = t.kal.notna()
    e = t.e - t.kal.fillna(0.0)
    ist = (e ** 2).groupby(t.istasyon_no).mean() ** 0.5
    sd = t.groupby(["istasyon_no", "derinlik"]).size()
    sd_duz = t[duz].groupby(["istasyon_no", "derinlik"]).size()
    return dict(havuz_rmse=float(np.sqrt((e ** 2).mean())), istasyon_rmse_ort=float(ist.mean()),
                duzeltilen_ist_derinlik=len(sd_duz), toplam_ist_derinlik=len(sd), duzeltilen_satir_orani=float(duz.mean()))


def kalibrasyon(dfs):
    satir, mevsim = [], []
    for N in [0, 1, 3, 6, 12, 24, "2015-2023"]:
        r = []
        for df in dfs:
            if N == 0:
                kal = None
            elif N == "2015-2023":
                kal = pencere_kayma(df, pd.Timestamp("2015-01-01"), TEST0)
            else:
                kal = pencere_kayma(df, TEST0 - pd.DateOffset(months=N), TEST0)
            r.append(test_rmse(df, kal))
        r = pd.DataFrame(r)
        satir.append(dict(N_ay=N, **r.mean().to_dict()))
    for N in (1, 3):
        for ay in range(1, 13):
            son = pd.Timestamp(2023, ay, 1) + pd.DateOffset(months=1)
            r = pd.DataFrame([test_rmse(df, pencere_kayma(df, son - pd.DateOffset(months=N), son)) for df in dfs])
            mevsim.append(dict(N_ay=N, pencere_son_ay=f"2023-{ay:02d}", **r.mean().to_dict()))
    return pd.DataFrame(satir), pd.DataFrame(mevsim)


def main():
    os.makedirs(OUT, exist_ok=True)
    tum = {m: [oku(*kd) for kd in v] for m, v in KAYNAK.items()}
    ref = tum["idw_rakim"][0].set_index(["istasyon_no", "derinlik", "tarih"]).y_true.sort_index()
    for m, dfs in tum.items():
        for df in dfs:
            x = df.set_index(["istasyon_no", "derinlik", "tarih"]).y_true.sort_index()
            if not (x.index.equals(ref.index) and np.allclose(x.to_numpy(), ref.to_numpy())):
                raise RuntimeError(f"{m}: satir kumesi / y_true rakim-IDW ile ayni degil")
    print(f"Satir kumesi ve y_true uc yontemde ayni ({len(ref)} satir).")

    print("\na) Kalicilik (rakim-IDW; istasyon x derinlik ortalama e, 2015-2019 vs 2020-2023):")
    k, j = kalicilik(tum["idw_rakim"][0])
    print(k.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    k.to_csv(os.path.join(OUT, "a_kalicilik.csv"), index=False)
    j.to_csv(os.path.join(OUT, "a_kayma_donem.csv"))

    print("\nb) Derinlikler arasi tutarlilik (rakim-IDW, 2015-2023 kayma):")
    isaret, kor, kk = derinlik_tutarliligi(tum["idw_rakim"][0])
    print(f"  istasyon {len(isaret)}: butun derinlikler ayni isaret {int(isaret.hepsi_ayni.sum())}/{len(isaret)} "
          f"({isaret.hepsi_ayni.mean():.2f}); derinlik ciftlerinde ayni isaret orani ort {isaret.cift_ayni.mean():.3f}")
    print("  derinlikler arasi Spearman (istasyonlar uzerinden):")
    print(kor.round(3).to_string())
    kor.to_csv(os.path.join(OUT, "b_derinlik_spearman.csv"))
    isaret.to_csv(os.path.join(OUT, "b_isaret.csv"), index=False)

    print("\nc) Kalibrasyon (2024-25 RMSE; ogrenen hibritte 3 tohum ortalamasi):")
    for m, dfs in tum.items():
        s, mv = kalibrasyon(dfs)
        print(f"  {m}:")
        print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"  {m}, N = 1 ve 3, pencerenin bittigi ay (2023):")
        print(mv.pivot(index="pencere_son_ay", columns="N_ay", values=["havuz_rmse", "istasyon_rmse_ort"]).round(3).to_string())
        s.to_csv(os.path.join(OUT, f"c_kalibrasyon_{m}.csv"), index=False)
        mv.to_csv(os.path.join(OUT, f"c_kalibrasyon_mevsim_{m}.csv"), index=False)


if __name__ == "__main__":
    main()
