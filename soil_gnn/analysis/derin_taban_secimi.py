import os
import sys

import numpy as np
import pandas as pd

pd.set_option("compute.use_numexpr", False)
ANA_DIZIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ANA_DIZIN, "src"))
sys.path.insert(0, os.path.join(ANA_DIZIN, "analysis"))
import eksik_derinlik as E
import eksik_derinlik_2 as E2

OUT = os.path.join(ANA_DIZIN, "results", "folds", "analysis_deep_baseline")


def main():
    os.makedirs(OUT, exist_ok=True)
    df_m = pd.read_csv(E.SoilConfig().master_path, parse_dates=["tarih"])
    satir = []
    for k in range(5):
        c = E.SoilConfig(); c.holdout = E.kat_istasyonlari(k); c.ic_dogrulama = True
        P = E.hazirla(c, df_m)
        T, dates, st_no, lat, lon, rak = E.veri(P)
        ti = np.nonzero(dates.year <= 2022)[0]
        ictr = P["g_tr"].st_global
        if set(ictr) & set(P["ic_global"]):
            raise RuntimeError("ic dogrulama istasyonu secimde")
        r = E.IDWM.satirlar(T[ti], dates[ti], st_no, lat, lon, ictr, [np.array([i]) for i in ictr],
                            float(E.IDW_P[k]), E.IDW_K[k], rak).dropna(subset=["y_pred"])
        r["r"] = r.y_true - r.y_pred
        r20 = r[r.derinlik == 20][["istasyon_no", "tarih", "r"]]
        for d in E.HEDEF:
            t = E2.a5_ozellik(r20, E.artik_tablosu(r20, r[r.derinlik == d].copy()), E2.DZ[d])
            en = None
            for kp in E2.KAPPA:
                s = E.toplamlar(t, f"s_{kp}").sum()
                g = s.smr / s.smm
                rss = s.srr - 2 * g * s.smr + g ** 2 * s.smm
                satir.append(dict(kat=k, derinlik=d, kappa=kp, gamma=g, rmse_ic_egitim=float(np.sqrt(rss / s.n)),
                                  n_istasyon=len(ictr), n_satir=int(s.n)))
                if en is None or rss < en[0]:
                    en = (rss, kp, g)
            print(f"kat {k} {d} cm: kappa {en[1]}, gamma {en[2]:.4f}", flush=True)
    t = pd.DataFrame(satir)
    t.to_csv(os.path.join(OUT, "taban_param_tum.csv"), index=False)
    sec = t.loc[t.groupby(["kat", "derinlik"]).rmse_ic_egitim.idxmin()]
    sec.to_csv(os.path.join(OUT, "taban_param.csv"), index=False)
    for k, g in sec.groupby("kat"):
        print(f"kat {k}: --taban-param " + ",".join(f"{int(r.derinlik)}:{int(r.kappa)}:{r.gamma:.4f}" for r in g.itertuples()))


if __name__ == "__main__":
    main()
