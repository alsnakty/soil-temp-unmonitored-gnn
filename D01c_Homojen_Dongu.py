import os
import subprocess
import sys
import pandas as pd
from D01d_Derinlik_Yayma import yay, S as YS

KOK = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(KOK, "DATASET")
PY = sys.executable
MAX_TUR = 8
ORTA = dict(KIRILMA_T="25", KIRILMA_ADIM="0.5", KIRILMA_ISABET="3", KIRILMA_ORAN="0.6")
BIRLESIK = "kirilma_istasyon_birlesik.csv"


def kos(betik, **env):
    e = dict(os.environ, PYTHONIOENCODING="utf-8", **env)
    r = subprocess.run([PY, "-W", "ignore", os.path.join(KOK, betik)], env=e, capture_output=True, text=True,
                       encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"{betik} hata:\n{r.stderr[-2000:]}")
    return r.stdout


def yakin(s, d, ay, df, kol):
    p = pd.Period(str(ay)[:7], "M")
    x = df[(df.istasyon_no == s) & (df.derinlik == d)]
    return any(abs((pd.Period(str(v)[:7], "M") - p).n) <= 4 for v in x[kol])


META = pd.read_csv(os.path.join(DS, "MGM_istasyonlar_final.csv"), encoding="cp1254").set_index("istasyon_no")
L = pd.read_csv(os.path.join(DS, "kirilma_istasyon_orta.csv")).assign(tur=0, kaynak="tarama")
ELLE = os.path.join(DS, "kirilma_elle.csv")
if os.path.exists(ELLE):
    L = pd.concat([L, pd.read_csv(ELLE).assign(tur=0, kaynak="elle")], ignore_index=True)
    print(f"elle eklenen kirilma: {len(L[L.kaynak == 'elle'])}", flush=True)
for tur in range(1, MAX_TUR + 1):
    L.to_csv(os.path.join(DS, BIRLESIK), index=False, encoding="utf-8-sig")
    cikti = kos("D01_Hedef_QC.py", QC_KIRILMA=BIRLESIK)
    print(f"[tur {tur}] D01: " + [s for s in cikti.splitlines() if s.startswith("Duzeltilen")][0], flush=True)
    kos("D01a_Kirilma_Taramasi.py", KIRILMA_MASTER="master_dataset_qc.csv",
        KIRILMA_ETIKET=f"_dongu{tur}", **ORTA)
    yeni_tarama = pd.read_csv(os.path.join(DS, f"kirilma_istasyon_dongu{tur}.csv"))
    atl = pd.read_csv(os.path.join(DS, "hedef_qc_atlanan.csv"))
    kay = pd.read_csv(os.path.join(DS, "hedef_qc_duzeltmeler.csv"))
    yet = kay[kay.durum != "duzeltildi"]
    yeni = []
    for _, r in yeni_tarama.iterrows():
        if (yakin(r.istasyon_no, r.derinlik, r.ay, L, "ay") or yakin(r.istasyon_no, r.derinlik, r.ay, atl, "ay")
                or yakin(r.istasyon_no, r.derinlik, r.ay, yet, "kirilma")):
            continue
        yeni.append(r)
    print(f"[tur {tur}] yeniden tarama: {len(yeni_tarama)} kirilma, yeni {len(yeni)}: "
          + ", ".join(f"{r.istasyon_ad} {r.derinlik} {r.ay}" for r in yeni), flush=True)
    L2 = pd.concat([L, pd.DataFrame(yeni)], ignore_index=True) if yeni else L
    mq = pd.read_csv(os.path.join(DS, "master_dataset_qc.csv"), usecols=["istasyon_no", "tarih"] + list(YS.values()))
    yy = yay(mq, kay, META, L2)
    yy = [r for _, r in yy.iterrows() if not (yakin(r.istasyon_no, r.derinlik, r.ay, atl, "ay")
                                              or yakin(r.istasyon_no, r.derinlik, r.ay, yet, "kirilma"))]
    print(f"[tur {tur}] derinlik yayma: yeni {len(yy)}: "
          + ", ".join(f"{r.istasyon_ad} {r.derinlik} {r.ay} ({r.adim_medyan:+.2f})" for r in yy), flush=True)
    if not yeni and not yy:
        break
    parca = [L]
    if yeni:
        parca.append(pd.DataFrame(yeni).assign(tur=tur, kaynak="tarama"))
    if yy:
        parca.append(pd.DataFrame(yy).assign(tur=tur))
    L = pd.concat(parca, ignore_index=True)
L.to_csv(os.path.join(DS, BIRLESIK), index=False, encoding="utf-8-sig")
print(f"Bitti: toplam kirilma listesi {len(L)} (tur dagilimi {L.tur.value_counts().sort_index().to_dict()}, "
      f"kaynak {L.kaynak.value_counts().to_dict()})")
