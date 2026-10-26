"""
Step 1b: Pin down the EXACT data-retention windows per interaction type.
Fine 1-day bins from 0 to 130 days to locate the click / cart cutoffs precisely.
"""
import time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import os
SRC_USER = os.environ.get("JD_USER_TXT",
    "/Users/steff/Library/CloudStorage/OneDrive-Personal/Descargas/Steffany/Estudios/NUS/BT4222 - Mining Web Data for Business Insights/Project/JDsearch/user_behavior_data.txt")
BASE = Path("/Users/steff/Documents/GitHub/bt4212_finalProj/Final Project")
ART = BASE/"artifacts"; FIG = BASE/"figures"
TYPES=["CLICK","CART","ORD","FLW"]; tix={t:i for i,t in enumerate(TYPES)}

df = pd.read_csv(SRC_USER, sep="\t", dtype=str, usecols=["history_type_list","history_time_list"], low_memory=False)
MAXD=130
# counts[type, day] for day in 0..MAXD-1 (1-day resolution), plus overflow bin at MAXD
counts=np.zeros((4,MAXD+1),dtype=np.int64)
def split(v):
    if v is None or str(v)=="nan" or str(v).strip()=="" : return []
    return str(v).split("_")
t0=time.time()
for types_s,times_s in zip(df["history_type_list"].values, df["history_time_list"].values):
    types=split(types_s); times=split(times_s); n=len(types)
    if n==0: continue
    tarr=np.fromstring(times_s, sep="_") if times_s and str(times_s)!="nan" else np.array([])
    if len(tarr)!=n+1: continue
    suffix=np.cumsum(tarr[::-1])[::-1]; elapsed=suffix[1:]
    days=(elapsed/86400.0).astype(np.int64)
    days=np.clip(days,0,MAXD)
    for ti,dd in zip(types,days):
        ci=tix.get(ti)
        if ci is not None: counts[ci,dd]+=1
print(f"parsed in {time.time()-t0:.0f}s")

# find last day index where CLICK and CART still have >0 events
click_days=np.where(counts[0]>0)[0]; cart_days=np.where(counts[1]>0)[0]; ord_days=np.where(counts[2]>0)[0]
lines=[]
lines.append("EXACT RETENTION WINDOWS (1-day resolution, 0..130d; bin 130 = >=130d overflow)")
lines.append(f"CLICK present on days: min={click_days.min()} max<130={click_days[click_days<MAXD].max() if (click_days<MAXD).any() else 'none'}  total_click={counts[0].sum():,}")
lines.append(f"CART  present on days: min={cart_days.min()} max<130={cart_days[cart_days<MAXD].max() if (cart_days<MAXD).any() else 'none'} total_cart={counts[1].sum():,}")
lines.append(f"ORD   present on days: min={ord_days.min()} max<130={ord_days[ord_days<MAXD].max() if (ord_days<MAXD).any() else 'none'}  overflow(>=130d) ord={counts[2,MAXD]:,}")
lines.append("")
lines.append("CLICK events per day near the cutoff (days 3..12):")
for d in range(3,13):
    lines.append(f"   day {d:3d}: CLICK={counts[0,d]:>9,}  CART={counts[1,d]:>9,}  ORD={counts[2,d]:>9,}")
lines.append("")
lines.append("CART events per day near its cutoff (days 60..100):")
for d in range(60,101,5):
    lines.append(f"   day {d:3d}: CLICK={counts[0,d]:>9,}  CART={counts[1,d]:>9,}  ORD={counts[2,d]:>9,}")
txt="\n".join(lines)
(ART/"retention_cutoffs.txt").write_text(txt)
print(txt)

# figure: events per day by type, log-y, up to 130 d
fig,ax=plt.subplots(figsize=(10,5))
x=np.arange(MAXD)
for t,c in zip(TYPES,["#2b7bba","#2ca089","#d2691e","#999999"]):
    ax.plot(x, counts[tix[t],:MAXD]+1, label=t, color=c)
ax.set_yscale("log"); ax.set_xlabel("Whole days before test query"); ax.set_ylabel("Historical events (+1, log)")
ax.set_title("Per-type event volume vs age: click window ~7d, cart window ~90d, orders ~1yr")
ax.axvline(7,ls="--",c="#2b7bba",alpha=.5); ax.axvline(90,ls="--",c="#2ca089",alpha=.5)
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); fig.savefig(FIG/"fig_retention_cutoffs.png",dpi=150); plt.close(fig)
print("saved fig_retention_cutoffs.png")
