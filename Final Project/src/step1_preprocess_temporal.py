"""
BT4222 Final Project - Group 13 - JDSearch
Step 1: Parsing + Temporal History Analysis (addresses the instructor's feedback)

The instructor noted that in JDSearch, historical ORDERS and CARTS dramatically
outnumber CLICKS, with millions of orders/carts having no recorded clicks.
They asked us to trace the history timeline back in time: convert the consecutive
intervals (seconds in `history_time_list`) into elapsed days BEFORE the final
test query, and watch what happens to the click / cart / order distribution as
we trace user activity further back.

Semantics (verified on the raw file):
  history_{qry,wid,type}_list each have n values (one per historical event).
  history_time_list has n+1 values: [0, gap(e0->e1), ..., gap(e_{n-1}->test_query)].
  => elapsed_seconds_before_query[i] = sum(time[i+1:])   (suffix sum)
     elapsed_days[i] = elapsed_seconds[i] / 86400.

Outputs (written to ../artifacts and ../figures):
  - temporal_type_by_daybin.csv       : composition of CLICK/CART/ORD/FLW per elapsed-day bin
  - temporal_summary.txt               : headline numbers for the report
  - fig_temporal_composition.png       : stacked composition vs elapsed days
  - fig_temporal_clickshare.png        : click share decay vs elapsed days
"""
import os, sys, time, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SRC_USER, RES as ART, FIG, prepare
from data_utils import elapsed_days

TYPES = ["CLICK", "CART", "ORD", "FLW"]
# elapsed-day bin edges
EDGES = [0, 1/1440, 1/24, 1, 7, 30, 90, 180, 365, np.inf]  # min, hour, day, week, month...
LABELS = ["<1min", "1min-1h", "1h-1d", "1-7d", "7-30d", "30-90d", "90-180d", "180-365d", ">365d"]

def main():
    prepare()
    t0 = time.time()
    print("Loading user_behavior_data.txt (919MB)...", flush=True)
    df = pd.read_csv(SRC_USER, sep="\t", dtype=str, low_memory=False)
    n_users = len(df)
    print(f"  users: {n_users:,}  ({time.time()-t0:.0f}s)", flush=True)

    # global accumulators
    # counts[type][bin_index] = number of events
    counts = np.zeros((len(TYPES), len(LABELS)), dtype=np.int64)
    type_idx = {t: i for i, t in enumerate(TYPES)}
    total_events = 0
    total_time_zero = 0
    # also: per-event elapsed_days reservoir for a continuous plot (sample)
    sample_elapsed = []   # (elapsed_days, type_idx)
    rng = np.random.default_rng(42)

    # headline: among events in the OLDEST tercile of each history, click share
    old_counts = np.zeros(len(TYPES), dtype=np.int64)   # events older than 30 days
    recent_counts = np.zeros(len(TYPES), dtype=np.int64)  # events within 1 day

    def split(v):
        if v is None or (isinstance(v, float)) or str(v) == "nan" or str(v).strip() == "":
            return []
        return str(v).split("_")

    t1 = time.time()
    for ridx, (types_s, times_s) in enumerate(zip(df["history_type_list"].values,
                                                  df["history_time_list"].values)):
        types = split(types_s)
        times = split(times_s)
        n = len(types)
        if n == 0:
            continue
        gaps = np.asarray([float(x) for x in times])
        days_checked = elapsed_days(gaps, n)
        elapsed = days_checked * 86400.
        total_time_zero += int((gaps[1:] == 0).sum())
        days = elapsed / 86400.0
        bins = np.digitize(days, EDGES[1:], right=False)  # 0..len(LABELS)-1
        bins = np.clip(bins, 0, len(LABELS) - 1)
        for ti, b in zip(types, bins):
            ci = type_idx.get(ti)
            if ci is None:
                continue
            counts[ci, b] += 1
            total_events += 1
        # recent vs old headline
        d = days
        for ti, dd in zip(types, d):
            ci = type_idx.get(ti)
            if ci is None:
                continue
            if dd <= 1.0:
                recent_counts[ci] += 1
            elif dd > 30.0:
                old_counts[ci] += 1
        # reservoir sample for continuous plot
        if ridx % 20 == 0:
            k = min(n, 30)
            sel = rng.choice(n, size=k, replace=False)
            for s in sel:
                ci = type_idx.get(types[s])
                if ci is not None:
                    sample_elapsed.append((days[s], ci))
        if ridx % 20000 == 0 and ridx > 0:
            print(f"  processed {ridx:,}/{n_users:,} users  ({time.time()-t1:.0f}s)", flush=True)

    print(f"Parsed all histories. total events={total_events:,}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- build composition table ----
    col_tot = counts.sum(axis=0)  # events per bin
    comp = counts / np.maximum(col_tot, 1)  # share of each type within bin
    tbl = pd.DataFrame(counts.T, index=LABELS, columns=TYPES)
    tbl["total_events"] = col_tot
    for t in TYPES:
        tbl[f"{t}_share"] = (counts[type_idx[t]] / np.maximum(col_tot, 1))
    tbl.to_csv(ART / "temporal_type_by_daybin.csv")
    print("\n=== Interaction-type composition by elapsed-days-before-query bin ===")
    print(tbl[[*TYPES, "total_events", "CLICK_share", "CART_share", "ORD_share"]].to_string())

    # ---- headline numbers ----
    rec_tot = recent_counts.sum(); old_tot = old_counts.sum()
    lines = []
    lines.append("TEMPORAL HISTORY ANALYSIS - headline numbers (full data, 173,831 users)")
    lines.append("")
    lines.append(f"Total historical events parsed: {total_events:,}")
    lines.append("")
    lines.append("Within <=1 day before the test query:")
    for t in TYPES:
        c = recent_counts[type_idx[t]]
        lines.append(f"   {t:6s}: {c:>12,}  ({100*c/max(rec_tot,1):5.2f}% of recent events)")
    lines.append(f"   recent click share = {100*recent_counts[type_idx['CLICK']]/max(rec_tot,1):.2f}%")
    lines.append("")
    lines.append("More than 30 days before the test query:")
    for t in TYPES:
        c = old_counts[type_idx[t]]
        lines.append(f"   {t:6s}: {c:>12,}  ({100*c/max(old_tot,1):5.2f}% of old events)")
    lines.append(f"   old click share = {100*old_counts[type_idx['CLICK']]/max(old_tot,1):.2f}%")
    lines.append("")
    lines.append("Click share by bin (near -> far in time):")
    for lab in LABELS:
        lines.append(f"   {lab:10s}: CLICK {100*tbl.loc[lab,'CLICK_share']:5.2f}%  "
                     f"CART {100*tbl.loc[lab,'CART_share']:5.2f}%  "
                     f"ORD {100*tbl.loc[lab,'ORD_share']:5.2f}%  "
                     f"(n={int(tbl.loc[lab,'total_events']):,})")
    txt = "\n".join(lines)
    (ART / "temporal_summary.txt").write_text(txt)
    print("\n" + txt)

    # ---- Figure 1: stacked composition vs bin ----
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(LABELS))
    colors = {"CLICK": "#2b7bba", "CART": "#2ca089", "ORD": "#d2691e", "FLW": "#999999"}
    for t in TYPES:
        vals = tbl[f"{t}_share"].values * 100
        ax.bar(LABELS, vals, bottom=bottom, label=t, color=colors[t])
        bottom += vals
    ax.set_ylabel("Share of events within bin (%)")
    ax.set_xlabel("Elapsed time before the test query")
    ax.set_title("Interaction-type composition as history is traced back in time")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(FIG / "fig_temporal_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: click share decay line ----
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(LABELS))
    ax.plot(x, tbl["CLICK_share"].values * 100, "o-", color=colors["CLICK"], label="CLICK")
    ax.plot(x, tbl["CART_share"].values * 100, "s-", color=colors["CART"], label="CART")
    ax.plot(x, tbl["ORD_share"].values * 100, "^-", color=colors["ORD"], label="ORD")
    ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_ylabel("Share within bin (%)"); ax.set_xlabel("Elapsed time before the test query")
    ax.set_title("Observed action composition changes with history age")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG / "fig_temporal_clickshare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nDONE step1 temporal. ({time.time()-t0:.0f}s total)")

if __name__ == "__main__":
    main()
