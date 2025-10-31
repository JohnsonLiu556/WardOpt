"""
Reads:
  - optimal_allocation.csv
  - dept_totals_summary.csv
Produces:
  - plots/shortage_by_trust_top10.png
  - plots/naive_vs_optimal_totals.png
  - insights_summary.txt  
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ALLOC_FILE = Path("optimal_allocation.csv")
DEPT_SUM_FILE = Path("dept_totals_summary.csv")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

from yaml import safe_load

CFG = safe_load(Path("config.yaml").read_text())
PATHS = CFG["paths"]

ALLOC_FILE = Path(PATHS["alloc_csv"])
DEPT_SUM_FILE = Path(PATHS["dept_totals_csv"])
PLOTS_DIR = Path(PATHS["plots_dir"]); PLOTS_DIR.mkdir(exist_ok=True)


alloc = pd.read_csv(ALLOC_FILE)
dept = pd.read_csv(DEPT_SUM_FILE)

tot_naive = alloc["naive_shortage"].sum()
tot_opt   = alloc["shortage"].sum()
reduction = tot_naive - tot_opt
pct_red   = 0.0 if tot_naive == 0 else 100 * reduction / tot_naive

total_beds     = alloc["beds"].sum()
total_demand   = alloc["pred_patients"].sum()
total_served   = alloc["served"].sum()

trust_tot = (alloc
    .groupby("trust", as_index=False)[["naive_shortage","shortage"]]
    .sum())
trust_tot["improvement"] = trust_tot["naive_shortage"] - trust_tot["shortage"]
top10 = trust_tot.sort_values("improvement", ascending=False).head(10)

# Plot: naive vs optimal totals
plt.figure()
plt.bar(["naive","optimal"], [tot_naive, tot_opt])
plt.title("Total shortage (all trusts, all depts)")
plt.ylabel("Patients short")
plt.tight_layout()
plt.savefig(PLOTS_DIR/"naive_vs_optimal_totals.png", dpi=160)

# Plot: top-10 trusts by improvement
plt.figure(figsize=(9,5))
plt.barh(top10["trust"], top10["improvement"])
plt.gca().invert_yaxis()
plt.title("Top-10 trusts by shortage reduction (patients)")
plt.xlabel("Patients fewer short (optimal vs naive)")
plt.tight_layout()
plt.savefig(PLOTS_DIR/"shortage_by_trust_top10.png", dpi=160)

lines = []
lines.append("Healthcare Resource Allocation — Insights\n")
lines.append(f"Total demand:        {int(total_demand):,}")
lines.append(f"Total beds:          {int(total_beds):,}")
lines.append(f"Total served:        {int(total_served):,}")
lines.append(f"Naive shortage:      {int(tot_naive):,}")
lines.append(f"Optimal shortage:    {int(tot_opt):,}")
lines.append(f"Reduction (patients):{int(reduction):,}  ({pct_red:.1f}% ↓)")
lines.append("")
lines.append("By department (totals):")
dept_fmt = dept[["dept","pred_patients","served","shortage","naive_shortage","improvement"]].copy()
dept_fmt = dept_fmt.sort_values("shortage")
for _, r in dept_fmt.iterrows():
    lines.append(f"  - {r['dept']}: shortage {int(r['shortage'])} vs naive {int(r['naive_shortage'])} (improve {int(r['improvement'])})")
lines.append("")
lines.append("Top-10 trusts by improvement:")
for _, r in top10.iterrows():
    lines.append(f"  - {r['trust']}: {int(r['improvement'])} patients fewer short")

Path("insights_summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print("\nSaved plots in ./plots and summary in insights_summary.txt")
