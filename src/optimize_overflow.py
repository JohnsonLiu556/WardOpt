"""
Optimizer with OVERFLOW BEDS.
- Each department may open temporary 'overflow' beds up to max_pct * beds.
- Overflow has a cost; nurses must also scale to staff any added beds.

Reads:  config.yaml, Monthly_Predictions.csv
Writes: optimal_allocation_overflow.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import cvxpy as cp
from yaml import safe_load
import warnings
warnings.filterwarnings("ignore", message="Solution may be inaccurate.*")

CFG   = safe_load(Path("config.yaml").read_text())
PATHS = CFG["paths"]
STAFF = CFG["staffing"]

PRED_CSV   = Path(PATHS["pred_csv"])
OUT_CSV    = Path(PATHS["alloc_csv"].replace(".csv", "_overflow.csv"))

PPN_MAP    = STAFF["patients_per_nurse"]
NURSE_COST = float(STAFF["nurse_cost"])
SHORT_PEN  = float(STAFF["shortage_penalty"])

OVF = STAFF.get("overflow", {}) or {}
OVF_MAX_PCT = float(OVF.get("max_pct", 0.10))
OVF_BED_COST = float(OVF.get("bed_cost", 20.0))  # pick << SHORT_PEN, >> NURSE_COST

pred = pd.read_csv(PRED_CSV, parse_dates=["date"])
pred["dept"] = pred["dept"].astype(str)
pred["beds"] = pd.to_numeric(pred["beds"], errors="coerce").fillna(0.0)
pred["pred_patients"] = pd.to_numeric(pred["pred_patients"], errors="coerce").fillna(0.0)

rows = []

def solve_trust(g: pd.DataFrame):
    depts = g["dept"].tolist()
    beds  = g["beds"].to_numpy(float)
    dem   = g["pred_patients"].to_numpy(float)
    ppn   = np.array([PPN_MAP.get(d, 5) for d in depts], dtype=float)

    nurses = cp.Variable(len(depts), nonneg=True)
    served = cp.Variable(len(depts), nonneg=True)
    short  = cp.Variable(len(depts), nonneg=True)
    extra  = cp.Variable(len(depts), nonneg=True)  # overflow beds

    cons = []
    # overflow limits
    cons += [extra <= OVF_MAX_PCT * beds]
    # service capacity
    cons += [served <= beds + extra]
    cons += [served <= cp.multiply(ppn, nurses)]
    # staffing bound induced by (beds + extra)
    cons += [nurses <= (beds + extra) / ppn]
    # shortage definition
    cons += [short >= dem - served]
    cons += [short >= 0]

    # Objective: nurses + overflow beds + shortages
    obj = cp.Minimize(
        NURSE_COST   * cp.sum(nurses) +
        OVF_BED_COST * cp.sum(extra)  +
        SHORT_PEN    * cp.sum(short)
    )
    prob = cp.Problem(obj, cons)

    used = "none"
    for s in ["OSQP", "SCS", "CLARABEL", "SCIPY"]:
        if s in cp.installed_solvers():
            try:
                prob.solve(solver=getattr(cp, s), verbose=False)
                if prob.status in ("optimal","optimal_inaccurate"):
                    used = s; break
            except Exception: pass

    # extract values (fallbacks)
    n = nurses.value if nurses.value is not None else np.zeros(len(depts))
    s = served.value  if served.value  is not None else np.minimum(beds, ppn*n)
    h = short.value   if short.value   is not None else np.maximum(0, dem-s)
    e = extra.value   if extra.value   is not None else np.zeros(len(depts))
    n = np.rint(n)
    s = np.minimum(beds + e, ppn * n)
    h = np.maximum(0.0, dem - s)

    return n, s, h, e, used, prob.status

for trust, g in pred.groupby("trust"):
    n, s, h, e, solver_used, status = solve_trust(g)
    for i, d in enumerate(g["dept"].tolist()):
        rows.append({
            "trust": trust,
            "dept": d,
            "beds": float(g["beds"].iloc[i]),
            "extra_beds": float(e[i]),
            "pred_patients": float(g["pred_patients"].iloc[i]),
            "nurses_alloc": float(n[i]),
            "served": float(s[i]),
            "shortage": float(h[i]),
            "solver": solver_used,
            "status": status
        })

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print("Wrote:", OUT_CSV.resolve())

tot = out.groupby("dept", as_index=False)[["beds","extra_beds","pred_patients","served","shortage"]].sum()
tot["overflow_util_pct"] = 100 * tot["extra_beds"] / tot["beds"]
print("\nOverflow totals by dept:")
print(tot.to_string(index=False))
