"""
Policy-aware optimizer:
 - Dept-specific nurse costs (optional).
 - Minimum service levels per dept: served >= rate * demand.
 - Same continuous+round approach (works with OSQP/SCS/CLARABEL/SCIPY).
Outputs: optimal_allocation_policy.csv
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
OUT_CSV    = Path(PATHS["alloc_csv"].replace(".csv", "_policy.csv"))
PPN_MAP    = STAFF["patients_per_nurse"]
DEFAULT_NC = float(STAFF["nurse_cost"])
SHORT_PEN  = float(STAFF["shortage_penalty"])
NURSE_COSTS = {**{k: float(v) for k,v in STAFF.get("nurse_costs", {}).items()}}
MIN_SVC     = {**{k: float(v) for k,v in STAFF.get("min_service_rate", {}).items()}}

def dept_costs(depts):
    return np.array([float(NURSE_COSTS.get(d, DEFAULT_NC)) for d in depts], dtype=float)

def min_rates(depts):
    return np.array([float(MIN_SVC.get(d, 0.0)) for d in depts], dtype=float)  # 0 = no lower bound

pred = pd.read_csv(PRED_CSV, parse_dates=["date"])
pred["dept"] = pred["dept"].astype(str)
pred["beds"] = pd.to_numeric(pred["beds"], errors="coerce").fillna(0.0)
pred["pred_patients"] = pd.to_numeric(pred["pred_patients"], errors="coerce").fillna(0.0)

rows = []
for trust, g in pred.groupby("trust"):
    g = g.copy()
    depts = g["dept"].tolist()
    beds  = g["beds"].to_numpy(float)
    dem   = g["pred_patients"].to_numpy(float)
    ppn   = np.array([PPN_MAP.get(d, 5) for d in depts], float)

    max_n = np.ceil(beds / ppn)
    budget = int(max_n.sum())              # full-bed staffing
    nc = dept_costs(depts)                 # per-dept nurse cost
    r  = min_rates(depts)                  # min service rates per dept

    nurses = cp.Variable(len(depts), nonneg=True)
    served = cp.Variable(len(depts), nonneg=True)
    short  = cp.Variable(len(depts), nonneg=True)

    cons = [
        served <= beds,
        served <= cp.multiply(ppn, nurses),
        short  >= dem - served,
        nurses <= max_n,
        cp.sum(nurses) <= budget,
        # min service levels: served >= r * demand
        served >= cp.multiply(r, dem)
    ]
    obj = cp.Minimize(cp.sum(cp.multiply(nc, nurses)) + SHORT_PEN * cp.sum(short))
    prob = cp.Problem(obj, cons)

    used = "none"
    for s in ["OSQP","SCS","CLARABEL","SCIPY"]:
        if s in cp.installed_solvers():
            try:
                prob.solve(solver=getattr(cp, s), verbose=False)
                if prob.status in ("optimal","optimal_inaccurate"):
                    used = s; break
            except Exception: pass

    n = nurses.value if nurses.value is not None else np.zeros(len(depts))
    s = served.value  if served.value  is not None else np.minimum(beds, ppn*n)
    h = short.value   if short.value   is not None else np.maximum(0, dem-s)

    # round nurses; recompute after rounding
    n = np.rint(n)
    s = np.minimum(beds, ppn*n)
    h = np.maximum(0.0, dem - s)

    for i,d in enumerate(depts):
        rows.append({
            "trust": trust, "dept": d, "beds": float(beds[i]),
            "pred_patients": float(dem[i]),
            "nurses_alloc": float(n[i]),
            "served": float(s[i]),
            "shortage": float(h[i]),
            "solver": used,
            "status": prob.status
        })

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print("Wrote:", OUT_CSV.resolve())

tot = out.groupby("dept", as_index=False)[["pred_patients","served","shortage"]].sum()
print("\nPolicy totals by dept:")
print(tot.to_string(index=False))
