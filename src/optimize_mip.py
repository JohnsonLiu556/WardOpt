from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
from yaml import safe_load

warnings.filterwarnings("ignore", message="Solution may be inaccurate.*")

# ---------- config ----------
CFG = safe_load(Path("config.yaml").read_text())
PATHS = CFG["paths"]
STAFF = CFG["staffing"]

PRED_CSV = Path(PATHS["pred_csv"])
OUT_CSV  = Path(PATHS["alloc_csv"].replace(".csv", "_mip.csv"))
PPN_MAP = STAFF["patients_per_nurse"]          # patients per nurse by dept
NURSE_COST = float(STAFF["nurse_cost"])
SHORTAGE_PENALTY = float(STAFF["shortage_penalty"])

# ---------- load data ----------
pred = pd.read_csv(PRED_CSV, parse_dates=["date"])
pred["dept"] = pred["dept"].astype(str)
pred["beds"] = pd.to_numeric(pred["beds"], errors="coerce").fillna(0.0)
pred["pred_patients"] = pd.to_numeric(pred["pred_patients"], errors="coerce").fillna(0.0)

rows = []

def solve_trust(block: pd.DataFrame):
    """Solve continuous problem, then round nurses to ints and recompute served/shortage."""
    depts = block["dept"].tolist()
    beds  = block["beds"].to_numpy(float)
    dem   = block["pred_patients"].to_numpy(float)
    ppn   = np.array([PPN_MAP.get(d, 5) for d in depts], dtype=float)

    # caps/budget: allow full-bed staffing
    max_n = np.ceil(beds / ppn)
    budget = int(max_n.sum())

    # decision vars (continuous)
    nurses = cp.Variable(len(depts), nonneg=True)
    served = cp.Variable(len(depts), nonneg=True)
    short  = cp.Variable(len(depts), nonneg=True)

    cons = [
        served <= beds,
        served <= cp.multiply(ppn, nurses),
        short  >= dem - served,
        nurses <= max_n,
        cp.sum(nurses) <= budget,
    ]
    obj = cp.Minimize(NURSE_COST * cp.sum(nurses) + SHORTAGE_PENALTY * cp.sum(short))
    prob = cp.Problem(obj, cons)

    # try available continuous solvers
    used = "none"
    for s in ["OSQP", "SCS", "CLARABEL", "SCIPY"]:
        if s in cp.installed_solvers():
            try:
                prob.solve(solver=getattr(cp, s), verbose=False)
                if prob.status in ("optimal", "optimal_inaccurate"):
                    used = s
                    break
            except Exception:
                pass

    # read values (fallback to zeros)
    n = nurses.value if nurses.value is not None else np.zeros(len(depts))
    s = served.value  if served.value  is not None else np.minimum(beds, ppn * n)
    h = short.value   if short.value   is not None else np.maximum(0.0, dem - s)

    # integerize nurses and recompute served/shortage
    n_int = np.rint(n)  # nearest integer
    s_int = np.minimum(beds, ppn * n_int)
    h_int = np.maximum(0.0, dem - s_int)

    return n_int, s_int, h_int, used, prob.status

# ---------- solve per trust ----------
for trust, g in pred.groupby("trust"):
    n, s, h, solver_used, status = solve_trust(g)
    for i, d in enumerate(g["dept"].tolist()):
        rows.append({
            "trust": trust,
            "dept": d,
            "beds": float(g["beds"].iloc[i]),
            "pred_patients": float(g["pred_patients"].iloc[i]),
            "nurses_alloc": float(n[i]),    # integer (rounded)
            "served": float(s[i]),
            "shortage": float(h[i]),
            "solver": solver_used,
            "status": status,
        })

# ---------- save & print ----------
out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print("Wrote:", OUT_CSV.resolve())

tot = out.groupby("dept", as_index=False)[["pred_patients","served","shortage"]].sum()
print("\nInteger-like solution (rounded) totals by dept:")
print(tot.to_string(index=False))

print("\nSolvers used (counts):")
print(out["solver"].value_counts(dropna=False))
print("\nStatuses (counts):")
print(out["status"].value_counts(dropna=False))

