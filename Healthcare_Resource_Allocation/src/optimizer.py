"""
Optimizer using CVXPY:

Allocate nurses across departments to meet forecasted demand while
respecting bed capacity and a per-trust nurse budget.

Inputs:
  - Monthly_Predictions.csv  (date, trust, dept, beds, patients, pred_patients)
  - Hospital_Capacity_Data.csv  (used to estimate a baseline, but we also allow full-bed staffing)

Outputs:
  - optimal_allocation.csv
  - dept_totals_summary.csv
  - a bar chart comparing naive vs optimal shortages
"""

from pathlib import Path
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

from yaml import safe_load

CFG = safe_load(Path("config.yaml").read_text())
PATHS = CFG["paths"]
STAFF = CFG["staffing"]

INPUT_PRED = Path(PATHS["pred_csv"])
INPUT_CAP  = Path(PATHS["cap_csv"])
OUTPUT_ALLOCS = Path(PATHS["alloc_csv"])
OUTPUT_DEPT_SUM = Path(PATHS["dept_totals_csv"])

PATIENTS_PER_NURSE = STAFF["patients_per_nurse"]
NURSE_COST = float(STAFF["nurse_cost"])
SHORTAGE_PENALTY = float(STAFF["shortage_penalty"])
BUDGET_MODE = STAFF.get("budget_mode", "full_beds")

INPUT_PRED = Path("Monthly_Predictions.csv")
INPUT_CAP  = Path("Hospital_Capacity_Data.csv")
OUTPUT_ALLOCS = Path("optimal_allocation.csv")
OUTPUT_DEPT_SUM = Path("dept_totals_summary.csv")

# patients served per 1 nurse (simple staffing ratios)
PATIENTS_PER_NURSE = {
    "Critical Care Adult": 2,   # ICU 1:2
    "G&A Adult": 5,             # Med/Surg 1:5
    "G&A Paediatric": 4         # Paeds 1:4
}

NURSE_COST = 1.0
SHORTAGE_PENALTY = 200.0


def to_num(s):
    """coerce to numeric with NaN -> 0"""
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def solve_with_fallback(problem):
    """try multiple solvers; return (status, solver_name)"""
    for solver in [cp.ECOS, cp.OSQP, cp.SCS]:
        try:
            problem.solve(solver=solver, verbose=False)
            if problem.status in ("optimal", "optimal_inaccurate"):
                return problem.status, solver.__name__
        except Exception:
            pass
    # last try: default settings
    try:
        problem.solve(verbose=False)
    except Exception:
        pass
    return problem.status, "none"

pred = pd.read_csv(INPUT_PRED, parse_dates=["date"])
cap  = pd.read_csv(INPUT_CAP,  parse_dates=["date"])

for df in (pred, cap):
    df["dept"]  = df["dept"].astype(str)
    df["beds"]  = to_num(df.get("beds", 0))
    df["patients"] = to_num(df.get("patients", 0))

pred["pred_patients"] = to_num(pred.get("pred_patients", 0))

latest_date = pred["date"].max()

# If Hospital_Capacity_Data.csv is month-aligned, grab that month; else fall back to pred
cap_latest = cap[cap["date"] == latest_date][["trust","dept","patients"]].copy()
if cap_latest.empty:
    cap_latest = pred[["trust","dept","patients"]].copy()


rows_out = []

trusts = sorted(pred["trust"].unique().tolist())
print(f"Optimizing {len(trusts)} trusts for {latest_date.date()} ...")

for trust in trusts:
    block = pred[pred["trust"] == trust].copy()
    if block.empty:
        continue

    # vectors
    depts = block["dept"].tolist()
    beds = block["beds"].to_numpy(float)
    demand = block["pred_patients"].to_numpy(float)
    ppn = np.array([PATIENTS_PER_NURSE.get(d, 5) for d in depts], dtype=float)

    # upper bound on nurses per dept implied by beds
    nurses_cap_by_beds = np.ceil(beds / ppn)

    # budget = allow full-bed staffing (so beds are the main bottleneck)
    # budget = int(nurses_cap_by_beds.sum()) --> improved version below
    if BUDGET_MODE == "from_actual":
        # estimate from actual patients in cap_latest (if available)
        base = cap_latest[cap_latest["trust"] == trust].copy()
        if not base.empty:
            base = base.set_index("dept").reindex(depts).fillna(0.0)
            base_ppn = np.array([PATIENTS_PER_NURSE.get(d, 5) for d in depts], float)
            est_nurses = np.ceil(pd.to_numeric(base["patients"], errors="coerce").fillna(0).to_numpy() / base_ppn)
            budget = int(max(est_nurses.sum(), 0))
        else:
            budget = int(nurses_cap_by_beds.sum())
    else:
        # default: allow full-bed staffing
        budget = int(nurses_cap_by_beds.sum())


    nurses_alloc = cp.Variable(len(depts), nonneg=True)  # nurses per dept
    served = cp.Variable(len(depts), nonneg=True)        # patients served per dept
    shortage = cp.Variable(len(depts), nonneg=True)      # unmet patients

    cons = []
    cons += [served <= beds]
    cons += [served <= cp.multiply(ppn, nurses_alloc)]
    cons += [shortage >= demand - served]
    cons += [shortage >= 0]
    cons += [cp.sum(nurses_alloc) <= budget]
    cons += [nurses_alloc <= nurses_cap_by_beds]

    obj = cp.Minimize(NURSE_COST * cp.sum(nurses_alloc) + SHORTAGE_PENALTY * cp.sum(shortage))
    prob = cp.Problem(obj, cons)
    status, used_solver = solve_with_fallback(prob)

    # read values (fallback to zeros if solver failed)
    n_alloc = nurses_alloc.value if nurses_alloc.value is not None else np.zeros(len(depts))
    s_served = served.value if served.value is not None else np.zeros(len(depts))
    s_short = shortage.value if shortage.value is not None else demand.copy()

    base = cap_latest[cap_latest["trust"] == trust].copy()
    if not base.empty:
        base = base.set_index("dept").reindex(depts).fillna(0.0)
        base_ppn = np.array([PATIENTS_PER_NURSE.get(d, 5) for d in depts], dtype=float)
        naive_nurses = np.ceil(to_num(base["patients"]).to_numpy() / base_ppn)
        naive_nurses = np.minimum(naive_nurses, nurses_cap_by_beds)  # cap by beds

        tot = naive_nurses.sum()
        if tot > budget and tot > 0:
            scale = budget / tot
            naive_nurses = np.floor(naive_nurses * scale)

        naive_served = np.minimum(beds, base_ppn * naive_nurses)
        naive_short = np.maximum(0.0, demand - naive_served)
    else:
        naive_short = demand.copy()

    for i, d in enumerate(depts):
        rows_out.append({
            "trust": trust,
            "dept": d,
            "beds": float(beds[i]),
            "pred_patients": float(demand[i]),
            "nurses_alloc": float(n_alloc[i]),
            "served": float(s_served[i]),
            "shortage": float(s_short[i]),
            "naive_shortage": float(naive_short[i]),
            "status": status,
            "solver": used_solver
        })


out = pd.DataFrame(rows_out)
out.to_csv(OUTPUT_ALLOCS, index=False)
print(f"\nWrote allocations to: {OUTPUT_ALLOCS.resolve()}")

print("\nStatus counts:")
print(out["status"].value_counts(dropna=False))
print("\nSolver usage:")
print(out["solver"].value_counts(dropna=False))

totals = out.groupby("dept", as_index=False)[["beds","pred_patients","served","shortage","naive_shortage"]].sum()
totals["improvement"] = totals["naive_shortage"] - totals["shortage"]
totals = totals.sort_values("shortage").reset_index(drop=True)
totals.to_csv(OUTPUT_DEPT_SUM, index=False)

print("\n=== Shortage summary (lower is better) ===")
print(totals[["dept","shortage","naive_shortage","improvement"]].to_string(index=False))
print(f"\nSaved: {OUTPUT_DEPT_SUM.resolve()}")

# Plot: naive vs optimal shortages by dept
plt.figure()
plt.bar(totals["dept"], totals["naive_shortage"], label="naive")
plt.bar(totals["dept"], totals["shortage"], label="optimal", alpha=0.7)
plt.title("Total shortage by department (naive vs optimal)")
plt.ylabel("Patients short")
plt.xticks(rotation=18)
plt.legend()
plt.tight_layout()
plt.show()
