"""
Runs the optimizer over a small grid of (penalty_x, budget_x) multipliers.
Writes: sensitivity_grid.csv and plots/sensitivity_heatmap.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import cvxpy as cp
from yaml import safe_load
import matplotlib.pyplot as plt

CFG = safe_load(Path("config.yaml").read_text())
PATHS = CFG["paths"]
STAFF = CFG["staffing"]

PRED_CSV = Path(PATHS["pred_csv"])
PLOTS_DIR = Path(PATHS["plots_dir"]); PLOTS_DIR.mkdir(exist_ok=True)

PPN = STAFF["patients_per_nurse"]
NURSE_COST = float(STAFF["nurse_cost"])
SHORTAGE_PENALTY = float(STAFF["shortage_penalty"])

def solve_once(penalty_x=1.0, budget_x=1.0):
    """Return (total_shortage, total_nurses_used) for given multipliers."""
    pred = pd.read_csv(PRED_CSV, parse_dates=["date"])
    total_short = 0.0
    total_nurses = 0.0

    for trust, g in pred.groupby("trust"):
        depts = g["dept"].astype(str).tolist()
        beds  = g["beds"].to_numpy(float)
        dem   = g["pred_patients"].to_numpy(float)
        ppn   = np.array([PPN.get(d, 5) for d in depts], dtype=float)

        max_nurses = np.ceil(beds / ppn)                 # nurses cap by beds
        budget = int(max_nurses.sum() * budget_x)        # scaled budget

        nurses = cp.Variable(len(depts), nonneg=True)
        served = cp.Variable(len(depts), nonneg=True)
        short  = cp.Variable(len(depts), nonneg=True)

        cons = [
            served <= beds,
            served <= cp.multiply(ppn, nurses),
            short  >= dem - served,
            nurses <= max_nurses,
            cp.sum(nurses) <= budget
        ]
        obj = cp.Minimize(NURSE_COST * cp.sum(nurses) + (SHORTAGE_PENALTY * penalty_x) * cp.sum(short))
        prob = cp.Problem(obj, cons)

        for solver in [cp.ECOS, cp.OSQP, cp.SCS]:
            try:
                prob.solve(solver=solver, verbose=False)
                if prob.status in ("optimal", "optimal_inaccurate"):
                    break
            except Exception:
                pass
        if nurses.value is None:
            # infeasible/unresolved → count full demand as shortage
            total_short += float(np.maximum(0.0, dem).sum())
            continue

        total_short  += float(short.value.sum())
        total_nurses += float(nurses.value.sum())

    return total_short, total_nurses

if __name__ == "__main__":
    grid = []
    penalty_grid = [0.5, 1.0, 2.0, 4.0]   # multiply the shortage penalty
    budget_grid  = [0.7, 1.0, 1.3]        # multiply the nurse budget

    for px in penalty_grid:
        for bx in budget_grid:
            tot_short, tot_nurses = solve_once(penalty_x=px, budget_x=bx)
            grid.append({"penalty_x": px, "budget_x": bx,
                         "total_shortage": tot_short, "total_nurses": tot_nurses})

    df = pd.DataFrame(grid)
    out_csv = Path("sensitivity_grid.csv")
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv.resolve())
    print(df)

    piv = df.pivot(index="penalty_x", columns="budget_x", values="total_shortage").sort_index(ascending=True)
    plt.figure(figsize=(6,4))
    plt.imshow(piv.values, aspect="auto")
    plt.xticks(range(len(piv.columns)), piv.columns)
    plt.yticks(range(len(piv.index)), piv.index)
    plt.title("Total shortage vs (penalty_x, budget_x)")
    plt.colorbar(label="patients short")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "sensitivity_heatmap.png", dpi=160)
    plt.show()
