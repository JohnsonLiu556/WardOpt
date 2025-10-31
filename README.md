# WardOpt
An end to end pipeline that predicts the number of patients each month while optimizing efficient use of nurses and beds for each trust and department. Includes integer-like staffing, policy minimum service levels, overflow beds, sensitivity analysis, and a Streamlit dashboard. Shortage has gone down by about 85% compared to naive on recent data.

<p align="center">
  <img src="plots/naive_vs_optimal_totals.png" width="520" alt="Total shortage: naive vs optimal">
</p>

**Figure 1. Sensitivity heatmap (shortage vs penalty×budget).**  
![Sensitivity](plots/sensitivity_heatmap.png)

**Figure 2. Top-10 trusts by shortage reduction (patients).**  
![Top-10 trusts](plots/shortage_by_trust_top10.png)


---

## Features
- **Forecasting** (lag-1, MA(3), Random Forest; auto-selects best per series)
- **Convex allocation** with clinical constraints (beds, patients-per-nurse)
- **Policy levers**: department-specific costs & minimum service levels
- **Overflow units**: temporary beds with a cost and per-dept caps
- **Integer-like staffing**: continuous solve → rounding → recompute
- **Sensitivity analysis**: shortage vs budget × penalty
- **Dashboard**: compare Base / Policy / Overflow / Integer plans / export CSVs
