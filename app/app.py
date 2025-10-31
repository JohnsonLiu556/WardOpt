import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Healthcare Resource Allocation", layout="wide")
st.title("Healthcare Resource Allocation — Dashboard")

FILES = {
    "Naive vs Optimal (base)": "optimal_allocation.csv",
    "Policy (costs + min service)": "optimal_allocation_policy.csv",
    "Overflow (temp beds)": "optimal_allocation_overflow.csv",
    "Integer-like (rounded)": "optimal_allocation_mip.csv",
}

available = {k: Path(v) for k,v in FILES.items() if Path(v).exists()}
if not available:
    st.warning("No outputs found. Run optimize.py / optimize_policy.py / optimize_overflow.py first.")
    st.stop()

dfs = {}
trusts = set()
for label, path in available.items():
    df = pd.read_csv(path)
    df["source"] = label
    dfs[label] = df
    trusts |= set(df["trust"].unique())

trust = st.selectbox("Select Trust", sorted(trusts))

tabs = st.tabs(list(available.keys()))
for tab, (label, df) in zip(tabs, available.items()):
    with tab:
        d = dfs[label]
        d = d[d["trust"] == trust].copy()
        if d.empty:
            st.info(f"No rows for {trust} in {label}.")
            continue

        # show metrics
        tot = d[["pred_patients","served","shortage"]].sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Patients", f"{int(tot['pred_patients']):,}")
        col2.metric("Served", f"{int(tot['served']):,}")
        col3.metric("Shortage", f"{int(tot['shortage']):,}")

        # table
        show_cols = [c for c in ["dept","beds","extra_beds","nurses_alloc","pred_patients","served","shortage"] if c in d.columns]
        st.dataframe(d[show_cols].sort_values("dept"), use_container_width=True)

        # bar chart
        chart_df = d.set_index("dept")[["pred_patients","served","shortage"]].sort_index()
        st.bar_chart(chart_df)

        # download
        st.download_button(
            label=f"Download {label} CSV",
            data=d.to_csv(index=False).encode("utf-8"),
            file_name=f"{Path(FILES[label]).stem}_{trust}.csv",
            mime="text/csv"
        )

# Overall comparison across sources (not just one trust)
st.markdown("---")
st.subheader("Overall comparison across sources")
overall = []
for label, df in dfs.items():
    agg = df.groupby("dept", as_index=False)[["pred_patients","served","shortage"]].sum()
    agg["source"] = label
    overall.append(agg)
overall = pd.concat(overall, ignore_index=True)

st.dataframe(overall, use_container_width=True)
