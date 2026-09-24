import os
import pandas as pd
import streamlit as st

from forecast_ledger_v83 import ForecastLedgerV83
from gold_model import price_interval, technical_snapshot
from gold_model_v82 import (
    INTRADAY_HORIZON_LABEL, INTRADAY_INTERVAL, apply_elliott_overlay,
    download_intraday_bundle, fit_intraday_system, validate_market_data,
)
from gold_model_v83 import MODEL_VERSION_V83, decide_v83, fit_v83_system
from macro_econometrics_v825 import combine_macro_sources, download_fred_macro, parse_slow_factor_csv

st.set_page_config(page_title="Gold Version 8.3", page_icon="🟡", layout="wide")
st.title("Gold Version 8.3")
st.caption("Calibrated regime-ensemble · one-hour forecast · market-only research")

with st.sidebar:
    st.header("Version 8.3 settings")
    st.text_input("Candle interval", INTRADAY_INTERVAL, disabled=True)
    st.text_input("Forecast horizon", INTRADAY_HORIZON_LABEL, disabled=True)
    threshold = st.slider("Timing probability threshold", .55, .75, .60, .01)
    cost_bps = st.number_input("Estimated total cost (basis points)", 0, 100, 10, 5)
    splits = st.slider("Walk-forward folds", 4, 8, 5)
    major_event = st.checkbox("Major US event within next hour")
    slow_file = st.file_uploader("Optional release-timestamped macro CSV", type=["csv"])
    run = st.button("Run Version 8.3", type="primary", width="stretch")

if not run:
    st.info("Run after a completed 15-minute candle. Version 8.2.5 remains unchanged on port 8502.")
    st.stop()

try:
    key = str(st.secrets["TWELVE_DATA_API_KEY"])
except (KeyError, FileNotFoundError):
    key = os.getenv("TWELVE_DATA_API_KEY", "")
if not key:
    st.error("Missing TWELVE_DATA_API_KEY in Streamlit secrets.")
    st.stop()

try:
    with st.spinner("Running Version 8.3 calibrated walk-forward research…"):
        gold, confirmations, source_status = download_intraday_bundle(key)
        validate_market_data(gold, confirmations)
        result = fit_v83_system(gold, confirmations, splits, cost_bps, threshold)
        fred, macro_audit = download_fred_macro()
        slow = parse_slow_factor_csv(slow_file) if slow_file else None
        macro = combine_macro_sources(fred, slow)
        elliott = apply_elliott_overlay(result.probability_up, result.median_return,
                                        threshold, cost_bps, gold)
        decision = decide_v83(result, macro, elliott, threshold, cost_bps, major_event)
except Exception as exc:
    st.error(f"Version 8.3 could not run: {exc}")
    st.stop()

lower, median, upper = price_interval(result)
forecast_time = result.as_of + pd.Timedelta(hours=1)
st.subheader("Version 8.3 decision")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Macro regime", decision.macro_regime)
c2.metric("Timing candidate", decision.candidate)
c3.metric("Risk-controlled action", decision.action)
c4.metric("Probability up", f"{elliott.probability_up:.1%}")
c5.metric("Predicted price in 1 hour", f"USD {median:,.2f}")
c6.metric("80% range", f"{lower:,.2f}–{upper:,.2f}")
st.caption(f"Data {result.as_of:%Y-%m-%d %H:%M UTC} | Expiry {forecast_time:%Y-%m-%d %H:%M UTC} | Spot USD {result.spot:,.2f}")
if decision.reasons:
    st.warning("Action withheld: " + "; ".join(decision.reasons) + ".")

st.subheader("Elliott Wave audit")
e1, e2, e3, e4, e5 = st.columns(5)
e1.metric("Current bias", elliott.current_bias)
e2.metric("Current structure", elliott.current_structure)
e3.metric("Probability adjustment", f"{elliott.adjustment:+.1%}")
e4.metric("Holdout accuracy", f"{elliott.accuracy:.1%}")
e5.metric("Reliability gate", "PASS" if elliott.qualified else "FAIL")
st.caption(
    f"Causal confirmed-pivot evidence | Holdout observations: {elliott.observations:,} | "
    f"90% Wilson lower bound: {elliott.lower_bound:.1%}"
)
if elliott.reasons:
    st.info("Elliott evidence not applied: " + "; ".join(elliott.reasons) + ".")
elif elliott.adjustment:
    st.success(f"Qualified Elliott evidence adjusted probability by {elliott.adjustment:+.1%}.")
else:
    st.info("Qualified Elliott evidence is neutral; no probability adjustment was applied.")

ledger = ForecastLedgerV83()
settled = ledger.settle(gold)
ledger.record(
    model_version=MODEL_VERSION_V83, data_timestamp=result.as_of,
    forecast_timestamp=forecast_time, starting_price=result.spot,
    directional_outlook=decision.candidate, decision=decision.action,
    market_probability_up=result.probability_up,
    adjusted_probability_up=elliott.probability_up,
    lower_target=lower, median_target=median, upper_target=upper,
    release_gate="PASS" if decision.action in {"BUY", "SELL"} else decision.action,
    gate_reasons="; ".join(decision.reasons),
)
if settled:
    st.success(f"Settled {settled} previous Version 8.3 forecast(s).")

st.subheader("Non-overlapping cost-aware evaluation")
rows = []
for name, value in decision.evaluation.items():
    if name in {"Win rate", "Net return", "Max drawdown"} and pd.notna(value):
        shown = f"{value:.1%}"
    elif pd.isna(value):
        shown = "N/A"
    elif value == float("inf"):
        shown = "∞"
    else:
        shown = f"{value:.3f}"
    rows.append({"Metric": name, "Result": shown})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.subheader("Accuracy and calibration challenge")
st.dataframe(pd.DataFrame([
    {"Model": "Version 8.3", "ROC-AUC": result.metrics.get("ROC-AUC"),
     "Brier score": result.metrics.get("Brier score")},
    {"Model": "Version 8.2.5 champion",
     "ROC-AUC": result.baseline_metrics.get("Champion ROC-AUC"),
     "Brier score": result.baseline_metrics.get("Champion Brier score")},
]), hide_index=True, width="stretch")
st.caption("Higher ROC-AUC and lower Brier score are better. Version 8.3 abstains unless it beats 8.2.5 on both.")

st.subheader("Version 8.3 forecast ledger")
history = ledger.frame()
st.dataframe(history, hide_index=True, width="stretch")
st.download_button("Download Version 8.3 ledger (CSV)", history.to_csv(index=False).encode(),
                   "gold_v83_forecast_ledger.csv", "text/csv")

st.subheader("Technical indicators and previous-session pivots")
technical = technical_snapshot(gold)
st.dataframe(pd.DataFrame([{"Measure": key, "Value": value} for key, value in technical.items()]),
             hide_index=True, width="stretch")

st.subheader("Macro release audit")
st.dataframe(macro_audit, hide_index=True, width="stretch")
st.caption("Slow factors classify regime at their true release frequency; they do not create synthetic 15-minute releases.")
