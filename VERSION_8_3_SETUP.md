# Gold Version 8.3 — Calibrated Regime-Ensemble Challenger

Version 8.3 is separate from Version 8.2.5. It does not replace `app_v82.py` or `forecast_ledger_v825.sqlite` and contains no broker execution.

## Design

- Slow, release-timestamped macro variables classify the regime as BULLISH, BEARISH or NEUTRAL.
- Three diverse models form a calibrated next-hour ensemble.
- Every fold has a purged fit window, a separate calibration window and an untouched test window.
- Causal regime, normalized cross-asset shock and market-session features time the candidate.
- Expected movement must exceed twice the configured trading cost.
- Evaluation uses every fourth 15-minute observation so one-hour outcomes do not overlap.
- Version 8.3 must beat Version 8.2.5 on ROC-AUC and Brier score or report NO EDGE.
- Elliott is an independently gated confirmation and cannot rescue failed statistical evidence.
- Output is BUY, SELL, NO EDGE or BLOCKED.
- Version 8.3 writes only to `forecast_ledger_v83.sqlite`.

## Run

Copy all packaged files into the existing `gold_forecaster` folder, activate the environment, and run:

```powershell
python -m pip install -r requirements.txt
python -m py_compile app_v83.py gold_model_v83.py forecast_ledger_v83.py
python -m streamlit run app_v83.py --server.port 8503
```

Open `http://localhost:8503`. Version 8.2.5 may remain available separately on port 8502.
