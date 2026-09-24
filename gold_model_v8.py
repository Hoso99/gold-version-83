from __future__ import annotations

import numpy as np
import pandas as pd

from gold_model import (
    CONFIRMATION_SYMBOLS,
    HORIZON_BARS as INTRADAY_HORIZON_BARS,
    HORIZON_LABEL as INTRADAY_HORIZON_LABEL,
    INTERVAL as INTRADAY_INTERVAL,
    _download_symbol,
    download_market_data,
    expected_market_open,
    fit_and_backtest,
    price_interval,
    validate_market_data,
)
from gold_model_v6 import _fred, _twelve_daily, download_daily_bundle, validate_daily_data
from gold_model_v621 import fit_research_system_v621

MODEL_VERSION = "8.1-multihorizon-research"

EXTRA_FRED = {
    "breakeven_10y": "T10YIE",
    "fed_funds": "DFF",
    "credit_spread_hy": "BAMLH0A0HYM2",
    "financial_conditions": "NFCI",
    "fed_balance_sheet": "WALCL",
}

MARKET_PROXIES = {
    "etf_gld": "GLD",
    "etf_slv": "SLV",
    "etf_tlt": "TLT",
    "etf_uup": "UUP",
}


def _stationary_proxy(close: pd.Series) -> pd.Series:
    """Convert an ETF price level into a rolling log-price z-score."""
    logged = np.log(close.where(close > 0))
    mean = logged.rolling(252, min_periods=126).mean()
    deviation = logged.rolling(252, min_periods=126).std()
    return ((logged - mean) / deviation.replace(0, np.nan)).dropna()


def download_enhanced_bundle(
    api_key: str,
) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, str]]:
    gold, sources, status = download_daily_bundle(api_key)
    for name, series_id in EXTRA_FRED.items():
        try:
            series = _fred(series_id)
            sources[name] = series
            status[series_id] = f"{len(series):,} observations"
        except Exception as exc:
            status[series_id] = f"unavailable: {exc}"
    for name, symbol in MARKET_PROXIES.items():
        try:
            frame = _twelve_daily(api_key, symbol)
            proxy = _stationary_proxy(frame.close)
            if len(proxy) < 750:
                raise RuntimeError(f"only {len(proxy):,} usable observations")
            sources[name] = proxy
            status[symbol] = f"{len(frame):,} daily rows; {len(proxy):,} normalized observations"
        except Exception as exc:
            status[symbol] = f"unavailable: {exc}"
    return gold, sources, status


def fit_enhanced_system(
    gold: pd.DataFrame,
    sources: dict[str, pd.Series],
    source_status: dict[str, str],
    splits: int = 5,
    cost_bps: float = 15,
    risk_fraction: float = 0.0025,
):
    return fit_research_system_v621(
        gold, sources, source_status, splits=splits,
        cost_bps=cost_bps, risk_fraction=risk_fraction,
    )


def download_intraday_bundle(
    api_key: str, outputsize: int = 5000,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, str]]:
    gold = download_market_data(api_key, outputsize)
    if not expected_market_open(gold.index[-1]):
        raise ValueError(
            f"Latest XAU/USD candle is {gold.index[-1]:%A %Y-%m-%d %H:%M UTC}, "
            "outside the accepted market session.")
    confirmations: dict[str, pd.DataFrame] = {}
    status = {"XAU/USD": f"{len(gold):,} 15-minute candles"}
    for name, symbol in CONFIRMATION_SYMBOLS.items():
        try:
            frame = _download_symbol(api_key, symbol, outputsize)
            confirmations[name] = frame
            status[symbol] = f"{len(frame):,} 15-minute candles"
        except Exception as exc:
            status[symbol] = f"unavailable: {exc}"
    return gold, confirmations, status


def fit_intraday_system(
    gold: pd.DataFrame,
    confirmations: dict[str, pd.DataFrame],
    splits: int = 5,
    cost_bps: float = 10,
    threshold: float = 0.65,
):
    return fit_and_backtest(
        gold, splits=splits, cost_bps=cost_bps,
        threshold=threshold, confirmations=confirmations,
    )


def intraday_release_reasons(result) -> list[str]:
    reasons: list[str] = []
    metrics = result.metrics
    baseline = result.baseline_metrics
    if metrics.get("ROC-AUC", 0) < 0.53:
        reasons.append("four-hour ROC-AUC is below 0.53")
    if metrics.get("ROC-AUC", 0) < baseline.get("ROC-AUC", 0) + 0.01:
        reasons.append("confirmation features do not improve ROC-AUC by at least 0.01")
    if metrics.get("Strategy total return", 0) <= 0:
        reasons.append("four-hour walk-forward strategy return is not positive")
    if metrics.get("Strategy max drawdown", -1) < -0.10:
        reasons.append("four-hour walk-forward drawdown exceeds 10%")
    coverage = metrics.get("80% interval coverage", 0)
    if not 0.70 <= coverage <= 0.90:
        reasons.append("four-hour interval coverage is outside 70% to 90%")
    if metrics.get("Signal changes", 0) < 20:
        reasons.append("fewer than 20 non-overlapping signal changes")
    return reasons
