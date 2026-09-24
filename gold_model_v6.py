from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

MODEL_VERSION = "6.1-research"
SYMBOL = "XAU/USD"
INTERVAL = "1day"
HORIZON_BARS = 5
TARGET_R = 2.0
STOP_R = 1.0
FRED_SERIES = {
    "real_yield_10y": "DFII10",
    "nominal_yield_10y": "DGS10",
    "broad_dollar": "DTWEXBGS",
    "vix": "VIXCLS",
    "wti": "DCOILWTICO",
}


@dataclass
class DailyResult:
    as_of: pd.Timestamp
    spot: float
    atr: float
    long_probability: float
    short_probability: float
    long_expected_r: float
    signal: str
    regime: str
    metrics: dict[str, float]
    predictions: pd.DataFrame
    feature_names: list[str]
    observations: int
    source_status: dict[str, str]
    release_passed: bool
    release_reasons: list[str]


def _read_url(url: str) -> str:
    request = Request(url, headers={"User-Agent": "GoldResearchLab/6.0"})
    try:
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(f"Data request returned HTTP {exc.code}.") from exc
    except URLError as exc:
        raise RuntimeError(f"Data connection failed: {exc.reason}") from exc


def _twelve_daily(api_key: str, symbol: str, outputsize: int = 5000) -> pd.DataFrame:
    query = urlencode({
        "symbol": symbol, "interval": INTERVAL, "outputsize": outputsize,
        "timezone": "UTC", "format": "JSON", "apikey": api_key,
    })
    payload = json.loads(_read_url(f"https://api.twelvedata.com/time_series?{query}"))
    if payload.get("status") == "error" or "values" not in payload:
        raise RuntimeError(f"Twelve Data {symbol}: {payload.get('message', 'no data returned')}")
    frame = pd.DataFrame(payload["values"])
    required = ["datetime", "open", "high", "low", "close"]
    if any(column not in frame for column in required):
        raise RuntimeError(f"Twelve Data {symbol} response is incomplete.")
    frame["date"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (frame.set_index("date")[["open", "high", "low", "close"]]
            .sort_index().loc[lambda x: ~x.index.duplicated(keep="last")].dropna())


def _fred(series_id: str) -> pd.Series:
    text = _read_url(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}")
    frame = pd.read_csv(StringIO(text))
    date_column = frame.columns[0]
    value_column = frame.columns[1]
    dates = pd.to_datetime(frame[date_column], utc=True, errors="coerce").dt.normalize()
    values = pd.to_numeric(frame[value_column], errors="coerce")
    return pd.Series(values.to_numpy(), index=dates, name=series_id).dropna().sort_index()


def download_daily_bundle(api_key: str) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, str]]:
    if not api_key:
        raise ValueError("TWELVE_DATA_API_KEY is missing.")
    gold = _twelve_daily(api_key, SYMBOL)
    if len(gold) < 1200:
        raise RuntimeError(f"Only {len(gold)} gold days were returned; at least 1,200 are required.")
    sources: dict[str, pd.Series] = {}
    status: dict[str, str] = {SYMBOL: f"{len(gold):,} daily rows"}
    for name, symbol in {"silver": "XAG/USD", "eurusd": "EUR/USD"}.items():
        try:
            frame = _twelve_daily(api_key, symbol)
            sources[name] = frame["close"]
            status[symbol] = f"{len(frame):,} daily rows"
        except Exception as exc:
            status[symbol] = f"unavailable: {exc}"
    for name, series_id in FRED_SERIES.items():
        try:
            series = _fred(series_id)
            sources[name] = series
            status[series_id] = f"{len(series):,} observations"
        except Exception as exc:
            status[series_id] = f"unavailable: {exc}"
    independent = [name for name in sources if name != "eurusd"]
    if len(independent) < 2:
        raise RuntimeError("Fewer than two independent confirmation series are available.")
    return gold, sources, status


def validate_daily_data(gold: pd.DataFrame, now: pd.Timestamp | None = None) -> None:
    if gold.index.has_duplicates or not gold.index.is_monotonic_increasing:
        raise ValueError("Gold dates are duplicated or unsorted.")
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    age_days = (current.normalize() - gold.index[-1]).days
    if age_days > 4:
        raise ValueError(f"Latest daily gold bar is stale ({age_days} calendar days old).")


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = -change.clip(upper=0).ewm(alpha=1 / window, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _atr(gold: pd.DataFrame, window: int = 14) -> pd.Series:
    previous = gold.close.shift(1)
    tr = pd.concat([
        gold.high - gold.low,
        (gold.high - previous).abs(),
        (gold.low - previous).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def make_daily_features(gold: pd.DataFrame, sources: dict[str, pd.Series]) -> pd.DataFrame:
    close = gold.close
    out = pd.DataFrame(index=gold.index)
    for days in (1, 2, 5, 10, 20, 60, 120, 252):
        out[f"gold_return_{days}"] = close.pct_change(days)
    for days in (10, 20, 60, 120, 252):
        out[f"gold_ema_gap_{days}"] = close / close.ewm(span=days, adjust=False).mean() - 1
    daily_return = close.pct_change()
    for days in (10, 20, 60):
        out[f"gold_vol_{days}"] = daily_return.rolling(days).std() * np.sqrt(252)
    out["atr_pct"] = _atr(gold) / close
    out["rsi_14"] = _rsi(close) / 100
    out["range_pct"] = (gold.high - gold.low) / gold.open
    out["body_pct"] = (gold.close - gold.open) / gold.open
    for name, raw in sources.items():
        aligned = raw.reindex(gold.index).ffill(limit=7)
        if name in {"silver", "eurusd", "broad_dollar", "vix", "wti"}:
            for days in (1, 5, 20, 60):
                out[f"{name}_change_{days}"] = aligned.pct_change(days)
        else:
            out[f"{name}_level"] = aligned
            out[f"{name}_change_5"] = aligned.diff(5)
            out[f"{name}_change_20"] = aligned.diff(20)
    if "silver" in sources:
        silver = sources["silver"].reindex(gold.index).ffill(limit=7)
        ratio = close / silver
        out["gold_silver_ratio_gap_60"] = ratio / ratio.rolling(60).mean() - 1
    month = out.index.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    return out.replace([np.inf, -np.inf], np.nan)


def _barrier_targets(gold: pd.DataFrame) -> pd.DataFrame:
    atr = _atr(gold)
    rows = []
    for i in range(len(gold)):
        if i + HORIZON_BARS >= len(gold) or pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0:
            rows.append((np.nan, np.nan, np.nan, np.nan))
            continue
        entry, unit = gold.close.iloc[i], atr.iloc[i]
        long_result = short_result = None
        for j in range(i + 1, i + HORIZON_BARS + 1):
            high, low = gold.high.iloc[j], gold.low.iloc[j]
            if long_result is None:
                if low <= entry - STOP_R * unit:
                    long_result = -STOP_R
                elif high >= entry + TARGET_R * unit:
                    long_result = TARGET_R
            if short_result is None:
                if high >= entry + STOP_R * unit:
                    short_result = -STOP_R
                elif low <= entry - TARGET_R * unit:
                    short_result = TARGET_R
        terminal_r = (gold.close.iloc[i + HORIZON_BARS] - entry) / unit
        long_r = float(np.clip(terminal_r, -STOP_R, TARGET_R)) if long_result is None else long_result
        short_r = float(np.clip(-terminal_r, -STOP_R, TARGET_R)) if short_result is None else short_result
        rows.append((float(long_r >= TARGET_R), float(short_r >= TARGET_R), long_r, short_r))
    return pd.DataFrame(rows, index=gold.index,
                        columns=["long_success", "short_success", "long_r", "short_r"])


def _ensemble(train_x, train_y, test_x, seed: int) -> np.ndarray:
    if pd.Series(train_y).nunique() < 2:
        return np.full(len(test_x), float(pd.Series(train_y).mean()))
    tree = HistGradientBoostingClassifier(
        learning_rate=.035, max_iter=220, max_leaf_nodes=15,
        min_samples_leaf=35, l2_regularization=2.0, random_state=seed)
    # No class weighting: weighted logistic probabilities were severely
    # overconfident in the Version 6.0 out-of-sample calibration audit.
    linear = LogisticRegression(C=.25, max_iter=2000, random_state=seed)
    tree.fit(train_x, train_y)
    linear.fit(train_x, train_y)
    return .70 * tree.predict_proba(test_x)[:, 1] + .30 * linear.predict_proba(test_x)[:, 1]


def _return_ensemble(train_x, train_y, test_x, seed: int) -> np.ndarray:
    nonlinear = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=.035, max_iter=220,
        max_leaf_nodes=15, min_samples_leaf=35, l2_regularization=3.0,
        random_state=seed)
    linear = Ridge(alpha=20.0)
    nonlinear.fit(train_x, train_y)
    linear.fit(train_x, train_y)
    return .70 * nonlinear.predict(test_x) + .30 * linear.predict(test_x)


def _decision(long_p: float, long_expected_r: float,
              threshold: float, minimum_expected_r: float) -> str:
    # Version 6.1 is deliberately long-only. The attached Version 6.0 audit
    # showed short ROC-AUC below random and negative short expectancy.
    if long_p >= threshold and long_expected_r >= minimum_expected_r:
        return "BUY"
    return "WAIT"


def _safe_auc(actual: pd.Series, probability: pd.Series) -> float:
    return float(roc_auc_score(actual, probability)) if actual.nunique() > 1 else float("nan")


def fit_daily_system(
    gold: pd.DataFrame,
    sources: dict[str, pd.Series],
    source_status: dict[str, str],
    splits: int = 6,
    threshold: float = .30,
    minimum_expected_r: float = .15,
    cost_bps: float = 15,
    risk_fraction: float = .0025,
) -> DailyResult:
    features = make_daily_features(gold, sources)
    targets = _barrier_targets(gold)
    labelled = features.join(targets).dropna()
    if len(labelled) < 900:
        raise ValueError(f"Only {len(labelled)} complete daily rows; at least 900 are required.")
    feature_names = list(features.columns)
    X = labelled[feature_names]
    cv = TimeSeriesSplit(n_splits=splits, gap=HORIZON_BARS)
    rows = []
    for fold, (train_index, test_index) in enumerate(cv.split(X), 1):
        scaler = StandardScaler()
        train_x = scaler.fit_transform(X.iloc[train_index])
        test_x = scaler.transform(X.iloc[test_index])
        row = labelled.iloc[test_index][["long_success", "short_success", "long_r", "short_r"]].copy()
        row["long_probability"] = _ensemble(
            train_x, labelled.long_success.iloc[train_index].astype(int), test_x, 100 + fold)
        row["short_probability"] = _ensemble(
            train_x, labelled.short_success.iloc[train_index].astype(int), test_x, 200 + fold)
        row["long_expected_r"] = _return_ensemble(
            train_x, labelled.long_r.iloc[train_index], test_x, 300 + fold)
        row["fold"] = fold
        rows.append(row)
    predictions = pd.concat(rows).sort_index()
    predictions["signal"] = [
        _decision(lp, er, threshold, minimum_expected_r)
        for lp, er in zip(predictions.long_probability, predictions.long_expected_r)]
    non_overlap = predictions.iloc[::HORIZON_BARS].copy()
    non_overlap["gross_r"] = np.select(
        [non_overlap.signal == "BUY", non_overlap.signal == "SELL"],
        [non_overlap.long_r, non_overlap.short_r], default=0.0)
    atr_pct = features.atr_pct.reindex(non_overlap.index)
    cost_r = (cost_bps / 10_000) / atr_pct.replace(0, np.nan)
    non_overlap["net_r"] = np.where(non_overlap.signal == "WAIT", 0.0,
                                     non_overlap.gross_r - cost_r)
    non_overlap["account_return"] = non_overlap.net_r * risk_fraction
    non_overlap["equity"] = (1 + non_overlap.account_return).cumprod()
    predictions["equity"] = non_overlap.equity.reindex(predictions.index).ffill()
    trades = non_overlap[non_overlap.signal != "WAIT"].copy()
    wins = trades.loc[trades.net_r > 0, "net_r"].sum()
    losses = -trades.loc[trades.net_r < 0, "net_r"].sum()
    profit_factor = float(wins / losses) if losses > 0 else float("inf")
    peak = non_overlap.equity.cummax()
    expectancy = float(trades.net_r.mean()) if len(trades) else float("nan")
    standard_error = float(trades.net_r.std(ddof=1) / np.sqrt(len(trades))) if len(trades) > 1 else float("inf")
    expectancy_lcb = expectancy - 1.645 * standard_error
    metrics = {
        "Long ROC-AUC": _safe_auc(predictions.long_success, predictions.long_probability),
        "Short ROC-AUC": _safe_auc(predictions.short_success, predictions.short_probability),
        "Long Brier": float(brier_score_loss(predictions.long_success, predictions.long_probability)),
        "Short Brier": float(brier_score_loss(predictions.short_success, predictions.short_probability)),
        "OOS trades": float(len(trades)),
        "Trade coverage": float(len(trades) / len(non_overlap)),
        "Win rate": float((trades.net_r > 0).mean()) if len(trades) else float("nan"),
        "Expectancy R": expectancy,
        "90% expectancy lower bound R": expectancy_lcb,
        "Profit factor": profit_factor,
        "Strategy return": float(non_overlap.equity.iloc[-1] - 1),
        "Maximum drawdown": float((non_overlap.equity / peak - 1).min()),
    }
    release_reasons = []
    if len(trades) < 50:
        release_reasons.append("fewer than 50 non-overlapping out-of-sample trades")
    if not np.isfinite(profit_factor) or profit_factor < 1.20:
        release_reasons.append("profit factor is below 1.20 or undefined")
    if not np.isfinite(expectancy_lcb) or expectancy_lcb <= 0:
        release_reasons.append("90% lower confidence bound for expectancy is not positive")
    if metrics["Maximum drawdown"] < -.10:
        release_reasons.append("maximum drawdown exceeds 10%")
    if metrics["Long ROC-AUC"] < .53:
        release_reasons.append("long barrier classifier ROC-AUC is below 0.53")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    latest_features = features.dropna().iloc[[-1]]
    latest_scaled = scaler.transform(latest_features)
    long_probability = float(_ensemble(
        scaled, labelled.long_success.astype(int), latest_scaled, 901)[0])
    short_probability = float(_ensemble(
        scaled, labelled.short_success.astype(int), latest_scaled, 902)[0])
    long_expected_r = float(_return_ensemble(
        scaled, labelled.long_r, latest_scaled, 903)[0])
    signal = _decision(long_probability, long_expected_r, threshold, minimum_expected_r)
    if release_reasons:
        signal = "WAIT"
    latest = latest_features.iloc[0]
    high_vol = latest.gold_vol_20 >= features.gold_vol_20.dropna().tail(500).quantile(.75)
    trend = latest.gold_ema_gap_60
    regime = "HIGH VOLATILITY" if high_vol else (
        "UPTREND" if trend > .03 else "DOWNTREND" if trend < -.03 else "RANGE")
    as_of = latest_features.index[-1]
    return DailyResult(
        as_of=as_of, spot=float(gold.close.loc[as_of]), atr=float(_atr(gold).loc[as_of]),
        long_probability=long_probability, short_probability=short_probability,
        long_expected_r=long_expected_r,
        signal=signal, regime=regime, metrics=metrics, predictions=predictions,
        feature_names=feature_names, observations=len(gold), source_status=source_status,
        release_passed=not release_reasons, release_reasons=release_reasons)
