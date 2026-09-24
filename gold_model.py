from __future__ import annotations

from dataclasses import dataclass
import json
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

SYMBOL = "XAU/USD"
CONFIRMATION_SYMBOLS = {"silver": "XAG/USD", "eurusd": "EUR/USD"}
INTERVAL = "15min"
HORIZON_BARS = 16
HORIZON_LABEL = "4 hours (16 × 15-minute bars)"
MODEL_VERSION = "5.0"


@dataclass
class ForecastResult:
    as_of: pd.Timestamp
    spot: float
    probability_up: float
    median_return: float
    lower_return: float
    upper_return: float
    signal: str
    regime: str
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    predictions: pd.DataFrame
    importance: pd.Series
    features_used: list[str]
    observations: int


def _download_symbol(api_key: str, symbol: str, outputsize: int) -> pd.DataFrame:
    query = urlencode({
        "symbol": symbol, "interval": INTERVAL, "outputsize": outputsize,
        "timezone": "UTC", "format": "JSON", "apikey": api_key,
    })
    try:
        with urlopen(f"https://api.twelvedata.com/time_series?{query}", timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(
            f"Twelve Data request for {symbol} returned HTTP {exc.code}. "
            "Check whether this symbol is included in your API plan.") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not connect to Twelve Data for {symbol}: {exc.reason}") from exc
    if payload.get("status") == "error" or "values" not in payload:
        message = payload.get("message", "No candle data returned.")
        raise RuntimeError(f"Twelve Data error for {symbol}: {message}")
    frame = pd.DataFrame(payload["values"])
    required = ["datetime", "open", "high", "low", "close"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise RuntimeError(f"{symbol} response is missing: {', '.join(missing)}")
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.set_index("datetime").sort_index()[["open", "high", "low", "close"]]
    return frame[~frame.index.duplicated(keep="last")].dropna()


def download_market_data(api_key: str, outputsize: int = 5000) -> pd.DataFrame:
    """Backward-compatible XAU/USD downloader."""
    if not api_key:
        raise ValueError("TWELVE_DATA_API_KEY is missing from Streamlit secrets.")
    frame = _download_symbol(api_key, SYMBOL, outputsize)
    if len(frame) < 1200:
        raise RuntimeError(
            f"Only {len(frame)} complete XAU/USD candles were returned; at least 1,200 are required.")
    return frame


def download_market_bundle(api_key: str, outputsize: int = 5000) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Download gold plus independent silver and dollar confirmation series."""
    gold = download_market_data(api_key, outputsize)
    # Reject a closed gold session before spending API credits on confirmations.
    if not expected_market_open(gold.index[-1]):
        raise ValueError(
            f"Latest XAU/USD candle is {gold.index[-1]:%A %Y-%m-%d %H:%M UTC}, "
            "outside the accepted market session.")
    confirmations = {
        name: _download_symbol(api_key, symbol, outputsize)
        for name, symbol in CONFIRMATION_SYMBOLS.items()
    }
    return gold, confirmations


def expected_market_open(timestamp: pd.Timestamp) -> bool:
    """Conservative UTC spot-metals weekend gate."""
    ts = pd.Timestamp(timestamp)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    weekday, hour = ts.weekday(), ts.hour
    if weekday == 5:
        return False
    if weekday == 6 and hour < 22:
        return False
    if weekday == 4 and hour >= 22:
        return False
    return True


def validate_market_data(
    gold: pd.DataFrame,
    confirmations: dict[str, pd.DataFrame],
    now: pd.Timestamp | None = None,
) -> dict[str, float | str]:
    if gold.empty:
        raise ValueError("No XAU/USD candles were returned.")
    as_of = gold.index[-1]
    if not expected_market_open(as_of):
        raise ValueError(
            f"Latest XAU/USD candle is {as_of:%A %Y-%m-%d %H:%M UTC}, outside the accepted market session.")
    recent = gold.index[-20:]
    if recent.has_duplicates:
        raise ValueError("Duplicate XAU/USD timestamps were detected.")
    gaps = pd.Series(recent).diff().dropna().dt.total_seconds().div(60)
    if len(gaps) and (gaps > 30).any():
        raise ValueError("Recent XAU/USD candles contain a gap longer than 30 minutes.")
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    age_minutes = max(0.0, (current - as_of).total_seconds() / 60)
    if age_minutes > 45:
        raise ValueError(f"Latest XAU/USD candle is stale ({age_minutes:.0f} minutes old).")
    report: dict[str, float | str] = {"XAU/USD": as_of.strftime("%Y-%m-%d %H:%M UTC")}
    for name, frame in confirmations.items():
        if frame.empty:
            raise ValueError(f"No {CONFIRMATION_SYMBOLS[name]} candles were returned.")
        confirmation_age = abs((as_of - frame.index[-1]).total_seconds()) / 60
        if confirmation_age > 60:
            raise ValueError(
                f"{CONFIRMATION_SYMBOLS[name]} is not aligned with gold ({confirmation_age:.0f}-minute difference).")
        report[CONFIRMATION_SYMBOLS[name]] = frame.index[-1].strftime("%Y-%m-%d %H:%M UTC")
    report["Data age minutes"] = age_minutes
    return report


def _rsi(price: pd.Series, window: int = 14) -> pd.Series:
    change = price.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = -change.clip(upper=0).ewm(alpha=1 / window, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Causal technical indicators and prior-session classic pivot levels."""
    close, high, low = prices.close, prices.high, prices.low
    out = pd.DataFrame(index=prices.index)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_pct"] = macd / close
    out["macd_signal_pct"] = macd_signal / close
    out["macd_hist_pct"] = (macd - macd_signal) / close

    middle = close.rolling(20).mean()
    deviation = close.rolling(20).std()
    upper, lower = middle + 2 * deviation, middle - 2 * deviation
    out["bollinger_width_20"] = (upper - lower) / middle
    out["bollinger_position_20"] = (close - lower) / (upper - lower).replace(0, np.nan)

    low_14, high_14 = low.rolling(14).min(), high.rolling(14).max()
    stochastic_k = 100 * (close - low_14) / (high_14 - low_14).replace(0, np.nan)
    out["stochastic_k_14"] = stochastic_k / 100
    out["stochastic_d_3"] = stochastic_k.rolling(3).mean() / 100

    previous_close = close.shift(1)
    true_range = pd.concat([
        high - low, (high - previous_close).abs(), (low - previous_close).abs()
    ], axis=1).max(axis=1)
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out["adx_14"] = dx.ewm(alpha=1 / 14, adjust=False).mean() / 100
    out["di_spread_14"] = (plus_di - minus_di) / 100

    sessions = prices.index.floor("D")
    daily = prices.assign(_session=sessions).groupby("_session").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"))
    previous = daily.shift(1)
    pivot = (previous.high + previous.low + previous.close) / 3
    levels = pd.DataFrame(index=daily.index)
    levels["pivot"] = pivot
    levels["r1"] = 2 * pivot - previous.low
    levels["s1"] = 2 * pivot - previous.high
    levels["r2"] = pivot + previous.high - previous.low
    levels["s2"] = pivot - previous.high + previous.low
    aligned = levels.reindex(sessions)
    aligned.index = prices.index
    for name in ("pivot", "r1", "s1", "r2", "s2"):
        out[f"distance_{name}"] = close / aligned[name] - 1
    out["pivot_position_s1_r1"] = (
        (close - aligned.s1) / (aligned.r1 - aligned.s1).replace(0, np.nan))
    return out.replace([np.inf, -np.inf], np.nan)


def technical_snapshot(prices: pd.DataFrame) -> dict[str, float]:
    """Latest display values; pivot prices come only from the prior completed session."""
    features = _technical_features(prices)
    row = features.iloc[-1]
    close = float(prices.close.iloc[-1])
    return {
        "RSI 14": float(_rsi(prices.close).iloc[-1]),
        "MACD histogram (%)": float(row["macd_hist_pct"] * 100),
        "Bollinger position": float(row["bollinger_position_20"]),
        "Stochastic %K": float(row["stochastic_k_14"] * 100),
        "ADX 14": float(row["adx_14"] * 100),
        "Pivot": float(close / (1 + row["distance_pivot"])),
        "Resistance 1": float(close / (1 + row["distance_r1"])),
        "Resistance 2": float(close / (1 + row["distance_r2"])),
        "Support 1": float(close / (1 + row["distance_s1"])),
        "Support 2": float(close / (1 + row["distance_s2"])),
    }


def _base_features(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices["close"]
    out = pd.DataFrame(index=prices.index)
    returns = close.pct_change()
    for bars in (1, 2, 4, 8, 16, 32, 64, 96):
        out[f"return_{bars}"] = close.pct_change(bars)
    for bars in (8, 16, 32, 64, 96):
        out[f"ema_gap_{bars}"] = close / close.ewm(span=bars, adjust=False).mean() - 1
    for bars in (8, 16, 32, 64):
        out[f"volatility_{bars}"] = returns.rolling(bars).std() * np.sqrt(bars)
    previous_close = close.shift(1)
    true_range = pd.concat([
        prices.high - prices.low,
        (prices.high - previous_close).abs(),
        (prices.low - previous_close).abs(),
    ], axis=1).max(axis=1)
    out["atr_14_pct"] = true_range.rolling(14).mean() / close
    out["rsi_14"] = _rsi(close) / 100
    out["candle_body"] = (prices.close - prices.open) / prices.open
    out["range_pct"] = (prices.high - prices.low) / prices.open
    out["range_position_32"] = (
        (close - prices.low.rolling(32).min()) /
        (prices.high.rolling(32).max() - prices.low.rolling(32).min()))
    out["hour_sin"] = np.sin(2 * np.pi * out.index.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out.index.hour / 24)
    out["weekday"] = out.index.dayofweek / 4
    out = out.join(_technical_features(prices))
    return out.replace([np.inf, -np.inf], np.nan)


def make_features(
    prices: pd.DataFrame,
    confirmations: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    out = _base_features(prices)
    if not confirmations:
        return out
    aligned: dict[str, pd.Series] = {}
    for name, frame in confirmations.items():
        aligned[name] = frame["close"].reindex(prices.index, method="ffill", limit=4)
        for bars in (1, 4, 16, 64):
            out[f"{name}_return_{bars}"] = aligned[name].pct_change(bars)
    if "silver" in aligned:
        ratio = prices.close / aligned["silver"]
        out["gold_silver_ratio_gap_64"] = ratio / ratio.rolling(64).mean() - 1
        out["gold_silver_corr_64"] = prices.close.pct_change().rolling(64).corr(
            aligned["silver"].pct_change())
    if "eurusd" in aligned:
        out["gold_eurusd_corr_64"] = prices.close.pct_change().rolling(64).corr(
            aligned["eurusd"].pct_change())
    return out.replace([np.inf, -np.inf], np.nan)


def _targets(
    prices: pd.DataFrame,
    horizon_bars: int = HORIZON_BARS,
) -> tuple[pd.Series, pd.Series]:
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be at least 1")
    future_return = prices.close.shift(-horizon_bars) / prices.close - 1
    direction = (future_return > 0).astype(float)
    direction[future_return.isna()] = np.nan
    return direction, future_return


def _models(seed: int):
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.035, max_iter=160, max_leaf_nodes=15,
        min_samples_leaf=30, l2_regularization=1.5, random_state=seed)
    regressors = {
        name: GradientBoostingRegressor(
            loss="quantile", alpha=alpha, n_estimators=130, max_depth=2,
            learning_rate=0.035, min_samples_leaf=25, random_state=seed)
        for name, alpha in (("lower", 0.10), ("median", 0.50), ("upper", 0.90))
    }
    return classifier, regressors


def _signal(probability: float, expected_return: float, cost_bps: float, threshold: float) -> str:
    cost = cost_bps / 10_000
    if probability >= threshold and expected_return > cost:
        return "BUY"
    if probability <= 1 - threshold and expected_return < -cost:
        return "SELL"
    return "WAIT"


def _classification_metrics(actual: pd.Series, probability: pd.Series) -> dict[str, float]:
    return {
        "ROC-AUC": roc_auc_score(actual, probability),
        "Brier score": brier_score_loss(actual, probability),
        "Direction accuracy": accuracy_score(actual, probability >= 0.5),
        "Always-up accuracy": float(actual.mean()),
    }


def _latest_regime(features: pd.DataFrame) -> str:
    clean = features.dropna()
    row = clean.iloc[-1]
    high_vol = row["volatility_32"] >= clean["volatility_32"].tail(500).quantile(0.75)
    trend = row["ema_gap_64"]
    if high_vol:
        return "HIGH VOLATILITY"
    if trend > 0.002:
        return "UPTREND"
    if trend < -0.002:
        return "DOWNTREND"
    return "RANGE"


def fit_and_backtest(
    prices: pd.DataFrame,
    splits: int = 5,
    cost_bps: float = 10,
    threshold: float = 0.65,
    confirmations: dict[str, pd.DataFrame] | None = None,
    horizon_bars: int = HORIZON_BARS,
) -> ForecastResult:
    base_features = _base_features(prices)
    features = make_features(prices, confirmations)
    y_cls, y_ret = _targets(prices, horizon_bars=horizon_bars)
    labelled = features.join(y_cls.rename("direction")).join(
        y_ret.rename("forward_return")).dropna()
    if len(labelled) < 1000:
        raise ValueError(f"Only {len(labelled)} labelled rows; at least 1,000 are required.")
    X = labelled[features.columns]
    X_base = base_features.reindex(X.index).dropna()
    common_index = X.index.intersection(X_base.index)
    X, X_base = X.loc[common_index], X_base.loc[common_index]
    y = labelled.loc[common_index, "direction"].astype(int)
    returns = labelled.loc[common_index, "forward_return"]
    effective_splits = min(splits, max(3, len(X) // 500))
    cv = TimeSeriesSplit(n_splits=effective_splits, gap=horizon_bars)
    rows, base_rows = [], []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        scaler = StandardScaler()
        train = scaler.fit_transform(X.iloc[train_idx])
        test = scaler.transform(X.iloc[test_idx])
        classifier, regressors = _models(40 + fold)
        classifier.fit(train, y.iloc[train_idx])
        pred = pd.DataFrame(index=X.iloc[test_idx].index)
        pred["actual_return"] = returns.iloc[test_idx]
        pred["actual_up"] = y.iloc[test_idx]
        pred["probability_up"] = classifier.predict_proba(test)[:, 1]
        for name, model in regressors.items():
            model.fit(train, returns.iloc[train_idx])
            pred[name] = model.predict(test)
        pred["fold"] = fold
        rows.append(pred)

        base_scaler = StandardScaler()
        base_train = base_scaler.fit_transform(X_base.iloc[train_idx])
        base_test = base_scaler.transform(X_base.iloc[test_idx])
        base_classifier, _ = _models(140 + fold)
        base_classifier.fit(base_train, y.iloc[train_idx])
        base_rows.append(pd.DataFrame({
            "actual_up": y.iloc[test_idx],
            "probability_up": base_classifier.predict_proba(base_test)[:, 1],
        }, index=X_base.iloc[test_idx].index))

    predictions = pd.concat(rows).sort_index()
    base_predictions = pd.concat(base_rows).sort_index()
    predictions["signal"] = [
        _signal(p, m, cost_bps, threshold)
        for p, m in zip(predictions.probability_up, predictions["median"])]
    predictions["position"] = predictions.signal.map({"BUY": 1, "SELL": -1, "WAIT": 0})
    events = predictions.iloc[::horizon_bars].copy()
    turnover = events.position.diff().abs().fillna(events.position.abs())
    events["strategy_return"] = events.position * events.actual_return - turnover * cost_bps / 10_000
    events["strategy_equity"] = (1 + events.strategy_return).cumprod()
    events["gold_equity"] = (1 + events.actual_return).cumprod()
    predictions["strategy_equity"] = events.strategy_equity.reindex(predictions.index).ffill()
    predictions["gold_equity"] = events.gold_equity.reindex(predictions.index).ffill()
    peak = events.strategy_equity.cummax()
    metrics = _classification_metrics(predictions.actual_up, predictions.probability_up)
    metrics.update({
        "80% interval coverage": ((predictions.actual_return >= predictions.lower) &
                                  (predictions.actual_return <= predictions.upper)).mean(),
        "Strategy total return": events.strategy_equity.iloc[-1] - 1,
        "Strategy max drawdown": (events.strategy_equity / peak - 1).min(),
        "Signal changes": float((events.position.diff().fillna(events.position) != 0).sum()),
    })
    baseline_metrics = _classification_metrics(
        base_predictions.actual_up, base_predictions.probability_up)

    full_features = features.dropna()
    latest_x = full_features.iloc[[-1]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    latest_scaled = scaler.transform(latest_x)
    classifier, regressors = _models(99)
    classifier.fit(scaled, y)
    probability = float(classifier.predict_proba(latest_scaled)[0, 1])
    forecast = {}
    for name, model in regressors.items():
        model.fit(scaled, returns)
        forecast[name] = float(model.predict(latest_scaled)[0])
    sample = min(500, len(X))
    perm = permutation_importance(
        classifier, scaled[-sample:], y.iloc[-sample:], n_repeats=3,
        random_state=42, scoring="neg_brier_score")
    importance = pd.Series(perm.importances_mean, index=X.columns).nlargest(15)
    as_of = latest_x.index[0]
    spot = float(prices.close.loc[:as_of].iloc[-1])
    return ForecastResult(
        as_of=as_of, spot=spot, probability_up=probability,
        median_return=forecast["median"], lower_return=forecast["lower"],
        upper_return=forecast["upper"],
        signal=_signal(probability, forecast["median"], cost_bps, threshold),
        regime=_latest_regime(features), metrics=metrics,
        baseline_metrics=baseline_metrics, predictions=predictions,
        importance=importance, features_used=list(X.columns), observations=len(prices))


def price_interval(result: ForecastResult) -> tuple[float, float, float]:
    return tuple(result.spot * (1 + value) for value in
                 (result.lower_return, result.median_return, result.upper_return))
