from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from gold_model import ForecastResult, _models, _signal, make_features, permutation_importance
from gold_model_v82 import INTRADAY_HORIZON_BARS, intraday_release_reasons

MODEL_VERSION_V83 = "8.3-calibrated-regime-ensemble-one-hour-research"


@dataclass
class V83Decision:
    action: str
    candidate: str
    macro_regime: str
    expected_move: float
    minimum_move: float
    reasons: list[str]
    evaluation: dict[str, float]


def _safe_auc(y, p):
    return float(roc_auc_score(y, p)) if pd.Series(y).nunique() > 1 else 0.5


def _v83_features(gold, confirmations):
    """Causal features available at the forecast timestamp."""
    out = make_features(gold, confirmations).copy()
    returns = gold.close.pct_change()
    for short, long in ((4, 32), (8, 64), (16, 96)):
        sv, lv = returns.rolling(short).std(), returns.rolling(long).std()
        out[f"vol_ratio_{short}_{long}"] = sv / lv.replace(0, np.nan)
        out[f"trend_vol_{short}_{long}"] = (
            gold.close.pct_change(short) / (lv * math.sqrt(short)).replace(0, np.nan))
    minute = out.index.hour * 60 + out.index.minute
    out["tod_sin"] = np.sin(2 * np.pi * minute / 1440)
    out["tod_cos"] = np.cos(2 * np.pi * minute / 1440)
    out["london_ny_overlap"] = ((out.index.hour >= 13) & (out.index.hour < 17)).astype(float)
    for name, frame in confirmations.items():
        aligned = frame.close.reindex(gold.index, method="ffill", limit=4)
        out[f"{name}_shock_4"] = (
            aligned.pct_change(4) /
            aligned.pct_change().rolling(64).std().replace(0, np.nan))
    return out.replace([np.inf, -np.inf], np.nan)


def _raw_models(seed):
    return [
        HistGradientBoostingClassifier(
            learning_rate=.035, max_iter=180, max_leaf_nodes=15,
            min_samples_leaf=35, l2_regularization=2.0, random_state=seed),
        ExtraTreesClassifier(
            n_estimators=180, max_depth=7, min_samples_leaf=20,
            max_features="sqrt", class_weight="balanced", n_jobs=-1,
            random_state=seed + 100),
        LogisticRegression(
            C=.20, max_iter=2000, class_weight="balanced", random_state=seed + 200),
    ]


def _fit_ensemble(train_x, train_y, predict_x, seed):
    models, probabilities = _raw_models(seed), []
    for model in models:
        model.fit(train_x, train_y)
        probabilities.append(model.predict_proba(predict_x)[:, 1])
    return models, np.mean(probabilities, axis=0)


def _sigmoid_calibrate(raw_cal, y_cal, raw_test):
    """Fold-local Platt calibration that never sees test outcomes."""
    raw_cal = np.clip(raw_cal, 1e-5, 1 - 1e-5)
    raw_test = np.clip(raw_test, 1e-5, 1 - 1e-5)
    if pd.Series(y_cal).nunique() < 2:
        return raw_test
    logit = lambda p: np.log(p / (1 - p)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, max_iter=1000)
    calibrator.fit(logit(raw_cal), y_cal)
    return calibrator.predict_proba(logit(raw_test))[:, 1]


def _folds(n, splits, gap):
    """Expanding chronological folds with a label purge."""
    test_size = n // (splits + 1)
    if test_size < 120:
        raise ValueError("Too little history for reliable Version 8.3 folds.")
    for fold in range(splits):
        test_start = n - (splits - fold) * test_size
        train_end = test_start - gap
        test_end = min(n, test_start + test_size)
        if train_end >= 600:
            yield np.arange(train_end), np.arange(test_start, test_end)


def fit_v83_system(gold, confirmations, splits=5, cost_bps=10, threshold=.60):
    """Purged, calibrated one-hour ensemble with an 8.2.5 champion comparison."""
    features = _v83_features(gold, confirmations)
    future = gold.close.shift(-INTRADAY_HORIZON_BARS) / gold.close - 1
    labelled = features.join(future.rename("future_return")).dropna()
    if len(labelled) < 1200:
        raise ValueError(f"Only {len(labelled)} labelled rows; Version 8.3 requires 1,200.")
    X, returns = labelled[features.columns], labelled.future_return
    y = (returns > 0).astype(int)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(
            _folds(len(X), splits, INTRADAY_HORIZON_BARS), 1):
        cal_size = max(160, int(len(train_idx) * .20))
        fit_end = len(train_idx) - cal_size - INTRADAY_HORIZON_BARS
        fit_idx = train_idx[:fit_end]
        cal_idx = train_idx[fit_end + INTRADAY_HORIZON_BARS:]
        scaler = StandardScaler().fit(X.iloc[fit_idx])
        fit_x, cal_x = scaler.transform(X.iloc[fit_idx]), scaler.transform(X.iloc[cal_idx])
        test_x = scaler.transform(X.iloc[test_idx])
        _, raw_cal = _fit_ensemble(fit_x, y.iloc[fit_idx], cal_x, 100 + fold)
        _, raw_test = _fit_ensemble(fit_x, y.iloc[fit_idx], test_x, 100 + fold)
        pred = pd.DataFrame(index=X.iloc[test_idx].index)
        pred["actual_return"], pred["actual_up"] = returns.iloc[test_idx], y.iloc[test_idx]
        pred["raw_probability_up"] = raw_test
        pred["probability_up"] = _sigmoid_calibrate(raw_cal, y.iloc[cal_idx], raw_test)
        _, regressors = _models(300 + fold)
        for name, model in regressors.items():
            model.fit(fit_x, returns.iloc[fit_idx])
            pred[name] = model.predict(test_x)
        pred["fold"] = fold
        rows.append(pred)
    if not rows:
        raise ValueError("No valid Version 8.3 folds could be constructed.")
    predictions = pd.concat(rows).sort_index()
    predictions["signal"] = [
        _signal(p, m, cost_bps * 2, threshold)
        for p, m in zip(predictions.probability_up, predictions["median"])]
    predictions["position"] = predictions.signal.map({"BUY": 1, "SELL": -1, "WAIT": 0})
    events = predictions.iloc[::INTRADAY_HORIZON_BARS].copy()
    turnover = events.position.diff().abs().fillna(events.position.abs())
    events["strategy_return"] = (
        events.position * events.actual_return - turnover * cost_bps / 10_000)
    equity = (1 + events.strategy_return).cumprod()
    brier = float(brier_score_loss(predictions.actual_up, predictions.probability_up))
    raw_brier = float(brier_score_loss(
        predictions.actual_up, predictions.raw_probability_up))
    metrics = {
        "ROC-AUC": _safe_auc(predictions.actual_up, predictions.probability_up),
        "Raw ensemble ROC-AUC": _safe_auc(
            predictions.actual_up, predictions.raw_probability_up),
        "Brier score": brier,
        "Raw ensemble Brier score": raw_brier,
        "Log loss": float(log_loss(
            predictions.actual_up, predictions.probability_up)),
        "Direction accuracy": float(accuracy_score(
            predictions.actual_up, predictions.probability_up >= .5)),
        "Always-up accuracy": float(predictions.actual_up.mean()),
        "Calibration improvement": raw_brier - brier,
        "80% interval coverage": float((
            (predictions.actual_return >= predictions.lower) &
            (predictions.actual_return <= predictions.upper)).mean()),
        "Strategy total return": float(equity.iloc[-1] - 1),
        "Strategy max drawdown": float((equity / equity.cummax() - 1).min()),
        "Signal changes": float(
            (events.position.diff().fillna(events.position) != 0).sum()),
    }
    from gold_model_v82 import fit_intraday_system
    champion = fit_intraday_system(
        gold, confirmations, splits, cost_bps, threshold)
    baseline_metrics = dict(champion.metrics)
    baseline_metrics["Champion ROC-AUC"] = champion.metrics.get("ROC-AUC", np.nan)
    baseline_metrics["Champion Brier score"] = champion.metrics.get("Brier score", np.nan)

    clean, latest = features.dropna(), features.dropna().iloc[[-1]]
    scaler = StandardScaler().fit(X)
    scaled, latest_scaled = scaler.transform(X), scaler.transform(latest)
    cal_size = max(200, int(len(X) * .20))
    fit_end = len(X) - cal_size - INTRADAY_HORIZON_BARS
    fit_idx = np.arange(fit_end)
    cal_idx = np.arange(fit_end + INTRADAY_HORIZON_BARS, len(X))
    _, raw_cal = _fit_ensemble(
        scaled[fit_idx], y.iloc[fit_idx], scaled[cal_idx], 900)
    final_models, raw_latest = _fit_ensemble(
        scaled[fit_idx], y.iloc[fit_idx], latest_scaled, 900)
    probability = float(_sigmoid_calibrate(
        raw_cal, y.iloc[cal_idx], raw_latest)[0])
    _, regressors = _models(999)
    forecast = {}
    for name, model in regressors.items():
        model.fit(scaled[fit_idx], returns.iloc[fit_idx])
        forecast[name] = float(model.predict(latest_scaled)[0])
    sample = min(500, len(X))
    perm = permutation_importance(
        final_models[0], scaled[-sample:], y.iloc[-sample:], n_repeats=2,
        random_state=42, scoring="neg_brier_score")
    importance = pd.Series(
        perm.importances_mean, index=X.columns).nlargest(15)
    as_of = latest.index[0]
    spot = float(gold.close.loc[:as_of].iloc[-1])
    regime = (
        "HIGH VOLATILITY"
        if clean.vol_ratio_8_64.iloc[-1] > 1.35 else "NORMAL")
    return ForecastResult(
        as_of, spot, probability, forecast["median"], forecast["lower"],
        forecast["upper"],
        _signal(probability, forecast["median"], cost_bps * 2, threshold),
        regime, metrics, baseline_metrics, predictions, importance,
        list(X.columns), len(gold))


def classify_macro_regime(macro, as_of):
    if macro is None or macro.empty:
        return "NEUTRAL", 0.0, ["no released macro history is available"]
    known = macro.loc[macro.index <= pd.Timestamp(as_of)].copy()
    directions = {
        "real_rate_pct": -1, "dollar_index": -1,
        "macro_uncertainty_index": 1, "market_volatility_vix": 1,
        "central_bank_demand_tonnes": 1, "gold_managed_money_net": 1,
        "geopolitical_risk_index": 1}
    contributions, used = [], []
    for name, direction in directions.items():
        if name not in known:
            continue
        values = pd.to_numeric(known[name], errors="coerce").dropna()
        if len(values) < 20:
            continue
        window = values.tail(min(252, len(values)))
        deviation = window.std()
        if not np.isfinite(deviation) or deviation == 0:
            continue
        z = (window.iloc[-1] - window.mean()) / deviation
        contributions.append(float(np.clip(direction * z, -2, 2)))
        used.append(name)
    if len(contributions) < 2:
        return "NEUTRAL", 0.0, ["fewer than two mature macro regime factors"]
    score = float(np.mean(contributions))
    regime = "BULLISH" if score >= .35 else "BEARISH" if score <= -.35 else "NEUTRAL"
    return regime, score, used


def non_overlapping_evaluation(result, threshold, cost_bps):
    sample = result.predictions.iloc[::INTRADAY_HORIZON_BARS].copy()
    minimum = 2 * cost_bps / 10_000
    sample["position"] = 0
    sample.loc[
        (sample.probability_up >= threshold) &
        (sample["median"] > minimum), "position"] = 1
    sample.loc[
        (sample.probability_up <= 1 - threshold) &
        (sample["median"] < -minimum), "position"] = -1
    traded = sample[sample.position != 0].copy()
    if traded.empty:
        return {"Observations": float(len(sample)), "Trades": 0.0,
                "Win rate": np.nan, "Profit factor": np.nan,
                "Net return": 0.0, "Max drawdown": 0.0}
    traded["net_return"] = (
        traded.position * traded.actual_return - cost_bps / 10_000)
    equity = (1 + traded.net_return).cumprod()
    gains = float(traded.loc[traded.net_return > 0, "net_return"].sum())
    losses = float(-traded.loc[traded.net_return < 0, "net_return"].sum())
    return {"Observations": float(len(sample)), "Trades": float(len(traded)),
            "Win rate": float((traded.net_return > 0).mean()),
            "Profit factor": gains / losses if losses > 0 else np.inf,
            "Net return": float(equity.iloc[-1] - 1),
            "Max drawdown": float((equity / equity.cummax() - 1).min())}


def decide_v83(result, macro, elliott, threshold, cost_bps, major_event=False):
    regime, _, evidence = classify_macro_regime(macro, result.as_of)
    evaluation = non_overlapping_evaluation(result, threshold, cost_bps)
    minimum = 2 * cost_bps / 10_000
    probability = float(elliott.probability_up)
    candidate = (
        "BUY" if probability >= threshold and result.median_return > minimum
        else "SELL" if probability <= 1 - threshold and result.median_return < -minimum
        else "NO EDGE")
    reasons = intraday_release_reasons(result)
    champion_auc = result.baseline_metrics.get("Champion ROC-AUC")
    champion_brier = result.baseline_metrics.get("Champion Brier score")
    if champion_auc is not None and result.metrics.get("ROC-AUC", 0) <= champion_auc:
        reasons.append("Version 8.3 did not beat Version 8.2.5 ROC-AUC")
    if champion_brier is not None and result.metrics.get("Brier score", 1) >= champion_brier:
        reasons.append("Version 8.3 did not beat Version 8.2.5 calibration")
    if result.metrics.get("Calibration improvement", 0) < 0:
        reasons.append("fold-local calibration did not improve Brier score")
    if evaluation["Trades"] < 20:
        reasons.append("fewer than 20 non-overlapping cost-aware trades")
    if evaluation["Net return"] <= 0:
        reasons.append("non-overlapping cost-aware return is not positive")
    if np.isfinite(evaluation["Profit factor"]) and evaluation["Profit factor"] < 1.20:
        reasons.append("cost-aware profit factor is below 1.20")
    if candidate == "BUY" and regime == "BEARISH":
        reasons.append("BUY candidate conflicts with bearish macro regime")
    if candidate == "SELL" and regime == "BULLISH":
        reasons.append("SELL candidate conflicts with bullish macro regime")
    if (elliott.qualified and candidate != "NO EDGE" and
            elliott.current_bias not in {candidate, "NEUTRAL"}):
        reasons.append("qualified Elliott structure contradicts the candidate")
    if evidence and evidence[0].startswith("fewer"):
        reasons.append(evidence[0])
    if major_event:
        reasons.append("major US event flagged within the next hour")
        action = "BLOCKED"
    elif candidate == "NO EDGE" or reasons:
        action = "NO EDGE"
    else:
        action = candidate
    return V83Decision(
        action, candidate, regime, float(result.median_return), minimum,
        list(dict.fromkeys(reasons)), evaluation)
