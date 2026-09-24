from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from gold_model_v6 import (
    SYMBOL,
    _atr,
    download_daily_bundle,
    make_daily_features,
    validate_daily_data,
)

MODEL_VERSION = "6.2-research"
STOP_R = 1.0
MAX_HORIZON = 20
HOLDOUT_BARS = 504
PROBABILITY_THRESHOLDS = (0.20, 0.25, 0.30)
CANDIDATES = (
    (5, 1.5), (5, 2.0),
    (10, 1.5), (10, 2.0),
    (20, 1.5), (20, 2.0),
)


@dataclass(frozen=True)
class Candidate:
    horizon: int
    target_r: float
    threshold: float

    @property
    def label(self) -> str:
        return f"{self.horizon}d / {self.target_r:.1f}R / p≥{self.threshold:.2f}"


@dataclass
class ResearchResult:
    as_of: pd.Timestamp
    spot: float
    atr: float
    probability: float
    signal: str
    candidate: Candidate | None
    regime: str
    metrics: dict[str, float]
    development_folds: pd.DataFrame
    holdout_predictions: pd.DataFrame
    candidate_audit: pd.DataFrame
    feature_names: list[str]
    observations: int
    source_status: dict[str, str]
    release_passed: bool
    release_reasons: list[str]


def _barrier_targets(
    gold: pd.DataFrame, horizon: int, target_r: float, stop_r: float = STOP_R,
) -> pd.DataFrame:
    atr = _atr(gold)
    values: list[tuple[float, float]] = []
    for i in range(len(gold)):
        if i + horizon >= len(gold) or not np.isfinite(atr.iloc[i]) or atr.iloc[i] <= 0:
            values.append((np.nan, np.nan))
            continue
        entry, unit = float(gold.close.iloc[i]), float(atr.iloc[i])
        result = None
        for j in range(i + 1, i + horizon + 1):
            # Conservative convention: if both barriers are inside one daily bar,
            # count the stop first because intraday ordering is unknown.
            if gold.low.iloc[j] <= entry - stop_r * unit:
                result = -stop_r
                break
            if gold.high.iloc[j] >= entry + target_r * unit:
                result = target_r
                break
        if result is None:
            terminal = (float(gold.close.iloc[i + horizon]) - entry) / unit
            result = float(np.clip(terminal, -stop_r, target_r))
        values.append((float(result >= target_r), float(result)))
    return pd.DataFrame(values, index=gold.index, columns=["success", "gross_r"])


def _fit_probability(
    train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame,
) -> tuple[np.ndarray, StandardScaler, LogisticRegression | None]:
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_x)
    scaled_test = scaler.transform(test_x)
    if train_y.nunique() < 2:
        return np.full(len(test_x), float(train_y.mean())), scaler, None
    model = LogisticRegression(C=0.15, max_iter=2500, random_state=620)
    model.fit(scaled_train, train_y.astype(int))
    return model.predict_proba(scaled_test)[:, 1], scaler, model


def _predict_latest(
    train_x: pd.DataFrame, train_y: pd.Series, latest_x: pd.DataFrame,
) -> float:
    probability, _, _ = _fit_probability(train_x, train_y, latest_x)
    return float(probability[0])


def _trade_sample(
    frame: pd.DataFrame, horizon: int, threshold: float, cost_bps: float,
) -> pd.DataFrame:
    scheduled = frame.iloc[::horizon].copy()
    scheduled = scheduled[scheduled.probability >= threshold].copy()
    if scheduled.empty:
        scheduled["net_r"] = pd.Series(dtype=float)
        return scheduled
    cost_r = (cost_bps / 10_000) / scheduled.atr_pct.replace(0, np.nan)
    scheduled["net_r"] = scheduled.gross_r - cost_r
    return scheduled.dropna(subset=["net_r"])


def _trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "trades": 0.0, "win_rate": float("nan"), "expectancy": float("nan"),
            "lower_bound": float("nan"), "profit_factor": float("nan"),
        }
    net = trades.net_r.astype(float)
    se = net.std(ddof=1) / sqrt(len(net)) if len(net) > 1 else float("inf")
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()
    return {
        "trades": float(len(net)),
        "win_rate": float((net > 0).mean()),
        "expectancy": float(net.mean()),
        "lower_bound": float(net.mean() - 1.645 * se),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
    }


def _candidate_data(
    features: pd.DataFrame, target_map: dict[tuple[int, float], pd.DataFrame],
    horizon: int, target_r: float,
) -> pd.DataFrame:
    return features.join(target_map[(horizon, target_r)]).dropna()


def _select_candidate(
    features: pd.DataFrame,
    target_map: dict[tuple[int, float], pd.DataFrame],
    end_position: int,
    cost_bps: float,
    minimum_trades: int = 15,
) -> tuple[Candidate | None, pd.DataFrame]:
    start = max(500, int(end_position * 0.75))
    audit: list[dict[str, float | str]] = []
    for horizon, target_r in CANDIDATES:
        data = _candidate_data(features, target_map, horizon, target_r).iloc[:end_position]
        split = min(start, len(data) - horizon - 1)
        if split < 400 or len(data) - split <= horizon:
            continue
        train = data.iloc[: split - horizon]
        validation = data.iloc[split:]
        probabilities, _, _ = _fit_probability(
            train[features.columns], train.success, validation[features.columns])
        scored = validation[["gross_r"]].copy()
        scored["probability"] = probabilities
        scored["atr_pct"] = validation.atr_pct
        for threshold in PROBABILITY_THRESHOLDS:
            trades = _trade_sample(scored, horizon, threshold, cost_bps)
            stats = _trade_stats(trades)
            audit.append({
                "candidate": f"{horizon}d / {target_r:.1f}R",
                "horizon": horizon,
                "target_r": target_r,
                "threshold": threshold,
                **stats,
            })
    table = pd.DataFrame(audit)
    if table.empty:
        return None, pd.DataFrame(columns=[
            "candidate", "horizon", "target_r", "threshold", "trades",
            "win_rate", "expectancy", "lower_bound", "profit_factor",
        ])
    eligible = table[
        (table.trades >= minimum_trades)
        & np.isfinite(table.lower_bound)
        & (table.lower_bound > 0)
        & (table.profit_factor >= 1.10)
    ]
    if eligible.empty:
        return None, table
    best = eligible.sort_values(
        ["lower_bound", "expectancy", "trades"], ascending=False).iloc[0]
    return Candidate(int(best.horizon), float(best.target_r), float(best.threshold)), table


def _score_period(
    data: pd.DataFrame,
    feature_names: list[str],
    train_end: int,
    test_start: int,
    test_end: int,
    candidate: Candidate,
    cost_bps: float,
    fold: int | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data.iloc[: max(0, train_end - candidate.horizon)]
    test = data.iloc[test_start:test_end]
    if len(train) < 400 or test.empty:
        return pd.DataFrame(), pd.DataFrame()
    probabilities, _, _ = _fit_probability(
        train[feature_names], train.success, test[feature_names])
    scored = test[["success", "gross_r", "atr_pct"]].copy()
    scored["probability"] = probabilities
    scored["signal"] = np.where(scored.probability >= candidate.threshold, "BUY", "WAIT")
    scored["fold"] = fold
    scored["candidate"] = candidate.label
    trades = _trade_sample(scored, candidate.horizon, candidate.threshold, cost_bps)
    trades["fold"] = fold
    trades["candidate"] = candidate.label
    return scored, trades


def _safe_auc(actual: pd.Series, probability: pd.Series) -> float:
    if actual.empty or actual.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(actual, probability))


def fit_research_system(
    gold: pd.DataFrame,
    sources: dict[str, pd.Series],
    source_status: dict[str, str],
    splits: int = 5,
    cost_bps: float = 15,
    risk_fraction: float = 0.0025,
) -> ResearchResult:
    features = make_daily_features(gold, sources).dropna()
    if len(features) < 1400:
        raise ValueError(f"Only {len(features)} complete daily feature rows; at least 1,400 are required.")
    feature_names = list(features.columns)
    target_map = {
        candidate: _barrier_targets(gold, *candidate).reindex(features.index)
        for candidate in CANDIDATES
    }
    common_end = min(len(_candidate_data(features, target_map, *c)) for c in CANDIDATES)
    if common_end < HOLDOUT_BARS + 900:
        raise ValueError("Not enough labelled history for the locked recent holdout.")
    development_end = common_end - HOLDOUT_BARS

    fold_rows: list[dict[str, float | str]] = []
    outer_trades: list[pd.DataFrame] = []
    outer = TimeSeriesSplit(n_splits=splits, gap=MAX_HORIZON)
    base_positions = np.arange(development_end)
    for fold, (train_idx, test_idx) in enumerate(outer.split(base_positions), 1):
        candidate, _ = _select_candidate(
            features, target_map, int(train_idx[-1] + 1), cost_bps)
        if candidate is None:
            fold_rows.append({"fold": fold, "candidate": "NO STRATEGY", "trades": 0,
                              "expectancy": np.nan, "profit_factor": np.nan})
            continue
        data = _candidate_data(features, target_map, candidate.horizon, candidate.target_r)
        _, trades = _score_period(
            data, feature_names, int(train_idx[-1] + 1), int(test_idx[0]),
            int(test_idx[-1] + 1), candidate, cost_bps, fold)
        stats = _trade_stats(trades)
        fold_rows.append({"fold": fold, "candidate": candidate.label, **stats})
        outer_trades.append(trades)

    candidate, candidate_audit = _select_candidate(
        features, target_map, development_end, cost_bps, minimum_trades=25)
    if candidate is None:
        holdout = pd.DataFrame(columns=[
            "success", "gross_r", "atr_pct", "probability", "signal", "fold", "candidate"])
        holdout_trades = pd.DataFrame(columns=["net_r"])
    else:
        data = _candidate_data(features, target_map, candidate.horizon, candidate.target_r)
        holdout, holdout_trades = _score_period(
            data, feature_names, development_end, development_end, common_end,
            candidate, cost_bps, "HOLDOUT")

    stats = _trade_stats(holdout_trades)
    if holdout.empty:
        auc = brier = float("nan")
    else:
        auc = _safe_auc(holdout.success, holdout.probability)
        brier = float(brier_score_loss(holdout.success, holdout.probability))
    if holdout_trades.empty:
        maximum_drawdown = strategy_return = float("nan")
    else:
        returns = holdout_trades.net_r * risk_fraction
        equity = (1 + returns).cumprod()
        peak = equity.cummax()
        maximum_drawdown = float((equity / peak - 1).min())
        strategy_return = float(equity.iloc[-1] - 1)
        holdout.loc[:, "equity"] = equity.reindex(holdout.index).ffill()

    fold_table = pd.DataFrame(fold_rows)
    traded_folds = fold_table[fold_table.trades > 0] if not fold_table.empty else fold_table
    positive_fold_share = (
        float((traded_folds.expectancy > 0).mean()) if len(traded_folds) else 0.0)
    metrics = {
        "Holdout ROC-AUC": auc,
        "Holdout Brier": brier,
        "Holdout trades": stats["trades"],
        "Holdout win rate": stats["win_rate"],
        "Holdout expectancy R": stats["expectancy"],
        "90% expectancy lower bound R": stats["lower_bound"],
        "Holdout profit factor": stats["profit_factor"],
        "Holdout strategy return": strategy_return,
        "Holdout maximum drawdown": maximum_drawdown,
        "Positive development-fold share": positive_fold_share,
    }
    reasons: list[str] = []
    if candidate is None:
        reasons.append("no candidate passed the training-only selection gate")
    if stats["trades"] < 30:
        reasons.append("fewer than 30 untouched-holdout trades")
    if not np.isfinite(stats["profit_factor"]) or stats["profit_factor"] < 1.20:
        reasons.append("untouched-holdout profit factor is below 1.20 or undefined")
    if not np.isfinite(stats["lower_bound"]) or stats["lower_bound"] <= 0:
        reasons.append("untouched-holdout 90% expectancy lower bound is not positive")
    if np.isfinite(maximum_drawdown) and maximum_drawdown < -0.10:
        reasons.append("untouched-holdout maximum drawdown exceeds 10%")
    if not np.isfinite(auc) or auc < 0.53:
        reasons.append("untouched-holdout classifier ROC-AUC is below 0.53")
    if positive_fold_share < 0.60:
        reasons.append("fewer than 60% of traded development folds have positive expectancy")

    latest_features = features.iloc[[-1]]
    if candidate is None:
        probability = float("nan")
    else:
        selected = _candidate_data(features, target_map, candidate.horizon, candidate.target_r)
        probability = _predict_latest(
            selected[feature_names], selected.success.astype(int), latest_features[feature_names])
    signal = "BUY" if candidate and probability >= candidate.threshold and not reasons else "WAIT"
    latest = latest_features.iloc[0]
    high_vol = latest.gold_vol_20 >= features.gold_vol_20.tail(500).quantile(.75)
    trend = latest.gold_ema_gap_60
    regime = "HIGH VOLATILITY" if high_vol else (
        "UPTREND" if trend > .03 else "DOWNTREND" if trend < -.03 else "RANGE")
    as_of = latest_features.index[-1]
    return ResearchResult(
        as_of=as_of,
        spot=float(gold.close.loc[as_of]),
        atr=float(_atr(gold).loc[as_of]),
        probability=probability,
        signal=signal,
        candidate=candidate,
        regime=regime,
        metrics=metrics,
        development_folds=fold_table,
        holdout_predictions=holdout,
        candidate_audit=candidate_audit,
        feature_names=feature_names,
        observations=len(gold),
        source_status=source_status,
        release_passed=not reasons,
        release_reasons=reasons,
    )
