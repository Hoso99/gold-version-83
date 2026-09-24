from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

from gold_model_v6 import SYMBOL, _atr, download_daily_bundle, make_daily_features, validate_daily_data
from gold_model_v62 import (
    CANDIDATES,
    HOLDOUT_BARS,
    MAX_HORIZON,
    PROBABILITY_THRESHOLDS,
    STOP_R,
    Candidate,
    ResearchResult,
    _barrier_targets,
    _candidate_data,
    _fit_probability,
    _predict_latest,
    _safe_auc,
    _trade_sample,
    _trade_stats,
)

MODEL_VERSION = "6.2.1-research"
MIN_COVERAGE = 0.10
MAX_COVERAGE = 0.65


def remove_weekends(
    gold: pd.DataFrame, sources: dict[str, pd.Series]
) -> tuple[pd.DataFrame, dict[str, pd.Series], int]:
    """Remove Saturday/Sunday observations before features or targets are made."""
    weekend_count = int((gold.index.dayofweek >= 5).sum())
    weekday_gold = gold.loc[gold.index.dayofweek < 5].copy()
    weekday_sources = {
        name: series.loc[series.index.dayofweek < 5].copy()
        for name, series in sources.items()
    }
    return weekday_gold, weekday_sources, weekend_count


def _paired_strategy_stats(
    scored: pd.DataFrame, horizon: int, threshold: float, cost_bps: float,
) -> dict[str, float]:
    scheduled = scored.iloc[::horizon].copy()
    cost_r = (cost_bps / 10_000) / scheduled.atr_pct.replace(0, np.nan)
    scheduled["benchmark_net_r"] = scheduled.gross_r - cost_r
    scheduled["strategy_net_r"] = np.where(
        scheduled.probability >= threshold, scheduled.benchmark_net_r, 0.0)
    scheduled = scheduled.dropna(subset=["benchmark_net_r", "strategy_net_r"])
    trades = scheduled[scheduled.probability >= threshold].copy()
    trades["net_r"] = trades.strategy_net_r
    stats = _trade_stats(trades)
    coverage = float(len(trades) / len(scheduled)) if len(scheduled) else float("nan")
    difference = scheduled.strategy_net_r - scheduled.benchmark_net_r
    difference_se = (
        difference.std(ddof=1) / sqrt(len(difference)) if len(difference) > 1 else float("inf"))
    stats.update({
        "coverage": coverage,
        "benchmark_expectancy": float(scheduled.benchmark_net_r.mean()) if len(scheduled) else float("nan"),
        "incremental_expectancy": float(difference.mean()) if len(difference) else float("nan"),
        "incremental_lower_bound": (
            float(difference.mean() - 1.645 * difference_se) if len(difference) else float("nan")),
    })
    return stats


def _select_candidate_v621(
    features: pd.DataFrame,
    target_map: dict[tuple[int, float], pd.DataFrame],
    end_position: int,
    cost_bps: float,
    minimum_trades: int = 15,
) -> tuple[Candidate | None, pd.DataFrame]:
    validation_start = max(500, int(end_position * 0.75))
    rows: list[dict[str, float | str]] = []
    for horizon, target_r in CANDIDATES:
        data = _candidate_data(features, target_map, horizon, target_r).iloc[:end_position]
        split = min(validation_start, len(data) - horizon - 1)
        if split < 400 or len(data) - split <= horizon:
            continue
        train = data.iloc[: split - horizon]
        validation = data.iloc[split:]
        probabilities, _, _ = _fit_probability(
            train[features.columns], train.success, validation[features.columns])
        scored = validation[["gross_r", "atr_pct"]].copy()
        scored["probability"] = probabilities
        for threshold in PROBABILITY_THRESHOLDS:
            stats = _paired_strategy_stats(scored, horizon, threshold, cost_bps)
            rows.append({
                "candidate": f"{horizon}d / {target_r:.1f}R",
                "horizon": horizon,
                "target_r": target_r,
                "threshold": threshold,
                **stats,
            })
    table = pd.DataFrame(rows)
    if table.empty:
        return None, pd.DataFrame(columns=[
            "candidate", "horizon", "target_r", "threshold", "trades", "coverage",
            "expectancy", "lower_bound", "profit_factor", "benchmark_expectancy",
            "incremental_expectancy", "incremental_lower_bound",
        ])
    eligible = table[
        (table.trades >= minimum_trades)
        & table.coverage.between(MIN_COVERAGE, MAX_COVERAGE)
        & np.isfinite(table.lower_bound)
        & (table.lower_bound > 0)
        & (table.profit_factor >= 1.10)
        & np.isfinite(table.incremental_lower_bound)
        & (table.incremental_lower_bound > 0)
    ]
    if eligible.empty:
        return None, table
    best = eligible.sort_values(
        ["incremental_lower_bound", "lower_bound", "trades"], ascending=False).iloc[0]
    return Candidate(int(best.horizon), float(best.target_r), float(best.threshold)), table


def _score_period_v621(
    data: pd.DataFrame,
    feature_names: list[str],
    train_end: int,
    test_start: int,
    test_end: int,
    candidate: Candidate,
    cost_bps: float,
    fold: int | str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    train = data.iloc[: max(0, train_end - candidate.horizon)]
    test = data.iloc[test_start:test_end]
    if len(train) < 400 or test.empty:
        return pd.DataFrame(), pd.DataFrame(), _paired_strategy_stats(
            pd.DataFrame(columns=["gross_r", "atr_pct", "probability"]),
            candidate.horizon, candidate.threshold, cost_bps)
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
    return scored, trades, _paired_strategy_stats(
        scored, candidate.horizon, candidate.threshold, cost_bps)


def fit_research_system_v621(
    gold: pd.DataFrame,
    sources: dict[str, pd.Series],
    source_status: dict[str, str],
    splits: int = 5,
    cost_bps: float = 15,
    risk_fraction: float = 0.0025,
) -> ResearchResult:
    gold, sources, weekends_removed = remove_weekends(gold, sources)
    status = dict(source_status)
    status["Weekend filter"] = f"{weekends_removed:,} Saturday/Sunday rows removed"
    features = make_daily_features(gold, sources).dropna()
    if len(features) < 1400:
        raise ValueError(f"Only {len(features)} complete weekday rows; at least 1,400 are required.")
    feature_names = list(features.columns)
    target_map = {
        candidate: _barrier_targets(gold, *candidate).reindex(features.index)
        for candidate in CANDIDATES
    }
    common_end = min(len(_candidate_data(features, target_map, *c)) for c in CANDIDATES)
    if common_end < HOLDOUT_BARS + 900:
        raise ValueError("Not enough labelled weekday history for the locked holdout.")
    development_end = common_end - HOLDOUT_BARS

    fold_rows: list[dict[str, float | str]] = []
    outer = TimeSeriesSplit(n_splits=splits, gap=MAX_HORIZON)
    for fold, (train_idx, test_idx) in enumerate(outer.split(np.arange(development_end)), 1):
        candidate, _ = _select_candidate_v621(
            features, target_map, int(train_idx[-1] + 1), cost_bps)
        if candidate is None:
            fold_rows.append({
                "fold": fold, "candidate": "NO STRATEGY", "trades": 0,
                "coverage": 0.0, "expectancy": np.nan, "profit_factor": np.nan,
                "benchmark_expectancy": np.nan, "incremental_expectancy": np.nan,
            })
            continue
        data = _candidate_data(features, target_map, candidate.horizon, candidate.target_r)
        _, _, stats = _score_period_v621(
            data, feature_names, int(train_idx[-1] + 1), int(test_idx[0]),
            int(test_idx[-1] + 1), candidate, cost_bps, fold)
        fold_rows.append({"fold": fold, "candidate": candidate.label, **stats})

    candidate, candidate_audit = _select_candidate_v621(
        features, target_map, development_end, cost_bps, minimum_trades=25)
    if candidate is None:
        holdout = pd.DataFrame(columns=[
            "success", "gross_r", "atr_pct", "probability", "signal", "fold", "candidate"])
        holdout_trades = pd.DataFrame(columns=["net_r"])
        paired = {
            "coverage": np.nan, "benchmark_expectancy": np.nan,
            "incremental_expectancy": np.nan, "incremental_lower_bound": np.nan,
        }
    else:
        data = _candidate_data(features, target_map, candidate.horizon, candidate.target_r)
        holdout, holdout_trades, paired = _score_period_v621(
            data, feature_names, development_end, development_end, common_end,
            candidate, cost_bps, "HOLDOUT")

    trade_stats = _trade_stats(holdout_trades)
    auc = _safe_auc(holdout.success, holdout.probability) if not holdout.empty else float("nan")
    brier = (
        float(brier_score_loss(holdout.success, holdout.probability))
        if not holdout.empty else float("nan"))
    if holdout_trades.empty:
        maximum_drawdown = strategy_return = float("nan")
    else:
        returns = holdout_trades.net_r * risk_fraction
        equity = (1 + returns).cumprod()
        maximum_drawdown = float((equity / equity.cummax() - 1).min())
        strategy_return = float(equity.iloc[-1] - 1)
        holdout.loc[:, "equity"] = equity.reindex(holdout.index).ffill()

    fold_table = pd.DataFrame(fold_rows)
    traded_folds = fold_table[fold_table.trades > 0]
    positive_fold_share = (
        float((traded_folds.incremental_expectancy > 0).mean()) if len(traded_folds) else 0.0)
    metrics = {
        "Holdout ROC-AUC": auc,
        "Holdout Brier": brier,
        "Holdout trades": trade_stats["trades"],
        "Holdout trade coverage": paired["coverage"],
        "Holdout win rate": trade_stats["win_rate"],
        "Holdout expectancy R": trade_stats["expectancy"],
        "90% expectancy lower bound R": trade_stats["lower_bound"],
        "Holdout profit factor": trade_stats["profit_factor"],
        "Always-long benchmark expectancy R": paired["benchmark_expectancy"],
        "Incremental expectancy versus benchmark R": paired["incremental_expectancy"],
        "90% incremental lower bound R": paired["incremental_lower_bound"],
        "Holdout strategy return": strategy_return,
        "Holdout maximum drawdown": maximum_drawdown,
        "Positive development-fold share": positive_fold_share,
        "Weekend rows removed": float(weekends_removed),
    }
    reasons: list[str] = []
    if candidate is None:
        reasons.append("no candidate passed the training-only benchmark and coverage gates")
    if trade_stats["trades"] < 30:
        reasons.append("fewer than 30 untouched-holdout trades")
    if not np.isfinite(trade_stats["profit_factor"]) or trade_stats["profit_factor"] < 1.20:
        reasons.append("untouched-holdout profit factor is below 1.20 or undefined")
    if not np.isfinite(trade_stats["lower_bound"]) or trade_stats["lower_bound"] <= 0:
        reasons.append("untouched-holdout 90% expectancy lower bound is not positive")
    if not np.isfinite(paired["incremental_lower_bound"]) or paired["incremental_lower_bound"] <= 0:
        reasons.append("strategy does not beat always-long with a positive 90% lower bound")
    if not np.isfinite(paired["coverage"]) or not MIN_COVERAGE <= paired["coverage"] <= MAX_COVERAGE:
        reasons.append("untouched-holdout trade coverage is outside the 10% to 65% range")
    if np.isfinite(maximum_drawdown) and maximum_drawdown < -0.10:
        reasons.append("untouched-holdout maximum drawdown exceeds 10%")
    if not np.isfinite(auc) or auc < 0.53:
        reasons.append("untouched-holdout classifier ROC-AUC is below 0.53")
    if positive_fold_share < 0.60:
        reasons.append("fewer than 60% of traded development folds beat always-long")

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
        as_of=as_of, spot=float(gold.close.loc[as_of]), atr=float(_atr(gold).loc[as_of]),
        probability=probability, signal=signal, candidate=candidate, regime=regime,
        metrics=metrics, development_folds=fold_table, holdout_predictions=holdout,
        candidate_audit=candidate_audit, feature_names=feature_names,
        observations=len(gold), source_status=status, release_passed=not reasons,
        release_reasons=reasons,
    )
