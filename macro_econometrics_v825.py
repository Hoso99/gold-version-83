from __future__ import annotations

from dataclasses import dataclass
import io
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

HORIZON_BARS = 4
FRED_SERIES = {
    "real_rate_pct": "DFII10",
    "dollar_index": "DTWEXBGS",
    "macro_uncertainty_index": "USEPUINDXD",
    "market_volatility_vix": "VIXCLS",
}
OPTIONAL_COLUMNS = [
    "central_bank_demand_tonnes",
    "gold_managed_money_net",
    "geopolitical_risk_index",
]
ALL_COLUMNS = list(FRED_SERIES) + OPTIONAL_COLUMNS
INTRADAY_PREFIX = "intraday_"


@dataclass
class EconometricOverlay:
    probability_up: float
    adjustment: float
    qualified: bool
    roc_auc: float
    brier: float
    observations: int
    features: list[str]
    latest: dict[str, float]
    source_audit: pd.DataFrame
    reasons: list[str]


def _fred_csv(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        with urlopen(url, timeout=30) as response:
            payload = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"FRED {series_id} returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not connect to FRED {series_id}: {exc.reason}") from exc
    frame = pd.read_csv(io.BytesIO(payload))
    if frame.shape[1] < 2:
        raise RuntimeError(f"FRED {series_id} returned an unexpected file")
    dates = pd.to_datetime(frame.iloc[:, 0], utc=True, errors="coerce")
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    return pd.Series(values.to_numpy(), index=dates, name=series_id).dropna().sort_index()


def download_fred_macro() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download latest-known public macro series and conservatively lag them one day."""
    series, audit = {}, []
    for feature, series_id in FRED_SERIES.items():
        try:
            values = _fred_csv(series_id)
            # Observation dates are not point-in-time vintages. A one-day lag is a
            # conservative minimum; production research should use ALFRED vintages.
            values.index = values.index + pd.Timedelta(days=1)
            series[feature] = values
            audit.append({"Factor": feature, "Source": f"FRED {series_id}",
                          "Latest release used": values.index[-1], "Status": "AVAILABLE"})
        except Exception as exc:
            audit.append({"Factor": feature, "Source": f"FRED {series_id}",
                          "Latest release used": pd.NaT, "Status": f"UNAVAILABLE: {exc}"})
    frame = pd.concat(series, axis=1).sort_index() if series else pd.DataFrame()
    return frame, pd.DataFrame(audit)


def parse_slow_factor_csv(upload) -> pd.DataFrame:
    """Read release-timestamped WGC/CFTC inputs without inventing unavailable history."""
    if upload is None:
        return pd.DataFrame(columns=OPTIONAL_COLUMNS)
    frame = pd.read_csv(upload)
    if "release_timestamp" not in frame:
        raise ValueError("Macro CSV must contain release_timestamp.")
    present = [name for name in OPTIONAL_COLUMNS if name in frame]
    if not present:
        raise ValueError(
            "Macro CSV needs central_bank_demand_tonnes, gold_managed_money_net, "
            "or geopolitical_risk_index."
        )
    frame["release_timestamp"] = pd.to_datetime(frame.release_timestamp, utc=True, errors="coerce")
    for name in present:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return (frame.dropna(subset=["release_timestamp"]).set_index("release_timestamp")[present]
            .sort_index().groupby(level=0).last())


def combine_macro_sources(
    fred: pd.DataFrame,
    slow: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames = [frame for frame in (fred, slow) if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=ALL_COLUMNS)
    return pd.concat(frames, axis=1).sort_index().groupby(level=0).last()


def _aligned_features(macro: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    if macro.empty:
        return pd.DataFrame(index=index)
    available = macro.loc[macro.index <= index[-1]].copy()
    aligned = available.reindex(available.index.union(index)).sort_index().ffill().reindex(index)
    out = pd.DataFrame(index=index)
    eligible = [
        column for column in ALL_COLUMNS
        if column in aligned and macro.loc[macro.index <= index[-1], column].dropna().shape[0] >= 20
    ]
    for name in eligible:
        values = pd.to_numeric(aligned[name], errors="coerce")
        out[name] = values
        out[f"{name}_change"] = values.diff(96 if name in FRED_SERIES else 1)
        expanding_mean = values.expanding(min_periods=20).mean()
        expanding_std = values.expanding(min_periods=20).std().replace(0, np.nan)
        out[f"{name}_z"] = (values - expanding_mean) / expanding_std
    return out.replace([np.inf, -np.inf], np.nan)


def _intraday_features(
    confirmations: dict[str, pd.DataFrame] | None,
    index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, list[dict]]:
    """Build causal market-proxy changes for one-hour timing."""
    out = pd.DataFrame(index=index)
    audit = []
    for name, frame in (confirmations or {}).items():
        if frame is None or frame.empty or "close" not in frame:
            continue
        close = pd.to_numeric(frame.close, errors="coerce").copy()
        close.index = pd.to_datetime(close.index, utc=True)
        close = close[~close.index.duplicated(keep="last")].sort_index()
        aligned = close.reindex(close.index.union(index)).sort_index().ffill().reindex(index)
        safe_name = "".join(character if character.isalnum() else "_" for character in name.lower())
        prefix = f"{INTRADAY_PREFIX}{safe_name}"
        out[f"{prefix}_return_1"] = aligned.pct_change(1, fill_method=None)
        out[f"{prefix}_return_4"] = aligned.pct_change(HORIZON_BARS, fill_method=None)
        out[f"{prefix}_volatility_16"] = aligned.pct_change(fill_method=None).rolling(16).std()
        available = aligned.dropna()
        audit.append({
            "Factor": prefix,
            "Observations": int(len(available)),
            "Latest release used": available.index[-1] if len(available) else pd.NaT,
            "Latest value": float(available.iloc[-1]) if len(available) else np.nan,
            "Frequency": "15-minute market proxy",
        })
    return out.replace([np.inf, -np.inf], np.nan), audit


def fit_econometric_overlay(
    gold: pd.DataFrame,
    macro: pd.DataFrame,
    base_probability: float,
    splits: int = 5,
    confirmations: dict[str, pd.DataFrame] | None = None,
) -> EconometricOverlay:
    source_rows = []
    for name in ALL_COLUMNS:
        values = (macro.loc[macro.index <= gold.index[-1], name].dropna()
                  if name in macro else pd.Series(dtype=float))
        source_rows.append({"Factor": name, "Observations": len(values),
                            "Latest release used": values.index[-1] if len(values) else pd.NaT,
                            "Latest value": float(values.iloc[-1]) if len(values) else np.nan,
                            "Frequency": "slow release/regime"})
    slow_features = _aligned_features(macro, gold.index)
    fast_features, fast_audit = _intraday_features(confirmations, gold.index)
    source_audit = pd.DataFrame(source_rows + fast_audit)
    features = pd.concat([slow_features, fast_features], axis=1)
    target_return = gold.close.shift(-HORIZON_BARS) / gold.close - 1
    target = (target_return > 0).astype(float); target[target_return.isna()] = np.nan
    joined = features.join(target.rename("up")).dropna(axis=1, how="all")
    candidate_names = [column for column in joined if column != "up"]
    feature_names = [name for name in candidate_names if joined[name].nunique(dropna=True) >= 3]
    labelled = joined[feature_names + ["up"]].dropna()
    fast_names = [name for name in feature_names if name.startswith(INTRADAY_PREFIX)]
    reasons = []
    if len(fast_names) < 2:
        reasons.append("fewer than two usable intraday cross-asset factors")
    if len(labelled) < 1000:
        reasons.append("fewer than 1,000 aligned intraday observations")
    if reasons:
        latest = {name: float(macro.loc[macro.index <= gold.index[-1], name].dropna().iloc[-1])
                  for name in macro if len(macro.loc[macro.index <= gold.index[-1], name].dropna())}
        return EconometricOverlay(base_probability, 0.0, False, np.nan, np.nan,
                                  len(labelled), feature_names, latest, source_audit, reasons)

    X = labelled[feature_names]; y = labelled.up.astype(int)
    effective_splits = min(splits, max(3, len(X) // 500))
    probabilities = []
    actual = []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=effective_splits, gap=HORIZON_BARS).split(X):
        scaler = StandardScaler(); train = scaler.fit_transform(X.iloc[train_idx]); test = scaler.transform(X.iloc[test_idx])
        model = LogisticRegression(C=0.25, max_iter=1000, class_weight="balanced")
        model.fit(train, y.iloc[train_idx])
        probabilities.extend(model.predict_proba(test)[:, 1]); actual.extend(y.iloc[test_idx])
    probability_series = np.asarray(probabilities); actual_series = np.asarray(actual)
    auc = float(roc_auc_score(actual_series, probability_series))
    brier = float(brier_score_loss(actual_series, probability_series))
    qualified = bool(auc >= 0.53 and brier < 0.25)
    if not qualified:
        reasons.append("mixed-frequency overlay did not demonstrate out-of-sample one-hour skill")

    scaler = StandardScaler(); scaled = scaler.fit_transform(X)
    model = LogisticRegression(C=0.25, max_iter=1000, class_weight="balanced").fit(scaled, y)
    latest_x = features[feature_names].dropna().iloc[[-1]]
    macro_probability = float(model.predict_proba(scaler.transform(latest_x))[0, 1])
    adjustment = float(np.clip((macro_probability - 0.5) * 0.10, -0.03, 0.03)) if qualified else 0.0
    final_probability = float(np.clip(base_probability + adjustment, 0.01, 0.99))
    latest = {name: float(macro.loc[macro.index <= gold.index[-1], name].dropna().iloc[-1])
              for name in macro if len(macro.loc[macro.index <= gold.index[-1], name].dropna())}
    return EconometricOverlay(final_probability, adjustment, qualified, auc, brier,
                              len(labelled), feature_names, latest, source_audit, reasons)
