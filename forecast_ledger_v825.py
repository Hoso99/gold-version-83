from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


LEDGER_COLUMNS = [
    "forecast_id", "created_at_utc", "model_version", "data_timestamp_utc",
    "forecast_timestamp_utc", "starting_price", "directional_outlook", "decision",
    "market_probability_up", "adjusted_probability_up", "lower_target", "median_target", "upper_target",
    "release_gate", "gate_reasons", "status", "actual_timestamp_utc",
    "actual_price", "actual_return", "target_error", "outlook_result", "direction_result",
]


class ForecastLedger:
    """Persistent, research-only forecast history with causal one-hour settlement."""

    def __init__(self, path: str | Path = "forecast_ledger_v825.sqlite"):
        self.path = Path(path)
        self._initialize()

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def _initialize(self):
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS forecasts (
                    forecast_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    data_timestamp_utc TEXT NOT NULL,
                    forecast_timestamp_utc TEXT NOT NULL,
                    starting_price REAL NOT NULL,
                    directional_outlook TEXT,
                    decision TEXT NOT NULL,
                    market_probability_up REAL NOT NULL,
                    adjusted_probability_up REAL NOT NULL,
                    lower_target REAL,
                    median_target REAL NOT NULL,
                    upper_target REAL,
                    release_gate TEXT NOT NULL,
                    gate_reasons TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    actual_timestamp_utc TEXT,
                    actual_price REAL,
                    actual_return REAL,
                    target_error REAL,
                    outlook_result TEXT,
                    direction_result TEXT
                )"""
            )
            existing = {row[1] for row in connection.execute("PRAGMA table_info(forecasts)")}
            additions = {
                "directional_outlook": "TEXT",
                "lower_target": "REAL",
                "upper_target": "REAL",
                "outlook_result": "TEXT",
            }
            for name, kind in additions.items():
                if name not in existing:
                    connection.execute(f"ALTER TABLE forecasts ADD COLUMN {name} {kind}")

    @staticmethod
    def _utc(value) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")

    def record(
        self,
        *,
        model_version: str,
        data_timestamp,
        forecast_timestamp,
        starting_price: float,
        directional_outlook: str,
        decision: str,
        market_probability_up: float,
        adjusted_probability_up: float,
        lower_target: float,
        median_target: float,
        upper_target: float,
        release_gate: str,
        gate_reasons: str,
    ) -> bool:
        data_time = self._utc(data_timestamp)
        forecast_time = self._utc(forecast_timestamp)
        forecast_id = f"{model_version}|{data_time.isoformat()}"
        now = pd.Timestamp.now(tz="UTC").isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO forecasts (
                    forecast_id, created_at_utc, model_version, data_timestamp_utc,
                    forecast_timestamp_utc, starting_price, directional_outlook, decision,
                    market_probability_up, adjusted_probability_up, lower_target, median_target, upper_target,
                    release_gate, gate_reasons, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')""",
                (
                    forecast_id, now, model_version, data_time.isoformat(),
                    forecast_time.isoformat(), float(starting_price), directional_outlook, decision,
                    float(market_probability_up), float(adjusted_probability_up),
                    float(lower_target), float(median_target), float(upper_target),
                    release_gate, gate_reasons,
                ),
            )
        return cursor.rowcount == 1

    def settle(self, gold: pd.DataFrame, tolerance_minutes: int = 20) -> int:
        if gold.empty or "close" not in gold:
            return 0
        prices = gold[["close"]].copy().sort_index()
        prices.index = pd.to_datetime(prices.index, utc=True)
        with self._connect() as connection:
            pending = pd.read_sql_query(
                "SELECT * FROM forecasts WHERE status = 'PENDING' ORDER BY forecast_timestamp_utc",
                connection,
            )
            settled = 0
            for row in pending.itertuples(index=False):
                expiry = self._utc(row.forecast_timestamp_utc)
                position = prices.index.searchsorted(expiry, side="left")
                if position >= len(prices):
                    continue
                actual_time = prices.index[position]
                delay = (actual_time - expiry).total_seconds() / 60
                if delay < 0 or delay > tolerance_minutes:
                    continue
                actual_price = float(prices.iloc[position].close)
                actual_return = actual_price / float(row.starting_price) - 1
                target_error = actual_price - float(row.median_target)
                outlook = row.directional_outlook or (
                    row.decision if row.decision in {"BUY", "SELL"} else None)
                if outlook == "BUY":
                    outlook_result = "CORRECT" if actual_return > 0 else "INCORRECT"
                elif outlook == "SELL":
                    outlook_result = "CORRECT" if actual_return < 0 else "INCORRECT"
                else:
                    outlook_result = "NOT_SCORED"
                if row.decision == "BUY":
                    direction_result = "CORRECT" if actual_return > 0 else "INCORRECT"
                elif row.decision == "SELL":
                    direction_result = "CORRECT" if actual_return < 0 else "INCORRECT"
                else:
                    direction_result = "NOT_SCORED"
                connection.execute(
                    """UPDATE forecasts SET status = 'SETTLED', actual_timestamp_utc = ?,
                       actual_price = ?, actual_return = ?, target_error = ?, outlook_result = ?, direction_result = ?
                       WHERE forecast_id = ? AND status = 'PENDING'""",
                    (
                        actual_time.isoformat(), actual_price, actual_return,
                        target_error, outlook_result, direction_result, row.forecast_id,
                    ),
                )
                settled += 1
        return settled

    def frame(self) -> pd.DataFrame:
        with self._connect() as connection:
            frame = pd.read_sql_query(
                "SELECT * FROM forecasts ORDER BY data_timestamp_utc DESC", connection)
        if frame.empty:
            return pd.DataFrame(columns=LEDGER_COLUMNS)
        for column in ("created_at_utc", "data_timestamp_utc", "forecast_timestamp_utc",
                       "actual_timestamp_utc"):
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        for column in ("lower_target", "median_target", "upper_target",
                       "actual_price", "actual_return", "target_error"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame[LEDGER_COLUMNS]
