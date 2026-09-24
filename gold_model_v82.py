from __future__ import annotations

from dataclasses import dataclass
import io
import json
import math
from pathlib import PurePosixPath
import re
from typing import BinaryIO, Callable
import zipfile

import numpy as np
import pandas as pd

from gold_model import (
    _download_symbol, _signal, download_market_data, expected_market_open,
    fit_and_backtest,
    technical_snapshot,
    validate_market_data as _validate_market_data_base,
)
from gold_model_v8 import *  # noqa: F401,F403 - V8.2 extends the validated V8.1 engines

MODEL_VERSION = "8.2.5-one-hour-structural-behavioral-statistical-research"
INTRADAY_INTERVAL = "15min"
INTRADAY_HORIZON_BARS = 4
INTRADAY_HORIZON_LABEL = "1 hour (4 x 15-minute bars)"

COMMUNITY_COLUMNS = [
    "channel", "message_id", "timestamp", "direction", "text", "evidence_type"
]
MAX_TELEGRAM_ARCHIVE_BYTES = 500 * 1024 * 1024
MAX_TELEGRAM_IMAGE_BYTES = 8 * 1024 * 1024
MAX_TELEGRAM_OCR_IMAGES = 300
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

BOT_CANDIDATES = {
    "FXPremiere Gold Signals Bot": {
        "url": "https://goldsignals.co/tag/gold-signals-bot/",
        "access": "provider website / Telegram delivery",
        "status": "UNVERIFIED",
    },
    "FXCryptoTools Gold Signals Bot": {
        "url": "https://fxcryptotools.com/",
        "access": "paid provider product",
        "status": "UNVERIFIED",
    },
    "Ben Gold Trader Signals Bot": {
        "url": "https://www.youtube.com/hashtag/bengoldtrader",
        "access": "MT5 product referenced in provider videos",
        "status": "UNVERIFIED",
    },
}

EXTRA_INTRADAY_PROXIES = {
    "dollar_uup": "UUP",
    "treasury_tlt": "TLT",
    "equity_spy": "SPY",
    "oil_uso": "USO",
    "bitcoin": "BTC/USD",
}

RESEARCH_SOURCES = {
    "World Gold Council Goldhub": "https://www.gold.org/goldhub/data",
    "CME Gold futures": "https://www.cmegroup.com/markets/metals/precious/gold.html",
    "LBMA benchmarks": "https://www.lbma.org.uk/",
    "US Treasury yields": "https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics",
    "CFTC positioning": "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm",
    "TradingView XAUUSD": "https://www.tradingview.com/symbols/XAUUSD/",
    "GoldPrice.org": "https://goldprice.org/",
}

# Public existence and self-described signal coverage were checked on 2026-09-21.
# Inclusion is not an endorsement. A channel receives zero model weight until its
# uploaded history passes the gates in audit_telegram_signals().
TELEGRAM_CHANNELS = {
    "gold_forex_signals_vip": {
        "title": "Gold Signals VIP (XAUUSD FOREX)",
        "url": "https://t.me/gold_forex_signals_vip",
    },
    "anabelsignals": {
        "title": "AnabelSignals – Forex & Gold Signals",
        "url": "https://t.me/anabelsignals",
    },
    "top_tradingsignals": {
        "title": "TopTradingSignals – Forex & Gold Signals",
        "url": "https://t.me/top_tradingsignals",
    },
    "firepipsignals": {
        "title": "FirePips Forex Academy",
        "url": "https://t.me/firepipsignals",
    },
}

_GOLD = re.compile(r"\b(?:xau\s*/?\s*usd|gold)\b", re.I)
_BUY = re.compile(r"\b(?:buy|long)\b", re.I)
_SELL = re.compile(r"\b(?:sell|short)\b", re.I)


@dataclass
class TelegramOverlay:
    probability_up: float
    adjustment: float
    decision: str
    consensus: str
    qualified_channels: int
    current_signals: int
    reasons: list[str]
    audit: pd.DataFrame
    messages: pd.DataFrame


@dataclass
class ElliottOverlay:
    probability_up: float
    adjustment: float
    decision: str
    current_bias: str
    current_structure: str
    observations: int
    accuracy: float
    lower_bound: float
    qualified: bool
    reasons: list[str]
    history: pd.DataFrame


def _message_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            parts.append(item.get("text", "") if isinstance(item, dict) else str(item))
        return "".join(parts)
    return ""


def _direction(text: str) -> int:
    if not _GOLD.search(text):
        return 0
    buy, sell = bool(_BUY.search(text)), bool(_SELL.search(text))
    if buy == sell:
        return 0
    return 1 if buy else -1


def _iter_chats(payload: dict):
    if isinstance(payload.get("chats"), dict):
        for chat in payload["chats"].get("list", []):
            yield chat
    elif "messages" in payload:
        yield payload


def _empty_community_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=COMMUNITY_COLUMNS)


def _telegram_media_path(message: dict) -> str | None:
    """Return a safe relative image path referenced by a Telegram export message."""
    candidate = message.get("photo") or message.get("file")
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    normalized = candidate.replace("\\", "/").lstrip("/")
    path = PurePosixPath(normalized)
    if ".." in path.parts or path.suffix.lower() not in _IMAGE_SUFFIXES:
        return None
    return str(path)


def _ocr_signal_image(image_bytes: bytes) -> str:
    """OCR one Telegram image. Imports are lazy so text-only exports stay lightweight."""
    try:
        from PIL import Image, ImageEnhance, ImageOps
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "Image OCR requires Pillow and pytesseract; install requirements.txt and tesseract-ocr."
        ) from exc
    if len(image_bytes) > MAX_TELEGRAM_IMAGE_BYTES:
        return ""
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            image = ImageOps.exif_transpose(opened).convert("L")
            image.thumbnail((2200, 2200))
            image = ImageEnhance.Contrast(image).enhance(1.7)
            return pytesseract.image_to_string(image, config="--psm 6", timeout=15)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed or is not available on PATH."
        ) from exc
    except (OSError, ValueError, RuntimeError, pytesseract.TesseractError):
        return ""


def _parse_telegram_payload(
    payload: dict,
    fallback_name: str,
    media_reader: Callable[[str], bytes | None] | None = None,
    ocr_image: Callable[[bytes], str] = _ocr_signal_image,
    max_ocr_images: int = MAX_TELEGRAM_OCR_IMAGES,
) -> pd.DataFrame:
    rows = []
    ocr_attempts = 0
    chats = list(_iter_chats(payload))
    image_candidates = []
    for chat in chats:
        for message in chat.get("messages", []):
            if (message.get("type") == "message" and
                    not message.get("edited") and not message.get("edited_unixtime") and
                    _telegram_media_path(message)):
                timestamp = pd.to_datetime(
                    message.get("date_unixtime", message.get("date")),
                    utc=True, errors="coerce")
                image_candidates.append((timestamp, id(message)))
    image_candidates.sort(
        key=lambda item: pd.Timestamp.min.tz_localize("UTC") if pd.isna(item[0]) else item[0],
        reverse=True)
    allowed_ocr_messages = {item[1] for item in image_candidates[:max_ocr_images]}

    for chat in chats:
        channel = str(chat.get("name") or fallback_name)
        for message in chat.get("messages", []):
            if message.get("type") != "message":
                continue
            # Export JSON cannot reconstruct pre-edit text, so edited calls are excluded.
            if message.get("edited") or message.get("edited_unixtime"):
                continue
            text = _message_text(message.get("text", ""))
            evidence_type = "text"
            direction = _direction(text)
            if (not direction and media_reader is not None and
                    id(message) in allowed_ocr_messages and ocr_attempts < max_ocr_images):
                media_path = _telegram_media_path(message)
                if media_path:
                    image_bytes = media_reader(media_path)
                    if image_bytes:
                        ocr_attempts += 1
                        ocr_text = ocr_image(image_bytes)
                        direction = _direction(ocr_text)
                        if direction:
                            text = ocr_text
                            evidence_type = "image_ocr"
            if not direction:
                continue
            timestamp = pd.to_datetime(
                message.get("date_unixtime", message.get("date")), utc=True, errors="coerce")
            if pd.isna(timestamp):
                continue
            rows.append({
                "channel": channel,
                "message_id": str(message.get("id", "")),
                "timestamp": timestamp,
                "direction": direction,
                "text": text[:500],
                "evidence_type": evidence_type,
            })
    if not rows:
        return _empty_community_frame()
    result = pd.DataFrame(rows).sort_values("timestamp")
    return result.drop_duplicates(["channel", "message_id", "timestamp", "direction"])


def parse_telegram_export(file: BinaryIO | bytes, fallback_name: str = "telegram_export") -> pd.DataFrame:
    """Parse a text-only Telegram Desktop JSON export."""
    raw = file if isinstance(file, bytes) else file.read()
    payload = json.loads(raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw)
    return _parse_telegram_payload(payload, fallback_name)


def parse_telegram_archive(
    file: BinaryIO | bytes,
    fallback_name: str = "telegram_export",
    ocr_image: Callable[[bytes], str] = _ocr_signal_image,
    max_ocr_images: int = MAX_TELEGRAM_OCR_IMAGES,
) -> pd.DataFrame:
    """Parse a zipped Telegram export and OCR referenced images under strict limits."""
    raw = file if isinstance(file, bytes) else file.read()
    if len(raw) > MAX_TELEGRAM_ARCHIVE_BYTES:
        raise ValueError("Telegram ZIP exceeds the 500 MB safety limit.")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if sum(member.file_size for member in members) > MAX_TELEGRAM_ARCHIVE_BYTES:
            raise ValueError("Telegram ZIP expands beyond the 500 MB safety limit.")
        json_members = [
            member for member in members
            if not member.is_dir() and PurePosixPath(member.filename).name.lower() == "result.json"
        ]
        if len(json_members) != 1:
            raise ValueError("Telegram ZIP must contain exactly one result.json file.")
        json_member = json_members[0]
        root = str(PurePosixPath(json_member.filename).parent)
        root = "" if root == "." else root.rstrip("/") + "/"
        names = {member.filename.replace("\\", "/"): member for member in members}
        payload = json.loads(archive.read(json_member).decode("utf-8-sig"))

        def read_media(relative_path: str) -> bytes | None:
            member = names.get(root + relative_path)
            if member is None or member.file_size > MAX_TELEGRAM_IMAGE_BYTES:
                return None
            return archive.read(member)

        return _parse_telegram_payload(
            payload, fallback_name, media_reader=read_media,
            ocr_image=ocr_image, max_ocr_images=max_ocr_images)


def combine_telegram_exports(files) -> pd.DataFrame:
    frames = []
    for item in files or []:
        try:
            item.seek(0)
        except (AttributeError, OSError):
            pass
        name = getattr(item, "name", "telegram_export")
        if str(name).lower().endswith(".zip"):
            frames.append(parse_telegram_archive(item, name))
        else:
            frames.append(parse_telegram_export(item, name))
    if not frames:
        return _empty_community_frame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["channel", "message_id", "timestamp", "direction"])


_WHATSAPP_LINE = re.compile(
    r"^\[?(?P<date>\d{1,2}[/.]\d{1,2}[/.]\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?"
    r"\s*(?:-|–)?\s*(?P<sender>[^:]{1,100}):\s*(?P<text>.*)$")


def parse_whatsapp_export(
    file: BinaryIO | bytes,
    source_name: str = "whatsapp_export",
    dayfirst: bool = True,
) -> pd.DataFrame:
    """Parse an exported WhatsApp TXT chat without accessing a WhatsApp account."""
    raw = file if isinstance(file, bytes) else file.read()
    text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, bytes) else str(raw)
    records, current = [], None
    for line in text.splitlines():
        match = _WHATSAPP_LINE.match(line.strip())
        if match:
            if current:
                records.append(current)
            current = match.groupdict()
        elif current:
            current["text"] += "\n" + line
    if current:
        records.append(current)
    rows = []
    for number, record in enumerate(records):
        direction = _direction(record["text"])
        if not direction:
            continue
        timestamp = pd.to_datetime(record["date"], utc=True, errors="coerce", dayfirst=dayfirst)
        if pd.isna(timestamp):
            continue
        rows.append({
            "channel": f"WhatsApp · {source_name}", "message_id": str(number),
            "timestamp": timestamp, "direction": direction,
            "text": record["text"][:500], "evidence_type": "text",
        })
    return pd.DataFrame(rows, columns=COMMUNITY_COLUMNS)


def parse_bot_csv(file: BinaryIO | bytes, source_name: str = "bot_export") -> pd.DataFrame:
    """Parse a bot CSV with timestamp plus direction/signal or message/text columns."""
    raw = file if isinstance(file, bytes) else file.read()
    frame = pd.read_csv(io.BytesIO(raw) if isinstance(raw, bytes) else io.StringIO(str(raw)))
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    time_column = normalized.get("timestamp") or normalized.get("date") or normalized.get("datetime")
    direction_column = normalized.get("direction") or normalized.get("signal")
    text_column = normalized.get("text") or normalized.get("message")
    source_column = normalized.get("source") or normalized.get("bot")
    if time_column is None or (direction_column is None and text_column is None):
        raise ValueError("Bot CSV requires timestamp/date and direction/signal or text/message columns.")
    rows = []
    for number, row in frame.iterrows():
        raw_text = str(row[text_column]) if text_column is not None else ""
        raw_direction = str(row[direction_column]).upper() if direction_column is not None else ""
        if raw_direction in {"BUY", "LONG", "1", "+1"}:
            direction = 1
        elif raw_direction in {"SELL", "SHORT", "-1"}:
            direction = -1
        else:
            direction = _direction(raw_text)
        if not direction:
            continue
        timestamp = pd.to_datetime(row[time_column], utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue
        source = str(row[source_column]) if source_column is not None else source_name
        text = raw_text or f"{raw_direction} GOLD"
        rows.append({"channel": f"Bot · {source}", "message_id": str(number),
                     "timestamp": timestamp, "direction": direction, "text": text[:500],
                     "evidence_type": "text"})
    return pd.DataFrame(rows, columns=COMMUNITY_COLUMNS)


def combine_bot_exports(files=None) -> pd.DataFrame:
    frames = []
    for item in files or []:
        name = getattr(item, "name", "bot_export")
        try:
            item.seek(0)
        except (AttributeError, OSError):
            pass
        if str(name).lower().endswith(".csv"):
            frame = parse_bot_csv(item, name)
        else:
            frame = parse_telegram_export(item, name)
            if not frame.empty:
                frame["channel"] = "Bot · " + frame["channel"].astype(str)
        frames.append(frame)
    frames = [frame for frame in frames if not frame.empty]
    return (pd.concat(frames, ignore_index=True) if frames else
            _empty_community_frame())


def combine_community_exports(
    telegram_files=None,
    whatsapp_files=None,
    bot_files=None,
    dayfirst: bool = True,
) -> pd.DataFrame:
    frames = [combine_telegram_exports(telegram_files)]
    for item in whatsapp_files or []:
        try:
            item.seek(0)
        except (AttributeError, OSError):
            pass
        frames.append(parse_whatsapp_export(
            item, getattr(item, "name", "whatsapp_export"), dayfirst=dayfirst))
    frames.append(combine_bot_exports(bot_files))
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return _empty_community_frame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["channel", "message_id", "timestamp", "direction"])


def download_intraday_bundle(
    api_key: str,
    outputsize: int = 5000,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, str]]:
    """Gold plus optional cross-asset confirmation proxies for the one-hour model."""
    gold = download_market_data(api_key, outputsize)
    if not expected_market_open(gold.index[-1]):
        raise ValueError(
            f"Latest XAU/USD candle is {gold.index[-1]:%A %Y-%m-%d %H:%M UTC}, "
            "outside the accepted market session.")
    symbols = dict(CONFIRMATION_SYMBOLS)
    symbols.update(EXTRA_INTRADAY_PROXIES)
    confirmations, status = {}, {"XAU/USD": f"{len(gold):,} 15-minute candles"}
    for name, symbol in symbols.items():
        try:
            frame = _download_symbol(api_key, symbol, outputsize)
            age = abs((gold.index[-1] - frame.index[-1]).total_seconds()) / 60
            if age > 60:
                raise RuntimeError(f"latest candle is {age:.0f} minutes out of alignment")
            confirmations[name] = frame
            status[symbol] = f"{len(frame):,} 15-minute candles"
        except Exception as exc:
            status[symbol] = f"unavailable: {exc}"
    return gold, confirmations, status


def validate_market_data(
    gold: pd.DataFrame,
    confirmations: dict[str, pd.DataFrame],
    now: pd.Timestamp | None = None,
) -> dict[str, float | str]:
    """Validate required market data and arbitrary aligned optional proxies."""
    core = {name: frame for name, frame in confirmations.items() if name in CONFIRMATION_SYMBOLS}
    report = _validate_market_data_base(gold, core, now=now)
    as_of = gold.index[-1]
    for name, frame in confirmations.items():
        if name in core:
            continue
        if frame.empty:
            raise ValueError(f"No {name} candles were returned.")
        age = abs((as_of - frame.index[-1]).total_seconds()) / 60
        if age > 60:
            raise ValueError(f"{name} is not aligned with gold ({age:.0f}-minute difference).")
        report[name] = frame.index[-1].strftime("%Y-%m-%d %H:%M UTC")
    return report


def fit_intraday_system(
    gold: pd.DataFrame,
    confirmations: dict[str, pd.DataFrame],
    splits: int = 5,
    cost_bps: float = 10,
    threshold: float = 0.65,
):
    """Fit the leakage-aware next-hour system using four 15-minute bars."""
    return fit_and_backtest(
        gold,
        splits=splits,
        cost_bps=cost_bps,
        threshold=threshold,
        confirmations=confirmations,
        horizon_bars=INTRADAY_HORIZON_BARS,
    )


def intraday_release_reasons(result) -> list[str]:
    """Apply release gates to the one-hour walk-forward evidence."""
    reasons: list[str] = []
    metrics = result.metrics
    baseline = result.baseline_metrics
    if metrics.get("ROC-AUC", 0) < 0.53:
        reasons.append("one-hour ROC-AUC is below 0.53")
    if metrics.get("ROC-AUC", 0) < baseline.get("ROC-AUC", 0) + 0.01:
        reasons.append("confirmation features do not improve ROC-AUC by at least 0.01")
    if metrics.get("Strategy total return", 0) <= 0:
        reasons.append("one-hour walk-forward strategy return is not positive")
    if metrics.get("Strategy max drawdown", -1) < -0.10:
        reasons.append("one-hour walk-forward drawdown exceeds 10%")
    coverage = metrics.get("80% interval coverage", 0)
    if not 0.70 <= coverage <= 0.90:
        reasons.append("one-hour interval coverage is outside 70% to 90%")
    if metrics.get("Signal changes", 0) < 20:
        reasons.append("fewer than 20 non-overlapping signal changes")
    return reasons


def _wilson_lower(wins: int, total: int, z: float = 1.645) -> float:
    if total <= 0:
        return float("nan")
    p = wins / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (centre - margin) / denominator


def resolve_telegram_signals(messages: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Resolve each call from the next candle open to the close four bars later."""
    if messages.empty:
        return messages.assign(entry_time=pd.NaT, exit_time=pd.NaT, return_1h=np.nan, success=np.nan)
    rows = []
    index = gold.index
    for row in messages.itertuples(index=False):
        position = index.searchsorted(row.timestamp, side="left")
        exit_position = position + INTRADAY_HORIZON_BARS
        resolved = position < len(index) and exit_position < len(index)
        record = row._asdict()
        if resolved:
            entry = float(gold.iloc[position].open)
            exit_price = float(gold.iloc[exit_position].close)
            future_return = exit_price / entry - 1
            record.update({
                "entry_time": index[position], "exit_time": index[exit_position],
                "return_1h": future_return,
                "success": bool(row.direction * future_return > 0),
            })
        else:
            record.update({"entry_time": pd.NaT, "exit_time": pd.NaT,
                           "return_1h": np.nan, "success": np.nan})
        rows.append(record)
    return pd.DataFrame(rows)


def audit_telegram_signals(
    messages: pd.DataFrame,
    gold: pd.DataFrame,
    min_resolved: int = 30,
    min_win_rate: float = 0.52,
    min_lower_bound: float = 0.45,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved = resolve_telegram_signals(messages, gold)
    rows = []
    for channel, group in resolved.groupby("channel"):
        done = group.dropna(subset=["success"])
        total = len(done)
        wins = int(done.success.astype(bool).sum()) if total else 0
        win_rate = wins / total if total else np.nan
        lower = _wilson_lower(wins, total)
        qualified = bool(
            total >= min_resolved and win_rate >= min_win_rate and lower >= min_lower_bound)
        rows.append({
            "Channel": channel,
            "Resolved signals": total,
            "Win rate": win_rate,
            "90% Wilson lower bound": lower,
            "Qualified": qualified,
            "Weight": max(0.0, min(1.0, (win_rate - 0.5) * 5)) if qualified else 0.0,
        })
    audit = pd.DataFrame(rows)
    if audit.empty:
        audit = pd.DataFrame(columns=[
            "Channel", "Resolved signals", "Win rate", "90% Wilson lower bound",
            "Qualified", "Weight"])
    return audit, resolved


def apply_telegram_overlay(
    base_probability: float,
    median_return: float,
    threshold: float,
    cost_bps: float,
    as_of: pd.Timestamp,
    messages: pd.DataFrame,
    gold: pd.DataFrame,
) -> TelegramOverlay:
    audit, resolved = audit_telegram_signals(messages, gold)
    qualified = audit.loc[audit.Qualified, ["Channel", "Weight"]]
    recent = messages[
        (messages.timestamp <= as_of) &
        (messages.timestamp > as_of - pd.Timedelta(hours=1))
    ].copy()
    recent = recent.merge(qualified, left_on="channel", right_on="Channel", how="inner")
    recent = recent.sort_values("timestamp").drop_duplicates("channel", keep="last")
    reasons = []
    adjustment = 0.0
    consensus = "NEUTRAL"
    if qualified.empty:
        reasons.append("no community source has at least 30 resolved signals and passed reliability gates")
    elif recent.empty:
        reasons.append("no fresh qualified community gold signal in the last hour")
    else:
        score = float(np.average(recent.direction, weights=recent.Weight.clip(lower=.01)))
        adjustment = float(np.clip(score * 0.05, -0.05, 0.05))
        consensus = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "MIXED"
    probability = float(np.clip(base_probability + adjustment, 0.01, 0.99))
    return TelegramOverlay(
        probability_up=probability,
        adjustment=adjustment,
        decision=_signal(probability, median_return, cost_bps, threshold),
        consensus=consensus,
        qualified_channels=len(qualified),
        current_signals=len(recent),
        reasons=reasons,
        audit=audit,
        messages=resolved,
    )


def causal_elliott_states(gold: pd.DataFrame, pivot_window: int = 4) -> pd.DataFrame:
    """Causal Elliott-style swing states; pivots appear only after confirmation."""
    if pivot_window < 2:
        raise ValueError("pivot_window must be at least 2")
    highs, lows = gold.high.to_numpy(float), gold.low.to_numpy(float)
    pivots: list[dict] = []
    output = []
    for available in range(len(gold)):
        candidate = available - pivot_window
        if candidate >= pivot_window:
            left, right = candidate - pivot_window, available + 1
            is_high = highs[candidate] >= np.max(highs[left:right])
            is_low = lows[candidate] <= np.min(lows[left:right])
            if is_high != is_low:
                kind = "H" if is_high else "L"
                price = highs[candidate] if is_high else lows[candidate]
                pivot = {"kind": kind, "price": float(price), "at": candidate}
                if pivots and pivots[-1]["kind"] == kind:
                    better = price > pivots[-1]["price"] if kind == "H" else price < pivots[-1]["price"]
                    if better:
                        pivots[-1] = pivot
                else:
                    pivots.append(pivot)
        bias, structure = 0, "INSUFFICIENT SWINGS"
        if len(pivots) >= 4:
            recent = pivots[-4:]
            high_points = [p["price"] for p in recent if p["kind"] == "H"]
            low_points = [p["price"] for p in recent if p["kind"] == "L"]
            if len(high_points) == 2 and len(low_points) == 2:
                if high_points[1] > high_points[0] and low_points[1] > low_points[0]:
                    bias, structure = 1, "BULLISH IMPULSE/CORRECTION"
                elif high_points[1] < high_points[0] and low_points[1] < low_points[0]:
                    bias, structure = -1, "BEARISH IMPULSE/CORRECTION"
                else:
                    structure = "OVERLAPPING/RANGE"
        output.append({"timestamp": gold.index[available], "elliott_bias": bias,
                       "elliott_structure": structure, "confirmed_pivots": len(pivots)})
    return pd.DataFrame(output).set_index("timestamp")


def audit_elliott_overlay(
    gold: pd.DataFrame,
    min_observations: int = 80,
    min_accuracy: float = 0.53,
    min_lower_bound: float = 0.50,
    validation_fraction: float = 0.30,
) -> tuple[dict, pd.DataFrame]:
    """Audit Elliott evidence on a chronological, non-overlapping holdout.

    The newest labelled observations are reserved for validation so the
    qualification gate is not based on the same history used to establish
    the structure.  This is deliberately conservative: weak or unstable
    Elliott evidence must remain neutral.
    """
    if not 0.20 <= validation_fraction <= 0.50:
        raise ValueError("validation_fraction must be between 0.20 and 0.50")
    states = causal_elliott_states(gold)
    states["future_return"] = gold.close.shift(-INTRADAY_HORIZON_BARS) / gold.close - 1
    sample = states.iloc[::INTRADAY_HORIZON_BARS].copy()
    sample = sample[(sample.elliott_bias != 0) & sample.future_return.notna()]
    sample["success"] = sample.elliott_bias * sample.future_return > 0
    validation_size = max(min_observations, int(math.ceil(len(sample) * validation_fraction)))
    validation = sample.tail(min(len(sample), validation_size)).copy()
    total = len(validation)
    wins = int(validation.success.sum()) if total else 0
    accuracy = wins / total if total else np.nan
    lower = _wilson_lower(wins, total)
    qualified = bool(total >= min_observations and accuracy >= min_accuracy and lower >= min_lower_bound)
    audit = {"Observations": total, "Accuracy": accuracy,
             "90% Wilson lower bound": lower, "Qualified": qualified,
             "Validation start": validation.index.min() if total else pd.NaT,
             "Validation end": validation.index.max() if total else pd.NaT}
    return audit, states.join(sample[["success"]], how="left")


def apply_elliott_overlay(
    base_probability: float,
    median_return: float,
    threshold: float,
    cost_bps: float,
    gold: pd.DataFrame,
) -> ElliottOverlay:
    audit, history = audit_elliott_overlay(gold)
    latest = history.iloc[-1]
    bias = int(latest.elliott_bias)
    reasons, adjustment = [], 0.0
    if not audit["Qualified"]:
        reasons.append("causal Elliott structure has not passed its historical reliability gate")
    elif bias == 0:
        reasons.append("current swing structure is overlapping or incomplete")
    else:
        # Use the conservative confidence bound, not headline accuracy, to
        # determine influence.  A marginal pass therefore has little weight.
        strength = min(1.0, max(0.0, (audit["90% Wilson lower bound"] - 0.50) / 0.08))
        adjustment = float(bias * strength * 0.04)
    probability = float(np.clip(base_probability + adjustment, 0.01, 0.99))
    return ElliottOverlay(
        probability_up=probability, adjustment=adjustment,
        decision=_signal(probability, median_return, cost_bps, threshold),
        current_bias="BUY" if bias > 0 else "SELL" if bias < 0 else "NEUTRAL",
        current_structure=str(latest.elliott_structure),
        observations=int(audit["Observations"]), accuracy=float(audit["Accuracy"]),
        lower_bound=float(audit["90% Wilson lower bound"]), qualified=bool(audit["Qualified"]),
        reasons=reasons, history=history)
