import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

"""
The audit log in server.py (runtime/predictions.jsonl, via append_log) records
every full API response for debugging, but nothing ever checks those
predictions against what actually happened -- there's no real, tracked win
rate. This module adds that missing piece:

  log_prediction(...)         -> append one row per horizon to pending.jsonl
  resolve_due_predictions(...) -> for any pending row whose target_time has
                                   passed, look up the realized close price
                                   in the freshly-fetched closed-candle window
                                   and move it to resolved.jsonl with the
                                   actual outcome attached
  live_accuracy_stats()       -> aggregate resolved.jsonl into a real,
                                   growing win-rate per horizon

File-based and single-process by design (matches --workers 1 used for the
stabilizer). If this is ever scaled to multiple workers/instances, this
should move to a shared DB instead of local disk.
"""

DATA_DIR = Path("runtime")
PENDING_PATH = DATA_DIR / "pending.jsonl"
RESOLVED_PATH = DATA_DIR / "resolved.jsonl"
EXPIRE_DAYS = 14
RESOLVE_TOLERANCE_MIN = 90


def _read_jsonl(path):
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(path, rows):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _append_jsonl(path, row):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_prediction(horizon_hours, direction, probability_up, entry_price, now, target_time):
    row = {
        "id": uuid.uuid4().hex[:12],
        "created_at": now.isoformat(),
        "horizon_hours": horizon_hours,
        "direction": direction,
        "probability_up": probability_up,
        "entry_price": entry_price,
        "target_time": target_time.isoformat(),
    }
    _append_jsonl(PENDING_PATH, row)
    return row


def resolve_due_predictions(closed_candles_df, now):
    pending = _read_jsonl(PENDING_PATH)
    if not pending or closed_candles_df is None or len(closed_candles_df) == 0:
        return

    cdf = closed_candles_df[["t", "c"]].copy()
    cdf["t"] = pd.to_numeric(cdf["t"], errors="coerce")
    cdf = cdf.dropna().sort_values("t")
    window_start_ms = int(cdf["t"].iloc[0])
    window_end_ms = int(cdf["t"].iloc[-1])

    still_pending, newly_resolved, expired = [], [], []

    for row in pending:
        target_time = datetime.fromisoformat(row["target_time"])
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)
        target_ms = int(target_time.timestamp() * 1000)

        if target_time > now:
            still_pending.append(row)
            continue

        tol_ms = RESOLVE_TOLERANCE_MIN * 60_000
        if window_start_ms - tol_ms <= target_ms <= window_end_ms + tol_ms:
            idx = (cdf["t"] - target_ms).abs().idxmin()
            nearest_t = float(cdf.loc[idx, "t"])
            if abs(nearest_t - target_ms) <= tol_ms:
                actual_price = float(cdf.loc[idx, "c"])
                entry_price = float(row["entry_price"])
                actual_return_pct = (actual_price / entry_price - 1.0) * 100.0
                actual_direction = "LONG" if actual_price > entry_price else "SHORT"
                row = dict(row)
                row.update({
                    "resolved_at": now.isoformat(),
                    "actual_price": actual_price,
                    "actual_return_pct": round(actual_return_pct, 3),
                    "actual_direction": actual_direction,
                    "correct": bool(actual_direction == row["direction"]),
                })
                newly_resolved.append(row)
                continue

        created_at = datetime.fromisoformat(row["created_at"])
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if (now - created_at).total_seconds() / 86400.0 > EXPIRE_DAYS:
            row = dict(row)
            row["expired_no_data"] = True
            expired.append(row)
        else:
            still_pending.append(row)

    if newly_resolved or expired:
        for r in newly_resolved + expired:
            _append_jsonl(RESOLVED_PATH, r)
        _write_jsonl(PENDING_PATH, still_pending)


def live_accuracy_stats():
    resolved = [r for r in _read_jsonl(RESOLVED_PATH) if not r.get("expired_no_data")]
    if not resolved:
        return {}
    df = pd.DataFrame(resolved)
    out = {}
    for h, g in df.groupby("horizon_hours"):
        long_g = g[g["direction"] == "LONG"]
        short_g = g[g["direction"] == "SHORT"]
        out[f"h{h}"] = {
            "n": int(len(g)),
            "win_rate_pct": round(float(g["correct"].mean()) * 100, 1),
            "avg_actual_return_pct": round(float(g["actual_return_pct"].mean()), 3),
            "long_n": int(len(long_g)),
            "long_win_rate_pct": round(float(long_g["correct"].mean()) * 100, 1) if len(long_g) else None,
            "short_n": int(len(short_g)),
            "short_win_rate_pct": round(float(short_g["correct"].mean()) * 100, 1) if len(short_g) else None,
        }
    return out
