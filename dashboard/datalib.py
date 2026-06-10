"""
datalib.py
----------
Strat de date pentru dashboard, aliniat la schema reala de eveniment din
DynamoDB (tabela PlasticDetectorEvents). NU depinde de Streamlit -> testabil.

Schema eveniment (ce trimite main.py prin aws_publisher.publish_event):
    device_id, source, action (COLLECT|ALARM|STOP), reason, detections_count,
    selected_label, selected_confidence, selected_area_ratio, selected_track_id,
    timestamp (ISO).

"Resetare zilnica" = grupare dupa data din timestamp. Nimic nu se sterge.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, date
from typing import Any

import pandas as pd

WASTE_LABELS = [
    "plastic-bottle", "plastic-bag", "plastic-cup", "plastic-wrapper",
    "plastic-other", "can", "carton", "foam", "other-waste",
]
ACTIVE_ACTIONS = ["COLLECT", "ALARM"]
HIGH_POLLUTION_THRESHOLD = 15   # = MAX_OBJECTS_BEFORE_ALARM din decision_logic

EVENT_COLUMNS = [
    "timestamp", "date", "time", "hour", "action", "reason", "detections_count",
    "selected_label", "selected_confidence", "selected_area_ratio",
    "selected_track_id", "device_id",
]


# --------------------------------------------------------------------------- #
# 1. Normalizare
# --------------------------------------------------------------------------- #
def events_to_dataframe(events: list[dict[str, Any]]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    df = pd.DataFrame(events)
    defaults = {
        "action": "STOP", "reason": "", "detections_count": 0,
        "selected_label": None, "selected_confidence": None,
        "selected_area_ratio": None, "selected_track_id": None,
        "device_id": "plastic-detector-01", "source": "laptop-ai",
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.strftime("%H:%M:%S")
    df["hour"] = df["timestamp"].dt.hour
    for col in ["detections_count", "selected_confidence", "selected_area_ratio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["detections_count"] = df["detections_count"].fillna(0).astype(int)

    return (df[EVENT_COLUMNS]
            .sort_values("timestamp", ascending=False)
            .reset_index(drop=True))


# --------------------------------------------------------------------------- #
# 2. KPI-uri generice + perioade + variatii
# --------------------------------------------------------------------------- #
def _compute_kpis(d: pd.DataFrame) -> dict[str, Any]:
    active = d[d["action"].isin(ACTIVE_ACTIONS)]
    collected = d[d["action"] == "COLLECT"]
    alarms = d[d["action"] == "ALARM"]
    n_active = len(active)
    return {
        "events": len(d),
        "collected": len(collected),
        "alarms": len(alarms),
        "alarms_large": int((alarms["reason"] == "object_too_large").sum()),
        "alarms_pollution": int((alarms["reason"] == "high_pollution_level").sum()),
        "collection_rate": round(100 * len(collected) / n_active, 1) if n_active else 0.0,
        "avg_confidence": round(float(active["selected_confidence"].mean()), 2) if n_active else 0.0,
        "peak_load": int(d["detections_count"].max()) if len(d) else 0,
        "avg_load": round(float(d["detections_count"].mean()), 1) if len(d) else 0.0,
    }


def kpis_for_day(df: pd.DataFrame, day: date) -> dict[str, Any]:
    return _compute_kpis(df[df["date"] == day])


def period_slice(df: pd.DataFrame, end_day: date, days: int) -> pd.DataFrame:
    start = end_day - timedelta(days=days - 1)
    return df[(df["date"] >= start) & (df["date"] <= end_day)]


def kpis_with_delta(df: pd.DataFrame, end_day: date, days: int):
    current = period_slice(df, end_day, days)
    prev = period_slice(df, end_day - timedelta(days=days), days)
    return _compute_kpis(current), _compute_kpis(prev)


def pct_change(now: float, before: float):
    if before == 0:
        return None
    return round(100 * (now - before) / before, 0)


# --------------------------------------------------------------------------- #
# 3. Agregari pentru grafice
# --------------------------------------------------------------------------- #
def waste_by_type(d: pd.DataFrame) -> pd.DataFrame:
    a = d[d["action"].isin(ACTIVE_ACTIONS)].dropna(subset=["selected_label"])
    out = a.groupby("selected_label").size().reset_index(name="count")
    return out.sort_values("count", ascending=False).reset_index(drop=True)


def alarm_reasons(d: pd.DataFrame) -> dict[str, int]:
    a = d[d["action"] == "ALARM"]
    return {
        "Deseu prea mare": int((a["reason"] == "object_too_large").sum()),
        "Poluare ridicata": int((a["reason"] == "high_pollution_level").sum()),
    }


def daily_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "Colectate", "Alarme"])
    rows = []
    for day, g in df.groupby("date"):
        rows.append({
            "date": day,
            "Colectate": int((g["action"] == "COLLECT").sum()),
            "Alarme": int((g["action"] == "ALARM").sum()),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def hourly_activity(d: pd.DataFrame) -> pd.DataFrame:
    a = d[d["action"].isin(ACTIVE_ACTIONS)]
    counts = a.groupby("hour").size()
    return pd.DataFrame({"hour": list(range(24)),
                         "count": [int(counts.get(h, 0)) for h in range(24)]})


def confidence_values(d: pd.DataFrame) -> list[float]:
    a = d[d["action"].isin(ACTIVE_ACTIONS)]
    return a["selected_confidence"].dropna().tolist()


def system_stats(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"total_events": 0, "active_days": 0, "last_seen": "—",
                "devices": [], "collected_all": 0, "alarms_all": 0}
    return {
        "total_events": len(df),
        "active_days": int(df["date"].nunique()),
        "last_seen": df["timestamp"].max().strftime("%d %b %Y, %H:%M:%S"),
        "devices": sorted(df["device_id"].dropna().unique().tolist()),
        "collected_all": int((df["action"] == "COLLECT").sum()),
        "alarms_all": int((df["action"] == "ALARM").sum()),
    }


# --------------------------------------------------------------------------- #
# 4. Date de DEMO (~5 saptamani, ca sa mearga si perioada de 30 zile)
# --------------------------------------------------------------------------- #
def generate_demo_data(days: int = 35, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    events: list[dict] = []
    today = datetime.now()

    for d in range(days):
        day = today - timedelta(days=d)
        base = 28 if d < 7 else 16
        n = rng.randint(base - 8, base + 12)
        for _ in range(n):
            ts = day.replace(hour=min(23, max(0, int(rng.gauss(13, 4)))),
                             minute=rng.randint(0, 59), second=rng.randint(0, 59),
                             microsecond=0)
            roll = rng.random()
            label = rng.choices(WASTE_LABELS,
                                weights=[6, 4, 3, 3, 2, 4, 2, 2, 2])[0]
            count = rng.randint(1, 6)
            area = round(rng.uniform(0.02, 0.12), 3)

            if roll < 0.66:
                action, reason = "COLLECT", "collectable_garbage_detected"
            elif roll < 0.82:
                action, reason = "ALARM", "object_too_large"
                area = round(rng.uniform(0.16, 0.34), 3)
            elif roll < 0.90:
                action, reason = "ALARM", "high_pollution_level"
                count = rng.randint(15, 26)
            else:
                action, reason = "STOP", "no_garbage_detected"

            ev = {
                "device_id": rng.choice(["plastic-detector-01"] * 5 + ["plastic-detector-02"]),
                "source": "laptop-ai",
                "action": action, "reason": reason,
                "detections_count": count, "timestamp": ts.isoformat(),
            }
            if action != "STOP":
                ev.update({
                    "selected_label": label,
                    "selected_confidence": round(rng.uniform(0.55, 0.97), 2),
                    "selected_area_ratio": area,
                    "selected_track_id": rng.randint(1, 300),
                })
            events.append(ev)
    return events


def daily_series(df: pd.DataFrame, end_day: date, days: int, metric: str) -> list:
    """Valoarea unui KPI pe fiecare zi (vechi -> nou), pentru sparkline-uri."""
    out = []
    for i in range(days - 1, -1, -1):
        day = end_day - timedelta(days=i)
        out.append(_compute_kpis(df[df["date"] == day]).get(metric, 0))
    return out


def load_live_events(path) -> list[dict]:
    """Citeste evenimentele scrise de live_camera.py (fisier JSON Lines)."""
    import json
    import os
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out