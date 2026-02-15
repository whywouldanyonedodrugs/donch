#!/usr/bin/env python3
"""
Daily Autopar export package builder for live-vs-backtest parity checks.

Creates:
  - live_decisions.csv   (required)
  - live_trades.csv      (recommended)
  - symbols_active.txt   (recommended)
  - run_context.json     (recommended)
  - live.log             (required; raw service logs for window)
  - settings_snapshot.json (required)
  - schema_diagnostics.json (required)

Default output:
  results/autopar_exports/autopar_YYYY-MM-DD/
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import yaml

try:
    import ccxt
except Exception:  # pragma: no cover
    ccxt = None

try:
    import psycopg2
    import psycopg2.extras
except Exception:  # pragma: no cover
    psycopg2 = None


META_DECISION_RE = re.compile(
    r"META_DECISION\s+"
    r"bundle=(?P<bundle>\S+)\s+"
    r"symbol=(?P<symbol>\S+)\s+"
    r"decision_ts=(?P<decision_ts>\S+)\s+"
    r"schema_ok=(?P<schema_ok>\S+)\s+"
    r"p_cal=(?P<p_cal>\S+)\s+"
    r"pstar=(?P<pstar>\S+)\s+"
    r"pstar_scope=(?P<pstar_scope>\S+)\s+"
    r"risk_on_1=(?P<risk_on_1>\S+)\s+"
    r"risk_on=(?P<risk_on>\S+)\s+"
    r"scope_val=(?P<scope_val>\S+)\s+"
    r"scope_src=(?P<scope_src>\S+)\s+"
    r"scope_ok=(?P<scope_ok>\S+)\s+"
    r"meta_ok=(?P<meta_ok>\S+)\s+"
    r"strat_ok=(?P<strat_ok>\S+)\s+"
    r"reason=(?P<reason>\S+)\s+"
    r"err=(?P<err>.*)$"
)


DECISION_COLUMNS = [
    "symbol",
    "decision_ts",
    "decision",
    "reason",
    "reason_raw",
    "reason_canonical",
    "schema_fail_class",
    "symbol_listed_at_utc",
    "symbol_age_days",
    "err",
    "bundle",
    "schema_ok",
    "p_cal",
    "pstar",
    "pstar_scope",
    "scope_ok",
    "meta_ok",
    "strat_ok",
    "size_mult",
    "risk_usd",
    "risk_on",
]


SYMBOL_METADATA_COLUMNS = [
    "symbol",
    "symbol_listed_at_utc",
    "symbol_age_days",
    "metadata_source",
]


REASON_MAPPING_COLUMNS = [
    "reason_raw",
    "reason_canonical",
    "count",
]


TRADES_COLUMNS = [
    "symbol",
    "opened_at",
    "closed_at",
    "pnl",
    "size",
    "entry_price",
    "exit_reason",
    "risk_usd",
    "win_probability_at_entry",
]


@dataclass(frozen=True)
class ExportWindow:
    start_utc: datetime
    end_utc: datetime
    label: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc_iso(dt: datetime | None) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def parse_date_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def build_window(day_str: str | None, start_str: str | None, end_str: str | None) -> ExportWindow:
    if start_str or end_str:
        if not (start_str and end_str):
            raise ValueError("Both --window-start-utc and --window-end-utc are required together.")
        start = datetime.fromisoformat(start_str.replace("Z", "+00:00")).astimezone(timezone.utc)
        end = datetime.fromisoformat(end_str.replace("Z", "+00:00")).astimezone(timezone.utc)
        if end <= start:
            raise ValueError("window_end_utc must be greater than window_start_utc.")
        label = start.date().isoformat()
        return ExportWindow(start_utc=start, end_utc=end, label=label)

    if day_str:
        d = parse_date_yyyy_mm_dd(day_str)
    else:
        d = utc_now().date()
    start = datetime.combine(d, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return ExportWindow(start_utc=start, end_utc=end, label=d.isoformat())


def parse_bool_token(s: str) -> bool | None:
    x = (s or "").strip().lower()
    if x in {"true", "1", "yes", "on"}:
        return True
    if x in {"false", "0", "no", "off"}:
        return False
    return None


def parse_float_token(s: str) -> float | None:
    x = (s or "").strip()
    if x == "" or x.lower() in {"none", "null", "nan", "na"}:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if v != v:  # NaN check
        return None
    return v


def parse_int_token(s: str) -> int | None:
    f = parse_float_token(s)
    if f is None:
        return None
    return int(round(f))


def parse_timestamp_utc(value: str) -> datetime | None:
    x = (value or "").strip()
    if not x:
        return None
    try:
        ts = datetime.fromisoformat(x.replace("Z", "+00:00"))
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def norm_none_token(s: str) -> str:
    x = (s or "").strip()
    if x.lower() in {"none", "null", "nan"}:
        return ""
    return x


def norm_err_token(s: str) -> str:
    x = (s or "").strip()
    if x.lower() in {"none", "null", "nan"}:
        return ""
    return x


def parse_bool_env(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    x = str(v).strip().lower()
    if x in {"1", "true", "yes", "on"}:
        return True
    if x in {"0", "false", "no", "off"}:
        return False
    return default


def canonicalize_reason(
    reason_raw: str,
    decision: str,
    schema_ok: bool | None,
    meta_ok: bool | None,
    strat_ok: bool | None,
    err: str,
) -> str:
    """
    Canonical taxonomy for parity metrics.
    """
    rr = str(reason_raw or "").strip()
    rl = rr.lower()
    err_l = str(err or "").lower()
    is_skipped = str(decision or "").strip().lower() == "skipped"

    if (schema_ok is False) or rl == "schema_fail" or "missing_required" in err_l:
        return "schema_fail"

    if "below_pstar" in rl:
        return "meta_prob"

    if rl.startswith("scope_") or "scope_fail" in rl or "scope_error" in rl:
        return "meta_scope"

    if "regime_slope" in rl:
        return "regime_slope_down"
    if "regime_down" in rl or rl == "entry_skip_regime":
        return "regime_down"
    if "atr_too_small" in rl:
        return "atr_too_small"
    if "atr_invalid" in rl:
        return "atr_invalid"
    if "dedup" in rl:
        return "dedup_entry"
    if "in_position" in rl or "already_in_position" in rl:
        return "in_position"
    if "cooldown" in rl:
        return "cooldown"
    if "daycap" in rl or "daily_cap" in rl:
        return "daycap"
    if "max_open" in rl:
        return "max_open_positions"
    if "no_5m_data_after_signal" in rl:
        return "no_5m_data_after_signal"
    if "simulation_none" in rl:
        return "simulation_none"
    if "error" in rl or "exception" in rl or "internal" in rl:
        return "internal_error"

    if rl in {"ok", "no_prob_gate", "prob_invalid_no_gate", "meta_disabled", "scope_bypass"}:
        if is_skipped and (meta_ok is True) and (strat_ok is False):
            return "strategy_fail"
        return "ok"

    if is_skipped and (meta_ok is True) and (strat_ok is False):
        return "strategy_fail"
    if is_skipped and (meta_ok is False):
        return "meta_prob"

    return "other"


def parse_exchange_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        ts = value
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    s = str(value).strip()
    if not s:
        return None

    # Numeric epoch (s/ms/us/ns)
    try:
        num = float(s)
        if num > 0:
            # normalize to seconds by magnitude
            if num > 1e18:
                num = num / 1e9
            elif num > 1e15:
                num = num / 1e6
            elif num > 1e12:
                num = num / 1e3
            ts = datetime.fromtimestamp(num, tz=timezone.utc)
            return ts
    except Exception:
        pass

    # ISO-like fallback
    try:
        ts = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    except Exception:
        return None


def _canonical_symbol_id(market: dict[str, Any]) -> str:
    sid = str(market.get("id") or "").upper().strip()
    if sid:
        sid = sid.replace("/", "").split(":")[0]
    if not sid:
        base = str(market.get("base") or "").upper().strip()
        quote = str(market.get("quote") or "").upper().strip()
        if base and quote:
            sid = f"{base}{quote}"
    return sid


def extract_symbol_listing_ts(market: dict[str, Any]) -> datetime | None:
    info = market.get("info") if isinstance(market.get("info"), dict) else {}
    candidates = [
        market.get("launchTime"),
        market.get("listTime"),
        market.get("onboardDate"),
        market.get("created"),
        market.get("createdTime"),
        info.get("launchTime"),
        info.get("listTime"),
        info.get("onboardDate"),
        info.get("onboardDateTime"),
        info.get("createdTime"),
        info.get("deliveryTime"),
    ]
    for v in candidates:
        ts = parse_exchange_timestamp(v)
        if ts is not None:
            return ts
    return None


def resolve_bybit_testnet(env_map: dict[str, str], mode: str) -> bool:
    x = str(mode or "auto").strip().lower()
    if x == "true":
        return True
    if x == "false":
        return False
    return parse_bool_env(env_map.get("BYBIT_TESTNET"), default=False)


def fetch_symbol_listing_metadata(
    symbols: list[str],
    *,
    bybit_testnet: bool,
    timeout_ms: int = 15000,
) -> tuple[dict[str, dict[str, Any]], str | None]:
    """
    Fetch objective listing metadata from exchange instrument metadata.
    """
    if ccxt is None:
        return {}, "ccxt unavailable"
    if not symbols:
        return {}, None

    try:
        ex = ccxt.bybit(
            {
                "enableRateLimit": True,
                "timeout": int(timeout_ms),
                "options": {"defaultType": "swap"},
                "testnet": bool(bybit_testnet),
            }
        )
        markets = ex.load_markets()
    except Exception as e:
        return {}, f"load_markets_failed:{type(e).__name__}:{e}"

    wanted = {str(s).upper().strip() for s in symbols if str(s).strip()}
    out: dict[str, dict[str, Any]] = {}

    for m in markets.values():
        try:
            sid = _canonical_symbol_id(m)
            if sid not in wanted:
                continue
            ts = extract_symbol_listing_ts(m)
            out[sid] = {
                "symbol": sid,
                "symbol_listed_at_utc": as_utc_iso(ts) if ts is not None else "",
                "listed_at_dt": ts,
                "metadata_source": "exchange_instrument_metadata" if ts is not None else "exchange_instrument_metadata_missing",
            }
        except Exception:
            continue

    # Ensure every requested symbol has a row.
    for s in sorted(wanted):
        if s not in out:
            out[s] = {
                "symbol": s,
                "symbol_listed_at_utc": "",
                "listed_at_dt": None,
                "metadata_source": "exchange_instrument_metadata_unavailable",
            }

    return out, None


def compute_symbol_age_days(listed_at_dt: datetime | None, asof_dt: datetime | None) -> int | None:
    if listed_at_dt is None or asof_dt is None:
        return None
    try:
        delta = asof_dt - listed_at_dt
    except Exception:
        return None
    days = int(delta.total_seconds() // 86400)
    if days < 0:
        return 0
    return days


def classify_schema_fail(
    reason_raw: str,
    schema_ok: bool | None,
    err: str,
    symbol_age_days: int | None,
    warmup_max_age_days: int,
) -> str:
    """
    Classify schema failures into warmup vs integration defect.
    """
    rr = str(reason_raw or "").strip().lower()
    is_schema_fail = (schema_ok is False) or (rr == "schema_fail")
    if not is_schema_fail:
        return "unknown"

    missing = parse_missing_fields_from_err(err)
    if not missing:
        return "unknown"

    warmup_fields = {"days_since_prev_break", "S6_fresh_x_compress"}
    only_warmup = all(f in warmup_fields for f in missing)
    if only_warmup:
        if symbol_age_days is None:
            return "unknown"
        return "warmup_expected" if int(symbol_age_days) <= int(warmup_max_age_days) else "integration_defect"

    return "integration_defect"


def parse_missing_fields_from_err(err: str) -> list[str]:
    """
    Parse missing required feature names from strict-schema error payloads.

    Expected shape:
      missing_required:['col_a', 'col_b']
    """
    if not err:
        return []
    raw = str(err).strip()
    if "missing_required" not in raw:
        return []

    # Best-effort extraction of the payload after `missing_required`.
    m = re.search(r"missing_required\s*[:=]\s*(.+)$", raw)
    if not m:
        return []
    payload = m.group(1).strip()
    if not payload:
        return []

    # Preferred: python-list payload from current logger formatting.
    if payload.startswith("["):
        try:
            parsed = ast.literal_eval(payload)
            if isinstance(parsed, list):
                out = []
                for v in parsed:
                    s = str(v).strip()
                    if s:
                        out.append(s)
                return out
        except Exception:
            pass

    # Fallback tokenizer for non-list payloads.
    toks = re.findall(r"[A-Za-z0-9_]+", payload)
    return [t for t in toks if t and t.lower() not in {"missing", "required"}]


def build_schema_diagnostics(decision_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Daily schema diagnostics for offline autopar monitoring.
    """
    decisions_count = len(decision_rows)
    schema_fail_rows: list[dict[str, Any]] = []
    for r in decision_rows:
        reason = str(r.get("reason_raw", r.get("reason", "")) or "").strip().lower()
        schema_ok = r.get("schema_ok", None)
        if reason == "schema_fail" or schema_ok is False:
            schema_fail_rows.append(r)

    schema_fail_count = len(schema_fail_rows)
    schema_fail_rate = (float(schema_fail_count) / float(decisions_count)) if decisions_count > 0 else 0.0

    symbol_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    non_warmup_counts: Counter[str] = Counter()
    warmup_only_count = 0

    expected_warmup_fields = {
        "days_since_prev_break",
        "S6_fresh_x_compress",
    }
    class_counts: Counter[str] = Counter()
    missing_by_class: dict[str, Counter[str]] = {
        "warmup_expected": Counter(),
        "integration_defect": Counter(),
        "unknown": Counter(),
    }
    symbols_by_class: dict[str, Counter[str]] = {
        "warmup_expected": Counter(),
        "integration_defect": Counter(),
        "unknown": Counter(),
    }

    for r in schema_fail_rows:
        sym = str(r.get("symbol", "") or "").upper().strip()
        if sym:
            symbol_counts[sym] += 1

        cls_raw = str(r.get("schema_fail_class", "unknown") or "unknown").strip()
        cls = cls_raw if cls_raw in {"warmup_expected", "integration_defect", "unknown"} else "unknown"
        class_counts[cls] += 1
        if sym:
            symbols_by_class[cls][sym] += 1

        err = str(r.get("err", "") or "")
        fields = parse_missing_fields_from_err(err)
        if not fields:
            continue

        for f in fields:
            missing_counts[f] += 1
            missing_by_class[cls][f] += 1
        non_warm = [f for f in fields if f not in expected_warmup_fields]
        if non_warm:
            for f in non_warm:
                non_warmup_counts[f] += 1
        else:
            warmup_only_count += 1

    top_missing_fields = [{"field": k, "count": int(v)} for k, v in missing_counts.most_common(20)]
    top_affected_symbols = [{"symbol": k, "count": int(v)} for k, v in symbol_counts.most_common(20)]
    top_non_warmup_missing_fields = [
        {"field": k, "count": int(v)} for k, v in non_warmup_counts.most_common(20)
    ]
    top_missing_fields_by_class = {
        cls: [{"field": k, "count": int(v)} for k, v in cnt.most_common(20)]
        for cls, cnt in missing_by_class.items()
    }
    top_affected_symbols_by_class = {
        cls: [{"symbol": k, "count": int(v)} for k, v in cnt.most_common(20)]
        for cls, cnt in symbols_by_class.items()
    }
    class_counts_out = {
        "warmup_expected": int(class_counts.get("warmup_expected", 0)),
        "integration_defect": int(class_counts.get("integration_defect", 0)),
        "unknown": int(class_counts.get("unknown", 0)),
    }

    if non_warmup_counts:
        max_non_warmup_count = int(max(non_warmup_counts.values()))
    else:
        max_non_warmup_count = 0

    incident_recommended = max_non_warmup_count >= 3

    return {
        "decisions_count": decisions_count,
        "schema_fail_count": schema_fail_count,
        "schema_fail_rate": schema_fail_rate,
        "top_missing_fields": top_missing_fields,
        "top_affected_symbols": top_affected_symbols,
        "expected_warmup_missing_fields": sorted(expected_warmup_fields),
        "warmup_only_schema_fail_count": warmup_only_count,
        "non_warmup_schema_fail_count": int(max(0, schema_fail_count - warmup_only_count)),
        "top_non_warmup_missing_fields": top_non_warmup_missing_fields,
        "schema_fail_class_counts": class_counts_out,
        "top_missing_fields_by_class": top_missing_fields_by_class,
        "top_affected_symbols_by_class": top_affected_symbols_by_class,
        "incident_recommended": incident_recommended,
        "incident_rule": "recommended when non-warmup missing field repeats >= 3 rows/day",
    }


def parse_meta_decision_lines(lines: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Parse META_DECISION logs to normalized decision rows.

    Dedup policy:
      - for duplicate (symbol, decision_ts), keep latest row encountered.
    """
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    kept_lines: list[str] = []

    for line in lines:
        if "META_DECISION" not in line:
            continue
        m = META_DECISION_RE.search(line)
        if not m:
            continue
        kept_lines.append(line.rstrip("\n"))
        g = m.groupdict()

        symbol = g["symbol"].upper().strip()
        try:
            decision_ts = datetime.fromisoformat(g["decision_ts"].replace("Z", "+00:00"))
            decision_ts_s = as_utc_iso(decision_ts)
        except Exception:
            decision_ts_s = g["decision_ts"].strip()

        schema_ok = parse_bool_token(g["schema_ok"])
        scope_ok = parse_bool_token(g["scope_ok"])
        meta_ok = parse_bool_token(g["meta_ok"])
        strat_ok = parse_bool_token(g["strat_ok"])

        decision = "taken" if (meta_ok is True and strat_ok is True) else "skipped"
        reason_raw = g["reason"]
        reason_canonical = canonicalize_reason(
            reason_raw=reason_raw,
            decision=decision,
            schema_ok=schema_ok,
            meta_ok=meta_ok,
            strat_ok=strat_ok,
            err=g["err"],
        )

        row = {
            "symbol": symbol,
            "decision_ts": decision_ts_s,
            "decision": decision,
            "reason": reason_raw,
            "reason_raw": reason_raw,
            "reason_canonical": reason_canonical,
            "schema_fail_class": "unknown",
            "symbol_listed_at_utc": "",
            "symbol_age_days": "",
            "err": norm_err_token(g["err"]),
            "bundle": norm_none_token(g["bundle"]),
            "schema_ok": schema_ok,
            "p_cal": parse_float_token(g["p_cal"]),
            "pstar": parse_float_token(g["pstar"]),
            "pstar_scope": norm_none_token(g["pstar_scope"]),
            "scope_ok": scope_ok,
            "meta_ok": meta_ok,
            "strat_ok": strat_ok,
            "size_mult": "",
            "risk_usd": "",
            "risk_on": parse_int_token(g["risk_on"]),
        }
        dedup[(symbol, decision_ts_s)] = row

    rows = list(dedup.values())
    rows.sort(key=lambda r: (r["decision_ts"], r["symbol"]))
    return rows, kept_lines


def load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.split("#", 1)[0].strip()
    return out


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def config_hash_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strategy_version(config: dict[str, Any], repo_root: Path) -> str:
    rel = str(config.get("STRATEGY_SPEC_PATH", "") or "").strip()
    if not rel:
        return ""
    p = (repo_root / rel).resolve()
    if not p.exists():
        return rel
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    return f"{rel}@{h}"


def load_symbols_active(symbols_path: Path, invalid_symbols_path: Path) -> list[str]:
    if not symbols_path.exists():
        return []

    symbols: list[str] = []
    for raw in symbols_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip().upper()
        if line:
            symbols.append(line)

    invalid: set[str] = set()
    if invalid_symbols_path.exists():
        for raw in invalid_symbols_path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip().upper()
            if line:
                invalid.add(line)

    return [s for s in symbols if s not in invalid]


def run_journalctl(service: str, start_utc: datetime, end_utc: datetime) -> list[str]:
    since = start_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    until = end_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    cmd = [
        "journalctl",
        "-u",
        service,
        "--since",
        since,
        "--until",
        until,
        "--no-pager",
        "-o",
        "cat",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"journalctl failed rc={proc.returncode}: {stderr}")
    return proc.stdout.splitlines()


def fetch_trades_with_timeout(
    dsn: str,
    start_utc: datetime,
    end_utc: datetime,
    timeout_sec: float,
) -> tuple[list[dict[str, Any]], str | None]:
    if psycopg2 is None:
        return [], "psycopg2 is unavailable in current Python environment"

    q = """
        SELECT
            symbol,
            opened_at,
            closed_at,
            pnl,
            size,
            entry_price,
            exit_reason,
            risk_usd,
            win_probability_at_entry
        FROM positions
        WHERE status='CLOSED'
          AND closed_at >= %s
          AND closed_at < %s
        ORDER BY closed_at ASC
    """
    timeout_i = max(1, int(round(timeout_sec)))
    conn = None
    try:
        conn = psycopg2.connect(
            dsn=dsn,
            connect_timeout=timeout_i,
            options=f"-c statement_timeout={int(timeout_sec * 1000)}",
        )
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(q, (start_utc, end_utc))
            recs = cur.fetchall()

        rows: list[dict[str, Any]] = []
        for r in recs:
            row = {
                "symbol": str(r.get("symbol") or "").upper(),
                "opened_at": as_utc_iso(r.get("opened_at")),
                "closed_at": as_utc_iso(r.get("closed_at")),
                "pnl": float(r["pnl"]) if r.get("pnl") is not None else "",
                "size": float(r["size"]) if r.get("size") is not None else "",
                "entry_price": float(r["entry_price"]) if r.get("entry_price") is not None else "",
                "exit_reason": str(r.get("exit_reason") or ""),
                "risk_usd": float(r["risk_usd"]) if r.get("risk_usd") is not None else "",
                "win_probability_at_entry": (
                    float(r["win_probability_at_entry"])
                    if r.get("win_probability_at_entry") is not None
                    else ""
                ),
            }
            rows.append(row)
        return rows, None
    except Exception as e:
        return [], str(e)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def enrich_decision_rows(
    decision_rows: list[dict[str, Any]],
    symbol_meta: dict[str, dict[str, Any]],
    warmup_max_age_days: int,
) -> None:
    for r in decision_rows:
        sym = str(r.get("symbol", "") or "").upper().strip()
        md = symbol_meta.get(sym, {})
        listed_at_dt = md.get("listed_at_dt")
        listed_at_s = str(md.get("symbol_listed_at_utc", "") or "")
        dt = parse_timestamp_utc(str(r.get("decision_ts", "") or ""))
        age_days = compute_symbol_age_days(listed_at_dt, dt)

        reason_raw = str(r.get("reason_raw", r.get("reason", "")) or "")
        reason_canonical = canonicalize_reason(
            reason_raw=reason_raw,
            decision=str(r.get("decision", "") or ""),
            schema_ok=r.get("schema_ok", None),
            meta_ok=r.get("meta_ok", None),
            strat_ok=r.get("strat_ok", None),
            err=str(r.get("err", "") or ""),
        )
        r["reason"] = reason_raw
        r["reason_raw"] = reason_raw
        r["reason_canonical"] = reason_canonical
        r["symbol_listed_at_utc"] = listed_at_s
        r["symbol_age_days"] = age_days if age_days is not None else ""
        r["schema_fail_class"] = classify_schema_fail(
            reason_raw=reason_raw,
            schema_ok=r.get("schema_ok", None),
            err=str(r.get("err", "") or ""),
            symbol_age_days=age_days,
            warmup_max_age_days=warmup_max_age_days,
        )


def build_symbol_metadata_rows(
    symbols: list[str],
    symbol_meta: dict[str, dict[str, Any]],
    asof_dt: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in sorted({str(x).upper().strip() for x in symbols if str(x).strip()}):
        md = symbol_meta.get(s, {})
        listed_at_dt = md.get("listed_at_dt")
        age_days = compute_symbol_age_days(listed_at_dt, asof_dt)
        rows.append(
            {
                "symbol": s,
                "symbol_listed_at_utc": str(md.get("symbol_listed_at_utc", "") or ""),
                "symbol_age_days": age_days if age_days is not None else "",
                "metadata_source": str(md.get("metadata_source", "unknown") or "unknown"),
            }
        )
    return rows


def build_reason_mapping_rows(decision_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for r in decision_rows:
        raw = str(r.get("reason_raw", r.get("reason", "")) or "")
        can = str(r.get("reason_canonical", "") or "")
        counts[(raw, can)] += 1
    rows: list[dict[str, Any]] = []
    for (raw, can), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])):
        rows.append({"reason_raw": raw, "reason_canonical": can, "count": int(n)})
    return rows


def build_reason_agreement(decision_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(decision_rows)
    agree = 0
    legacy_same = 0

    for r in decision_rows:
        raw = str(r.get("reason_raw", r.get("reason", "")) or "")
        can = str(r.get("reason_canonical", "") or "")
        expected = canonicalize_reason(
            reason_raw=raw,
            decision=str(r.get("decision", "") or ""),
            schema_ok=r.get("schema_ok", None),
            meta_ok=r.get("meta_ok", None),
            strat_ok=r.get("strat_ok", None),
            err=str(r.get("err", "") or ""),
        )
        if can == expected:
            agree += 1
        if raw == can:
            legacy_same += 1

    agreement_rate = (float(agree) / float(total)) if total > 0 else 1.0
    legacy_match_rate = (float(legacy_same) / float(total)) if total > 0 else 0.0
    return {
        "rows": total,
        "canonical_agreement_rate": agreement_rate,
        "canonical_agreement_count": int(agree),
        "legacy_reason_equals_canonical_rate": legacy_match_rate,
        "note": "agreement computed on canonical reasons for stability across version-specific raw reason variants",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            out: dict[str, Any] = {}
            for c in columns:
                v = r.get(c, "")
                if isinstance(v, bool):
                    out[c] = "true" if v else "false"
                elif v is None:
                    out[c] = ""
                else:
                    out[c] = v
            writer.writerow(out)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def zip_package(package_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for p in sorted(package_dir.rglob("*")):
            if p.is_file():
                arc = f"{package_dir.name}/{p.relative_to(package_dir).as_posix()}"
                zf.write(p, arcname=arc)


def publish_to_dir(package_dir: Path, zip_path: Path | None, publish_dir: Path) -> None:
    publish_dir.mkdir(parents=True, exist_ok=True)
    dst_pkg = publish_dir / package_dir.name
    if dst_pkg.exists():
        shutil.rmtree(dst_pkg)
    shutil.copytree(package_dir, dst_pkg)
    if zip_path and zip_path.exists():
        shutil.copy2(zip_path, publish_dir / zip_path.name)


def resolve_db_dsn(args_dsn: str | None, env_path: Path) -> str:
    if args_dsn:
        return args_dsn
    if os.getenv("DATABASE_URL"):
        return os.getenv("DATABASE_URL", "")
    env_map = load_env_file(env_path)
    return env_map.get("DATABASE_URL", "")


def build_run_context(
    window: ExportWindow,
    config: dict[str, Any],
    config_hash: str,
    bundle_id: str,
    decision_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    active_symbols: list[str],
    schema_diag: dict[str, Any],
    reason_agreement: dict[str, Any],
    bybit_testnet: bool,
    symbol_meta_error: str | None,
    repo_root: Path,
    db_error: str | None,
) -> dict[str, Any]:
    risk_mode = str(config.get("RISK_MODE", "") or "").lower()
    if risk_mode in {"cash", "fixed"}:
        risk_usd_base = config.get("FIXED_RISK_CASH", config.get("RISK_USD"))
    else:
        risk_usd_base = config.get("RISK_PCT", config.get("RISK_EQUITY_PCT"))

    out = {
        "exported_at_utc": as_utc_iso(utc_now()),
        "window_start_utc": as_utc_iso(window.start_utc),
        "window_end_utc": as_utc_iso(window.end_utc),
        "timezone": "UTC",
        "bundle_id": bundle_id,
        "config_hash": config_hash,
        "strategy_version": strategy_version(config, repo_root),
        "meta_threshold": config.get("META_PROB_THRESHOLD"),
        "meta_scope": config.get("META_GATE_SCOPE"),
        "risk_mode": config.get("RISK_MODE"),
        "risk_usd_base": risk_usd_base,
        "decisions_count": len(decision_rows),
        "trades_count": len(trade_rows),
        "symbols_count": len(active_symbols),
        "schema_fail_count": int(schema_diag.get("schema_fail_count", 0) or 0),
        "schema_fail_rate": float(schema_diag.get("schema_fail_rate", 0.0) or 0.0),
        "schema_incident_recommended": bool(schema_diag.get("incident_recommended", False)),
        "schema_top_non_warmup_missing_fields": schema_diag.get("top_non_warmup_missing_fields", [])[:10],
        "schema_fail_class_counts": schema_diag.get("schema_fail_class_counts", {}),
        "reason_canonical_agreement_rate": float(reason_agreement.get("canonical_agreement_rate", 1.0)),
        "symbol_metadata_source": "exchange_instrument_metadata",
        "bybit_testnet": bool(bybit_testnet),
    }
    if db_error:
        out["trades_export_error"] = db_error
    if symbol_meta_error:
        out["symbol_metadata_error"] = symbol_meta_error
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export daily live autopar package.")
    ap.add_argument("--date", help="UTC date YYYY-MM-DD. Default: today UTC.")
    ap.add_argument("--window-start-utc", help="Override window start (ISO-8601 UTC).")
    ap.add_argument("--window-end-utc", help="Override window end (ISO-8601 UTC).")
    ap.add_argument("--service", default="donch.service", help="systemd unit name")
    ap.add_argument("--output-root", default="results/autopar_exports", help="root export dir")
    ap.add_argument("--config-path", default="config.yaml", help="path to live config yaml")
    ap.add_argument("--symbols-path", default="symbols.txt", help="symbols file")
    ap.add_argument(
        "--invalid-symbols-path",
        default="results/runtime/invalid_symbols.txt",
        help="invalid symbols file to subtract from universe",
    )
    ap.add_argument("--env-path", default=".env", help="env file path for DATABASE_URL fallback")
    ap.add_argument("--db-dsn", default=None, help="postgres dsn override")
    ap.add_argument("--skip-trades", action="store_true", help="skip DB trades export and write header-only CSV")
    ap.add_argument("--trades-timeout-sec", type=float, default=12.0, help="hard timeout for DB trades export")
    ap.add_argument(
        "--bybit-testnet-mode",
        default="auto",
        choices=["auto", "true", "false"],
        help="for instrument metadata fetch: auto from env BYBIT_TESTNET, or force true/false",
    )
    ap.add_argument(
        "--warmup-max-age-days",
        type=int,
        default=45,
        help="max symbol age for warmup_expected schema classification",
    )
    ap.add_argument("--zip", action="store_true", help="also write autopar_YYYY-MM-DD.zip")
    ap.add_argument("--publish-dir", default=None, help="optional mounted/shared destination dir")
    # Kept for CLI compatibility; both files are always exported now.
    ap.add_argument("--include-settings-snapshot", action="store_true", help="deprecated (always included)")
    ap.add_argument("--include-live-log", action="store_true", help="deprecated (always included)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing package dir if present")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()

    try:
        window = build_window(args.date, args.window_start_utc, args.window_end_utc)
    except Exception as e:
        print(f"[ERROR] Invalid window arguments: {e}", file=sys.stderr)
        return 2

    output_root = Path(args.output_root)
    package_dir = output_root / f"autopar_{window.label}"
    zip_path = output_root / f"autopar_{window.label}.zip"

    if package_dir.exists():
        if args.overwrite:
            shutil.rmtree(package_dir)
        else:
            print(
                f"[ERROR] Package directory already exists: {package_dir}. "
                "Use --overwrite to replace.",
                file=sys.stderr,
            )
            return 2
    package_dir.mkdir(parents=True, exist_ok=True)

    try:
        journal_lines = run_journalctl(args.service, window.start_utc, window.end_utc)
    except Exception as e:
        print(f"[ERROR] Failed to fetch journal logs: {e}", file=sys.stderr)
        return 1

    cfg_path = Path(args.config_path)
    config = load_yaml(cfg_path)
    cfg_hash = config_hash_sha256(cfg_path)
    env_map = load_env_file(Path(args.env_path))

    symbols_active = load_symbols_active(Path(args.symbols_path), Path(args.invalid_symbols_path))
    (package_dir / "symbols_active.txt").write_text(
        ("\n".join(symbols_active) + ("\n" if symbols_active else "")),
        encoding="utf-8",
    )

    decision_rows, _meta_lines = parse_meta_decision_lines(journal_lines)

    # Objective listing-age metadata from exchange instrument metadata.
    all_symbols_for_meta = sorted(
        {
            str(r.get("symbol", "")).upper().strip()
            for r in decision_rows
            if str(r.get("symbol", "")).strip()
        }
        | {str(s).upper().strip() for s in symbols_active if str(s).strip()}
    )
    bybit_testnet = resolve_bybit_testnet(env_map, args.bybit_testnet_mode)
    symbol_meta, symbol_meta_error = fetch_symbol_listing_metadata(
        all_symbols_for_meta,
        bybit_testnet=bybit_testnet,
    )
    enrich_decision_rows(
        decision_rows,
        symbol_meta=symbol_meta,
        warmup_max_age_days=int(args.warmup_max_age_days),
    )
    write_csv(package_dir / "live_decisions.csv", decision_rows, DECISION_COLUMNS)
    write_csv(
        package_dir / "symbol_metadata.csv",
        build_symbol_metadata_rows(
            all_symbols_for_meta,
            symbol_meta=symbol_meta,
            asof_dt=window.end_utc,
        ),
        SYMBOL_METADATA_COLUMNS,
    )
    write_csv(
        package_dir / "reason_mapping.csv",
        build_reason_mapping_rows(decision_rows),
        REASON_MAPPING_COLUMNS,
    )
    reason_agreement = build_reason_agreement(decision_rows)
    write_json(package_dir / "reason_agreement.json", reason_agreement)

    schema_diag = build_schema_diagnostics(decision_rows)
    write_json(package_dir / "schema_diagnostics.json", schema_diag)

    dsn = resolve_db_dsn(args.db_dsn, Path(args.env_path))
    trade_rows: list[dict[str, Any]] = []
    db_error: str | None = None
    if args.skip_trades:
        db_error = "trade export skipped by --skip-trades"
    elif dsn:
        trade_rows, db_error = fetch_trades_with_timeout(
            dsn=dsn,
            start_utc=window.start_utc,
            end_utc=window.end_utc,
            timeout_sec=float(args.trades_timeout_sec),
        )
    else:
        db_error = "DATABASE_URL is unavailable (no --db-dsn, env, or .env value)"
    write_csv(package_dir / "live_trades.csv", trade_rows, TRADES_COLUMNS)

    bundle_counter = Counter(
        str(r.get("bundle", "")).strip() for r in decision_rows if str(r.get("bundle", "")).strip()
    )
    bundle_id = bundle_counter.most_common(1)[0][0] if bundle_counter else "unknown"

    run_ctx = build_run_context(
        window=window,
        config=config,
        config_hash=cfg_hash,
        bundle_id=bundle_id,
        decision_rows=decision_rows,
        trade_rows=trade_rows,
        active_symbols=symbols_active,
        schema_diag=schema_diag,
        reason_agreement=reason_agreement,
        bybit_testnet=bybit_testnet,
        symbol_meta_error=symbol_meta_error,
        repo_root=repo_root,
        db_error=db_error,
    )
    write_json(package_dir / "run_context.json", run_ctx)

    # Required for offline schema diagnostics/parity replay support.
    (package_dir / "live.log").write_text(
        ("\n".join(journal_lines) + ("\n" if journal_lines else "")),
        encoding="utf-8",
    )
    write_json(package_dir / "settings_snapshot.json", config)

    did_zip = False
    if args.zip:
        zip_package(package_dir, zip_path)
        did_zip = True

    if args.publish_dir:
        publish_to_dir(package_dir, zip_path if did_zip else None, Path(args.publish_dir))

    print(f"[OK] package_dir={package_dir}")
    if did_zip:
        print(f"[OK] zip={zip_path}")
    print(f"[OK] decisions={len(decision_rows)} trades={len(trade_rows)} symbols={len(symbols_active)}")
    print(
        "[OK] schema_fail_count=%d schema_fail_rate=%.6f incident_recommended=%s"
        % (
            int(schema_diag.get("schema_fail_count", 0) or 0),
            float(schema_diag.get("schema_fail_rate", 0.0) or 0.0),
            str(bool(schema_diag.get("incident_recommended", False))).lower(),
        )
    )
    print(
        "[OK] canonical_reason_agreement=%.6f"
        % float(reason_agreement.get("canonical_agreement_rate", 1.0) or 0.0)
    )
    if symbol_meta_error:
        print(f"[WARN] symbol_metadata_error={symbol_meta_error}")
    if db_error:
        print(f"[WARN] trades_export_error={db_error}")
    if not decision_rows:
        print("[WARN] No META_DECISION rows in window; live.log may still be useful.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
