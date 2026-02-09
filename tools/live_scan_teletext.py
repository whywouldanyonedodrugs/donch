#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import queue
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras


LOG_RE_CHECK = re.compile(r"Checking\s+([A-Z0-9]+)\.\.\.")
LOG_RE_OHLCV = re.compile(
    r"OHLCV_LIMIT\s+symbol=([A-Z0-9]+)\s+base_tf=([0-9a-zA-Z]+).*?base_limit=(\d+)"
)
LOG_RE_BACKFILL = re.compile(
    r"ENTRY_QUAL_BACKFILL\s+symbol=([A-Z0-9]+).*?from=(\d+)\s+to=(\d+)"
)
LOG_RE_META = re.compile(
    r"META_DECISION.*?symbol=([A-Z0-9]+).*?decision_ts=([0-9T:\-+]+)\s+"
    r"schema_ok=([A-Za-z0-9_.-]+)\s+p_cal=([A-Za-z0-9_.-]+)\s+pstar=([A-Za-z0-9_.-]+).*?"
    r"scope_ok=([A-Za-z0-9_.-]+)\s+meta_ok=([A-Za-z0-9_.-]+)\s+strat_ok=([A-Za-z0-9_.-]+)\s+"
    r"reason=([A-Za-z0-9_.:-]+)\s+err=([A-Za-z0-9_.:-]+)"
)
LOG_RE_VETO = re.compile(r"Signal for\s+([A-Z0-9]+)\s+vetoed:\s+(.+)$")
LOG_RE_SIGNAL = re.compile(
    r"SIGNAL\s+([A-Z0-9]+)\s+@\s+([0-9.]+)\s+\|\s+regime=([A-Z_]+)\s+\|\s+wp=([0-9.]+)%"
)
LOG_RE_OPEN = re.compile(r"OPENED\s+(LONG|SHORT)\s+([A-Z0-9]+)")
LOG_RE_WATCHDOG = re.compile(r"WATCHDOG\s+symbol=([A-Z0-9]+)\s+stage=([a-z_]+)\s+age=([0-9.]+)s")
LOG_RE_OHLCV_INSUFF = re.compile(
    r"OHLCV_INSUFFICIENT\s+symbol=([A-Z0-9]+)\s+tf=([0-9a-zA-Z]+)\s+have=(\d+)\s+need=(\d+)"
)
LOG_RE_SCAN_SKIP = re.compile(r"SCAN_SKIP\s+symbol=([A-Z0-9]+)\s+stage=([A-Za-z0-9_:-]+).*?(?:\s+n=(\d+))?")
LOG_RE_FEATURE_MISSING = re.compile(r"FEATURE_MISSING.*?symbol=([A-Z0-9]+).*?missing=\[(.*)\]")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


C_RESET = "\033[0m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_MAGENTA = "\033[95m"
C_CYAN = "\033[96m"
C_GRAY = "\033[90m"


def _c(text: str, color: str, use_ansi: bool) -> str:
    if not use_ansi:
        return text
    return f"{color}{text}{C_RESET}"


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _ansi_supported() -> bool:
    term = os.getenv("TERM", "").strip().lower()
    return sys.stdout.isatty() and term not in {"", "dumb"}


class Screen:
    def __init__(self, use_ansi: bool) -> None:
        self.use_ansi = use_ansi

    def start(self) -> None:
        if self.use_ansi:
            sys.stdout.write("\x1b[?25l\x1b[2J\x1b[H")
            sys.stdout.flush()

    def draw(self, text: str) -> None:
        if self.use_ansi:
            sys.stdout.write("\x1b[H")
            sys.stdout.write(text)
            sys.stdout.write("\x1b[J")
            sys.stdout.flush()
            return
        print(text, flush=True)
        print("-" * 80, flush=True)

    def stop(self) -> None:
        if self.use_ansi:
            sys.stdout.write("\x1b[?25h\n")
            sys.stdout.flush()


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        if value.lower() in {"nan", "none", "null"}:
            return None
        return float(value)
    except Exception:
        return None


def _safe_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in {"true", "1"}:
        return True
    if v in {"false", "0"}:
        return False
    return None


def _fmt_bool(v: Optional[bool]) -> str:
    if v is None:
        return " ? "
    return " Y " if v else " N "


def _fmt_bool_c(v: Optional[bool], use_ansi: bool) -> str:
    if v is None:
        return _c(" ? ", C_GRAY, use_ansi)
    if v:
        return _c(" Y ", C_GREEN, use_ansi)
    return _c(" N ", C_RED, use_ansi)


def _fmt_float(v: Optional[float], width: int = 7, prec: int = 3) -> str:
    if v is None:
        return " " * (width - 1) + "-"
    fmt = f"{{:>{width}.{prec}f}}"
    return fmt.format(v)


def _fmt_ago(ts: Optional[datetime]) -> str:
    if ts is None:
        return "   -"
    delta = datetime.now(timezone.utc) - ts
    sec = max(0, int(delta.total_seconds()))
    if sec < 60:
        return f"{sec:>3}s"
    if sec < 3600:
        return f"{sec // 60:>3}m"
    return f"{sec // 3600:>3}h"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, value = s.split("=", 1)
        key = key.strip()
        value = value.strip()
        if "#" in value:
            value = value.split("#", 1)[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _sanitize_text(text: str) -> str:
    s = text.replace("\r", " ").replace("\t", " ")
    s = ANSI_RE.sub("", s)
    out: List[str] = []
    for ch in s:
        oc = ord(ch)
        if 32 <= oc <= 126:
            out.append(ch)
    return " ".join("".join(out).split())


@dataclass
class SymbolState:
    symbol: str
    last_seen: Optional[datetime] = None
    last_event: str = "-"
    decision_ts: Optional[str] = None
    p_cal: Optional[float] = None
    pstar: Optional[float] = None
    schema_ok: Optional[bool] = None
    meta_ok: Optional[bool] = None
    strat_ok: Optional[bool] = None
    scope_ok: Optional[bool] = None
    reason: Optional[str] = None
    err: Optional[str] = None
    veto: Optional[str] = None
    ohlcv_tf: Optional[str] = None
    ohlcv_limit: Optional[int] = None
    backfill_from: Optional[int] = None
    backfill_to: Optional[int] = None
    last_wp_pct: Optional[float] = None
    regime: Optional[str] = None
    watchdog_age_s: Optional[float] = None
    opens: int = 0
    ohlcv_have: Optional[int] = None
    ohlcv_need: Optional[int] = None
    feature_missing: Optional[str] = None
    feature_missing_n: Optional[int] = None
    scan_skip_stage: Optional[str] = None
    scan_skip_n: Optional[int] = None


@dataclass
class TradeRow:
    closed_at: datetime
    symbol: str
    side: str
    pnl: float
    exit_reason: Optional[str]
    market_regime: Optional[str]


@dataclass
class Snapshot:
    equity: Optional[float]
    equity_ts: Optional[datetime]
    open_positions: int
    closed_total: int
    closed_period: int
    winrate_period: Optional[float]
    sharpe_period: Optional[float]
    pnl_period: Optional[float]
    last_closed_regime: Optional[str]
    recent_trades: List[TradeRow] = field(default_factory=list)


class OhlcvSparkReader:
    def __init__(self, sqlite_path: Optional[str], tf: str, bars: int, width: int) -> None:
        self.path = str(sqlite_path or "").strip()
        self.tf = tf
        self.bars = max(8, int(bars))
        self.width = max(12, int(width))
        self.conn: Optional[sqlite3.Connection] = None
        self.enabled = False

    def connect(self) -> None:
        if not self.path:
            return
        p = Path(self.path)
        if not p.exists():
            return
        self.conn = sqlite3.connect(str(p))
        self.enabled = True

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass

    def render_symbol(self, symbol: str, use_ansi: bool) -> Optional[str]:
        if not self.enabled or self.conn is None:
            return None123Cryptozjbs123
            
        rows = self._load_recent_closes(symbol=symbol, tf=self.tf, limit=self.bars)
        if not rows:
            return None
        vals = [float(x) for x in rows]
        spark = self._spark(vals, self.width)
        if not spark:
            return None
        first = vals[0]
        last = vals[-1]
        change = ((last / first) - 1.0) * 100.0 if first > 0 else 0.0
        color = C_GREEN if change >= 0 else C_RED
        spark_s = _c(spark, color, use_ansi)
        change_s = _c(f"{change:+.2f}%", color, use_ansi)
        return f"{spark_s}  {change_s}"

    def render_big_chart(self, symbol: str, use_ansi: bool, height: int, width: int) -> List[str]:
        if not self.enabled or self.conn is None:
            return []

        # Reserve space for labels (e.g. 12 chars)
        chart_w = max(10, width - 12)
        data = self._load_recent_ohlc(symbol, self.tf, chart_w)
        if not data:
            return []

        highs = [x[1] for x in data]
        lows = [x[2] for x in data]
        g_max = max(highs)
        g_min = min(lows)
        if g_max <= g_min:
            g_max = g_min + 1e-9

        scale = (height - 1) / (g_max - g_min)

        grid = [[" " for _ in range(len(data))] for _ in range(height)]

        for x, (o, h, l, c) in enumerate(data):
            y_h = int((h - g_min) * scale)
            y_l = int((l - g_min) * scale)
            y_o = int((o - g_min) * scale)
            y_c = int((c - g_min) * scale)

            # Clamp
            y_h = min(height - 1, max(0, y_h))
            y_l = min(height - 1, max(0, y_l))
            y_o = min(height - 1, max(0, y_o))
            y_c = min(height - 1, max(0, y_c))

            is_up = c >= o
            color = C_GREEN if is_up else C_RED

            # Wick
            for y in range(y_l, y_h + 1):
                grid[height - 1 - y][x] = _c("│", color, use_ansi)

            # Body
            b_start = min(y_o, y_c)
            b_end = max(y_o, y_c)
            for y in range(b_start, b_end + 1):
                grid[height - 1 - y][x] = _c("█", color, use_ansi)

        lines = []
        # Add symbol info at top
        change = ((data[-1][3] / data[0][0]) - 1) * 100
        c_color = C_GREEN if change >= 0 else C_RED
        info = f"{symbol} {self.tf} {len(data)} bars  Change: " + _c(f"{change:+.2f}%", c_color, use_ansi)
        lines.append(info)

        # Add price labels on the right
        for i, row_chars in enumerate(grid):
            price_level = g_max - (i / max(1, height - 1)) * (g_max - g_min)
            label = f" {price_level:.4f}"
            lines.append("".join(row_chars) + _c(label, C_GRAY, use_ansi))

        return lines

    def _load_recent_closes(self, symbol: str, tf: str, limit: int) -> List[float]:
        assert self.conn is not None
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT close
            FROM ohlcv
            WHERE symbol=? AND tf=?
            ORDER BY ts DESC
            LIMIT ?
            """,
            (symbol, tf, int(limit)),
        )
        rows = cur.fetchall() or []
        vals: List[float] = []
        for r in reversed(rows):
            try:
                vals.append(float(r[0]))
            except Exception:
                continue
        return vals

    def _load_recent_ohlc(self, symbol: str, tf: str, limit: int) -> List[Tuple[float, float, float, float]]:
        assert self.conn is not None
        cur = self.conn.cursor()
        try:
            cur.execute(
                "SELECT open, high, low, close FROM ohlcv WHERE symbol=? AND tf=? ORDER BY ts DESC LIMIT ?",
                (symbol, tf, int(limit)),
            )
            rows = cur.fetchall()
            if not rows:
                return []
            return [(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in reversed(rows)]
        except Exception:
            return []

    @staticmethod
    def _spark(values: List[float], width: int) -> str:
        if not values:
            return ""
        if len(values) > width:
            step = len(values) / float(width)
            sampled: List[float] = []
            for i in range(width):
                idx = min(len(values) - 1, int((i + 1) * step) - 1)
                sampled.append(values[idx])
            values = sampled
        chars = " ▂▃▄▅▆▇█"
        lo = min(values)
        hi = max(values)
        if hi <= lo:
            return "▄" * len(values)
        out: List[str] = []
        scale = float(len(chars) - 1) / float(hi - lo)
        for v in values:
            idx = int((v - lo) * scale)
            if idx < 0:
                idx = 0
            if idx >= len(chars):
                idx = len(chars) - 1
            out.append(chars[idx])
        return "".join(out)


class LogTailer:
    def __init__(self, units: List[str], since_lines: int) -> None:
        self.units = units
        self.since_lines = since_lines
        self.proc: Optional[subprocess.Popen] = None
        self.queue: queue.Queue[str] = queue.Queue(maxsize=5000)
        self.stop_evt = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        cmd = ["journalctl"]
        for unit in self.units:
            cmd.extend(["-u", unit])
        cmd.extend(["-f", "-n", str(self.since_lines), "-o", "cat"])
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        while not self.stop_evt.is_set():
            line = self.proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            s = _sanitize_text(line.rstrip("\n"))
            if not s:
                continue
            try:
                self.queue.put_nowait(s)
            except queue.Full:
                try:
                    _ = self.queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.queue.put_nowait(s)
                except queue.Full:
                    pass

    def poll(self, max_items: int = 500) -> List[str]:
        out: List[str] = []
        for _ in range(max_items):
            try:
                out.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return out

    def stop(self) -> None:
        self.stop_evt.set()
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=1.0)
            except Exception:
                pass


class LogParser:
    def __init__(self) -> None:
        self.symbols: Dict[str, SymbolState] = {}
        self.events: deque[str] = deque(maxlen=14)
        self.last_signal_regime: Optional[str] = None

    def _get(self, symbol: str) -> SymbolState:
        st = self.symbols.get(symbol)
        if st is None:
            st = SymbolState(symbol=symbol)
            self.symbols[symbol] = st
        return st

    def _touch(self, st: SymbolState, event: str) -> None:
        st.last_seen = datetime.now(timezone.utc)
        st.last_event = event

    def _push_event(self, text: str) -> None:
        t = _sanitize_text(text)
        if not t:
            return
        self.events.append(t[:180])

    def feed(self, line: str) -> None:
        m = LOG_RE_CHECK.search(line)
        if m:
            sym = m.group(1)
            st = self._get(sym)
            self._touch(st, "scan")
            self._push_event(f"SCAN {sym}")
            return

        m = LOG_RE_OHLCV.search(line)
        if m:
            sym, tf, lim = m.groups()
            st = self._get(sym)
            st.ohlcv_tf = tf
            st.ohlcv_limit = int(lim)
            self._touch(st, "ohlcv")
            self._push_event(f"OHLCV {sym} tf={tf} need={lim}")
            return

        m = LOG_RE_BACKFILL.search(line)
        if m:
            sym, bf_from, bf_to = m.groups()
            st = self._get(sym)
            st.backfill_from = int(bf_from)
            st.backfill_to = int(bf_to)
            self._touch(st, "backfill")
            self._push_event(f"BACKFILL {sym} {bf_from}->{bf_to}")
            return

        m = LOG_RE_META.search(line)
        if m:
            (
                sym,
                decision_ts,
                schema_ok,
                p_cal,
                pstar,
                scope_ok,
                meta_ok,
                strat_ok,
                reason,
                err,
            ) = m.groups()
            st = self._get(sym)
            st.decision_ts = decision_ts
            st.schema_ok = _safe_bool(schema_ok)
            st.p_cal = _safe_float(p_cal)
            st.pstar = _safe_float(pstar)
            st.scope_ok = _safe_bool(scope_ok)
            st.meta_ok = _safe_bool(meta_ok)
            st.strat_ok = _safe_bool(strat_ok)
            st.reason = None if reason in {"-", "None", "none", "null", "nan"} else reason
            st.err = None if err in {"-", "None", "none", "null", "nan"} else err
            self._touch(st, "meta")
            ptxt = "-" if st.p_cal is None else f"{st.p_cal:.3f}"
            self._push_event(
                f"META {sym} p={ptxt} schema={schema_ok} meta={meta_ok} strat={strat_ok} reason={reason}"
            )
            return

        m = LOG_RE_VETO.search(line)
        if m:
            sym, veto = m.groups()
            st = self._get(sym)
            st.veto = veto.strip()
            self._touch(st, "veto")
            self._push_event(f"VETO {sym} {st.veto}")
            return

        m = LOG_RE_SIGNAL.search(line)
        if m:
            sym, _px, regime, wp = m.groups()
            st = self._get(sym)
            st.regime = regime
            st.last_wp_pct = _safe_float(wp)
            self.last_signal_regime = regime
            self._touch(st, "signal")
            self._push_event(f"SIGNAL {sym} regime={regime} wp={wp}%")
            return

        m = LOG_RE_OPEN.search(line)
        if m:
            _side, sym = m.groups()
            st = self._get(sym)
            st.opens += 1
            self._touch(st, "open")
            self._push_event(f"OPEN {sym}")
            return

        m = LOG_RE_WATCHDOG.search(line)
        if m:
            sym, _stage, age = m.groups()
            st = self._get(sym)
            st.watchdog_age_s = _safe_float(age)
            self._touch(st, "watchdog")
            self._push_event(f"WATCHDOG {sym} age={age}s")
            return

        m = LOG_RE_OHLCV_INSUFF.search(line)
        if m:
            sym, tf, have, need = m.groups()
            st = self._get(sym)
            st.ohlcv_tf = tf
            st.ohlcv_have = int(have)
            st.ohlcv_need = int(need)
            self._touch(st, "ohlcv_short")
            self._push_event(f"OHLCV_SHORT {sym} {have}/{need}")
            return

        m = LOG_RE_SCAN_SKIP.search(line)
        if m:
            sym, stage, n = m.groups()
            st = self._get(sym)
            st.scan_skip_stage = stage
            st.scan_skip_n = int(n) if n else None
            self._touch(st, "scan_skip")
            if n:
                self._push_event(f"SKIP {sym} stage={stage} n={n}")
            else:
                self._push_event(f"SKIP {sym} stage={stage}")
            return

        m = LOG_RE_FEATURE_MISSING.search(line)
        if m:
            sym, missing = m.groups()
            st = self._get(sym)
            st.feature_missing = missing.strip()
            items = [x.strip() for x in missing.split(",") if x.strip()]
            st.feature_missing_n = len(items)
            self._touch(st, "feat_miss")
            self._push_event(f"FEATURE_MISS {sym} n={st.feature_missing_n}")
            return

        if "[WARNING]" in line or "[ERROR]" in line or "[CRITICAL]" in line:
            self._push_event(line)


class DBReader:
    def __init__(self, dsn: str, lookback_hours: int, recent_trades: int) -> None:
        self.dsn = dsn
        self.lookback_hours = lookback_hours
        self.recent_trades_n = recent_trades
        self.conn = None
        self.columns_positions: set[str] = set()
        self.regime_col: Optional[str] = None

    def connect(self) -> None:
        self.conn = psycopg2.connect(self.dsn)
        self.conn.autocommit = True
        self._load_schema_hints()

    def _load_schema_hints(self) -> None:
        if self.conn is None:
            return
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'positions'
                """
            )
            rows = cur.fetchall() or []
        self.columns_positions = {str(r[0]) for r in rows if r and r[0]}
        for candidate in ("market_regime_at_entry", "market_regime"):
            if candidate in self.columns_positions:
                self.regime_col = candidate
                break

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass

    def snapshot(self) -> Snapshot:
        if self.conn is None:
            self.connect()
        assert self.conn is not None
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=self.lookback_hours)

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT equity, ts FROM equity_snapshots ORDER BY ts DESC LIMIT 1")
            row = cur.fetchone()
            equity = float(row["equity"]) if row and row["equity"] is not None else None
            equity_ts = row["ts"] if row else None

            cur.execute("SELECT COUNT(*) AS n FROM positions WHERE status='OPEN'")
            open_positions = int(cur.fetchone()["n"])

            cur.execute("SELECT COUNT(*) AS n FROM positions WHERE status='CLOSED'")
            closed_total = int(cur.fetchone()["n"])

            cur.execute(
                """
                SELECT pnl, closed_at
                FROM positions
                WHERE status='CLOSED' AND closed_at >= %s
                ORDER BY closed_at ASC
                """,
                (start,),
            )
            period_rows = cur.fetchall() or []
            pnls: List[float] = [float(r["pnl"]) for r in period_rows if r["pnl"] is not None]
            closed_period = len(pnls)
            wins = sum(1 for p in pnls if p > 0)
            winrate = (wins / closed_period) if closed_period > 0 else None
            pnl_period = float(sum(pnls)) if pnls else 0.0

            sharpe = self._compute_sharpe_from_equity(cur, start)

            last_closed_regime = self._fetch_last_closed_regime(cur)

            trades = self._fetch_recent_trades(cur)

            return Snapshot(
                equity=equity,
                equity_ts=equity_ts,
                open_positions=open_positions,
                closed_total=closed_total,
                closed_period=closed_period,
                winrate_period=winrate,
                sharpe_period=sharpe,
                pnl_period=pnl_period,
                last_closed_regime=last_closed_regime,
                recent_trades=trades,
            )

    def _fetch_last_closed_regime(self, cur) -> Optional[str]:
        if not self.regime_col:
            return None
        query = f"SELECT {self.regime_col} AS regime FROM positions WHERE closed_at IS NOT NULL ORDER BY closed_at DESC LIMIT 1"
        cur.execute(query)
        row = cur.fetchone()
        if not row:
            return None
        value = row.get("regime")
        return str(value) if value is not None else None

    def _fetch_recent_trades(self, cur) -> List[TradeRow]:
        regime_expr = "NULL::text AS regime"
        if self.regime_col:
            regime_expr = f"{self.regime_col} AS regime"
        cur.execute(
            f"""
            SELECT symbol, side, pnl, exit_reason, {regime_expr}, closed_at
            FROM positions
            WHERE closed_at IS NOT NULL
            ORDER BY closed_at DESC
            LIMIT %s
            """,
            (self.recent_trades_n,),
        )
        trades: List[TradeRow] = []
        for tr in cur.fetchall() or []:
            closed_at = tr.get("closed_at")
            if closed_at is None:
                continue
            trades.append(
                TradeRow(
                    closed_at=closed_at,
                    symbol=str(tr.get("symbol") or "-"),
                    side=str(tr.get("side") or "-"),
                    pnl=float(tr["pnl"]) if tr.get("pnl") is not None else 0.0,
                    exit_reason=str(tr["exit_reason"]) if tr.get("exit_reason") is not None else None,
                    market_regime=str(tr["regime"]) if tr.get("regime") is not None else None,
                )
            )
        return trades

    def _compute_sharpe_from_equity(self, cur, start: datetime) -> Optional[float]:
        cur.execute(
            """
            SELECT ts, equity
            FROM equity_snapshots
            WHERE ts >= %s
            ORDER BY ts ASC
            """,
            (start,),
        )
        rows = cur.fetchall() or []
        if len(rows) < 3:
            return None
        series: List[Tuple[datetime, float]] = []
        for r in rows:
            if r["equity"] is None:
                continue
            eq = float(r["equity"])
            if eq <= 0:
                continue
            series.append((r["ts"], eq))
        if len(series) < 3:
            return None

        rets: List[float] = []
        dts: List[float] = []
        prev_t, prev_e = series[0]
        for t, e in series[1:]:
            if prev_e > 0 and e > 0:
                rets.append((e / prev_e) - 1.0)
                dts.append(max((t - prev_t).total_seconds(), 1.0))
            prev_t, prev_e = t, e
        if len(rets) < 2:
            return None

        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        std = var ** 0.5
        if std == 0:
            return None

        step_sec = sorted(dts)[len(dts) // 2]
        periods_per_year = (365.0 * 24.0 * 3600.0) / step_sec
        return (mean / std) * (periods_per_year ** 0.5)


def _trim(text: str, width: int) -> str:
    vis = _strip_ansi(text)
    if len(vis) < width:
        return text + " " * (width - len(vis))
    if len(vis) == width:
        return text
    if width <= 1:
        return vis[:width]
    if width <= 3:
        return vis[:width]
    return vis[: width - 3] + "..."


def _parse_units(unit_arg: str) -> List[str]:
    parts = [u.strip() for u in str(unit_arg).split(",") if u.strip()]
    if not parts:
        parts = ["donch", "DONCH", "donch.service", "DONCH.service"]
    out: List[str] = []
    seen = set()
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def render(
    parser: LogParser,
    snap: Optional[Snapshot],
    max_symbols: int,
    lookback_hours: int,
    units: List[str],
    use_ansi: bool,
    spark_rows: List[str],
    chart_lines: Optional[List[str]] = None,
) -> str:
    term_cols = shutil.get_terminal_size((140, 40)).columns
    cols = max(100, min(int(term_cols), 170))
    sep = "=" * cols
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: List[str] = []
    lines.append(sep)
    lines.append(_trim(f"DONCH TELETEXT MONITOR   {now}", cols))

    if snap is None:
        lines.append(_trim("DB: unavailable", cols))
    else:
        eq = "-" if snap.equity is None else f"{snap.equity:.2f}"
        wr = "-" if snap.winrate_period is None else f"{snap.winrate_period * 100.0:.1f}%"
        sh = "-" if snap.sharpe_period is None else f"{snap.sharpe_period:.2f}"
        pnl = "-" if snap.pnl_period is None else f"{snap.pnl_period:+.2f}"
        if snap.pnl_period is not None:
            c = C_GREEN if snap.pnl_period >= 0 else C_RED
            pnl = _c(pnl, c, use_ansi)

        reg = snap.last_closed_regime or "n/a"
        lines.append(
            _trim(
                f"EQ {eq} | OPEN {snap.open_positions} | CLOSED {snap.closed_total} | "
                f"{lookback_hours}h: N={snap.closed_period} WR={wr} SHARPE={sh} PNL={pnl} | LAST_CLOSED_REGIME={reg}",
                cols,
            )
        )
    lines.append(sep)

    head = "SYMBOL      AGE EVT       p_cal  p*   sch met str scp reason       notes"
    lines.append(_trim(head, cols))
    lines.append("-" * min(len(head), cols))

    states = sorted(
        parser.symbols.values(),
        key=lambda s: s.last_seen or datetime(1970, 1, 1, tzinfo=timezone.utc),
        reverse=True,
    )[:max_symbols]
    note_width = max(20, cols - 68)
    for st in states:
        reason = st.reason or "-"
        note_parts: List[str] = []
        if st.scan_skip_stage:
            if st.scan_skip_n is not None:
                note_parts.append(f"skip:{st.scan_skip_stage}:{st.scan_skip_n}")
            else:
                note_parts.append(f"skip:{st.scan_skip_stage}")
        if st.ohlcv_need is not None:
            note_parts.append(f"bars:{st.ohlcv_have or 0}/{st.ohlcv_need}")
        if st.feature_missing_n is not None:
            note_parts.append(f"miss:{st.feature_missing_n}")
        if st.ohlcv_tf and st.ohlcv_limit:
            note_parts.append(f"need:{st.ohlcv_tf}:{st.ohlcv_limit}")
        if st.backfill_to and st.backfill_from is not None:
            note_parts.append(f"bf:{st.backfill_from}->{st.backfill_to}")
        if st.watchdog_age_s is not None:
            note_parts.append(f"wd={st.watchdog_age_s:.1f}s")
        if st.regime:
            note_parts.append(f"reg={st.regime}")
        if st.veto:
            note_parts.append("veto")
        note = ",".join(note_parts[:5]) if note_parts else "-"
        note = _trim(note, note_width)

        p_cal_str = _fmt_float(st.p_cal, 7, 3)
        if st.p_cal is not None and st.pstar is not None and st.p_cal >= st.pstar:
            p_cal_str = _c(p_cal_str, C_GREEN, use_ansi)

        row = (
            f"{st.symbol:<10} {_fmt_ago(st.last_seen):>4} {st.last_event[:9]:<9} "
            f"{p_cal_str} {_fmt_float(st.pstar,4,2)} "
            f"{_fmt_bool_c(st.schema_ok, use_ansi)}{_fmt_bool_c(st.meta_ok, use_ansi)}{_fmt_bool_c(st.strat_ok, use_ansi)}{_fmt_bool_c(st.scope_ok, use_ansi)} "
            f"{reason[:12]:<12} {note}"
        )
        lines.append(_trim(row, cols))

    if spark_rows:
        lines.append(sep)
        lines.append(_trim("SCAN CANDLES (5m close, newest on right)", cols))
        for s in spark_rows:
            lines.append(_trim(s, cols))

    lines.append(sep)
    lines.append(_trim("RECENT EVENTS", cols))
    for e in list(parser.events)[-10:]:
        lines.append(_trim(e, cols))

    if snap is not None and snap.recent_trades:
        lines.append(sep)
        lines.append(_trim("LAST TRADES", cols))
        for tr in snap.recent_trades[:6]:
            ts = tr.closed_at.astimezone(timezone.utc).strftime("%m-%d %H:%M")
            pnl_s = f"{tr.pnl:+8.2f}"
            if tr.pnl >= 0:
                pnl_s = _c(pnl_s, C_GREEN, use_ansi)
            else:
                pnl_s = _c(pnl_s, C_RED, use_ansi)

            row = (
                f"{ts} {tr.symbol:<10} {tr.side:<5} pnl={pnl_s} "
                f"exit={tr.exit_reason or '-'} reg={tr.market_regime or '-'}"
            )
            lines.append(_trim(row, cols))

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Teletext-style DONCH live monitor (scans, rejects, equity, regime)."
    )
    ap.add_argument(
        "--unit",
        default="donch,DONCH,donch.service,DONCH.service",
        help="systemd unit(s), comma-separated",
    )
    ap.add_argument("--lines", type=int, default=300, help="journal bootstrap lines")
    ap.add_argument("--refresh-sec", type=float, default=1.0, help="screen refresh interval")
    ap.add_argument("--stats-sec", type=float, default=5.0, help="DB stats refresh interval")
    ap.add_argument("--lookback-hours", type=int, default=24, help="lookback for winrate/sharpe/pnl")
    ap.add_argument("--max-symbols", type=int, default=28, help="max symbol rows")
    ap.add_argument("--recent-trades", type=int, default=8, help="last trades rows")
    ap.add_argument("--env-file", default=".env", help="path to env file for DATABASE_URL")
    ap.add_argument("--no-ansi", action="store_true", help="disable ANSI in-place screen refresh")
    ap.add_argument("--spark-bars", type=int, default=48, help="bars used for spark candles")
    ap.add_argument("--spark-width", type=int, default=36, help="spark width in characters")
    ap.add_argument("--spark-rows", type=int, default=8, help="how many symbols to render with sparks")
    ap.add_argument("--spark-tf", default="5m", help="timeframe for spark candles")
    ap.add_argument(
        "--ohlcv-sqlite",
        default=os.getenv("OHLCV_DISK_CACHE_SQLITE", "results/runtime/ohlcv_cache.sqlite"),
        help="path to OHLCV sqlite cache for spark candles",
    )
    args = ap.parse_args()

    _load_env_file(Path(args.env_file))
    dsn = os.getenv("DATABASE_URL", "").strip()
    if not dsn:
        print("DATABASE_URL is missing (set env or .env).", file=sys.stderr)
        return 2

    units = _parse_units(args.unit)
    tailer = LogTailer(units=units, since_lines=args.lines)
    parser = LogParser()
    db = DBReader(dsn=dsn, lookback_hours=args.lookback_hours, recent_trades=args.recent_trades)
    sparks = OhlcvSparkReader(
        sqlite_path=args.ohlcv_sqlite,
        tf=str(args.spark_tf),
        bars=int(args.spark_bars),
        width=int(args.spark_width),
    )
    use_ansi = _ansi_supported() and not args.no_ansi
    screen = Screen(use_ansi=use_ansi)

    stop = {"flag": False}

    def _handle_sig(_sig, _frm):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    tailer.start()
    sparks.connect()
    screen.start()
    snap: Optional[Snapshot] = None
    next_stats_at = 0.0

    try:
        while not stop["flag"]:
            for line in tailer.poll(max_items=2000):
                parser.feed(line)

            now_mono = time.monotonic()
            if now_mono >= next_stats_at:
                try:
                    snap = db.snapshot()
                except Exception as e:
                    err_text = str(e).strip()
                    if err_text:
                        parser.events.append(f"DB snapshot error: {type(e).__name__}: {err_text}")
                    else:
                        parser.events.append(f"DB snapshot error: {type(e).__name__}")
                next_stats_at = now_mono + max(1.0, args.stats_sec)

            states = sorted(
                parser.symbols.values(),
                key=lambda s: s.last_seen or datetime(1970, 1, 1, tzinfo=timezone.utc),
                reverse=True,
            )

            spark_rows: List[str] = []
            if sparks.enabled and args.spark_rows > 0:
                for st in states[: args.spark_rows]:
                    rendered = sparks.render_symbol(st.symbol, use_ansi)
                    if rendered:
                        spark_rows.append(f"{st.symbol:<10} {rendered}")

            chart_lines: List[str] = []
            if sparks.enabled:
                if states:
                    term_w = shutil.get_terminal_size((140, 40)).columns
                    chart_w = max(40, min(int(term_w), 170) - 2)
                    chart_lines = sparks.render_big_chart(states[0].symbol, use_ansi, height=12, width=chart_w)

            screen.draw(
                render(
                    parser,
                    snap,
                    max_symbols=args.max_symbols,
                    lookback_hours=args.lookback_hours,
                    units=units,
                    use_ansi=use_ansi,
                    spark_rows=spark_rows,
                    chart_lines=chart_lines,
                )
            )
            time.sleep(max(0.1, args.refresh_sec))

    finally:
        tailer.stop()
        db.close()
        sparks.close()
        screen.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
