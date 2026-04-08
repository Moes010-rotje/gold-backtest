"""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — PARAMETER OPTIMIZER v1.0          ║
║  Test 100+ combinaties, vind de beste settings automatisch   ║
╚══════════════════════════════════════════════════════════════╝

Optimaliseert:
  - ATR SL multiplier (0.8 → 2.5)
  - Min confluence (2 → 5)
  - RR ratio (1.0 → 3.0)
  - EMA periodes
  - RSI thresholds
  - Partial close % en timing
  - Cooldowns

Output: Top 10 beste combinaties + aanbevolen live settings
"""

import sys
import json
import math
import time
import itertools
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
from copy import deepcopy

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("pip install pandas numpy")
    sys.exit(1)

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════════════

class Direction(Enum):
    LONG = "buy"
    SHORT = "sell"

class TradePhase(Enum):
    OPEN = "open"
    TP1_HIT = "tp1_hit"
    CLOSED = "closed"


# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Config:
    START_BALANCE: float = 5000.0
    COMMISSION_PER_LOT: float = 7.0
    SIMULATED_SPREAD: float = 0.30
    RISK_PERCENT: float = 0.5
    MAX_DAILY_LOSS_PERCENT: float = 3.0
    MAX_TOTAL_DRAWDOWN_PERCENT: float = 10.0
    MAX_CONCURRENT_TRADES: int = 2
    MAX_DAILY_TRADES: int = 20
    MAX_CONSECUTIVE_LOSSES: int = 5

    # ─── Parameters to optimize ───────────────────────────────
    ATR_PERIOD: int = 10
    ATR_SL_MULTIPLIER: float = 1.5
    MIN_SL_POINTS: float = 2.0
    MAX_SL_POINTS: float = 10.0
    RR_RATIO: float = 2.0

    PARTIAL_PERCENT: float = 0.50
    TP1_RR: float = 1.0
    MOVE_SL_TO_BE: bool = True

    SWING_LOOKBACK: int = 3
    EMA_FAST: int = 9
    EMA_SLOW: int = 21
    USE_EMA_FILTER: bool = True

    ENGULF_BODY_RATIO: float = 0.60
    MOMENTUM_CANDLE_ATR: float = 0.8
    EXHAUSTION_WICK_RATIO: float = 0.65

    USE_MEAN_REVERSION: bool = True
    BB_PERIOD: int = 20
    BB_STD_DEV: float = 2.0
    RSI_PERIOD: int = 7
    RSI_OVERSOLD: float = 25.0
    RSI_OVERBOUGHT: float = 75.0
    MR_CONFLUENCE_SCORE: int = 2

    ROUND_NUMBER_INTERVAL: float = 50.0
    ROUND_NUMBER_ZONE: float = 3.0

    MIN_CONFLUENCE: int = 3
    TRADE_COOLDOWN_BARS: int = 12
    LOSS_COOLDOWN_BARS: int = 30

    # Sessions
    LONDON_START: int = 7
    LONDON_END: int = 12
    OVERLAP_START: int = 12
    OVERLAP_END: int = 15
    NY_START: int = 12
    NY_END: int = 17

    # ─── Nieuwe filters ──────────────────────────────────────
    REQUIRE_VOLUME_CONFIRM: bool = False
    REQUIRE_CANDLE_CLOSE_CONFIRM: bool = True
    MIN_BODY_TO_RANGE_RATIO: float = 0.0  # minimum body/range for entry candle


# ═══════════════════════════════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════════════════════════════

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema9"] = df["Close"].ewm(span=9).mean()
    df["ema21"] = df["Close"].ewm(span=21).mean()
    df["ema50"] = df["Close"].ewm(span=50).mean()

    df["tr"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"] - df["Close"].shift(1))
        )
    )
    df["atr"] = df["tr"].rolling(14).mean()
    df["atr10"] = df["tr"].rolling(10).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(7).mean()
    loss = (-delta.clip(upper=0)).rolling(7).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    delta14 = df["Close"].diff()
    gain14 = delta14.clip(lower=0).rolling(14).mean()
    loss14 = (-delta14.clip(upper=0)).rolling(14).mean()
    rs14 = gain14 / loss14.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs14))

    df["bb_mid"] = df["Close"].rolling(20).mean()
    df["bb_std"] = df["Close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    df["body"] = abs(df["Close"] - df["Open"])
    df["candle_range"] = df["High"] - df["Low"]
    df["upper_wick"] = df["High"] - df[["Close", "Open"]].max(axis=1)
    df["lower_wick"] = df[["Close", "Open"]].min(axis=1) - df["Low"]

    if "Volume" in df.columns:
        df["avg_volume"] = df["Volume"].rolling(20).mean()
    else:
        df["Volume"] = 0
        df["avg_volume"] = 0

    return df


def detect_swings(df: pd.DataFrame, lookback: int = 3) -> Tuple[List[int], List[int]]:
    swing_highs, swing_lows = [], []
    for i in range(lookback, len(df) - lookback):
        h = df["High"].iloc[i]
        l = df["Low"].iloc[i]
        if all(h >= df["High"].iloc[i + j] and h >= df["High"].iloc[i - j] for j in range(1, lookback + 1)):
            swing_highs.append(i)
        if all(l <= df["Low"].iloc[i + j] and l <= df["Low"].iloc[i - j] for j in range(1, lookback + 1)):
            swing_lows.append(i)
    return swing_highs, swing_lows


# ═══════════════════════════════════════════════════════════════════
#  SIGNAL DETECTION
# ═══════════════════════════════════════════════════════════════════

def get_session(hour: int) -> str:
    if 12 <= hour < 15:
        return "overlap"
    if 7 <= hour < 12:
        return "london"
    if 12 <= hour < 17:
        return "new_york"
    return "off"


def detect_signal(df: pd.DataFrame, i: int, cfg: Config,
                  swing_highs: List[int], swing_lows: List[int]
                  ) -> Optional[Tuple[Direction, float, str]]:
    if i < 60:
        return None

    price = df["Close"].iloc[i]
    atr = df["atr"].iloc[i]
    if pd.isna(atr) or atr <= 0:
        return None

    ema_f = df["ema9"].iloc[i] if cfg.EMA_FAST == 9 else df["ema21"].iloc[i]
    ema_s = df["ema21"].iloc[i] if cfg.EMA_SLOW == 21 else df["ema50"].iloc[i]
    if pd.isna(ema_f) or pd.isna(ema_s):
        return None

    ts = df.index[i]
    if not hasattr(ts, 'hour'):
        return None
    hour = ts.hour

    session = get_session(hour)
    if session == "off":
        return None

    # Candle close confirmation
    if cfg.REQUIRE_CANDLE_CLOSE_CONFIRM:
        body = df["body"].iloc[i]
        cr = df["candle_range"].iloc[i]
        if cr > 0 and body / cr < cfg.MIN_BODY_TO_RANGE_RATIO:
            return None

    confluence = 0
    reasons = []
    votes = {Direction.LONG: 0, Direction.SHORT: 0}

    # 1. EMA trend
    if cfg.USE_EMA_FILTER:
        if ema_f > ema_s:
            votes[Direction.LONG] += 1
            reasons.append("ema")
        else:
            votes[Direction.SHORT] += 1
            reasons.append("ema")

    # 2. Liquidity sweep
    recent_sh = [s for s in swing_highs if s < i and s > i - 25]
    recent_sl = [s for s in swing_lows if s < i and s > i - 25]

    ch = df["High"].iloc[i]
    cl = df["Low"].iloc[i]
    cc = df["Close"].iloc[i]

    for si in recent_sl[-3:]:
        sl_p = df["Low"].iloc[si]
        if cl < sl_p and cc > sl_p:
            wick = min(df["Close"].iloc[i], df["Open"].iloc[i]) - cl
            if wick > atr * 0.25:
                votes[Direction.LONG] += 2
                reasons.append("sweep")
                break

    for si in recent_sh[-3:]:
        sh_p = df["High"].iloc[si]
        if ch > sh_p and cc < sh_p:
            wick = ch - max(df["Close"].iloc[i], df["Open"].iloc[i])
            if wick > atr * 0.25:
                votes[Direction.SHORT] += 2
                reasons.append("sweep")
                break

    # 3. Order block / engulfing
    if i >= 2:
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        pb = abs(prev["Close"] - prev["Open"])
        cb = abs(curr["Close"] - curr["Open"])

        if (curr["Close"] > curr["Open"] and prev["Close"] < prev["Open"]
                and cb > pb * cfg.ENGULF_BODY_RATIO
                and curr["Close"] > prev["High"]):
            votes[Direction.LONG] += 1
            reasons.append("ob")
        elif (curr["Close"] < curr["Open"] and prev["Close"] > prev["Open"]
                and cb > pb * cfg.ENGULF_BODY_RATIO
                and curr["Close"] < prev["Low"]):
            votes[Direction.SHORT] += 1
            reasons.append("ob")

    # 4. FVG
    if i >= 2:
        c1h = df["High"].iloc[i - 2]
        c3l = df["Low"].iloc[i]
        c1l = df["Low"].iloc[i - 2]
        c3h = df["High"].iloc[i]
        min_gap = atr * cfg.MOMENTUM_CANDLE_ATR * 0.3

        if c3l > c1h and (c3l - c1h) >= min_gap:
            votes[Direction.LONG] += 1
            reasons.append("fvg")
        elif c1l > c3h and (c1l - c3h) >= min_gap:
            votes[Direction.SHORT] += 1
            reasons.append("fvg")

    # 5. Momentum candle
    body = df["body"].iloc[i]
    total = df["candle_range"].iloc[i]
    if total > 0 and body / total >= cfg.ENGULF_BODY_RATIO and body >= atr * cfg.MOMENTUM_CANDLE_ATR:
        if df["Close"].iloc[i] > df["Open"].iloc[i]:
            votes[Direction.LONG] += 1
            reasons.append("mom")
        else:
            votes[Direction.SHORT] += 1
            reasons.append("mom")

    # 6. Exhaustion
    if total > 0:
        if df["upper_wick"].iloc[i] / total >= cfg.EXHAUSTION_WICK_RATIO:
            votes[Direction.SHORT] += 1
            reasons.append("exh")
        elif df["lower_wick"].iloc[i] / total >= cfg.EXHAUSTION_WICK_RATIO:
            votes[Direction.LONG] += 1
            reasons.append("exh")

    # 7. Round number
    nearest = round(price / cfg.ROUND_NUMBER_INTERVAL) * cfg.ROUND_NUMBER_INTERVAL
    if abs(price - nearest) <= cfg.ROUND_NUMBER_ZONE:
        confluence += 1
        reasons.append("rn")

    # 8. Mean Reversion
    if cfg.USE_MEAN_REVERSION:
        bb_u = df["bb_upper"].iloc[i]
        bb_l = df["bb_lower"].iloc[i]
        rsi = df["rsi"].iloc[i]

        if not (pd.isna(bb_u) or pd.isna(rsi)):
            bb_range = bb_u - bb_l
            if bb_range > 0:
                pct_b = (price - bb_l) / bb_range

                if pct_b <= 0.05 and rsi <= cfg.RSI_OVERSOLD:
                    votes[Direction.LONG] += cfg.MR_CONFLUENCE_SCORE
                    reasons.append("MR")
                elif pct_b >= 0.95 and rsi >= cfg.RSI_OVERBOUGHT:
                    votes[Direction.SHORT] += cfg.MR_CONFLUENCE_SCORE
                    reasons.append("MR")

    # 9. RSI divergence (nieuw)
    rsi14 = df["rsi14"].iloc[i]
    if not pd.isna(rsi14):
        if rsi14 < 30 and df["Close"].iloc[i] > df["Close"].iloc[i - 5]:
            votes[Direction.LONG] += 1
            reasons.append("rsi_div")
        elif rsi14 > 70 and df["Close"].iloc[i] < df["Close"].iloc[i - 5]:
            votes[Direction.SHORT] += 1
            reasons.append("rsi_div")

    # Volume confirm
    if cfg.REQUIRE_VOLUME_CONFIRM and "Volume" in df.columns:
        vol = df["Volume"].iloc[i]
        avg_vol = df["avg_volume"].iloc[i]
        if not pd.isna(avg_vol) and avg_vol > 0 and vol < avg_vol * 1.0:
            return None

    # Determine direction
    ls = votes[Direction.LONG]
    ss = votes[Direction.SHORT]

    if ls > ss and ls >= 1:
        direction = Direction.LONG
        confluence += ls
    elif ss > ls and ss >= 1:
        direction = Direction.SHORT
        confluence += ss
    else:
        return None

    # EMA filter
    if cfg.USE_EMA_FILTER:
        if direction == Direction.LONG and ema_f <= ema_s:
            if "MR" not in reasons:
                return None
        elif direction == Direction.SHORT and ema_f >= ema_s:
            if "MR" not in reasons:
                return None

    if confluence < cfg.MIN_CONFLUENCE:
        return None

    return direction, confluence, "|".join(reasons)


# ═══════════════════════════════════════════════════════════════════
#  FAST BACKTESTER
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    direction: Direction
    entry: float
    sl: float
    tp: float
    tp1: float
    lots: float
    bar: int
    phase: TradePhase = TradePhase.OPEN
    pnl: float = 0.0
    remaining: float = 0.0

    def __post_init__(self):
        self.remaining = self.lots


def run_backtest(df: pd.DataFrame, cfg: Config,
                 swing_highs: List[int], swing_lows: List[int],
                 silent: bool = True) -> dict:
    """Fast backtest — returns performance metrics."""
    balance = cfg.START_BALANCE
    peak = balance
    max_dd_pct = 0.0
    active: List[Trade] = []
    closed: List[Trade] = []
    daily_trades = 0
    daily_date = ""
    consec_losses = 0
    last_trade_bar = -999
    last_loss_bar = -999

    for i in range(60, len(df)):
        price = df["Close"].iloc[i]
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        atr = df["atr"].iloc[i]

        if pd.isna(atr) or price <= 0:
            continue

        ts = df.index[i]
        today = str(ts.date()) if hasattr(ts, 'date') else str(ts)[:10]
        if today != daily_date:
            daily_date = today
            daily_trades = 0

        # Manage trades
        for t in list(active):
            sl_hit = (t.direction == Direction.LONG and low <= t.sl) or \
                     (t.direction == Direction.SHORT and high >= t.sl)
            if sl_hit:
                if t.direction == Direction.LONG:
                    t.pnl += (t.sl - t.entry) * t.remaining * 100
                else:
                    t.pnl += (t.entry - t.sl) * t.remaining * 100
                t.pnl -= cfg.COMMISSION_PER_LOT * t.remaining
                balance += t.pnl
                consec_losses += 1
                last_loss_bar = i
                active.remove(t)
                closed.append(t)
                continue

            tp_hit = (t.direction == Direction.LONG and high >= t.tp) or \
                     (t.direction == Direction.SHORT and low <= t.tp)
            if tp_hit:
                if t.direction == Direction.LONG:
                    t.pnl += (t.tp - t.entry) * t.remaining * 100
                else:
                    t.pnl += (t.entry - t.tp) * t.remaining * 100
                t.pnl -= cfg.COMMISSION_PER_LOT * t.remaining
                balance += t.pnl
                consec_losses = 0
                active.remove(t)
                closed.append(t)
                continue

            # Partial close
            if t.phase == TradePhase.OPEN:
                tp1_hit = (t.direction == Direction.LONG and high >= t.tp1) or \
                          (t.direction == Direction.SHORT and low <= t.tp1)
                if tp1_hit:
                    cl = round(t.lots * cfg.PARTIAL_PERCENT, 2)
                    if cl >= 0.01:
                        if t.direction == Direction.LONG:
                            p = (t.tp1 - t.entry) * cl * 100
                        else:
                            p = (t.entry - t.tp1) * cl * 100
                        p -= cfg.COMMISSION_PER_LOT * cl
                        balance += p
                        t.pnl += p
                        t.remaining = round(t.remaining - cl, 2)
                        t.phase = TradePhase.TP1_HIT
                        if cfg.MOVE_SL_TO_BE:
                            t.sl = t.entry

        # Equity tracking
        unrealized = sum(
            ((price - t.entry) if t.direction == Direction.LONG else (t.entry - price)) * t.remaining * 100
            for t in active
        )
        equity = balance + unrealized
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd_pct:
            max_dd_pct = dd

        # Gates
        if daily_trades >= cfg.MAX_DAILY_TRADES:
            continue
        if len(active) >= cfg.MAX_CONCURRENT_TRADES:
            continue
        if consec_losses >= cfg.MAX_CONSECUTIVE_LOSSES:
            if i - last_loss_bar < cfg.LOSS_COOLDOWN_BARS * 2:
                continue
            consec_losses = 0
        if i - last_trade_bar < cfg.TRADE_COOLDOWN_BARS:
            continue
        if i - last_loss_bar < cfg.LOSS_COOLDOWN_BARS:
            continue
        total_dd = (cfg.START_BALANCE - balance) / cfg.START_BALANCE * 100
        if total_dd >= cfg.MAX_TOTAL_DRAWDOWN_PERCENT:
            continue

        # Signal
        signal = detect_signal(df, i, cfg, swing_highs, swing_lows)
        if not signal:
            continue

        direction, score, reason = signal

        # SL/TP
        sl_dist = max(atr * cfg.ATR_SL_MULTIPLIER, cfg.MIN_SL_POINTS)
        sl_dist = min(sl_dist, cfg.MAX_SL_POINTS)
        tp_dist = sl_dist * cfg.RR_RATIO
        tp1_dist = sl_dist * cfg.TP1_RR

        entry = price + cfg.SIMULATED_SPREAD if direction == Direction.LONG else price

        if direction == Direction.LONG:
            sl = entry - sl_dist
            tp = entry + tp_dist
            tp1 = entry + tp1_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist
            tp1 = entry - tp1_dist

        risk = balance * (cfg.RISK_PERCENT / 100)
        lots = risk / (sl_dist * 100)
        lots = max(0.01, min(round(lots, 2), 0.5))

        trade = Trade(direction=direction, entry=entry, sl=sl, tp=tp,
                      tp1=tp1, lots=lots, bar=i)
        active.append(trade)
        daily_trades += 1
        last_trade_bar = i

    # Close remaining
    if active:
        last_price = df["Close"].iloc[-1]
        for t in active:
            if t.direction == Direction.LONG:
                t.pnl += (last_price - t.entry) * t.remaining * 100
            else:
                t.pnl += (t.entry - last_price) * t.remaining * 100
            closed.append(t)

    # Calculate metrics
    if not closed:
        return {"trades": 0, "pf": 0, "wr": 0, "pnl": 0, "dd": 0, "balance": balance}

    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    total_win = sum(t.pnl for t in wins)
    total_loss = abs(sum(t.pnl for t in losses))

    return {
        "trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "wr": round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "pf": round(total_win / total_loss, 2) if total_loss > 0 else 99,
        "pnl": round(balance - cfg.START_BALANCE, 2),
        "return_pct": round((balance - cfg.START_BALANCE) / cfg.START_BALANCE * 100, 2),
        "dd": round(max_dd_pct, 2),
        "balance": round(balance, 2),
        "avg_win": round(total_win / len(wins), 2) if wins else 0,
        "avg_loss": round(total_loss / len(losses), 2) if losses else 0,
    }


# ═══════════════════════════════════════════════════════════════════
#  OPTIMIZER
# ═══════════════════════════════════════════════════════════════════

def optimize(df: pd.DataFrame):
    """Test parameter combinations and find the best settings."""

    print("\n" + "=" * 60)
    print("  🔧 PARAMETER OPTIMIZER")
    print("  Testing combinations...")
    print("=" * 60)

    df = calculate_indicators(df)
    swing_h, swing_l = detect_swings(df, 3)

    # ─── Parameter grid ──────────────────────────────────────
    param_grid = {
        "ATR_SL_MULTIPLIER": [1.0, 1.5, 2.0, 2.5],
        "MIN_CONFLUENCE": [3, 4, 5],
        "RR_RATIO": [1.5, 2.0, 2.5, 3.0],
        "TP1_RR": [0.8, 1.0, 1.5],
        "PARTIAL_PERCENT": [0.33, 0.50],
        "TRADE_COOLDOWN_BARS": [6, 12, 24],
        "LOSS_COOLDOWN_BARS": [15, 30, 60],
        "MIN_SL_POINTS": [2.0, 3.0, 4.0],
        "REQUIRE_CANDLE_CLOSE_CONFIRM": [True, False],
        "MIN_BODY_TO_RANGE_RATIO": [0.0, 0.3, 0.5],
    }

    # Smart sampling: test key params first, then fine-tune
    # Phase 1: SL multiplier + confluence + RR (48 combos)
    print("\n  📊 Phase 1: Core parameters (SL × Confluence × RR)")
    phase1_results = []

    total = len(param_grid["ATR_SL_MULTIPLIER"]) * len(param_grid["MIN_CONFLUENCE"]) * len(param_grid["RR_RATIO"])
    count = 0

    for sl_mult in param_grid["ATR_SL_MULTIPLIER"]:
        for min_conf in param_grid["MIN_CONFLUENCE"]:
            for rr in param_grid["RR_RATIO"]:
                count += 1
                cfg = Config()
                cfg.ATR_SL_MULTIPLIER = sl_mult
                cfg.MIN_CONFLUENCE = min_conf
                cfg.RR_RATIO = rr

                result = run_backtest(df, cfg, swing_h, swing_l)
                result["params"] = {
                    "sl_mult": sl_mult, "min_conf": min_conf, "rr": rr,
                }
                phase1_results.append(result)

                if count % 10 == 0:
                    print(f"    {count}/{total} tested...")

    # Sort by profit factor, then by PnL
    phase1_results.sort(key=lambda x: (x["pf"], x["pnl"]), reverse=True)

    print(f"\n  ✅ Phase 1 complete: {len(phase1_results)} combinations")
    print(f"\n  Top 5 core combos:")
    for i, r in enumerate(phase1_results[:5]):
        p = r["params"]
        print(f"    {i+1}. SL×{p['sl_mult']} | Conf≥{p['min_conf']} | RR 1:{p['rr']} "
              f"→ PF:{r['pf']} | WR:{r['wr']}% | PnL:${r['pnl']:+.2f} | "
              f"DD:{r['dd']:.1f}% | {r['trades']} trades")

    # Phase 2: Fine-tune top 3 with secondary params
    print(f"\n  📊 Phase 2: Fine-tuning top 3 with secondary parameters")
    phase2_results = []
    count = 0

    top3 = phase1_results[:3]
    for base in top3:
        bp = base["params"]
        for tp1_rr in param_grid["TP1_RR"]:
            for partial in param_grid["PARTIAL_PERCENT"]:
                for cd_trade in param_grid["TRADE_COOLDOWN_BARS"]:
                    for cd_loss in param_grid["LOSS_COOLDOWN_BARS"]:
                        for min_sl in param_grid["MIN_SL_POINTS"]:
                            count += 1
                            cfg = Config()
                            cfg.ATR_SL_MULTIPLIER = bp["sl_mult"]
                            cfg.MIN_CONFLUENCE = bp["min_conf"]
                            cfg.RR_RATIO = bp["rr"]
                            cfg.TP1_RR = tp1_rr
                            cfg.PARTIAL_PERCENT = partial
                            cfg.TRADE_COOLDOWN_BARS = cd_trade
                            cfg.LOSS_COOLDOWN_BARS = cd_loss
                            cfg.MIN_SL_POINTS = min_sl

                            result = run_backtest(df, cfg, swing_h, swing_l)
                            result["params"] = {
                                "sl_mult": bp["sl_mult"],
                                "min_conf": bp["min_conf"],
                                "rr": bp["rr"],
                                "tp1_rr": tp1_rr,
                                "partial": partial,
                                "cd_trade": cd_trade,
                                "cd_loss": cd_loss,
                                "min_sl": min_sl,
                            }
                            phase2_results.append(result)

                            if count % 50 == 0:
                                print(f"    {count} tested...")

    phase2_results.sort(key=lambda x: (x["pf"], x["pnl"]), reverse=True)

    print(f"\n  ✅ Phase 2 complete: {len(phase2_results)} combinations")

    # Phase 3: Filter tweaks on best combo
    print(f"\n  📊 Phase 3: Filter optimization on best combo")
    phase3_results = []
    count = 0

    if phase2_results:
        best_p = phase2_results[0]["params"]
    else:
        best_p = phase1_results[0]["params"]

    for candle_confirm in param_grid["REQUIRE_CANDLE_CLOSE_CONFIRM"]:
        for body_ratio in param_grid["MIN_BODY_TO_RANGE_RATIO"]:
            count += 1
            cfg = Config()
            cfg.ATR_SL_MULTIPLIER = best_p["sl_mult"]
            cfg.MIN_CONFLUENCE = best_p["min_conf"]
            cfg.RR_RATIO = best_p["rr"]
            cfg.TP1_RR = best_p.get("tp1_rr", 1.0)
            cfg.PARTIAL_PERCENT = best_p.get("partial", 0.50)
            cfg.TRADE_COOLDOWN_BARS = best_p.get("cd_trade", 12)
            cfg.LOSS_COOLDOWN_BARS = best_p.get("cd_loss", 30)
            cfg.MIN_SL_POINTS = best_p.get("min_sl", 2.0)
            cfg.REQUIRE_CANDLE_CLOSE_CONFIRM = candle_confirm
            cfg.MIN_BODY_TO_RANGE_RATIO = body_ratio

            result = run_backtest(df, cfg, swing_h, swing_l)
            result["params"] = {**best_p, "candle_confirm": candle_confirm, "body_ratio": body_ratio}
            phase3_results.append(result)

    phase3_results.sort(key=lambda x: (x["pf"], x["pnl"]), reverse=True)

    # ─── FINAL RESULTS ────────────────────────────────────────
    all_results = phase1_results + phase2_results + phase3_results

    # Filter: min 20 trades, PF > 0
    viable = [r for r in all_results if r["trades"] >= 20 and r["pf"] > 0]
    viable.sort(key=lambda x: (x["pf"], x["pnl"]), reverse=True)

    # Score: PF * sqrt(trades) * (1 - dd/100) — balances profitability, activity, safety
    for r in viable:
        r["score"] = round(r["pf"] * math.sqrt(r["trades"]) * (1 - r["dd"] / 100), 2)
    viable.sort(key=lambda x: x["score"], reverse=True)

    print("\n" + "=" * 60)
    print("  🏆 TOP 10 BESTE SETTINGS")
    print("=" * 60)

    for i, r in enumerate(viable[:10]):
        p = r["params"]
        print(f"""
  #{i+1} — Score: {r['score']}
  ├── SL: ATR×{p['sl_mult']} | Min SL: ${p.get('min_sl', 2.0)}
  ├── Confluence: ≥{p['min_conf']} | RR: 1:{p['rr']}
  ├── TP1: {p.get('tp1_rr', 1.0)}R | Partial: {p.get('partial', 0.50)*100:.0f}%
  ├── Cooldown: {p.get('cd_trade', 12)} bars / {p.get('cd_loss', 30)} loss
  ├── Trades: {r['trades']} | Wins: {r['wins']} | WR: {r['wr']}%
  ├── PF: {r['pf']} | PnL: ${r['pnl']:+,.2f} | Return: {r['return_pct']:+.2f}%
  └── Max DD: {r['dd']:.1f}% | Avg Win: ${r['avg_win']} | Avg Loss: ${r['avg_loss']}""")

    # ─── RECOMMENDED SETTINGS ─────────────────────────────────
    if viable:
        best = viable[0]
        bp = best["params"]
        print(f"""
{'='*60}
  ⭐ AANBEVOLEN SETTINGS VOOR LIVE BOT:
{'='*60}

  ATR_SL_MULTIPLIER = {bp['sl_mult']}
  MIN_SL_POINTS = {bp.get('min_sl', 2.0)}
  MAX_SL_POINTS = 10.0
  RR_RATIO / LONDON_RR = {bp['rr']}
  OVERLAP_RR = {bp['rr']}
  NY_RR = {bp['rr']}
  TP1_RR = {bp.get('tp1_rr', 1.0)}
  PARTIAL_PERCENT = {bp.get('partial', 0.50)}
  MIN_CONFLUENCE = {bp['min_conf']}
  TRADE_COOLDOWN_BARS = {bp.get('cd_trade', 12)}
  LOSS_COOLDOWN_BARS = {bp.get('cd_loss', 30)}

  Verwachte performance:
  ├── Win Rate: {best['wr']}%
  ├── Profit Factor: {best['pf']}
  ├── Trades/periode: {best['trades']}
  ├── Max Drawdown: {best['dd']:.1f}%
  └── Return: {best['return_pct']:+.2f}%

{'='*60}""")

    # Save results
    output = {
        "top_10": [{
            "rank": i + 1,
            "params": r["params"],
            "trades": r["trades"],
            "wr": r["wr"],
            "pf": r["pf"],
            "pnl": r["pnl"],
            "dd": r["dd"],
            "score": r["score"],
        } for i, r in enumerate(viable[:10])],
        "total_tested": len(all_results),
        "viable": len(viable),
    }

    with open("optimization_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  📁 Results saved to: optimization_results.json")

    return viable[:10] if viable else []


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — PARAMETER OPTIMIZER v1.0          ║
║  Vindt automatisch de beste settings voor jouw strategie     ║
╚══════════════════════════════════════════════════════════════╝
    """)

    print("📥 Downloading 60-day gold data (5-minute)...")
    df = yf.download("GC=F", period="60d", interval="5m", progress=True)

    if df.empty:
        print("⚠️  5M failed, trying 1H...")
        df = yf.download("GC=F", period="60d", interval="1h", progress=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[df["Close"] > 0]

    print(f"✅ {len(df)} bars: {df.index[0]} → {df.index[-1]}")

    start_time = time.time()
    results = optimize(df)
    elapsed = time.time() - start_time

    print(f"\n  ⏱️  Optimizer klaar in {elapsed:.0f} seconden")
    print(f"  📊 Totaal combinaties getest")

    if results:
        print(f"\n  ✅ Gebruik de aanbevolen settings om je live bot te updaten!")
    else:
        print(f"\n  ⚠️  Geen winstgevende combinatie gevonden op deze data.")
        print(f"      De strategie moet mogelijk fundamenteel aangepast worden.")


if __name__ == "__main__":
    main()
