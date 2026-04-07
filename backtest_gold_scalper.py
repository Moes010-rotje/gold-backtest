"""
╔══════════════════════════════════════════════════════════════╗
║        XAUUSD GOLD SCALPER — BACKTESTER v1.0                ║
║     Mode 1: 5 jaar op 15M/1H (gratis via yfinance)          ║
║     Mode 2: 60 dagen op 5M/1M (exact als live bot)          ║
║     Dezelfde strategie als xauusd_gold_scalper.py            ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python backtest_gold_scalper.py --mode short    # 60 dagen, 5M/1M
    python backtest_gold_scalper.py --mode long     # 5 jaar, 15M/1H
    python backtest_gold_scalper.py --mode both     # beide

Requires:
    pip install yfinance pandas numpy
"""

import os
import sys
import argparse
import json
import math
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum

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
#  CONFIGURATION (mirrors live bot)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    # ─── Mode ─────────────────────────────────────────────────────
    SYMBOL: str = "GC=F"              # yfinance gold futures ticker
    START_BALANCE: float = 10000.0
    COMMISSION_PER_LOT: float = 7.0   # $7 round trip per lot

    # ─── Risk ─────────────────────────────────────────────────────
    RISK_PERCENT: float = 0.5
    MAX_DAILY_LOSS_PERCENT: float = 3.0
    MAX_TOTAL_DRAWDOWN_PERCENT: float = 6.0
    MAX_CONCURRENT_TRADES: int = 2
    MAX_DAILY_TRADES: int = 20
    MAX_CONSECUTIVE_LOSSES: int = 4

    # ─── Spread Simulation ────────────────────────────────────────
    SIMULATED_SPREAD: float = 0.30    # $0.30 average gold spread
    MAX_SPREAD: float = 3.50

    # ─── SL/TP ────────────────────────────────────────────────────
    ATR_PERIOD: int = 10
    ATR_SL_MULTIPLIER: float = 0.8
    MIN_SL_POINTS: float = 1.5
    MAX_SL_POINTS: float = 6.0

    # ─── Session RR (by session) ──────────────────────────────────
    LONDON_RR: float = 1.5
    OVERLAP_RR: float = 1.8
    NY_RR: float = 2.0
    DEFAULT_RR: float = 1.5

    # ─── Partial Close ────────────────────────────────────────────
    PARTIAL_PERCENT: float = 0.50
    TP1_RR: float = 1.0
    MOVE_SL_TO_BE: bool = True

    # ─── SMC ──────────────────────────────────────────────────────
    SWING_LOOKBACK: int = 3
    OB_MAX_AGE: int = 20
    FVG_MIN_SIZE_ATR: float = 0.2
    ENGULF_BODY_RATIO: float = 0.60
    MOMENTUM_CANDLE_ATR: float = 0.8
    EXHAUSTION_WICK_RATIO: float = 0.65

    # ─── EMA ──────────────────────────────────────────────────────
    EMA_FAST: int = 9
    EMA_SLOW: int = 21
    USE_EMA_FILTER: bool = True

    # ─── Mean Reversion ───────────────────────────────────────────
    USE_MEAN_REVERSION: bool = True
    BB_PERIOD: int = 20
    BB_STD_DEV: float = 2.0
    RSI_PERIOD: int = 7
    RSI_OVERSOLD: float = 25.0
    RSI_OVERBOUGHT: float = 75.0
    RSI_EXTREME_OVERSOLD: float = 15.0
    RSI_EXTREME_OVERBOUGHT: float = 85.0
    STOCH_RSI_PERIOD: int = 7
    STOCH_RSI_K: int = 3
    STOCH_RSI_OVERSOLD: float = 15.0
    STOCH_RSI_OVERBOUGHT: float = 85.0
    MR_REQUIRE_BB_TOUCH: bool = True
    MR_REQUIRE_RSI: bool = True
    MR_CONFLUENCE_SCORE: int = 2

    # ─── Round Numbers ────────────────────────────────────────────
    ROUND_NUMBER_INTERVAL: float = 50.0
    ROUND_NUMBER_ZONE: float = 3.0

    # ─── Confluence ───────────────────────────────────────────────
    MIN_CONFLUENCE: int = 3

    # ─── Sessions (UTC hours) ─────────────────────────────────────
    ASIA_START: int = 0
    ASIA_END: int = 7
    LONDON_START: int = 7
    LONDON_END: int = 12
    OVERLAP_START: int = 12
    OVERLAP_END: int = 15
    NY_START: int = 12
    NY_END: int = 17

    # ─── Cooldowns (in bars) ──────────────────────────────────────
    TRADE_COOLDOWN_BARS: int = 12      # ~2 min on 1M, ~1h on 5M
    LOSS_COOLDOWN_BARS: int = 30       # ~5 min on 1M


# ═══════════════════════════════════════════════════════════════════
#  ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

class Direction(Enum):
    LONG = "buy"
    SHORT = "sell"

class TradePhase(Enum):
    OPEN = "open"
    TP1_HIT = "tp1_hit"
    CLOSED = "closed"

@dataclass
class BacktestTrade:
    direction: Direction
    entry: float
    sl: float
    tp: float
    tp1: float
    lots: float
    entry_bar: int
    entry_time: str
    phase: TradePhase = TradePhase.OPEN
    exit_price: float = 0.0
    exit_bar: int = 0
    exit_time: str = ""
    pnl: float = 0.0
    reason: str = ""
    exit_reason: str = ""
    mae: float = 0.0           # max adverse excursion
    mfe: float = 0.0           # max favorable excursion
    remaining_lots: float = 0.0

    def __post_init__(self):
        self.remaining_lots = self.lots


# ═══════════════════════════════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════════════════════════════

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # EMA
    df["ema_fast"] = df["Close"].ewm(span=9).mean()
    df["ema_slow"] = df["Close"].ewm(span=21).mean()

    # ATR
    df["tr"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"] - df["Close"].shift(1))
        )
    )
    df["atr"] = df["tr"].rolling(10).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(7).mean()
    loss = (-delta.clip(upper=0)).rolling(7).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["bb_mid"] = df["Close"].rolling(20).mean()
    df["bb_std"] = df["Close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    # Candle metrics
    df["body"] = abs(df["Close"] - df["Open"])
    df["candle_range"] = df["High"] - df["Low"]
    df["upper_wick"] = df["High"] - df[["Close", "Open"]].max(axis=1)
    df["lower_wick"] = df[["Close", "Open"]].min(axis=1) - df["Low"]

    # Volume
    if "Volume" in df.columns:
        df["avg_volume"] = df["Volume"].rolling(20).mean()
    else:
        df["Volume"] = 0
        df["avg_volume"] = 0

    return df


def stoch_rsi(rsi_series: pd.Series, period: int = 7, k_smooth: int = 3) -> pd.Series:
    rsi_min = rsi_series.rolling(period).min()
    rsi_max = rsi_series.rolling(period).max()
    stoch = ((rsi_series - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)) * 100
    return stoch.rolling(k_smooth).mean()


# ═══════════════════════════════════════════════════════════════════
#  SWING DETECTION
# ═══════════════════════════════════════════════════════════════════

def detect_swings(df: pd.DataFrame, lookback: int = 3) -> Tuple[List[int], List[int]]:
    """Returns indices of swing highs and swing lows."""
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(df) - lookback):
        h = df["High"].iloc[i]
        l = df["Low"].iloc[i]

        is_high = all(
            h >= df["High"].iloc[i + j] and h >= df["High"].iloc[i - j]
            for j in range(1, lookback + 1)
        )
        is_low = all(
            l <= df["Low"].iloc[i + j] and l <= df["Low"].iloc[i - j]
            for j in range(1, lookback + 1)
        )

        if is_high:
            swing_highs.append(i)
        if is_low:
            swing_lows.append(i)

    return swing_highs, swing_lows


# ═══════════════════════════════════════════════════════════════════
#  SIGNAL DETECTION (mirrors live bot logic)
# ═══════════════════════════════════════════════════════════════════

def get_session(hour: int, cfg: BacktestConfig) -> str:
    if cfg.OVERLAP_START <= hour < cfg.OVERLAP_END:
        return "overlap"
    if cfg.LONDON_START <= hour < cfg.LONDON_END:
        return "london"
    if cfg.NY_START <= hour < cfg.NY_END:
        return "new_york"
    if cfg.ASIA_START <= hour < cfg.ASIA_END:
        return "asia"
    return "off"


def is_tradeable(hour: int, cfg: BacktestConfig) -> bool:
    session = get_session(hour, cfg)
    return session in ("london", "overlap", "new_york")


def get_rr(session: str, cfg: BacktestConfig) -> float:
    if session == "overlap":
        return cfg.OVERLAP_RR
    if session == "new_york":
        return cfg.NY_RR
    if session == "london":
        return cfg.LONDON_RR
    return cfg.DEFAULT_RR


def detect_signal(df: pd.DataFrame, i: int, cfg: BacktestConfig,
                  swing_highs: List[int], swing_lows: List[int]
                  ) -> Optional[Tuple[Direction, float, str]]:
    """
    Evaluate bar i for a trade signal. Returns (direction, confluence_score, reasons) or None.
    """
    if i < 50:  # need enough history
        return None

    price = df["Close"].iloc[i]
    atr = df["atr"].iloc[i]
    if pd.isna(atr) or atr <= 0:
        return None

    ema_f = df["ema_fast"].iloc[i]
    ema_s = df["ema_slow"].iloc[i]
    if pd.isna(ema_f) or pd.isna(ema_s):
        return None

    # Session check
    ts = df.index[i]
    if hasattr(ts, 'hour'):
        hour = ts.hour
    else:
        return None

    if not is_tradeable(hour, cfg):
        return None

    # ─── Build confluence ─────────────────────────────────────
    confluence = 0
    reasons = []
    votes = {Direction.LONG: 0, Direction.SHORT: 0}

    # 1. EMA trend
    if cfg.USE_EMA_FILTER:
        if ema_f > ema_s:
            votes[Direction.LONG] += 1
            reasons.append("ema_buy")
        else:
            votes[Direction.SHORT] += 1
            reasons.append("ema_sell")

    # 2. Liquidity sweep (check if price swept recent swing and closed back)
    recent_sh = [s for s in swing_highs if s < i and s > i - 20]
    recent_sl = [s for s in swing_lows if s < i and s > i - 20]

    last_candle_high = df["High"].iloc[i]
    last_candle_low = df["Low"].iloc[i]
    last_close = df["Close"].iloc[i]

    for si in recent_sl[-3:]:
        sl_price = df["Low"].iloc[si]
        if last_candle_low < sl_price and last_close > sl_price:
            votes[Direction.LONG] += 2
            reasons.append("liq_sweep")
            break

    for si in recent_sh[-3:]:
        sh_price = df["High"].iloc[si]
        if last_candle_high > sh_price and last_close < sh_price:
            votes[Direction.SHORT] += 2
            reasons.append("liq_sweep")
            break

    # 3. Order block (simplified: look for engulfing after structure)
    if i >= 2:
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        prev_body = abs(prev["Close"] - prev["Open"])
        curr_body = abs(curr["Close"] - curr["Open"])

        if (curr["Close"] > curr["Open"] and prev["Close"] < prev["Open"]
                and curr_body > prev_body * cfg.ENGULF_BODY_RATIO):
            votes[Direction.LONG] += 1
            reasons.append("bull_ob")
        elif (curr["Close"] < curr["Open"] and prev["Close"] > prev["Open"]
                and curr_body > prev_body * cfg.ENGULF_BODY_RATIO):
            votes[Direction.SHORT] += 1
            reasons.append("bear_ob")

    # 4. FVG detection
    if i >= 2:
        c1_high = df["High"].iloc[i - 2]
        c3_low = df["Low"].iloc[i]
        c1_low = df["Low"].iloc[i - 2]
        c3_high = df["High"].iloc[i]
        min_gap = atr * cfg.FVG_MIN_SIZE_ATR

        if c3_low > c1_high and (c3_low - c1_high) >= min_gap:
            if price <= c3_low and price >= c1_high:
                votes[Direction.LONG] += 1
                reasons.append("fvg")
        elif c1_low > c3_high and (c1_low - c3_high) >= min_gap:
            if price >= c3_high and price <= c1_low:
                votes[Direction.SHORT] += 1
                reasons.append("fvg")

    # 5. Momentum candle
    body = df["body"].iloc[i]
    total = df["candle_range"].iloc[i]
    if total > 0 and body / total >= cfg.ENGULF_BODY_RATIO and body >= atr * cfg.MOMENTUM_CANDLE_ATR:
        if df["Close"].iloc[i] > df["Open"].iloc[i]:
            votes[Direction.LONG] += 1
            reasons.append("momentum")
        else:
            votes[Direction.SHORT] += 1
            reasons.append("momentum")

    # 6. Exhaustion candle
    if total > 0:
        uw = df["upper_wick"].iloc[i]
        lw = df["lower_wick"].iloc[i]
        if uw / total >= cfg.EXHAUSTION_WICK_RATIO:
            votes[Direction.SHORT] += 1
            reasons.append("exhaustion")
        elif lw / total >= cfg.EXHAUSTION_WICK_RATIO:
            votes[Direction.LONG] += 1
            reasons.append("exhaustion")

    # 7. Round number
    nearest = round(price / cfg.ROUND_NUMBER_INTERVAL) * cfg.ROUND_NUMBER_INTERVAL
    if abs(price - nearest) <= cfg.ROUND_NUMBER_ZONE:
        confluence += 1
        reasons.append("round_num")

    # 8. Mean Reversion
    if cfg.USE_MEAN_REVERSION:
        bb_upper = df["bb_upper"].iloc[i]
        bb_lower = df["bb_lower"].iloc[i]
        bb_mid = df["bb_mid"].iloc[i]
        rsi = df["rsi"].iloc[i]

        if not (pd.isna(bb_upper) or pd.isna(rsi)):
            bb_range = bb_upper - bb_lower
            if bb_range > 0:
                pct_b = (price - bb_lower) / bb_range

                # Oversold
                if pct_b <= 0.05 and rsi <= cfg.RSI_OVERSOLD:
                    mr_score = 2 if rsi <= cfg.RSI_EXTREME_OVERSOLD else 1
                    votes[Direction.LONG] += cfg.MR_CONFLUENCE_SCORE
                    reasons.append(f"MR:bb_low+rsi_{rsi:.0f}")

                # Overbought
                elif pct_b >= 0.95 and rsi >= cfg.RSI_OVERBOUGHT:
                    mr_score = 2 if rsi >= cfg.RSI_EXTREME_OVERBOUGHT else 1
                    votes[Direction.SHORT] += cfg.MR_CONFLUENCE_SCORE
                    reasons.append(f"MR:bb_high+rsi_{rsi:.0f}")

    # ─── Determine direction ──────────────────────────────────
    long_score = votes[Direction.LONG]
    short_score = votes[Direction.SHORT]

    if long_score > short_score and long_score >= 1:
        direction = Direction.LONG
        confluence += long_score
    elif short_score > long_score and short_score >= 1:
        direction = Direction.SHORT
        confluence += short_score
    else:
        return None

    # EMA filter
    if cfg.USE_EMA_FILTER:
        if direction == Direction.LONG and ema_f <= ema_s:
            # Allow counter-trend only for strong mean reversion
            if "MR:" not in str(reasons):
                return None
        elif direction == Direction.SHORT and ema_f >= ema_s:
            if "MR:" not in str(reasons):
                return None

    # Min confluence
    if confluence < cfg.MIN_CONFLUENCE:
        return None

    reason_str = " | ".join(reasons)
    return direction, confluence, reason_str


# ═══════════════════════════════════════════════════════════════════
#  BACKTESTER ENGINE
# ═══════════════════════════════════════════════════════════════════

class Backtester:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.balance = cfg.START_BALANCE
        self.equity = cfg.START_BALANCE
        self.peak_balance = cfg.START_BALANCE
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0

        self.trades: List[BacktestTrade] = []
        self.active_trades: List[BacktestTrade] = []
        self.closed_trades: List[BacktestTrade] = []

        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.daily_date = ""
        self.consecutive_losses = 0
        self.last_trade_bar = -999
        self.last_loss_bar = -999

        # Equity curve
        self.equity_curve: List[dict] = []

        # Monthly tracking
        self.monthly_pnl: Dict[str, float] = {}

    def calc_lots(self, sl_dist: float) -> float:
        risk = self.balance * (self.cfg.RISK_PERCENT / 100)
        per_lot = 100.0
        if sl_dist <= 0:
            return 0.01
        lots = risk / (sl_dist * per_lot)
        return max(0.01, min(round(lots, 2), 0.5))

    def run(self, df: pd.DataFrame, mode: str = "short"):
        """Run backtest over the dataframe."""
        print(f"\n{'='*60}")
        print(f"  BACKTESTING: {mode.upper()} MODE")
        print(f"  Period: {df.index[0]} → {df.index[-1]}")
        print(f"  Bars: {len(df)} | Balance: ${self.cfg.START_BALANCE:,.2f}")
        print(f"{'='*60}\n")

        df = calculate_indicators(df)
        swing_highs, swing_lows = detect_swings(df, self.cfg.SWING_LOOKBACK)

        total_bars = len(df)
        report_interval = max(1, total_bars // 20)

        for i in range(50, len(df)):
            # Progress
            if (i - 50) % report_interval == 0:
                pct = ((i - 50) / (total_bars - 50)) * 100
                print(f"  Progress: {pct:.0f}% | Bar {i}/{total_bars} | "
                      f"Balance: ${self.balance:,.2f} | Trades: {len(self.closed_trades)}")

            bar_time = df.index[i]
            price = df["Close"].iloc[i]
            high = df["High"].iloc[i]
            low = df["Low"].iloc[i]
            atr = df["atr"].iloc[i]

            if pd.isna(atr) or price <= 0:
                continue

            # ─── Daily reset ──────────────────────────────────
            if hasattr(bar_time, 'date'):
                today = str(bar_time.date())
            else:
                today = str(bar_time)[:10]

            if today != self.daily_date:
                self.daily_date = today
                self.daily_trades = 0
                self.daily_pnl = 0.0

            # ─── Manage active trades ─────────────────────────
            for t in list(self.active_trades):
                # Track MAE/MFE
                if t.direction == Direction.LONG:
                    adverse = t.entry - low
                    favorable = high - t.entry
                else:
                    adverse = high - t.entry
                    favorable = t.entry - low

                t.mae = max(t.mae, adverse)
                t.mfe = max(t.mfe, favorable)

                # Check SL hit
                sl_hit = (
                    (t.direction == Direction.LONG and low <= t.sl) or
                    (t.direction == Direction.SHORT and high >= t.sl)
                )
                if sl_hit:
                    t.exit_price = t.sl
                    t.exit_bar = i
                    t.exit_time = str(bar_time)
                    t.phase = TradePhase.CLOSED
                    t.exit_reason = "SL"

                    if t.direction == Direction.LONG:
                        t.pnl = (t.sl - t.entry) * t.remaining_lots * 100
                    else:
                        t.pnl = (t.entry - t.sl) * t.remaining_lots * 100

                    t.pnl -= self.cfg.COMMISSION_PER_LOT * t.remaining_lots
                    self._close_trade(t)
                    continue

                # Check TP hit
                tp_hit = (
                    (t.direction == Direction.LONG and high >= t.tp) or
                    (t.direction == Direction.SHORT and low <= t.tp)
                )
                if tp_hit:
                    t.exit_price = t.tp
                    t.exit_bar = i
                    t.exit_time = str(bar_time)
                    t.phase = TradePhase.CLOSED
                    t.exit_reason = "TP"

                    if t.direction == Direction.LONG:
                        t.pnl = (t.tp - t.entry) * t.remaining_lots * 100
                    else:
                        t.pnl = (t.entry - t.tp) * t.remaining_lots * 100

                    t.pnl -= self.cfg.COMMISSION_PER_LOT * t.remaining_lots
                    self._close_trade(t)
                    continue

                # Check TP1 (partial close)
                if t.phase == TradePhase.OPEN:
                    tp1_hit = (
                        (t.direction == Direction.LONG and high >= t.tp1) or
                        (t.direction == Direction.SHORT and low <= t.tp1)
                    )
                    if tp1_hit:
                        close_lots = round(t.lots * self.cfg.PARTIAL_PERCENT, 2)
                        if close_lots >= 0.01:
                            # Book partial profit
                            if t.direction == Direction.LONG:
                                partial_pnl = (t.tp1 - t.entry) * close_lots * 100
                            else:
                                partial_pnl = (t.entry - t.tp1) * close_lots * 100

                            partial_pnl -= self.cfg.COMMISSION_PER_LOT * close_lots
                            self.balance += partial_pnl
                            self.daily_pnl += partial_pnl
                            t.pnl += partial_pnl
                            t.remaining_lots = round(t.remaining_lots - close_lots, 2)
                            t.phase = TradePhase.TP1_HIT

                            # Move SL to breakeven
                            if self.cfg.MOVE_SL_TO_BE:
                                t.sl = t.entry

            # ─── Record equity ────────────────────────────────
            unrealized = 0.0
            for t in self.active_trades:
                if t.direction == Direction.LONG:
                    unrealized += (price - t.entry) * t.remaining_lots * 100
                else:
                    unrealized += (t.entry - price) * t.remaining_lots * 100

            self.equity = self.balance + unrealized

            if self.equity > self.peak_balance:
                self.peak_balance = self.equity

            dd = self.peak_balance - self.equity
            dd_pct = (dd / self.peak_balance * 100) if self.peak_balance > 0 else 0
            if dd_pct > self.max_drawdown_pct:
                self.max_drawdown_pct = dd_pct
                self.max_drawdown = dd

            # Record for equity curve (every 10 bars to save memory)
            if i % 10 == 0:
                self.equity_curve.append({
                    "bar": i,
                    "time": str(bar_time),
                    "balance": round(self.balance, 2),
                    "equity": round(self.equity, 2),
                    "drawdown_pct": round(dd_pct, 2),
                })

            # Monthly PnL tracking
            if hasattr(bar_time, 'strftime'):
                month_key = bar_time.strftime("%Y-%m")
            else:
                month_key = str(bar_time)[:7]

            # ─── Gate checks ──────────────────────────────────
            if self.daily_trades >= self.cfg.MAX_DAILY_TRADES:
                continue
            if len(self.active_trades) >= self.cfg.MAX_CONCURRENT_TRADES:
                continue
            if self.daily_pnl <= -(self.balance * self.cfg.MAX_DAILY_LOSS_PERCENT / 100):
                continue
            if self.consecutive_losses >= self.cfg.MAX_CONSECUTIVE_LOSSES:
                if i - self.last_loss_bar < self.cfg.LOSS_COOLDOWN_BARS * 2:
                    continue
                self.consecutive_losses = 0
            if i - self.last_trade_bar < self.cfg.TRADE_COOLDOWN_BARS:
                continue
            if i - self.last_loss_bar < self.cfg.LOSS_COOLDOWN_BARS:
                continue

            # Total drawdown check
            total_dd = (self.cfg.START_BALANCE - self.balance) / self.cfg.START_BALANCE * 100
            if total_dd >= self.cfg.MAX_TOTAL_DRAWDOWN_PERCENT:
                continue

            # ─── Signal detection ─────────────────────────────
            signal = detect_signal(df, i, self.cfg, swing_highs, swing_lows)
            if not signal:
                continue

            direction, score, reason = signal

            # ─── Calculate SL/TP ──────────────────────────────
            sl_dist = max(atr * self.cfg.ATR_SL_MULTIPLIER, self.cfg.MIN_SL_POINTS)
            sl_dist = min(sl_dist, self.cfg.MAX_SL_POINTS)

            hour = bar_time.hour if hasattr(bar_time, 'hour') else 12
            session = get_session(hour, self.cfg)
            rr = get_rr(session, self.cfg)

            tp_dist = sl_dist * rr
            tp1_dist = sl_dist * self.cfg.TP1_RR

            entry = price + self.cfg.SIMULATED_SPREAD if direction == Direction.LONG else price

            if direction == Direction.LONG:
                sl = entry - sl_dist
                tp = entry + tp_dist
                tp1 = entry + tp1_dist
            else:
                sl = entry + sl_dist
                tp = entry - tp_dist
                tp1 = entry - tp1_dist

            lots = self.calc_lots(sl_dist)
            if lots < 0.01:
                continue

            # ─── Open trade ───────────────────────────────────
            trade = BacktestTrade(
                direction=direction,
                entry=entry,
                sl=sl,
                tp=tp,
                tp1=tp1,
                lots=lots,
                entry_bar=i,
                entry_time=str(bar_time),
                reason=reason,
            )

            self.active_trades.append(trade)
            self.trades.append(trade)
            self.daily_trades += 1
            self.last_trade_bar = i

        # Close remaining trades at last price
        last_price = df["Close"].iloc[-1]
        for t in list(self.active_trades):
            if t.direction == Direction.LONG:
                t.pnl += (last_price - t.entry) * t.remaining_lots * 100
            else:
                t.pnl += (t.entry - last_price) * t.remaining_lots * 100
            t.pnl -= self.cfg.COMMISSION_PER_LOT * t.remaining_lots
            t.exit_price = last_price
            t.exit_reason = "END"
            t.phase = TradePhase.CLOSED
            self._close_trade(t)

        print(f"\n  Backtest complete! {len(self.closed_trades)} trades executed.\n")

    def _close_trade(self, t: BacktestTrade):
        self.balance += t.pnl if t.exit_reason != "END" else t.pnl
        self.daily_pnl += t.pnl

        # Monthly PnL
        month = t.exit_time[:7] if t.exit_time else t.entry_time[:7]
        if month not in self.monthly_pnl:
            self.monthly_pnl[month] = 0.0
        self.monthly_pnl[month] += t.pnl

        if t.pnl > 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.last_loss_bar = t.exit_bar

        if t in self.active_trades:
            self.active_trades.remove(t)
        self.closed_trades.append(t)

    def report(self) -> dict:
        """Generate full performance report."""
        if not self.closed_trades:
            print("No trades executed!")
            return {}

        wins = [t for t in self.closed_trades if t.pnl > 0]
        losses = [t for t in self.closed_trades if t.pnl <= 0]

        total_profit = sum(t.pnl for t in wins)
        total_loss = abs(sum(t.pnl for t in losses))
        net_pnl = total_profit - total_loss

        win_rate = len(wins) / len(self.closed_trades) * 100
        profit_factor = total_profit / total_loss if total_loss > 0 else 99.0
        avg_win = total_profit / len(wins) if wins else 0
        avg_loss = total_loss / len(losses) if losses else 0
        avg_rr = avg_win / avg_loss if avg_loss > 0 else 0

        # Expectancy
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)

        # MAE/MFE stats
        avg_mae = np.mean([t.mae for t in self.closed_trades]) if self.closed_trades else 0
        avg_mfe = np.mean([t.mfe for t in self.closed_trades]) if self.closed_trades else 0

        # Longest streaks
        max_win_streak = 0
        max_loss_streak = 0
        curr_win = 0
        curr_loss = 0
        for t in self.closed_trades:
            if t.pnl > 0:
                curr_win += 1
                curr_loss = 0
                max_win_streak = max(max_win_streak, curr_win)
            else:
                curr_loss += 1
                curr_win = 0
                max_loss_streak = max(max_loss_streak, curr_loss)

        # Exit reason breakdown
        exit_reasons = {}
        for t in self.closed_trades:
            r = t.exit_reason
            if r not in exit_reasons:
                exit_reasons[r] = {"count": 0, "pnl": 0}
            exit_reasons[r]["count"] += 1
            exit_reasons[r]["pnl"] += t.pnl

        # Direction breakdown
        long_trades = [t for t in self.closed_trades if t.direction == Direction.LONG]
        short_trades = [t for t in self.closed_trades if t.direction == Direction.SHORT]
        long_wins = len([t for t in long_trades if t.pnl > 0])
        short_wins = len([t for t in short_trades if t.pnl > 0])

        # Session breakdown
        session_stats = {}
        for t in self.closed_trades:
            try:
                hour = int(t.entry_time[11:13])
                s = get_session(hour, self.cfg)
            except:
                s = "unknown"
            if s not in session_stats:
                session_stats[s] = {"trades": 0, "wins": 0, "pnl": 0.0}
            session_stats[s]["trades"] += 1
            if t.pnl > 0:
                session_stats[s]["wins"] += 1
            session_stats[s]["pnl"] += t.pnl

        report = {
            "total_trades": len(self.closed_trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "net_pnl": round(net_pnl, 2),
            "total_profit": round(total_profit, 2),
            "total_loss": round(total_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_rr": round(avg_rr, 2),
            "expectancy": round(expectancy, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "final_balance": round(self.balance, 2),
            "return_pct": round((self.balance - self.cfg.START_BALANCE) / self.cfg.START_BALANCE * 100, 2),
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "avg_mae": round(avg_mae, 2),
            "avg_mfe": round(avg_mfe, 2),
            "long_trades": len(long_trades),
            "long_win_rate": round(long_wins / max(len(long_trades), 1) * 100, 1),
            "short_trades": len(short_trades),
            "short_win_rate": round(short_wins / max(len(short_trades), 1) * 100, 1),
        }

        # ─── Print Report ─────────────────────────────────────
        print("=" * 60)
        print("  📊 BACKTEST RESULTS")
        print("=" * 60)
        print(f"""
  💰 Starting Balance:  ${self.cfg.START_BALANCE:>12,.2f}
  💰 Final Balance:     ${self.balance:>12,.2f}
  📈 Net P&L:           ${net_pnl:>+12,.2f}
  📈 Return:            {report['return_pct']:>+11.2f}%
  📉 Max Drawdown:      ${self.max_drawdown:>12,.2f} ({self.max_drawdown_pct:.2f}%)

  ─── TRADE STATISTICS ────────────────────────
  Total Trades:         {len(self.closed_trades):>8}
  Wins:                 {len(wins):>8}
  Losses:               {len(losses):>8}
  Win Rate:             {win_rate:>7.1f}%
  Profit Factor:        {profit_factor:>8.2f}
  Expectancy:           ${expectancy:>+11.2f}

  ─── AVERAGES ────────────────────────────────
  Avg Win:              ${avg_win:>+11.2f}
  Avg Loss:             ${avg_loss:>11.2f}
  Avg RR:               {avg_rr:>8.2f}
  Avg MAE:              ${avg_mae:>11.2f}
  Avg MFE:              ${avg_mfe:>11.2f}

  ─── STREAKS ─────────────────────────────────
  Max Win Streak:       {max_win_streak:>8}
  Max Loss Streak:      {max_loss_streak:>8}

  ─── DIRECTION ───────────────────────────────
  Long:   {len(long_trades):>4} trades | WR: {report['long_win_rate']:.1f}%
  Short:  {len(short_trades):>4} trades | WR: {report['short_win_rate']:.1f}%""")

        print("\n  ─── EXIT REASONS ────────────────────────────")
        for reason, data in sorted(exit_reasons.items()):
            print(f"  {reason:>6}: {data['count']:>5} trades | PnL: ${data['pnl']:>+10,.2f}")

        print("\n  ─── SESSION PERFORMANCE ─────────────────────")
        for session, data in sorted(session_stats.items()):
            wr = data['wins'] / max(data['trades'], 1) * 100
            print(f"  {session:>10}: {data['trades']:>4} trades | WR: {wr:.0f}% | PnL: ${data['pnl']:>+10,.2f}")

        print("\n  ─── MONTHLY P&L ─────────────────────────────")
        for month in sorted(self.monthly_pnl.keys()):
            pnl = self.monthly_pnl[month]
            emoji = "🟢" if pnl >= 0 else "🔴"
            print(f"  {emoji} {month}: ${pnl:>+10,.2f}")

        print("\n" + "=" * 60)

        return report

    def save_results(self, filename: str):
        """Save detailed results to JSON."""
        data = {
            "report": self.report(),
            "equity_curve": self.equity_curve[-500:],  # last 500 points
            "monthly_pnl": self.monthly_pnl,
            "trades": [
                {
                    "direction": t.direction.value,
                    "entry": t.entry,
                    "sl": t.sl,
                    "tp": t.tp,
                    "lots": t.lots,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason,
                    "pnl": round(t.pnl, 2),
                    "mae": round(t.mae, 2),
                    "mfe": round(t.mfe, 2),
                    "reason": t.reason,
                }
                for t in self.closed_trades[-200:]  # last 200 trades
            ],
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  📁 Results saved to: {filename}")


# ═══════════════════════════════════════════════════════════════════
#  DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════

def download_data(mode: str) -> pd.DataFrame:
    """Download XAUUSD data from yfinance."""
    ticker = "GC=F"  # Gold futures

    if mode == "short":
        print("\n📥 Downloading 60-day gold data (5-minute)...")
        df = yf.download(ticker, period="60d", interval="5m", progress=True)
        if df.empty:
            print("⚠️  5M data failed, trying 1H for 60 days...")
            df = yf.download(ticker, period="60d", interval="1h", progress=True)
    elif mode == "long":
        print("\n📥 Downloading 5-year gold data (1-hour)...")
        df = yf.download(ticker, period="5y", interval="1h", progress=True)
        if df.empty:
            print("⚠️  1H data failed, trying daily...")
            df = yf.download(ticker, period="5y", interval="1d", progress=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if df.empty:
        print("❌ No data downloaded!")
        sys.exit(1)

    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Remove timezone info for consistency
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Clean
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[df["Close"] > 0]

    print(f"✅ Downloaded {len(df)} bars: {df.index[0]} → {df.index[-1]}")
    return df


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="XAUUSD Gold Scalper Backtester")
    parser.add_argument("--mode", choices=["short", "long", "both"], default="both",
                        help="short=60d 5M, long=5y 1H, both=run both")
    parser.add_argument("--balance", type=float, default=10000,
                        help="Starting balance (default: $10,000)")
    parser.add_argument("--risk", type=float, default=0.5,
                        help="Risk per trade %% (default: 0.5)")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║        XAUUSD GOLD SCALPER — BACKTESTER v1.0                ║
╚══════════════════════════════════════════════════════════════╝
    """)

    cfg = BacktestConfig()
    cfg.START_BALANCE = args.balance
    cfg.RISK_PERCENT = args.risk

    modes = ["short", "long"] if args.mode == "both" else [args.mode]

    for mode in modes:
        # Download data
        df = download_data(mode)

        # Adjust cooldowns based on timeframe
        if mode == "long":
            cfg.TRADE_COOLDOWN_BARS = 2    # 2 bars = 2 hours
            cfg.LOSS_COOLDOWN_BARS = 5     # 5 hours
        else:
            cfg.TRADE_COOLDOWN_BARS = 12   # 12 bars = 1 hour on 5M
            cfg.LOSS_COOLDOWN_BARS = 30    # 2.5 hours on 5M

        # Run backtest
        bt = Backtester(cfg)
        bt.run(df, mode)
        report = bt.report()

        # Save results
        output_file = f"backtest_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        bt.save_results(output_file)

        print()


if __name__ == "__main__":
    main()
