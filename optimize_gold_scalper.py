"""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — v1.5.2 IMPROVEMENT FINDER        ║
║  Baseline: v1.5.2 (WR 77.2%, PF 1.63, 6 signalen)          ║
║  Test: 50+ variaties op strategy, indicators, risk mgmt     ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys, json, time, math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from enum import Enum
from collections import defaultdict

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("pip install pandas numpy"); sys.exit(1)
try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance"); sys.exit(1)


class Direction(Enum):
    LONG = "buy"
    SHORT = "sell"

class TradePhase(Enum):
    OPEN = "open"
    TP1_HIT = "tp1_hit"


@dataclass
class Config:
    LABEL: str = "v1.5.2 Base"
    START_BALANCE: float = 5000.0
    COMMISSION_PER_LOT: float = 7.0
    SIMULATED_SPREAD: float = 0.30
    RISK_PERCENT: float = 1.0
    MAX_DAILY_LOSS_PERCENT: float = 3.0
    MAX_TOTAL_DRAWDOWN_PERCENT: float = 10.0
    MAX_CONCURRENT_TRADES: int = 3
    MAX_DAILY_TRADES: int = 30
    MAX_CONSECUTIVE_LOSSES: int = 5

    ATR_SL_MULTIPLIER: float = 2.5
    MIN_SL_POINTS: float = 2.0
    MAX_SL_POINTS: float = 10.0
    RR_RATIO: float = 2.0
    PARTIAL_PERCENT: float = 0.67
    TP1_RR: float = 0.4
    MOVE_SL_TO_BE: bool = True

    SWING_LOOKBACK: int = 3
    ENGULF_BODY_RATIO: float = 0.60
    MOMENTUM_CANDLE_ATR: float = 0.8
    RSI_OVERSOLD: float = 25.0
    RSI_OVERBOUGHT: float = 75.0
    MR_CONFLUENCE_SCORE: int = 2
    MIN_CONFLUENCE: int = 3
    TRADE_COOLDOWN_BARS: int = 5
    LOSS_COOLDOWN_BARS: int = 12
    ADX_THRESHOLD: float = 25.0
    FVG_MIN_ATR: float = 0.25

    # v1.5.2 defaults
    SKIP_BAD_HOURS: bool = True
    BAD_HOURS: tuple = (7, 12, 16)

    # Tunable extras
    USE_RSI_GATE: bool = False        # block overbought buys / oversold sells
    RSI_GATE_HIGH: float = 70.0
    RSI_GATE_LOW: float = 30.0

    USE_ATR_GATE: bool = False        # skip if ATR too low (choppy)
    ATR_GATE_MIN: float = 3.0

    USE_TREND_GATE: bool = False      # only trade WITH the 1H trend
    TREND_EMA_PERIOD: int = 50

    USE_DAILY_TARGET: bool = False    # stop trading after daily target hit
    DAILY_TARGET: float = 200.0

    USE_DAILY_LOSS_CUT: bool = False  # tighter daily loss cut
    DAILY_LOSS_CUT: float = 100.0

    USE_SCALING: bool = False         # scale lots based on streak
    SCALE_AFTER_WINS: int = 3
    SCALE_MULT: float = 1.5

    USE_ANTI_SCALE: bool = False      # reduce lots after losses
    ANTI_SCALE_AFTER: int = 2
    ANTI_SCALE_MULT: float = 0.5

    USE_SESSION_WEIGHT: bool = False  # different risk per session
    LONDON_RISK_MULT: float = 1.0
    NY_RISK_MULT: float = 1.0

    USE_MOMENTUM_FILTER: bool = False # only trade if last 3 candles show direction
    MOM_LOOKBACK: int = 3

    ONLY_SHORTS: bool = False         # only short (shorts were +$1,200)
    ONLY_LONGS: bool = False

    TP2_ENABLED: bool = False         # second partial at 1.0R
    TP2_RR: float = 1.0
    TP2_PERCENT: float = 0.50


def calculate_indicators(df):
    df = df.copy()
    df["ema9"] = df["Close"].ewm(span=9).mean()
    df["ema21"] = df["Close"].ewm(span=21).mean()
    df["ema50"] = df["Close"].ewm(span=50).mean()
    df["tr"] = np.maximum(df["High"]-df["Low"], np.maximum(abs(df["High"]-df["Close"].shift(1)), abs(df["Low"]-df["Close"].shift(1))))
    df["atr"] = df["tr"].rolling(14).mean()
    d = df["Close"].diff()
    g = d.clip(lower=0).rolling(7).mean()
    l = (-d.clip(upper=0)).rolling(7).mean()
    rs = g / l.replace(0, np.nan)
    df["rsi"] = 100 - (100/(1+rs))
    d14 = df["Close"].diff()
    g14 = d14.clip(lower=0).rolling(14).mean()
    l14 = (-d14.clip(upper=0)).rolling(14).mean()
    rs14 = g14 / l14.replace(0, np.nan)
    df["rsi14"] = 100 - (100/(1+rs14))
    df["bb_mid"] = df["Close"].rolling(20).mean()
    df["bb_std"] = df["Close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2*df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2*df["bb_std"]
    df["body"] = abs(df["Close"]-df["Open"])
    df["candle_range"] = df["High"]-df["Low"]
    df["upper_wick"] = df["High"] - df[["Close","Open"]].max(axis=1)
    df["lower_wick"] = df[["Close","Open"]].min(axis=1) - df["Low"]
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr14 = df["tr"].rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.rolling(14).mean()
    # Momentum direction
    df["mom_dir"] = np.where(df["Close"] > df["Close"].shift(3), 1, -1)
    return df


def detect_swings(df, lookback=3):
    sh, sl = [], []
    for i in range(lookback, len(df)-lookback):
        h, l = df["High"].iloc[i], df["Low"].iloc[i]
        if all(h >= df["High"].iloc[i+j] and h >= df["High"].iloc[i-j] for j in range(1, lookback+1)): sh.append(i)
        if all(l <= df["Low"].iloc[i+j] and l <= df["Low"].iloc[i-j] for j in range(1, lookback+1)): sl.append(i)
    return sh, sl


def detect_signal(df, i, cfg, swing_highs, swing_lows):
    if i < 60: return None
    price = df["Close"].iloc[i]
    atr = df["atr"].iloc[i]
    if pd.isna(atr) or atr <= 0: return None
    ema9, ema21 = df["ema9"].iloc[i], df["ema21"].iloc[i]
    rsi = df["rsi"].iloc[i]
    if pd.isna(ema9) or pd.isna(ema21): return None
    ts = df.index[i]
    if not hasattr(ts, 'hour') or not (7 <= ts.hour < 17): return None
    if cfg.SKIP_BAD_HOURS and ts.hour in cfg.BAD_HOURS: return None

    # ATR gate
    if cfg.USE_ATR_GATE and atr < cfg.ATR_GATE_MIN: return None

    # Momentum filter
    if cfg.USE_MOMENTUM_FILTER and i >= cfg.MOM_LOOKBACK:
        mom = df["mom_dir"].iloc[i]
        # Will check after direction is determined

    confluence = 0; reasons = []
    votes = {Direction.LONG: 0, Direction.SHORT: 0}
    ch, cl, cc, co = df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i], df["Open"].iloc[i]
    body, total = df["body"].iloc[i], df["candle_range"].iloc[i]

    # 1. EMA
    if ema9 > ema21: votes[Direction.LONG] += 1; reasons.append("ema")
    else: votes[Direction.SHORT] += 1; reasons.append("ema")

    # 2. OB
    if i >= 2:
        prev, curr = df.iloc[i-1], df.iloc[i]
        pb, cb = abs(prev["Close"]-prev["Open"]), abs(curr["Close"]-curr["Open"])
        if curr["Close"]>curr["Open"] and prev["Close"]<prev["Open"] and cb>pb*cfg.ENGULF_BODY_RATIO and curr["Close"]>prev["High"]:
            votes[Direction.LONG] += 1; reasons.append("ob")
        elif curr["Close"]<curr["Open"] and prev["Close"]>prev["Open"] and cb>pb*cfg.ENGULF_BODY_RATIO and curr["Close"]<prev["Low"]:
            votes[Direction.SHORT] += 1; reasons.append("ob")

    # 3. FVG
    if i >= 2:
        c1h, c3l = df["High"].iloc[i-2], df["Low"].iloc[i]
        c1l, c3h = df["Low"].iloc[i-2], df["High"].iloc[i]
        mg = atr*cfg.FVG_MIN_ATR
        if c3l > c1h and (c3l-c1h) >= mg: votes[Direction.LONG] += 1; reasons.append("fvg")
        elif c1l > c3h and (c1l-c3h) >= mg: votes[Direction.SHORT] += 1; reasons.append("fvg")

    # 4. Momentum
    if total > 0 and body/total >= cfg.ENGULF_BODY_RATIO and body >= atr*cfg.MOMENTUM_CANDLE_ATR:
        if cc > co: votes[Direction.LONG] += 1; reasons.append("mom")
        else: votes[Direction.SHORT] += 1; reasons.append("mom")

    # 5. Mean Reversion
    bb_u, bb_l = df["bb_upper"].iloc[i], df["bb_lower"].iloc[i]
    if not (pd.isna(bb_u) or pd.isna(rsi)):
        bbr = bb_u - bb_l
        if bbr > 0:
            pct_b = (price-bb_l)/bbr
            if pct_b <= 0.05 and rsi <= cfg.RSI_OVERSOLD: votes[Direction.LONG] += cfg.MR_CONFLUENCE_SCORE; reasons.append("MR")
            elif pct_b >= 0.95 and rsi >= cfg.RSI_OVERBOUGHT: votes[Direction.SHORT] += cfg.MR_CONFLUENCE_SCORE; reasons.append("MR")

    # 6. RSI div
    rsi14 = df["rsi14"].iloc[i]
    if not pd.isna(rsi14):
        if rsi14 < 30 and cc > df["Close"].iloc[i-5]: votes[Direction.LONG] += 1; reasons.append("rsi_div")
        elif rsi14 > 70 and cc < df["Close"].iloc[i-5]: votes[Direction.SHORT] += 1; reasons.append("rsi_div")

    # 7. ADX
    adx = df["adx"].iloc[i]
    if not pd.isna(adx) and adx >= cfg.ADX_THRESHOLD:
        confluence += 1; reasons.append("adx")

    # Direction
    ls, ss = votes[Direction.LONG], votes[Direction.SHORT]
    if ls > ss and ls >= 1: direction = Direction.LONG; confluence += ls
    elif ss > ls and ss >= 1: direction = Direction.SHORT; confluence += ss
    else: return None

    # EMA filter
    if direction == Direction.LONG and ema9 <= ema21 and "MR" not in reasons: return None
    if direction == Direction.SHORT and ema9 >= ema21 and "MR" not in reasons: return None

    # Direction filters
    if cfg.ONLY_SHORTS and direction == Direction.LONG: return None
    if cfg.ONLY_LONGS and direction == Direction.SHORT: return None

    # RSI gate
    if cfg.USE_RSI_GATE:
        if direction == Direction.LONG and rsi > cfg.RSI_GATE_HIGH: return None
        if direction == Direction.SHORT and rsi < cfg.RSI_GATE_LOW: return None

    # Trend gate (EMA50)
    if cfg.USE_TREND_GATE:
        ema50 = df["ema50"].iloc[i]
        if not pd.isna(ema50):
            if direction == Direction.LONG and price < ema50 and "MR" not in reasons: return None
            if direction == Direction.SHORT and price > ema50 and "MR" not in reasons: return None

    # Momentum filter
    if cfg.USE_MOMENTUM_FILTER:
        mom = df["mom_dir"].iloc[i]
        if direction == Direction.LONG and mom < 0: return None
        if direction == Direction.SHORT and mom > 0: return None

    if confluence < cfg.MIN_CONFLUENCE: return None
    return direction, confluence, "|".join(reasons)


@dataclass
class Trade:
    direction: Direction; entry: float; sl: float; tp: float; tp1: float
    lots: float; bar: int; day: str = ""; hour: int = 0
    phase: TradePhase = TradePhase.OPEN; pnl: float = 0.0; remaining: float = 0.0
    tp2: float = 0.0
    def __post_init__(self): self.remaining = self.lots


def run_backtest(df, cfg, sh, sl):
    balance = cfg.START_BALANCE; peak = balance; max_dd = 0.0
    active, closed = [], []; daily_trades = 0; daily_date = ""
    consec_losses = 0; consec_wins = 0; ltb = -999; llb = -999
    daily_pnl_today = 0.0

    for i in range(60, len(df)):
        price, high, low = df["Close"].iloc[i], df["High"].iloc[i], df["Low"].iloc[i]
        atr = df["atr"].iloc[i]
        if pd.isna(atr) or price <= 0: continue
        ts = df.index[i]
        today = str(ts.date()) if hasattr(ts,'date') else str(ts)[:10]
        if today != daily_date:
            daily_date = today; daily_trades = 0; daily_pnl_today = 0.0

        for t in list(active):
            if (t.direction==Direction.LONG and low<=t.sl) or (t.direction==Direction.SHORT and high>=t.sl):
                t.pnl += ((t.sl-t.entry) if t.direction==Direction.LONG else (t.entry-t.sl))*t.remaining*100
                t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl; daily_pnl_today += t.pnl
                consec_losses += 1; consec_wins = 0; llb = i; active.remove(t); closed.append(t); continue
            if (t.direction==Direction.LONG and high>=t.tp) or (t.direction==Direction.SHORT and low<=t.tp):
                t.pnl += ((t.tp-t.entry) if t.direction==Direction.LONG else (t.entry-t.tp))*t.remaining*100
                t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl; daily_pnl_today += t.pnl
                consec_losses = 0; consec_wins += 1; active.remove(t); closed.append(t); continue
            if t.phase == TradePhase.OPEN:
                if (t.direction==Direction.LONG and high>=t.tp1) or (t.direction==Direction.SHORT and low<=t.tp1):
                    cl = round(t.lots*cfg.PARTIAL_PERCENT, 2)
                    if cl >= 0.01:
                        p = ((t.tp1-t.entry) if t.direction==Direction.LONG else (t.entry-t.tp1))*cl*100
                        p -= cfg.COMMISSION_PER_LOT*cl; balance += p; t.pnl += p; daily_pnl_today += p
                        t.remaining = round(t.remaining-cl, 2); t.phase = TradePhase.TP1_HIT
                        if cfg.MOVE_SL_TO_BE: t.sl = t.entry

        eq = balance + sum(((price-t.entry) if t.direction==Direction.LONG else (t.entry-price))*t.remaining*100 for t in active)
        if eq > peak: peak = eq
        dd = (peak-eq)/peak*100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd

        # Gates
        if daily_trades >= cfg.MAX_DAILY_TRADES: continue
        if len(active) >= cfg.MAX_CONCURRENT_TRADES: continue
        if consec_losses >= cfg.MAX_CONSECUTIVE_LOSSES:
            if i-llb < cfg.LOSS_COOLDOWN_BARS*2: continue
            consec_losses = 0
        if i-ltb < cfg.TRADE_COOLDOWN_BARS: continue
        if i-llb < cfg.LOSS_COOLDOWN_BARS: continue
        if (cfg.START_BALANCE-balance)/cfg.START_BALANCE*100 >= cfg.MAX_TOTAL_DRAWDOWN_PERCENT: continue

        # Daily target
        if cfg.USE_DAILY_TARGET and daily_pnl_today >= cfg.DAILY_TARGET: continue
        # Daily loss cut
        if cfg.USE_DAILY_LOSS_CUT and daily_pnl_today <= -cfg.DAILY_LOSS_CUT: continue

        signal = detect_signal(df, i, cfg, sh, sl)
        if not signal: continue
        direction, score, reason = signal

        sl_dist = max(atr*cfg.ATR_SL_MULTIPLIER, cfg.MIN_SL_POINTS)
        sl_dist = min(sl_dist, cfg.MAX_SL_POINTS)
        entry = price + cfg.SIMULATED_SPREAD if direction==Direction.LONG else price

        # Session-based risk
        risk_pct = cfg.RISK_PERCENT
        if cfg.USE_SESSION_WEIGHT and hasattr(ts, 'hour'):
            if 7 <= ts.hour < 12: risk_pct *= cfg.LONDON_RISK_MULT
            elif 12 <= ts.hour < 17: risk_pct *= cfg.NY_RISK_MULT

        # Scaling
        if cfg.USE_SCALING and consec_wins >= cfg.SCALE_AFTER_WINS:
            risk_pct *= cfg.SCALE_MULT
        if cfg.USE_ANTI_SCALE and consec_losses >= cfg.ANTI_SCALE_AFTER:
            risk_pct *= cfg.ANTI_SCALE_MULT

        if direction == Direction.LONG:
            s,t,t1 = entry-sl_dist, entry+sl_dist*cfg.RR_RATIO, entry+sl_dist*cfg.TP1_RR
        else:
            s,t,t1 = entry+sl_dist, entry-sl_dist*cfg.RR_RATIO, entry-sl_dist*cfg.TP1_RR

        lots = max(0.01, min(round((balance*risk_pct/100)/(sl_dist*100), 2), 0.5))
        trade = Trade(direction=direction, entry=entry, sl=s, tp=t, tp1=t1,
                      lots=lots, bar=i, day=today, hour=ts.hour if hasattr(ts,'hour') else 0)
        active.append(trade)
        daily_trades += 1; ltb = i

    if active:
        lp = df["Close"].iloc[-1]
        for t in active:
            t.pnl += ((lp-t.entry) if t.direction==Direction.LONG else (t.entry-lp))*t.remaining*100
            closed.append(t)

    if not closed: return None
    wins = [t for t in closed if t.pnl > 0]; losses = [t for t in closed if t.pnl <= 0]
    tw = sum(t.pnl for t in wins); tl = abs(sum(t.pnl for t in losses))

    # Weekly
    weekly_pnl = defaultdict(float)
    for t in closed:
        try:
            from datetime import datetime as dt2, timedelta
            d = dt2.strptime(t.day, "%Y-%m-%d")
            yr, wk, _ = d.isocalendar()
            weekly_pnl[f"{yr}-W{wk:02d}"] += t.pnl
        except: pass
    green_w = sum(1 for p in weekly_pnl.values() if p >= 0)
    red_w = sum(1 for p in weekly_pnl.values() if p < 0)

    # Monthly
    monthly_pnl = defaultdict(float)
    for t in closed: monthly_pnl[t.day[:7]] += t.pnl
    all_months_green = all(p >= 0 for p in monthly_pnl.values())

    # Daily
    daily_pnls = defaultdict(float)
    for t in closed: daily_pnls[t.day] += t.pnl
    trading_days = [d for d,p in daily_pnls.items() if abs(p) > 0.01]
    daily_vals = [daily_pnls[d] for d in trading_days]
    avg_day = sum(daily_vals)/max(len(daily_vals),1)
    days_100 = sum(1 for p in daily_vals if p >= 100)
    green_d = sum(1 for p in daily_vals if p > 0)

    return {
        "label": cfg.LABEL, "trades": len(closed),
        "wr": round(len(wins)/max(len(closed),1)*100,1),
        "pf": round(tw/tl,2) if tl > 0 else 99,
        "pnl": round(balance-cfg.START_BALANCE,2),
        "dd": round(max_dd,2),
        "avg_daily": round(avg_day,2),
        "days_100": days_100,
        "green_weeks": green_w, "red_weeks": red_w,
        "green_days": green_d, "trading_days": len(trading_days),
        "all_months_green": all_months_green,
        "avg_win": round(tw/max(len(wins),1),2),
        "avg_loss": round(tl/max(len(losses),1),2),
    }


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — v1.5.2 IMPROVEMENT FINDER        ║
║     🎯 Zoekt betere WR, PF, winst, consistentie             ║
║     50+ variaties op de huidige strategie                    ║
╚══════════════════════════════════════════════════════════════╝
    """)
    print("📥 Downloading 60-day gold data...")
    df = yf.download("GC=F", period="60d", interval="5m", progress=True)
    if df.empty: df = yf.download("GC=F", period="60d", interval="1h", progress=True)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    df = df.dropna(subset=["Open","High","Low","Close"])
    df = df[df["Close"]>0]
    print(f"✅ {len(df)} bars: {df.index[0]} → {df.index[-1]}")

    df = calculate_indicators(df)
    sh, sl = detect_swings(df, 3)
    start = time.time()

    tests = []

    # ═══════════════════════════════════════════════════════
    #  CATEGORY 1: BASELINE
    # ═══════════════════════════════════════════════════════
    print(f"\n  📊 Cat 1: Baseline")
    cfg = Config(); cfg.LABEL = "v1.5.2 BASE"
    tests.append(cfg)

    # ═══════════════════════════════════════════════════════
    #  CATEGORY 2: RISK MANAGEMENT
    # ═══════════════════════════════════════════════════════
    print(f"  📊 Cat 2: Risk Management ({8} tests)")

    # Daily target stop
    for target in [150, 200, 300]:
        c = Config(); c.LABEL = f"DailyTarget ${target}"; c.USE_DAILY_TARGET = True; c.DAILY_TARGET = target
        tests.append(c)

    # Daily loss cut
    for cut in [75, 100, 150]:
        c = Config(); c.LABEL = f"LossCut ${cut}"; c.USE_DAILY_LOSS_CUT = True; c.DAILY_LOSS_CUT = cut
        tests.append(c)

    # Anti-scale after losses
    c = Config(); c.LABEL = "AntiScale 2L→50%"; c.USE_ANTI_SCALE = True; c.ANTI_SCALE_AFTER = 2; c.ANTI_SCALE_MULT = 0.5
    tests.append(c)
    c = Config(); c.LABEL = "AntiScale 3L→50%"; c.USE_ANTI_SCALE = True; c.ANTI_SCALE_AFTER = 3; c.ANTI_SCALE_MULT = 0.5
    tests.append(c)

    # ═══════════════════════════════════════════════════════
    #  CATEGORY 3: ENTRY FILTERS
    # ═══════════════════════════════════════════════════════
    print(f"  📊 Cat 3: Entry Filters ({8} tests)")

    # RSI gate
    c = Config(); c.LABEL = "RSI Gate 65/35"; c.USE_RSI_GATE = True; c.RSI_GATE_HIGH = 65; c.RSI_GATE_LOW = 35
    tests.append(c)
    c = Config(); c.LABEL = "RSI Gate 70/30"; c.USE_RSI_GATE = True; c.RSI_GATE_HIGH = 70; c.RSI_GATE_LOW = 30
    tests.append(c)
    c = Config(); c.LABEL = "RSI Gate 75/25"; c.USE_RSI_GATE = True; c.RSI_GATE_HIGH = 75; c.RSI_GATE_LOW = 25
    tests.append(c)

    # ATR gate
    for atr_min in [2.0, 3.0, 4.0, 5.0]:
        c = Config(); c.LABEL = f"ATR Gate ≥{atr_min}"; c.USE_ATR_GATE = True; c.ATR_GATE_MIN = atr_min
        tests.append(c)

    # Momentum filter
    c = Config(); c.LABEL = "Mom Filter 3bar"; c.USE_MOMENTUM_FILTER = True; c.MOM_LOOKBACK = 3
    tests.append(c)

    # ═══════════════════════════════════════════════════════
    #  CATEGORY 4: TREND FILTERS
    # ═══════════════════════════════════════════════════════
    print(f"  📊 Cat 4: Trend Filters ({3} tests)")

    c = Config(); c.LABEL = "EMA50 Trend Gate"; c.USE_TREND_GATE = True
    tests.append(c)

    # Only shorts (since shorts made +$1,200 and longs -$206)
    c = Config(); c.LABEL = "SHORTS ONLY"; c.ONLY_SHORTS = True
    tests.append(c)
    c = Config(); c.LABEL = "LONGS ONLY"; c.ONLY_LONGS = True
    tests.append(c)

    # ═══════════════════════════════════════════════════════
    #  CATEGORY 5: SL/TP TUNING
    # ═══════════════════════════════════════════════════════
    print(f"  📊 Cat 5: SL/TP Tuning ({12} tests)")

    for sl_m in [2.0, 2.5, 3.0, 3.5]:
        for tp1 in [0.3, 0.4, 0.5]:
            c = Config(); c.LABEL = f"SL×{sl_m} TP1={tp1}R"
            c.ATR_SL_MULTIPLIER = sl_m; c.TP1_RR = tp1
            tests.append(c)

    # ═══════════════════════════════════════════════════════
    #  CATEGORY 6: PARTIAL CLOSE TUNING
    # ═══════════════════════════════════════════════════════
    print(f"  📊 Cat 6: Partial Close ({4} tests)")

    for partial in [0.50, 0.60, 0.67, 0.75]:
        c = Config(); c.LABEL = f"Partial {partial*100:.0f}%"; c.PARTIAL_PERCENT = partial
        tests.append(c)

    # ═══════════════════════════════════════════════════════
    #  CATEGORY 7: SESSION OPTIMIZATION
    # ═══════════════════════════════════════════════════════
    print(f"  📊 Cat 7: Session ({6} tests)")

    # Different bad hours combos
    c = Config(); c.LABEL = "Skip 7,12,16"; c.BAD_HOURS = (7, 12, 16)
    tests.append(c)
    c = Config(); c.LABEL = "Skip 7,12"; c.BAD_HOURS = (7, 12)
    tests.append(c)
    c = Config(); c.LABEL = "Skip 7,16"; c.BAD_HOURS = (7, 16)
    tests.append(c)
    c = Config(); c.LABEL = "Skip 12,16"; c.BAD_HOURS = (12, 16)
    tests.append(c)

    # Session weighting
    c = Config(); c.LABEL = "London 0.5x NY 1.5x"; c.USE_SESSION_WEIGHT = True; c.LONDON_RISK_MULT = 0.5; c.NY_RISK_MULT = 1.5
    tests.append(c)
    c = Config(); c.LABEL = "London 1.5x NY 0.5x"; c.USE_SESSION_WEIGHT = True; c.LONDON_RISK_MULT = 1.5; c.NY_RISK_MULT = 0.5
    tests.append(c)

    # ═══════════════════════════════════════════════════════
    #  CATEGORY 8: SCALING / STREAK
    # ═══════════════════════════════════════════════════════
    print(f"  📊 Cat 8: Streak Scaling ({4} tests)")

    c = Config(); c.LABEL = "Scale 3W→1.5x"; c.USE_SCALING = True; c.SCALE_AFTER_WINS = 3; c.SCALE_MULT = 1.5
    tests.append(c)
    c = Config(); c.LABEL = "Scale 4W→2x"; c.USE_SCALING = True; c.SCALE_AFTER_WINS = 4; c.SCALE_MULT = 2.0
    tests.append(c)
    c = Config(); c.LABEL = "Scale+Anti combo"; c.USE_SCALING = True; c.SCALE_AFTER_WINS = 3; c.SCALE_MULT = 1.5
    c.USE_ANTI_SCALE = True; c.ANTI_SCALE_AFTER = 2; c.ANTI_SCALE_MULT = 0.5
    tests.append(c)
    c = Config(); c.LABEL = "Conserv Scale+Anti"; c.USE_SCALING = True; c.SCALE_AFTER_WINS = 4; c.SCALE_MULT = 1.3
    c.USE_ANTI_SCALE = True; c.ANTI_SCALE_AFTER = 2; c.ANTI_SCALE_MULT = 0.7
    tests.append(c)

    # ═══════════════════════════════════════════════════════
    #  CATEGORY 9: COMBOS (best ideas together)
    # ═══════════════════════════════════════════════════════
    print(f"  📊 Cat 9: Best Combos ({5} tests)")

    c = Config(); c.LABEL = "RSI+LossCut"
    c.USE_RSI_GATE = True; c.RSI_GATE_HIGH = 70; c.RSI_GATE_LOW = 30
    c.USE_DAILY_LOSS_CUT = True; c.DAILY_LOSS_CUT = 100
    tests.append(c)

    c = Config(); c.LABEL = "RSI+Anti+LossCut"
    c.USE_RSI_GATE = True; c.RSI_GATE_HIGH = 70; c.RSI_GATE_LOW = 30
    c.USE_ANTI_SCALE = True; c.ANTI_SCALE_AFTER = 2; c.ANTI_SCALE_MULT = 0.5
    c.USE_DAILY_LOSS_CUT = True; c.DAILY_LOSS_CUT = 100
    tests.append(c)

    c = Config(); c.LABEL = "Full Defense"
    c.USE_RSI_GATE = True; c.RSI_GATE_HIGH = 70; c.RSI_GATE_LOW = 30
    c.USE_ANTI_SCALE = True; c.ANTI_SCALE_AFTER = 2; c.ANTI_SCALE_MULT = 0.5
    c.USE_DAILY_LOSS_CUT = True; c.DAILY_LOSS_CUT = 100
    c.USE_ATR_GATE = True; c.ATR_GATE_MIN = 3.0
    tests.append(c)

    c = Config(); c.LABEL = "Max Profit"
    c.USE_SCALING = True; c.SCALE_AFTER_WINS = 3; c.SCALE_MULT = 1.5
    c.USE_DAILY_TARGET = True; c.DAILY_TARGET = 300
    tests.append(c)

    c = Config(); c.LABEL = "Balanced Pro"
    c.USE_RSI_GATE = True; c.RSI_GATE_HIGH = 70; c.RSI_GATE_LOW = 30
    c.USE_ANTI_SCALE = True; c.ANTI_SCALE_AFTER = 2; c.ANTI_SCALE_MULT = 0.7
    c.USE_SCALING = True; c.SCALE_AFTER_WINS = 4; c.SCALE_MULT = 1.3
    tests.append(c)

    # ═══════════════════════════════════════════════════════
    #  RUN ALL TESTS
    # ═══════════════════════════════════════════════════════
    print(f"\n  Running {len(tests)} configurations...\n")

    results = []
    for i, cfg in enumerate(tests):
        r = run_backtest(df, cfg, sh, sl)
        if r:
            results.append(r)
            if i % 10 == 0 or i == len(tests)-1:
                print(f"  [{i+1}/{len(tests)}] {r['label']}: WR:{r['wr']}% PF:{r['pf']} ${r['pnl']:+,.0f} GW:{r['green_weeks']}")

    # ═══════════════════════════════════════════════════════
    #  RESULTS
    # ═══════════════════════════════════════════════════════
    baseline = [r for r in results if r["label"] == "v1.5.2 BASE"][0]

    print(f"\n{'='*90}")
    print(f"  🏆 ALLE {len(results)} RESULTATEN — GERANKT OP TOTALE PnL")
    print(f"{'='*90}")
    print(f"\n  {'#':>3} {'Config':<22} {'Trades':>6} {'WR':>6} {'PF':>5} {'$/Day':>7} {'PnL':>9} {'DD':>5} {'GW':>4} {'RW':>4} {'$100+':>5} {'AllM🟢':>6} {'vs Base':>9}")
    print(f"  {'-'*92}")

    for rank, r in enumerate(sorted(results, key=lambda x: x["pnl"], reverse=True), 1):
        diff = r["pnl"] - baseline["pnl"]
        marker = " ⬅️" if r["label"] == "v1.5.2 BASE" else ""
        am = "✅" if r["all_months_green"] else "❌"
        print(f"  {rank:>3} {r['label']:<22} {r['trades']:>6} {r['wr']:>5.1f}% {r['pf']:>4.2f} ${r['avg_daily']:>+5.0f} ${r['pnl']:>+8,.0f} {r['dd']:>4.1f}% {r['green_weeks']:>3} {r['red_weeks']:>3} {r['days_100']:>5} {am:>5} ${diff:>+8,.0f}{marker}")

    # Top improvements by category
    better = [r for r in results if r["pnl"] > baseline["pnl"] and r["label"] != "v1.5.2 BASE"]
    better_wr = [r for r in results if r["wr"] > baseline["wr"] and r["label"] != "v1.5.2 BASE"]
    better_consistency = [r for r in results if r["red_weeks"] < baseline["red_weeks"] and r["label"] != "v1.5.2 BASE"]
    better_daily = [r for r in results if r["avg_daily"] > baseline["avg_daily"] and r["label"] != "v1.5.2 BASE"]

    print(f"\n{'='*90}")
    print(f"  ⭐ TOP VERBETERINGEN PER CATEGORIE")
    print(f"{'='*90}")

    if better:
        top_pnl = sorted(better, key=lambda x: x["pnl"], reverse=True)[:5]
        print(f"\n  💰 MEER WINST (top 5):")
        for r in top_pnl:
            print(f"    {r['label']:<25} ${r['pnl']:>+8,.0f} (${r['pnl']-baseline['pnl']:>+,.0f} meer) | WR:{r['wr']}% PF:{r['pf']}")

    if better_wr:
        top_wr = sorted(better_wr, key=lambda x: x["wr"], reverse=True)[:5]
        print(f"\n  🎯 HOGERE WIN RATE (top 5):")
        for r in top_wr:
            print(f"    {r['label']:<25} WR:{r['wr']}% (+{r['wr']-baseline['wr']:.1f}%) | PF:{r['pf']} ${r['pnl']:>+,.0f}")

    if better_consistency:
        top_con = sorted(better_consistency, key=lambda x: x["red_weeks"])[:5]
        print(f"\n  📅 MEEST CONSISTENT (minste rode weken):")
        for r in top_con:
            print(f"    {r['label']:<25} {r['green_weeks']}G/{r['red_weeks']}R weken | ${r['pnl']:>+,.0f} | AllMonths🟢:{r['all_months_green']}")

    if better_daily:
        top_day = sorted(better_daily, key=lambda x: x["avg_daily"], reverse=True)[:5]
        print(f"\n  📈 MEER PER DAG (top 5):")
        for r in top_day:
            print(f"    {r['label']:<25} ${r['avg_daily']:>+.0f}/dag (+${r['avg_daily']-baseline['avg_daily']:.0f}) | $100+ dagen: {r['days_100']}")

    # Overall recommendation
    # Score each: normalize pnl, wr, pf, consistency
    for r in results:
        r["score"] = (
            (r["pnl"] / max(baseline["pnl"], 1)) * 30 +
            (r["wr"] / max(baseline["wr"], 1)) * 25 +
            (r["pf"] / max(baseline["pf"], 1)) * 25 +
            (r["green_weeks"] / max(r["green_weeks"] + r["red_weeks"], 1)) * 20
        )

    best = sorted(results, key=lambda x: x["score"], reverse=True)[0]

    print(f"""
{'='*90}
  🏆 AANBEVOLEN VERBETERING: {best['label']}
{'='*90}
  {'':>20} {'v1.5.2':>12} {'Winner':>12} {'Verschil':>12}
  {'WR':>20} {f"{baseline['wr']}%":>12} {f"{best['wr']}%":>12} {f"{best['wr']-baseline['wr']:+.1f}%":>12}
  {'PF':>20} {baseline['pf']:>12} {best['pf']:>12} {f"{best['pf']-baseline['pf']:+.2f}":>12}
  {'PnL':>20} {f"${baseline['pnl']:+,.0f}":>12} {f"${best['pnl']:+,.0f}":>12} {f"${best['pnl']-baseline['pnl']:+,.0f}":>12}
  {'$/Day':>20} {f"${baseline['avg_daily']:+,.0f}":>12} {f"${best['avg_daily']:+,.0f}":>12} {f"${best['avg_daily']-baseline['avg_daily']:+,.0f}":>12}
  {'DD':>20} {f"{baseline['dd']}%":>12} {f"{best['dd']}%":>12} {f"{best['dd']-baseline['dd']:+.1f}%":>12}
  {'Green Weeks':>20} {f"{baseline['green_weeks']}":>12} {f"{best['green_weeks']}":>12} {"":>12}
  {'Red Weeks':>20} {f"{baseline['red_weeks']}":>12} {f"{best['red_weeks']}":>12} {f"{best['red_weeks']-baseline['red_weeks']:+d}":>12}
  {'All Months 🟢':>20} {str(baseline['all_months_green']):>12} {str(best['all_months_green']):>12} {"":>12}
{'='*90}""")

    print(f"\n  ⏱️  Klaar in {time.time()-start:.0f}s")

    output = {"baseline": baseline, "winner": best,
              "all_results": sorted(results, key=lambda x: x["pnl"], reverse=True)}
    with open("improvement_finder.json","w") as f:
        json.dump(output, f, indent=2)
    print(f"  📁 Saved: improvement_finder.json")


if __name__ == "__main__":
    main()
