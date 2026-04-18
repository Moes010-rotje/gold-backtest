"""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — WEEKLY PERFORMANCE ANALYZER       ║
║  Elke week apart: WR, PnL, trades, beste/slechtste dag      ║
║  + Automatisch verbeteringen vinden voor consistentie        ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys, json, time, math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
from collections import defaultdict
from datetime import datetime, timedelta

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
    LABEL: str = "v1.5.1"
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
    SWEEP_WICK_ATR_MIN: float = 0.25
    FVG_MIN_ATR: float = 0.25
    DOUBLE_TOLERANCE_ATR: float = 0.3

    # Regime filter
    USE_REGIME_FILTER: bool = False
    REGIME_ATR_LOOKBACK: int = 50
    REGIME_LOW_VOL_MULT: float = 0.6

    # Session bias
    USE_SESSION_BIAS: bool = False
    LONDON_BIAS_WEIGHT: float = 1.0
    NY_BIAS_WEIGHT: float = 1.0

    # Consecutive loss pause
    USE_LOSS_PAUSE: bool = False
    LOSS_PAUSE_THRESHOLD: int = 3
    LOSS_PAUSE_BARS: int = 60


def calculate_indicators(df):
    df = df.copy()
    df["ema9"] = df["Close"].ewm(span=9).mean()
    df["ema21"] = df["Close"].ewm(span=21).mean()
    df["tr"] = np.maximum(df["High"]-df["Low"], np.maximum(abs(df["High"]-df["Close"].shift(1)), abs(df["Low"]-df["Close"].shift(1))))
    df["atr"] = df["tr"].rolling(14).mean()
    df["atr50"] = df["tr"].rolling(50).mean()
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
    # ADX
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr14 = df["tr"].rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.rolling(14).mean()
    # Trend direction (EMA slope over 20 bars)
    df["trend_slope"] = df["ema21"] - df["ema21"].shift(20)
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

    # Regime filter: skip low volatility
    if cfg.USE_REGIME_FILTER:
        atr50 = df["atr50"].iloc[i]
        if not pd.isna(atr50) and atr50 > 0:
            if atr < atr50 * cfg.REGIME_LOW_VOL_MULT:
                return None

    confluence = 0; reasons = []
    votes = {Direction.LONG: 0, Direction.SHORT: 0}
    ch, cl, cc, co = df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i], df["Open"].iloc[i]
    body, total = df["body"].iloc[i], df["candle_range"].iloc[i]

    # Determine session for bias
    hour = ts.hour
    session_weight = 1.0
    if cfg.USE_SESSION_BIAS:
        if 7 <= hour < 12: session_weight = cfg.LONDON_BIAS_WEIGHT
        elif 12 <= hour < 17: session_weight = cfg.NY_BIAS_WEIGHT

    # 1. EMA
    if ema9 > ema21: votes[Direction.LONG] += 1; reasons.append("ema")
    else: votes[Direction.SHORT] += 1; reasons.append("ema")

    # 2. Sweep
    for si in [s for s in swing_lows if s < i and s > i-25][-3:]:
        if cl < df["Low"].iloc[si] and cc > df["Low"].iloc[si]:
            wick = min(cc, co) - cl
            if wick > atr*cfg.SWEEP_WICK_ATR_MIN: votes[Direction.LONG] += 2; reasons.append("sweep"); break
    for si in [s for s in swing_highs if s < i and s > i-25][-3:]:
        if ch > df["High"].iloc[si] and cc < df["High"].iloc[si]:
            wick = ch - max(cc, co)
            if wick > atr*cfg.SWEEP_WICK_ATR_MIN: votes[Direction.SHORT] += 2; reasons.append("sweep"); break

    # 3. OB
    if i >= 2:
        prev, curr = df.iloc[i-1], df.iloc[i]
        pb, cb = abs(prev["Close"]-prev["Open"]), abs(curr["Close"]-curr["Open"])
        if curr["Close"]>curr["Open"] and prev["Close"]<prev["Open"] and cb>pb*cfg.ENGULF_BODY_RATIO and curr["Close"]>prev["High"]:
            votes[Direction.LONG] += 1; reasons.append("ob")
        elif curr["Close"]<curr["Open"] and prev["Close"]>prev["Open"] and cb>pb*cfg.ENGULF_BODY_RATIO and curr["Close"]<prev["Low"]:
            votes[Direction.SHORT] += 1; reasons.append("ob")

    # 4. FVG
    if i >= 2:
        c1h, c3l = df["High"].iloc[i-2], df["Low"].iloc[i]
        c1l, c3h = df["Low"].iloc[i-2], df["High"].iloc[i]
        mg = atr*cfg.FVG_MIN_ATR
        if c3l > c1h and (c3l-c1h) >= mg: votes[Direction.LONG] += 1; reasons.append("fvg")
        elif c1l > c3h and (c1l-c3h) >= mg: votes[Direction.SHORT] += 1; reasons.append("fvg")

    # 5. Momentum
    if total > 0 and body/total >= cfg.ENGULF_BODY_RATIO and body >= atr*cfg.MOMENTUM_CANDLE_ATR:
        if cc > co: votes[Direction.LONG] += 1; reasons.append("mom")
        else: votes[Direction.SHORT] += 1; reasons.append("mom")

    # 6. Mean Reversion
    bb_u, bb_l = df["bb_upper"].iloc[i], df["bb_lower"].iloc[i]
    if not (pd.isna(bb_u) or pd.isna(rsi)):
        bbr = bb_u - bb_l
        if bbr > 0:
            pct_b = (price-bb_l)/bbr
            if pct_b <= 0.05 and rsi <= cfg.RSI_OVERSOLD: votes[Direction.LONG] += cfg.MR_CONFLUENCE_SCORE; reasons.append("MR")
            elif pct_b >= 0.95 and rsi >= cfg.RSI_OVERBOUGHT: votes[Direction.SHORT] += cfg.MR_CONFLUENCE_SCORE; reasons.append("MR")

    # 7. RSI div
    rsi14 = df["rsi14"].iloc[i]
    if not pd.isna(rsi14):
        if rsi14 < 30 and cc > df["Close"].iloc[i-5]: votes[Direction.LONG] += 1; reasons.append("rsi_div")
        elif rsi14 > 70 and cc < df["Close"].iloc[i-5]: votes[Direction.SHORT] += 1; reasons.append("rsi_div")

    # 8. Double
    recent_sl = [s for s in swing_lows if s < i and s > i-30]
    recent_sh = [s for s in swing_highs if s < i and s > i-30]
    tol = atr * cfg.DOUBLE_TOLERANCE_ATR
    if len(recent_sl) >= 2:
        l1, l2 = df["Low"].iloc[recent_sl[-2]], df["Low"].iloc[recent_sl[-1]]
        if abs(l1-l2) < tol and price > max(l1,l2): votes[Direction.LONG] += 1; reasons.append("dbl")
    if len(recent_sh) >= 2:
        h1, h2 = df["High"].iloc[recent_sh[-2]], df["High"].iloc[recent_sh[-1]]
        if abs(h1-h2) < tol and price < min(h1,h2): votes[Direction.SHORT] += 1; reasons.append("dbl")

    # 9. ADX
    adx = df["adx"].iloc[i]
    if not pd.isna(adx) and adx >= cfg.ADX_THRESHOLD:
        confluence += 1; reasons.append("adx")

    ls, ss = votes[Direction.LONG], votes[Direction.SHORT]
    if ls > ss and ls >= 1: direction = Direction.LONG; confluence += ls
    elif ss > ls and ss >= 1: direction = Direction.SHORT; confluence += ss
    else: return None

    if direction == Direction.LONG and ema9 <= ema21 and "MR" not in reasons: return None
    if direction == Direction.SHORT and ema9 >= ema21 and "MR" not in reasons: return None
    if confluence < cfg.MIN_CONFLUENCE: return None
    return direction, confluence, "|".join(reasons)


@dataclass
class Trade:
    direction: Direction; entry: float; sl: float; tp: float; tp1: float
    lots: float; bar: int; sl_dist: float; day: str = ""; hour: int = 0
    phase: TradePhase = TradePhase.OPEN; pnl: float = 0.0; remaining: float = 0.0
    reasons: str = ""
    def __post_init__(self): self.remaining = self.lots


def run_backtest(df, cfg, sh, sl):
    balance = cfg.START_BALANCE; peak = balance; max_dd = 0.0
    active, closed = [], []; daily_trades = 0; daily_date = ""
    consec_losses = 0; ltb = -999; llb = -999
    loss_pause_until = -999

    for i in range(60, len(df)):
        price, high, low = df["Close"].iloc[i], df["High"].iloc[i], df["Low"].iloc[i]
        atr = df["atr"].iloc[i]
        if pd.isna(atr) or price <= 0: continue
        ts = df.index[i]
        today = str(ts.date()) if hasattr(ts,'date') else str(ts)[:10]
        if today != daily_date: daily_date = today; daily_trades = 0

        for t in list(active):
            if (t.direction==Direction.LONG and low<=t.sl) or (t.direction==Direction.SHORT and high>=t.sl):
                t.pnl += ((t.sl-t.entry) if t.direction==Direction.LONG else (t.entry-t.sl))*t.remaining*100
                t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl
                consec_losses += 1; llb = i; active.remove(t); closed.append(t)
                if cfg.USE_LOSS_PAUSE and consec_losses >= cfg.LOSS_PAUSE_THRESHOLD:
                    loss_pause_until = i + cfg.LOSS_PAUSE_BARS
                continue
            if (t.direction==Direction.LONG and high>=t.tp) or (t.direction==Direction.SHORT and low<=t.tp):
                t.pnl += ((t.tp-t.entry) if t.direction==Direction.LONG else (t.entry-t.tp))*t.remaining*100
                t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl
                consec_losses = 0; active.remove(t); closed.append(t); continue
            if t.phase == TradePhase.OPEN:
                if (t.direction==Direction.LONG and high>=t.tp1) or (t.direction==Direction.SHORT and low<=t.tp1):
                    cl = round(t.lots*cfg.PARTIAL_PERCENT, 2)
                    if cl >= 0.01:
                        p = ((t.tp1-t.entry) if t.direction==Direction.LONG else (t.entry-t.tp1))*cl*100
                        p -= cfg.COMMISSION_PER_LOT*cl; balance += p; t.pnl += p
                        t.remaining = round(t.remaining-cl, 2); t.phase = TradePhase.TP1_HIT
                        if cfg.MOVE_SL_TO_BE: t.sl = t.entry

        eq = balance + sum(((price-t.entry) if t.direction==Direction.LONG else (t.entry-price))*t.remaining*100 for t in active)
        if eq > peak: peak = eq
        dd = (peak-eq)/peak*100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd

        if daily_trades >= cfg.MAX_DAILY_TRADES: continue
        if len(active) >= cfg.MAX_CONCURRENT_TRADES: continue
        if consec_losses >= cfg.MAX_CONSECUTIVE_LOSSES:
            if i-llb < cfg.LOSS_COOLDOWN_BARS*2: continue
            consec_losses = 0
        if i-ltb < cfg.TRADE_COOLDOWN_BARS: continue
        if i-llb < cfg.LOSS_COOLDOWN_BARS: continue
        if cfg.USE_LOSS_PAUSE and i < loss_pause_until: continue
        if (cfg.START_BALANCE-balance)/cfg.START_BALANCE*100 >= cfg.MAX_TOTAL_DRAWDOWN_PERCENT: continue

        signal = detect_signal(df, i, cfg, sh, sl)
        if not signal: continue
        direction, score, reason = signal

        sl_dist = max(atr*cfg.ATR_SL_MULTIPLIER, cfg.MIN_SL_POINTS)
        sl_dist = min(sl_dist, cfg.MAX_SL_POINTS)
        entry = price + cfg.SIMULATED_SPREAD if direction==Direction.LONG else price
        if direction == Direction.LONG:
            s,t,t1 = entry-sl_dist, entry+sl_dist*cfg.RR_RATIO, entry+sl_dist*cfg.TP1_RR
        else:
            s,t,t1 = entry+sl_dist, entry-sl_dist*cfg.RR_RATIO, entry-sl_dist*cfg.TP1_RR

        lots = max(0.01, min(round((balance*cfg.RISK_PERCENT/100)/(sl_dist*100), 2), 0.5))
        trade = Trade(direction=direction, entry=entry, sl=s, tp=t, tp1=t1,
                      lots=lots, bar=i, sl_dist=sl_dist, day=today,
                      hour=ts.hour if hasattr(ts,'hour') else 0, reasons=reason)
        active.append(trade)
        daily_trades += 1; ltb = i

    if active:
        lp = df["Close"].iloc[-1]
        for t in active:
            t.pnl += ((lp-t.entry) if t.direction==Direction.LONG else (t.entry-lp))*t.remaining*100
            closed.append(t)

    return closed, balance, max_dd


def analyze_weekly(closed, cfg):
    """Break down trades by week."""
    weekly = defaultdict(lambda: {"trades":0,"wins":0,"losses":0,"pnl":0.0,"days":set(),"long_w":0,"long_l":0,"short_w":0,"short_l":0,"signals":defaultdict(int),"hourly_pnl":defaultdict(float)})

    for t in closed:
        try:
            dt = datetime.strptime(t.day, "%Y-%m-%d")
            # ISO week
            yr, wk, _ = dt.isocalendar()
            week_key = f"{yr}-W{wk:02d}"
            monday = dt - timedelta(days=dt.weekday())
            week_label = f"{week_key} ({monday.strftime('%d %b')})"
        except:
            week_label = "unknown"

        w = weekly[week_label]
        w["trades"] += 1
        w["pnl"] += t.pnl
        w["days"].add(t.day)
        w["hourly_pnl"][t.hour] += t.pnl

        for r in t.reasons.split("|"):
            if r: w["signals"][r] += 1

        if t.pnl > 0:
            w["wins"] += 1
            if t.direction == Direction.LONG: w["long_w"] += 1
            else: w["short_w"] += 1
        else:
            w["losses"] += 1
            if t.direction == Direction.LONG: w["long_l"] += 1
            else: w["short_l"] += 1

    return dict(weekly)


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — WEEKLY PERFORMANCE ANALYZER       ║
║     🎯 Elke week apart + verbeteringen vinden               ║
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
    start_time = time.time()

    # ═══════════════════════════════════════════════════════════
    #  PHASE 1: BASELINE WEEKLY BREAKDOWN
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  📊 PHASE 1: v1.5.1 BASELINE — WEEKLY BREAKDOWN")
    print(f"{'='*70}")

    cfg = Config()
    closed, final_bal, max_dd = run_backtest(df, cfg, sh, sl)
    weekly = analyze_weekly(closed, cfg)

    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    tw = sum(t.pnl for t in wins); tl = abs(sum(t.pnl for t in losses))
    total_wr = len(wins)/max(len(closed),1)*100
    total_pf = tw/tl if tl > 0 else 99

    print(f"\n  TOTAAL: {len(closed)} trades | WR:{total_wr:.1f}% | PF:{total_pf:.2f} | ${final_bal-cfg.START_BALANCE:+,.0f} | DD:{max_dd:.1f}%\n")

    green_weeks = 0; red_weeks = 0
    worst_week = None; best_week = None

    print(f"  {'Week':<25} {'Trades':>7} {'WR':>6} {'PnL':>10} {'Long W/L':>10} {'Short W/L':>10} {'Days':>5}")
    print(f"  {'-'*75}")

    for week in sorted(weekly.keys()):
        w = weekly[week]
        wr = w["wins"]/max(w["trades"],1)*100
        emoji = "🟢" if w["pnl"] >= 0 else "🔴"
        if w["pnl"] >= 0: green_weeks += 1
        else: red_weeks += 1
        if worst_week is None or w["pnl"] < weekly.get(worst_week,{}).get("pnl",0): worst_week = week
        if best_week is None or w["pnl"] > weekly.get(best_week,{}).get("pnl",0): best_week = week

        long_str = f"{w['long_w']}/{w['long_l']}"
        short_str = f"{w['short_w']}/{w['short_l']}"
        print(f"  {emoji} {week:<23} {w['trades']:>7} {wr:>5.0f}% ${w['pnl']:>+9,.0f} {long_str:>10} {short_str:>10} {len(w['days']):>5}")

    print(f"\n  Groene weken: {green_weeks} | Rode weken: {red_weeks} | Win%: {green_weeks/max(green_weeks+red_weeks,1)*100:.0f}%")

    # Analyze worst week
    if worst_week:
        ww = weekly[worst_week]
        print(f"\n  📉 SLECHTSTE WEEK: {worst_week}")
        print(f"     PnL: ${ww['pnl']:+,.2f} | WR: {ww['wins']/max(ww['trades'],1)*100:.0f}%")
        print(f"     Signalen: {dict(ww['signals'])}")
        print(f"     Uur-analyse:")
        for h in sorted(ww["hourly_pnl"].keys()):
            hp = ww["hourly_pnl"][h]
            emoji = "🟢" if hp >= 0 else "🔴"
            print(f"       {emoji} {h:02d}:00 UTC: ${hp:+.2f}")

    if best_week:
        bw = weekly[best_week]
        print(f"\n  📈 BESTE WEEK: {best_week}")
        print(f"     PnL: ${bw['pnl']:+,.2f} | WR: {bw['wins']/max(bw['trades'],1)*100:.0f}%")
        print(f"     Signalen: {dict(bw['signals'])}")

    # Overall session analysis
    print(f"\n  📊 SESSIE ANALYSE (alle weken):")
    hourly_pnl = defaultdict(float)
    hourly_trades = defaultdict(int)
    hourly_wins = defaultdict(int)
    for t in closed:
        hourly_pnl[t.hour] += t.pnl
        hourly_trades[t.hour] += 1
        if t.pnl > 0: hourly_wins[t.hour] += 1

    for h in sorted(hourly_pnl.keys()):
        wr = hourly_wins[h]/max(hourly_trades[h],1)*100
        emoji = "🟢" if hourly_pnl[h] >= 0 else "🔴"
        session = "London" if 7 <= h < 12 else "NY Overlap" if 12 <= h < 15 else "New York" if 15 <= h < 17 else "Off"
        print(f"    {emoji} {h:02d}:00 ({session:>11}): {hourly_trades[h]:>4}t | WR:{wr:>5.0f}% | ${hourly_pnl[h]:>+9,.0f}")

    # Signal performance
    print(f"\n  📊 SIGNAAL PERFORMANCE:")
    signal_pnl = defaultdict(float)
    signal_count = defaultdict(int)
    signal_wins = defaultdict(int)
    for t in closed:
        for r in t.reasons.split("|"):
            if r:
                signal_pnl[r] += t.pnl
                signal_count[r] += 1
                if t.pnl > 0: signal_wins[r] += 1

    for sig in sorted(signal_pnl.keys(), key=lambda x: signal_pnl[x], reverse=True):
        wr = signal_wins[sig]/max(signal_count[sig],1)*100
        emoji = "🟢" if signal_pnl[sig] >= 0 else "🔴"
        print(f"    {emoji} {sig:<10}: {signal_count[sig]:>5}t | WR:{wr:>5.0f}% | ${signal_pnl[sig]:>+9,.0f}")

    # Direction analysis per week
    total_long = sum(1 for t in closed if t.direction == Direction.LONG)
    total_short = sum(1 for t in closed if t.direction == Direction.SHORT)
    long_pnl = sum(t.pnl for t in closed if t.direction == Direction.LONG)
    short_pnl = sum(t.pnl for t in closed if t.direction == Direction.SHORT)
    long_wr = sum(1 for t in closed if t.direction == Direction.LONG and t.pnl > 0)/max(total_long,1)*100
    short_wr = sum(1 for t in closed if t.direction == Direction.SHORT and t.pnl > 0)/max(total_short,1)*100

    print(f"\n  📊 RICHTING:")
    print(f"    Long:  {total_long} trades | WR:{long_wr:.0f}% | ${long_pnl:+,.0f}")
    print(f"    Short: {total_short} trades | WR:{short_wr:.0f}% | ${short_pnl:+,.0f}")

    # ═══════════════════════════════════════════════════════════
    #  PHASE 2: TEST IMPROVEMENTS
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  🔧 PHASE 2: VERBETERINGEN TESTEN")
    print(f"{'='*70}")

    improvements = []

    # Test 1: Regime filter (skip low vol)
    print(f"\n  Testing: Regime filter (skip lage volatiliteit)...")
    for low_mult in [0.5, 0.6, 0.7]:
        cfg2 = Config(); cfg2.USE_REGIME_FILTER = True; cfg2.REGIME_LOW_VOL_MULT = low_mult
        cfg2.LABEL = f"Regime {low_mult}"
        c2, b2, d2 = run_backtest(df, cfg2, sh, sl)
        w2 = [t for t in c2 if t.pnl > 0]
        l2 = [t for t in c2 if t.pnl <= 0]
        wr2 = len(w2)/max(len(c2),1)*100
        pf2 = sum(t.pnl for t in w2)/max(abs(sum(t.pnl for t in l2)),0.01)
        pnl2 = b2 - cfg2.START_BALANCE
        improvements.append({"label":cfg2.LABEL,"trades":len(c2),"wr":wr2,"pf":round(pf2,2),"pnl":round(pnl2,2),"dd":round(d2,2)})
        print(f"    {cfg2.LABEL}: {len(c2)}t | WR:{wr2:.1f}% | PF:{pf2:.2f} | ${pnl2:+,.0f} | DD:{d2:.1f}%")

    # Test 2: Loss pause (langere pauze na 3 verliezen)
    print(f"\n  Testing: Loss pause na 3 verliezen...")
    for pause in [30, 60, 120]:
        cfg2 = Config(); cfg2.USE_LOSS_PAUSE = True; cfg2.LOSS_PAUSE_BARS = pause
        cfg2.LABEL = f"Pause {pause}bars"
        c2, b2, d2 = run_backtest(df, cfg2, sh, sl)
        w2 = [t for t in c2 if t.pnl > 0]; l2 = [t for t in c2 if t.pnl <= 0]
        wr2 = len(w2)/max(len(c2),1)*100
        pf2 = sum(t.pnl for t in w2)/max(abs(sum(t.pnl for t in l2)),0.01)
        pnl2 = b2 - cfg2.START_BALANCE
        improvements.append({"label":cfg2.LABEL,"trades":len(c2),"wr":wr2,"pf":round(pf2,2),"pnl":round(pnl2,2),"dd":round(d2,2)})
        print(f"    {cfg2.LABEL}: {len(c2)}t | WR:{wr2:.1f}% | PF:{pf2:.2f} | ${pnl2:+,.0f} | DD:{d2:.1f}%")

    # Test 3: Skip bad hours
    bad_hours = [h for h in hourly_pnl if hourly_pnl[h] < -50]
    if bad_hours:
        print(f"\n  Testing: Skip slechte uren {bad_hours}...")
        # Create version that skips bad hours
        cfg2 = Config(); cfg2.LABEL = f"Skip uren {bad_hours}"
        # We need custom signal detection for this, simulate by filtering
        c2_all, b2, d2 = run_backtest(df, cfg2, sh, sl)
        c2 = [t for t in c2_all if t.hour not in bad_hours]
        # Recalculate balance
        sim_bal = cfg2.START_BALANCE
        for t in c2: sim_bal += t.pnl
        w2 = [t for t in c2 if t.pnl > 0]; l2 = [t for t in c2 if t.pnl <= 0]
        wr2 = len(w2)/max(len(c2),1)*100
        pf2 = sum(t.pnl for t in w2)/max(abs(sum(t.pnl for t in l2)),0.01)
        pnl2 = sum(t.pnl for t in c2)
        improvements.append({"label":cfg2.LABEL,"trades":len(c2),"wr":wr2,"pf":round(pf2,2),"pnl":round(pnl2,2),"dd":round(d2,2)})
        print(f"    {cfg2.LABEL}: {len(c2)}t | WR:{wr2:.1f}% | PF:{pf2:.2f} | ${pnl2:+,.0f}")

    # Test 4: Different risk levels
    print(f"\n  Testing: Risk levels...")
    for risk in [0.75, 1.0, 1.5]:
        cfg2 = Config(); cfg2.RISK_PERCENT = risk; cfg2.LABEL = f"Risk {risk}%"
        c2, b2, d2 = run_backtest(df, cfg2, sh, sl)
        w2 = [t for t in c2 if t.pnl > 0]; l2 = [t for t in c2 if t.pnl <= 0]
        wr2 = len(w2)/max(len(c2),1)*100
        pf2 = sum(t.pnl for t in w2)/max(abs(sum(t.pnl for t in l2)),0.01)
        pnl2 = b2 - cfg2.START_BALANCE
        improvements.append({"label":cfg2.LABEL,"trades":len(c2),"wr":wr2,"pf":round(pf2,2),"pnl":round(pnl2,2),"dd":round(d2,2)})
        wk2 = analyze_weekly(c2, cfg2)
        red = sum(1 for w in wk2.values() if w["pnl"] < 0)
        print(f"    {cfg2.LABEL}: {len(c2)}t | WR:{wr2:.1f}% | PF:{pf2:.2f} | ${pnl2:+,.0f} | DD:{d2:.1f}% | {red} rode weken")

    # Test 5: Confluence 4
    print(f"\n  Testing: Hogere confluence...")
    for conf in [4, 5]:
        cfg2 = Config(); cfg2.MIN_CONFLUENCE = conf; cfg2.LABEL = f"Conf≥{conf}"
        c2, b2, d2 = run_backtest(df, cfg2, sh, sl)
        w2 = [t for t in c2 if t.pnl > 0]; l2 = [t for t in c2 if t.pnl <= 0]
        wr2 = len(w2)/max(len(c2),1)*100
        pf2 = sum(t.pnl for t in w2)/max(abs(sum(t.pnl for t in l2)),0.01)
        pnl2 = b2 - cfg2.START_BALANCE
        improvements.append({"label":cfg2.LABEL,"trades":len(c2),"wr":wr2,"pf":round(pf2,2),"pnl":round(pnl2,2),"dd":round(d2,2)})
        print(f"    {cfg2.LABEL}: {len(c2)}t | WR:{wr2:.1f}% | PF:{pf2:.2f} | ${pnl2:+,.0f} | DD:{d2:.1f}%")

    # ═══════════════════════════════════════════════════════════
    #  PHASE 3: RANKED RESULTS
    # ═══════════════════════════════════════════════════════════
    baseline = {"label":"v1.5.1 CURRENT","trades":len(closed),"wr":round(total_wr,1),"pf":round(total_pf,2),
                "pnl":round(final_bal-cfg.START_BALANCE,2),"dd":round(max_dd,2)}
    all_results = [baseline] + improvements

    print(f"\n{'='*70}")
    print(f"  🏆 ALLE RESULTATEN GERANKT (op PnL)")
    print(f"{'='*70}")
    print(f"\n  {'Config':<25} {'Trades':>7} {'WR':>6} {'PF':>6} {'PnL':>10} {'DD':>6} {'vs Base':>10}")
    print(f"  {'-'*72}")
    for r in sorted(all_results, key=lambda x: x["pnl"], reverse=True):
        diff = r["pnl"] - baseline["pnl"]
        marker = " ⬅️ HUIDIG" if r["label"] == "v1.5.1 CURRENT" else ""
        emoji = "🟢" if diff > 0 else "🔴" if diff < 0 else "⚪"
        print(f"  {emoji} {r['label']:<23} {r['trades']:>7} {r['wr']:>5.1f}% {r['pf']:>5.2f} ${r['pnl']:>+9,.0f} {r['dd']:>5.1f}% ${diff:>+9,.0f}{marker}")

    # Recommendations
    better = [r for r in improvements if r["pnl"] > baseline["pnl"] and r["pf"] >= 1.0]
    better.sort(key=lambda x: x["pnl"], reverse=True)

    print(f"\n{'='*70}")
    print(f"  ⭐ AANBEVELINGEN")
    print(f"{'='*70}")

    if better:
        best = better[0]
        print(f"""
  BESTE VERBETERING: {best['label']}
  ├── PnL: ${baseline['pnl']:+,.0f} → ${best['pnl']:+,.0f} (${best['pnl']-baseline['pnl']:+,.0f} meer)
  ├── WR: {baseline['wr']}% → {best['wr']}%
  ├── PF: {baseline['pf']} → {best['pf']}
  └── DD: {baseline['dd']}% → {best['dd']}%
""")
    else:
        print(f"\n  ⚠️ Geen verbetering gevonden die beter is dan v1.5.1")
        print(f"  De huidige strategie is al geoptimaliseerd.")

    # Weekly consistency advice
    if red_weeks > 0:
        red_week_data = [(w, weekly[w]) for w in sorted(weekly.keys()) if weekly[w]["pnl"] < 0]
        print(f"\n  📅 RODE WEKEN ANALYSE ({red_weeks} van {green_weeks+red_weeks}):")
        for wk, wd in red_week_data:
            print(f"    🔴 {wk}: ${wd['pnl']:+,.0f} | {wd['trades']}t | WR:{wd['wins']/max(wd['trades'],1)*100:.0f}%")
            print(f"       Signalen: {dict(wd['signals'])}")
    else:
        print(f"\n  ✅ ALLE WEKEN GROEN! Geen verbeteringen nodig.")

    print(f"\n  ⏱️  Klaar in {time.time()-start_time:.0f}s")

    # Save
    output = {
        "baseline": baseline,
        "weekly": {k: {"pnl":round(v["pnl"],2),"trades":v["trades"],"wins":v["wins"],"losses":v["losses"],
                       "wr":round(v["wins"]/max(v["trades"],1)*100,1)} for k,v in weekly.items()},
        "improvements": improvements,
        "green_weeks": green_weeks, "red_weeks": red_weeks,
    }
    with open("weekly_analysis.json","w") as f:
        json.dump(output, f, indent=2)
    print(f"  📁 Saved: weekly_analysis.json")


if __name__ == "__main__":
    main()
