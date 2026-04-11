"""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — DAILY PROFIT ANALYZER             ║
║  Target: $100-200 per dag consistent                         ║
║  Test: Risk levels, extra indicators, strategiewijzigingen   ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys, json, time
from dataclasses import dataclass, field
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
    CLOSED = "closed"


@dataclass
class Config:
    LABEL: str = "v1.5 Base"
    START_BALANCE: float = 5000.0
    COMMISSION_PER_LOT: float = 7.0
    SIMULATED_SPREAD: float = 0.30
    RISK_PERCENT: float = 0.5
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

    SWEEP_WICK_ATR_MIN: float = 0.25
    FVG_MIN_ATR: float = 0.25
    DOUBLE_TOLERANCE_ATR: float = 0.3
    ADX_THRESHOLD: float = 25.0

    # Extra toggles
    USE_MACD: bool = False
    USE_STOCH: bool = False
    USE_EMA50_FILTER: bool = False
    USE_HIGHER_TF_TREND: bool = False


def calculate_indicators(df):
    df = df.copy()
    df["ema9"] = df["Close"].ewm(span=9).mean()
    df["ema21"] = df["Close"].ewm(span=21).mean()
    df["ema50"] = df["Close"].ewm(span=50).mean()
    df["tr"] = np.maximum(df["High"]-df["Low"], np.maximum(abs(df["High"]-df["Close"].shift(1)), abs(df["Low"]-df["Close"].shift(1))))
    df["atr"] = df["tr"].rolling(14).mean()

    # RSI 7
    d = df["Close"].diff()
    g = d.clip(lower=0).rolling(7).mean()
    l = (-d.clip(upper=0)).rolling(7).mean()
    rs = g / l.replace(0, np.nan)
    df["rsi"] = 100 - (100/(1+rs))

    # RSI 14
    d14 = df["Close"].diff()
    g14 = d14.clip(lower=0).rolling(14).mean()
    l14 = (-d14.clip(upper=0)).rolling(14).mean()
    rs14 = g14 / l14.replace(0, np.nan)
    df["rsi14"] = 100 - (100/(1+rs14))

    # BB
    df["bb_mid"] = df["Close"].rolling(20).mean()
    df["bb_std"] = df["Close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2*df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2*df["bb_std"]

    # Candle
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

    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Stochastic
    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["stoch_k"] = ((df["Close"] - low14) / (high14 - low14).replace(0, np.nan)) * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # Higher TF trend (EMA50 slope)
    df["ema50_slope"] = df["ema50"] - df["ema50"].shift(10)

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

    confluence = 0; reasons = []
    votes = {Direction.LONG: 0, Direction.SHORT: 0}
    ch, cl, cc, co = df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i], df["Open"].iloc[i]
    body, total = df["body"].iloc[i], df["candle_range"].iloc[i]

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

    # 10. MACD (extra)
    if cfg.USE_MACD:
        macd_h = df["macd_hist"].iloc[i]
        macd_h_prev = df["macd_hist"].iloc[i-1]
        if not pd.isna(macd_h) and not pd.isna(macd_h_prev):
            if macd_h > 0 and macd_h > macd_h_prev: votes[Direction.LONG] += 1; reasons.append("macd")
            elif macd_h < 0 and macd_h < macd_h_prev: votes[Direction.SHORT] += 1; reasons.append("macd")

    # 11. Stochastic (extra)
    if cfg.USE_STOCH:
        stk = df["stoch_k"].iloc[i]
        std = df["stoch_d"].iloc[i]
        if not pd.isna(stk) and not pd.isna(std):
            if stk < 20 and stk > std: votes[Direction.LONG] += 1; reasons.append("stoch")
            elif stk > 80 and stk < std: votes[Direction.SHORT] += 1; reasons.append("stoch")

    # 12. EMA50 filter (extra)
    if cfg.USE_EMA50_FILTER:
        ema50 = df["ema50"].iloc[i]
        if not pd.isna(ema50):
            if price > ema50 and ema9 > ema50: votes[Direction.LONG] += 1; reasons.append("ema50")
            elif price < ema50 and ema9 < ema50: votes[Direction.SHORT] += 1; reasons.append("ema50")

    # 13. Higher TF trend (extra)
    if cfg.USE_HIGHER_TF_TREND:
        slope = df["ema50_slope"].iloc[i]
        if not pd.isna(slope):
            if slope > 0: votes[Direction.LONG] += 1; reasons.append("htf")
            elif slope < 0: votes[Direction.SHORT] += 1; reasons.append("htf")

    # Direction
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
    lots: float; bar: int; sl_dist: float; day: str = ""
    phase: TradePhase = TradePhase.OPEN
    pnl: float = 0.0; remaining: float = 0.0
    def __post_init__(self): self.remaining = self.lots


def run_backtest(df, cfg, sh, sl):
    balance = cfg.START_BALANCE; peak = balance; max_dd = 0.0
    active, closed = [], []; daily_trades = 0; daily_date = ""
    consec_losses = 0; ltb = -999; llb = -999
    daily_pnl_tracker: Dict[str, float] = {}

    for i in range(60, len(df)):
        price, high, low = df["Close"].iloc[i], df["High"].iloc[i], df["Low"].iloc[i]
        atr = df["atr"].iloc[i]
        if pd.isna(atr) or price <= 0: continue
        ts = df.index[i]
        today = str(ts.date()) if hasattr(ts,'date') else str(ts)[:10]
        if today != daily_date: daily_date = today; daily_trades = 0
        if today not in daily_pnl_tracker: daily_pnl_tracker[today] = 0.0

        for t in list(active):
            if (t.direction==Direction.LONG and low<=t.sl) or (t.direction==Direction.SHORT and high>=t.sl):
                t.pnl += ((t.sl-t.entry) if t.direction==Direction.LONG else (t.entry-t.sl))*t.remaining*100
                t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl
                daily_pnl_tracker[today] += t.pnl
                consec_losses += 1; llb = i; active.remove(t); closed.append(t); continue
            if (t.direction==Direction.LONG and high>=t.tp) or (t.direction==Direction.SHORT and low<=t.tp):
                t.pnl += ((t.tp-t.entry) if t.direction==Direction.LONG else (t.entry-t.tp))*t.remaining*100
                t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl
                daily_pnl_tracker[today] += t.pnl
                consec_losses = 0; active.remove(t); closed.append(t); continue
            if t.phase == TradePhase.OPEN:
                if (t.direction==Direction.LONG and high>=t.tp1) or (t.direction==Direction.SHORT and low<=t.tp1):
                    cl = round(t.lots*cfg.PARTIAL_PERCENT, 2)
                    if cl >= 0.01:
                        p = ((t.tp1-t.entry) if t.direction==Direction.LONG else (t.entry-t.tp1))*cl*100
                        p -= cfg.COMMISSION_PER_LOT*cl; balance += p; t.pnl += p
                        daily_pnl_tracker[today] += p
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
                      lots=lots, bar=i, sl_dist=sl_dist, day=today)
        active.append(trade)
        daily_trades += 1; ltb = i

    # Close remaining
    if active:
        lp = df["Close"].iloc[-1]
        for t in active:
            t.pnl += ((lp-t.entry) if t.direction==Direction.LONG else (t.entry-lp))*t.remaining*100
            today = t.day
            if today in daily_pnl_tracker: daily_pnl_tracker[today] += t.pnl
            closed.append(t)

    if not closed: return None

    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    tw = sum(t.pnl for t in wins); tl = abs(sum(t.pnl for t in losses))
    wr = len(wins)/len(closed)*100
    pf = tw/tl if tl > 0 else 99
    net = balance - cfg.START_BALANCE

    # Daily stats
    trading_days = [d for d, p in daily_pnl_tracker.items() if abs(p) > 0.01]
    daily_pnls = [daily_pnl_tracker[d] for d in trading_days]
    green_days = [p for p in daily_pnls if p > 0]
    red_days = [p for p in daily_pnls if p <= 0]
    days_100 = [p for p in daily_pnls if p >= 100]
    days_200 = [p for p in daily_pnls if p >= 200]
    days_50 = [p for p in daily_pnls if p >= 50]

    return {
        "label": cfg.LABEL,
        "risk": cfg.RISK_PERCENT,
        "trades": len(closed), "wr": round(wr,1), "pf": round(pf,2),
        "pnl": round(net,2), "dd": round(max_dd,2),
        "return_pct": round(net/cfg.START_BALANCE*100,2),
        "balance": round(balance,2),
        "avg_win": round(tw/max(len(wins),1),2),
        "avg_loss": round(tl/max(len(losses),1),2),
        "trading_days": len(trading_days),
        "green_days": len(green_days),
        "red_days": len(red_days),
        "green_pct": round(len(green_days)/max(len(trading_days),1)*100,1),
        "avg_daily": round(sum(daily_pnls)/max(len(trading_days),1),2),
        "best_day": round(max(daily_pnls, default=0),2),
        "worst_day": round(min(daily_pnls, default=0),2),
        "days_50": len(days_50),
        "days_100": len(days_100),
        "days_200": len(days_200),
        "median_day": round(np.median(daily_pnls),2) if daily_pnls else 0,
        "daily_pnls": {d: round(p,2) for d, p in sorted(daily_pnl_tracker.items()) if abs(p) > 0.01},
    }


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — DAILY PROFIT ANALYZER             ║
║     🎯 Target: $100-200 per dag consistent                  ║
║     Test: Risk levels + extra indicators                     ║
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

    # ═══════════════════════════════════════════════════════════
    #  TEST SUITE
    # ═══════════════════════════════════════════════════════════

    tests = []

    # 1. Risk level comparison
    for risk in [0.5, 0.75, 1.0, 1.5, 2.0]:
        cfg = Config()
        cfg.LABEL = f"Risk {risk}%"
        cfg.RISK_PERCENT = risk
        if risk >= 1.5: cfg.MAX_DAILY_LOSS_PERCENT = 5.0
        if risk >= 2.0: cfg.MAX_TOTAL_DRAWDOWN_PERCENT = 15.0
        tests.append(cfg)

    # 2. Risk 1% + MACD extra confluence
    cfg = Config(); cfg.LABEL = "1% + MACD"; cfg.RISK_PERCENT = 1.0; cfg.USE_MACD = True
    tests.append(cfg)

    # 3. Risk 1% + Stochastic
    cfg = Config(); cfg.LABEL = "1% + Stoch"; cfg.RISK_PERCENT = 1.0; cfg.USE_STOCH = True
    tests.append(cfg)

    # 4. Risk 1% + EMA50 filter
    cfg = Config(); cfg.LABEL = "1% + EMA50"; cfg.RISK_PERCENT = 1.0; cfg.USE_EMA50_FILTER = True
    tests.append(cfg)

    # 5. Risk 1% + Higher TF trend
    cfg = Config(); cfg.LABEL = "1% + HTF"; cfg.RISK_PERCENT = 1.0; cfg.USE_HIGHER_TF_TREND = True
    tests.append(cfg)

    # 6. Risk 1% + ALL extra indicators
    cfg = Config(); cfg.LABEL = "1% + ALL EXTRA"; cfg.RISK_PERCENT = 1.0
    cfg.USE_MACD = True; cfg.USE_STOCH = True; cfg.USE_EMA50_FILTER = True; cfg.USE_HIGHER_TF_TREND = True
    tests.append(cfg)

    # 7. Risk 1% + Confluence 4 (stricter)
    cfg = Config(); cfg.LABEL = "1% Conf≥4"; cfg.RISK_PERCENT = 1.0; cfg.MIN_CONFLUENCE = 4
    tests.append(cfg)

    # 8. Risk 1% + Max 4 concurrent
    cfg = Config(); cfg.LABEL = "1% Max4 Open"; cfg.RISK_PERCENT = 1.0; cfg.MAX_CONCURRENT_TRADES = 4
    tests.append(cfg)

    # 9. Risk 1% + Shorter cooldowns
    cfg = Config(); cfg.LABEL = "1% Fast CD"; cfg.RISK_PERCENT = 1.0
    cfg.TRADE_COOLDOWN_BARS = 3; cfg.LOSS_COOLDOWN_BARS = 5
    tests.append(cfg)

    # 10. Risk 1% + ALL extras + Conf 4 (quality + volume)
    cfg = Config(); cfg.LABEL = "1% QUALITY MAX"; cfg.RISK_PERCENT = 1.0
    cfg.USE_MACD = True; cfg.USE_STOCH = True; cfg.MIN_CONFLUENCE = 4
    tests.append(cfg)

    # Run all tests
    results = []
    print(f"\n  Running {len(tests)} configurations...\n")

    for i, cfg in enumerate(tests):
        print(f"  [{i+1}/{len(tests)}] {cfg.LABEL}...")
        r = run_backtest(df, cfg, sh, sl)
        if r:
            results.append(r)
            print(f"       WR:{r['wr']}% | PF:{r['pf']} | Avg/day:${r['avg_daily']:+.0f} | "
                  f"$100+ days:{r['days_100']} | DD:{r['dd']:.1f}%")

    # ═══════════════════════════════════════════════════════════
    #  RESULTS
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"  📊 DAILY PROFIT ANALYSIS — {len(results)} CONFIGURATIONS TESTED")
    print(f"{'='*80}")

    # Overview table
    print(f"\n  {'Config':<20} {'Risk':>5} {'WR':>6} {'PF':>5} {'$/Day':>8} {'$100+':>6} {'$200+':>6} {'Green%':>7} {'DD':>6} {'Total':>10}")
    print(f"  {'-'*85}")
    for r in sorted(results, key=lambda x: x["avg_daily"], reverse=True):
        print(f"  {r['label']:<20} {r['risk']:>4.1f}% {r['wr']:>5.1f}% {r['pf']:>4.2f} "
              f"${r['avg_daily']:>+6.0f} {r['days_100']:>5}d {r['days_200']:>5}d "
              f"{r['green_pct']:>5.0f}%  {r['dd']:>5.1f}% ${r['pnl']:>+9,.0f}")

    # Detailed daily breakdown for top 3
    top3 = sorted(results, key=lambda x: x["avg_daily"], reverse=True)[:3]

    for r in top3:
        print(f"\n  {'='*60}")
        print(f"  📅 DAILY BREAKDOWN: {r['label']}")
        print(f"  {'='*60}")
        print(f"""
  Avg per dag:    ${r['avg_daily']:>+8.2f}
  Mediaan dag:    ${r['median_day']:>+8.2f}
  Beste dag:      ${r['best_day']:>+8.2f}
  Slechtste dag:  ${r['worst_day']:>+8.2f}

  Trading dagen:  {r['trading_days']:>8}
  Groene dagen:   {r['green_days']:>8} ({r['green_pct']:.0f}%)
  Rode dagen:     {r['red_days']:>8}
  $50+ dagen:     {r['days_50']:>8}
  $100+ dagen:    {r['days_100']:>8}
  $200+ dagen:    {r['days_200']:>8}

  Dagelijks overzicht:""")

        # Show each day
        for day, pnl in sorted(r["daily_pnls"].items()):
            bars = "█" * min(int(abs(pnl) / 10), 30)
            emoji = "🟢" if pnl > 0 else "🔴"
            target = " ← $100+" if pnl >= 100 else " ← $200+!" if pnl >= 200 else ""
            print(f"    {emoji} {day}: ${pnl:>+8.2f} {bars}{target}")

    # Recommendation
    best = sorted(results, key=lambda x: x["avg_daily"], reverse=True)[0]

    # Find best for $100/day target
    target_100 = [r for r in results if r["days_100"] >= 5 and r["dd"] < 8]
    target_100.sort(key=lambda x: x["days_100"], reverse=True)

    print(f"""
{'='*80}
  ⭐ AANBEVELINGEN
{'='*80}

  🏆 HOOGSTE DAGELIJKSE WINST: {best['label']}
     ${best['avg_daily']:+.2f}/dag | {best['days_100']} dagen $100+ | DD: {best['dd']:.1f}%

  🎯 VOOR $100/DAG TARGET:""")

    if target_100:
        t = target_100[0]
        print(f"     {t['label']} → {t['days_100']} dagen ≥$100 van {t['trading_days']} dagen")
        print(f"     Risk: {t['risk']}% | WR: {t['wr']}% | PF: {t['pf']}")
    else:
        print(f"     ⚠️ Geen config haalt consistent $100/dag met <8% DD")
        print(f"     Groei je account naar $10,000+ voor dit target")

    print(f"""
  📈 GROEI PROJECTIE (met {best['label']}):
     Maand 1: $5,000 → ${5000 + best['avg_daily']*22:,.0f}
     Maand 2: → ${5000 + best['avg_daily']*44:,.0f}
     Maand 3: → ${5000 + best['avg_daily']*66:,.0f}
     (zonder compound effect)

  💡 MET COMPOUND (winst herbelegd):
     Maand 1: ${5000 * (1 + best['return_pct']/100 * 22/len(max(best['daily_pnls'],default={'a':1})))**1:,.0f}
     Let op: compound versnelt maar risico groeit mee

{'='*80}""")

    # Save
    output = {
        "results": [{k:v for k,v in r.items() if k != "daily_pnls"} for r in results],
        "recommendation": best["label"],
    }
    with open("daily_profit_analysis.json","w") as f:
        json.dump(output, f, indent=2)
    print(f"  📁 Saved: daily_profit_analysis.json")


if __name__ == "__main__":
    main()
