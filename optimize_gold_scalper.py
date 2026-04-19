"""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD RISK LEVELS + MULTI-PAIR SCANNER                 ║
║  Part 1: Gold 1% vs 1.5% vs 2% risk                         ║
║  Part 2: Scan 10+ pairs voor beste 2e bot                    ║
║  Zelfde v1.5.3 strategie op elk pair                         ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys, json, time
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
class PairConfig:
    SYMBOL: str = "GC=F"
    PAIR_NAME: str = "XAUUSD"
    PIP_VALUE_PER_LOT: float = 100.0   # gold: $1 per pip per lot × 100oz
    TYPICAL_SPREAD: float = 0.30
    MIN_SL: float = 2.0
    MAX_SL: float = 10.0

    START_BALANCE: float = 5000.0
    COMMISSION_PER_LOT: float = 7.0
    RISK_PERCENT: float = 1.0
    MAX_DAILY_LOSS_PERCENT: float = 3.0
    MAX_TOTAL_DRAWDOWN_PERCENT: float = 10.0
    MAX_CONCURRENT_TRADES: int = 3
    MAX_DAILY_TRADES: int = 30
    MAX_CONSECUTIVE_LOSSES: int = 5

    ATR_SL_MULTIPLIER: float = 2.5
    RR_RATIO: float = 2.0
    PARTIAL_PERCENT: float = 0.67
    TP1_RR: float = 0.5
    MOVE_SL_TO_BE: bool = True
    MIN_CONFLUENCE: int = 3
    TRADE_COOLDOWN_BARS: int = 5
    LOSS_COOLDOWN_BARS: int = 12
    ADX_THRESHOLD: float = 25.0

    ENGULF_BODY_RATIO: float = 0.60
    MOMENTUM_CANDLE_ATR: float = 0.8
    RSI_OVERSOLD: float = 25.0
    RSI_OVERBOUGHT: float = 75.0
    MR_CONFLUENCE_SCORE: int = 2
    FVG_MIN_ATR: float = 0.25

    SKIP_BAD_HOURS: bool = True
    BAD_HOURS: tuple = (7, 12, 16)


# ═══════════════════════════════════════════════════════
#  PAIR DEFINITIONS
# ═══════════════════════════════════════════════════════

PAIRS = [
    {"symbol": "GC=F",     "name": "XAUUSD (Gold)",  "pip_val": 100.0, "spread": 0.30, "min_sl": 2.0, "max_sl": 10.0},
    {"symbol": "GBPJPY=X", "name": "GBPJPY",         "pip_val": 6.5,   "spread": 0.025, "min_sl": 0.15, "max_sl": 0.80},
    {"symbol": "USDJPY=X", "name": "USDJPY",         "pip_val": 6.5,   "spread": 0.015, "min_sl": 0.10, "max_sl": 0.60},
    {"symbol": "EURUSD=X", "name": "EURUSD",         "pip_val": 10.0,  "spread": 0.00012, "min_sl": 0.0008, "max_sl": 0.0050},
    {"symbol": "GBPUSD=X", "name": "GBPUSD",         "pip_val": 10.0,  "spread": 0.00015, "min_sl": 0.0010, "max_sl": 0.0060},
    {"symbol": "EURJPY=X", "name": "EURJPY",         "pip_val": 6.5,   "spread": 0.020, "min_sl": 0.12, "max_sl": 0.70},
    {"symbol": "AUDUSD=X", "name": "AUDUSD",         "pip_val": 10.0,  "spread": 0.00012, "min_sl": 0.0006, "max_sl": 0.0040},
    {"symbol": "USDCHF=X", "name": "USDCHF",         "pip_val": 10.0,  "spread": 0.00015, "min_sl": 0.0008, "max_sl": 0.0050},
    {"symbol": "EURGBP=X", "name": "EURGBP",         "pip_val": 12.5,  "spread": 0.00012, "min_sl": 0.0005, "max_sl": 0.0035},
    {"symbol": "NQ=F",     "name": "USTEC (Nasdaq)",  "pip_val": 20.0,  "spread": 1.5, "min_sl": 10.0, "max_sl": 60.0},
    {"symbol": "YM=F",     "name": "US30 (Dow)",      "pip_val": 5.0,   "spread": 2.0, "min_sl": 15.0, "max_sl": 80.0},
    {"symbol": "SI=F",     "name": "XAGUSD (Silver)", "pip_val": 50.0,  "spread": 0.03, "min_sl": 0.10, "max_sl": 0.50},
]


def calculate_indicators(df):
    df = df.copy()
    df["ema9"] = df["Close"].ewm(span=9).mean()
    df["ema21"] = df["Close"].ewm(span=21).mean()
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
    return df


def detect_swings(df, lookback=3):
    sh, sl = [], []
    for i in range(lookback, len(df)-lookback):
        h, l = df["High"].iloc[i], df["Low"].iloc[i]
        if all(h >= df["High"].iloc[i+j] and h >= df["High"].iloc[i-j] for j in range(1, lookback+1)): sh.append(i)
        if all(l <= df["Low"].iloc[i+j] and l <= df["Low"].iloc[i-j] for j in range(1, lookback+1)): sl.append(i)
    return sh, sl


def detect_signal(df, i, cfg):
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

    confluence = 0; reasons = []
    votes = {Direction.LONG: 0, Direction.SHORT: 0}
    ch, cl, cc, co = df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i], df["Open"].iloc[i]
    body = df["body"].iloc[i]; total = df["candle_range"].iloc[i]

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
    lots: float; bar: int; day: str = ""
    phase: TradePhase = TradePhase.OPEN; pnl: float = 0.0; remaining: float = 0.0
    def __post_init__(self): self.remaining = self.lots


def run_backtest(df, cfg):
    balance = cfg.START_BALANCE; peak = balance; max_dd = 0.0
    active, closed = [], []; daily_trades = 0; daily_date = ""
    consec_losses = 0; ltb = -999; llb = -999
    pip_val = cfg.PIP_VALUE_PER_LOT

    for i in range(60, len(df)):
        price, high, low = df["Close"].iloc[i], df["High"].iloc[i], df["Low"].iloc[i]
        atr = df["atr"].iloc[i]
        if pd.isna(atr) or price <= 0: continue
        ts = df.index[i]
        today = str(ts.date()) if hasattr(ts,'date') else str(ts)[:10]
        if today != daily_date: daily_date = today; daily_trades = 0

        for t in list(active):
            if (t.direction==Direction.LONG and low<=t.sl) or (t.direction==Direction.SHORT and high>=t.sl):
                t.pnl += ((t.sl-t.entry) if t.direction==Direction.LONG else (t.entry-t.sl))*t.remaining*pip_val
                t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl
                consec_losses += 1; llb = i; active.remove(t); closed.append(t); continue
            if (t.direction==Direction.LONG and high>=t.tp) or (t.direction==Direction.SHORT and low<=t.tp):
                t.pnl += ((t.tp-t.entry) if t.direction==Direction.LONG else (t.entry-t.tp))*t.remaining*pip_val
                t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl
                consec_losses = 0; active.remove(t); closed.append(t); continue
            if t.phase == TradePhase.OPEN:
                if (t.direction==Direction.LONG and high>=t.tp1) or (t.direction==Direction.SHORT and low<=t.tp1):
                    cl = round(t.lots*cfg.PARTIAL_PERCENT, 2)
                    if cl >= 0.01:
                        p = ((t.tp1-t.entry) if t.direction==Direction.LONG else (t.entry-t.tp1))*cl*pip_val
                        p -= cfg.COMMISSION_PER_LOT*cl; balance += p; t.pnl += p
                        t.remaining = round(t.remaining-cl, 2); t.phase = TradePhase.TP1_HIT
                        if cfg.MOVE_SL_TO_BE: t.sl = t.entry

        eq = balance + sum(((price-t.entry) if t.direction==Direction.LONG else (t.entry-price))*t.remaining*pip_val for t in active)
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

        signal = detect_signal(df, i, cfg)
        if not signal: continue
        direction, score, reason = signal

        sl_dist = max(atr*cfg.ATR_SL_MULTIPLIER, cfg.MIN_SL)
        sl_dist = min(sl_dist, cfg.MAX_SL)
        entry = price + cfg.TYPICAL_SPREAD if direction==Direction.LONG else price
        if direction == Direction.LONG:
            s,t,t1 = entry-sl_dist, entry+sl_dist*cfg.RR_RATIO, entry+sl_dist*cfg.TP1_RR
        else:
            s,t,t1 = entry+sl_dist, entry-sl_dist*cfg.RR_RATIO, entry-sl_dist*cfg.TP1_RR

        lots = max(0.01, min(round((balance*cfg.RISK_PERCENT/100)/(sl_dist*pip_val), 2), 1.0))
        trade = Trade(direction=direction, entry=entry, sl=s, tp=t, tp1=t1, lots=lots, bar=i, day=today)
        active.append(trade)
        daily_trades += 1; ltb = i

    if active:
        lp = df["Close"].iloc[-1]
        for t in active:
            t.pnl += ((lp-t.entry) if t.direction==Direction.LONG else (t.entry-lp))*t.remaining*pip_val
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
    all_months_green = all(p >= 0 for p in monthly_pnl.values()) if monthly_pnl else False

    # Daily
    daily_pnls = defaultdict(float)
    for t in closed: daily_pnls[t.day] += t.pnl
    trading_days = [d for d,p in daily_pnls.items() if abs(p) > 0.01]
    daily_vals = [daily_pnls[d] for d in trading_days]
    avg_day = sum(daily_vals)/max(len(daily_vals),1)
    days_100 = sum(1 for p in daily_vals if p >= 100)
    days_50 = sum(1 for p in daily_vals if p >= 50)
    green_d = sum(1 for p in daily_vals if p > 0)
    best_day = max(daily_vals, default=0)
    worst_day = min(daily_vals, default=0)

    return {
        "pair": cfg.PAIR_NAME, "trades": len(closed),
        "wr": round(len(wins)/max(len(closed),1)*100,1),
        "pf": round(tw/tl,2) if tl > 0 else 99,
        "pnl": round(balance-cfg.START_BALANCE,2),
        "dd": round(max_dd,2),
        "avg_daily": round(avg_day,2),
        "days_100": days_100, "days_50": days_50,
        "green_weeks": green_w, "red_weeks": red_w,
        "green_days": green_d, "trading_days": len(trading_days),
        "all_months_green": all_months_green,
        "best_day": round(best_day,2), "worst_day": round(worst_day,2),
        "avg_win": round(tw/max(len(wins),1),2),
        "avg_loss": round(tl/max(len(losses),1),2),
        "monthly": {m: round(v,2) for m,v in sorted(monthly_pnl.items())},
        "risk": cfg.RISK_PERCENT,
    }


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     GOLD RISK LEVELS + MULTI-PAIR SCANNER                   ║
║     Part 1: Gold 1% vs 1.5% vs 2%                           ║
║     Part 2: 12 pairs scannen voor beste 2e bot              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    start = time.time()

    # ═══════════════════════════════════════════════════════
    #  PART 1: GOLD RISK LEVELS
    # ═══════════════════════════════════════════════════════
    print("=" * 70)
    print("  📊 PART 1: XAUUSD RISK LEVEL VERGELIJKING")
    print("=" * 70)

    print("\n  📥 Downloading XAUUSD 60-day data...")
    df_gold = yf.download("GC=F", period="60d", interval="5m", progress=False)
    if isinstance(df_gold.columns, pd.MultiIndex): df_gold.columns = df_gold.columns.get_level_values(0)
    if df_gold.index.tz is not None: df_gold.index = df_gold.index.tz_localize(None)
    df_gold = df_gold.dropna(subset=["Open","High","Low","Close"])
    df_gold = df_gold[df_gold["Close"]>0]
    df_gold = calculate_indicators(df_gold)
    print(f"  ✅ {len(df_gold)} bars")

    gold_results = []
    for risk in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        cfg = PairConfig()
        cfg.RISK_PERCENT = risk
        cfg.PAIR_NAME = f"XAUUSD {risk}%"
        r = run_backtest(df_gold, cfg)
        if r:
            gold_results.append(r)
            print(f"  Risk {risk}%: {r['trades']}t | WR:{r['wr']}% | PF:{r['pf']} | ${r['pnl']:+,.0f} | DD:{r['dd']:.1f}% | ${r['avg_daily']:+.0f}/day")

    print(f"\n  {'Risk':>8} {'Trades':>7} {'WR':>6} {'PF':>5} {'$/Day':>8} {'PnL':>10} {'DD':>6} {'$100+':>6} {'GW/RW':>7} {'Best Day':>10} {'Worst Day':>10}")
    print(f"  {'-'*90}")
    for r in gold_results:
        print(f"  {r['pair']:>8} {r['trades']:>7} {r['wr']:>5.1f}% {r['pf']:>4.2f} ${r['avg_daily']:>+6.0f} ${r['pnl']:>+9,.0f} {r['dd']:>5.1f}% {r['days_100']:>5} {r['green_weeks']:>2}/{r['red_weeks']:<2} ${r['best_day']:>+9,.0f} ${r['worst_day']:>+9,.0f}")

    # Monthly per risk
    print(f"\n  MAANDELIJKS per risk level:")
    for r in gold_results:
        months = " | ".join([f"{'🟢' if v>=0 else '🔴'}${v:+,.0f}" for m,v in sorted(r["monthly"].items())])
        print(f"  {r['pair']:>12}: {months}")

    # ═══════════════════════════════════════════════════════
    #  PART 2: MULTI-PAIR SCANNER
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  📊 PART 2: MULTI-PAIR SCANNER (v1.5.3 strategie op elk pair)")
    print(f"{'='*70}")

    pair_results = []

    for pair_info in PAIRS:
        sym = pair_info["symbol"]
        name = pair_info["name"]
        print(f"\n  📥 {name} ({sym})...", end=" ")

        try:
            df = yf.download(sym, period="60d", interval="5m", progress=False)
            if df.empty:
                df = yf.download(sym, period="60d", interval="1h", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            df = df.dropna(subset=["Open","High","Low","Close"])
            df = df[df["Close"]>0]

            if len(df) < 100:
                print(f"❌ Te weinig data ({len(df)} bars)")
                continue

            df = calculate_indicators(df)

            cfg = PairConfig()
            cfg.SYMBOL = sym
            cfg.PAIR_NAME = name
            cfg.PIP_VALUE_PER_LOT = pair_info["pip_val"]
            cfg.TYPICAL_SPREAD = pair_info["spread"]
            cfg.MIN_SL = pair_info["min_sl"]
            cfg.MAX_SL = pair_info["max_sl"]
            cfg.RISK_PERCENT = 1.0

            r = run_backtest(df, cfg)
            if r and r["trades"] > 0:
                pair_results.append(r)
                emoji = "🟢" if r["pnl"] > 0 else "🔴"
                print(f"{emoji} {r['trades']}t | WR:{r['wr']}% | PF:{r['pf']} | ${r['pnl']:+,.0f} | DD:{r['dd']:.1f}%")
            else:
                print(f"⚪ Geen trades")

        except Exception as e:
            print(f"❌ Error: {str(e)[:60]}")

    # ═══════════════════════════════════════════════════════
    #  PAIR RANKING
    # ═══════════════════════════════════════════════════════
    if pair_results:
        print(f"\n{'='*90}")
        print(f"  🏆 ALLE PAIRS GERANKT OP WINST")
        print(f"{'='*90}")
        print(f"\n  {'#':>3} {'Pair':<20} {'Trades':>7} {'WR':>6} {'PF':>5} {'$/Day':>8} {'PnL':>10} {'DD':>6} {'GW':>4} {'RW':>4} {'$100+':>5} {'AllM🟢':>6}")
        print(f"  {'-'*92}")

        for rank, r in enumerate(sorted(pair_results, key=lambda x: x["pnl"], reverse=True), 1):
            am = "✅" if r["all_months_green"] else "❌"
            emoji = "🟢" if r["pnl"] > 0 else "🔴"
            print(f"  {rank:>3} {emoji} {r['pair']:<18} {r['trades']:>7} {r['wr']:>5.1f}% {r['pf']:>4.2f} ${r['avg_daily']:>+6.0f} ${r['pnl']:>+9,.0f} {r['dd']:>5.1f}% {r['green_weeks']:>3} {r['red_weeks']:>3} {r['days_100']:>5} {am:>5}")

        # Rank by WR
        print(f"\n  🎯 GERANKT OP WIN RATE:")
        for rank, r in enumerate(sorted(pair_results, key=lambda x: x["wr"], reverse=True), 1):
            print(f"  {rank:>3} {r['pair']:<20} WR:{r['wr']:>5.1f}% | PF:{r['pf']} | ${r['pnl']:>+,.0f}")

        # Rank by consistency
        print(f"\n  📅 GERANKT OP CONSISTENTIE (minste rode weken):")
        for rank, r in enumerate(sorted(pair_results, key=lambda x: (x["red_weeks"], -x["pnl"])), 1):
            print(f"  {rank:>3} {r['pair']:<20} {r['green_weeks']}G/{r['red_weeks']}R | AllM🟢:{r['all_months_green']} | ${r['pnl']:>+,.0f}")

        # Monthly breakdown for top 3
        top3 = sorted(pair_results, key=lambda x: x["pnl"], reverse=True)[:3]
        print(f"\n  📅 MAANDELIJKS TOP 3:")
        for r in top3:
            months = " | ".join([f"{'🟢' if v>=0 else '🔴'}${v:+,.0f}" for m,v in sorted(r["monthly"].items())])
            print(f"  {r['pair']:<20}: {months}")

        # Recommendation
        profitable = [r for r in pair_results if r["pnl"] > 0 and r["pf"] >= 1.0]
        profitable.sort(key=lambda x: x["pnl"], reverse=True)

        print(f"\n{'='*70}")
        print(f"  ⭐ AANBEVELING VOOR 2E BOT")
        print(f"{'='*70}")

        if profitable:
            best = profitable[0]
            print(f"""
  🏆 BESTE PAIR: {best['pair']}
  ├── Trades: {best['trades']} | WR: {best['wr']}%
  ├── PF: {best['pf']} | PnL: ${best['pnl']:+,.2f}
  ├── $/dag: ${best['avg_daily']:+.2f}
  ├── DD: {best['dd']:.1f}% | $100+ dagen: {best['days_100']}
  ├── Groene weken: {best['green_weeks']}/{best['green_weeks']+best['red_weeks']}
  └── Alle maanden groen: {best['all_months_green']}""")

            if len(profitable) >= 2:
                second = profitable[1]
                print(f"""
  🥈 RUNNER UP: {second['pair']}
  ├── Trades: {second['trades']} | WR: {second['wr']}%
  ├── PF: {second['pf']} | PnL: ${second['pnl']:+,.2f}
  └── DD: {second['dd']:.1f}%""")

            # Combined projection
            gold_1pct = [r for r in gold_results if "1.0%" in r["pair"]][0] if gold_results else None
            if gold_1pct:
                combined_daily = gold_1pct["avg_daily"] + best["avg_daily"]
                print(f"""
  💰 GECOMBINEERD DAGELIJKS (Gold + {best['pair']}):
  ├── Gold: ${gold_1pct['avg_daily']:+.0f}/dag
  ├── {best['pair']}: ${best['avg_daily']:+.0f}/dag
  ├── TOTAAL: ${combined_daily:+.0f}/dag
  └── Maand: ${combined_daily*22:+,.0f}
""")
        else:
            print(f"\n  ⚠️ Geen pair gevonden met PF ≥ 1.0")
            print(f"  Overweeg de strategie aan te passen per pair")

    print(f"\n  ⏱️  Klaar in {time.time()-start:.0f}s")

    # Save
    output = {
        "gold_risk_levels": gold_results,
        "pair_scan": [{k:v for k,v in r.items() if k != "monthly"} for r in pair_results],
    }
    with open("risk_and_pairs.json","w") as f:
        json.dump(output, f, indent=2)
    print(f"  📁 Saved: risk_and_pairs.json")


if __name__ == "__main__":
    main()
