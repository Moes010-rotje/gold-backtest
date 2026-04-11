"""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — BACKTEST v1.5 + TRAILING STOP     ║
║  Test: v1.5 zonder trail vs v1.5 MET trail                  ║
║  Zie exact hoeveel de trailing runner toevoegt               ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys, json, time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum

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
    TRAILING = "trailing"
    CLOSED = "closed"


@dataclass
class Config:
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

    # Trailing stop
    USE_TRAILING: bool = False
    TRAIL_ACTIVATION_RR: float = 1.0
    TRAIL_ATR_MULT: float = 0.8
    TRAIL_STEP: float = 0.5
    REMOVE_TP: bool = True

    SWEEP_WICK_ATR_MIN: float = 0.25
    FVG_MIN_ATR: float = 0.25
    DOUBLE_TOLERANCE_ATR: float = 0.3
    ADX_THRESHOLD: float = 25.0


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

    # EMA
    if ema9 > ema21: votes[Direction.LONG] += 1; reasons.append("ema")
    else: votes[Direction.SHORT] += 1; reasons.append("ema")

    # Sweep
    for si in [s for s in swing_lows if s < i and s > i-25][-3:]:
        if cl < df["Low"].iloc[si] and cc > df["Low"].iloc[si]:
            wick = min(cc, co) - cl
            if wick > atr*cfg.SWEEP_WICK_ATR_MIN: votes[Direction.LONG] += 2; reasons.append("sweep"); break
    for si in [s for s in swing_highs if s < i and s > i-25][-3:]:
        if ch > df["High"].iloc[si] and cc < df["High"].iloc[si]:
            wick = ch - max(cc, co)
            if wick > atr*cfg.SWEEP_WICK_ATR_MIN: votes[Direction.SHORT] += 2; reasons.append("sweep"); break

    # OB
    if i >= 2:
        prev, curr = df.iloc[i-1], df.iloc[i]
        pb, cb = abs(prev["Close"]-prev["Open"]), abs(curr["Close"]-curr["Open"])
        if curr["Close"]>curr["Open"] and prev["Close"]<prev["Open"] and cb>pb*cfg.ENGULF_BODY_RATIO and curr["Close"]>prev["High"]:
            votes[Direction.LONG] += 1; reasons.append("ob")
        elif curr["Close"]<curr["Open"] and prev["Close"]>prev["Open"] and cb>pb*cfg.ENGULF_BODY_RATIO and curr["Close"]<prev["Low"]:
            votes[Direction.SHORT] += 1; reasons.append("ob")

    # FVG
    if i >= 2:
        c1h, c3l = df["High"].iloc[i-2], df["Low"].iloc[i]
        c1l, c3h = df["Low"].iloc[i-2], df["High"].iloc[i]
        mg = atr*cfg.FVG_MIN_ATR
        if c3l > c1h and (c3l-c1h) >= mg: votes[Direction.LONG] += 1; reasons.append("fvg")
        elif c1l > c3h and (c1l-c3h) >= mg: votes[Direction.SHORT] += 1; reasons.append("fvg")

    # Momentum
    if total > 0 and body/total >= cfg.ENGULF_BODY_RATIO and body >= atr*cfg.MOMENTUM_CANDLE_ATR:
        if cc > co: votes[Direction.LONG] += 1; reasons.append("mom")
        else: votes[Direction.SHORT] += 1; reasons.append("mom")

    # Mean Reversion
    bb_u, bb_l = df["bb_upper"].iloc[i], df["bb_lower"].iloc[i]
    if not (pd.isna(bb_u) or pd.isna(rsi)):
        bbr = bb_u - bb_l
        if bbr > 0:
            pct_b = (price-bb_l)/bbr
            if pct_b <= 0.05 and rsi <= cfg.RSI_OVERSOLD: votes[Direction.LONG] += cfg.MR_CONFLUENCE_SCORE; reasons.append("MR")
            elif pct_b >= 0.95 and rsi >= cfg.RSI_OVERBOUGHT: votes[Direction.SHORT] += cfg.MR_CONFLUENCE_SCORE; reasons.append("MR")

    # RSI div
    rsi14 = df["rsi14"].iloc[i]
    if not pd.isna(rsi14):
        if rsi14 < 30 and cc > df["Close"].iloc[i-5]: votes[Direction.LONG] += 1; reasons.append("rsi_div")
        elif rsi14 > 70 and cc < df["Close"].iloc[i-5]: votes[Direction.SHORT] += 1; reasons.append("rsi_div")

    # Double
    recent_sl = [s for s in swing_lows if s < i and s > i-30]
    recent_sh = [s for s in swing_highs if s < i and s > i-30]
    tol = atr * cfg.DOUBLE_TOLERANCE_ATR
    if len(recent_sl) >= 2:
        l1, l2 = df["Low"].iloc[recent_sl[-2]], df["Low"].iloc[recent_sl[-1]]
        if abs(l1-l2) < tol and price > max(l1,l2): votes[Direction.LONG] += 1; reasons.append("dbl")
    if len(recent_sh) >= 2:
        h1, h2 = df["High"].iloc[recent_sh[-2]], df["High"].iloc[recent_sh[-1]]
        if abs(h1-h2) < tol and price < min(h1,h2): votes[Direction.SHORT] += 1; reasons.append("dbl")

    # ADX
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
    lots: float; bar: int; sl_dist: float; entry_time: str = ""
    phase: TradePhase = TradePhase.OPEN
    pnl: float = 0.0; remaining: float = 0.0
    exit_reason: str = ""; max_profit_r: float = 0.0
    trail_activated: bool = False
    def __post_init__(self): self.remaining = self.lots


def run_backtest(df, cfg, sh, sl, label=""):
    balance = cfg.START_BALANCE; peak = balance; max_dd = 0.0
    active, closed = [], []; daily_trades = 0; daily_date = ""
    consec_losses = 0; ltb = -999; llb = -999
    monthly_pnl = {}
    big_winners = []  # trades that made $50+

    total_bars = len(df)
    report_interval = max(1, total_bars // 10)

    for i in range(60, len(df)):
        if (i-60) % report_interval == 0:
            pct = (i-60)/(total_bars-60)*100
            print(f"  {pct:.0f}% | Balance: ${balance:,.2f} | Trades: {len(closed)}")

        price, high, low = df["Close"].iloc[i], df["High"].iloc[i], df["Low"].iloc[i]
        atr = df["atr"].iloc[i]
        if pd.isna(atr) or price <= 0: continue
        ts = df.index[i]
        today = str(ts.date()) if hasattr(ts,'date') else str(ts)[:10]
        if today != daily_date: daily_date = today; daily_trades = 0

        for t in list(active):
            # Check SL
            if (t.direction==Direction.LONG and low<=t.sl) or (t.direction==Direction.SHORT and high>=t.sl):
                if t.direction == Direction.LONG:
                    t.pnl += (t.sl-t.entry)*t.remaining*100
                else:
                    t.pnl += (t.entry-t.sl)*t.remaining*100
                t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl
                t.exit_reason = "TRAIL_SL" if t.trail_activated else "SL"
                consec_losses += 1 if not t.trail_activated else 0; llb = i
                active.remove(t); closed.append(t)
                if t.pnl >= 50: big_winners.append(t)
                continue

            # Check TP (only if trailing not active or TP not removed)
            if not (cfg.USE_TRAILING and t.trail_activated and cfg.REMOVE_TP):
                if (t.direction==Direction.LONG and high>=t.tp) or (t.direction==Direction.SHORT and low<=t.tp):
                    if t.direction == Direction.LONG:
                        t.pnl += (t.tp-t.entry)*t.remaining*100
                    else:
                        t.pnl += (t.entry-t.tp)*t.remaining*100
                    t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl
                    t.exit_reason = "TP"; consec_losses = 0
                    active.remove(t); closed.append(t)
                    if t.pnl >= 50: big_winners.append(t)
                    continue

            # Track max profit in R
            if t.direction == Direction.LONG:
                curr_r = (high - t.entry) / t.sl_dist if t.sl_dist > 0 else 0
            else:
                curr_r = (t.entry - low) / t.sl_dist if t.sl_dist > 0 else 0
            if curr_r > t.max_profit_r: t.max_profit_r = curr_r

            # Partial close at TP1
            if t.phase == TradePhase.OPEN:
                if (t.direction==Direction.LONG and high>=t.tp1) or (t.direction==Direction.SHORT and low<=t.tp1):
                    cl = round(t.lots*cfg.PARTIAL_PERCENT, 2)
                    if cl >= 0.01:
                        if t.direction == Direction.LONG:
                            p = (t.tp1-t.entry)*cl*100
                        else:
                            p = (t.entry-t.tp1)*cl*100
                        p -= cfg.COMMISSION_PER_LOT*cl; balance += p; t.pnl += p
                        t.remaining = round(t.remaining-cl, 2)
                        t.phase = TradePhase.TP1_HIT
                        if cfg.MOVE_SL_TO_BE: t.sl = t.entry

            # Trailing stop logic
            if cfg.USE_TRAILING and t.phase == TradePhase.TP1_HIT:
                if t.direction == Direction.LONG:
                    profit_r = (price - t.entry) / t.sl_dist if t.sl_dist > 0 else 0
                else:
                    profit_r = (t.entry - price) / t.sl_dist if t.sl_dist > 0 else 0

                if profit_r >= cfg.TRAIL_ACTIVATION_RR:
                    t.trail_activated = True
                    t.phase = TradePhase.TRAILING
                    trail_dist = atr * cfg.TRAIL_ATR_MULT

                    if t.direction == Direction.LONG:
                        new_sl = price - trail_dist
                        if new_sl > t.sl + cfg.TRAIL_STEP:
                            t.sl = new_sl
                    else:
                        new_sl = price + trail_dist
                        if new_sl < t.sl - cfg.TRAIL_STEP:
                            t.sl = new_sl

                elif t.phase == TradePhase.TRAILING:
                    trail_dist = atr * cfg.TRAIL_ATR_MULT
                    if t.direction == Direction.LONG:
                        new_sl = price - trail_dist
                        if new_sl > t.sl + cfg.TRAIL_STEP:
                            t.sl = new_sl
                    else:
                        new_sl = price + trail_dist
                        if new_sl < t.sl - cfg.TRAIL_STEP:
                            t.sl = new_sl

            # Continue trailing if already activated
            if cfg.USE_TRAILING and t.phase == TradePhase.TRAILING:
                trail_dist = atr * cfg.TRAIL_ATR_MULT
                if t.direction == Direction.LONG:
                    new_sl = price - trail_dist
                    if new_sl > t.sl + cfg.TRAIL_STEP:
                        t.sl = new_sl
                else:
                    new_sl = price + trail_dist
                    if new_sl < t.sl - cfg.TRAIL_STEP:
                        t.sl = new_sl

        # Equity
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
                      lots=lots, bar=i, sl_dist=sl_dist, entry_time=str(ts))
        active.append(trade)
        daily_trades += 1; ltb = i

    # Close remaining
    if active:
        lp = df["Close"].iloc[-1]
        for t in active:
            t.pnl += ((lp-t.entry) if t.direction==Direction.LONG else (t.entry-lp))*t.remaining*100
            t.exit_reason = "END"; closed.append(t)

    if not closed:
        print("  No trades!"); return {}

    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    tw = sum(t.pnl for t in wins); tl = abs(sum(t.pnl for t in losses))
    wr = len(wins)/len(closed)*100
    pf = tw/tl if tl > 0 else 99
    net = balance - cfg.START_BALANCE

    # Monthly
    for t in closed:
        m = t.entry_time[:7] if t.entry_time else "?"
        if m not in monthly_pnl: monthly_pnl[m] = 0
        monthly_pnl[m] += t.pnl

    # Exit stats
    exit_stats = {}
    for t in closed:
        r = t.exit_reason
        if r not in exit_stats: exit_stats[r] = {"count":0,"pnl":0}
        exit_stats[r]["count"] += 1; exit_stats[r]["pnl"] += t.pnl

    # Runner stats
    trail_trades = [t for t in closed if t.trail_activated]
    trail_wins = [t for t in trail_trades if t.pnl > 0]
    max_runner_pnl = max([t.pnl for t in trail_trades], default=0)
    avg_runner_pnl = sum(t.pnl for t in trail_trades)/len(trail_trades) if trail_trades else 0
    max_r_reached = max([t.max_profit_r for t in closed], default=0)

    # Big winners
    big_50 = [t for t in closed if t.pnl >= 50]
    big_100 = [t for t in closed if t.pnl >= 100]

    # Streaks
    max_ws, max_ls, cw, cl = 0, 0, 0, 0
    for t in closed:
        if t.pnl > 0: cw += 1; cl = 0; max_ws = max(max_ws, cw)
        else: cl += 1; cw = 0; max_ls = max(max_ls, cl)

    print(f"\n{'='*60}")
    print(f"  📊 {label} RESULTS")
    print(f"{'='*60}")
    print(f"""
  💰 Start:     ${cfg.START_BALANCE:>10,.2f}
  💰 Final:     ${balance:>10,.2f}
  📈 Net P&L:   ${net:>+10,.2f}
  📈 Return:    {net/cfg.START_BALANCE*100:>+9.2f}%
  📉 Max DD:    {max_dd:>9.2f}%

  ─── TRADES ──────────────────────────────────
  Total:        {len(closed):>8}
  Wins:         {len(wins):>8}
  Losses:       {len(losses):>8}
  Win Rate:     {wr:>7.1f}%
  PF:           {pf:>8.2f}

  ─── AVERAGES ────────────────────────────────
  Avg Win:      ${tw/max(len(wins),1):>+9.2f}
  Avg Loss:     ${tl/max(len(losses),1):>9.2f}
  Max Win:      ${max(t.pnl for t in closed):>+9.2f}
  Max Loss:     ${min(t.pnl for t in closed):>9.2f}

  ─── STREAKS ─────────────────────────────────
  Max Win:      {max_ws:>8}
  Max Loss:     {max_ls:>8}

  ─── BIG WINNERS 💰 ─────────────────────────
  $50+ trades:  {len(big_50):>8}
  $100+ trades: {len(big_100):>8}
  Max R reached:{max_r_reached:>7.1f}R""")

    if cfg.USE_TRAILING:
        print(f"""
  ─── RUNNER / TRAILING STATS 🏃 ──────────────
  Runners activated:  {len(trail_trades):>5}
  Runner wins:        {len(trail_wins):>5}
  Runner WR:          {len(trail_wins)/max(len(trail_trades),1)*100:>5.0f}%
  Avg runner PnL:     ${avg_runner_pnl:>+8.2f}
  Best runner:        ${max_runner_pnl:>+8.2f}""")

    print(f"\n  ─── EXIT REASONS ────────────────────────────")
    for r, d in sorted(exit_stats.items()):
        print(f"  {r:>10}: {d['count']:>5}t | PnL: ${d['pnl']:>+10,.2f}")

    print(f"\n  ─── MONTHLY P&L ─────────────────────────────")
    for m in sorted(monthly_pnl.keys()):
        e = "🟢" if monthly_pnl[m] >= 0 else "🔴"
        print(f"  {e} {m}: ${monthly_pnl[m]:>+10,.2f}")

    return {
        "trades": len(closed), "wr": round(wr,1), "pf": round(pf,2),
        "pnl": round(net,2), "dd": round(max_dd,2),
        "return": round(net/cfg.START_BALANCE*100,2),
        "big_50": len(big_50), "big_100": len(big_100),
        "runners": len(trail_trades), "best_runner": round(max_runner_pnl,2),
        "avg_win": round(tw/max(len(wins),1),2),
    }


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — v1.5 TRAILING STOP BACKTEST       ║
║     Test: ZONDER trail vs MET trail                          ║
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

    # Test 1: v1.5 WITHOUT trailing
    print(f"\n{'='*60}")
    print(f"  🔵 TEST 1: v1.5 ZONDER TRAILING STOP")
    print(f"{'='*60}")
    cfg1 = Config()
    cfg1.USE_TRAILING = False
    r1 = run_backtest(df, cfg1, sh, sl, "v1.5 ZONDER TRAIL")

    # Test 2: v1.5 WITH trailing
    print(f"\n{'='*60}")
    print(f"  🟢 TEST 2: v1.5 MET TRAILING STOP")
    print(f"{'='*60}")
    cfg2 = Config()
    cfg2.USE_TRAILING = True
    r2 = run_backtest(df, cfg2, sh, sl, "v1.5 MET TRAIL")

    # Comparison
    if r1 and r2:
        print(f"""
{'='*60}
  ⚖️ VERGELIJKING: TRAIL vs GEEN TRAIL
{'='*60}
  {'':>20} {'Zonder':>12} {'Met Trail':>12} {'Verschil':>12}
  {'WR':>20} {f"{r1['wr']}%":>12} {f"{r2['wr']}%":>12} {f"{r2['wr']-r1['wr']:+.1f}%":>12}
  {'PF':>20} {f"{r1['pf']}":>12} {f"{r2['pf']}":>12} {f"{r2['pf']-r1['pf']:+.2f}":>12}
  {'Return':>20} {f"+{r1['return']:.0f}%":>12} {f"+{r2['return']:.0f}%":>12} {f"{r2['return']-r1['return']:+.1f}%":>12}
  {'Net PnL':>20} {f"${r1['pnl']:+,.0f}":>12} {f"${r2['pnl']:+,.0f}":>12} {f"${r2['pnl']-r1['pnl']:+,.0f}":>12}
  {'Trades':>20} {f"{r1['trades']}":>12} {f"{r2['trades']}":>12} {f"{r2['trades']-r1['trades']:+d}":>12}
  {'Max DD':>20} {f"{r1['dd']}%":>12} {f"{r2['dd']}%":>12} {f"{r2['dd']-r1['dd']:+.1f}%":>12}
  {'Avg Win':>20} {f"${r1['avg_win']}":>12} {f"${r2['avg_win']}":>12} {f"${r2['avg_win']-r1['avg_win']:+.2f}":>12}
  {'$50+ trades':>20} {f"{r1['big_50']}":>12} {f"{r2['big_50']}":>12} {f"{r2['big_50']-r1['big_50']:+d}":>12}
  {'$100+ trades':>20} {f"{r1['big_100']}":>12} {f"{r2['big_100']}":>12} {f"{r2['big_100']-r1['big_100']:+d}":>12}
  {'Runners':>20} {'N/A':>12} {f"{r2['runners']}":>12} {'':>12}
  {'Best Runner':>20} {'N/A':>12} {f"${r2['best_runner']:+,.0f}":>12} {'':>12}

  {'✅ TRAILING WINT!' if r2['pnl'] > r1['pnl'] else '⚠️ TRAILING HELPT NIET — houd v1.5 zonder trail'}
{'='*60}""")


if __name__ == "__main__":
    main()
