"""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SWING BOT — BACKTESTER + OPTIMIZER          ║
║  Phase 1: Baseline test of current v1.0 strategy             ║
║  Phase 2: Signal audit (what helps, what hurts)              ║
║  Phase 3: Parameter optimization for higher WR               ║
║  Phase 4: Daily/Weekly P&L analysis                          ║
║  Phase 5: Compare with scalper results                       ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys, json, math, time
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
    TRAILING = "trailing"
    CLOSED = "closed"


@dataclass
class Config:
    LABEL: str = "Swing v1.0"
    START_BALANCE: float = 5000.0
    COMMISSION_PER_LOT: float = 7.0
    SIMULATED_SPREAD: float = 0.30
    RISK_PERCENT: float = 1.0
    MAX_DAILY_LOSS_PERCENT: float = 3.0
    MAX_TOTAL_DRAWDOWN_PERCENT: float = 10.0
    MAX_CONCURRENT_TRADES: int = 2
    MAX_DAILY_TRADES: int = 3
    MAX_CONSECUTIVE_LOSSES: int = 3

    # SL/TP
    ATR_SL_MULTIPLIER: float = 3.0
    MIN_SL_POINTS: float = 5.0
    MAX_SL_POINTS: float = 30.0
    RR_RATIO: float = 3.0
    PARTIAL_PERCENT: float = 0.50
    TP1_RR: float = 1.5
    MOVE_SL_TO_BE: bool = True

    # Trailing
    USE_TRAILING: bool = True
    TRAIL_ACTIVATION_RR: float = 2.0
    TRAIL_ATR_MULT: float = 1.5
    TRAIL_STEP: float = 1.0

    # Structure
    SWING_LOOKBACK: int = 5
    BOS_MIN_BREAK_ATR: float = 0.5
    OB_BODY_RATIO: float = 0.5
    FVG_MIN_ATR: float = 0.3

    # Indicators
    EMA_FAST: int = 50
    EMA_SLOW: int = 200
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: float = 30.0
    RSI_OVERBOUGHT: float = 70.0
    ADX_THRESHOLD: float = 20.0
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    BB_SQUEEZE_RATIO: float = 0.5
    MR_SCORE: int = 2

    MIN_CONFLUENCE: int = 4
    TRADE_COOLDOWN_BARS: int = 6    # ~6 hours on 1H
    LOSS_COOLDOWN_BARS: int = 12    # ~12 hours

    # Toggleable signals
    SIG_GOLDEN_CROSS: bool = True    # Daily EMA 50/200
    SIG_STRUCTURE: bool = True       # 4H BOS/CHoCH
    SIG_PREMIUM_DISCOUNT: bool = True # 4H zone
    SIG_SWEEP: bool = True           # 1H liquidity sweep
    SIG_OB: bool = True              # 1H order block
    SIG_FVG: bool = True             # 1H FVG
    SIG_MOMENTUM: bool = True        # 1H momentum candle
    SIG_RSI_DIV: bool = True         # RSI divergence
    SIG_BB_SQUEEZE: bool = True      # BB squeeze breakout
    SIG_MR: bool = True              # Mean reversion
    SIG_DOUBLE: bool = True          # Double bottom/top
    SIG_ADX: bool = True             # ADX trend filter


def calculate_indicators(df):
    df = df.copy()
    # EMAs
    df["ema9"] = df["Close"].ewm(span=9).mean()
    df["ema21"] = df["Close"].ewm(span=21).mean()
    df["ema50"] = df["Close"].ewm(span=50).mean()
    df["ema200"] = df["Close"].ewm(span=200).mean()

    # ATR
    df["tr"] = np.maximum(df["High"]-df["Low"],
                np.maximum(abs(df["High"]-df["Close"].shift(1)),
                           abs(df["Low"]-df["Close"].shift(1))))
    df["atr"] = df["tr"].rolling(14).mean()

    # RSI 14
    d = df["Close"].diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    rs = g / l.replace(0, np.nan)
    df["rsi"] = 100 - (100/(1+rs))

    # Bollinger Bands
    df["bb_mid"] = df["Close"].rolling(20).mean()
    df["bb_std"] = df["Close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2*df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2*df["bb_std"]

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

    # Candle metrics
    df["body"] = abs(df["Close"]-df["Open"])
    df["candle_range"] = df["High"]-df["Low"]
    df["upper_wick"] = df["High"] - df[["Close","Open"]].max(axis=1)
    df["lower_wick"] = df[["Close","Open"]].min(axis=1) - df["Low"]

    # Swing range (for premium/discount)
    df["swing_high"] = df["High"].rolling(50, min_periods=10).max()
    df["swing_low"] = df["Low"].rolling(50, min_periods=10).min()
    df["equilibrium"] = (df["swing_high"] + df["swing_low"]) / 2

    return df


def detect_swings(df, lookback=5):
    sh, sl = [], []
    for i in range(lookback, len(df)-lookback):
        h, l = df["High"].iloc[i], df["Low"].iloc[i]
        if all(h >= df["High"].iloc[i+j] and h >= df["High"].iloc[i-j] for j in range(1, lookback+1)):
            sh.append(i)
        if all(l <= df["Low"].iloc[i+j] and l <= df["Low"].iloc[i-j] for j in range(1, lookback+1)):
            sl.append(i)
    return sh, sl


def detect_signal(df, i, cfg, swing_highs, swing_lows):
    if i < 200: return None
    price = df["Close"].iloc[i]
    atr = df["atr"].iloc[i]
    if pd.isna(atr) or atr <= 0: return None

    ema50 = df["ema50"].iloc[i]
    ema200 = df["ema200"].iloc[i]
    rsi = df["rsi"].iloc[i]
    adx = df["adx"].iloc[i]
    if pd.isna(ema50) or pd.isna(ema200): return None

    ts = df.index[i]
    if not hasattr(ts, 'hour') or not (7 <= ts.hour < 17): return None

    # ADX gate: don't trade ranging markets
    if cfg.SIG_ADX and not pd.isna(adx) and adx < cfg.ADX_THRESHOLD:
        return None

    confluence = 0
    reasons = []
    votes = {Direction.LONG: 0, Direction.SHORT: 0}
    cc, co = df["Close"].iloc[i], df["Open"].iloc[i]
    ch, cl = df["High"].iloc[i], df["Low"].iloc[i]
    body, total = df["body"].iloc[i], df["candle_range"].iloc[i]

    # 1. Golden Cross (EMA 50/200) — macro trend
    if cfg.SIG_GOLDEN_CROSS:
        if ema50 > ema200:
            votes[Direction.LONG] += 2; reasons.append("golden_cross")
        elif ema50 < ema200:
            votes[Direction.SHORT] += 2; reasons.append("death_cross")

    # 2. Market Structure (BOS/CHoCH simulation using swing points)
    if cfg.SIG_STRUCTURE:
        recent_sh = [s for s in swing_highs if s < i and s > i-50]
        recent_sl = [s for s in swing_lows if s < i and s > i-50]
        min_break = atr * cfg.BOS_MIN_BREAK_ATR

        if len(recent_sh) >= 2 and len(recent_sl) >= 2:
            # Higher highs + higher lows = bullish
            hh = df["High"].iloc[recent_sh[-1]] > df["High"].iloc[recent_sh[-2]]
            hl = df["Low"].iloc[recent_sl[-1]] > df["Low"].iloc[recent_sl[-2]]
            # Lower highs + lower lows = bearish
            lh = df["High"].iloc[recent_sh[-1]] < df["High"].iloc[recent_sh[-2]]
            ll = df["Low"].iloc[recent_sl[-1]] < df["Low"].iloc[recent_sl[-2]]

            last_sh_price = df["High"].iloc[recent_sh[-1]]
            last_sl_price = df["Low"].iloc[recent_sl[-1]]

            # BOS
            if price > last_sh_price + min_break and (hh or hl):
                votes[Direction.LONG] += 2; reasons.append("bos_bull")
            elif price < last_sl_price - min_break and (lh or ll):
                votes[Direction.SHORT] += 2; reasons.append("bos_bear")
            # CHoCH
            elif hh and hl and price < last_sl_price - min_break:
                votes[Direction.SHORT] += 2; reasons.append("choch_bear")
            elif lh and ll and price > last_sh_price + min_break:
                votes[Direction.LONG] += 2; reasons.append("choch_bull")

    # 3. Premium/Discount zone
    if cfg.SIG_PREMIUM_DISCOUNT:
        eq = df["equilibrium"].iloc[i]
        if not pd.isna(eq) and eq > 0:
            if price < eq:
                votes[Direction.LONG] += 1; reasons.append("discount")
            elif price > eq:
                votes[Direction.SHORT] += 1; reasons.append("premium")

    # 4. Liquidity Sweep
    if cfg.SIG_SWEEP:
        for si in [s for s in swing_lows if s < i and s > i-30][-5:]:
            if cl < df["Low"].iloc[si] and cc > df["Low"].iloc[si]:
                wick = min(cc, co) - cl
                if wick > atr * 0.3:
                    votes[Direction.LONG] += 2; reasons.append("sweep"); break
        for si in [s for s in swing_highs if s < i and s > i-30][-5:]:
            if ch > df["High"].iloc[si] and cc < df["High"].iloc[si]:
                wick = ch - max(cc, co)
                if wick > atr * 0.3:
                    votes[Direction.SHORT] += 2; reasons.append("sweep"); break

    # 5. Order Block
    if cfg.SIG_OB and i >= 2:
        prev, curr = df.iloc[i-1], df.iloc[i]
        pb = abs(prev["Close"]-prev["Open"])
        cb = abs(curr["Close"]-curr["Open"])
        if curr["Close"]>curr["Open"] and prev["Close"]<prev["Open"] and cb>pb*cfg.OB_BODY_RATIO and curr["Close"]>prev["High"]:
            votes[Direction.LONG] += 1; reasons.append("ob")
        elif curr["Close"]<curr["Open"] and prev["Close"]>prev["Open"] and cb>pb*cfg.OB_BODY_RATIO and curr["Close"]<prev["Low"]:
            votes[Direction.SHORT] += 1; reasons.append("ob")

    # 6. FVG
    if cfg.SIG_FVG and i >= 2:
        c1h, c3l = df["High"].iloc[i-2], df["Low"].iloc[i]
        c1l, c3h = df["Low"].iloc[i-2], df["High"].iloc[i]
        mg = atr * cfg.FVG_MIN_ATR
        if c3l > c1h and (c3l-c1h) >= mg:
            votes[Direction.LONG] += 1; reasons.append("fvg")
        elif c1l > c3h and (c1l-c3h) >= mg:
            votes[Direction.SHORT] += 1; reasons.append("fvg")

    # 7. Momentum candle
    if cfg.SIG_MOMENTUM and total > 0 and body/total >= 0.6 and body >= atr*0.8:
        if cc > co: votes[Direction.LONG] += 1; reasons.append("momentum")
        else: votes[Direction.SHORT] += 1; reasons.append("momentum")

    # 8. RSI Divergence
    if cfg.SIG_RSI_DIV and not pd.isna(rsi) and i >= 8:
        p_now = cc
        p_prev = df["Close"].iloc[i-8]
        if rsi < 35 and p_now > p_prev:
            votes[Direction.LONG] += 1; reasons.append("rsi_div")
        elif rsi > 65 and p_now < p_prev:
            votes[Direction.SHORT] += 1; reasons.append("rsi_div")

    # 9. BB Squeeze breakout
    if cfg.SIG_BB_SQUEEZE:
        bb_u, bb_l = df["bb_upper"].iloc[i], df["bb_lower"].iloc[i]
        if not pd.isna(bb_u) and bb_u > bb_l:
            bb_width = bb_u - bb_l
            if bb_width < atr * cfg.BB_SQUEEZE_RATIO:
                # Squeeze active — check breakout direction
                if total > 0 and body/total >= 0.6 and body >= atr*0.8:
                    if cc > co: votes[Direction.LONG] += 1; reasons.append("squeeze_break")
                    else: votes[Direction.SHORT] += 1; reasons.append("squeeze_break")

    # 10. Mean Reversion
    if cfg.SIG_MR:
        bb_u, bb_l = df["bb_upper"].iloc[i], df["bb_lower"].iloc[i]
        if not pd.isna(bb_u) and not pd.isna(rsi) and bb_u > bb_l:
            bbr = bb_u - bb_l
            pct_b = (price-bb_l)/bbr
            if pct_b <= 0.05 and rsi <= cfg.RSI_OVERSOLD:
                votes[Direction.LONG] += cfg.MR_SCORE; reasons.append("MR")
            elif pct_b >= 0.95 and rsi >= cfg.RSI_OVERBOUGHT:
                votes[Direction.SHORT] += cfg.MR_SCORE; reasons.append("MR")

    # 11. Double Bottom/Top
    if cfg.SIG_DOUBLE:
        recent_sl = [s for s in swing_lows if s < i and s > i-40]
        recent_sh = [s for s in swing_highs if s < i and s > i-40]
        tol = atr * 0.4
        if len(recent_sl) >= 2:
            l1, l2 = df["Low"].iloc[recent_sl[-2]], df["Low"].iloc[recent_sl[-1]]
            if abs(l1-l2) < tol and price > max(l1,l2):
                votes[Direction.LONG] += 1; reasons.append("dbl_bot")
        if len(recent_sh) >= 2:
            h1, h2 = df["High"].iloc[recent_sh[-2]], df["High"].iloc[recent_sh[-1]]
            if abs(h1-h2) < tol and price < min(h1,h2):
                votes[Direction.SHORT] += 1; reasons.append("dbl_top")

    # 12. ADX bonus
    if cfg.SIG_ADX and not pd.isna(adx) and adx >= cfg.ADX_THRESHOLD:
        confluence += 1; reasons.append(f"adx{adx:.0f}")

    # Direction
    ls, ss = votes[Direction.LONG], votes[Direction.SHORT]
    if ls > ss and ls >= 2: direction = Direction.LONG; confluence += ls
    elif ss > ls and ss >= 2: direction = Direction.SHORT; confluence += ss
    else: return None

    # Alignment filters
    if cfg.SIG_GOLDEN_CROSS:
        if direction == Direction.LONG and ema50 < ema200 and "MR" not in str(reasons):
            return None
        if direction == Direction.SHORT and ema50 > ema200 and "MR" not in str(reasons):
            return None

    # Premium/Discount alignment
    if cfg.SIG_PREMIUM_DISCOUNT:
        eq = df["equilibrium"].iloc[i]
        if not pd.isna(eq):
            if direction == Direction.LONG and price > eq and "MR" not in str(reasons):
                return None
            if direction == Direction.SHORT and price < eq and "MR" not in str(reasons):
                return None

    if confluence < cfg.MIN_CONFLUENCE: return None
    return direction, confluence, "|".join(reasons)


@dataclass
class Trade:
    direction: Direction; entry: float; sl: float; tp: float; tp1: float
    lots: float; bar: int; sl_dist: float; day: str = ""
    phase: TradePhase = TradePhase.OPEN
    pnl: float = 0.0; remaining: float = 0.0
    trail_active: bool = False
    def __post_init__(self): self.remaining = self.lots


def run_backtest(df, cfg, sh, sl, verbose=False):
    balance = cfg.START_BALANCE; peak = balance; max_dd = 0.0
    active, closed = [], []; daily_trades = 0; daily_date = ""
    consec_losses = 0; ltb = -999; llb = -999
    daily_pnl = {}; weekly_pnl = {}

    for i in range(200, len(df)):
        price, high, low = df["Close"].iloc[i], df["High"].iloc[i], df["Low"].iloc[i]
        atr = df["atr"].iloc[i]
        if pd.isna(atr) or price <= 0: continue
        ts = df.index[i]
        today = str(ts.date()) if hasattr(ts,'date') else str(ts)[:10]
        week = today[:4] + "-W" + str(ts.isocalendar()[1]) if hasattr(ts,'isocalendar') else today[:7]
        if today != daily_date: daily_date = today; daily_trades = 0
        if today not in daily_pnl: daily_pnl[today] = 0.0
        if week not in weekly_pnl: weekly_pnl[week] = 0.0

        for t in list(active):
            # SL
            if (t.direction==Direction.LONG and low<=t.sl) or (t.direction==Direction.SHORT and high>=t.sl):
                if t.direction==Direction.LONG: t.pnl += (t.sl-t.entry)*t.remaining*100
                else: t.pnl += (t.entry-t.sl)*t.remaining*100
                t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl
                daily_pnl[today] += t.pnl; weekly_pnl[week] += t.pnl
                if not t.trail_active: consec_losses += 1; llb = i
                active.remove(t); closed.append(t); continue

            # TP
            if not (cfg.USE_TRAILING and t.trail_active):
                if (t.direction==Direction.LONG and high>=t.tp) or (t.direction==Direction.SHORT and low<=t.tp):
                    if t.direction==Direction.LONG: t.pnl += (t.tp-t.entry)*t.remaining*100
                    else: t.pnl += (t.entry-t.tp)*t.remaining*100
                    t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl
                    daily_pnl[today] += t.pnl; weekly_pnl[week] += t.pnl
                    consec_losses = 0; active.remove(t); closed.append(t); continue

            # TP1 partial
            if t.phase == TradePhase.OPEN:
                if (t.direction==Direction.LONG and high>=t.tp1) or (t.direction==Direction.SHORT and low<=t.tp1):
                    cl_lots = round(t.lots*cfg.PARTIAL_PERCENT, 2)
                    if cl_lots >= 0.01:
                        if t.direction==Direction.LONG: p = (t.tp1-t.entry)*cl_lots*100
                        else: p = (t.entry-t.tp1)*cl_lots*100
                        p -= cfg.COMMISSION_PER_LOT*cl_lots; balance += p; t.pnl += p
                        daily_pnl[today] += p; weekly_pnl[week] += p
                        t.remaining = round(t.remaining-cl_lots, 2)
                        t.phase = TradePhase.TP1_HIT
                        if cfg.MOVE_SL_TO_BE: t.sl = t.entry

            # Trailing
            if cfg.USE_TRAILING and t.phase in (TradePhase.TP1_HIT, TradePhase.TRAILING):
                if t.sl_dist > 0:
                    if t.direction==Direction.LONG: pr = (price-t.entry)/t.sl_dist
                    else: pr = (t.entry-price)/t.sl_dist
                    if pr >= cfg.TRAIL_ACTIVATION_RR:
                        t.trail_active = True; t.phase = TradePhase.TRAILING
                        trail_d = atr * cfg.TRAIL_ATR_MULT
                        if t.direction==Direction.LONG:
                            new_sl = price - trail_d
                            if new_sl > t.sl + cfg.TRAIL_STEP: t.sl = new_sl
                        else:
                            new_sl = price + trail_d
                            if new_sl < t.sl - cfg.TRAIL_STEP: t.sl = new_sl

        # Equity tracking
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

        lots = max(0.01, min(round((balance*cfg.RISK_PERCENT/100)/(sl_dist*100), 2), 1.0))
        trade = Trade(direction=direction, entry=entry, sl=s, tp=t, tp1=t1,
                      lots=lots, bar=i, sl_dist=sl_dist, day=today)
        active.append(trade)
        daily_trades += 1; ltb = i

    # Close remaining
    if active:
        lp = df["Close"].iloc[-1]
        for t in active:
            t.pnl += ((lp-t.entry) if t.direction==Direction.LONG else (t.entry-lp))*t.remaining*100
            closed.append(t)

    if not closed: return None

    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    tw = sum(t.pnl for t in wins); tl = abs(sum(t.pnl for t in losses))
    wr = len(wins)/len(closed)*100 if closed else 0
    pf = tw/tl if tl > 0 else 99
    net = balance - cfg.START_BALANCE

    trading_days = [d for d,p in daily_pnl.items() if abs(p) > 0.01]
    daily_vals = [daily_pnl[d] for d in trading_days]
    green_days = [p for p in daily_vals if p > 0]
    trading_weeks = [w for w,p in weekly_pnl.items() if abs(p) > 0.01]
    weekly_vals = [weekly_pnl[w] for w in trading_weeks]
    green_weeks = [p for p in weekly_vals if p > 0]

    return {
        "label": cfg.LABEL, "trades": len(closed), "wins": len(wins), "losses": len(losses),
        "wr": round(wr,1), "pf": round(pf,2), "pnl": round(net,2),
        "return_pct": round(net/cfg.START_BALANCE*100,2), "dd": round(max_dd,2),
        "balance": round(balance,2),
        "avg_win": round(tw/max(len(wins),1),2), "avg_loss": round(tl/max(len(losses),1),2),
        "trading_days": len(trading_days), "green_days": len(green_days),
        "green_day_pct": round(len(green_days)/max(len(trading_days),1)*100,1),
        "avg_daily": round(sum(daily_vals)/max(len(trading_days),1),2),
        "best_day": round(max(daily_vals, default=0),2),
        "worst_day": round(min(daily_vals, default=0),2),
        "trading_weeks": len(trading_weeks), "green_weeks": len(green_weeks),
        "green_week_pct": round(len(green_weeks)/max(len(trading_weeks),1)*100,1),
        "avg_weekly": round(sum(weekly_vals)/max(len(trading_weeks),1),2),
        "trades_per_week": round(len(closed)/max(len(trading_weeks),1),1),
        "daily_pnl": {d: round(p,2) for d,p in sorted(daily_pnl.items()) if abs(p)>0.01},
        "weekly_pnl": {w: round(p,2) for w,p in sorted(weekly_pnl.items()) if abs(p)>0.01},
    }


# ═══════════════════════════════════════════════════════════════════
SIGNAL_NAMES = {
    "SIG_GOLDEN_CROSS": "Golden Cross (EMA 50/200)",
    "SIG_STRUCTURE": "BOS/CHoCH Structure",
    "SIG_PREMIUM_DISCOUNT": "Premium/Discount Zone",
    "SIG_SWEEP": "Liquidity Sweep",
    "SIG_OB": "Order Block",
    "SIG_FVG": "Fair Value Gap",
    "SIG_MOMENTUM": "Momentum Candle",
    "SIG_RSI_DIV": "RSI Divergence",
    "SIG_BB_SQUEEZE": "BB Squeeze Breakout",
    "SIG_MR": "Mean Reversion (BB+RSI)",
    "SIG_DOUBLE": "Double Bottom/Top",
    "SIG_ADX": "ADX Trend Filter",
}


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SWING BOT — BACKTESTER + OPTIMIZER          ║
║     🎯 Test current strategy + optimize for higher WR       ║
╚══════════════════════════════════════════════════════════════╝
    """)
    print("📥 Downloading 60-day gold data (1H)...")
    df = yf.download("GC=F", period="60d", interval="1h", progress=True)
    if df.empty:
        print("❌ No data!"); return
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    df = df.dropna(subset=["Open","High","Low","Close"])
    df = df[df["Close"]>0]
    print(f"✅ {len(df)} bars: {df.index[0]} → {df.index[-1]}")

    df = calculate_indicators(df)
    sh, sl = detect_swings(df, 5)
    start = time.time()

    # ═══════════════════════════════════════════════════════════
    #  PHASE 1: BASELINE
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  📊 PHASE 1: BASELINE (v1.0 settings)")
    print(f"{'='*60}")
    baseline = run_backtest(df, Config(), sh, sl)
    if not baseline:
        print("  ❌ No trades!"); return
    print(f"  ✅ {baseline['trades']}t | WR:{baseline['wr']}% | PF:{baseline['pf']} | ${baseline['pnl']:+.0f} | DD:{baseline['dd']:.1f}%")
    print(f"     {baseline['trades_per_week']:.1f} trades/week | {baseline['green_day_pct']:.0f}% green days | {baseline['green_week_pct']:.0f}% green weeks")

    # ═══════════════════════════════════════════════════════════
    #  PHASE 2: SIGNAL AUDIT
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  📊 PHASE 2: SIGNAL AUDIT")
    print(f"{'='*60}")
    print(f"  {'Signal':<30} {'Trades':>7} {'WR%':>6} {'PF':>6} {'PnL':>10} {'Impact':>10}")
    print(f"  {'-'*70}")

    signal_impact = {}
    for sig_key, sig_name in SIGNAL_NAMES.items():
        cfg = Config(); cfg.LABEL = f"No {sig_name[:20]}"
        setattr(cfg, sig_key, False)
        r = run_backtest(df, cfg, sh, sl)
        if not r: r = {"trades":0,"wr":0,"pf":0,"pnl":0}
        pnl_diff = r["pnl"] - baseline["pnl"]
        signal_impact[sig_key] = {"name":sig_name, "result":r, "pnl_diff":pnl_diff}
        emoji = "🟢" if pnl_diff < -50 else "🔴" if pnl_diff > 50 else "⚪"
        print(f"  {emoji} {sig_name:<28} {r['trades']:>7} {r['wr']:>5.1f}% {r['pf']:>5.2f} ${r['pnl']:>+9.0f} ${pnl_diff:>+9.0f}")

    helpers = [(k,v) for k,v in signal_impact.items() if v["pnl_diff"] < -50]
    hurters = [(k,v) for k,v in signal_impact.items() if v["pnl_diff"] > 50]
    helpers.sort(key=lambda x: x[1]["pnl_diff"])
    hurters.sort(key=lambda x: x[1]["pnl_diff"], reverse=True)

    print(f"\n  ✅ HOUDEN: {', '.join([v['name'][:15] for k,v in helpers])}")
    print(f"  ❌ VERWIJDEREN: {', '.join([v['name'][:15] for k,v in hurters]) if hurters else 'Geen'}")

    # ═══════════════════════════════════════════════════════════
    #  PHASE 3: PARAMETER OPTIMIZATION
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  📊 PHASE 3: PARAMETER OPTIMIZATION")
    print(f"{'='*60}")

    all_results = []
    count = 0

    for sl_m in [2.0, 2.5, 3.0, 3.5, 4.0]:
        for rr in [2.0, 2.5, 3.0, 3.5, 4.0]:
            for tp1 in [1.0, 1.5, 2.0]:
                for conf in [3, 4, 5]:
                    for partial in [0.33, 0.50, 0.67]:
                        count += 1
                        cfg = Config()
                        for sig_key, _ in hurters:
                            setattr(cfg, sig_key, False)
                        cfg.ATR_SL_MULTIPLIER = sl_m
                        cfg.RR_RATIO = rr
                        cfg.TP1_RR = tp1
                        cfg.MIN_CONFLUENCE = conf
                        cfg.PARTIAL_PERCENT = partial
                        cfg.LABEL = f"SL{sl_m}_RR{rr}_TP1{tp1}_C{conf}_P{partial}"
                        r = run_backtest(df, cfg, sh, sl)
                        if r:
                            r["params"] = {"sl":sl_m,"rr":rr,"tp1":tp1,"conf":conf,"partial":partial}
                            all_results.append(r)
                        if count % 100 == 0: print(f"    {count} tested...")

    print(f"  ✅ {count} combinations tested")

    viable = [r for r in all_results if r["trades"] >= 5 and r["pf"] >= 1.0]
    viable.sort(key=lambda x: (x["wr"], x["pf"]), reverse=True)

    print(f"\n  🎯 TOP 10 HIGHEST WIN RATE:")
    for i, r in enumerate(viable[:10]):
        p = r["params"]
        print(f"""
  #{i+1} — WR: {r['wr']}% {'⭐' if r['wr']>=60 else ''}
  ├── SL: ATR×{p['sl']} | RR: 1:{p['rr']} | TP1: {p['tp1']}R | Partial: {p['partial']*100:.0f}%
  ├── Conf≥{p['conf']} | Trades: {r['trades']} | PF: {r['pf']}
  ├── PnL: ${r['pnl']:+,.2f} | DD: {r['dd']:.1f}% | {r['trades_per_week']:.1f}t/week
  └── Green days: {r['green_day_pct']:.0f}% | Green weeks: {r['green_week_pct']:.0f}%""")

    # ═══════════════════════════════════════════════════════════
    #  PHASE 4: DAILY/WEEKLY BREAKDOWN OF BEST
    # ═══════════════════════════════════════════════════════════
    if viable:
        best = viable[0]
        print(f"\n{'='*60}")
        print(f"  📅 PHASE 4: WEEKLY P&L — BEST CONFIG")
        print(f"{'='*60}")
        for w, p in sorted(best.get("weekly_pnl",{}).items()):
            e = "🟢" if p > 0 else "🔴"
            bars = "█" * min(int(abs(p)/20), 30)
            print(f"    {e} {w}: ${p:>+8.2f} {bars}")

        print(f"\n  📅 DAILY P&L:")
        for d, p in sorted(best.get("daily_pnl",{}).items()):
            e = "🟢" if p > 0 else "🔴"
            print(f"    {e} {d}: ${p:>+8.2f}")

    # ═══════════════════════════════════════════════════════════
    #  PHASE 5: COMPARISON WITH SCALPER
    # ═══════════════════════════════════════════════════════════
    best_swing = viable[0] if viable else baseline
    print(f"""
{'='*60}
  ⚖️ PHASE 5: SWING vs SCALPER VERGELIJKING
{'='*60}
  {'':>20} {'Scalper v1.5':>15} {'Swing Best':>15}
  {'WR':>20} {'70.7%':>15} {f"{best_swing['wr']}%":>15}
  {'PF':>20} {'1.30':>15} {f"{best_swing['pf']}":>15}
  {'Return':>20} {'+56%':>15} {f"+{best_swing['return_pct']:.0f}%":>15}
  {'Trades':>20} {'334':>15} {f"{best_swing['trades']}":>15}
  {'Trades/week':>20} {'~56':>15} {f"{best_swing['trades_per_week']:.1f}":>15}
  {'DD':>20} {'2.2%':>15} {f"{best_swing['dd']:.1f}%":>15}
  {'Avg/day':>20} {'$57':>15} {f"${best_swing['avg_daily']:.0f}":>15}
  {'Green days':>20} {'84%':>15} {f"{best_swing['green_day_pct']:.0f}%":>15}
  {'Green weeks':>20} {'100%':>15} {f"{best_swing['green_week_pct']:.0f}%":>15}
  {'Commissie impact':>20} {'Hoog (42%)':>15} {'Laag (~15%)':>15}
{'='*60}""")

    # Recommendation
    if best_swing:
        bp = best_swing.get("params", {})
        print(f"""
  ⭐ AANBEVOLEN SWING SETTINGS:
  ATR_SL_MULTIPLIER = {bp.get('sl', 3.0)}
  RR_RATIO = {bp.get('rr', 3.0)}
  TP1_RR = {bp.get('tp1', 1.5)}
  PARTIAL_PERCENT = {bp.get('partial', 0.50)}
  MIN_CONFLUENCE = {bp.get('conf', 4)}
  USE_TRAILING = True

  Signalen:""")
        for sig_key, sig_name in SIGNAL_NAMES.items():
            is_on = sig_key not in [h[0] for h in hurters]
            print(f"    {'✅' if is_on else '❌'} {sig_name}")

    # Save
    output = {
        "baseline": {k:v for k,v in baseline.items() if k not in ("daily_pnl","weekly_pnl")},
        "signal_audit": {k:{"name":v["name"],"impact":v["pnl_diff"],"verdict":"KEEP" if v["pnl_diff"]<-50 else "REMOVE" if v["pnl_diff"]>50 else "NEUTRAL"} for k,v in signal_impact.items()},
        "top_10": [{k:v for k,v in r.items() if k not in ("daily_pnl","weekly_pnl")} for r in viable[:10]],
        "total_tested": count,
    }
    with open("swing_backtest_results.json","w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  📁 Saved: swing_backtest_results.json")
    print(f"  ⏱️  Klaar in {time.time()-start:.0f}s")


if __name__ == "__main__":
    main()
