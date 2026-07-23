"""
Stock Dashboard v3 — price data (with intraday support), technical
indicators, cleaned news summaries with links, and a narrative connecting
the technical picture to recent news.

IMPORTANT: Nothing here predicts future prices or tells you to buy/sell.
The "Indicator Lean" and "narrative" describe the CURRENT picture and the
tension between different signals — they are not forecasts. This is not
financial advice.
"""

import re
import json
import time
import html as html_lib
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ----------------------------------------------------------------------
# 1. PAGE SETUP
# ----------------------------------------------------------------------
st.set_page_config(page_title="BullBear", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------
# 1b. VISUAL STYLE — elegant private-bank look: deep charcoal-navy, muted
#     gold accent, serif headers + monospace numbers (professional data
#     feel without the harsh black/neon-orange contrast). Includes a
#     mobile breakpoint so headers/metrics don't overflow small screens.
# ----------------------------------------------------------------------
# The color/theme basics also live in .streamlit/config.toml (the
# officially supported way to theme Streamlit). This block layers on
# fonts and panel styling that config.toml can't control on its own.
# Note: Streamlit's internal HTML structure can change between versions —
# if a future Streamlit update changes these class names, this styling
# may need small selector updates, but the app will still work either way.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,500;0,600;1,500&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    font-family: 'Fraunces', serif !important;
    color: #E7DFC6 !important;
    letter-spacing: 0.3px;
    font-weight: 600;
}
h1 {
    border-bottom: 1px solid #C9A96166;
    padding-bottom: 0.4rem;
    font-size: 1.7rem !important;
}
h2, h3 {
    border-bottom: 1px solid #2A2E38;
    padding-bottom: 0.25rem;
    font-size: 1.1rem !important;
}
/* Metric numbers: soft gold, monospace, tabular — professional data feel */
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #C9A961 !important;
    font-weight: 600;
}
[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    color: #9198A6 !important;
    text-transform: uppercase;
    font-size: 0.72rem !important;
    letter-spacing: 0.4px;
}
/* Panels/cards: soft rounded corners, gentle border — elegant not stark */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #262B36 !important;
    border-radius: 8px !important;
    background-color: #191D26 !important;
}
/* Sidebar: slightly darker, soft gold divider */
[data-testid="stSidebar"] {
    background-color: #12151C !important;
    border-right: 1px solid #C9A96133;
}
/* Buttons: soft gold outline, rounded */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    border: 1px solid #C9A961 !important;
    border-radius: 6px !important;
    color: #C9A961 !important;
    background-color: transparent !important;
}
.stButton > button:hover {
    background-color: #C9A961 !important;
    color: #14171F !important;
}
[data-testid="stCaptionContainer"] {
    color: #8A90A0 !important;
}

/* --- Mobile: shrink headers/metrics so they don't overflow a phone screen --- */
@media (max-width: 640px) {
    h1 { font-size: 1.25rem !important; letter-spacing: 0.1px; }
    h2, h3 { font-size: 0.95rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.65rem !important; }
}
</style>
""", unsafe_allow_html=True)

st.title("BullBear")
st.caption(
    "Educational tool only. Shows public price data, indicators, news, and "
    "a statistical price range based on past volatility. Nothing here is "
    "a prediction or a buy/sell recommendation."
)


with st.expander("📖 New to these terms? Quick glossary"):
    st.markdown("""
- **SMA (Simple Moving Average)** — the average closing price over the last N days. Smooths out daily noise so the underlying trend is easier to see. A "20-day average" is just yesterday's price blended with the past month.
- **RSI (Relative Strength Index)** — a 0–100 score for how fast and how far a price has moved recently. Traditionally, above 70 is called "overbought" (may be due to cool off), below 30 "oversold" (may be due to bounce). These are rules of thumb, not guarantees.
- **MACD** — short for Moving Average Convergence Divergence. It compares a faster and slower moving average to gauge whether upward or downward momentum is building or fading.
- **Bollinger Bands** — a band drawn above and below the moving average, sized by how much the price has recently fluctuated. Price hugging the outer band suggests an unusually strong move relative to its own recent history.
- **Standard deviation ("1st"/"2nd deviation")** — a statistics term for how spread out past price moves have been. If daily moves were random, 1 standard deviation ("68% range") covers roughly 68% of past outcomes, and 2 standard deviations ("95% range") covers about 95%. Wider historical swings (higher volatility) mean a wider range here — it describes the past, not a forecast of the future.
- **Indicator Lean** — a simple count of how many well-known indicators (above) currently point up vs. down. It's shown for transparency, not because it's a validated trading strategy.
""")


# ----------------------------------------------------------------------
# 2. SIDEBAR — USER INPUTS
# ----------------------------------------------------------------------
st.sidebar.header("BullBear Settings")

# ----------------------------------------------------------------------
# WATCHLIST PERSISTENCE — saves your watchlist to a small local file so
# it's still there next time you open the app. Lives next to the app
# itself; nothing is sent anywhere.
# ----------------------------------------------------------------------
WATCHLIST_FILE = "bullbear_watchlist.json"
DEFAULT_WATCHLIST = "AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, XOM, JNJ"


def load_saved_watchlist() -> str:
    try:
        with open(WATCHLIST_FILE, "r") as f:
            return json.load(f).get("watchlist", DEFAULT_WATCHLIST)
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_WATCHLIST


def save_watchlist(watchlist_text: str) -> None:
    try:
        with open(WATCHLIST_FILE, "w") as f:
            json.dump({"watchlist": watchlist_text}, f)
    except OSError:
        pass  # non-critical — worst case, it just doesn't persist this time


TELEGRAM_FILE = "bullbear_telegram.json"


def load_saved_telegram() -> dict:
    try:
        with open(TELEGRAM_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"bot_token": "", "chat_id": ""}


def save_telegram(bot_token: str, chat_id: str) -> None:
    try:
        with open(TELEGRAM_FILE, "w") as f:
            json.dump({"bot_token": bot_token, "chat_id": chat_id}, f)
    except OSError:
        pass  # non-critical — worst case, it just doesn't persist this time


with st.sidebar.expander("🔍 Ticker & Chart", expanded=True):
    # International markets: yfinance reaches these via ticker suffixes.
    # We auto-append the right one so you can just type the base symbol.
    MARKET_SUFFIX_MAP = {
        "US (default)": "",
        "Hong Kong (HKEX)": ".HK",
        "South Korea — KOSPI": ".KS",
        "South Korea — KOSDAQ": ".KQ",
        "Singapore (SGX)": ".SI",
    }
    market_choice = st.selectbox(
        "Market",
        options=list(MARKET_SUFFIX_MAP.keys()),
        help="Pick a market and just type the base ticker/code — the right suffix is added automatically. E.g. Hong Kong: '0700' becomes '0700.HK'.",
    )
    raw_ticker_input = st.text_input("Stock ticker or code (e.g. AAPL, 0700, 005930, D05)", "AAPL").upper().strip()
    _suffix = MARKET_SUFFIX_MAP[market_choice]
    if _suffix and "." not in raw_ticker_input:
        ticker = raw_ticker_input + _suffix
    else:
        ticker = raw_ticker_input

    CHART_OPTIONS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"]
    # Maps each display option to a (yfinance period, yfinance interval) pair.
    # 1d/5d use intraday bars WITH pre/post-market data included (prepost=True)
    # so the chart doesn't just flatline at the last regular-session close —
    # it reflects overnight/extended-hours trading where available.
    CHART_PERIOD_MAP = {
        "1d": ("1d", "5m"),
        "5d": ("5d", "15m"),
        "1mo": ("1mo", "1d"),
        "3mo": ("3mo", "1d"),
        "6mo": ("6mo", "1d"),
        "1y": ("1y", "1d"),
        "2y": ("2y", "1d"),
        "5y": ("5y", "1d"),
    }

    period_choice = st.selectbox("Chart timeframe", options=CHART_OPTIONS, index=4)  # default "6mo"
    horizon_days = st.slider("Price range horizon (trading days ahead)", 1, 30, 5)
    run_button = st.button("Run analysis", type="primary")

with st.sidebar.expander("📋 Watchlist Scan"):
    if "watchlist_text" not in st.session_state:
        st.session_state.watchlist_text = load_saved_watchlist()

    watchlist_input = st.text_area(
        "Tickers to scan (comma-separated)",
        key="watchlist_text",
        help="Edit this to any tickers you want compared. Saved automatically so it's here next time you open the app.",
    )
    scan_button = st.button("Scan watchlist")
    if scan_button:
        save_watchlist(st.session_state.watchlist_text)

with st.sidebar.expander("📨 Telegram Alerts (optional)"):
    if "telegram_creds_loaded" not in st.session_state:
        _saved_tg = load_saved_telegram()
        st.session_state.telegram_bot_token = _saved_tg["bot_token"]
        st.session_state.telegram_chat_id = _saved_tg["chat_id"]
        st.session_state.telegram_creds_loaded = True

    telegram_enabled = st.checkbox(
        "Enable",
        help="Sends a neutral tilt update to a Telegram bot you control — not automatic 24/7 monitoring (see note below).",
    )

    if telegram_enabled:
        with st.expander("First time? 3-step setup (2 min)"):
            st.markdown(
                "1. In **Telegram**, message **@BotFather** → send `/newbot` → follow "
                "prompts → copy the **token** it gives you.\n"
                "2. Message your new bot anything (e.g. \"hi\").\n"
                "3. Visit `api.telegram.org/bot<TOKEN>/getUpdates` in a browser "
                "(replace `<TOKEN>` — no brackets) → find `\"chat\":{\"id\":...}` → "
                "that number is your **Chat ID**."
            )

        telegram_bot_token = st.text_input("Bot token", key="telegram_bot_token", type="password")
        telegram_chat_id = st.text_input("Chat ID", key="telegram_chat_id", type="password")
        if telegram_bot_token or telegram_chat_id:
            save_telegram(telegram_bot_token, telegram_chat_id)

        st.caption(
            "Only sends when you tap the button below — Streamlit can't run in "
            "the background and monitor unattended (that would need a separate "
            "always-on script)."
        )

        # This button lives in the sidebar so it's always in the same place —
        # it sends whatever ticker/tilt was last computed by Run analysis.
        if st.button("📨 Send last tilt to Telegram"):
            if "last_lean" not in st.session_state:
                st.warning("Run an analysis first (click 'Run analysis' above) — then this button sends that result.")
            elif not telegram_bot_token or not telegram_chat_id:
                st.error("Enter both a bot token and chat ID above first.")
            else:
                message = (
                    f"BullBear update for {st.session_state['last_ticker']}: {st.session_state['last_lean']}. "
                    "Descriptive technical reading, not a buy/sell recommendation."
                )
                sent, info = send_telegram_message(telegram_bot_token, telegram_chat_id, message)
                if sent:
                    st.success("Sent.")
                else:
                    st.error(f"Couldn't send: {info}")
    else:
        telegram_bot_token, telegram_chat_id = "", ""


# ----------------------------------------------------------------------
# RATE-LIMIT HANDLING — yfinance scrapes Yahoo's public site rather than
# using an official paid API, and Yahoo rate-limits more aggressively for
# cloud-hosted traffic (many apps sharing the same IP range) than for a
# home connection. This retries briefly before giving up, and longer
# cache times below reduce how often we hit Yahoo in the first place.
# ----------------------------------------------------------------------
def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


def yf_call_with_retry(fn, retries: int = 2, delay_seconds: float = 4.0):
    """Runs a yfinance call, retrying briefly on rate-limit errors before giving up."""
    last_exception = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if _is_rate_limit_error(e) and attempt < retries:
                time.sleep(delay_seconds)
                continue
            raise
    raise last_exception


# ----------------------------------------------------------------------
# 3. FETCH PRICE DATA
# ----------------------------------------------------------------------
@st.cache_data(ttl=900)  # was 300s — longer cache means fewer Yahoo requests
def load_chart_data(ticker_symbol: str, yf_period: str, yf_interval: str) -> pd.DataFrame:
    """
    Data for the chart — resolution depends on the timeframe picked.
    prepost=True includes pre-market/after-hours bars for intraday views,
    so the chart doesn't just flatline at the last regular-session close.
    """
    stock = yf.Ticker(ticker_symbol)
    df = yf_call_with_retry(lambda: stock.history(period=yf_period, interval=yf_interval, prepost=True))
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker_symbol}'.")
    return df


@st.cache_data(ttl=900)
def load_daily_data(ticker_symbol: str) -> pd.DataFrame:
    """
    Always-daily data (1 year), used for the Indicator Lean and the
    statistical price range — so those stay stable even when the chart
    above is zoomed into an intraday view.
    """
    stock = yf.Ticker(ticker_symbol)
    df = yf_call_with_retry(lambda: stock.history(period="1y", interval="1d"))
    if df.empty:
        raise ValueError(f"No daily data found for ticker '{ticker_symbol}'.")
    return df


@st.cache_data(ttl=120)  # was 60s — still fairly fresh, but fewer calls
def get_realtime_price_info(ticker_symbol: str) -> dict:
    """
    Pulls whatever real-time/extended-hours price fields Yahoo has for this
    ticker right now: regular session price plus pre-market or after-hours
    price if the market is currently closed. Not all tickers/exchanges
    report extended-hours data (this is mainly a US thing) — fields are
    None when unavailable, and the app falls back to the last close.
    """
    stock = yf.Ticker(ticker_symbol)
    info = yf_call_with_retry(lambda: stock.info) or {}
    return {
        "market_state": info.get("marketState"),  # e.g. "REGULAR", "PRE", "POST", "CLOSED"
        "regular_price": info.get("regularMarketPrice"),
        "post_price": info.get("postMarketPrice"),
        "post_change": info.get("postMarketChange"),
        "pre_price": info.get("preMarketPrice"),
        "pre_change": info.get("preMarketChange"),
        "currency": info.get("currency") or "USD",
    }


CURRENCY_SYMBOLS = {
    "USD": "$", "HKD": "HK$", "KRW": "₩", "SGD": "S$",
    "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥",
}


def money(value, currency_code: str = "USD") -> str:
    """Formats a number with the right currency symbol for the ticker's home market."""
    if value is None:
        return "N/A"
    symbol = CURRENCY_SYMBOLS.get(currency_code, currency_code + " ")
    return f"{symbol}{value:,.2f}"


@st.cache_data(ttl=3600)  # fundamentals barely change hour to hour — cache 1hr
def get_fundamentals(ticker_symbol: str) -> dict:
    """
    Pulls company fundamentals + analyst data from yfinance's info dict.
    Uses .get() defensively throughout since not every field is populated
    for every ticker (e.g. dividendYield is often missing for non-payers,
    and analyst target fields are often missing for non-US tickers).
    """
    stock = yf.Ticker(ticker_symbol)
    info = yf_call_with_retry(lambda: stock.info) or {}
    return {
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "dividend_yield": info.get("dividendYield"),
        "trailing_eps": info.get("trailingEps"),
        "beta": info.get("beta"),
        "target_mean_price": info.get("targetMeanPrice"),
        "target_high_price": info.get("targetHighPrice"),
        "target_low_price": info.get("targetLowPrice"),
        "recommendation_key": info.get("recommendationKey"),
        "num_analyst_opinions": info.get("numberOfAnalystOpinions"),
        "currency": info.get("currency") or "USD",
    }


@st.cache_data(ttl=3600)
def get_earnings_and_dividends(ticker_symbol: str) -> dict:
    """
    Next earnings date (if available) and recent dividend history.
    yfinance's exact fields for earnings dates have shifted across
    versions, so this tries a couple of approaches defensively.
    """
    stock = yf.Ticker(ticker_symbol)
    next_earnings = None
    try:
        edates = yf_call_with_retry(lambda: stock.get_earnings_dates(limit=8))
        if edates is not None and not edates.empty:
            future = edates[edates.index > pd.Timestamp.now(tz=edates.index.tz)]
            if not future.empty:
                next_earnings = future.index.min()
            else:
                next_earnings = edates.index.max()  # most recent past one, as fallback
    except Exception:
        pass

    dividends = pd.Series(dtype=float)
    try:
        dividends = yf_call_with_retry(lambda: stock.dividends)
        if dividends is not None and not dividends.empty:
            dividends = dividends.tail(8)  # most recent 8 payments
    except Exception:
        pass

    return {"next_earnings": next_earnings, "dividends": dividends}


def format_market_cap(value, currency_code: str = "USD") -> str:
    """Formats a raw market cap number into X.XT / X.XB / X.XM with the right currency symbol."""
    if not value:
        return "N/A"
    symbol = CURRENCY_SYMBOLS.get(currency_code, currency_code + " ")
    if value >= 1e12:
        return f"{symbol}{value / 1e12:.2f}T"
    elif value >= 1e9:
        return f"{symbol}{value / 1e9:.1f}B"
    elif value >= 1e6:
        return f"{symbol}{value / 1e6:.1f}M"
    return f"{symbol}{value:,.0f}"


SECTOR_BENCHMARK_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
}


def compute_risk_metrics(daily_df: pd.DataFrame) -> dict:
    """
    Max drawdown = the largest peak-to-trough decline in the lookback
    window. Annualized volatility scales daily volatility up to a
    yearly figure (roughly 252 trading days) so it's comparable across
    stocks regardless of how long each one's history is.
    Both describe the PAST — not a forecast of future risk.
    """
    close = daily_df["Close"]
    running_max = close.cummax()
    drawdown = close / running_max - 1
    max_drawdown = drawdown.min()

    annualized_vol = daily_df["LogReturn"].std() * np.sqrt(252)

    return {"max_drawdown": max_drawdown, "annualized_vol": annualized_vol}


# ----------------------------------------------------------------------
# 4. TECHNICAL INDICATORS (works on any OHLC dataframe, daily or intraday)
# ----------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    std20 = df["Close"].rolling(window=20).std()
    df["BB_upper"] = df["SMA20"] + (2 * std20)
    df["BB_lower"] = df["SMA20"] - (2 * std20)

    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))

    return df


# ----------------------------------------------------------------------
# 5. INDICATOR LEAN (transparent heuristic — NOT a recommendation)
# ----------------------------------------------------------------------
def compute_indicator_lean(latest: pd.Series) -> tuple[str, list[str], int]:
    bullish_points = 0
    bearish_points = 0
    reasons = []

    if latest["Close"] > latest["SMA20"]:
        bullish_points += 1
        reasons.append("Price is above its 20-day average (short-term trend is upward).")
    else:
        bearish_points += 1
        reasons.append("Price is below its 20-day average (short-term trend is downward).")

    if latest["SMA20"] > latest["SMA50"]:
        bullish_points += 1
        reasons.append("20-day average is above the 50-day average (recent trend is stronger than the longer-term one).")
    else:
        bearish_points += 1
        reasons.append("20-day average is below the 50-day average (recent trend is weaker than the longer-term one).")

    if latest["MACD"] > latest["MACD_signal"]:
        bullish_points += 1
        reasons.append("MACD is above its signal line (momentum is strengthening).")
    else:
        bearish_points += 1
        reasons.append("MACD is below its signal line (momentum is weakening).")

    if 30 <= latest["RSI14"] <= 70:
        reasons.append(f"RSI ({latest['RSI14']:.0f}) is in a neutral range (neither stretched up nor down).")
    elif latest["RSI14"] > 70:
        bearish_points += 1
        reasons.append(f"RSI ({latest['RSI14']:.0f}) is in the overbought zone (price has risen quickly — may be due to cool off).")
    else:
        bullish_points += 1
        reasons.append(f"RSI ({latest['RSI14']:.0f}) is in the oversold zone (price has fallen quickly — may be due to bounce).")

    indicator_score = bullish_points - bearish_points  # ranges roughly -4 to +4

    if bullish_points > bearish_points:
        lean = f"Bullish tilt ({bullish_points} of 4 indicators positive)"
    elif bearish_points > bullish_points:
        lean = f"Bearish tilt ({bearish_points} of 4 indicators negative)"
    else:
        lean = "Mixed / no clear tilt"

    return lean, reasons, indicator_score


# ----------------------------------------------------------------------
# 6. STATISTICAL PRICE RANGE (volatility-based, NOT a prediction)
# ----------------------------------------------------------------------
def compute_price_range(df: pd.DataFrame, days_ahead: int, confidence: float = 0.68):
    daily_vol = df["LogReturn"].std()
    horizon_vol = daily_vol * np.sqrt(days_ahead)
    last_price = df["Close"].iloc[-1]
    z = 1.0 if confidence == 0.68 else 1.96
    upper = last_price * np.exp(z * horizon_vol)
    lower = last_price * np.exp(-z * horizon_vol)
    return lower, upper


def whats_changed_today(daily_df: pd.DataFrame) -> list[str]:
    """
    Factual, computed comparison of today's bar vs. yesterday's — only
    reports things that actually happened (a threshold crossed, a volume
    spike), never an invented narrative. Empty list if nothing notable.
    """
    if len(daily_df) < 21:
        return []

    today = daily_df.iloc[-1]
    yesterday = daily_df.iloc[-2]
    changes = []

    if pd.notna(today["RSI14"]) and pd.notna(yesterday["RSI14"]):
        if yesterday["RSI14"] >= 30 > today["RSI14"]:
            changes.append(f"RSI crossed below 30 today (now {today['RSI14']:.0f}) — entered the oversold zone.")
        elif yesterday["RSI14"] <= 70 < today["RSI14"]:
            changes.append(f"RSI crossed above 70 today (now {today['RSI14']:.0f}) — entered the overbought zone.")

    if pd.notna(today["SMA20"]) and pd.notna(yesterday["SMA20"]):
        if yesterday["Close"] <= yesterday["SMA20"] < today["Close"]:
            changes.append("Price crossed above its 20-day average today.")
        elif yesterday["Close"] >= yesterday["SMA20"] > today["Close"]:
            changes.append("Price crossed below its 20-day average today.")

    if pd.notna(today["MACD"]) and pd.notna(yesterday["MACD"]):
        if yesterday["MACD"] <= yesterday["MACD_signal"] < today["MACD"]:
            changes.append("MACD crossed above its signal line today.")
        elif yesterday["MACD"] >= yesterday["MACD_signal"] > today["MACD"]:
            changes.append("MACD crossed below its signal line today.")

    avg_volume_20d = daily_df["Volume"].iloc[-21:-1].mean()
    if avg_volume_20d and today["Volume"] > avg_volume_20d * 1.4:
        pct_above = (today["Volume"] / avg_volume_20d - 1) * 100
        changes.append(f"Volume today is {pct_above:.0f}% above its 20-day average — notably heavier trading than usual.")

    return changes


def historical_scenario_ranges(daily_df: pd.DataFrame, days_ahead: int) -> dict | None:
    """
    Instead of inventing bull/base/bear percentage outcomes, this looks at
    this stock's OWN actual historical N-day forward returns over the past
    year and reports real percentiles from that distribution. Still not a
    forecast — it's "here's the spread of what actually happened over
    similar-length windows in the past," which the person can weigh
    themselves rather than being handed a made-up number.
    """
    closes = daily_df["Close"]
    n = len(closes)
    if n < days_ahead + 30:
        return None

    forward_returns = [
        (closes.iloc[i + days_ahead] - closes.iloc[i]) / closes.iloc[i]
        for i in range(n - days_ahead)
    ]
    forward_returns = np.array(forward_returns)
    return {
        "worst_10pct": np.percentile(forward_returns, 10) * 100,
        "median": np.percentile(forward_returns, 50) * 100,
        "best_10pct": np.percentile(forward_returns, 90) * 100,
        "sample_size": len(forward_returns),
    }


def compute_signal_agreement(indicator_score: int, news_items: list[dict]) -> str:
    """
    An honest alternative to a fabricated 'confidence %': just reports
    whether the technical read and headline tone happen to point the same
    way right now, or not. Not a probability — a simple factual observation.
    """
    if not news_items:
        return "Not enough headlines to compare against the technical read."

    sentiments = [tag_sentiment(i["title"] + " " + i.get("description", "")) for i in news_items]
    n_pos = sentiments.count("positive")
    n_neg = sentiments.count("negative")
    news_lean = "positive" if n_pos > n_neg else ("negative" if n_neg > n_pos else "mixed")
    tech_lean = "positive" if indicator_score > 0 else ("negative" if indicator_score < 0 else "mixed")

    if tech_lean == "mixed" or news_lean == "mixed":
        return "Signals are mixed — technical and headline tone don't clearly point the same way."
    elif tech_lean == news_lean:
        return "Aligned — both the technical read and headline tone point the same direction right now."
    else:
        return "Conflicting — the technical read and headline tone point in different directions right now."


# ----------------------------------------------------------------------
# 7. TEXT CLEANUP — strips HTML tags/entities out of RSS descriptions
# ----------------------------------------------------------------------
def clean_html(raw_text: str) -> str:
    """Removes HTML tags (e.g. <p>) and decodes entities (e.g. &amp;) from RSS text."""
    if not raw_text:
        return ""
    text = html_lib.unescape(raw_text)
    text = re.sub(r"<[^>]+>", " ", text)   # strip any tag like <p>, </p>, <a href=...>
    text = re.sub(r"\s+", " ", text).strip()  # collapse extra whitespace
    return text


# ----------------------------------------------------------------------
# 8. NEWS: headlines + links + cleaned short summaries
# ----------------------------------------------------------------------
@st.cache_data(ttl=600)  # was 300s
def get_news_items(ticker_symbol: str, max_items: int = 6) -> list[dict]:
    """
    Uses yfinance's own news lookup (tied to the specific ticker) instead
    of a generic RSS feed, since the RSS feed was returning unrelated
    market-wide stories. yfinance has changed its exact news response
    shape across versions, so this defensively checks a few possible
    key layouts rather than assuming one fixed structure.
    """
    stock = yf.Ticker(ticker_symbol)
    try:
        raw_news = yf_call_with_retry(lambda: stock.news) or []
    except Exception:
        raw_news = []

    items = []
    for n in raw_news[:max_items]:
        # Newer yfinance versions nest the actual fields under "content";
        # older versions put them directly on the item.
        content = n.get("content", n) if isinstance(n, dict) else {}

        title = content.get("title") or n.get("title") or ""
        if not title:
            continue  # skip anything we can't even get a headline from

        link = (
            (content.get("canonicalUrl") or {}).get("url")
            or (content.get("clickThroughUrl") or {}).get("url")
            or n.get("link")
            or ""
        )
        published = content.get("pubDate") or n.get("providerPublishTime") or ""
        description = content.get("summary") or content.get("description") or ""

        items.append({
            "title": clean_html(title),
            "link": link,
            "published": str(published),
            "description": clean_html(description),
        })
    return items


# ----------------------------------------------------------------------
# 9. FREE, LOCAL SENTIMENT (VADER — no API, no cost)
# ----------------------------------------------------------------------
# VADER is a well-established rule-based sentiment tool tuned for short,
# informal text like headlines. It runs entirely on your own machine —
# no API key, no cost — and handles negation ("not bad") and intensity
# better than plain keyword counting, though it's still far from perfect:
# it can't verify facts, catch subtle sarcasm, or know which headlines
# actually matter more than others.
@st.cache_resource
def get_sentiment_analyzer():
    return SentimentIntensityAnalyzer()


def tag_sentiment(text: str) -> str:
    """Returns 'positive', 'negative', or 'neutral' using VADER's compound score."""
    analyzer = get_sentiment_analyzer()
    compound = analyzer.polarity_scores(text)["compound"]
    # Standard VADER thresholds
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"


# ----------------------------------------------------------------------
# 10. FREE NARRATIVE — rule-based, no API, connects tilt + headline tone
# ----------------------------------------------------------------------
def generate_narrative(ticker_symbol: str, lean: str, reasons: list[str],
                        news_items: list[dict]) -> str:
    """
    Builds a plain-language paragraph from the indicator lean (grounded in
    real price data) and the tone of recent headlines (a much weaker,
    illustrative signal from simple keyword matching on a handful of
    articles). Deliberately does NOT treat a small headline tally as
    something meaningful enough to weigh in a buy/sell direction — a
    handful of headlines skewing one way is easy to happen by chance and
    says very little on its own.
    """
    sentiments = [tag_sentiment(i["title"] + " " + i.get("description", "")) for i in news_items]
    n_pos = sentiments.count("positive")
    n_neg = sentiments.count("negative")
    total = len(sentiments)

    lines = [
        f"The technical readout for {ticker_symbol} is: **{lean}**, based on where the "
        "price sits relative to its moving averages, RSI, and MACD — this part is "
        "grounded in actual price data.",
    ]

    if total == 0:
        lines.append("No recent headlines were found to add context.")
    else:
        # Only describe headline tone in soft, qualitative terms — no
        # implication that a few articles' wording is itself informative.
        if n_pos == n_neg:
            tone_desc = "an even mix of tone"
        elif abs(n_pos - n_neg) == 1:
            tone_desc = "essentially an even mix of tone (within a single headline of each other)"
        elif n_pos > n_neg:
            tone_desc = "a somewhat more positive-sounding mix of wording"
        else:
            tone_desc = "a somewhat more negative-sounding mix of wording"

        lines.append(
            f"The {total} recent headlines shown above have {tone_desc}, based on simple "
            "keyword matching (words like 'beat' or 'surge' vs. 'miss' or 'downgrade'). "
            "On its own, this is a weak signal — a small sample of headlines skewing one "
            "way is easy to happen by chance, and keyword matching can't tell sarcasm, "
            "negation, or genuine importance apart. It is not, by itself, a reasonable "
            "basis to buy or sell — read the actual headlines above for the substance."
        )

    lines.append(
        "This is a rule-based description of the current picture, not a forecast — "
        "keyword matching is blunt and can misread sarcasm, negation, or nuance, "
        "so treat it as a starting point for your own reading of the news, not a verdict."
    )
    return " ".join(lines)


def explain_like_im_5(ticker_symbol: str, lean: str, news_items: list[dict]) -> str:
    """
    A super-simplified, plain-language restatement of the SAME factual
    lean already computed above — no new claims, just simpler words.
    Deliberately still hedged; simplicity isn't a license to overstate.
    """
    if "Bullish" in lean:
        direction = f"a few signs are pointing up for {ticker_symbol} right now"
    elif "Bearish" in lean:
        direction = f"a few signs are pointing down for {ticker_symbol} right now"
    else:
        direction = f"the signals for {ticker_symbol} are mixed right now — no clear direction"

    return (
        f"In simple terms: {direction}, based on how the price has been moving "
        "compared to its own recent averages. That's just a description of the "
        "recent past, though — nobody, including this app, actually knows what "
        "happens next. Please don't treat this as advice to buy or sell."
    )


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 11. HISTORICAL SIGNAL CHECK (a fixed, honestly-caveated backtest)
# ----------------------------------------------------------------------
def historical_signal_check(daily_df: pd.DataFrame, forward_days: int = 5):
    """
    Retrospectively checks whether the Indicator Lean's underlying score
    has lined up with this ONE stock's own subsequent price moves over
    its own past year. This is a single-asset, single-window check with
    no transaction costs and no out-of-sample data — a clean-looking
    correlation here is easy to get from noise alone and does NOT mean
    the same pattern will hold going forward.
    """
    scores, forward_returns = [], []
    n = len(daily_df)

    for i in range(60, n - forward_days):
        latest = daily_df.iloc[i]
        if pd.isna(latest[["SMA20", "SMA50", "RSI14", "MACD", "MACD_signal"]]).any():
            continue
        _, _, score = compute_indicator_lean(latest)
        future_ret = (daily_df["Close"].iloc[i + forward_days] - daily_df["Close"].iloc[i]) / daily_df["Close"].iloc[i]
        scores.append(score)
        forward_returns.append(future_ret)

    if len(scores) < 20:
        return None

    return float(np.corrcoef(scores, forward_returns)[0, 1])


# ----------------------------------------------------------------------
# 12. PORTFOLIO ALLOCATION (backward-looking Sharpe optimization)
# ----------------------------------------------------------------------
def optimize_portfolio(tickers: list[str], iterations: int = 3000):
    """
    Randomly samples portfolio weightings and keeps whichever had the best
    historical Sharpe ratio (return divided by volatility, ignoring a
    risk-free rate) over the lookback window. IMPORTANT: this literally
    finds whatever WOULD have worked best in the past — a well-known trap
    among quants (sometimes called Sharpe-ratio overfitting), since the
    mix of stocks that happened to do best historically is not reliably
    the mix that will do best going forward. Shown for research interest,
    not as a recommended allocation.
    """
    price_data = {}
    for t in tickers:
        try:
            price_data[t] = load_daily_data(t)["Close"]
        except Exception:
            continue

    if len(price_data) < 2:
        return None, None

    prices = pd.DataFrame(price_data).dropna()
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov = returns.cov()
    valid_tickers = list(prices.columns)

    best_sharpe = -np.inf
    best_weights = None
    for _ in range(iterations):
        w = np.random.random(len(valid_tickers))
        w /= w.sum()
        port_return = np.dot(w, mean_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        sharpe_ratio = port_return / port_vol if port_vol != 0 else -np.inf
        if sharpe_ratio > best_sharpe:
            best_sharpe = sharpe_ratio
            best_weights = w

    return dict(zip(valid_tickers, best_weights)), best_sharpe


# ----------------------------------------------------------------------
# 13. TELEGRAM NOTIFICATIONS — neutral tilt updates, sent manually.
#     Note: Streamlit only runs when someone has the page open or clicks
#     something. This can NOT silently monitor the market and text you
#     unattended 24/7 — true background monitoring needs separate
#     infrastructure (e.g. a scheduled script running on its own).
# ----------------------------------------------------------------------
def send_telegram_message(bot_token: str, chat_id: str, message: str) -> tuple[bool, str]:
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=10)
        resp.raise_for_status()
        return True, "Sent."
    except Exception as e:
        return False, str(e)


# ----------------------------------------------------------------------
# 14. WATCHLIST SCAN — ranks multiple tickers by the same transparent
#     heuristics used above (indicator lean + headline tone). This is a
#     ranked scan of well-known signals, not a "best stocks" verdict.
# ----------------------------------------------------------------------
def scan_watchlist(tickers: list[str]) -> pd.DataFrame:
    rows = []
    progress = st.progress(0.0, text="Starting scan...")

    for idx, t in enumerate(tickers):
        progress.progress((idx) / max(len(tickers), 1), text=f"Scanning {t}...")
        try:
            df = load_daily_data(t)
            df = add_indicators(df)
            latest = df.iloc[-1]
            lean, _reasons, indicator_score = compute_indicator_lean(latest)

            news = get_news_items(t, max_items=5)
            sentiments = [tag_sentiment(i["title"] + " " + i.get("description", "")) for i in news]
            n_pos = sentiments.count("positive")
            n_neg = sentiments.count("negative")

            combined_score = indicator_score + (n_pos - n_neg)

            sector = get_fundamentals(t).get("sector") or "Unknown"

            rows.append({
                "Ticker": t,
                "Last price": round(float(latest["Close"]), 2),
                "Sector": sector,
                "Indicator lean": lean,
                "Headlines +/-": f"{n_pos}/{n_neg}",
                "Combined tilt score": combined_score,
            })
        except Exception:
            continue  # skip tickers that fail to load (bad symbol, no data, etc.)

    progress.progress(1.0, text="Done.")
    progress.empty()
    return pd.DataFrame(rows)


def sector_concentration(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Factual sector breakdown of the watchlist — equal-weighted (counts each
    ticker once, not by dollar amount, since we don't know your actual
    position sizes). Flags concentration, doesn't recommend a fix.
    """
    counts = results_df["Sector"].value_counts()
    pct = (counts / counts.sum() * 100).round(1)
    return pd.DataFrame({"Sector": counts.index, "Count": counts.values, "% of watchlist": pct.values})


# ----------------------------------------------------------------------
# GENERAL MARKET NEWS — not tied to any one ticker. Reuses the same
# get_news_items() function (proven reliable via yfinance) pointed at a
# broad market proxy (SPY, the S&P 500 ETF) instead of a single stock.
# ----------------------------------------------------------------------
with st.expander("🗞️ Market news (general, not stock-specific)", expanded=False):
    with st.spinner("Loading market headlines..."):
        try:
            market_news = get_news_items("SPY", max_items=6)
        except Exception:
            market_news = []

    if not market_news:
        st.write("Couldn't load general market news right now.")
    else:
        for item in market_news:
            sentiment = tag_sentiment(item["title"] + " " + item.get("description", ""))
            badge = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}[sentiment]
            st.markdown(f"{badge} **[{item['title']}]({item['link']})**")
            if item["description"]:
                st.caption(item["description"])


# ----------------------------------------------------------------------
# 12. MAIN LOGIC
# ----------------------------------------------------------------------
if run_button:
    try:
        yf_period, yf_interval = CHART_PERIOD_MAP[period_choice]
        unit_label = "day" if yf_interval == "1d" else "bar"

        with st.spinner(f"Loading {ticker} data..."):
            chart_df = load_chart_data(ticker, yf_period, yf_interval)
            chart_df = add_indicators(chart_df)

            daily_df = load_daily_data(ticker)
            daily_df = add_indicators(daily_df)

            rt = get_realtime_price_info(ticker)

        latest_chart = chart_df.iloc[-1]
        latest_daily = daily_df.iloc[-1]
        currency = rt["currency"]

        # Prefer the real-time regular price if Yahoo has it; otherwise
        # fall back to the last bar we already downloaded.
        display_price = rt["regular_price"] if rt["regular_price"] else latest_chart["Close"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Last price", money(display_price, currency),
                    help="The most recent regular-session price where available, otherwise the last downloaded bar.")
        col2.metric(f"20-{unit_label} avg", money(latest_chart['SMA20'], currency) if pd.notna(latest_chart['SMA20']) else "N/A",
                    help="Average closing price over the last 20 periods. Smooths out day-to-day noise to show the short-term trend.")
        col3.metric(f"50-{unit_label} avg", money(latest_chart['SMA50'], currency) if pd.notna(latest_chart['SMA50']) else "N/A",
                    help="Average closing price over the last 50 periods — a slower-moving, longer-term trend line.")
        col4.metric("RSI (14)", f"{latest_chart['RSI14']:.1f}" if pd.notna(latest_chart['RSI14']) else "N/A",
                    help="Relative Strength Index: a 0-100 momentum score. Traditionally, >70 is 'overbought', <30 is 'oversold' — rules of thumb, not guarantees.")

        # --- Extended-hours indicator (pre-market / after-hours) ---
        # Mainly available for US tickers — Yahoo doesn't report this for
        # every exchange, so this quietly does nothing if data's missing.
        if rt["market_state"] in ("POST", "PREPRE", "POSTPOST") and rt["post_price"]:
            st.caption(
                f"🌙 After-hours: {money(rt['post_price'], currency)} "
                f"({'+' if (rt['post_change'] or 0) >= 0 else ''}{rt['post_change']:.2f})"
            )
        elif rt["market_state"] == "PRE" and rt["pre_price"]:
            st.caption(
                f"🌅 Pre-market: {money(rt['pre_price'], currency)} "
                f"({'+' if (rt['pre_change'] or 0) >= 0 else ''}{rt['pre_change']:.2f})"
            )

        # --- Price chart ---
        st.subheader(f"{ticker} price chart ({period_choice})")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Close"], name="Price"))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA20"], name=f"20-{unit_label} avg"))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA50"], name=f"50-{unit_label} avg"))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_upper"], name="Upper band",
                                  line=dict(dash="dot"), opacity=0.4))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_lower"], name="Lower band",
                                  line=dict(dash="dot"), opacity=0.4))
        fig.update_layout(height=450, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        if yf_interval != "1d":
            st.caption(
                f"Showing intraday {yf_interval} bars. The Indicator Lean and price range "
                "below always use standard daily data, so they stay stable regardless of "
                "this chart's zoom level."
            )

        # --- Indicator Lean (always from daily data) ---
        st.subheader("Indicator lean")
        lean, reasons, _score = compute_indicator_lean(latest_daily)
        st.session_state["last_lean"] = lean
        st.session_state["last_ticker"] = ticker
        st.metric("Current tilt (based on daily data)", lean,
                  help="A simple count of how many well-known indicators (moving averages, MACD, RSI) point bullish vs. bearish right now. A transparency tool, not a validated strategy.")
        for r in reasons:
            st.write(f"- {r}")

        # --- What changed today (factual, computed — not a narrative) ---
        changes_today = whats_changed_today(daily_df)
        if changes_today:
            st.markdown("**What changed today:**")
            for c in changes_today:
                st.write(f"- {c}")
        else:
            st.caption("No notable threshold crossings or volume spikes today.")

        # --- Statistical price range (always from daily data) ---
        st.subheader(f"Statistical price range — next {horizon_days} trading day(s)")
        lower68, upper68 = compute_price_range(daily_df, horizon_days, confidence=0.68)
        lower95, upper95 = compute_price_range(daily_df, horizon_days, confidence=0.95)
        rcol1, rcol2 = st.columns(2)
        rcol1.metric("~68% range (1 std dev)", f"{money(lower68, currency)} – {money(upper68, currency)}",
                     help="1 standard deviation. Based on this stock's own past daily price swings, if those swings behaved like a bell curve, about 68% of past outcomes fell within 1 standard deviation of the average move. Describes typical past volatility, not a forecast.")
        rcol2.metric("~95% range (2 std dev)", f"{money(lower95, currency)} – {money(upper95, currency)}",
                     help="2 standard deviations — a wider band. About 95% of past outcomes fell within this range if moves behaved like a bell curve. Still based on the past, not a prediction.")
        st.warning(
            "Based purely on this stock's own historical volatility, assuming no "
            "directional drift. Not a prediction — real prices can move outside "
            "this range, especially around news events."
        )

        # --- Historical scenario ranges — REAL percentiles from this
        #     stock's own past N-day returns, not invented bull/base/bear
        #     numbers. Answers "what actually happened over similar windows
        #     historically" rather than forecasting what will happen now. ---
        st.subheader(f"Historical {horizon_days}-day outcomes (this stock's own past year)")
        scenario = historical_scenario_ranges(daily_df, horizon_days)
        if scenario:
            scol1, scol2, scol3 = st.columns(3)
            scol1.metric("Worst 10% of past outcomes", f"{scenario['worst_10pct']:+.1f}%")
            scol2.metric("Median past outcome", f"{scenario['median']:+.1f}%")
            scol3.metric("Best 10% of past outcomes", f"{scenario['best_10pct']:+.1f}%")
            st.caption(
                f"Based on {scenario['sample_size']} overlapping {horizon_days}-day windows from this "
                "stock's own past year. This is what actually happened historically over "
                "similar-length periods — not a forecast, and past windows overlap so they "
                "aren't fully independent samples. A genuinely new event (earnings, news) can "
                "produce an outcome outside this entire historical range."
            )
        else:
            st.caption("Not enough price history to compute this yet.")

        # --- Signal agreement — honest alternative to a fabricated confidence score ---
        st.subheader("Signal agreement")
        st.write(compute_signal_agreement(_score, get_news_items(ticker)))
        st.caption(
            "This just states whether the technical read and headline tone happen to "
            "point the same way right now — it is NOT a probability or confidence score. "
            "Agreement between two weak signals doesn't make either one strong."
        )

        # --- News ---
        st.subheader("Recent news")
        with st.spinner("Fetching headlines..."):
            news_items = get_news_items(ticker)

        if not news_items:
            st.write("No headlines found for this ticker right now.")
        else:
            for item in news_items:
                with st.container(border=True):
                    sentiment = tag_sentiment(item["title"] + " " + item.get("description", ""))
                    badge = {"positive": "🟢 positive tone", "negative": "🔴 negative tone", "neutral": "⚪ neutral tone"}[sentiment]
                    st.markdown(f"**[{item['title']}]({item['link']})** — {badge}")
                    if item["published"]:
                        st.caption(item["published"])
                    if item["description"]:
                        st.caption(item["description"])

        # --- Narrative connecting technicals + news (free, rule-based) ---
        st.subheader("What's driving the current picture")
        st.write(generate_narrative(ticker, lean, reasons, news_items))
        if st.toggle("🧒 Explain like I'm 5"):
            st.info(explain_like_im_5(ticker, lean, news_items))

        st.divider()
        st.markdown("### 📋 Detailed breakdown")
        st.caption("Everything below is supporting detail — fundamentals, analyst estimates, earnings, and risk metrics.")

        # --- Fundamentals + Analyst view + Earnings/Dividends + Risk profile ---
        with st.spinner("Loading fundamentals..."):
            fundamentals = get_fundamentals(ticker)
            ed_data = get_earnings_and_dividends(ticker)
            risk = compute_risk_metrics(daily_df)

        st.subheader("Fundamentals")
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        fcol1.metric("Sector", fundamentals["sector"] or "N/A")
        fcol2.metric("Market cap", format_market_cap(fundamentals["market_cap"], currency))
        fcol3.metric("P/E (trailing)", f"{fundamentals['trailing_pe']:.1f}" if fundamentals["trailing_pe"] else "N/A",
                     help="Price divided by trailing 12-month earnings per share. Higher generally means the market is pricing in more future growth (or the stock is more expensive relative to current profit) — context-dependent, not good/bad on its own.")
        div_yield = fundamentals["dividend_yield"]
        # yfinance has changed whether this field is a fraction (0.02) or
        # already a percent (2.0) across versions — this handles either.
        div_yield_pct = div_yield * 100 if (div_yield and div_yield < 1) else div_yield
        fcol4.metric("Dividend yield", f"{div_yield_pct:.2f}%" if div_yield_pct else "None",
                     help="Annual dividend payments as a percentage of the current share price.")

        st.subheader("Analyst view")
        st.caption(
            "These are published estimates from Wall Street analysts covering this "
            "stock — not this app's own calculation, and not a guarantee of future price. "
            "Coverage is often thinner or unavailable for non-US tickers."
        )
        if fundamentals["target_mean_price"]:
            acol1, acol2 = st.columns(2)
            with acol1:
                st.metric("Analyst avg. target", money(fundamentals['target_mean_price'], currency))
                st.metric("# of analysts", fundamentals["num_analyst_opinions"] or "N/A")
            with acol2:
                target_range_text = (
                    f"{money(fundamentals['target_low_price'], currency)} – {money(fundamentals['target_high_price'], currency)}"
                    if fundamentals["target_low_price"] and fundamentals["target_high_price"] else "N/A"
                )
                st.markdown(f"**Target range**  \n{target_range_text}")
                st.markdown(f"**Consensus**  \n{(fundamentals['recommendation_key'] or 'N/A').replace('_', ' ').title()}")
        else:
            st.write("No analyst target data available for this ticker.")

        st.subheader("Earnings & dividends")
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            if ed_data["next_earnings"] is not None:
                st.metric("Next/most recent earnings date", ed_data["next_earnings"].strftime("%Y-%m-%d"))
            else:
                st.write("No earnings date data available.")
        with ecol2:
            if not ed_data["dividends"].empty:
                st.caption("Last 8 dividend payments")
                st.bar_chart(ed_data["dividends"])
            else:
                st.write("No dividend history — this stock may not pay dividends.")

        st.subheader("Risk profile")
        rcol_a, rcol_b, rcol_c = st.columns(3)
        rcol_a.metric("Beta", f"{fundamentals['beta']:.2f}" if fundamentals["beta"] else "N/A",
                      help="How much this stock has historically moved relative to the overall market. 1.0 = moves roughly with the market; above 1 = historically more volatile than the market; below 1 = historically less volatile.")
        rcol_b.metric("Max drawdown (1y)", f"{risk['max_drawdown']*100:.1f}%",
                      help="The largest peak-to-trough decline over the past year. Describes the worst historical stretch in this window — not a cap on future losses.")
        rcol_c.metric("Annualized volatility", f"{risk['annualized_vol']*100:.1f}%",
                      help="How much this stock's price has typically swung over a year, scaled up from its daily moves. Higher = historically choppier.")

        sector = fundamentals["sector"]
        benchmark_ticker = SECTOR_BENCHMARK_MAP.get(sector, "SPY")
        benchmark_label = f"{sector} sector (via {benchmark_ticker})" if sector in SECTOR_BENCHMARK_MAP else f"the overall market (via {benchmark_ticker})"
        try:
            benchmark_df = add_indicators(load_daily_data(benchmark_ticker))
            benchmark_vol = benchmark_df["LogReturn"].std() * np.sqrt(252)
            st.caption(
                f"For comparison, {benchmark_label} has had roughly {benchmark_vol*100:.1f}% "
                f"annualized volatility over the same period — "
                + ("higher" if risk["annualized_vol"] > benchmark_vol else "lower")
                + f" than {ticker}'s {risk['annualized_vol']*100:.1f}%."
            )
        except Exception:
            pass  # benchmark comparison is a nice-to-have, not essential

        # --- Historical signal check (fixed backtest, honestly caveated) ---
        st.subheader("Historical signal check")
        corr = historical_signal_check(daily_df)
        if corr is not None:
            st.metric("Correlation: signal vs. 5-day forward return (past year)", f"{corr:.2f}")
        else:
            st.write("Not enough history to compute this check.")
        st.warning(
            "This checks whether the Indicator Lean's score has lined up with "
            "*this one stock's own* subsequent price moves, over its own past "
            "year only. A clean-looking number here is easy to get from noise "
            "alone with a single asset and window — it is not validation of a "
            "trading strategy, and does not mean the same pattern continues."
        )
        if telegram_enabled:
            st.caption("💡 Use the '📨 Send last tilt to Telegram' button in the sidebar to send this result.")

        st.caption(f"Data as of {datetime.now().strftime('%Y-%m-%d %H:%M')}. Not financial advice.")

    except Exception as e:
        if _is_rate_limit_error(e):
            st.error(
                "⏳ Yahoo Finance is rate-limiting requests right now. This happens "
                "sometimes on cloud hosting (many apps share the same IP range) and "
                "isn't something wrong with your app — it usually clears up within a "
                "minute or two. Try again shortly."
            )
        else:
            st.error(f"Something went wrong: {e}")

else:
    st.write("👈 Enter a ticker in the sidebar and click **Run analysis** to get started.")


# ----------------------------------------------------------------------
# WATCHLIST SCAN SECTION
# ----------------------------------------------------------------------
st.divider()
st.header("Watchlist scan")
st.caption(
    "Ranks the tickers you list by the same Indicator Lean + headline-tone "
    "heuristics used above — not a 'best stocks to buy' list. A high score "
    "just means more of these simple signals currently point the same way; "
    "it says nothing about future performance."
)

if scan_button:
    tickers_to_scan = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
    if not tickers_to_scan:
        st.warning("Add at least one ticker to scan.")
    else:
        with st.spinner(f"Scanning {len(tickers_to_scan)} tickers..."):
            results_df = scan_watchlist(tickers_to_scan)

        if results_df.empty:
            st.error("Couldn't fetch data for any of these tickers — check the symbols and try again.")
        else:
            sorted_df = results_df.sort_values("Combined tilt score", ascending=False).reset_index(drop=True)

            col_bull, col_bear = st.columns(2)
            with col_bull:
                st.subheader("Most bullish-tilted")
                st.dataframe(sorted_df.head(5), hide_index=True, use_container_width=True)
            with col_bear:
                st.subheader("Most bearish-tilted")
                st.dataframe(
                    sorted_df.sort_values("Combined tilt score", ascending=True).head(5).reset_index(drop=True),
                    hide_index=True, use_container_width=True,
                )

            with st.expander("See full scan results"):
                st.dataframe(sorted_df, hide_index=True, use_container_width=True)

            st.warning(
                "'Combined tilt score' just adds the indicator count (moving averages, MACD, RSI) "
                "to the net headline tone (positive minus negative mentions). It's a simple, transparent "
                "tally — not a validated ranking, and it can flip quickly as prices and headlines change."
            )

            # --- Sector concentration (factual — counts tickers, not $ exposure) ---
            st.subheader("Sector concentration in this watchlist")
            sec_df = sector_concentration(results_df)
            st.dataframe(sec_df, hide_index=True, use_container_width=True)
            top_sector_pct = sec_df["% of watchlist"].max() if not sec_df.empty else 0
            if top_sector_pct >= 40:
                top_sector_name = sec_df.iloc[0]["Sector"]
                st.warning(
                    f"⚠️ {top_sector_pct:.0f}% of this watchlist is in {top_sector_name} — "
                    "concentrated in one sector means sector-specific news affects a large "
                    "share of it at once. This counts tickers equally, not by dollar amount, "
                    "so it may not match your actual portfolio weighting."
                )
            else:
                st.caption("Counts each ticker equally (not by dollar exposure) — not a substitute for your actual position sizes.")

            # --- Portfolio allocation (backward-looking, research interest only) ---
            st.subheader("Historical Sharpe-optimal allocation")
            with st.spinner("Running allocation search..."):
                weights, best_sharpe = optimize_portfolio(tickers_to_scan)

            if weights is None:
                st.write("Need at least 2 valid tickers with price history to compute this.")
            else:
                weights_df = pd.DataFrame(
                    {"Ticker": list(weights.keys()), "Weight %": [w * 100 for w in weights.values()]}
                ).sort_values("Weight %", ascending=False).reset_index(drop=True)
                st.dataframe(weights_df, hide_index=True, use_container_width=True)
                st.metric("Historical Sharpe ratio of this mix", f"{best_sharpe:.2f}")
                st.warning(
                    "This is the weighting that WOULD have had the best risk-adjusted return "
                    "over the exact past window used here — found by randomly trying thousands "
                    "of combinations. This kind of backward-looking optimization is well known "
                    "(sometimes called 'Sharpe ratio overfitting') to not reliably predict which "
                    "mix will do best going forward. It's shown for research interest, not as a "
                    "recommended portfolio."
                )
else:
    st.write("👈 Edit the watchlist in the sidebar and click **Scan watchlist** to compare tickers.")

