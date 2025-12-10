import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional deps
try:
    import tushare as ts
except Exception:
    ts = None

try:
    import baostock as bs
except Exception:
    bs = None

# -----------------------------
# Basic page config
# -----------------------------
st.set_page_config(
    page_title="A股深度复盘系统 Pro",
    layout="wide",
    page_icon="📈"
)

# -----------------------------
# Data helpers
# -----------------------------
def _to_ts_code(symbol: str) -> str:
    """转换为 Tushare 所需的格式 (e.g., 600519.SH)"""
    symbol = symbol.strip()
    if symbol.endswith(".SH") or symbol.endswith(".SZ"):
        return symbol
    if symbol.isdigit():
        return f"{symbol}.SH" if symbol.startswith("6") else f"{symbol}.SZ"
    return symbol

def _to_bs_code(symbol: str) -> str:
    """转换为 Baostock 所需的格式 (e.g., sh.600519)"""
    symbol = symbol.strip()
    # 如果已经是 sh. 或 sz. 开头，直接返回
    if symbol.startswith("sh.") or symbol.startswith("sz."):
        return symbol
    # 如果是 Tushare 格式 (600519.SH)，转为 sh.600519
    if symbol.endswith(".SH"):
        return f"sh.{symbol[:6]}"
    if symbol.endswith(".SZ"):
        return f"sz.{symbol[:6]}"
    # 纯数字
    if symbol.isdigit():
        return f"sh.{symbol}" if symbol.startswith("6") else f"sz.{symbol}"
    return symbol

@st.cache_data(ttl=60 * 60 * 24) # 缓存一天
def get_stock_name(symbol: str, token: str = "") -> str:
    """
    统一获取股票名称：
    1. 优先尝试 Tushare (如果提供了 Token)
    2. 失败或无 Token 则尝试 Baostock
    3. 都失败返回空字符串
    """
    name = ""
    
    # --- 尝试 Tushare ---
    if token and ts is not None:
        try:
            ts_code = _to_ts_code(symbol)
            pro = ts.pro_api(token)
            # 只查询单只股票，速度快
            df = pro.stock_basic(ts_code=ts_code, fields='name')
            if not df.empty:
                return df.iloc[0]['name']
        except Exception:
            pass # Tushare 失败，静默进入 Baostock

    # --- 尝试 Baostock ---
    if bs is not None:
        try:
            bs_code = _to_bs_code(symbol)
            # 登录
            lg = bs.login()
            if lg.error_code == '0':
                rs = bs.query_stock_basic(code=bs_code)
                if rs.error_code == '0':
                    row = rs.get_row_data()
                    # Baostock 返回的 list 顺序: [code, code_name, ipoDate, outDate, type, status]
                    if row and len(row) > 1:
                        name = row[1]
            bs.logout()
        except Exception:
            pass

    return name

@st.cache_data(ttl=60 * 15, show_spinner=False)
def fetch_hist_tushare(symbol: str, token: str, days: int = 180,
                       adjust: str = "qfq", retry: int = 3) -> pd.DataFrame:
    """TuShare daily bars with optional qfq/hfq adjustment using adj_factor."""
    if ts is None or not token:
        return pd.DataFrame()

    pro = ts.pro_api(token)
    end = pd.Timestamp.today().strftime("%Y%m%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=days * 3)).strftime("%Y%m%d")

    ts_code = _to_ts_code(symbol)

    last_err = None
    for _ in range(retry):
        try:
            df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
            if df is None or df.empty:
                return pd.DataFrame()

            # adj factors
            if adjust in ("qfq", "hfq"):
                af = pro.adj_factor(ts_code=ts_code, start_date=start, end_date=end)
                if af is not None and not af.empty:
                    af = af.rename(columns={"trade_date": "date", "adj_factor": "factor"})
                    df = df.merge(af[["date", "factor"]], on="date", how="left")
                    df["factor"] = df["factor"].ffill().bfill()

                    if adjust == "qfq":
                        df["adj"] = df["factor"] / df["factor"].iloc[-1]
                    else:  # hfq
                        df["adj"] = df["factor"] / df["factor"].iloc[0]

                    for col in ["open", "high", "low", "close"]:
                        df[col] = df[col] * df["adj"]

            df = df.rename(columns={
                "trade_date": "date",
                "vol": "volume",
                "pct_chg": "pct_change"
            })
            df["date"] = pd.to_datetime(df["date"])
            for col in ["open", "high", "low", "close", "volume", "amount", "pct_change"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True).tail(days)
            return df[["date","open","high","low","close","volume","amount","pct_change"]]
        except Exception as e:
            last_err = e
            time.sleep(1.0)

    raise last_err

@st.cache_data(ttl=60 * 15, show_spinner=False)
def fetch_hist_baostock(symbol: str, days: int = 180, adjust: str = "qfq") -> pd.DataFrame:
    """Baostock daily bars; adjustflag supports qfq/hfq/no."""
    if bs is None:
        return pd.DataFrame()

    lg = bs.login()
    if lg.error_code != "0":
        bs.logout()
        return pd.DataFrame()

    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=days * 3)

    code = _to_bs_code(symbol) # 使用新的转换函数

    adj_flag = "3"
    if adjust == "qfq":
        adj_flag = "2"
    elif adjust == "hfq":
        adj_flag = "1"

    rs = bs.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume,amount,pctChg",
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        frequency="d",
        adjustflag=adj_flag
    )

    data = []
    while rs.error_code == "0" and rs.next():
        data.append(rs.get_row_data())

    bs.logout()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=rs.fields)
    df = df.rename(columns={"pctChg": "pct_change"})
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open","high","low","close","volume","amount","pct_change"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True).tail(days)
    return df

def fetch_hist(symbol: str, token: str, days: int = 180, adjust: str = "qfq") -> pd.DataFrame:
    """Unified entry: tushare -> baostock fallback."""
    if token:
        try:
            df = fetch_hist_tushare(symbol, token, days=days, adjust=adjust)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            st.warning(f"TuShare 拉取失败，自动切换 Baostock：{e}")
    return fetch_hist_baostock(symbol, days=days, adjust=adjust)

# -----------------------------
# Indicator functions
# -----------------------------
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # SMA / EMA
    for n in [5, 10, 20, 60, 120]:
        df[f"MA{n}"] = close.rolling(n).mean()
        df[f"EMA{n}"] = close.ewm(span=n, adjust=False).mean()

    # Bollinger
    mid = df["MA20"]
    std = close.rolling(20).std()
    df["Upper"] = mid + 2 * std
    df["Lower"] = mid - 2 * std

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # StochRSI
    rsi_min = df["RSI"].rolling(14).min()
    rsi_max = df["RSI"].rolling(14).max()
    df["StochRSI"] = (df["RSI"] - rsi_min) / (rsi_max - rsi_min + 1e-9)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["HIST"] = df["DIF"] - df["DEA"]

    # KDJ (stochastic)
    low_n = low.rolling(9).min()
    high_n = high.rolling(9).max()
    rsv = (close - low_n) / (high_n - low_n + 1e-9) * 100
    df["K"] = rsv.ewm(com=2).mean()
    df["D"] = df["K"].ewm(com=2).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # ADX
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).sum() / (tr14 + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).sum() / (tr14 + 1e-9)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    df["ADX"] = dx.rolling(14).mean()
    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di

    # OBV
    direction = np.sign(close.diff()).fillna(0)
    df["OBV"] = (direction * vol).cumsum()

    # CCI
    tp = (high + low + close) / 3
    ma_tp = tp.rolling(20).mean()
    md = (tp - ma_tp).abs().rolling(20).mean()
    df["CCI"] = (tp - ma_tp) / (0.015 * md + 1e-9)

    # MFI
    raw_mf = tp * vol
    pos_mf = raw_mf.where(tp.diff() > 0, 0).rolling(14).sum()
    neg_mf = raw_mf.where(tp.diff() < 0, 0).rolling(14).sum()
    mfr = pos_mf / (neg_mf + 1e-9)
    df["MFI"] = 100 - (100 / (1 + mfr))

    # Ichimoku (9,26,52)
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    df["TENKAN"] = tenkan
    df["KIJUN"] = kijun
    df["SPAN_A"] = span_a
    df["SPAN_B"] = span_b

    # Parabolic SAR (simplified)
    af = 0.02
    max_af = 0.2
    sar = close.copy()
    trend = 1
    ep = low.iloc[0]
    sar.iloc[0] = low.iloc[0]
    for i in range(1, len(df)):
        prev_sar = sar.iloc[i-1]
        if trend == 1:
            sar.iloc[i] = prev_sar + af * (ep - prev_sar)
            if low.iloc[i] < sar.iloc[i]:
                trend = -1
                sar.iloc[i] = ep
                ep = high.iloc[i]
                af = 0.02
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + 0.02, max_af)
        else:
            sar.iloc[i] = prev_sar + af * (ep - prev_sar)
            if high.iloc[i] > sar.iloc[i]:
                trend = 1
                sar.iloc[i] = ep
                ep = low.iloc[i]
                af = 0.02
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + 0.02, max_af)
    df["SAR"] = sar

    # Volume ratio
    df["VOL_MA20"] = vol.rolling(20).mean()
    df["VOL_RATIO"] = vol / (df["VOL_MA20"] + 1e-9)

    return df

def detect_fractals(df: pd.DataFrame, k: int = 2):
    """Simplified Chanlun fractals (top/bottom)."""
    highs = df["high"]
    lows = df["low"]
    top = (highs.shift(k) < highs) & (highs.shift(-k) < highs)
    bot = (lows.shift(k) > lows) & (lows.shift(-k) > lows)
    df["FRACTAL_TOP"] = top
    df["FRACTAL_BOT"] = bot
    return df

def build_bi_segments(df: pd.DataFrame):
    """Very simplified 'bi' segments connecting alternating fractals."""
    pts = []
    for _, row in df.iterrows():
        if row.get("FRACTAL_TOP"):
            pts.append((row["date"], row["high"], "top"))
        if row.get("FRACTAL_BOT"):
            pts.append((row["date"], row["low"], "bot"))

    segs = []
    last = None
    for p in pts:
        if last is None:
            last = p
            continue
        if p[2] != last[2]:
            segs.append((last, p))
            last = p
        else:
            if p[2] == "top" and p[1] >= last[1]:
                last = p
            if p[2] == "bot" and p[1] <= last[1]:
                last = p
    return segs

def gann_lines(df: pd.DataFrame, pivot_idx: int = None):
    """Simple Gann 1x1, 1x2, 2x1 from pivot low."""
    if pivot_idx is None:
        pivot_idx = df["low"].idxmin()
    pivot_date = df.loc[pivot_idx, "date"]
    pivot_price = df.loc[pivot_idx, "low"]

    days_from_pivot = (df["date"] - pivot_date).dt.days
    step = (df["ATR14"].iloc[-1] or (pivot_price * 0.01))

    lines = {}
    for name, ratio in [("1x1", 1.0), ("1x2", 0.5), ("2x1", 2.0)]:
        y = pivot_price + days_from_pivot * step * ratio
        lines[name] = y
    return lines, pivot_date, pivot_price

def fib_levels(df: pd.DataFrame, lookback: int = 120):
    chunk = df.tail(lookback)
    hi = chunk["high"].max()
    lo = chunk["low"].min()
    diff = hi - lo
    levels = {
        "0.236": hi - diff * 0.236,
        "0.382": hi - diff * 0.382,
        "0.5": hi - diff * 0.5,
        "0.618": hi - diff * 0.618,
        "0.786": hi - diff * 0.786
    }
    return hi, lo, levels

def main_uptrend_state(df: pd.DataFrame):
    latest = df.iloc[-1]
    ma20_slope = df["MA20"].diff().tail(5).mean()
    cloud_top = max(latest["SPAN_A"], latest["SPAN_B"])
    cloud_bot = min(latest["SPAN_A"], latest["SPAN_B"])
    above_cloud = latest["close"] > cloud_top
    adx_strong = latest["ADX"] > 25
    ma_rise = ma20_slope > 0
    if above_cloud and adx_strong and ma_rise:
        return "✅ 主升浪/强趋势", "success"
    if latest["close"] > cloud_bot and ma_rise:
        return "🟡 趋势孕育中", "warning"
    return "❌ 震荡/下行", "error"

def make_signals(df: pd.DataFrame):
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    score = 0
    reasons = []

    # Trend / MA
    if latest["MA5"] > latest["MA20"]:
        score += 2; reasons.append("✅ MA5>MA20：短线多头")
    else:
        score -= 2; reasons.append("❌ MA5<MA20：短线弱势")

    if latest["close"] > latest["MA60"]:
        score += 1; reasons.append("✅ 站上MA60：中期偏强")
    else:
        score -= 1; reasons.append("❌ 跌破MA60：中期偏弱")

    # MACD
    if latest["DIF"] > latest["DEA"] and latest["HIST"] > prev["HIST"]:
        score += 1; reasons.append("✅ MACD金叉且柱子放大")
    elif latest["DIF"] < latest["DEA"]:
        score -= 1; reasons.append("❌ MACD死叉/走弱")

    # RSI / MFI
    if latest["RSI"] < 30:
        score += 2; reasons.append("📉 RSI<30：超卖反弹区")
    elif latest["RSI"] > 70:
        score -= 2; reasons.append("📈 RSI>70：超买风险区")

    if latest["MFI"] < 20:
        score += 1; reasons.append("💧 MFI<20：资金过度流出，反弹概率↑")
    elif latest["MFI"] > 80:
        score -= 1; reasons.append("💧 MFI>80：资金过热，注意回撤")

    # ADX
    if latest["ADX"] > 25 and latest["PLUS_DI"] > latest["MINUS_DI"]:
        score += 1; reasons.append("✅ ADX强趋势且多头占优")
    elif latest["ADX"] > 25:
        score -= 1; reasons.append("⚠️ ADX强趋势但空头占优")

    # Bollinger position
    if latest["close"] > latest["Upper"]:
        score -= 1; reasons.append("⚠️ 突破布林上轨：短线过热")
    elif latest["close"] < latest["Lower"]:
        score += 1; reasons.append("✅ 跌破布林下轨：情绪极端")

    # Volume
    if latest["VOL_RATIO"] >= 1.2:
        score += 1; reasons.append("✅ 放量（量比>1.2）")
    elif latest["VOL_RATIO"] <= 0.8:
        score -= 1; reasons.append("❌ 缩量（量比<0.8）")

    # SAR
    if latest["close"] > latest["SAR"]:
        score += 0.5; reasons.append("✅ SAR多头")
    else:
        score -= 0.5; reasons.append("❌ SAR空头")

    # Position suggestion
    if score >= 5:
        action, position, color = "🚀 强势买入", "70% - 100%", "success"
    elif score >= 3:
        action, position, color = "✅ 试探加仓", "30% - 50%", "success"
    elif score >= 0:
        action, position, color = "👀 观望/小仓", "0% - 20%", "warning"
    else:
        action, position, color = "🛑 减仓/清仓", "0% - 10%", "error"

    support = df["low"].tail(20).min()
    resistance = df["high"].tail(20).max()

    # Buy/Sell points (signals)
    buy_signal = (prev["MA5"] <= prev["MA20"]) and (latest["MA5"] > latest["MA20"])
    sell_signal = (prev["MA5"] >= prev["MA20"]) and (latest["MA5"] < latest["MA20"])

    return score, action, position, reasons, support, resistance, color, buy_signal, sell_signal

# -----------------------------
# Plotting
# -----------------------------
def plot_kline(df: pd.DataFrame, title: str,
               show_gann: bool = True,
               show_chanlun: bool = True,
               show_fib: bool = True):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.55, 0.15, 0.15, 0.15]
    )

    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="K线"
    ), row=1, col=1)

    for ma in ["MA5","MA10","MA20","MA60","MA120"]:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[ma], name=ma, line=dict(width=1)
        ), row=1, col=1)

    # Bollinger
    fig.add_trace(go.Scatter(x=df["date"], y=df["Upper"], name="BOLL上轨",
                             line=dict(dash="dash", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["Lower"], name="BOLL下轨",
                             line=dict(dash="dash", width=1)), row=1, col=1)

    # Ichimoku cloud
    fig.add_trace(go.Scatter(x=df["date"], y=df["SPAN_A"], name="云A", line=dict(width=0.7)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["SPAN_B"], name="云B", line=dict(width=0.7)),
                  row=1, col=1)

    # Chanlun fractals + bi
    if show_chanlun:
        tops = df[df["FRACTAL_TOP"]]
        bots = df[df["FRACTAL_BOT"]]
        fig.add_trace(go.Scatter(
            x=tops["date"], y=tops["high"], mode="markers",
            name="缠论顶分型", marker_symbol="triangle-down", marker_size=8
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=bots["date"], y=bots["low"], mode="markers",
            name="缠论底分型", marker_symbol="triangle-up", marker_size=8
        ), row=1, col=1)
        segs = build_bi_segments(df)
        for s, e in segs:
            fig.add_trace(go.Scatter(
                x=[s[0], e[0]], y=[s[1], e[1]],
                mode="lines", name="缠论笔(简)", line=dict(width=1.2)
            ), row=1, col=1)

    # Gann fan lines
    if show_gann:
        lines, _, _ = gann_lines(df)
        for name, y in lines.items():
            fig.add_trace(go.Scatter(
                x=df["date"], y=y, name=f"江恩{name}", line=dict(dash="dot", width=1)
            ), row=1, col=1)

    # Fibonacci retracements
    if show_fib:
        _, _, levels = fib_levels(df)
        for k, v in levels.items():
            fig.add_hline(y=v, line_dash="dash", annotation_text=f"Fib {k}",
                          row=1, col=1)

    # Volume
    colors = np.where(df["close"] >= df["open"], "red", "green")
    fig.add_trace(go.Bar(
        x=df["date"], y=df["volume"], name="成交量", marker_color=colors
    ), row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=df["date"], y=df["DIF"], name="DIF"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["DEA"], name="DEA"), row=3, col=1)
    fig.add_trace(go.Bar(x=df["date"], y=df["HIST"], name="MACD柱"), row=3, col=1)

    # RSI / KDJ
    fig.add_trace(go.Scatter(x=df["date"], y=df["RSI"], name="RSI"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["K"], name="K"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["D"], name="D"), row=4, col=1)

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=920,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.markdown("## 🎛️ 操盘控制台 Pro")

    # 1. 输入 Token (添加了错误处理，防止本地运行报错)
    default_token = ""
    try:
        # 尝试读取 secrets，如果文件不存在则跳过
        if "TUSHARE_TOKEN" in st.secrets:
            default_token = st.secrets["TUSHARE_TOKEN"]
    except Exception:
        pass
    
    tushare_token = st.text_input(
        "TuShare Token（可选，留空走 Baostock）",
        value=default_token,
        type="password"
    ).strip()

    # 2. 输入代码
    stock_code = st.text_input("股票代码(6位)", value="600519").strip()

    # 3. 自动匹配名称
    # 当 stock_code 改变时，重新运行 get_stock_name
    auto_name_fetched = get_stock_name(stock_code, tushare_token)
    
    # 如果没获取到，默认叫“未知股票”
    default_name = auto_name_fetched if auto_name_fetched else "未知股票"

    # 允许用户手动修改，但默认值是自动获取的
    stock_name = st.text_input("股票名称", value=default_name)

    window_days = st.radio(
        "分析窗口",
        [7, 15, 30, 60, 120, 180],
        index=5,
        horizontal=True
    )

    adjust = st.selectbox(
        "复权方式",
        ["qfq", "hfq", ""],
        index=0,
        format_func=lambda x: "前复权" if x == "qfq" else "后复权" if x == "hfq" else "不复权"
    )

    st.divider()
    st.markdown("### 📌 显示项")
    show_gann = st.checkbox("显示江恩线(简化)", value=True)
    show_chanlun = st.checkbox("显示缠论分型/笔(简化)", value=True)
    show_fib = st.checkbox("显示斐波那契回撤", value=True)

    st.divider()
    st.markdown("### 🔔 关键价位提醒")
    support_alert = st.number_input("回踩支撑价（提示补仓）", value=0.0, step=0.1)
    risk_alert = st.number_input("跌破风险价（提示减仓）", value=0.0, step=0.1)
    breakout_alert = st.number_input("突破价（提示加仓）", value=0.0, step=0.1)

    st.caption("⚠️ 数据源：TuShare / Baostock 自动切换")

# -----------------------------
# Main area
# -----------------------------
st.title(f"📈 {stock_name} ({stock_code}) 深度复盘系统 Pro")

with st.spinner("正在拉取数据..."):
    df = fetch_hist(stock_code, tushare_token, days=380, adjust=adjust)

if df is None or df.empty:
    st.error("未获取到数据：可能是代码不对 / 网络抽风 / 接口限流。")
    st.stop()

df = calc_indicators(df)
df = detect_fractals(df, k=2)

view_df = df.tail(window_days).copy()

latest = view_df.iloc[-1]
prev = view_df.iloc[-2] if len(view_df) > 1 else latest

last_close = float(latest["close"])
prev_close = float(prev["close"])
pct_change = (last_close - prev_close) / prev_close * 100 if prev_close else 0

# Key metrics panel
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("当前价格", f"{last_close:.2f}", f"{pct_change:.2f}%")
c2.metric("RSI(14)", f"{latest['RSI']:.1f}")
c3.metric("MA5", f"{latest['MA5']:.2f}")
c4.metric("MA20", f"{latest['MA20']:.2f}")
c5.metric("MACD柱", f"{latest['HIST']:.3f}")
c6.metric("ADX", f"{latest['ADX']:.1f}", help=">25 强趋势")
c7.metric("量比", f"{latest['VOL_RATIO']:.2f}x")

# Trend state
trend_text, trend_color = main_uptrend_state(view_df)
if trend_color == "success":
    st.success(f"主升浪识别：{trend_text}")
elif trend_color == "warning":
    st.warning(f"主升浪识别：{trend_text}")
else:
    st.error(f"主升浪识别：{trend_text}")

# Chart
plot_kline(view_df, f"{stock_name} | {window_days}日窗口",
           show_gann=show_gann, show_chanlun=show_chanlun, show_fib=show_fib)

# AI signals
score, action, position, reasons, support, resistance, color, buy_sig, sell_sig = make_signals(view_df)

st.subheader("🤖 AI 买卖点 & 仓位建议（多指标综合）")
if color == "success":
    st.success(f"**{action}** | 建议仓位：**{position}**")
elif color == "error":
    st.error(f"**{action}** | 建议仓位：**{position}**")
else:
    st.warning(f"**{action}** | 建议仓位：**{position}**")

# Concrete buy/sell & stop suggestions
atr = latest["ATR14"]
stop_loss = last_close - 2 * atr if pd.notna(atr) else support
take_profit = last_close + 3 * atr if pd.notna(atr) else resistance

scol1, scol2, scol3 = st.columns(3)
scol1.metric("短线止损参考(2ATR)", f"{stop_loss:.2f}")
scol2.metric("短线止盈参考(3ATR)", f"{take_profit:.2f}")
scol3.metric("缠论近端支撑", f"{support:.2f}")

if buy_sig:
    st.success("📌 **短线买点出现：MA5 上穿 MA20（黄金交叉）**")
if sell_sig:
    st.error("📌 **短线卖点出现：MA5 下穿 MA20（死亡交叉）**")

st.info(
    f"📌 近期支撑位：**{support:.2f}** |  "
    f"压力位：**{resistance:.2f}**"
)

with st.expander("展开查看评分逻辑 / 指标解释"):
    st.write(f"综合评分：**{score:.1f}**")
    for r in reasons:
        st.write(r)
    st.markdown("""
**评分说明：**
- 采用国际通用指标（MA/EMA、MACD、RSI、BOLL、ADX、MFI、SAR、KDJ、Ichimoku）综合打分；
- 缠论、江恩、斐波那契为“形态/位置”辅助，不直接决定仓位；
- 分值仅用于提示概率优势区，不保证收益。
""")

# Price alerts
if support_alert > 0 and last_close <= support_alert:
    st.warning(f"🟡 回踩支撑：股价 ≤ {support_alert:.2f}，可考虑分批补仓")
if risk_alert > 0 and last_close <= risk_alert:
    st.error(f"🔴 跌破风险：股价 ≤ {risk_alert:.2f}，注意控制回撤/减仓")
if breakout_alert > 0 and last_close >= breakout_alert:
    st.success(f"🟢 突破确认：股价 ≥ {breakout_alert:.2f}，趋势确认可加仓")
