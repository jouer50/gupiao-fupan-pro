import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import string
import os
import bcrypt  # 记得 requirements.txt 必须有 bcrypt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 0. 全局配置 (必须在最第一行)
# ==========================================
st.set_page_config(
    page_title="A股深度复盘系统 Pro (含登录)",
    layout="wide",
    page_icon="📈"
)

# Optional deps
try:
    import tushare as ts
except Exception:
    ts = None

try:
    import baostock as bs
except Exception:
    bs = None

# ==========================================
# 🔐 第一部分：登录/注册/验证码 核心逻辑
# ==========================================

USER_DB_FILE = "users.csv"

# 初始化用户数据库文件
if not os.path.exists(USER_DB_FILE):
    df_init = pd.DataFrame(columns=["username", "password_hash"])
    df_init.to_csv(USER_DB_FILE, index=False)

def load_users():
    return pd.read_csv(USER_DB_FILE)

def save_user(username, password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    df = load_users()
    new_user = pd.DataFrame({"username": [username], "password_hash": [hashed]})
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USER_DB_FILE, index=False)

def verify_login(username, password):
    df = load_users()
    user_row = df[df["username"] == username]
    if user_row.empty:
        return False
    stored_hash = user_row.iloc[0]["password_hash"]
    try:
        return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
    except:
        return False

def generate_captcha():
    chars = string.ascii_uppercase + string.digits
    code = ''.join(random.choice(chars) for _ in range(4))
    return code

def login_page():
    st.markdown("<h1 style='text-align: center;'>🔐 A股深度复盘系统 Pro</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if "captcha_code" not in st.session_state:
            st.session_state["captcha_code"] = generate_captcha()

        tab1, tab2 = st.tabs(["🔑 用户登录", "📝 注册新账号"])

        with tab1:
            login_user = st.text_input("用户名", key="l_user")
            login_pass = st.text_input("密码", type="password", key="l_pass")
            
            c1, c2 = st.columns([2, 1])
            with c1:
                captcha_input = st.text_input("验证码", placeholder="不区分大小写")
            with c2:
                st.markdown(f"## `{st.session_state['captcha_code']}`")
                if st.button("刷新", key="refresh_cap"):
                    st.session_state["captcha_code"] = generate_captcha()
                    st.rerun()

            if st.button("登录", type="primary", use_container_width=True):
                if captcha_input.upper() != st.session_state["captcha_code"]:
                    st.error("❌ 验证码错误")
                    st.session_state["captcha_code"] = generate_captcha()
                elif not verify_login(login_user, login_pass):
                    st.error("❌ 用户名或密码错误")
                    st.session_state["captcha_code"] = generate_captcha()
                else:
                    st.session_state["logged_in"] = True
                    st.session_state["current_user"] = login_user
                    st.success("登录成功！")
                    time.sleep(0.5)
                    st.rerun()

        with tab2:
            new_user = st.text_input("新用户名", key="r_user")
            new_pass = st.text_input("设置密码", type="password", key="r_pass")
            new_pass2 = st.text_input("确认密码", type="password", key="r_pass2")
            
            if st.button("注册账号", use_container_width=True):
                df = load_users()
                if new_user in df["username"].values:
                    st.warning("⚠️ 用户名已存在")
                elif len(new_pass) < 4:
                    st.warning("⚠️ 密码至少4位")
                elif new_pass != new_pass2:
                    st.error("❌ 两次密码不一致")
                else:
                    save_user(new_user, new_pass)
                    st.success("✅ 注册成功！请去登录页面登录。")

# ==========================================
# 📈 第二部分：完整的核心功能函数 (数据、指标、绘图)
# ==========================================

def _to_ts_code(symbol: str) -> str:
    symbol = symbol.strip()
    if symbol.endswith(".SH") or symbol.endswith(".SZ"): return symbol
    if symbol.isdigit(): return f"{symbol}.SH" if symbol.startswith("6") else f"{symbol}.SZ"
    return symbol

def _to_bs_code(symbol: str) -> str:
    symbol = symbol.strip()
    if symbol.startswith("sh.") or symbol.startswith("sz."): return symbol
    if symbol.endswith(".SH"): return f"sh.{symbol[:6]}"
    if symbol.endswith(".SZ"): return f"sz.{symbol[:6]}"
    if symbol.isdigit(): return f"sh.{symbol}" if symbol.startswith("6") else f"sz.{symbol}"
    return symbol

@st.cache_data(ttl=60 * 60 * 24)
def get_stock_name(symbol: str, token: str = "") -> str:
    name = ""
    if token and ts is not None:
        try:
            ts_code = _to_ts_code(symbol)
            pro = ts.pro_api(token)
            df = pro.stock_basic(ts_code=ts_code, fields='name')
            if not df.empty: return df.iloc[0]['name']
        except: pass
    if bs is not None:
        try:
            bs_code = _to_bs_code(symbol)
            lg = bs.login()
            if lg.error_code == '0':
                rs = bs.query_stock_basic(code=bs_code)
                if rs.error_code == '0':
                    row = rs.get_row_data()
                    if row and len(row) > 1: name = row[1]
            bs.logout()
        except: pass
    return name

@st.cache_data(ttl=60 * 15, show_spinner=False)
def fetch_hist_tushare(symbol: str, token: str, days: int = 180, adjust: str = "qfq") -> pd.DataFrame:
    if ts is None or not token: return pd.DataFrame()
    pro = ts.pro_api(token)
    end = pd.Timestamp.today().strftime("%Y%m%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=days * 3)).strftime("%Y%m%d")
    ts_code = _to_ts_code(symbol)
    try:
        df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
        if df is None or df.empty: return pd.DataFrame()
        if adjust in ("qfq", "hfq"):
            af = pro.adj_factor(ts_code=ts_code, start_date=start, end_date=end)
            if af is not None and not af.empty:
                af = af.rename(columns={"trade_date": "date", "adj_factor": "factor"})
                df = df.merge(af[["date", "factor"]], on="date", how="left")
                df["factor"] = df["factor"].ffill().bfill()
                adj_col = df["factor"] / df["factor"].iloc[-1] if adjust == "qfq" else df["factor"] / df["factor"].iloc[0]
                for col in ["open", "high", "low", "close"]: df[col] = df[col] * adj_col
        df = df.rename(columns={"trade_date": "date", "vol": "volume", "pct_chg": "pct_change"})
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "volume", "pct_change"]:
             if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("date").reset_index(drop=True).tail(days)
    except: return pd.DataFrame()

@st.cache_data(ttl=60 * 15, show_spinner=False)
def fetch_hist_baostock(symbol: str, days: int = 180, adjust: str = "qfq") -> pd.DataFrame:
    if bs is None: return pd.DataFrame()
    lg = bs.login()
    if lg.error_code != "0": return pd.DataFrame()
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=days * 3)
    code = _to_bs_code(symbol)
    adj_flag = "2" if adjust == "qfq" else "1" if adjust == "hfq" else "3"
    rs = bs.query_history_k_data_plus(code, "date,open,high,low,close,volume,amount,pctChg",
        start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d"), frequency="d", adjustflag=adj_flag)
    data = []
    while rs.error_code == "0" and rs.next(): data.append(rs.get_row_data())
    bs.logout()
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data, columns=rs.fields).rename(columns={"pctChg": "pct_change"})
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open","high","low","close","volume","amount","pct_change"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").reset_index(drop=True).tail(days)

def fetch_hist(symbol: str, token: str, days: int = 180, adjust: str = "qfq") -> pd.DataFrame:
    if token:
        df = fetch_hist_tushare(symbol, token, days, adjust)
        if df is not None and not df.empty: return df
    return fetch_hist_baostock(symbol, days, adjust)

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    for n in [5, 10, 20, 60, 120]:
        df[f"MA{n}"] = close.rolling(n).mean()
        df[f"EMA{n}"] = close.ewm(span=n, adjust=False).mean()
    mid, std = df["MA20"], close.rolling(20).std()
    df["Upper"], df["Lower"] = mid + 2*std, mid - 2*std
    
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    
    rsi_min, rsi_max = df["RSI"].rolling(14).min(), df["RSI"].rolling(14).max()
    df["StochRSI"] = (df["RSI"] - rsi_min) / (rsi_max - rsi_min + 1e-9)

    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["HIST"] = df["DIF"] - df["DEA"]
    
    low_n, high_n = low.rolling(9).min(), high.rolling(9).max()
    rsv = (close - low_n) / (high_n - low_n + 1e-9) * 100
    df["K"] = rsv.ewm(com=2).mean()
    df["D"] = df["K"].ewm(com=2).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).sum() / (tr14 + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).sum() / (tr14 + 1e-9)
    df["ADX"] = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9) * 100).rolling(14).mean()
    df["PLUS_DI"], df["MINUS_DI"] = plus_di, minus_di
    
    df["OBV"] = (np.sign(close.diff()).fillna(0) * vol).cumsum()
    
    tp = (high + low + close) / 3
    df["CCI"] = (tp - tp.rolling(20).mean()) / (0.015 * (tp - tp.rolling(20).mean()).abs().rolling(20).mean() + 1e-9)
    
    raw_mf = tp * vol
    pos_mf = raw_mf.where(tp.diff() > 0, 0).rolling(14).sum()
    neg_mf = raw_mf.where(tp.diff() < 0, 0).rolling(14).sum()
    df["MFI"] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-9)))

    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    df["SPAN_A"] = ((tenkan + kijun) / 2).shift(26)
    df["SPAN_B"] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    af, max_af, trend, ep, sar = 0.02, 0.2, 1, low.iloc[0], close.copy()
    sar.iloc[0] = low.iloc[0]
    for i in range(1, len(df)):
        prev_sar = sar.iloc[i-1]
        if trend == 1:
            sar.iloc[i] = prev_sar + af * (ep - prev_sar)
            if low.iloc[i] < sar.iloc[i]:
                trend, sar.iloc[i], ep, af = -1, ep, high.iloc[i], 0.02
            elif high.iloc[i] > ep:
                ep, af = high.iloc[i], min(af + 0.02, max_af)
        else:
            sar.iloc[i] = prev_sar + af * (ep - prev_sar)
            if high.iloc[i] > sar.iloc[i]:
                trend, sar.iloc[i], ep, af = 1, ep, low.iloc[i], 0.02
            elif low.iloc[i] < ep:
                ep, af = low.iloc[i], min(af + 0.02, max_af)
    df["SAR"] = sar
    df["VOL_RATIO"] = vol / (vol.rolling(20).mean() + 1e-9)
    return df

def detect_fractals(df: pd.DataFrame, k: int = 2):
    df["FRACTAL_TOP"] = (df["high"].shift(k) < df["high"]) & (df["high"].shift(-k) < df["high"])
    df["FRACTAL_BOT"] = (df["low"].shift(k) > df["low"]) & (df["low"].shift(-k) > df["low"])
    return df

def build_bi_segments(df: pd.DataFrame):
    pts = []
    for _, row in df.iterrows():
        if row.get("FRACTAL_TOP"): pts.append((row["date"], row["high"], "top"))
        if row.get("FRACTAL_BOT"): pts.append((row["date"], row["low"], "bot"))
    segs, last = [], None
    for p in pts:
        if last is None: last = p; continue
        if p[2] != last[2]: segs.append((last, p)); last = p
        else:
            if (p[2] == "top" and p[1] >= last[1]) or (p[2] == "bot" and p[1] <= last[1]): last = p
    return segs

def gann_lines(df: pd.DataFrame):
    pivot_idx = df["low"].idxmin()
    pivot_date, pivot_price = df.loc[pivot_idx, "date"], df.loc[pivot_idx, "low"]
    days = (df["date"] - pivot_date).dt.days
    step = df["ATR14"].iloc[-1] or pivot_price * 0.01
    return {name: pivot_price + days * step * ratio for name, ratio in [("1x1", 1), ("1x2", 0.5), ("2x1", 2)]}

def fib_levels(df: pd.DataFrame):
    chunk = df.tail(120)
    hi, lo = chunk["high"].max(), chunk["low"].min()
    diff = hi - lo
    return {k: hi - diff * v for k, v in {"0.236":0.236, "0.382":0.382, "0.5":0.5, "0.618":0.618}.items()}

def main_uptrend_state(df: pd.DataFrame):
    latest = df.iloc[-1]
    top, bot = max(latest["SPAN_A"], latest["SPAN_B"]), min(latest["SPAN_A"], latest["SPAN_B"])
    if latest["close"] > top and latest["ADX"] > 25 and df["MA20"].diff().tail(5).mean() > 0:
        return "✅ 主升浪/强趋势", "success"
    if latest["close"] > bot and df["MA20"].diff().tail(5).mean() > 0:
        return "🟡 趋势孕育中", "warning"
    return "❌ 震荡/下行", "error"

def make_signals(df: pd.DataFrame):
    latest, prev = df.iloc[-1], df.iloc[-2]
    score, reasons = 0, []
    if latest["MA5"] > latest["MA20"]: score += 2; reasons.append("✅ MA5>MA20：短线多头")
    else: score -= 2; reasons.append("❌ MA5<MA20：短线弱势")
    if latest["close"] > latest["MA60"]: score += 1; reasons.append("✅ 站上MA60：中期偏强")
    else: score -= 1; reasons.append("❌ 跌破MA60：中期偏弱")
    if latest["DIF"] > latest["DEA"] and latest["HIST"] > prev["HIST"]: score += 1; reasons.append("✅ MACD金叉增强")
    elif latest["DIF"] < latest["DEA"]: score -= 1; reasons.append("❌ MACD死叉")
    if latest["RSI"] < 30: score += 2; reasons.append("📉 RSI超卖")
    elif latest["RSI"] > 70: score -= 2; reasons.append("📈 RSI超买")
    if latest["MFI"] < 20: score += 1; reasons.append("💧 MFI资金流出极值")
    if latest["VOL_RATIO"] >= 1.2: score += 1; reasons.append("✅ 放量")
    
    if score >= 5: action, pos, color = "🚀 强势买入", "70%+", "success"
    elif score >= 3: action, pos, color = "✅ 试探加仓", "30-50%", "success"
    elif score >= 0: action, pos, color = "👀 观望", "20%↓", "warning"
    else: action, pos, color = "🛑 减仓/空仓", "0-10%", "error"
    
    return score, action, pos, reasons, color, \
           (prev["MA5"] <= prev["MA20"] and latest["MA5"] > latest["MA20"]), \
           (prev["MA5"] >= prev["MA20"] and latest["MA5"] < latest["MA20"])

def plot_kline(df: pd.DataFrame, title: str, show_gann: bool, show_chanlun: bool, show_fib: bool):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.55, 0.15, 0.15, 0.15])
    fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="K线"), 1, 1)
    for ma in ["MA5","MA20","MA60"]: fig.add_trace(go.Scatter(x=df["date"], y=df[ma], name=ma, line=dict(width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["Upper"], name="BOLL上", line=dict(dash="dash", width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["Lower"], name="BOLL下", line=dict(dash="dash", width=1)), 1, 1)
    
    if show_chanlun:
        tops, bots = df[df["FRACTAL_TOP"]], df[df["FRACTAL_BOT"]]
        fig.add_trace(go.Scatter(x=tops["date"], y=tops["high"], mode="markers", name="顶分型", marker_symbol="triangle-down", marker_size=8), 1, 1)
        fig.add_trace(go.Scatter(x=bots["date"], y=bots["low"], mode="markers", name="底分型", marker_symbol="triangle-up", marker_size=8), 1, 1)
        for s, e in build_bi_segments(df): fig.add_trace(go.Scatter(x=[s[0], e[0]], y=[s[1], e[1]], mode="lines", name="笔", line=dict(width=1.2, color='gray')), 1, 1)
    
    if show_gann:
        for n, y in gann_lines(df).items(): fig.add_trace(go.Scatter(x=df["date"], y=y, name=f"江恩{n}", line=dict(dash="dot", width=1)), 1, 1)
    
    if show_fib:
        for k, v in fib_levels(df).items(): fig.add_hline(y=v, line_dash="dash", annotation_text=f"Fib {k}", row=1, col=1)
        
    colors = np.where(df["close"] >= df["open"], "red", "green")
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="量", marker_color=colors), 2, 1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["DIF"], name="DIF"), 3, 1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["DEA"], name="DEA"), 3, 1)
    fig.add_trace(go.Bar(x=df["date"], y=df["HIST"], name="MACD"), 3, 1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["RSI"], name="RSI"), 4, 1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["K"], name="K"), 4, 1)
    
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=900, margin=dict(t=60, b=30))
    st.plotly_chart(fig, use_container_width=True)

def main_stock_system():
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state['current_user']}")
        if st.button("🚪 退出登录", type="primary"):
            st.session_state["logged_in"] = False
            st.rerun()
        st.divider()
        st.markdown("## 🎛️ 操盘控制台")
        
        default_token = ""
        try:
            if "TUSHARE_TOKEN" in st.secrets: default_token = st.secrets["TUSHARE_TOKEN"]
        except: pass
        tushare_token = st.text_input("Token", value=default_token, type="password")
        
        stock_code = st.text_input("代码", value="600519").strip()
        auto_name = get_stock_name(stock_code, tushare_token)
        stock_name = st.text_input("名称", value=auto_name or "未知")
        
        window_days = st.radio("窗口", [30, 60, 120, 180, 250], index=3, horizontal=True)
        adjust = st.selectbox("复权", ["qfq", "hfq", ""], index=0)
        st.divider()
        show_gann = st.checkbox("江恩线", True)
        show_chanlun = st.checkbox("缠论分型", True)
        show_fib = st.checkbox("斐波那契", True)
        
    st.title(f"📈 {stock_name} ({stock_code}) 深度复盘系统 Pro")
    
    with st.spinner("🚀 AI正在拉取数据并计算指标..."):
        df = fetch_hist(stock_code, tushare_token, 380, adjust)
    
    if df.empty:
        st.error("❌ 数据拉取失败，请检查代码或Token")
        st.stop()
        
    df = calc_indicators(df)
    df = detect_fractals(df)
    view_df = df.tail(window_days).copy()
    latest = view_df.iloc[-1]
    
    # 核心指标栏
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("当前价", f"{latest['close']:.2f}", f"{latest['pct_change']:.2f}%")
    c2.metric("RSI", f"{latest['RSI']:.1f}")
    c3.metric("MACD柱", f"{latest['HIST']:.3f}")
    c4.metric("ADX", f"{latest['ADX']:.1f}")
    c5.metric("量比", f"{latest['VOL_RATIO']:.2f}")

    # 趋势判断
    t_txt, t_col = main_uptrend_state(view_df)
    if t_col=="success": st.success(f"趋势识别：{t_txt}")
    elif t_col=="warning": st.warning(f"趋势识别：{t_txt}")
    else: st.error(f"趋势识别：{t_txt}")
    
    # 画图
    plot_kline(view_df, f"{stock_name} 行情分析", show_gann, show_chanlun, show_fib)
    
    # 信号生成
    score, action, pos, reasons, color, buy_sig, sell_sig = make_signals(view_df)
    
    st.subheader("🤖 AI 决策建议")
    if color=="success": st.success(f"**{action}** | 仓位：{pos} | 评分：{score}")
    elif color=="warning": st.warning(f"**{action}** | 仓位：{pos} | 评分：{score}")
    else: st.error(f"**{action}** | 仓位：{pos} | 评分：{score}")
    
    if buy_sig: st.success("🔥 触发短线金叉买点！")
    if sell_sig: st.error("❄️ 触发短线死叉卖点！")
    
    with st.expander("🔍 查看详细逻辑"):
        for r in reasons: st.write(r)

# ==========================================
# 🚀 主程序入口
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_page()
else:
    main_stock_system()
