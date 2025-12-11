import streamlit as st  # 修正了这里的大小写
import pandas as pd
import numpy as np
import time
import random
import string
import os
import bcrypt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 0. 全局配置 & 界面清理
# ==========================================
st.set_page_config(
    page_title="A股深度复盘系统 Pro",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# 隐藏菜单 CSS
hide_css = """
<style>
    header {visibility: hidden !important;}
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    .block-container {padding-top: 1rem !important;}
</style>
"""
st.markdown(hide_css, unsafe_allow_html=True)

# 👑 管理员账号 (硬编码，最高优先级)
ADMIN_USERNAME = "ZCX001"
ADMIN_PASS = "123456"

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
# 🔐 用户数据库 & 验证逻辑
# ==========================================

USER_DB_FILE = "users.csv"

def init_db():
    if not os.path.exists(USER_DB_FILE):
        df = pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"])
        df.to_csv(USER_DB_FILE, index=False)
    else:
        # 自动修复缺失列
        try:
            df = pd.read_csv(USER_DB_FILE)
            changed = False
            if "quota" not in df.columns: df["quota"] = 20; changed = True
            if "watchlist" not in df.columns: df["watchlist"] = ""; changed = True
            if changed: df.to_csv(USER_DB_FILE, index=False)
        except:
            pass

init_db()

def load_users():
    return pd.read_csv(USER_DB_FILE, dtype={"watchlist": str, "quota": int})

def save_users_df(df):
    df.to_csv(USER_DB_FILE, index=False)

def save_user(username, password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    df = load_users()
    new_user = pd.DataFrame({
        "username": [username], 
        "password_hash": [hashed], 
        "watchlist": [""],
        "quota": [20]
    })
    df = pd.concat([df, new_user], ignore_index=True)
    save_users_df(df)

def verify_login(username, password):
    # 👑 超级通道：如果是管理员，直接比对硬编码密码，不查数据库
    # 这能解决“账户已存在但密码不对”的死循环
    if username == ADMIN_USERNAME:
        if password == ADMIN_PASS:
            # 登录成功后，顺便检查一下数据库里有没有这个号，没有就补上，有就更新权限
            df = load_users()
            if df[df["username"] == ADMIN_USERNAME].empty:
                # 补录管理员到数据库
                salt = bcrypt.gensalt()
                hashed = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
                new_admin = pd.DataFrame({
                    "username": [ADMIN_USERNAME],
                    "password_hash": [hashed],
                    "watchlist": [""],
                    "quota": [999999]
                })
                df = pd.concat([df, new_admin], ignore_index=True)
                save_users_df(df)
            return True, "Login Success"
        else:
            return False, "❌ 管理员密码错误"

    # 普通用户走数据库验证
    df = load_users()
    user_row = df[df["username"] == username]
    if user_row.empty: return False, "❌ 用户不存在"
    
    stored_hash = user_row.iloc[0]["password_hash"]
    try:
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            return True, "Success"
        else:
            return False, "❌ 密码错误"
    except:
        return False, "❌ 校验失败"

def delete_user(target_username):
    if target_username == ADMIN_USERNAME: return
    df = load_users()
    df = df[df["username"] != target_username]
    save_users_df(df)

def update_user_quota(target_username, new_quota):
    df = load_users()
    idx = df[df["username"] == target_username].index
    if len(idx) > 0:
        df.loc[idx[0], "quota"] = int(new_quota)
        save_users_df(df)
        return True
    return False

def get_current_quota(username):
    if username == ADMIN_USERNAME: return 999999
    df = load_users()
    user = df[df["username"] == username]
    if user.empty: return 0
    return int(user.iloc[0]["quota"])

def consume_quota(username):
    if username == ADMIN_USERNAME: return True
    df = load_users()
    idx = df[df["username"] == username].index
    if len(idx) > 0:
        current_q = int(df.loc[idx[0], "quota"])
        if current_q > 0:
            df.loc[idx[0], "quota"] = current_q - 1
            save_users_df(df)
            return True
    return False

# --- 自选股 ---
def get_user_watchlist(username):
    df = load_users()
    user_row = df[df["username"] == username]
    if user_row.empty: return []
    w_str = str(user_row.iloc[0]["watchlist"])
    if pd.isna(w_str) or w_str == "nan" or w_str.strip() == "": return []
    return w_str.split(",")

def toggle_watchlist(username, stock_code):
    df = load_users()
    idx = df[df["username"] == username].index
    if len(idx) == 0: return False
    current_w = str(df.loc[idx[0], "watchlist"])
    if pd.isna(current_w) or current_w == "nan": current_w = ""
    codes = [c for c in current_w.split(",") if c.strip()]
    if stock_code in codes: codes.remove(stock_code); action = "remove"
    else: codes.append(stock_code); action = "add"
    new_w = ",".join(codes)
    df.loc[idx[0], "watchlist"] = new_w
    save_users_df(df)
    return action

# --- 验证码 ---
def generate_captcha():
    chars = string.ascii_uppercase + string.digits
    code = ''.join(random.choice(chars) for _ in range(4))
    return code

def login_page():
    st.markdown("<br><h1 style='text-align: center;'>🔐 A股深度复盘系统 Pro</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if "captcha_code" not in st.session_state: st.session_state["captcha_code"] = generate_captcha()
        
        tab1, tab2 = st.tabs(["🔑 登录", "📝 注册"])
        with tab1:
            st.info(f"管理员账号: **{ADMIN_USERNAME}** / **{ADMIN_PASS}**")
            login_user = st.text_input("用户名", key="l_user")
            login_pass = st.text_input("密码", type="password", key="l_pass")
            c1, c2 = st.columns([2, 1])
            with c1: captcha_input = st.text_input("验证码", placeholder="不区分大小写")
            with c2:
                st.markdown(f"### `{st.session_state['captcha_code']}`")
                if st.button("🔄"):
                    st.session_state["captcha_code"] = generate_captcha(); st.rerun()
            
            if st.button("🚀 登录", type="primary", use_container_width=True):
                if captcha_input.upper() != st.session_state["captcha_code"]:
                    st.error("验证码错误")
                    st.session_state["captcha_code"] = generate_captcha()
                else:
                    is_valid, msg = verify_login(login_user, login_pass)
                    if not is_valid:
                        st.error(msg)
                        st.session_state["captcha_code"] = generate_captcha()
                    else:
                        st.session_state["logged_in"] = True
                        st.session_state["current_user"] = login_user
                        st.success("登录成功")
                        time.sleep(0.5); st.rerun()
        with tab2:
            st.caption("新用户注册送 20 积分")
            new_user = st.text_input("新用户名", key="r_user")
            new_pass = st.text_input("设置密码", type="password", key="r_pass")
            new_pass2 = st.text_input("确认密码", type="password", key="r_pass2")
            if st.button("注册", use_container_width=True):
                df = load_users()
                if new_user in df["username"].values: st.warning("用户名已存在")
                elif len(new_pass) < 4: st.warning("密码太短")
                elif new_pass != new_pass2: st.error("两次密码不一致")
                else:
                    save_user(new_user, new_pass)
                    st.success("注册成功！请登录。")

# ==========================================
# 📈 股票核心逻辑
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
            pro = ts.pro_api(token)
            df = pro.stock_basic(ts_code=_to_ts_code(symbol), fields='name')
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

@st.cache_data(ttl=60 * 60 * 12)
def fetch_fundamentals(symbol: str, token: str):
    data = {"pe": "N/A", "pb": "N/A", "total_mv": "N/A", "float_mv": "N/A", "roe": "N/A"}
    if token and ts is not None:
        try:
            pro = ts.pro_api(token)
            ts_code = _to_ts_code(symbol)
            df = pro.daily_basic(ts_code=ts_code, fields='pe_ttm,pb,total_mv,circ_mv')
            if not df.empty:
                row = df.iloc[-1]
                data["pe"] = f"{row['pe_ttm']:.2f}" if row['pe_ttm'] else "N/A"
                data["pb"] = f"{row['pb']:.2f}" if row['pb'] else "N/A"
                data["total_mv"] = f"{row['total_mv']/10000:.2f}亿" if row['total_mv'] else "N/A"
                data["float_mv"] = f"{row['circ_mv']/10000:.2f}亿" if row['circ_mv'] else "N/A"
            df_fin = pro.fina_indicator(ts_code=ts_code, fields='roe,q_dt')
            if not df_fin.empty: data["roe"] = f"{df_fin.iloc[0]['roe']:.2f}%"
        except: pass
    return data

@st.cache_data(ttl=60 * 15, show_spinner=False)
def fetch_hist(symbol: str, token: str, days: int = 180, adjust: str = "qfq") -> pd.DataFrame:
    if token and ts is not None:
        try:
            pro = ts.pro_api(token)
            end = pd.Timestamp.today().strftime("%Y%m%d")
            start = (pd.Timestamp.today() - pd.Timedelta(days=days * 3)).strftime("%Y%m%d")
            ts_code = _to_ts_code(symbol)
            df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
            if df is not None and not df.empty:
                if adjust in ("qfq", "hfq"):
                    af = pro.adj_factor(ts_code=ts_code, start_date=start, end_date=end)
                    if af is not None and not af.empty:
                        af = af.rename(columns={"trade_date": "date", "adj_factor": "factor"})
                        df = df.merge(af[["date", "factor"]], on="date", how="left")
                        df["factor"] = df["factor"].ffill().bfill()
                        if adjust == "qfq": adj = df["factor"] / df["factor"].iloc[-1]
                        else: adj = df["factor"] / df["factor"].iloc[0]
                        for col in ["open", "high", "low", "close"]: df[col] = df[col] * adj
                df = df.rename(columns={"trade_date": "date", "vol": "volume", "pct_chg": "pct_change"})
                df["date"] = pd.to_datetime(df["date"])
                for col in ["open", "high", "low", "close", "volume", "pct_change"]:
                    if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
                return df.sort_values("date").reset_index(drop=True).tail(days)
        except: pass
    if bs is None: return pd.DataFrame()
    lg = bs.login()
    if lg.error_code != "0": return pd.DataFrame()
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=days * 3)
    code = _to_bs_code(symbol)
    adj = "2" if adjust == "qfq" else "1" if adjust == "hfq" else "3"
    rs = bs.query_history_k_data_plus(code, "date,open,high,low,close,volume,amount,pctChg",
        start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d"), frequency="d", adjustflag=adj)
    data = []
    while rs.error_code == "0" and rs.next(): data.append(rs.get_row_data())
    bs.logout()
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data, columns=rs.fields).rename(columns={"pctChg": "pct_change"})
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open","high","low","close","volume","amount","pct_change"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").reset_index(drop=True).tail(days)

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    for n in [5, 10, 20, 60, 120]: df[f"MA{n}"] = close.rolling(n).mean()
    mid, std = df["MA20"], close.rolling(20).std()
    df["Upper"], df["Lower"] = mid + 2*std, mid - 2*std
    delta = close.diff()
    gain, loss = delta.clip(lower=0).rolling(14).mean(), (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    ema12, ema26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
    df["DIF"], df["DEA"] = ema12 - ema26, (ema12 - ema26).ewm(span=9).mean()
    df["HIST"] = df["DIF"] - df["DEA"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    up, down = high.diff(), -low.diff()
    p_dm = np.where((up > down) & (up > 0), up, 0.0)
    m_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr14 = tr.rolling(14).sum()
    p_di, m_di = 100 * pd.Series(p_dm, index=df.index).rolling(14).sum() / (tr14+1e-9), 100 * pd.Series(m_dm, index=df.index).rolling(14).sum() / (tr14+1e-9)
    df["ADX"] = (abs(p_di - m_di) / (p_di + m_di + 1e-9) * 100).rolling(14).mean()
    df["VOL_RATIO"] = vol / (vol.rolling(20).mean() + 1e-9)
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    df["SPAN_A"] = ((tenkan + kijun) / 2).shift(26)
    df["SPAN_B"] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
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
    p_idx = df["low"].idxmin()
    p_date, p_price = df.loc[p_idx, "date"], df.loc[p_idx, "low"]
    days = (df["date"] - p_date).dt.days
    step = df["ATR14"].iloc[-1] or p_price * 0.01
    return {n: p_price + days * step * r for n, r in [("1x1", 1), ("1x2", 0.5), ("2x1", 2)]}

def fib_levels(df: pd.DataFrame):
    chunk = df.tail(120)
    hi, lo = chunk["high"].max(), chunk["low"].min()
    diff = hi - lo
    return {k: hi - diff * v for k, v in {"0.236":0.236, "0.382":0.382, "0.5":0.5, "0.618":0.618}.items()}

def main_uptrend_state(df: pd.DataFrame):
    latest = df.iloc[-1]
    top, bot = max(latest["SPAN_A"], latest["SPAN_B"]), min(latest["SPAN_A"], latest["SPAN_B"])
    ma_rise = df["MA20"].diff().tail(5).mean() > 0
    if latest["close"] > top and latest["ADX"] > 25 and ma_rise: return "🚀 强势主升浪", "success"
    if latest["close"] > top: return "📈 上升趋势中", "success"
    if latest["close"] > bot and ma_rise: return "🟡 震荡/趋势孕育", "warning"
    return "❌ 下行/调整趋势", "error"

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
    if latest["VOL_RATIO"] >= 1.2: score += 1; reasons.append("✅ 放量")
    
    if score >= 5: action, pos, color = "🚀 强势买入", "70%+", "success"
    elif score >= 3: action, pos, color = "✅ 试探加仓", "30-50%", "success"
    elif score >= 0: action, pos, color = "👀 观望", "20%↓", "warning"
    else: action, pos, color = "🛑 减仓/空仓", "0-10%", "error"
    
    support, resistance = df["low"].tail(20).min(), df["high"].tail(20).max()
    buy_sig = (prev["MA5"] <= prev["MA20"] and latest["MA5"] > latest["MA20"])
    sell_sig = (prev["MA5"] >= prev["MA20"] and latest["MA5"] < latest["MA20"])
    return score, action, pos, reasons, color, buy_sig, sell_sig, support, resistance

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

# ==========================================
# 🚀 主程序入口
# ==========================================
def main_stock_system():
    if "stock_code" not in st.session_state: st.session_state["stock_code"] = "600519"
    if "data_loaded" not in st.session_state: st.session_state["data_loaded"] = False
    
    with st.sidebar:
        user = st.session_state['current_user']
        
        # 1. 顶部醒目的积分显示
        quota = get_current_quota(user)
        if user == ADMIN_USERNAME:
            st.metric("👑 管理员", f"{user}", delta="无限积分", delta_color="normal")
        else:
            st.metric("💰 剩余积分", f"{quota} 次", help="每次刷新或查询消耗 1 积分")
            
        if st.button("🚪 退出登录"):
            st.session_state["logged_in"] = False; st.rerun()

        # 2. 管理员后台 (Strict Check)
        if user == ADMIN_USERNAME:
            with st.expander("👮‍♂️ 积分管理后台", expanded=True):
                all_users = load_users()
                st.dataframe(all_users[["username", "quota"]], use_container_width=True)
                
                user_list = all_users["username"].tolist()
                if ADMIN_USERNAME in user_list: user_list.remove(ADMIN_USERNAME)
                target = st.selectbox("选择用户", ["请选择"]+user_list)
                
                if target != "请选择":
                    try:
                        curr_val = int(all_users[all_users["username"]==target]["quota"].iloc[0])
                    except: curr_val = 0
                    new_val = st.number_input(f"修改 {target} 的积分", value=curr_val, step=10)
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💾 保存"):
                            update_user_quota(target, new_val)
                            st.success("成功"); time.sleep(0.5); st.rerun()
                    with col2:
                        if st.button("❌ 删除"):
                            delete_user(target); st.rerun()

        st.divider()
        st.caption("我的自选股")
        user_w = get_user_watchlist(user)
        if user_w:
            cols = st.columns(3)
            for i, c in enumerate(user_w):
                if cols[i%3].button(c, key=f"w_{c}"):
                    st.session_state["stock_code"] = c
                    st.session_state["data_loaded"] = False
                    st.rerun()

        st.divider()
        default_token = ""
        try:
            if "TUSHARE_TOKEN" in st.secrets: default_token = st.secrets["TUSHARE_TOKEN"]
        except: pass
        tushare_token = st.text_input("Tushare Token", value=default_token, type="password")
        
        code_input = st.text_input("股票代码", value=st.session_state["stock_code"]).strip()
        if code_input != st.session_state["stock_code"]:
            st.session_state["stock_code"] = code_input
            st.session_state["data_loaded"] = False
            st.rerun()
            
        stock_name = st.text_input("名称", value=get_stock_name(code_input, tushare_token) or "未知")
        window_days = st.radio("窗口", [60, 120, 250], index=1, horizontal=True)
        adjust = st.selectbox("复权", ["qfq", "hfq", ""], index=0)
        
        st.divider()
        st.caption("显示选项")
        show_gann = st.checkbox("江恩线", True)
        show_chanlun = st.checkbox("缠论分型", True)
        show_fib = st.checkbox("斐波那契", True)

    c_title, c_fav = st.columns([8, 2])
    with c_title: st.title(f"📈 {stock_name} ({st.session_state['stock_code']})")
    with c_fav:
        if st.session_state['stock_code'] in get_user_watchlist(user):
            if st.button("💔 移除自选"): toggle_watchlist(user, st.session_state['stock_code']); st.rerun()
        else:
            if st.button("❤️ 加入自选"): toggle_watchlist(user, st.session_state['stock_code']); st.rerun()

    # 积分消耗逻辑
    if not st.session_state["data_loaded"]:
        st.info("👋 欢迎回来！点击下方按钮开始分析 (消耗 1 积分)")
        if st.button("🔍 开始分析", type="primary"):
            if consume_quota(user):
                st.session_state["data_loaded"] = True
                st.rerun()
            else: st.error("❌ 积分不足！请联系管理员充值。")
        st.stop()
        
    if st.button("🔄 刷新数据 (消耗 1 积分)"):
        if consume_quota(user):
            st.cache_data.clear()
            st.rerun()
        else: st.error("❌ 积分不足！")

    with st.spinner("🚀 计算中..."):
        df = fetch_hist(st.session_state['stock_code'], tushare_token, 380, adjust)
        fund = fetch_fundamentals(st.session_state['stock_code'], tushare_token)
    
    if df.empty: st.error("无数据"); st.stop()
    
    df = calc_indicators(df); df = detect_fractals(df)
    v_df = df.tail(window_days).copy(); latest = v_df.iloc[-1]
    last_close = float(latest["close"])
    
    t_txt, t_col = main_uptrend_state(v_df)
    if t_col=="success": st.success(f"## {t_txt}")
    elif t_col=="warning": st.warning(f"## {t_txt}")
    else: st.error(f"## {t_txt}")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("价格", f"{latest['close']:.2f}", f"{latest['pct_change']:.2f}%")
    c2.metric("RSI", f"{latest['RSI']:.1f}")
    c3.metric("MACD", f"{latest['HIST']:.3f}")
    c4.metric("ADX", f"{latest['ADX']:.1f}")
    c5.metric("PE", fund['pe'])
    
    plot_kline(v_df, "K线分析", show_gann, show_chanlun, show_fib)
    score, action, pos, reasons, color, b_sig, s_sig, sup, res = make_signals(v_df)
    
    st.subheader(f"🤖 AI 建议: {action} (评分 {score})")
    if color=="success": st.success(f"建议仓位: {pos}")
    elif color=="warning": st.warning(f"建议仓位: {pos}")
    else: st.error(f"建议仓位: {pos}")
    
    atr = latest["ATR14"]
    stop_loss = last_close - 2 * atr if pd.notna(atr) else sup
    take_profit = last_close + 3 * atr if pd.notna(atr) else res
    
    scol1, scol2, scol3 = st.columns(3)
    scol1.metric("🛡️ 止损参考", f"{stop_loss:.2f}")
    scol2.metric("💰 止盈参考", f"{take_profit:.2f}")
    scol3.metric("🏗️ 支撑位", f"{sup:.2f}")
    
    if b_sig: st.success("🔥 触发短线金叉买点！")
    if s_sig: st.error("❄️ 触发短线死叉卖点！")
    
    with st.expander("查看详细逻辑"):
        for r in reasons: st.write(r)

if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if not st.session_state["logged_in"]: login_page()
else: main_stock_system()
