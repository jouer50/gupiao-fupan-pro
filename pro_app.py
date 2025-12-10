import streamlit as st
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
# 0. 全局配置
# ==========================================
st.set_page_config(
    page_title="A股深度复盘系统 Pro",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# 🚫 界面隐藏 CSS
hide_ui_css = """
<style>
    header {visibility: hidden !important;}
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    .block-container {padding-top: 1rem !important;}
    .stDeployButton {display: none !important;}
</style>
"""
st.markdown(hide_ui_css, unsafe_allow_html=True)

# 👑 管理员配置
ADMIN_USERNAME = "ZCX001"
ADMIN_DEFAULT_PASS = "123456"

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
# 🔐 用户数据库 & 强制修复管理员
# ==========================================

USER_DB_FILE = "users.csv"

def init_db():
    # 1. 读取或创建
    if not os.path.exists(USER_DB_FILE):
        df = pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"])
    else:
        try:
            df = pd.read_csv(USER_DB_FILE, dtype={"watchlist": str, "quota": int})
            # 补全列
            if "quota" not in df.columns: df["quota"] = 20
            if "watchlist" not in df.columns: df["watchlist"] = ""
        except:
            df = pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"])

    # 2. 🔨 强制重置管理员 (确保绝对干净)
    # 先删除旧的管理员记录
    df = df[df["username"] != ADMIN_USERNAME]
    
    # 再重新插入纯净的管理员记录
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(ADMIN_DEFAULT_PASS.encode('utf-8'), salt).decode('utf-8')
    new_admin = pd.DataFrame({
        "username": [ADMIN_USERNAME],
        "password_hash": [hashed],
        "watchlist": [""],
        "quota": [999999]
    })
    df = pd.concat([df, new_admin], ignore_index=True)
    
    df.to_csv(USER_DB_FILE, index=False)

# 每次启动都运行一次修复
init_db()

def load_users():
    return pd.read_csv(USER_DB_FILE, dtype={"watchlist": str, "quota": int})

def save_users_df(df):
    df.to_csv(USER_DB_FILE, index=False)

def save_user(username, password):
    username = username.strip() # 强制去空格
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

def verify_login(username, password):
    username = username.strip()
    df = load_users()
    user_row = df[df["username"] == username]
    if user_row.empty: return False, "❌ 用户不存在"
    stored_hash = user_row.iloc[0]["password_hash"]
    try:
        if not bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            return False, "❌ 密码错误"
    except: return False, "❌ 密码校验失败"
    return True, "Login Success"

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

def generate_captcha():
    chars = string.ascii_uppercase + string.digits
    code = ''.join(random.choice(chars) for _ in range(4))
    return code

# ==========================================
# 🔑 登录页面
# ==========================================
def login_page():
    st.markdown("<br><h1 style='text-align: center;'>🔐 A股深度复盘系统 Pro</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if "captcha_code" not in st.session_state: st.session_state["captcha_code"] = generate_captcha()
        
        tab1, tab2 = st.tabs(["🔑 登录", "📝 注册"])
        with tab1:
            st.success(f"💡 管理员账号已重置: **{ADMIN_USERNAME}** / **{ADMIN_DEFAULT_PASS}**")
            login_user = st.text_input("用户名", key="l_user")
            login_pass = st.text_input("密码", type="password", key="l_pass")
            c1, c2 = st.columns([2, 1])
            with c1: captcha_input = st.text_input("验证码", placeholder="不区分大小写")
            with c2:
                st.markdown(f"### `{st.session_state['captcha_code']}`")
                if st.button("🔄"): st.session_state["captcha_code"] = generate_captcha(); st.rerun()
            
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
                        st.session_state["current_user"] = login_user.strip()
                        st.success("登录成功")
                        time.sleep(0.5); st.rerun()
        with tab2:
            st.caption("新用户注册送 20 积分")
            new_user = st.text_input("新用户名", key="r_user")
            new_pass = st.text_input("设置密码", type="password", key="r_pass")
            new_pass2 = st.text_input("确认密码", type="password", key="r_pass2")
            if st.button("立即注册", use_container_width=True):
                df = load_users()
                clean = new_user.strip()
                if clean in df["username"].values: st.warning("用户名已存在")
                elif len(new_pass) < 4: st.warning("密码太短")
                elif new_pass != new_pass2: st.error("两次密码不一致")
                else:
                    save_user(clean, new_pass)
                    st.success("注册成功！请登录。")

# ==========================================
# 📈 股票逻辑
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

@st.cache_data(ttl=60*60*24)
def get_stock_name(symbol: str, token: str = "") -> str:
    name = ""
    if token and ts:
        try:
            pro = ts.pro_api(token)
            df = pro.stock_basic(ts_code=_to_ts_code(symbol), fields='name')
            if not df.empty: return df.iloc[0]['name']
        except: pass
    if bs:
        try:
            bs.login()
            rs = bs.query_stock_basic(code=_to_bs_code(symbol))
            if rs.error_code == '0':
                row = rs.get_row_data()
                if row and len(row) > 1: name = row[1]
            bs.logout()
        except: pass
    return name

@st.cache_data(ttl=60*60*12)
def fetch_fundamentals(symbol: str, token: str):
    data = {"pe": "N/A", "pb": "N/A", "total_mv": "N/A", "float_mv": "N/A", "roe": "N/A"}
    if token and ts:
        try:
            pro = ts.pro_api(token)
            ts_code = _to_ts_code(symbol)
            df = pro.daily_basic(ts_code=ts_code, fields='pe_ttm,pb,total_mv,circ_mv')
            if not df.empty:
                r = df.iloc[-1]
                data.update({"pe": f"{r['pe_ttm']:.2f}", "pb": f"{r['pb']:.2f}", 
                             "total_mv": f"{r['total_mv']/10000:.2f}亿", "float_mv": f"{r['circ_mv']/10000:.2f}亿"})
            df_f = pro.fina_indicator(ts_code=ts_code, fields='roe')
            if not df_f.empty: data["roe"] = f"{df_f.iloc[0]['roe']:.2f}%"
        except: pass
    return data

@st.cache_data(ttl=60*15, show_spinner=False)
def fetch_hist(symbol: str, token: str, days: int = 180, adjust: str = "qfq") -> pd.DataFrame:
    # Tushare
    if token and ts:
        try:
            pro = ts.pro_api(token)
            e = pd.Timestamp.today().strftime("%Y%m%d")
            s = (pd.Timestamp.today() - pd.Timedelta(days=days*3)).strftime("%Y%m%d")
            ts_code = _to_ts_code(symbol)
            df = pro.daily(ts_code=ts_code, start_date=s, end_date=e)
            if df is not None and not df.empty:
                if adjust in ("qfq", "hfq"):
                    af = pro.adj_factor(ts_code=ts_code, start_date=s, end_date=e)
                    if af is not None:
                        af = af.rename(columns={"trade_date":"date","adj_factor":"factor"})
                        df = df.merge(af[["date","factor"]], on="date", how="left")
                        df["factor"] = df["factor"].ffill().bfill()
                        adj = df["factor"] / df["factor"].iloc[-1] if adjust=="qfq" else df["factor"] / df["factor"].iloc[0]
                        for c in ["open","high","low","close"]: df[c] *= adj
                df = df.rename(columns={"trade_date":"date","vol":"volume","pct_chg":"pct_change"})
                df["date"] = pd.to_datetime(df["date"])
                for c in ["open","high","low","close","volume","pct_change"]: df[c] = pd.to_numeric(df[c], errors="coerce")
                return df.sort_values("date").reset_index(drop=True).tail(days)
        except: pass
    # Baostock
    if bs:
        bs.login()
        e = pd.Timestamp.today()
        s = e - pd.Timedelta(days=days*3)
        code = _to_bs_code(symbol)
        adj = "2" if adjust=="qfq" else "1" if adjust=="hfq" else "3"
        rs = bs.query_history_k_data_plus(code, "date,open,high,low,close,volume,pctChg",
            start_date=s.strftime("%Y-%m-%d"), end_date=e.strftime("%Y-%m-%d"), frequency="d", adjustflag=adj)
        data = rs.get_data()
        bs.logout()
        if not data.empty:
            df = data.rename(columns={"pctChg":"pct_change"})
            df["date"] = pd.to_datetime(df["date"])
            for c in ["open","high","low","close","volume","pct_change"]: df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.sort_values("date").reset_index(drop=True).tail(days)
    return pd.DataFrame()

def calc_indicators(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    for n in [5,10,20,60]: df[f"MA{n}"] = c.rolling(n).mean()
    m, s = df["MA20"], c.rolling(20).std()
    df["Upper"], df["Lower"] = m+2*s, m-2*s
    delta = c.diff()
    g, ls = delta.clip(lower=0).rolling(14).mean(), (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100/(1+g/(ls+1e-9)))
    e12, e26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    df["DIF"], df["DEA"] = e12-e26, (e12-e26).ewm(span=9).mean()
    df["HIST"] = df["DIF"] - df["DEA"]
    df["ADX"] = ((h.diff().clip(lower=0)+l.diff().clip(upper=0).abs()).rolling(14).mean()).rolling(14).mean() # simplified
    df["VOL_RATIO"] = v / (v.rolling(20).mean()+1e-9)
    df["ATR14"] = (h-l).rolling(14).mean()
    # Ichimoku
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    df["SPAN_A"] = ((tenkan + kijun) / 2).shift(26)
    df["SPAN_B"] = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    return df

def detect_fractals(df):
    df["F_TOP"] = (df["high"].shift(2)<df["high"]) & (df["high"].shift(-2)<df["high"])
    df["F_BOT"] = (df["low"].shift(2)>df["low"]) & (df["low"].shift(-2)>df["low"])
    return df

def build_bi(df):
    pts = []
    for _, r in df.iterrows():
        if r.get("F_TOP"): pts.append((r["date"], r["high"], "top"))
        if r.get("F_BOT"): pts.append((r["date"], r["low"], "bot"))
    segs, last = [], None
    for p in pts:
        if last is None: last=p; continue
        if p[2]!=last[2]: segs.append((last,p)); last=p
        else: 
            if (p[2]=="top" and p[1]>=last[1]) or (p[2]=="bot" and p[1]<=last[1]): last=p
    return segs

def gann(df):
    idx = df["low"].idxmin()
    p_d, p_p = df.loc[idx,"date"], df.loc[idx,"low"]
    days = (df["date"]-p_d).dt.days
    step = df["ATR14"].iloc[-1] or p_p*0.01
    return {n: p_p + days*step*r for n,r in [("1x1",1), ("1x2",0.5), ("2x1",2)]}

def fib(df):
    c = df.tail(120); h, l = c["high"].max(), c["low"].min()
    return {k: h-(h-l)*v for k,v in {"0.236":0.236, "0.5":0.5, "0.618":0.618}.items()}

def check_trend(df):
    l = df.iloc[-1]
    top = max(l["SPAN_A"], l["SPAN_B"])
    ma_up = df["MA20"].diff().tail(5).mean() > 0
    if l["close"] > top and ma_up: return "🚀 强势主升浪", "success"
    if l["close"] > top: return "📈 上升趋势", "success"
    return "🟡 震荡/调整", "warning"

def make_sig(df):
    l, p = df.iloc[-1], df.iloc[-2]
    s, r = 0, []
    if l["MA5"]>l["MA20"]: s+=2; r.append("✅ MA5金叉")
    else: s-=2; r.append("❌ MA5死叉")
    if l["DIF"]>l["DEA"]: s+=1; r.append("✅ MACD多头")
    if l["RSI"]<30: s+=2; r.append("📉 RSI超卖")
    if l["RSI"]>70: s-=2; r.append("📈 RSI超买")
    
    act = "买入" if s>=3 else "观望" if s>=0 else "减仓"
    col = "success" if s>=3 else "warning" if s>=0 else "error"
    pos = "50-80%" if s>=3 else "0-20%"
    
    sup = df["low"].tail(20).min()
    res = df["high"].tail(20).max()
    b_sig = (p["MA5"]<=p["MA20"] and l["MA5"]>l["MA20"])
    s_sig = (p["MA5"]>=p["MA20"] and l["MA5"]<l["MA20"])
    return s, act, pos, r, col, b_sig, s_sig, sup, res

def plot_k(df, title, g, c, f):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
    fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="K线"),1,1)
    for m in ["MA5","MA20"]: fig.add_trace(go.Scatter(x=df["date"], y=df[m], name=m, line=dict(width=1)),1,1)
    if c:
        for s,e in build_bi(df): fig.add_trace(go.Scatter(x=[s[0],e[0]], y=[s[1],e[1]], mode="lines", line=dict(color='gray', width=1)),1,1)
    if g:
        for n,y in gann(df).items(): fig.add_trace(go.Scatter(x=df["date"], y=y, name=f"Gann{n}", line=dict(dash="dot", width=1)),1,1)
    if f:
        for k,v in fib(df).items(): fig.add_hline(y=v, line_dash="dash", annotation_text=f"Fib {k}", row=1, col=1)
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="Vol"),2,1)
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=800, margin=dict(t=40,b=20))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 🚀 主程序
# ==========================================
def main_stock_system():
    if "stock_code" not in st.session_state: st.session_state["stock_code"] = "600519"
    if "data_loaded" not in st.session_state: st.session_state["data_loaded"] = False
    
    user = st.session_state['current_user']
    is_admin = (user == ADMIN_USERNAME)
    
    # --- 调试信息 (用来排错) ---
    st.sidebar.error(f"👤 当前登录: [{user}]") 
    st.sidebar.caption(f"系统管理员: [{ADMIN_USERNAME}]")
    if is_admin:
        st.sidebar.success("✅ 权限匹配成功：已识别为管理员")
    else:
        st.sidebar.warning("⚠️ 权限匹配失败：视为普通用户")
    st.sidebar.divider()
    # ---------------------------

    # 1. 顶部 Tab 分页 (把管理后台移到主界面)
    if is_admin:
        main_tab, admin_tab = st.tabs(["📈 股票分析", "👮‍♂️ 管理员后台"])
    else:
        main_tab = st.container() # 普通用户只有主界面
        admin_tab = None

    # --- 管理员后台逻辑 ---
    if is_admin and admin_tab:
        with admin_tab:
            st.subheader("👥 用户管理数据库")
            users = load_users()
            st.dataframe(users, use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                target = st.selectbox("选择用户", users["username"].unique())
                new_q = st.number_input("设置积分", value=100, step=10)
                if st.button("💾 更新积分"):
                    update_user_quota(target, new_q)
                    st.success("已更新"); time.sleep(0.5); st.rerun()
            with c2:
                if st.button("❌ 删除该用户"):
                    delete_user(target)
                    st.warning("已删除"); time.sleep(0.5); st.rerun()

    # --- 股票分析逻辑 ---
    with main_tab:
        # 侧边栏
        with st.sidebar:
            q = get_current_quota(user)
            st.metric("剩余积分", "无限" if is_admin else q)
            if st.button("🚪 退出"): st.session_state["logged_in"]=False; st.rerun()
            st.divider()
            
            # 自选股
            st.caption("我的自选")
            for c in get_user_watchlist(user):
                if st.button(c, key=f"fav_{c}"):
                    st.session_state["stock_code"] = c
                    st.session_state["data_loaded"] = False
                    st.rerun()
            
            st.divider()
            try:
                if "TUSHARE_TOKEN" in st.secrets: d_tok = st.secrets["TUSHARE_TOKEN"]
                else: d_tok=""
            except: d_tok=""
            token = st.text_input("Token", value=d_tok, type="password")
            
            # 输入框
            new_c = st.text_input("代码", value=st.session_state["stock_code"])
            if new_c != st.session_state["stock_code"]:
                st.session_state["stock_code"] = new_c
                st.session_state["data_loaded"] = False
                st.rerun()
                
            w = st.radio("窗口", [60,120,250], 1, horizontal=True)
            adj = st.selectbox("复权", ["qfq","hfq",""], 0)
            st.divider()
            s_g = st.checkbox("江恩", True)
            s_c = st.checkbox("缠论", True)
            s_f = st.checkbox("Fib", True)

        # 顶部标题区
        t_col, f_col = st.columns([8,2])
        with t_col: st.title(f"📈 {get_stock_name(st.session_state['stock_code'], token)} ({st.session_state['stock_code']})")
        with f_col:
            if st.session_state['stock_code'] in get_user_watchlist(user):
                if st.button("💔 移除自选"): toggle_watchlist(user, st.session_state['stock_code']); st.rerun()
            else:
                if st.button("❤️ 加入自选"): toggle_watchlist(user, st.session_state['stock_code']); st.rerun()

        # 积分墙
        if not st.session_state["data_loaded"]:
            st.info("👋 准备就绪")
            if st.button("🔍 消耗 1 积分查询", type="primary"):
                if consume_quota(user):
                    st.session_state["data_loaded"] = True
                    st.rerun()
                else: st.error("积分不足")
            st.stop()

        # 数据区
        if st.button("🔄 刷新数据"):
            if consume_quota(user): st.cache_data.clear(); st.rerun()
            else: st.error("积分不足")

        with st.spinner("Analyzing..."):
            df = fetch_hist(st.session_state['stock_code'], token, 380, adj)
            funda = fetch_fundamentals(st.session_state['stock_code'], token)
        
        if df.empty: st.error("No Data"); st.stop()
        
        df = calc_indicators(df); df = detect_fractals(df)
        v_df = df.tail(w).copy(); l = v_df.iloc[-1]
        
        tt, tc = check_trend(v_df)
        if tc=="success": st.success(f"## {tt}")
        else: st.warning(f"## {tt}")
        
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("价格", f"{l['close']:.2f}", f"{l['pct_change']:.2f}%")
        c2.metric("PE", funda['pe'])
        c3.metric("RSI", f"{l['RSI']:.1f}")
        c4.metric("VOL", f"{l['VOL_RATIO']:.2f}")
        
        plot_k(v_df, "K线", s_g, s_c, s_f)
        s, act, pos, r, col, b, sl, su, re = make_sig(v_df)
        
        st.subheader(f"AI: {act} (Score: {s})")
        if col=="success": st.success(f"仓位: {pos}")
        else: st.warning(f"仓位: {pos}")
        
        sc1, sc2 = st.columns(2)
        sc1.metric("支撑", f"{su:.2f}"); sc2.metric("压力", f"{re:.2f}")
        
        with st.expander("逻辑详情"):
            for i in r: st.write(i)

if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if not st.session_state["logged_in"]: login_page()
else: main_stock_system()
