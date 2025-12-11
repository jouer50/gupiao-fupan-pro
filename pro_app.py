import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import bcrypt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. 核心配置 & 界面隐藏
# ==========================================
st.set_page_config(
    page_title="A股复盘系统(修复版)",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# 🚫 隐藏菜单 CSS
hide_css = """
<style>
    header {visibility: hidden !important; height: 0px !important; padding: 0px !important; margin: 0px !important;}
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    .block-container {padding-top: 1rem !important;}
    .stDeployButton {display: none !important;}
</style>
"""
st.markdown(hide_css, unsafe_allow_html=True)

# 👑 管理员账号
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"
DB_FILE = "users_v4.csv" # 升级数据库版本

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
# 2. 数据库逻辑
# ==========================================
def init_db():
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"])
        df.to_csv(DB_FILE, index=False)

init_db()

def load_users():
    try:
        return pd.read_csv(DB_FILE, dtype={"watchlist": str, "quota": int})
    except:
        return pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"])

def save_users(df):
    df.to_csv(DB_FILE, index=False)

def verify_login(u, p):
    # 优先检查管理员
    if u == ADMIN_USER and p == ADMIN_PASS: return True
    
    df = load_users()
    row = df[df["username"] == u]
    if row.empty: return False
    try: return bcrypt.checkpw(p.encode(), row.iloc[0]["password_hash"].encode())
    except: return False

def consume_quota(u):
    if u == ADMIN_USER: return True
    df = load_users()
    idx = df[df["username"] == u].index
    if len(idx) > 0 and df.loc[idx[0], "quota"] > 0:
        df.loc[idx[0], "quota"] -= 1
        save_users(df)
        return True
    return False

def register_user(u, p):
    if u == ADMIN_USER: return False, "无法注册管理员名字"
    df = load_users()
    if u in df["username"].values: return False, "用户已存在"
    
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(p.encode(), salt).decode()
    new_row = {"username": u, "password_hash": hashed, "watchlist": "", "quota": 20}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_users(df)
    return True, "注册成功"

# ==========================================
# 3. 股票数据逻辑 (增强健壮性)
# ==========================================
def _to_ts_code(symbol):
    symbol = symbol.strip()
    if symbol.isdigit(): return f"{symbol}.SH" if symbol.startswith('6') else f"{symbol}.SZ"
    return symbol

def _to_bs_code(symbol):
    symbol = symbol.strip()
    if symbol.isdigit(): return f"sh.{symbol}" if symbol.startswith('6') else f"sz.{symbol}"
    return symbol

@st.cache_data(ttl=3600)
def get_name(code, token):
    if token and ts:
        try:
            pro = ts.pro_api(token)
            df = pro.stock_basic(ts_code=_to_ts_code(code), fields='name')
            if not df.empty: return df.iloc[0]['name']
        except: pass
    return code

@st.cache_data(ttl=3600)
def get_data(code, token, days, adjust):
    # Tushare
    if token and ts:
        try:
            pro = ts.pro_api(token)
            e = pd.Timestamp.today().strftime('%Y%m%d')
            s = (pd.Timestamp.today() - pd.Timedelta(days=days*2)).strftime('%Y%m%d')
            df = pro.daily(ts_code=_to_ts_code(code), start_date=s, end_date=e)
            if df is not None and not df.empty:
                df = df.rename(columns={'trade_date':'date','vol':'volume','pct_chg':'pct_change'})
                df['date'] = pd.to_datetime(df['date'])
                cols = ['open','high','low','close','volume']
                for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
                return df.sort_values('date').reset_index(drop=True)
        except: pass
        
    # Baostock
    if bs:
        bs.login()
        e = pd.Timestamp.today().strftime('%Y-%m-%d')
        s = (pd.Timestamp.today() - pd.Timedelta(days=days*2)).strftime('%Y-%m-%d')
        rs = bs.query_history_k_data_plus(_to_bs_code(code),
            "date,open,high,low,close,volume,pctChg",
            start_date=s, end_date=e, frequency="d", adjustflag="3")
        data = rs.get_data()
        bs.logout()
        if not data.empty:
            df = data.rename(columns={'pctChg':'pct_change'})
            df['date'] = pd.to_datetime(df['date'])
            cols = ['open','high','low','close','volume','pct_change']
            for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
            return df.sort_values('date').reset_index(drop=True)
            
    return pd.DataFrame()

def calc_indicators(df):
    if df.empty: return df
    # 强制转为float，防止报错
    for c in ['close','high','low','volume']:
        df[c] = df[c].astype(float)
        
    close = df['close']
    # MA
    for n in [5,10,20,60]: df[f'MA{n}'] = close.rolling(n).mean()
    
    # KDJ (增加异常处理)
    try:
        low_list = df['low'].rolling(9, min_periods=1).min()
        high_list = df['high'].rolling(9, min_periods=1).max()
        rsv = (close - low_list) / (high_list - low_list + 1e-9) * 100
        df['K'] = rsv.ewm(com=2, adjust=False).mean()
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
    except:
        df['K'] = 50; df['D'] = 50; df['J'] = 50

    # MACD
    try:
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        df['DIF'] = exp1 - exp2
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['HIST'] = 2 * (df['DIF'] - df['DEA'])
    except:
        df['DIF'] = 0; df['DEA'] = 0; df['HIST'] = 0

    # RSI
    try:
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / (ema_down + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
    except:
        df['RSI'] = 50

    return df

def plot_kline(df, title):
    if df.empty: return
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        row_heights=[0.5, 0.1, 0.2, 0.2], vertical_spacing=0.02)

    # K线
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'], name='K线'), row=1, col=1)
    
    # 均线 (防御性添加)
    for ma in ['MA5','MA20','MA60']:
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df['date'], y=df[ma], name=ma, line=dict(width=1)), row=1, col=1)

    # 成交量
    colors = ['red' if o >= c else 'green' for o, c in zip(df['open'], df['close'])]
    fig.add_trace(go.Bar(x=df['date'], y=df['volume'], marker_color=colors, name='成交量'), row=2, col=1)

    # MACD (防御性添加)
    if 'DIF' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['DIF'], name='DIF'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['DEA'], name='DEA'), row=3, col=1)
        fig.add_trace(go.Bar(x=df['date'], y=df['HIST'], name='MACD柱'), row=3, col=1)

    # KDJ (防御性添加 - 这里是之前报错的地方)
    if 'K' in df.columns and 'D' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name='K'), row=4, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name='D'), row=4, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['J'], name='J'), row=4, col=1)

    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=800, margin=dict(t=30,b=20,l=20,r=20))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. 主程序
# ==========================================
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False

# --- 登录 ---
if not st.session_state["logged_in"]:
    st.markdown("<br><br><h1 style='text-align:center'>🔐 复盘系统 Pro</h1>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        tab1, tab2 = st.tabs(["登录", "注册"])
        with tab1:
            u = st.text_input("账号")
            p = st.text_input("密码", type="password")
            if st.button("🚀 登录", type="primary", use_container_width=True):
                if verify_login(u.strip(), p):
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = u.strip()
                    st.rerun()
                else:
                    st.error("账号或密码错误 (管理员: ZCX001 / 123456)")
        with tab2:
            nu = st.text_input("新账号")
            np1 = st.text_input("新密码", type="password")
            if st.button("📝 注册", use_container_width=True):
                suc, msg = register_user(nu.strip(), np1)
                if suc: st.success(msg)
                else: st.error(msg)
    st.stop()

# --- 主界面 ---
user = st.session_state["user"]
is_admin = (user == ADMIN_USER)

with st.sidebar:
    st.header(f"👤 {user}")
    
    # 管理员后台
    if is_admin:
        st.success("✅ 管理员")
        with st.expander("👮‍♂️ 积分管理", expanded=True):
            df_u = load_users()
            st.dataframe(df_u[["username","quota"]], hide_index=True)
            u_list = [x for x in df_u["username"] if x != ADMIN_USER]
            if u_list:
                target = st.selectbox("修改用户", u_list)
                val = st.number_input("积分", value=50, step=10)
                if st.button("保存修改"):
                    idx = df_u[df_u["username"]==target].index[0]
                    df_u.loc[idx, "quota"] = val
                    save_users(df_u)
                    st.success("成功")
    else:
        # 普通用户看积分
        df_u = load_users()
        q = df_u[df_u["username"]==user]["quota"].iloc[0]
        st.metric("剩余积分", q)

    st.divider()
    
    # 令牌设置
    try:
        def_tok = st.secrets["TUSHARE_TOKEN"]
    except:
        def_tok = ""
    token = st.text_input("Tushare Token", value=def_tok, type="password")
    
    # 股票输入
    if "code" not in st.session_state: st.session_state.code = "600519"
    new_code = st.text_input("股票代码", st.session_state.code)
    if new_code != st.session_state.code:
        st.session_state.code = new_code
    
    name = get_name(st.session_state.code, token)
    st.caption(f"当前股票: {name}")
    
    # ✅ 修复1：找回了7天/30天等选项
    days = st.radio("时间窗口 (天)", [7, 30, 60, 90, 180, 360], index=2, horizontal=True)
    adjust = st.selectbox("复权方式", ["qfq", "hfq", ""], 0)
    
    st.divider()
    if st.button("🚪 退出登录"):
        st.session_state["logged_in"] = False
        st.rerun()

# 主区域
c1, c2 = st.columns([3, 1])
with c1: st.title(f"📈 {name} ({st.session_state.code})")
with c2:
    if st.button("🔄 刷新数据 (消耗1积分)", type="primary"):
        if consume_quota(user):
            st.session_state["refresh"] = time.time() # 强制刷新
            st.rerun()
        else:
            st.error("积分不足")

# 数据获取与展示
with st.spinner("正在分析数据..."):
    df = get_data(st.session_state.code, token, days, adjust)
    
    if df.empty:
        st.warning("⚠️ 未获取到数据，请检查：\n1. 股票代码是否正确\n2. Tushare Token 是否有效\n3. 刚开盘或收盘可能存在延迟")
    else:
        # ✅ 修复2：增强的指标计算，防止KeyError
        df = calc_indicators(df)
        
        latest = df.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("当前价格", f"{latest['close']:.2f}")
        c2.metric("涨跌幅", f"{latest['pct_change']:.2f}%")
        
        # 安全获取指标，防止报错
        rsi_val = f"{latest['RSI']:.1f}" if 'RSI' in df.columns else "N/A"
        c3.metric("RSI (14)", rsi_val)
        
        # ✅ 修复3：安全的绘图函数
        plot_kline(df, f"{name} K线走势")
        
        # 简单的信号提示
        st.subheader("💡 智能信号")
        msgs = []
        if 'MA5' in df.columns and latest['MA5'] > latest['MA20']: msgs.append("✅ 短线多头排列 (MA5 > MA20)")
        if 'MA5' in df.columns and latest['MA5'] < latest['MA20']: msgs.append("❌ 短线空头排列 (MA5 < MA20)")
        if 'RSI' in df.columns and latest['RSI'] < 30: msgs.append("📉 超卖区域 (RSI < 30)，注意反弹")
        
        if msgs:
            for m in msgs: st.write(m)
        else:
            st.info("暂无明显趋势信号")
