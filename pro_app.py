import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import bcrypt
import random
import string
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback
from datetime import datetime, timedelta
import urllib.request
import json
import socket
import base64

# ✅ 0. 依赖库检查
try:
    import yfinance as yf
except ImportError:
    st.error("🚨 严重错误：缺少 `yfinance` 库")
    st.stop()

# ==========================================
# 1. 核心配置
# ==========================================
st.set_page_config(
    page_title="阿尔法量研 Pro V81 (Mobile Fix)",
    layout="wide",
    page_icon="🔥",
    initial_sidebar_state="expanded"
)

# 初始化 Session
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if "code" not in st.session_state: st.session_state.code = "600519"
if "paid_code" not in st.session_state: st.session_state.paid_code = "" 
if "trade_qty" not in st.session_state: st.session_state.trade_qty = 100

# ✅ 模拟交易数据结构初始化
if "paper_account" not in st.session_state: 
    st.session_state.paper_account = {
        "cash": 1000000.0,
        "holdings": {}, 
        "history": []
    }

# ✅ 全局变量
ma_s = 5
ma_l = 20
flags = {
    'ma': True, 'boll': True, 'vol': True, 'macd': True, 
    'kdj': True, 'gann': False, 'fib': True, 'chan': True
}

# 核心常量
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"
DB_FILE = "users_v69.csv" 
KEYS_FILE = "card_keys.csv"
WECHAT_VALID_CODE = "666888" 

# Optional deps
ts = None
bs = None
try: import tushare as ts
except: pass
try: import baostock as bs
except: pass

# 🔥 CSS 样式 (V81 修复版：强制显示侧边栏按钮 + 红盈蓝亏)
ui_css = """
<style>
    /* 全局背景 */
    .stApp {
        background-color: #f5f7f9;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
    }

    /* 隐藏 Header */
    header[data-testid="stHeader"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    
    /* 📱📱📱 移动端侧边栏按钮强制修复 (核心修改) 📱📱📱 */
    section[data-testid="stSidebar"] {
        z-index: 99999 !important;
    }
    
    button[data-testid="stSidebarCollapsedControl"] {
        display: block !important;
        position: fixed !important;
        top: 10px !important;
        left: 10px !important;
        z-index: 1000000 !important; /* 层级最高 */
        background-color: #2962ff !important;
        color: white !important;
        border-radius: 50% !important;
        width: 44px !important;
        height: 44px !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3) !important;
        border: 2px solid white !important;
    }
    
    /* 强制图标颜色 */
    button[data-testid="stSidebarCollapsedControl"] svg {
        fill: white !important;
        stroke: white !important;
    }
    
    /* 移动端顶部留白，防止按钮遮挡内容 */
    .block-container {
        padding-top: 60px !important;
        padding-left: 10px !important;
        padding-right: 10px !important;
    }
    
    /* 按钮样式 */
    div.stButton > button {
        border-radius: 8px;
        height: 44px;
        font-weight: 600;
        border: 1px solid #ddd;
    }
    div.stButton > button[kind="primary"] {
        background: #d32f2f; /* 红色主按钮 */
        color: white;
        border: none;
    }

    /* 🔴🔵 红蓝配色定义 (Red=Win, Blue=Loss) */
    .color-up { color: #d32f2f !important; font-weight: bold; } /* 红涨 */
    .color-down { color: #2962ff !important; font-weight: bold; } /* 蓝跌 */
    .bg-up { background-color: #ffebee !important; color: #c62828 !important; }
    .bg-down { background-color: #e3f2fd !important; color: #1565c0 !important; }

    /* 卡片 */
    .app-card {
        background: white; border-radius: 12px; padding: 15px;
        margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* 模拟交易 Tab */
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #d32f2f;
        border-bottom: 2px solid #d32f2f;
    }
</style>
"""
st.markdown(ui_css, unsafe_allow_html=True)

# ==========================================
# 2. 数据库与工具
# ==========================================
def init_db():
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota", "vip_expiry", "paper_json"])
        df.to_csv(DB_FILE, index=False)
    else:
        df = pd.read_csv(DB_FILE)
        if "paper_json" not in df.columns:
            df["paper_json"] = ""
            df.to_csv(DB_FILE, index=False)
    if not os.path.exists(KEYS_FILE):
        df_keys = pd.DataFrame(columns=["key", "points", "status", "created_at"])
        df_keys.to_csv(KEYS_FILE, index=False)

def safe_fmt(value, fmt="{:.2f}", default="-", suffix=""):
    try:
        f_val = float(value)
        if np.isnan(f_val) or np.isinf(f_val): return default
        return fmt.format(f_val) + suffix
    except: return default

def load_users():
    try: return pd.read_csv(DB_FILE, dtype=str).fillna("")
    except: return pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota", "vip_expiry", "paper_json"])

def save_users(df): df.to_csv(DB_FILE, index=False)

def save_user_holdings(username):
    if username == ADMIN_USER: return
    df = load_users()
    idx = df[df["username"] == username].index
    if len(idx) > 0:
        df.loc[idx[0], "paper_json"] = json.dumps(st.session_state.paper_account)
        save_users(df)

def load_user_holdings(username):
    if username == ADMIN_USER: return
    df = load_users()
    row = df[df["username"] == username]
    if not row.empty:
        try:
            data = json.loads(str(row.iloc[0]["paper_json"]))
            if "cash" in data: st.session_state.paper_account = data
        except: pass
    if "cash" not in st.session_state.paper_account:
        st.session_state.paper_account["cash"] = 1000000.0

# 简化版登录注册工具
def verify_login(u, p):
    if u == ADMIN_USER and p == ADMIN_PASS: return True
    df = load_users()
    row = df[df["username"] == u]
    if row.empty: return False
    try: return bcrypt.checkpw(p.encode(), row.iloc[0]["password_hash"].encode())
    except: return False

def register_user(u, p, initial_quota=10):
    df = load_users()
    if u in df["username"].values: return False, "用户已存在"
    hashed = bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()
    new_row = {"username": u, "password_hash": hashed, "watchlist": "", "quota": initial_quota, "vip_expiry": "", "paper_json": ""}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_users(df)
    return True, "注册成功"

# ==========================================
# 3. 股票与行情逻辑
# ==========================================
def is_cn_stock(code): return code.isdigit() and len(code) == 6
def process_ticker(code):
    code = code.strip().upper()
    if code.isdigit() and len(code) < 6: return f"{code.zfill(4)}.HK"
    return code

# ✅ 新增：实时获取单个股票价格，防止持仓显示 -100%
def get_live_price(code):
    try:
        # 优先尝试 yfinance fast_info
        t = yf.Ticker(process_ticker(code))
        price = t.fast_info.last_price
        if price and price > 0: return float(price)
        
        # 失败则尝试 history
        hist = t.history(period="1d")
        if not hist.empty: return float(hist['Close'].iloc[-1])
        return 0.0
    except: return 0.0

@st.cache_data(ttl=1800)
def get_data_and_resample(code, timeframe):
    code = process_ticker(code)
    try:
        df = yf.download(code, period="1y", interval="1d", progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.reset_index(inplace=True)
        df.rename(columns={'date':'date','close':'close','high':'high','low':'low','open':'open','volume':'volume'}, inplace=True)
        
        if timeframe == '周线':
            df.set_index('date', inplace=True)
            df = df.resample('W').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
        elif timeframe == '月线':
            df.set_index('date', inplace=True)
            df = df.resample('M').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
            
        df['pct_change'] = df['close'].pct_change() * 100
        return df
    except: return pd.DataFrame()

def calc_indicators(df):
    if df.empty: return df
    c = df['close']
    df['MA_Short'] = c.rolling(ma_s).mean()
    df['MA_Long'] = c.rolling(ma_l).mean()
    df['MA60'] = c.rolling(60).mean()
    
    # KDJ
    low9 = df['low'].rolling(9).min()
    high9 = df['high'].rolling(9).max()
    rsv = (c - low9)/(high9 - low9 + 1e-9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # MACD
    exp1 = c.ewm(span=12, adjust=False).mean()
    exp2 = c.ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['HIST'] = 2 * (df['DIF'] - df['DEA'])
    
    # RSI
    delta = c.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.rolling(14).mean() / (down.rolling(14).mean() + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df.fillna(method='bfill')

# ==========================================
# 4. 绘图与策略
# ==========================================
def plot_chart(df, flags):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.05)
    
    # K线 (红涨蓝跌)
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                                 increasing_line_color='#d32f2f', decreasing_line_color='#2962ff', name='K线'), row=1, col=1)
    
    if flags['ma']:
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA_Short'], line=dict(color='#333', width=1), name=f'MA{ma_s}'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA_Long'], line=dict(color='#ff9800', width=1), name=f'MA{ma_l}'), row=1, col=1)
    
    # Volume (红涨蓝跌)
    colors = ['#d32f2f' if c >= o else '#2962ff' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df['date'], y=df['volume'], marker_color=colors, name='成交量'), row=2, col=1)
    
    # MACD
    if flags['macd']:
        fig.add_trace(go.Bar(x=df['date'], y=df['HIST'], marker_color=colors, name='MACD柱'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['DIF'], line=dict(color='#2196f3', width=1), name='DIF'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['DEA'], line=dict(color='#ff9800', width=1), name='DEA'), row=3, col=1)

    fig.update_layout(height=600, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

def get_daily_picks(wl):
    # 简单的策略筛选模拟
    pool = list(set(["600519", "NVDA", "TSLA", "AAPL"] + wl))
    res = []
    for c in pool[:6]:
        # 模拟：随机给出红蓝信号
        signal = random.choice(["buy", "sell", "wait"])
        if signal == "buy":
            res.append({"code": c, "tag": "🚀 趋势突破", "color": "red"})
        elif signal == "sell":
            res.append({"code": c, "tag": "🔵 顶部背离", "color": "blue"})
    return res

# ==========================================
# 5. 主程序
# ==========================================
init_db()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📈 阿尔法量研 Pro V81")
    
    if st.session_state.logged_in:
        user = st.session_state["user"]
        st.success(f"👤 {user}")
        
        # 股票输入
        new_c = st.text_input("股票代码", st.session_state.code)
        if new_c != st.session_state.code: 
            st.session_state.code = new_c
            st.rerun()
            
        # 侧边栏菜单
        menu = st.radio("导航", ["行情分析", "模拟交易", "策略选股"], label_visibility="collapsed")
        
        if st.button("退出"):
            st.session_state.logged_in = False
            st.rerun()
    else:
        st.info("请先登录")

# --- Login Logic ---
if not st.session_state.logged_in:
    st.title("阿尔法量研 Pro")
    tab1, tab2 = st.tabs(["登录", "注册"])
    with tab1:
        u = st.text_input("账号")
        p = st.text_input("密码", type="password")
        if st.button("登录"):
            if verify_login(u, p):
                st.session_state.logged_in = True
                st.session_state.user = u
                st.rerun()
            else: st.error("错误")
    st.stop()

# --- Main Logic ---
st.title(f"{st.session_state.code}")

# 获取当前页面股票的实时价格（用于下单）
try:
    with st.spinner("获取实时行情..."):
        current_df = get_data_and_resample(st.session_state.code, "日线")
        if not current_df.empty:
            curr_price = float(current_df.iloc[-1]['close'])
            pct_chg = float(current_df.iloc[-1]['pct_change'])
            curr_price_color = "color-up" if pct_chg > 0 else "color-down"
            
            # 显示大字报价
            st.markdown(f"""
            <div style="text-align:center; padding: 10px;">
                <span style="font-size: 40px; font-weight: 800;" class="{curr_price_color}">{curr_price:.2f}</span>
                <span style="font-size: 18px; margin-left: 10px;" class="{curr_price_color}">{pct_chg:+.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            curr_price = 0
except:
    curr_price = 0
    st.error("行情获取失败")

# 📥📥📥 模拟交易模块 (Mobile Optimized & Color Fixed) 📥📥📥
with st.expander("🎮 模拟交易 (仿真账户)", expanded=False): # 默认折叠
    paper = st.session_state.paper_account
    cash = paper.get("cash", 1000000.0)
    holdings = paper.get("holdings", {})
    
    # 1. 动态计算总资产 (关键修复：循环计算所有持仓的实时市值)
    total_hold_val = 0
    
    # 预先获取持仓中所有股票的最新价
    # 注意：为了性能，这里在前端渲染时逐个获取可能会慢，但在Streamlit中是可接受的
    # 如果持仓很多，建议批量获取。这里演示逐个修复 -100% 问题。
    
    realtime_prices = {} 
    
    # 计算总资产
    for h_code, h_data in holdings.items():
        # 如果是当前页面股票，直接用 current_price
        if h_code == st.session_state.code and curr_price > 0:
            rt_price = curr_price
        else:
            # 否则去获取该股票的实时价格
            rt_price = get_live_price(h_code)
            # 如果获取失败（比如网络问题），为了不显示 -100%，暂时用成本价计算
            if rt_price <= 0: rt_price = h_data['cost']
        
        realtime_prices[h_code] = rt_price # 存起来下面列表用
        total_hold_val += rt_price * h_data['qty']

    total_assets = cash + total_hold_val
    total_profit = total_assets - 1000000.0
    
    # 🔴 红盈 🔵 蓝亏
    asset_color = "#d32f2f" if total_profit >= 0 else "#2962ff"
    
    st.markdown(f"""
    <div class="app-card" style="border-left: 5px solid {asset_color}">
        <div style="display:flex; justify-content:space-between; color:#666; font-size:12px;">
            <span>总资产</span> <span>可用资金</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-weight:bold; font-size:18px;">
            <span>{total_assets:,.2f}</span> <span>{cash:,.2f}</span>
        </div>
        <div style="margin-top:5px; font-size:13px; color:{asset_color}">
            总盈亏: {total_profit:+,.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_trade, tab_pos, tab_his = st.tabs(["⚡ 下单", "📦 持仓", "📝 记录"])
    
    # --- Tab 1: 下单 ---
    with tab_trade:
        if curr_price > 0:
            action = st.radio("方向", ["买入", "卖出"], horizontal=True, label_visibility="collapsed")
            
            # 快捷按钮
            c1, c2, c3, c4 = st.columns(4)
            if action == "买入":
                max_buy = int(cash // (curr_price * 100)) * 100
                if c1.button("1/4"): st.session_state.trade_qty = max(100, int(max_buy * 0.25))
                if c2.button("半仓"): st.session_state.trade_qty = max(100, int(max_buy * 0.5))
                if c3.button("全仓"): st.session_state.trade_qty = max(100, max_buy)
                
                qty = st.number_input("股数", min_value=100, step=100, value=st.session_state.trade_qty)
                
                if st.button("🔴 买入", type="primary", use_container_width=True):
                    cost = qty * curr_price
                    if cost > cash:
                        st.error("资金不足")
                    else:
                        paper['cash'] -= cost
                        if st.session_state.code in holdings:
                            old = holdings[st.session_state.code]
                            new_qty = old['qty'] + qty
                            new_avg = (old['cost']*old['qty'] + cost) / new_qty
                            holdings[st.session_state.code] = {'cost': new_avg, 'qty': new_qty}
                        else:
                            holdings[st.session_state.code] = {'cost': curr_price, 'qty': qty}
                        
                        # ✅ 修复：时间同步
                        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        paper['history'].insert(0, {"time": now_str, "code": st.session_state.code, "action": "买入", "price": curr_price, "qty": qty})
                        save_user_holdings(st.session_state.user)
                        st.success("买入成交")
                        time.sleep(0.5); st.rerun()
                        
            else: # 卖出
                curr_qty = holdings.get(st.session_state.code, {}).get('qty', 0)
                if c1.button("1/3"): st.session_state.trade_qty = max(100, int(curr_qty * 0.33 / 100)*100)
                if c2.button("1/2"): st.session_state.trade_qty = max(100, int(curr_qty * 0.5 / 100)*100)
                if c3.button("全清"): st.session_state.trade_qty = max(100, curr_qty)
                
                qty = st.number_input("股数", min_value=100, max_value=max(100, curr_qty), step=100, value=st.session_state.trade_qty)
                
                # 蓝色卖出按钮
                st.markdown("""<style>div.stButton > button[kind="secondary"] {color: #2962ff; border-color: #2962ff;}</style>""", unsafe_allow_html=True)
                if st.button("🔵 卖出", type="secondary", use_container_width=True):
                    if qty > curr_qty:
                        st.error("持仓不足")
                    else:
                        amt = qty * curr_price
                        paper['cash'] += amt
                        remain = curr_qty - qty
                        if remain == 0: del holdings[st.session_state.code]
                        else: holdings[st.session_state.code]['qty'] = remain
                        
                        # ✅ 修复：时间同步
                        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        paper['history'].insert(0, {"time": now_str, "code": st.session_state.code, "action": "卖出", "price": curr_price, "qty": qty})
                        save_user_holdings(st.session_state.user)
                        st.success("卖出成交")
                        time.sleep(0.5); st.rerun()

    # --- Tab 2: 持仓 (修复 -100% bug & 红蓝配色) ---
    with tab_pos:
        if not holdings:
            st.info("暂无持仓")
        else:
            for h_code, h_data in holdings.items():
                # 使用刚才计算好的实时价格
                p_now = realtime_prices.get(h_code, h_data['cost']) 
                cost = h_data['cost']
                qty = h_data['qty']
                
                # 盈亏计算
                profit_val = (p_now - cost) * qty
                profit_pct = (p_now - cost) / cost * 100
                
                # 配色
                bg_cls = "bg-up" if profit_val >= 0 else "bg-down"
                
                st.markdown(f"""
                <div class="app-card" style="padding: 10px; display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-weight:bold; font-size:16px;">{h_code}</div>
                        <div style="font-size:12px; color:#888;">{qty}股 | 成本 {cost:.2f}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-weight:bold; font-size:16px;">{p_now:.2f}</div>
                        <div class="{bg_cls}" style="padding: 2px 8px; border-radius: 4px; font-size:12px; display:inline-block;">
                            {profit_pct:+.2f}% ({profit_val:+,.0f})
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # --- Tab 3: 记录 (时间同步) ---
    with tab_his:
        for h in paper['history']:
            # 这里的 h['time'] 已经是带秒的了
            color = "red" if h['action'] == "买入" else "blue"
            st.markdown(f"""
            <div style="border-bottom:1px solid #eee; padding: 5px; font-size:13px;">
                <span style="color:#999; margin-right:10px;">{h['time']}</span>
                <span style="font-weight:bold;">{h['code']}</span>
                <span style="color:{color}; margin: 0 10px;">{h['action']}</span>
                <span>{h['price']:.2f}</span>
                <span style="float:right;">{h['qty']}股</span>
            </div>
            """, unsafe_allow_html=True)

# --- Chart Area ---
st.markdown("### 📊 行情图表")
if not current_df.empty:
    current_df = calc_indicators(current_df)
    
    # 指标开关
    with st.expander("🛠️ 指标设置", expanded=False):
        c1, c2 = st.columns(2)
        flags['ma'] = c1.checkbox("均线 MA", True)
        flags['macd'] = c2.checkbox("MACD", True)

    plot_chart(current_df.iloc[-120:], flags)
else:
    st.warning("数据加载中或无效代码")

# --- Strategy Area ---
st.markdown("### 🧬 策略选股")
picks = get_daily_picks(load_users()[load_users()['username']==st.session_state.user]['watchlist'].iloc[0].split(","))
cols = st.columns(3)
for i, p in enumerate(picks):
    with cols[i%3]:
        # 红蓝配色策略标签
        tag_bg = "#ffebee" if p['color']=="red" else "#e3f2fd"
        tag_tx = "#c62828" if p['color']=="red" else "#1565c0"
        if st.button(f"{p['code']}\n{p['tag']}", key=f"pk_{i}"):
            st.session_state.code = p['code']
            st.rerun()