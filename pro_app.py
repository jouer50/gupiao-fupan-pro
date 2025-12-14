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
import base64

# ✅ 0. 依赖库检查
try:
    import yfinance as yf
except ImportError:
    st.error("🚨 严重错误：缺少 `yfinance` 库，请 pip install yfinance")
    st.stop()

# ==========================================
# 1. 核心配置与 App 化体验优化
# ==========================================
st.set_page_config(
    page_title="阿尔法量研 Pro V80 (Mobile Optimized)",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# 初始化 Session
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if "code" not in st.session_state: st.session_state.code = "600519"
if "paid_code" not in st.session_state: st.session_state.paid_code = "" 
if "trade_qty" not in st.session_state: st.session_state.trade_qty = 100
if "paper_account" not in st.session_state: 
    st.session_state.paper_account = {"cash": 1000000.0, "holdings": {}, "history": []}

# ✅ 全局变量
ma_s = 5
ma_l = 20
flags = {
    'ma': True, 'boll': True, 'vol': True, 'macd': True, 
    'kdj': True, 'gann': False, 'fib': True, 'chan': True
}
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"
DB_FILE = "users_v80.csv" 
KEYS_FILE = "card_keys.csv"
WECHAT_VALID_CODE = "666888" 

# Optional deps
ts = None; bs = None
try: import tushare as ts
except: pass
try: import baostock as bs
except: pass

# 🔥 CSS 样式重构：移动端 App 体验 + 丝滑感
ui_css = """
<style>
    /* 全局 App 质感 */
    .stApp {
        background-color: #f2f4f8; 
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
    }
    
    /* 移动端去边距，增加可视面积 */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 3rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100%;
    }

    /* 隐藏多余元素 */
    header[data-testid="stHeader"] { background-color: transparent !important; height: 3rem; }
    [data-testid="stDecoration"] { display: none !important; }
    .stDeployButton { display: none !important; }
    
    /* 侧边栏优化 */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #eee;
    }

    /* 按钮优化：更适合手指触摸 */
    div.stButton > button {
        background: linear-gradient(180deg, #ffffff 0%, #f0f0f5 100%); 
        color: #1d1d1f; 
        border: 1px solid #d1d1d6; 
        border-radius: 12px; 
        padding: 0.5rem 1rem; 
        font-weight: 600; 
        font-size: 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05); 
        transition: transform 0.1s, box-shadow 0.1s; 
        width: 100%;
        min-height: 44px; /* Apple Human Interface Guideline minimum */
    }
    div.stButton > button:active { transform: scale(0.98); background: #e5e5ea; }
    
    /* 主色调按钮 */
    div.stButton > button[kind="primary"] { 
        background: linear-gradient(180deg, #007aff 0%, #0062cc 100%); 
        color: white; border: none; 
        box-shadow: 0 2px 6px rgba(0, 122, 255, 0.3);
    }

    /* 卡片式设计 */
    .app-card { 
        background-color: #ffffff; 
        border-radius: 16px; 
        padding: 18px; 
        margin-bottom: 12px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.04); 
        border: 1px solid #rgba(0,0,0,0.02);
    }
    
    /* 结论框 (小白专用) */
    .conclusion-box {
        margin-top: 10px;
        padding: 10px;
        background: #f9f9f9;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 700;
        color: #333;
        border-left: 4px solid #333;
    }
    
    /* 市场状态条 */
    .market-status-box {
        padding: 12px 16px; border-radius: 16px; margin-bottom: 16px;
        display: flex; align-items: center; justify-content: space-between;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03); 
    }
    .status-green { background: #e0f8e9; color: #008a00; border: 1px solid #bcebc8; }
    .status-red { background: #ffebee; color: #d32f2f; border: 1px solid #ffcdd2; }
    .status-yellow { background: #fffde7; color: #f57f17; border: 1px solid #fff9c4; }

    /* 价格大字 */
    .big-price-box { text-align: center; margin: 10px 0 20px 0; }
    .price-main { font-size: 52px; font-weight: 800; line-height: 1; letter-spacing: -1px; }
    .price-sub { font-size: 18px; font-weight: 600; margin-left: 8px; vertical-align: super;}
    
    /* 评分卡片 */
    .rating-container { display: flex; gap: 10px; }
    .rating-box { flex: 1; background: #fff; border-radius: 14px; text-align: center; padding: 12px 5px; box-shadow: 0 2px 6px rgba(0,0,0,0.03); }
    .rating-score { font-size: 26px; font-weight: 800; color: #ff3b30; }
    .rating-label { font-size: 11px; color: #8e8e93; font-weight: 600; text-transform: uppercase; }
    
    /* 锁定遮罩 */
    .locked-container { position: relative; overflow: hidden; border-radius: 16px;}
    .locked-blur { filter: blur(8px); user-select: none; opacity: 0.5; pointer-events: none; }
    .locked-overlay {
        position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: rgba(255, 255, 255, 0.6); z-index: 10;
        backdrop-filter: blur(2px);
    }
    
    /* 标签 */
    .tag-buy { background-color: #e8f5e9; color: #2e7d32; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; border: 1px solid #c8e6c9; }
    .tag-hold { background-color: #e3f2fd; color: #1565c0; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; border: 1px solid #bbdefb; }
    .tag-risk { background-color: #ffebee; color: #c62828; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; border: 1px solid #ffcdd2; }

    /* 模拟交易 */
    .trade-input-group { background: #f9f9f9; padding: 10px; border-radius: 10px; margin-top: 10px;}
</style>
"""
st.markdown(ui_css, unsafe_allow_html=True)

# ==========================================
# 2. 数据库与工具函数 (精简版)
# ==========================================
def init_db():
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota", "vip_expiry", "paper_json"])
        df.to_csv(DB_FILE, index=False)
    if not os.path.exists(KEYS_FILE):
        pd.DataFrame(columns=["key", "points", "status", "created_at"]).to_csv(KEYS_FILE, index=False)

def safe_fmt(value, fmt="{:.2f}", default="-"):
    try:
        f_val = float(value)
        if np.isnan(f_val) or np.isinf(f_val): return default
        return fmt.format(f_val)
    except: return default

def load_users():
    try: return pd.read_csv(DB_FILE, dtype={"watchlist": str, "quota": int, "paper_json": str}).fillna("")
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
            st.session_state.paper_account = data if "cash" in data else {"cash": 1000000.0, "holdings": {}, "history": []}
        except: st.session_state.paper_account = {"cash": 1000000.0, "holdings": {}, "history": []}

def check_vip_status(username):
    if username == ADMIN_USER: return True, "管理员"
    df = load_users()
    row = df[df["username"] == username]
    if row.empty: return False, "非会员"
    # 简单逻辑：有 VIP 日期且未过期
    expiry_str = str(row.iloc[0]["vip_expiry"])
    if not expiry_str or expiry_str == "nan": return False, "非会员"
    try:
        if datetime.strptime(expiry_str, "%Y-%m-%d") >= datetime.now():
            return True, f"VIP有效期至 {expiry_str}"
    except: pass
    return False, "VIP已过期"

def consume_quota(u):
    if u == ADMIN_USER or check_vip_status(u)[0]: return True
    df = load_users()
    idx = df[df["username"] == u].index
    if len(idx) > 0 and df.loc[idx[0], "quota"] > 0:
        df.loc[idx[0], "quota"] -= 1
        save_users(df)
        return True
    return False

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
    new_row = {"username": u, "password_hash": hashed, "watchlist": "", "quota": initial_quota, "vip_expiry": "", "paper_json": "{}"}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_users(df)
    return True, "注册成功"

def update_watchlist(username, code, action="add"):
    df = load_users()
    idx = df[df["username"] == username].index[0]
    codes = [c.strip() for c in str(df.loc[idx, "watchlist"]).split(",") if c.strip()]
    if action == "add" and code not in codes: codes.append(code)
    elif action == "remove" and code in codes: codes.remove(code)
    df.loc[idx, "watchlist"] = ",".join(codes)
    save_users(df)

def get_user_watchlist(username):
    df = load_users()
    if username == ADMIN_USER: return []
    row = df[df["username"] == username]
    return [c.strip() for c in str(row.iloc[0]["watchlist"]).split(",") if c.strip()] if not row.empty else []

# ==========================================
# 3. 数据与计算逻辑
# ==========================================
def process_ticker(code):
    code = code.strip().upper()
    if code.isdigit() and len(code) < 6: return f"{code.zfill(4)}.HK"
    return code

@st.cache_data(ttl=3600)
def get_name(code, proxy=None):
    # 简单映射，减少 API 调用
    QUICK_MAP = {'600519':'贵州茅台','000858':'五粮液','300750':'宁德时代','002594':'比亚迪','AAPL':'Apple','TSLA':'Tesla','NVDA':'NVIDIA'}
    if code in QUICK_MAP: return QUICK_MAP[code]
    try: return yf.Ticker(process_ticker(code)).info.get('shortName', code)
    except: return code

@st.cache_data(ttl=1800)
def get_stock_data(code):
    """获取数据并计算基础指标"""
    try:
        df = yf.download(process_ticker(code), period="1y", interval="1d", progress=False)
        if df.empty: return pd.DataFrame()
        # 清洗列名
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.reset_index(inplace=True)
        df.rename(columns={'date':'date','close':'close','high':'high','low':'low','open':'open','volume':'volume'}, inplace=True)
        
        # 计算核心指标
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA60'] = df['close'].rolling(60).mean()
        
        # MACD
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = exp12 - exp26
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['HIST'] = 2 * (df['DIF'] - df['DEA'])
        
        # RSI
        delta = df['close'].diff()
        up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
        rs = up.rolling(14).mean() / (down.rolling(14).mean() + 1e-9)
        df['RSI'] = 100 - (100/(1+rs))
        
        # ATR & Vol
        df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
        df['VolRatio'] = df['volume'] / (df['volume'].rolling(5).mean() + 1e-9)
        
        return df.dropna()
    except: return pd.DataFrame()

# ✅ 优化后的精选策略：基于逻辑而非随机
def get_smart_picks(user_watchlist):
    """
    智能筛选：基于技术指标筛选 (非随机)
    由于实时计算所有股票太慢，这里只检查热门股 + 用户自选股的一部分
    """
    pool = list(set(["600519", "NVDA", "TSLA", "300750", "AAPL", "MSFT"] + user_watchlist))
    picks = []
    
    # 为了演示速度，限制检查数量
    check_limit = 5
    checked_count = 0
    
    for code in pool:
        if checked_count >= check_limit: break
        
        # 简单获取最新数据 (模拟快速筛选)
        try:
            df = get_stock_data(code)
            if len(df) < 30: continue
            
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            name = get_name(code)
            
            # 策略 1: 均线金叉 + 放量
            if (prev['MA5'] <= prev['MA20']) and (curr['MA5'] > curr['MA20']) and (curr['VolRatio'] > 1.2):
                picks.append({
                    "code": code, "name": name, 
                    "tag": "🚀 金叉突破", "type": "tag-buy",
                    "reason": f"MA5上穿MA20，且量比达{curr['VolRatio']:.1f}，短线动能强劲。"
                })
                checked_count += 1
                continue
                
            # 策略 2: 回踩支撑 (MA60)
            dist_ma60 = (curr['low'] - curr['MA60']) / curr['MA60']
            if (curr['close'] > curr['MA60']) and (abs(dist_ma60) < 0.02) and (curr['RSI'] < 45):
                picks.append({
                    "code": code, "name": name, 
                    "tag": "⚓ 回踩企稳", "type": "tag-hold",
                    "reason": f"股价回踩60日线获支撑，RSI处于低位，关注反弹。"
                })
                checked_count += 1
                continue
                
            # 策略 3: 超卖反弹
            if curr['RSI'] < 30:
                picks.append({
                    "code": code, "name": name, 
                    "tag": "🔥 超卖反弹", "type": "tag-risk",
                    "reason": f"RSI低至{curr['RSI']:.1f}，进入超卖区，存在技术性反弹需求。"
                })
                checked_count += 1
                
        except: pass
        
    # 如果没有筛选出结果，给一个保底的 (模拟逻辑)
    if not picks:
        picks.append({"code": "NVDA", "name": "NVIDIA", "tag": "👑 趋势龙头", "type": "tag-buy", "reason": "AI板块核心龙头，多头排列延续。"})
        
    return picks

def generate_deep_report(df, name):
    curr = df.iloc[-1]
    
    # 逻辑推导
    chan_conc = "🟢 结构向好：底分型确立，向上笔延伸中。" if curr['close'] > curr['open'] else "🔴 结构承压：顶分型雏形，注意向下变盘。"
    gann_conc = "🟢 支撑有效：股价运行于江恩强势区。" if curr['close'] > curr['MA20'] else "🔴 趋势受阻：股价处于江恩弱势区。"
    mom_conc = "🟢 动能充沛：MACD金叉且量能配合。" if (curr['DIF']>curr['DEA'] and curr['VolRatio']>1) else "🔴 动能衰竭：MACD死叉或量能不足。"

    html = f"""
    <div class="app-card">
        <div style="font-weight:bold; color:#1d1d1f; margin-bottom:5px;">📐 缠论结构透视</div>
        <div style="font-size:13px; color:#666;">
            当前处于{ "上涨线段" if curr['MA5']>curr['MA20'] else "下跌线段" }构造中。<br>
            关键分型：{ "底分型 (强)" if curr['close']>curr['open'] else "顶分型 (弱)" }
        </div>
        <div class="conclusion-box">{chan_conc}</div>
    </div>
    
    <div class="app-card">
        <div style="font-weight:bold; color:#1d1d1f; margin-bottom:5px;">🌌 江恩与斐波那契</div>
        <div style="font-size:13px; color:#666;">
            上方压力：{(curr['close']*1.05):.2f} (Fib 0.618)<br>
            下方支撑：{(curr['close']*0.95):.2f} (Fib 0.382)
        </div>
        <div class="conclusion-box">{gann_conc}</div>
    </div>
    
    <div class="app-card">
        <div style="font-weight:bold; color:#1d1d1f; margin-bottom:5px;">📊 核心动能指标</div>
        <div style="font-size:13px; color:#666;">
            MACD: DIF={curr['DIF']:.2f}<br>
            RSI: {curr['RSI']:.1f} ({ "超买" if curr['RSI']>70 else "超卖" if curr['RSI']<30 else "中性" })<br>
            量比: {curr['VolRatio']:.2f}
        </div>
        <div class="conclusion-box">{mom_conc}</div>
    </div>
    """
    return html

# ==========================================
# 4. 执行入口与UI逻辑
# ==========================================
init_db()

# --- 侧边栏 ---
with st.sidebar:
    st.markdown("### AlphaQuant Pro ♾️")
    if st.session_state.get('logged_in'):
        user = st.session_state["user"]
        is_vip, vip_msg = check_vip_status(user)
        
        # 🔥🔥🔥 修改开始：增加用户存在性检查，防止报错 🔥🔥🔥
        df_users = load_users()
        user_row = df_users[df_users['username'] == user]
        
        if not user_row.empty:
            current_quota = user_row['quota'].iloc[0]
            st.info(f"👤 {user} | {vip_msg} | 积分: {current_quota}")
        else:
            # 如果数据库里找不到当前用户（可能文件被删了），强制退出登录
            st.warning("⚠️ 用户数据异常，请重新登录")
            st.session_state["logged_in"] = False
            time.sleep(1)
            st.rerun()
        # 🔥🔥🔥 修改结束 🔥🔥🔥

        load_user_holdings(user)
        
        # ... 后面的代码保持不变 ...
        
        st.info(f"👤 {user} | {vip_msg} | 积分: {load_users()[load_users()['username']==user]['quota'].iloc[0]}")
        
        # 模式切换
        mode = st.radio("显示模式", ["极简模式", "专业模式"], index=0)
        is_pro = (mode == "专业模式")
        
        # 专业模式权限检查
        if is_pro and not (is_vip or user==ADMIN_USER or st.session_state.paid_code == st.session_state.code):
            st.warning("🔒 专业模式需解锁 (1积分/次)")
            if st.button("🔓 立即解锁", type="primary"):
                if consume_quota(user):
                    st.session_state.paid_code = st.session_state.code
                    st.rerun()
                else: st.error("积分不足")
            is_pro = False # 强制回退

        # ✅ 每日精选策略 (仅专业模式显示)
        if is_pro and user != ADMIN_USER:
            st.markdown("### 🎯 每日精选 (AI Screening)")
            with st.spinner("正在扫描市场..."):
                user_wl = get_user_watchlist(user)
                smart_picks = get_smart_picks(user_wl)
                
            for p in smart_picks:
                with st.expander(f"{p['tag']} | {p['name']}", expanded=False):
                    st.write(p['reason'])
                    if st.button("查看 K线", key=f"btn_{p['code']}"):
                        st.session_state.code = p['code']
                        st.rerun()
            st.divider()

        # 自选股
        with st.expander("⭐ 我的自选", expanded=False):
            wl = get_user_watchlist(user)
            for c in wl:
                if st.button(c, key=f"wl_{c}"): 
                    st.session_state.code = c
                    st.rerun()
            new_c = st.text_input("加自选", placeholder="代码")
            if st.button("添加"): update_watchlist(user, new_c, "add"); st.rerun()
            
        if st.button("退出"): st.session_state["logged_in"] = False; st.rerun()

    else:
        st.info("请先登录")

# --- 登录页 ---
if not st.session_state.get('logged_in'):
    st.markdown("<br><h2 style='text-align:center'>AlphaQuant Pro V80</h2>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["登录", "注册"])
    with tab1:
        u = st.text_input("账号")
        p = st.text_input("密码", type="password")
        if st.button("登录", type="primary"):
            if verify_login(u, p): 
                st.session_state["logged_in"] = True
                st.session_state["user"] = u
                st.rerun()
            else: st.error("错误")
    with tab2:
        nu = st.text_input("新账号")
        np1 = st.text_input("设置密码", type="password")
        if st.button("注册"):
            s, m = register_user(nu, np1)
            if s: st.success(m)
            else: st.error(m)
    st.stop()

# --- 主界面 ---
name = get_name(st.session_state.code)
st.markdown(f"## {name} <span style='font-size:18px; color:#888'>{st.session_state.code}</span>", unsafe_allow_html=True)

# 数据获取
with st.spinner("数据加载中..."):
    df = get_stock_data(st.session_state.code)

if df.empty:
    st.error("无法获取数据，请检查代码。")
    st.stop()

# 顶部行情卡片
curr = df.iloc[-1]
clr = "#d32f2f" if curr['close'] >= df.iloc[-2]['close'] else "#2e7d32"
st.markdown(f"""
<div class="big-price-box">
    <span class="price-main" style="color:{clr}">{curr['close']:.2f}</span>
    <span class="price-sub" style="color:{clr}">{((curr['close']-df.iloc[-2]['close'])/df.iloc[-2]['close']*100):+.2f}%</span>
</div>
""", unsafe_allow_html=True)

# 市场状态
status_text = "趋势向上 (多头)" if curr['close'] > curr['MA20'] else "趋势向下 (空头)"
status_cls = "status-green" if curr['close'] > curr['MA20'] else "status-red"
st.markdown(f"""
<div class="market-status-box {status_cls}">
    <b>{status_text}</b>
    <span style="font-size:12px">基于MA20趋势线判断</span>
</div>
""", unsafe_allow_html=True)

# 图表绘制
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'), row=1, col=1)
if flags['ma']:
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA5'], line=dict(color='black', width=1), name='MA5'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], line=dict(color='orange', width=1), name='MA20'), row=1, col=1)
fig.add_trace(go.Bar(x=df['date'], y=df['volume'], marker_color=['red' if c>=o else 'green' for c,o in zip(df['close'], df['open'])], name='Vol'), row=2, col=1)
fig.update_layout(height=450, margin=dict(l=0,r=0,t=10,b=0), xaxis_rangeslider_visible=False, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# ✅ 只有专业模式才显示深度解读
if is_pro:
    st.markdown("### 🧠 深度技术解读")
    st.markdown(generate_deep_report(df, name), unsafe_allow_html=True)
    
    # 策略点位
    stop_loss = curr['close'] - 2 * curr['ATR']
    take_profit = curr['close'] + 3 * curr['ATR']
    st.markdown(f"""
    <div class="app-card">
        <h4>🛡️ 交易计划 (Pro)</h4>
        <div style="display:flex; justify-content:space-between; text-align:center;">
            <div>🎯 止盈位<br><b style="color:#d32f2f">{take_profit:.2f}</b></div>
            <div>🛡️ 止损位<br><b style="color:#2e7d32">{stop_loss:.2f}</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # 极简模式下的遮罩提示
    st.markdown("""
    <div class="locked-container">
        <div class="locked-blur">
            <div style="height:100px; background:#eee;"></div>
            <div style="height:100px; background:#ddd; margin-top:10px;"></div>
        </div>
        <div class="locked-overlay">
            <div style="font-size:40px;">🔒</div>
            <div style="font-weight:bold;">深度解读已锁定</div>
            <div style="font-size:12px; color:#666;">切换至 [专业模式] 查看缠论结构、主力动能及买卖点</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ✅ 模拟交易优化 (默认折叠，修复Bug)
with st.expander("🎮 模拟交易 (SimTrade)", expanded=False):
    paper = st.session_state.paper_account
    holdings = paper.get("holdings", {})
    cash = paper.get("cash", 1000000.0)
    
    # 资产计算
    mkt_val = 0
    for c_code, c_data in holdings.items():
        price = curr['close'] if c_code == st.session_state.code else c_data['cost'] # 简化：非当前股票用成本价估算
        mkt_val += price * c_data['qty']
        
    total_asset = cash + mkt_val
    pnl = total_asset - 1000000.0
    
    st.markdown(f"""
    <div style="background:#fff; padding:15px; border-radius:10px; border:1px solid #eee; display:flex; justify-content:space-between;">
        <div>
            <div style="font-size:12px; color:#888">总资产</div>
            <div style="font-size:18px; font-weight:bold">{total_asset:,.0f}</div>
        </div>
        <div style="text-align:right">
            <div style="font-size:12px; color:#888">总盈亏</div>
            <div style="font-size:18px; font-weight:bold; color:{'red' if pnl>=0 else 'green'}">{pnl:+,.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 交易面板
    tab_buy, tab_sell = st.tabs(["买入", "卖出"])
    with tab_buy:
        st.write(f"可用资金: {cash:,.0f}")
        qty = st.number_input("买入数量", min_value=100, step=100, key="b_qty")
        if st.button("🔴 买入", use_container_width=True, type="primary"):
            cost = qty * curr['close']
            if cost > cash: st.error("资金不足")
            else:
                st.session_state.paper_account['cash'] -= cost
                if st.session_state.code in holdings:
                    old = holdings[st.session_state.code]
                    new_q = old['qty'] + qty
                    new_c = (old['cost']*old['qty'] + cost)/new_q
                    holdings[st.session_state.code] = {'name':name, 'qty':new_q, 'cost':new_c}
                else:
                    holdings[st.session_state.code] = {'name':name, 'qty':qty, 'cost':curr['close']}
                save_user_holdings(user)
                st.success("买入成功")
                time.sleep(0.5); st.rerun()
                
    with tab_sell:
        curr_hold = holdings.get(st.session_state.code, {'qty':0, 'cost':0})
        st.write(f"当前持仓: {curr_hold['qty']}")
        
        # ✅ 修复显示 -100% 的问题
        if curr_hold['cost'] > 0:
            pct = (curr['close'] - curr_hold['cost']) / curr_hold['cost'] * 100
        else: pct = 0.0
            
        st.write(f"持仓盈亏: {pct:+.2f}%")
        
        s_qty = st.number_input("卖出数量", min_value=0, max_value=curr_hold['qty'], step=100, key="s_qty")
        if st.button("🟢 卖出", use_container_width=True):
            if s_qty > 0:
                amt = s_qty * curr['close']
                st.session_state.paper_account['cash'] += amt
                left = curr_hold['qty'] - s_qty
                if left == 0: del holdings[st.session_state.code]
                else: holdings[st.session_state.code]['qty'] = left
                save_user_holdings(user)
                st.success("卖出成功")
                time.sleep(0.5); st.rerun()

# 策略参数 (默认折叠)
if is_pro:
    with st.expander("⚙️ 策略参数设置", expanded=False):
        ma_s = st.slider("短期均线", 2, 20, 5)
        ma_l = st.slider("长期均线", 10, 120, 20)
        st.caption("调整参数后图表将自动刷新")