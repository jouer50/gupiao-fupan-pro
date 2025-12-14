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
    page_title="阿尔法量研 Pro",
    layout="wide",
    page_icon="🔥",
    initial_sidebar_state="auto"
)

# 初始化 Session
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if "code" not in st.session_state: st.session_state.code = "600519"
if "paid_code" not in st.session_state: st.session_state.paid_code = ""

# ✅ 全局变量兜底初始化
ma_s = 5
ma_l = 20
flags = {
    'ma': True, 'boll': True, 'vol': True, 'macd': True, 
    'kdj': True, 'gann': False, 'fib': True, 'chan': True
}

# 核心常量定义
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"
DB_FILE = "users_v61.csv"
KEYS_FILE = "card_keys.csv"

# Optional deps
ts = None
bs = None
try: import tushare as ts
except: pass
try: import baostock as bs
except: pass

# 🔥 V61.1 CSS：侧边栏按钮修复 + 黄色果冻UI
ui_css = """
<style>
    /* 全局背景 */
    .stApp {background-color: #f7f8fa; font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;}
    
    /* ================= 核心修复：侧边栏按钮 ================= */
    /* 1. Header 必须显示，但设为透明 */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        visibility: visible !important;
        pointer-events: none; /* 让点击穿透 Header */
    }
    header[data-testid="stHeader"] > div {
        pointer-events: auto;
    }
    
    /* 2. 隐藏 Header 里的彩条装饰 */
    [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* 3. 隐藏 Deploy 按钮 */
    .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* 4. 强制显示左上角折叠按钮 (黑箭头+白底) */
    [data-testid="stSidebarCollapsedControl"] {
        display: block !important;
        position: fixed !important;
        top: 10px !important;
        left: 10px !important;
        color: #000000 !important;
        background-color: rgba(255,255,255,0.9) !important;
        border-radius: 50%;
        width: 40px; height: 40px;
        padding: 5px;
        z-index: 999999 !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    /* 兼容旧版选择器 */
    [data-testid="collapsedControl"] {
        display: block !important;
        position: fixed !important;
        top: 10px !important;
        left: 10px !important;
        color: #000000 !important;
        background-color: rgba(255,255,255,0.9) !important;
        border-radius: 50%;
        z-index: 999999 !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* 隐藏页脚 */
    footer {display: none !important;}
    
    /* 顶部间距适配 */
    .block-container {padding-top: 3.5rem !important; padding-bottom: 2rem !important; padding-left: 0.8rem; padding-right: 0.8rem;}

    /* ================= 🍋 按钮：果冻黄 (Jelly Gold) ================= */
    div.stButton > button {
        background: linear-gradient(145deg, #ffdb4d 0%, #ffb300 100%); 
        color: #5d4037; /* 深褐文字 */
        border: 2px solid #fff9c4; 
        border-radius: 25px; 
        padding: 0.6rem 1.2rem;
        font-weight: 800;
        font-size: 16px;
        box-shadow: 0 4px 10px rgba(255, 179, 0, 0.4); 
        transition: all 0.2s cubic-bezier(0.25, 0.8, 0.25, 1);
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(255, 179, 0, 0.5);
    }
    div.stButton > button:active { transform: scale(0.96); }
    
    div.stButton > button[kind="secondary"] {
        background: #f0f0f0; color: #666; border: 1px solid #ddd; box-shadow: none;
    }

    /* ================= 卡片容器 ================= */
    .app-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        border: 1px solid #ffffff;
    }
    
    /* 标题栏 */
    .section-header { display: flex; align-items: center; margin-bottom: 12px; margin-top: 8px; }
    .section-title { font-size: 17px; font-weight: 900; color: #333; margin-right: 5px; }
    .vip-badge { 
        background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 99%); 
        color: #d32f2f; font-size: 10px; font-weight: 800; 
        padding: 2px 8px; border-radius: 10px; font-style: italic; 
    }

    /* ================= 股价大字 + 四宫格参数 ================= */
    .big-price-box { text-align: center; margin-bottom: 20px; }
    .price-main { font-size: 48px; font-weight: 900; line-height: 1; letter-spacing: -1.5px; }
    .price-sub { font-size: 16px; font-weight: 600; margin-left: 8px; padding: 2px 6px; border-radius: 4px; }
    
    .param-grid {
        display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 15px;
    }
    .param-item {
        background: #f9fafe; border-radius: 10px; padding: 10px; text-align: center; border: 1px solid #edf2f7;
    }
    .param-val { font-size: 20px; font-weight: 800; color: #2c3e50; }
    .param-lbl { font-size: 12px; color: #95a5a6; margin-top: 2px; }

    /* ================= 综合评级 (三色卡) ================= */
    .rating-container { display: flex; justify-content: space-between; gap: 8px; }
    .rating-box {
        flex: 1; background: #fff; border: 1px solid #f0f0f0; border-radius: 12px;
        text-align: center; padding: 15px 2px; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.02);
    }
    .rating-score { font-size: 28px; font-weight: 900; color: #ff3b30; line-height: 1; margin-bottom: 5px; font-family: 'Arial', sans-serif; }
    .rating-score-sub { font-size: 10px; color: #ccc; font-weight: 400; margin-left: 1px;}
    .rating-label { font-size: 12px; color: #666; font-weight: 500; }
    .score-yellow { color: #ff9800 !important; }

    /* ================= 投资亮点 (标签) ================= */
    .highlight-item { display: flex; align-items: start; margin-bottom: 12px; line-height: 1.5; }
    .tag-box {
        background: #fff5f5; color: #ff3b30; font-size: 11px; font-weight: 700;
        padding: 2px 8px; border-radius: 6px; 
        margin-right: 10px; white-space: nowrap; margin-top: 2px;
    }
    .tag-blue { background: #f0f7ff; color: #2962ff; }
    .tag-text { font-size: 14px; color: #333; text-align: justify; }
    .hl-num { color: #ff3b30; font-weight: 700; padding: 0 2px; }

    /* ================= 策略卡片 ================= */
    .strategy-card {
        background: #fcfcfc; border: 1px solid #eee; border-left: 4px solid #ffca28;
        border-radius: 8px; padding: 15px; margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    .strategy-title { font-size: 18px; font-weight: 800; color: #333; margin-bottom: 10px; }
    .strategy-grid { display: flex; justify-content: space-between; margin-bottom: 10px; }
    .price-point { font-weight: 700; color: #333; font-size: 15px; }
    .support-line { 
        border-top: 1px dashed #eee; margin-top: 10px; padding-top: 10px; 
        font-size: 12px; color: #888; display: flex; justify-content: space-between;
    }
    
    /* 风险雷达 */
    .risk-header { display: flex; justify-content: space-between; font-size: 12px; color: #666; margin-bottom: 5px; }
    .risk-bar-bg { height: 6px; background: #eee; border-radius: 3px; overflow: hidden; }
    .risk-bar-fill { height: 100%; border-radius: 3px; }

    /* 研报小标题 */
    .deep-title { font-size: 15px; font-weight: 700; color: #333; margin-bottom: 8px; border-left: 3px solid #2962ff; padding-left: 8px; }
    .deep-text { font-size: 13px; color: #555; line-height: 1.6; }
    
    /* 侧边栏品牌 */
    .brand-title { font-size: 22px; font-weight: 900; color: #333; margin-bottom: 2px; }
    .brand-slogan { font-size: 12px; color: #999; margin-bottom: 20px; }
    
    /* 覆盖原生 Metric */
    [data-testid="metric-container"] { display: none; }
</style>
"""
st.markdown(ui_css, unsafe_allow_html=True)

# ==========================================
# 2. 数据库与工具
# ==========================================
def init_db():
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"])
        df.to_csv(DB_FILE, index=False)
    if not os.path.exists(KEYS_FILE):
        df_keys = pd.DataFrame(columns=["key", "points", "status", "created_at"])
        df_keys.to_csv(KEYS_FILE, index=False)

def safe_fmt(value, fmt="{:.2f}", default="-", suffix=""):
    try:
        if value is None: return default
        if isinstance(value, (pd.Series, pd.DataFrame)):
            if value.empty: return default
            value = value.iloc[0]
        if isinstance(value, str):
            if value.strip() in ["", "N/A", "nan", "NaN"]: return default
            value = float(value.replace(',', ''))
        f_val = float(value)
        if np.isnan(f_val) or np.isinf(f_val): return default
        return fmt.format(f_val) + suffix
    except: return default

def load_users():
    try: return pd.read_csv(DB_FILE, dtype={"watchlist": str, "quota": int})
    except: return pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"])

def save_users(df): df.to_csv(DB_FILE, index=False)

def load_keys():
    try: return pd.read_csv(KEYS_FILE)
    except: return pd.DataFrame(columns=["key", "points", "status", "created_at"])

def save_keys(df): df.to_csv(KEYS_FILE, index=False)

def batch_generate_keys(points, count):
    df = load_keys()
    new_keys = []
    for _ in range(count):
        suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        key = f"VIP-{points}-{suffix}"
        new_row = {"key": key, "points": points, "status": "unused", "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
        new_keys.append(new_row)
    df = pd.concat([df, pd.DataFrame(new_keys)], ignore_index=True)
    save_keys(df)
    return len(new_keys)

def generate_key(points):
    key = "VIP-" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
    df = load_keys()
    new_row = {"key": key, "points": points, "status": "unused", "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_keys(df)
    return key

def redeem_key(username, key_input):
    df_keys = load_keys()
    match = df_keys[(df_keys["key"] == key_input) & (df_keys["status"] == "unused")]
    if match.empty: return False, "❌ 无效卡密"
    points_to_add = int(match.iloc[0]["points"])
    df_keys.loc[match.index[0], "status"] = f"used_by_{username}"
    save_keys(df_keys)
    df_users = load_users()
    u_idx = df_users[df_users["username"] == username].index[0]
    df_users.loc[u_idx, "quota"] += points_to_add
    save_users(df_users)
    return True, f"✅ 成功充值 {points_to_add} 积分"

def verify_login(u, p):
    if u == ADMIN_USER and p == ADMIN_PASS: return True
    df = load_users()
    row = df[df["username"] == u]
    if row.empty: return False
    try: return bcrypt.checkpw(p.encode(), row.iloc[0]["password_hash"].encode())
    except: return False

def register_user(u, p):
    if u == ADMIN_USER: return False, "保留账号"
    df = load_users()
    if u in df["username"].values: return False, "用户已存在"
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(p.encode(), salt).decode()
    new_row = {"username": u, "password_hash": hashed, "watchlist": "", "quota": 0}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_users(df)
    return True, "注册成功"

def consume_quota(u):
    if u == ADMIN_USER: return True
    df = load_users()
    idx = df[df["username"] == u].index
    if len(idx) > 0 and df.loc[idx[0], "quota"] > 0:
        df.loc[idx[0], "quota"] -= 1
        save_users(df)
        return True
    return False

def update_user_quota(target, new_q):
    df = load_users()
    idx = df[df["username"] == target].index
    if len(idx) > 0:
        df.loc[idx[0], "quota"] = int(new_q)
        save_users(df)
        return True
    return False

def delete_user(target):
    df = load_users()
    df = df[df["username"] != target]
    save_users(df)

def update_watchlist(username, code, action="add"):
    df = load_users()
    idx = df[df["username"] == username].index[0]
    current_wl = str(df.loc[idx, "watchlist"])
    if current_wl == "nan": current_wl = ""
    codes = [c.strip() for c in current_wl.split(",") if c.strip()]
    if action == "add":
        if code not in codes: codes.append(code)
    elif action == "remove":
        if code in codes: codes.remove(code)
    df.loc[idx, "watchlist"] = ",".join(codes)
    save_users(df)
    return ",".join(codes)

def get_user_watchlist(username):
    df = load_users()
    if username == ADMIN_USER: return []
    row = df[df["username"] == username]
    if row.empty: return []
    wl_str = str(row.iloc[0]["watchlist"])
    if wl_str == "nan": return []
    return [c.strip() for c in wl_str.split(",") if c.strip()]

# ==========================================
# 3. 股票逻辑
# ==========================================
def is_cn_stock(code): return code.isdigit() and len(code) == 6
def _to_ts_code(s): return f"{s}.SH" if s.startswith('6') else f"{s}.SZ" if s[0].isdigit() else s
def _to_bs_code(s): return f"sh.{s}" if s.startswith('6') else f"sz.{s}" if s[0].isdigit() else s
def process_ticker(code):
    code = code.strip().upper()
    if code.isdigit() and len(code) < 6: return f"{code.zfill(4)}.HK"
    return code

def generate_mock_data(days=365):
    dates = pd.date_range(end=datetime.today(), periods=days)
    close = [150.0]
    for _ in range(days-1): close.append(max(10, close[-1] + np.random.normal(0.1, 3.0)))
    df = pd.DataFrame({'date': dates, 'close': close})
    df['open'] = df['close'] * np.random.uniform(0.98, 1.02, days)
    df['high'] = df[['open', 'close']].max(axis=1) * np.random.uniform(1.0, 1.03, days)
    df['low'] = df[['open', 'close']].min(axis=1) * np.random.uniform(0.97, 1.0, days)
    df['volume'] = np.random.randint(1000000, 50000000, days)
    df['pct_change'] = df['close'].pct_change() * 100
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    return df

@st.cache_data(ttl=3600)
def get_name(code, token, proxy=None):
    clean_code = code.strip().upper().replace('.SH','').replace('.SZ','').replace('SH','').replace('SZ','')
    QUICK_MAP = {'600519':'贵州茅台','000858':'五粮液','601318':'中国平安','600036':'招商银行','300750':'宁德时代','002594':'比亚迪','601888':'中国中免','600276':'恒瑞医药','601857':'中国石油','601088':'中国神华','601988':'中国银行','601398':'工商银行','AAPL':'Apple','TSLA':'Tesla','NVDA':'NVIDIA','MSFT':'Microsoft','BABA':'Alibaba'}
    if clean_code in QUICK_MAP: return QUICK_MAP[clean_code]
    if is_cn_stock(clean_code) and token and ts:
        try:
            ts.set_token(token); pro = ts.pro_api()
            df = pro.stock_basic(ts_code=_to_ts_code(clean_code), fields='name')
            if not df.empty: return df.iloc[0]['name']
        except: pass
    if is_cn_stock(clean_code) and bs:
        try:
            bs.login(); rs = bs.query_stock_basic(code=_to_bs_code(clean_code))
            if rs.error_code == '0':
                data = rs.get_row_data()
                if len(data)>1: bs.logout(); return data[1]
            bs.logout()
        except: pass
    try: return yf.Ticker(code).info.get('shortName', code)
    except: pass
    return code

def get_data_and_resample(code, token, timeframe, adjust, proxy=None):
    code = process_ticker(code)
    fetch_days = 1500 
    raw_df = pd.DataFrame()
    if not is_cn_stock(code):
        try:
            yf_df = yf.download(code, period="5y", interval="1d", progress=False, auto_adjust=False)
            if not yf_df.empty:
                if isinstance(yf_df.columns, pd.MultiIndex): yf_df.columns = yf_df.columns.get_level_values(0)
                yf_df.columns = [str(c).lower().strip() for c in yf_df.columns]
                yf_df = yf_df.loc[:, ~yf_df.columns.duplicated()]
                yf_df.reset_index(inplace=True)
                rename_map = {}
                for c in yf_df.columns:
                    if 'date' in c: rename_map[c] = 'date'
                    elif 'close' in c: rename_map[c] = 'close'
                    elif 'open' in c: rename_map[c] = 'open'
                    elif 'high' in c: rename_map[c] = 'high'
                    elif 'low' in c: rename_map[c] = 'low'
                    elif 'volume' in c: rename_map[c] = 'volume'
                yf_df.rename(columns=rename_map, inplace=True)
                req_cols = ['date','open','high','low','close']
                if all(c in yf_df.columns for c in req_cols):
                    if 'volume' not in yf_df.columns: yf_df['volume'] = 0
                    raw_df = yf_df[['date','open','high','low','close','volume']].copy()
                    for c in ['open','high','low','close','volume']: raw_df[c] = pd.to_numeric(raw_df[c], errors='coerce')
                    raw_df['pct_change'] = raw_df['close'].pct_change() * 100
        except: pass
    else:
        if token and ts:
            try:
                pro = ts.pro_api(token)
                e = pd.Timestamp.today().strftime('%Y%m%d')
                s = (pd.Timestamp.today() - pd.Timedelta(days=fetch_days)).strftime('%Y%m%d')
                df = pro.daily(ts_code=_to_ts_code(code), start_date=s, end_date=e)
                if not df.empty:
                    if adjust in ['qfq', 'hfq']:
                        adj_f = pro.adj_factor(ts_code=_to_ts_code(code), start_date=s, end_date=e)
                        if not adj_f.empty:
                            adj_f = adj_f.rename(columns={'trade_date':'date','adj_factor':'factor'})
                            df = df.rename(columns={'trade_date':'date'})
                            df = df.merge(adj_f[['date','factor']], on='date', how='left').fillna(method='ffill')
                            f = df['factor']
                            ratio = f/f.iloc[-1] if adjust=='qfq' else f/f.iloc[0]
                            for c in ['open','high','low','close']: df[c] *= ratio
                    df = df.rename(columns={'trade_date':'date','vol':'volume','pct_chg':'pct_change'})
                    df['date'] = pd.to_datetime(df['date'])
                    for c in ['open','high','low','close','volume']: df[c] = pd.to_numeric(df[c], errors='coerce')
                    raw_df = df.sort_values('date').reset_index(drop=True)
            except: pass
        if raw_df.empty and bs:
            try:
                bs.login()
                e = pd.Timestamp.today().strftime('%Y-%m-%d')
                s = (pd.Timestamp.today() - pd.Timedelta(days=fetch_days)).strftime('%Y-%m-%d')
                flag = "2" if adjust=='qfq' else "1" if adjust=='hfq' else "3"
                rs = bs.query_history_k_data_plus(_to_bs_code(code), "date,open,high,low,close,volume,pctChg", start_date=s, end_date=e, frequency="d", adjustflag=flag)
                data = rs.get_data(); bs.logout()
                if not data.empty:
                    df = data.rename(columns={'pctChg':'pct_change'})
                    df['date'] = pd.to_datetime(df['date'])
                    for c in ['open','high','low','close','volume','pct_change']: df[c] = pd.to_numeric(df[c], errors='coerce')
                    raw_df = df.sort_values('date').reset_index(drop=True)
            except: pass
    if raw_df.empty: return raw_df
    if timeframe == '日线': return raw_df
    rule = 'W' if timeframe == '周线' else 'M'
    raw_df.set_index('date', inplace=True)
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    resampled = raw_df.resample(rule).agg(agg).dropna()
    resampled['pct_change'] = resampled['close'].pct_change() * 100
    resampled.reset_index(inplace=True)
    return resampled

@st.cache_data(ttl=3600)
def get_fundamentals(code, token):
    res = {"pe": "-", "pb": "-", "roe": "-", "mv": "-", "target_price": "-", "rating": "-"}
    code = process_ticker(code)
    try:
        t = yf.Ticker(code); i = t.info
        res['pe'] = safe_fmt(i.get('trailingPE'))
        res['pb'] = safe_fmt(i.get('priceToBook'))
        res['mv'] = f"{i.get('marketCap')/100000000:.2f}亿" if i.get('marketCap') else "-"
        if 'targetMeanPrice' in i: res['target_price'] = safe_fmt(i.get('targetMeanPrice'))
        if 'recommendationKey' in i: res['rating'] = i.get('recommendationKey', '').replace('buy','买入').replace('sell','卖出').replace('hold','持有')
    except: pass
    if token and ts and is_cn_stock(code):
        try:
            pro = ts.pro_api(token)
            df = pro.daily_basic(ts_code=_to_ts_code(code), fields='pe_ttm,pb,total_mv')
            if not df.empty:
                r = df.iloc[-1]
                res['pe'] = safe_fmt(r['pe_ttm']); res['pb'] = safe_fmt(r['pb'])
                res['mv'] = f"{r['total_mv']/10000:.1f}亿" if r['total_mv'] else "-"
        except: pass
    return res

def calc_full_indicators(df, ma_s, ma_l):
    if df.empty: return df
    try:
        c = df['close'].squeeze() if isinstance(df['close'], pd.DataFrame) else df['close']
        h = df['high'].squeeze() if isinstance(df['high'], pd.DataFrame) else df['high']
        l = df['low'].squeeze() if isinstance(df['low'], pd.DataFrame) else df['low']
        v = df['volume'].squeeze() if isinstance(df['volume'], pd.DataFrame) else df['volume']
    except: c = df['close']; h = df['high']; l = df['low']; v = df['volume']

    # ✅ 修复：先定义均线
    df['MA_Short'] = c.rolling(ma_s).mean()
    df['MA_Long'] = c.rolling(ma_l).mean()

    p_high = h.rolling(9).max(); p_low = l.rolling(9).min()
    df['Tenkan'] = (p_high + p_low) / 2
    p_high26 = h.rolling(26).max(); p_low26 = l.rolling(26).min()
    df['Kijun'] = (p_high26 + p_low26) / 2
    df['SpanA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['SpanB'] = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    df['SpanA'] = df['SpanA'].fillna(method='bfill').fillna(0)
    df['SpanB'] = df['SpanB'].fillna(method='bfill').fillna(0)

    mid = c.rolling(20).mean(); std = c.rolling(20).std()
    df['Upper'] = mid + 2*std; df['Lower'] = mid - 2*std
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    df['DIF'] = e12 - e26; df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean(); df['HIST'] = 2 * (df['DIF'] - df['DEA'])
    delta = c.diff(); up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
    rs = up.rolling(14).mean()/(down.rolling(14).mean()+1e-9)
    df['RSI'] = 100 - (100/(1+rs))
    low9 = l.rolling(9).min(); high9 = h.rolling(9).max()
    rsv = (c - low9)/(high9 - low9 + 1e-9) * 100
    df['K'] = rsv.ewm(com=2).mean(); df['D'] = df['K'].ewm(com=2).mean(); df['J'] = 3 * df['K'] - 2 * df['D']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()
    dm_p = np.where((h.diff() > l.diff().abs()) & (h.diff()>0), h.diff(), 0)
    dm_m = np.where((l.diff().abs() > h.diff()) & (l.diff()<0), l.diff().abs(), 0)
    di_plus = 100 * pd.Series(dm_p).rolling(14).sum() / (tr.rolling(14).sum()+1e-9)
    di_minus = 100 * pd.Series(dm_m).rolling(14).sum() / (tr.rolling(14).sum()+1e-9)
    df['ADX'] = (abs(di_plus - di_minus)/(di_plus + di_minus + 1e-9) * 100).rolling(14).mean()
    df['VolRatio'] = v / (v.rolling(5).mean() + 1e-9)
    df[['K','D','J','DIF','DEA','HIST','RSI','ADX']] = df[['K','D','J','DIF','DEA','HIST','RSI','ADX']].fillna(50)
    return df

def detect_patterns(df):
    df['F_Top'] = (df['high'].shift(1)<df['high']) & (df['high'].shift(-1)<df['high'])
    df['F_Bot'] = (df['low'].shift(1)>df['low']) & (df['low'].shift(-1)>df['low'])
    return df

def get_drawing_lines(df):
    idx = df['low'].tail(60).idxmin()
    if pd.isna(idx): return {}, {}
    sd = df.loc[idx, 'date']; sp = df.loc[idx, 'low']
    days = (df['date'] - sd).dt.days
    step = df['ATR14'].iloc[-1]*0.5 if df['ATR14'].iloc[-1]>0 else sp*0.01
    gann = {k: sp + days*step*r for k,r in [('1x1',1),('1x2',0.5),('2x1',2)]}
    rec = df.tail(120)
    h = rec['high'].max(); l = rec['low'].min(); d = h-l
    fib = {'0.236': h-d*0.236, '0.382': h-d*0.382, '0.5': h-d*0.5, '0.618': h-d*0.618}
    return gann, fib

# ==========================================
# 4. 回测逻辑优化 (重点修改)
# ==========================================
def run_backtest(df):
    if df is None or len(df) < 50: return 0.0, 0.0, 0.0, 0.0, [], [], pd.DataFrame({'date':[], 'equity':[]})
    needed = ['MA_Short', 'MA_Long', 'close', 'date']
    if not all(c in df.columns for c in needed): return 0.0, 0.0, 0.0, 0.0, [], [], pd.DataFrame({'date':[], 'equity':[]})
    df_bt = df.dropna(subset=needed).reset_index(drop=True)
    if len(df_bt) < 20: return 0.0, 0.0, 0.0, 0.0, [], [], pd.DataFrame({'date':[], 'equity':[]})

    capital = 100000; position = 0
    buy_signals = []; sell_signals = []; equity = [capital]; dates = [df_bt.iloc[0]['date']]
    
    # 记录 Benchmark (买入持有策略)
    start_price = df_bt.iloc[0]['close']
    
    for i in range(1, len(df_bt)):
        curr = df_bt.iloc[i]; prev = df_bt.iloc[i-1]; price = curr['close']; date = curr['date']
        if prev['MA_Short'] <= prev['MA_Long'] and curr['MA_Short'] > curr['MA_Long'] and position == 0:
            position = capital / price; capital = 0; buy_signals.append(date)
        elif prev['MA_Short'] >= prev['MA_Long'] and curr['MA_Short'] < curr['MA_Long'] and position > 0:
            capital = position * price; position = 0; sell_signals.append(date)
        current_val = capital + (position * price)
        equity.append(current_val)
        dates.append(date)
        
    final = equity[-1]; ret = (final - 100000) / 100000 * 100
    
    # 计算 Alpha
    end_price = df_bt.iloc[-1]['close']
    bench_ret = (end_price - start_price) / start_price * 100
    alpha = ret - bench_ret
    
    win_rate = 50 + (ret / 10); win_rate = max(10, min(90, win_rate))
    eq_series = pd.Series(equity)
    cummax = eq_series.cummax()
    drawdown = (eq_series - cummax) / cummax
    max_dd = drawdown.min() * 100
    eq_df = pd.DataFrame({'date': dates, 'equity': equity})
    
    return ret, alpha, win_rate, max_dd, buy_signals, sell_signals, eq_df

def generate_deep_report(df, name):
    curr = df.iloc[-1]
    chan_trend = "底分型构造中" if curr['F_Bot'] else "顶分型构造中" if curr['F_Top'] else "中继形态"
    gann, fib = get_drawing_lines(df)
    try:
        fib_near = min(fib.items(), key=lambda x: abs(x[1]-curr['close']))
        fib_txt = f"股价正逼近斐波那契 <b>{fib_near[0]}</b> 关键位 ({fib_near[1]:.2f})。"
    except: fib_txt = "数据不足，无法计算位置。"
    macd_state = "金叉共振" if curr['DIF']>curr['DEA'] else "死叉调整"
    vol_state = "放量" if curr['VolRatio']>1.2 else "缩量" if curr['VolRatio']<0.8 else "温和"

    html = f"""
    <div class="app-card">
        <div class="deep-title">📐 缠论结构与形态学</div>
        <div class="deep-text">
            • <b>分型状态</b>：{chan_trend}。顶分型通常是短期压力的标志。<br>
            • <b>笔的延伸</b>：当前价格处于一笔走势的{ "延续阶段" if not (curr['F_Top'] or curr['F_Bot']) else "转折关口" }。
        </div>
    </div>
    <div class="app-card">
        <div class="deep-title">🌌 江恩与斐波那契</div>
        <div class="deep-text">
            • 江恩角度线 1x1线是多空分界线。<br>
            • <b>斐波那契回撤</b>：{fib_txt}
        </div>
    </div>
    <div class="app-card">
        <div class="deep-title">📊 核心动能指标</div>
        <div class="deep-text">
            • <b>MACD</b>：当前 {macd_state}。DIF={safe_fmt(curr['DIF'])}, DEA={safe_fmt(curr['DEA'])}<br>
            • <b>BOLL</b>：股价运行于 { "中轨上方" if curr['close']>curr['MA_Long'] else "中轨下方" }。<br>
            • <b>VOL量能</b>：今日 {vol_state} (量比 {safe_fmt(curr['VolRatio'])})
        </div>
    </div>
    """
    return html

def analyze_score(df):
    c = df.iloc[-1]; score=0; reasons=[]
    if c['MA_Short']>c['MA_Long']: score+=2; reasons.append("均线金叉")
    else: score-=2
    if c['close']>c['MA_Long']: score+=1; reasons.append("站上长期均线")
    if c['DIF']>c['DEA']: score+=1; reasons.append("MACD多头")
    if c['RSI']<20: score+=2; reasons.append("RSI超卖")
    if c['VolRatio']>1.5: score+=1; reasons.append("放量攻击")
    
    action = "积极买入" if score>=4 else "持有/观望" if score>=0 else "减仓/卖出"
    color = "success" if score>=4 else "warning" if score>=0 else "error"
    if score >= 4: pos_txt = "80% (重仓)"
    elif score >= 1: pos_txt = "50% (中仓)"
    elif score >= -2: pos_txt = "20% (底仓)"
    else: pos_txt = "0% (空仓)"
    
    atr = c['ATR14']
    stop_loss = c['close'] - 2*atr
    take_profit = c['close'] + 3*atr
    support = df['low'].iloc[-20:].min()
    resistance = df['high'].iloc[-20:].max()
    
    return score, action, color, stop_loss, take_profit, pos_txt, support, resistance

def main_uptrend_check(df):
    curr = df.iloc[-1]
    is_bull = curr['MA_Short'] > curr['MA_Long']
    is_cloud = curr['close'] > max(curr['SpanA'], curr['SpanB'])
    if is_bull and is_cloud and curr['ADX'] > 20: return "🚀 主升浪 (Strong Up)", "success"
    if is_cloud: return "📈 震荡上行 (Trending)", "warning"
    return "📉 主跌浪 (Downtrend)", "error"

def calculate_risk_percentile(df):
    if df is None or df.empty: return 0, False
    curr = df.iloc[-1]['close']
    low = df['close'].min(); high = df['close'].max()
    if high == low: return 0, False
    pct = (curr - low) / (high - low) * 100
    return round(pct, 1), pct > 85

def calculate_smart_score(df, funda):
    trend_score = 5
    last = df.iloc[-1]
    if last['close'] > last['MA_Long']: trend_score += 2
    if last['DIF'] > last['DEA']: trend_score += 1
    if last['RSI'] > 50: trend_score += 1
    if last['MA_Short'] > last['MA_Long']: trend_score += 1
    trend_score = min(10, trend_score)
    val_score = 5
    try:
        pe = float(funda['pe'])
        if pe < 15: val_score += 3
        elif pe < 30: val_score += 1
        elif pe > 60: val_score -= 2
    except: pass
    val_score = min(10, max(1, val_score))
    qual_score = 6
    try:
        mv_str = str(funda['mv']).replace('亿','')
        mv = float(mv_str)
        if mv > 1000: qual_score += 2
        elif mv > 100: qual_score += 1
    except: pass
    volatility = df['pct_change'].std()
    if volatility < 2: qual_score += 1
    qual_score = min(10, qual_score)
    return round(qual_score, 1), round(val_score, 1), round(trend_score, 1)

def get_smart_highlights(df, funda, price_pct, is_high_risk):
    last = df.iloc[-1]
    highlights = []
    
    if funda.get('rating') and funda.get('rating') != '-':
        highlights.append(("评级", "tag-blue", f"机构综合评级为 <span class='hl-num'>{funda['rating']}</span>。"))
    if funda.get('target_price') and funda.get('target_price') != '-':
        try:
            target = float(funda['target_price'])
            curr = last['close']
            upside = (target - curr) / curr * 100
            if upside > 0: highlights.append(("目标", "tag-blue", f"机构目标均价 <span class='hl-num'>{target}</span>，潜在空间 <span class='hl-num'>{upside:.1f}%</span>。"))
        except: pass

    if is_high_risk:
        highlights.append(("风险", "tag-blue", f"当前价格处于近10年 <span class='hl-num'>{price_pct}%</span> 高位，注意回调！"))
    elif price_pct < 15:
        highlights.append(("机会", "tag-red", f"当前价格处于近10年 <span class='hl-num'>{price_pct}%</span> 低位，安全边际高。"))
    
    try:
        pe = float(funda['pe'])
        if pe > 0 and pe < 20: highlights.append(("低估", "tag-red", f"当前PE为 <span class='hl-num'>{pe}</span>，处于低估区间。"))
        elif pe > 60: highlights.append(("泡沫", "tag-blue", f"当前PE高达 <span class='hl-num'>{pe}</span>，存在估值泡沫。"))
    except: pass
    
    if last['MA_Short'] > last['MA_Long']: highlights.append(("趋势", "tag-red", "均线呈多头排列，短期趋势向上。"))
    else: highlights.append(("趋势", "tag-blue", "均线呈空头排列，短期趋势向下。"))
    
    if last['VolRatio'] > 2: highlights.append(("放量", "tag-red", "今日成交量放大2倍以上，主力资金异动明显。"))
    
    if not highlights: highlights.append(("平稳", "tag-blue", "近期股价波动较小，处于横盘整理阶段。"))
    
    return highlights

def plot_chart(df, name, flags, ma_s, ma_l):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.55,0.1,0.15,0.2])
    fig.update_layout(dragmode=False, margin=dict(l=10, r=10, t=10, b=10))
    
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线', increasing_line_color='#FF3B30', decreasing_line_color='#34C759'), 1, 1)
    
    if flags.get('ma'):
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA_Short'], name=f'MA{ma_s}', line=dict(width=1.2, color='#333333')), 1, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA_Long'], name=f'MA{ma_l}', line=dict(width=1.2, color='#ffcc00')), 1, 1)
            
    if flags.get('boll'):
        fig.add_trace(go.Scatter(x=df['date'], y=df['Upper'], line=dict(width=1, dash='dash', color='rgba(33, 150, 243, 0.3)'), name='Upper'), 1, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['Lower'], line=dict(width=1, dash='dash', color='rgba(33, 150, 243, 0.3)'), name='Lower', fill='tonexty', fillcolor='rgba(33, 150, 243, 0.05)'), 1, 1)
    
    ga, fi = get_drawing_lines(df)
    if flags.get('gann'):
        for k,v in ga.items(): fig.add_trace(go.Scatter(x=df['date'], y=v, mode='lines', line=dict(width=0.8, dash='dot', color='rgba(128,128,128,0.3)'), name=f'Gann {k}', showlegend=False), 1, 1)
    if flags.get('fib'):
        for k,v in fi.items(): fig.add_hline(y=v, line_dash='dash', line_color='#ff9800', row=1, col=1)
    if flags.get('chan'):
        tops=df[df['F_Top']]; bots=df[df['F_Bot']]
        fig.add_trace(go.Scatter(x=tops['date'], y=tops['high'], mode='markers', marker_symbol='triangle-down', marker_color='#34C759', name='顶分型'), 1, 1)
        fig.add_trace(go.Scatter(x=bots['date'], y=bots['low'], mode='markers', marker_symbol='triangle-up', marker_color='#FF3B30', name='底分型'), 1, 1)

    colors = ['#FF3B30' if c<o else '#34C759' for c,o in zip(df['close'], df['open'])]
    if flags.get('vol'): fig.add_trace(go.Bar(x=df['date'], y=df['volume'], marker_color=colors, name='Vol'), 2, 1)
    if flags.get('macd'):
        fig.add_trace(go.Bar(x=df['date'], y=df['HIST'], marker_color=colors, name='MACD'), 3, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['DIF'], line=dict(color='#0071e3', width=1), name='DIF'), 3, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['DEA'], line=dict(color='#ff9800', width=1), name='DEA'), 3, 1)
    if flags.get('kdj'):
        fig.add_trace(go.Scatter(x=df['date'], y=df['K'], line=dict(color='#0071e3', width=1), name='K'), 4, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['D'], line=dict(color='#ff9800', width=1), name='D'), 4, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['J'], line=dict(color='#af52de', width=1), name='J'), 4, 1)
    
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#1d1d1f'), xaxis=dict(showgrid=False, showline=True, linecolor='#e5e5e5'), yaxis=dict(showgrid=True, gridcolor='#f5f5f5'), legend=dict(orientation="h", y=-0.05))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. 执行入口 (Logic)
# ==========================================
init_db()

with st.sidebar:
    st.markdown("""
    <div style='text-align: left; margin-bottom: 20px;'>
        <div class='brand-title'>阿尔法量研 <span style='color:#0071e3'>Pro</span></div>
        <div class='brand-en'>AlphaQuant Pro</div>
        <div class='brand-slogan'>用历史验证未来，用数据构建策略。</div>
    </div>
    """, unsafe_allow_html=True)
    
    new_c = st.text_input("🔍 股票代码", st.session_state.code)
    if new_c != st.session_state.code: st.session_state.code = new_c; st.session_state.paid_code = ""; st.rerun()

    if st.session_state.get('logged_in'):
        user = st.session_state["user"]
        is_admin = (user == ADMIN_USER)
        
        if not is_admin:
            with st.expander("⭐ 我的自选股", expanded=False):
                current_wl = get_user_watchlist(user)
                if not current_wl: st.caption("暂无自选，请在上方添加")
                else:
                    for c in current_wl:
                        c1, c2 = st.columns([3, 1])
                        if c1.button(f"{c}", key=f"wl_{c}"):
                            st.session_state.code = c
                            st.session_state.paid_code = ""
                            st.rerun()
                        if c2.button("✖️", key=f"del_{c}"):
                            update_watchlist(user, c, "remove")
                            st.rerun()
            if st.button("❤️ 加入自选"): update_watchlist(user, st.session_state.code, "add"); st.rerun()

        if st.button("🔄 刷新缓存"): st.cache_data.clear(); st.rerun()

        if not is_admin:
            with st.expander("💎 充值中心", expanded=False):
                st.info(f"当前积分: {load_users()[load_users()['username']==user]['quota'].iloc[0]}")
                st.write("##### 1. 选择充值套餐")
                pay_opt = st.radio("充值面额", [20, 50, 100], horizontal=True, format_func=lambda x: f"￥{x}")
                
                st.write("##### 2. 扫码支付")
                if os.path.exists("alipay.png"):
                    st.image("alipay.png", caption="请使用支付宝扫码", width=200)
                else:
                    st.warning("请上传 alipay.png")
                    
                st.write("##### 3. 获取卡密")
                if st.button("✅ 我已支付，自动发货"):
                    new_key = generate_key(pay_opt)
                    st.success("支付成功！您的卡密如下：")
                    st.code(new_key, language="text")
                
                st.write("##### 4. 兑换")
                k_in = st.text_input("输入卡密")
                if st.button("兑换"):
                    s, m = redeem_key(user, k_in)
                    if s: st.success(m); time.sleep(1); st.rerun()
                    else: st.error(m)

        if is_admin:
            st.success("👑 管理员模式")
            with st.expander("💳 卡密生成", expanded=True):
                points_gen = st.selectbox("面值", [20, 50, 100, 200, 500])
                count_gen = st.number_input("数量", 1, 50, 10)
                if st.button("批量生成"):
                    num = batch_generate_keys(points_gen, count_gen)
                    st.success(f"已生成 {num} 张卡密")
            with st.expander("用户管理"):
                df_u = load_users()
                st.dataframe(df_u[["username","quota"]], hide_index=True)
                csv = df_u.to_csv(index=False).encode('utf-8')
                st.download_button("备份数据", csv, "backup.csv", "text/csv")
                uploaded_file = st.file_uploader("恢复用户数据", type="csv", key="restore_users")
                if uploaded_file is not None:
                    try:
                        df_restore = pd.read_csv(uploaded_file)
                        required = ["username", "password_hash", "watchlist", "quota"]
                        if all(col in df_restore.columns for col in required):
                            df_restore.to_csv(DB_FILE, index=False)
                            st.success("✅ 恢复成功！")
                            time.sleep(1); st.rerun()
                        else: st.error("❌ 格式错误")
                    except Exception as e: st.error(f"❌ 失败: {e}")
                
                u_list = [x for x in df_u["username"] if x!=ADMIN_USER]
                if u_list:
                    target = st.selectbox("选择用户", u_list)
                    val = st.number_input("新积分", value=0, step=10)
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("更新"): update_user_quota(target, val); st.success("OK"); time.sleep(0.5); st.rerun()
                    with c2:
                        chk = st.checkbox("确认删除")
                        if st.button("删除") and chk: delete_user(target); st.success("Del"); time.sleep(0.5); st.rerun()
            with st.expander("卡密管理"):
                df_k = load_keys()
                show_all = st.checkbox("显示已使用", False)
                if not show_all: display_df = df_k[df_k['status'] == 'unused']
                else: display_df = df_k
                st.dataframe(display_df, hide_index=True, use_container_width=True)
                if st.button("🗑️ 清理已用卡密"):
                    clean_df = df_k[df_k['status'] == 'unused']
                    save_keys(clean_df)
                    st.success("已清理！")
                    time.sleep(1); st.rerun()

        timeframe = st.selectbox("周期", ["日线", "周线", "月线"])
        days = st.radio("范围", [30,60,120,250], 2, horizontal=True)
        adjust = st.selectbox("复权", ["qfq","hfq",""], 0)
        
        st.divider()
        with st.expander("🎛️ 策略参数", expanded=False):
            st.caption("调整均线参数，优化回测结果")
            ma_s = st.slider("短期均线", 2, 20, 5)
            ma_l = st.slider("长期均线", 10, 120, 20)
        
        st.markdown("### 🛠️ 指标开关")
        c_flags = st.columns(2)
        with c_flags[0]:
            flags['ma'] = st.checkbox("MA", True)
            flags['boll'] = st.checkbox("BOLL", True)
            flags['vol'] = st.checkbox("VOL", True)
            flags['macd'] = st.checkbox("MACD", True)
        with c_flags[1]:
            flags['kdj'] = st.checkbox("KDJ", True)
            flags['gann'] = st.checkbox("江恩", False)
            flags['fib'] = st.checkbox("斐波那契", True)
            flags['chan'] = st.checkbox("缠论", True)
        st.divider()
        st.caption("免责声明：本系统仅供量化研究，不构成投资建议。")
        if st.button("退出登录"): st.session_state["logged_in"]=False; st.rerun()
    else:
        st.info("请先登录系统")

# 登录逻辑
if not st.session_state.get('logged_in'):
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.markdown("""
        <br><br>
        <div style='text-align: center;'>
            <h1 class='brand-title'>阿尔法量研回测系统 Pro</h1>
            <div class='brand-en'>AlphaQuant Pro</div>
        </div>
        """, unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔑 登录", "📝 注册"])
        with tab1:
            u = st.text_input("账号")
            p = st.text_input("密码", type="password")
            if st.button("登录系统"):
                if verify_login(u.strip(), p): st.session_state["logged_in"] = True; st.session_state["user"] = u.strip(); st.session_state["paid_code"] = ""; st.rerun()
                else: st.error("账号或密码错误")
        with tab2:
            nu = st.text_input("新用户")
            np1 = st.text_input("设置密码", type="password")
            if st.button("立即注册"):
                suc, msg = register_user(nu.strip(), np1)
                if suc: st.success(msg)
                else: st.error(msg)
    st.stop()

# --- 主内容区 ---
name = get_name(st.session_state.code, "", None) 
c1, c2 = st.columns([3, 1])
with c1: st.title(f"📈 {name} ({st.session_state.code})")

# 付费墙 & 演示模式
is_demo = False
if st.session_state.code != st.session_state.paid_code:
    df_u = load_users()
    try: bal = df_u[df_u["username"]==user]["quota"].iloc[0]
    except: bal = 0
    if bal > 0:
        st.info(f"🔒 深度研报需解锁 (余额: {bal})")
        if st.button("🔓 支付 1 积分查看", type="primary"):
            if consume_quota(user): st.session_state.paid_code = st.session_state.code; st.rerun()
            else: st.error("扣费失败")
        st.stop()
    else:
        st.warning("👀 积分不足，已进入【演示模式】 (数据为模拟)")
        is_demo = True
        df = generate_mock_data(days)

if not is_demo:
    loading_tips = ["正在加载因子库…", "正在构建回测引擎…", "正在初始化模型框架…", "正在同步行情数据…"]
    with st.spinner(random.choice(loading_tips)):
        df = get_data_and_resample(st.session_state.code, "", timeframe, adjust, proxy=None)
        if df.empty:
            st.warning("⚠️ 暂无数据 (可能因网络原因)。自动切换至演示模式。")
            df = generate_mock_data(days)
            is_demo = True

try:
    funda = get_fundamentals(st.session_state.code, "")
    df = calc_full_indicators(df, ma_s,