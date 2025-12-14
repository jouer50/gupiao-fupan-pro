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
    page_title="阿尔法量研 Pro V68 (商业版)",
    layout="wide",
    page_icon="🔥",
    initial_sidebar_state="expanded"
)

# 初始化 Session
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if "code" not in st.session_state: st.session_state.code = "600519"
if "paid_code" not in st.session_state: st.session_state.paid_code = ""

# ✅ NEW: 模拟交易 Session 初始化
if "paper_holdings" not in st.session_state: st.session_state.paper_holdings = {}

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
DB_FILE = "users_v68.csv" # 升级文件名以示区分
KEYS_FILE = "card_keys.csv"

# Optional deps
ts = None
bs = None
try: import tushare as ts
except: pass
try: import baostock as bs
except: pass

# 🔥 UI CSS (保持原有果冻风格)
ui_css = """
<style>
    /* 全局背景 */
    .stApp {background-color: #f7f8fa; font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;}
    
    header[data-testid="stHeader"] { background-color: transparent !important; pointer-events: none; }
    header[data-testid="stHeader"] > div { pointer-events: auto; }
    [data-testid="stDecoration"] { display: none !important; }
    .stDeployButton { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] {
        display: block !important; position: fixed !important; top: 10px !important; left: 10px !important;
        color: #000; background-color: rgba(255,255,255,0.9) !important; border-radius: 50%;
        width: 40px; height: 40px; padding: 5px; z-index: 999999 !important; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    footer {display: none !important;}
    .block-container {padding-top: 3.5rem !important;}

    /* 按钮：果冻黄 */
    div.stButton > button {
        background: linear-gradient(145deg, #ffdb4d 0%, #ffb300 100%); 
        color: #5d4037; border: 2px solid #fff9c4; border-radius: 25px; 
        padding: 0.6rem 1.2rem; font-weight: 800; font-size: 16px;
        box-shadow: 0 4px 10px rgba(255, 179, 0, 0.4); 
        transition: all 0.2s; width: 100%;
    }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(255, 179, 0, 0.5); }
    div.stButton > button:active { transform: scale(0.96); }
    div.stButton > button[kind="secondary"] { background: #f0f0f0; color: #666; border: 1px solid #ddd; box-shadow: none; }
    
    /* 核心解锁按钮样式 (New) */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2962ff 0%, #0039cb 100%);
        color: white; border: none; box-shadow: 0 4px 15px rgba(41, 98, 255, 0.4);
    }

    /* 卡片容器 */
    .app-card { background-color: #ffffff; border-radius: 12px; padding: 16px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.02); }
    .section-header { display: flex; align-items: center; margin-bottom: 12px; margin-top: 8px; }
    .section-title { font-size: 17px; font-weight: 900; color: #333; margin-right: 5px; }
    .vip-badge { background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 99%); color: #d32f2f; font-size: 10px; font-weight: 800; padding: 2px 8px; border-radius: 10px; font-style: italic; }

    /* AI Copilot 对话框 */
    .ai-chat-box {
        background: #f0f7ff; border-radius: 12px; padding: 15px; margin-bottom: 20px;
        border-left: 5px solid #2962ff; box-shadow: 0 4px 12px rgba(41, 98, 255, 0.1);
    }
    .ai-avatar { font-size: 24px; margin-right: 10px; float: left; }
    .ai-content { overflow: hidden; font-size: 15px; line-height: 1.6; color: #2c3e50; }

    /* 大盘红绿灯 */
    .market-status-box {
        padding: 12px 20px; border-radius: 12px; margin-bottom: 20px;
        display: flex; align-items: center; justify-content: space-between;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.05);
    }
    .status-green { background: #e8f5e9; color: #2e7d32; border-left: 5px solid #2e7d32; }
    .status-red { background: #ffebee; color: #c62828; border-left: 5px solid #c62828; }
    .status-yellow { background: #fffde7; color: #f9a825; border-left: 5px solid #f9a825; }
    .status-icon { font-size: 24px; margin-right: 12px; }
    .status-text { font-weight: 800; font-size: 16px; }
    .status-sub { font-size: 12px; opacity: 0.8; margin-top: 2px;}

    /* 股价大字 + 四宫格参数 */
    .big-price-box { text-align: center; margin-bottom: 20px; }
    .price-main { font-size: 48px; font-weight: 900; line-height: 1; letter-spacing: -1.5px; }
    .price-sub { font-size: 16px; font-weight: 600; margin-left: 8px; padding: 2px 6px; border-radius: 4px; }
    .param-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 15px; }
    .param-item { background: #f9fafe; border-radius: 10px; padding: 10px; text-align: center; border: 1px solid #edf2f7; }
    .param-val { font-size: 20px; font-weight: 800; color: #2c3e50; }
    .param-lbl { font-size: 12px; color: #95a5a6; margin-top: 2px; }

    /* 综合评级 */
    .rating-container { display: flex; justify-content: space-between; gap: 8px; }
    .rating-box { flex: 1; background: #fff; border: 1px solid #f0f0f0; border-radius: 12px; text-align: center; padding: 15px 2px; box-shadow: 0 4px 10px rgba(0,0,0,0.02); }
    .rating-score { font-size: 28px; font-weight: 900; color: #ff3b30; line-height: 1; margin-bottom: 5px; }
    .rating-label { font-size: 12px; color: #666; font-weight: 500; }
    .score-yellow { color: #ff9800 !important; }

    /* 投资亮点 */
    .highlight-item { display: flex; align-items: start; margin-bottom: 12px; line-height: 1.5; }
    .tag-box { background: #fff5f5; color: #ff3b30; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 6px; margin-right: 10px; white-space: nowrap; margin-top: 2px; }
    .tag-blue { background: #f0f7ff; color: #2962ff; }
    .tag-text { font-size: 14px; color: #333; text-align: justify; }
    .hl-num { color: #ff3b30; font-weight: 700; padding: 0 2px; }

    /* 策略卡片 */
    .strategy-card { background: #fcfcfc; border: 1px solid #eee; border-left: 4px solid #ffca28; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.02); }
    .strategy-title { font-size: 18px; font-weight: 800; color: #333; margin-bottom: 10px; }
    .strategy-grid { display: flex; justify-content: space-between; margin-bottom: 10px; }
    .support-line { border-top: 1px dashed #eee; margin-top: 10px; padding-top: 10px; font-size: 12px; color: #888; display: flex; justify-content: space-between; }
    .reason-box { background: #f8f9fa; border-radius: 8px; padding: 10px; margin-top: 8px; font-size: 13px; color: #555; }
    
    /* 锁定遮罩 (New) */
    .locked-blur { filter: blur(6px); opacity: 0.6; pointer-events: none; user-select: none; }
    .lock-overlay {
        position: relative; top: -150px; left: 0; right: 0; text-align: center; z-index: 10;
        background: rgba(255,255,255,0.85); backdrop-filter: blur(4px);
        padding: 30px; border-radius: 12px; border: 1px solid #eee; margin: 0 20px;
    }
    
    [data-testid="metric-container"] { display: none; }
</style>
"""
st.markdown(ui_css, unsafe_allow_html=True)

# ==========================================
# 2. 数据库与工具 (升级版：支持 VIP)
# ==========================================
def init_db():
    # ✅ V68 Upgrade: 添加 vip_expiry 字段
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota", "vip_expiry"])
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
    try:
        df = pd.read_csv(DB_FILE, dtype={"watchlist": str, "quota": int})
        # ✅ V68 Upgrade: 动态兼容旧数据
        if "vip_expiry" not in df.columns:
            df["vip_expiry"] = ""
            save_users(df)
        return df
    except: return pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota", "vip_expiry"])

def save_users(df): df.to_csv(DB_FILE, index=False)

def load_keys():
    try: return pd.read_csv(KEYS_FILE)
    except: return pd.DataFrame(columns=["key", "points", "status", "created_at"])

def save_keys(df): df.to_csv(KEYS_FILE, index=False)

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
    new_row = {"username": u, "password_hash": hashed, "watchlist": "", "quota": 0, "vip_expiry": ""}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_users(df)
    return True, "注册成功"

# ✅ NEW: VIP 检查与逻辑
def check_is_vip(username):
    if username == ADMIN_USER: return True
    df = load_users()
    row = df[df["username"] == username]
    if row.empty: return False
    exp_str = str(row.iloc[0].get("vip_expiry", ""))
    if not exp_str or exp_str == "nan": return False
    try:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d %H:%M:%S")
        return exp_date > datetime.now()
    except: return False

def get_vip_expiry_str(username):
    if username == ADMIN_USER: return "永久管理员"
    df = load_users()
    row = df[df["username"] == username]
    if row.empty: return "-"
    exp_str = str(row.iloc[0].get("vip_expiry", ""))
    if not exp_str or exp_str == "nan": return "未开通"
    try:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d %H:%M:%S")
        if exp_date > datetime.now(): return exp_date.strftime("%Y-%m-%d 到期")
        else: return "已过期"
    except: return "未开通"

def add_vip_days(username, days):
    df = load_users()
    idx = df[df["username"] == username].index[0]
    current_exp = str(df.loc[idx, "vip_expiry"])
    now = datetime.now()
    
    if current_exp and current_exp != "nan":
        try:
            curr_date = datetime.strptime(current_exp, "%Y-%m-%d %H:%M:%S")
            start_date = max(now, curr_date)
        except: start_date = now
    else:
        start_date = now
        
    new_exp = start_date + timedelta(days=days)
    df.loc[idx, "vip_expiry"] = new_exp.strftime("%Y-%m-%d %H:%M:%S")
    save_users(df)
    return True

def consume_quota(u):
    if u == ADMIN_USER: return True
    # ✅ V68: 如果是VIP，不扣分
    if check_is_vip(u): return True
    
    df = load_users()
    idx = df[df["username"] == u].index
    if len(idx) > 0 and df.loc[idx[0], "quota"] > 0:
        df.loc[idx[0], "quota"] -= 1
        save_users(df)
        return True
    return False

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
# 3. 股票逻辑 (保持原样)
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
    df['MA60'] = df['close'].rolling(60).mean()
    return df

@st.cache_data(ttl=3600)
def get_name(code, token, proxy=None):
    clean_code = code.strip().upper().replace('.SH','').replace('.SZ','').replace('SH','').replace('SZ','')
    QUICK_MAP = {'600519':'贵州茅台','000858':'五粮液','601318':'中国平安','600036':'招商银行','300750':'宁德时代','002594':'比亚迪','601888':'中国中免','600276':'恒瑞医药','601857':'中国石油','601088':'中国神华','601988':'中国银行','601398':'工商银行','AAPL':'Apple','TSLA':'Tesla','NVDA':'NVIDIA','MSFT':'Microsoft','BABA':'Alibaba'}
    if clean_code in QUICK_MAP: return QUICK_MAP[clean_code]
    try: return yf.Ticker(code).info.get('shortName', code)
    except: return code

def get_data_and_resample(code, token, timeframe, adjust, proxy=None):
    code = process_ticker(code)
    raw_df = pd.DataFrame()
    # 简化版逻辑，保留yfinance核心
    try:
        yf_df = yf.download(code, period="2y", interval="1d", progress=False, auto_adjust=False)
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
            raw_df = yf_df[['date','open','high','low','close','volume']].copy()
            for c in ['open','high','low','close','volume']: raw_df[c] = pd.to_numeric(raw_df[c], errors='coerce')
            raw_df['pct_change'] = raw_df['close'].pct_change() * 100
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
    try:
        t = yf.Ticker(code); i = t.info
        res['pe'] = safe_fmt(i.get('trailingPE'))
        res['pb'] = safe_fmt(i.get('priceToBook'))
        res['mv'] = f"{i.get('marketCap')/100000000:.2f}亿" if i.get('marketCap') else "-"
        if 'targetMeanPrice' in i: res['target_price'] = safe_fmt(i.get('targetMeanPrice'))
        if 'recommendationKey' in i: res['rating'] = i.get('recommendationKey', '').replace('buy','买入').replace('sell','卖出').replace('hold','持有')
    except: pass
    return res

def calc_full_indicators(df, ma_s, ma_l):
    if df.empty: return df
    try:
        c = df['close']; h = df['high']; l = df['low']; v = df['volume']
    except: return df

    df['MA_Short'] = c.rolling(ma_s).mean()
    df['MA_Long'] = c.rolling(ma_l).mean()
    df['MA60'] = c.rolling(60).mean()

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

def check_market_status(df):
    if df is None or df.empty or len(df) < 60: return "neutral", "数据不足", "gray"
    curr = df.iloc[-1]
    if curr['close'] > curr['MA60']: return "green", "🚀 趋势向上 (可积极做多)", "status-green"
    elif curr['close'] < curr['MA60']: return "red", "🛑 趋势转弱 (建议空仓观望)", "status-red"
    else: return "yellow", "⚠️ 震荡整理 (轻仓操作)", "status-yellow"

def get_daily_picks(user_watchlist):
    hot_stocks = ["600519", "NVDA", "TSLA", "300750", "002594", "AAPL"]
    pool = list(set(hot_stocks + user_watchlist))
    results = []
    for code in pool[:6]: 
        name = get_name(code, "", None)
        status = random.choice(["buy", "hold", "wait"])
        if status == "buy": results.append({"code": code, "name": name, "tag": "今日买点", "type": "tag-buy"})
        elif status == "hold": results.append({"code": code, "name": name, "tag": "持股待涨", "type": "tag-hold"})
    return results

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
        <div class="deep-text">• 江恩角度线 1x1线是多空分界线。<br>• <b>斐波那契回撤</b>：{fib_txt}</div>
    </div>
    <div class="app-card">
        <div class="deep-title">📊 核心动能指标</div>
        <div class="deep-text">• <b>MACD</b>：当前 {macd_state}<br>• <b>VOL量能</b>：今日 {vol_state} (量比 {safe_fmt(curr['VolRatio'])})</div>
    </div>
    """
    return html

def generate_ai_copilot_text(df, name):
    c = df.iloc[-1]
    openers = ["主人好！", "Hi~ 老板，", "数据汇报："]
    mood = "neutral"
    advice = ""
    if c['close'] > c['MA60']:
        if c['MA_Short'] > c['MA_Long']: advice = f"现在的 {name} 走势很漂亮，多头排列，你可以继续持有享受泡沫。"; mood = "happy"
        else: advice = f"虽然还在牛熊线上方，但短期有回调压力，别追高哦。"; mood = "neutral"
    else: advice = f"目前趋势偏弱，处于空头掌控中，建议多看少动，保住本金最重要。"; mood = "worried"
    tech = ""
    if c['RSI'] < 30: tech = "不过我看 RSI 已经超卖了，短期随时可能反弹，如果你是左侧交易者可以轻仓试错。"
    elif c['RSI'] > 75: tech = "而且 RSI 有点过热了，小心主力骗炮出货，记得推高止损。"
    final_text = f"{random.choice(openers)} {advice} {tech} 切记，即使我看好，也要设好止损线 {c['close']*0.95:.2f} 保护自己。"
    return final_text, mood

def analyze_score(df):
    c = df.iloc[-1]; score=0; reasons=[]
    if c['MA_Short']>c['MA_Long']: score+=2; reasons.append("均线金叉 (短线看涨)")
    else: score-=2; reasons.append("均线死叉 (短线看跌)")
    if c['close']>c['MA_Long']: score+=1; reasons.append("站上长期生命线")
    else: reasons.append("跌破长期生命线")
    if c['DIF']>c['DEA']: score+=1; reasons.append("MACD 处于多头区域")
    if c['RSI']<20: score+=2; reasons.append("RSI 进入超卖区 (反弹概率大)")
    elif c['RSI']>80: reasons.append("RSI 进入超买区 (回调风险大)")
    if c['VolRatio']>1.5: score+=1; reasons.append("主力放量攻击")
    
    action = "积极买入" if score>=4 else "持有/观望" if score>=0 else "减仓/卖出"
    color = "success" if score>=4 else "warning" if score>=0 else "error"
    if score >= 4: pos_txt = "80% (重仓)"
    elif score >= 1: pos_txt = "50% (中仓)"
    elif score >= -2: pos_txt = "20% (底仓)"
    else: pos_txt = "0% (空仓)"
    atr = c['ATR14']; stop_loss = c['close'] - 2*atr; take_profit = c['close'] + 3*atr
    support = df['low'].iloc[-20:].min(); resistance = df['high'].iloc[-20:].max()
    return score, action, color, stop_loss, take_profit, pos_txt, support, resistance, reasons

def calculate_smart_score(df, funda):
    trend_score = 5; last = df.iloc[-1]
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

def plot_chart(df, name, flags, ma_s, ma_l, is_simple=False):
    # ✅ V68: 基础版只显示简单K线
    rows = 4 if not is_simple else 1
    row_heights = [0.55,0.1,0.15,0.2] if not is_simple else [1.0]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=row_heights)
    fig.update_layout(dragmode=False, margin=dict(l=10, r=10, t=10, b=10))
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线', increasing_line_color='#FF3B30', decreasing_line_color='#34C759'), 1, 1)
    
    if not is_simple:
        if flags.get('ma'):
            fig.add_trace(go.Scatter(x=df['date'], y=df['MA_Short'], name=f'MA{ma_s}', line=dict(width=1.2, color='#333333')), 1, 1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['MA_Long'], name=f'MA{ma_l}', line=dict(width=1.2, color='#ffcc00')), 1, 1)
        if flags.get('boll'):
            fig.add_trace(go.Scatter(x=df['date'], y=df['Upper'], line=dict(width=1, dash='dash', color='rgba(33, 150, 243, 0.3)'), name='Upper'), 1, 1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['Lower'], line=dict(width=1, dash='dash', color='rgba(33, 150, 243, 0.3)'), name='Lower', fill='tonexty', fillcolor='rgba(33, 150, 243, 0.05)'), 1, 1)
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
    
    fig.update_layout(height=600 if not is_simple else 300, xaxis_rangeslider_visible=False, paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#1d1d1f'), xaxis=dict(showgrid=False, showline=True, linecolor='#e5e5e5'), yaxis=dict(showgrid=True, gridcolor='#f5f5f5'), legend=dict(orientation="h", y=-0.05))
    st.plotly_chart(fig, use_container_width=True)

def create_download_link(df, name):
    html = f"<html><body><h1>{name} 研报</h1><p>生成时间: {datetime.now()}</p></body></html>"
    b64 = base64.b64encode(html.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{name}_report.html" style="text-decoration:none; padding:8px 15px; background:#ff3b30; color:white; border-radius:5px; font-weight:bold;">📄 导出研报 (VIP)</a>'
    return href

# ==========================================
# 5. 执行入口
# ==========================================
init_db()

with st.sidebar:
    st.markdown("""
    <div style='text-align: left; margin-bottom: 20px;'>
        <div class='brand-title'>阿尔法量研 <span style='color:#0071e3'>Pro</span></div>
        <div class='brand-en'>AlphaQuant Pro V68 (VIP)</div>
        <div class='brand-slogan'>只做加法，功能升级。</div>
    </div>
    """, unsafe_allow_html=True)
    
    new_c = st.text_input("🔍 股票代码", st.session_state.code)
    if new_c != st.session_state.code: st.session_state.code = new_c; st.session_state.paid_code = ""; st.rerun()

    if st.session_state.get('logged_in'):
        user = st.session_state["user"]
        is_admin = (user == ADMIN_USER)
        is_vip_user = check_is_vip(user)
        
        # ✅ NEW: VIP 状态展示
        if is_vip_user:
            st.success(f"👑 尊贵VIP会员\n\n到期：{get_vip_expiry_str(user)}")
        else:
            st.info(f"👤 普通用户 (积分: {load_users()[load_users()['username']==user]['quota'].iloc[0]})")
        
        if not is_admin:
            st.markdown("### 🎯 每日精选")
            user_wl = get_user_watchlist(user)
            picks = get_daily_picks(user_wl)
            for pick in picks:
                if st.button(f"{pick['tag']} | {pick['name']}", key=f"pick_{pick['code']}"):
                    st.session_state.code = pick['code']; st.rerun()
            st.divider()
        
        # Paper Trading
        if not is_admin:
            with st.expander("🎮 模拟交易 (Paper Trading)", expanded=True):
                curr_hold = st.session_state.paper_holdings.get(st.session_state.code, None)
                if curr_hold:
                    st.info(f"持仓: {curr_hold['price']}\n日期: {curr_hold['date']}")
                    if st.button("卖出止盈/止损", key="paper_sell"):
                        del st.session_state.paper_holdings[st.session_state.code]; st.rerun()
                else:
                    if st.button("➕ 模拟买入", key="paper_buy"):
                        st.session_state.paper_holdings[st.session_state.code] = {'price': 0, 'date': datetime.now().strftime("%Y-%m-%d")}; st.rerun()

        if not is_admin:
            with st.expander("⭐ 自选股", expanded=False):
                current_wl = get_user_watchlist(user)
                for c in current_wl:
                    if st.button(f"{c}", key=f"wl_{c}"): st.session_state.code = c; st.session_state.paid_code = ""; st.rerun()
            if st.button("❤️ 加入自选"): update_watchlist(user, st.session_state.code, "add"); st.rerun()

        if not is_admin:
            with st.expander("💎 会员中心", expanded=False):
                my_quota = load_users()[load_users()['username']==user]['quota'].iloc[0]
                st.write(f"当前积分: **{my_quota}**")
                
                tab_pay, tab_vip = st.tabs(["充值积分", "兑换月卡"])
                with tab_pay:
                    pay_opt = st.radio("充值面额", [20, 50, 100], horizontal=True, format_func=lambda x: f"￥{x}")
                    if st.button("✅ 模拟支付(获取卡密)"):
                        new_key = generate_key(pay_opt)
                        st.code(new_key, language="text")
                    k_in = st.text_input("输入卡密兑换积分")
                    if st.button("兑换积分"):
                        s, m = redeem_key(user, k_in)
                        if s: st.success(m); time.sleep(1); st.rerun()
                        else: st.error(m)
                
                with tab_vip:
                    st.write("🌟 **VIP特权**：无限次查看深度研报、AI解读、买卖点位。")
                    st.write("价格：300 积分 / 30天")
                    if st.button("🔥 兑换 30天 VIP"):
                        df_u = load_users()
                        idx = df_u[df_u["username"] == user].index[0]
                        if df_u.loc[idx, "quota"] >= 300:
                            df_u.loc[idx, "quota"] -= 300
                            save_users(df_u)
                            add_vip_days(user, 30)
                            st.balloons()
                            st.success("🎉 恭喜成为尊贵的 VIP 会员！")
                            time.sleep(2); st.rerun()
                        else:
                            st.error("积分不足，请先充值")

        timeframe = st.selectbox("周期", ["日线", "周线", "月线"])
        days = st.radio("范围", [30,60,120,250], 2, horizontal=True)
        adjust = st.selectbox("复权", ["qfq","hfq",""], 0)
        
        if st.button("退出登录"): st.session_state["logged_in"]=False; st.rerun()
    else:
        st.info("请先登录系统")

# 登录逻辑
if not st.session_state.get('logged_in'):
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.markdown("<h1 style='text-align:center'>AlphaQuant Pro V68</h1>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔑 登录", "📝 注册"])
        with tab1:
            u = st.text_input("账号"); p = st.text_input("密码", type="password")
            if st.button("登录"):
                if verify_login(u.strip(), p): st.session_state["logged_in"] = True; st.session_state["user"] = u.strip(); st.rerun()
                else: st.error("错误")
        with tab2:
            nu = st.text_input("新用户"); np1 = st.text_input("设置密码", type="password")
            if st.button("注册"):
                suc, msg = register_user(nu.strip(), np1)
                if suc: st.success(msg)
                else: st.error(msg)
    st.stop()

# --- 主内容区 ---
name = get_name(st.session_state.code, "", None) 
c1, c2 = st.columns([3, 1])
with c1: st.title(f"📈 {name} ({st.session_state.code})")

# 数据加载
df = get_data_and_resample(st.session_state.code, "", timeframe, adjust, None)
if df.empty: df = generate_mock_data(days) # 兜底

# 计算指标
funda = get_fundamentals(st.session_state.code, "")
df = calc_full_indicators(df, ma_s, ma_l)
df = detect_patterns(df)

# 更新 Paper Trading 价格
if st.session_state.code in st.session_state.paper_holdings:
    st.session_state.paper_holdings[st.session_state.code]['price'] = df.iloc[-1]['close']

# 1. 基础展示 (Free Tier)
status, msg, css_class = check_market_status(df)
st.markdown(f"""
<div class="market-status-box {css_class}">
    <div style="display:flex; align-items:center;">
        <span class="status-icon">{'🟢' if status=='green' else '🔴' if status=='red' else '🟡'}</span>
        <div><div class="status-text">{msg}</div><div class="status-sub">基于 MA60 牛熊线</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

l = df.iloc[-1]
color = "#ff3b30" if l['pct_change'] > 0 else "#00c853"
st.markdown(f"""
<div class="big-price-box">
    <span class="price-main" style="color:{color}">{l['close']:.2f}</span>
    <span class="price-sub" style="color:{color}">{l['pct_change']:.2f}%</span>
</div>
""", unsafe_allow_html=True)

sq, sv, st_ = calculate_smart_score(df, funda)
st.markdown(f"""
<div class="rating-container">
    <div class="rating-box"><div class="rating-score">{sq}</div><div class="rating-label">公司质量</div></div>
    <div class="rating-box"><div class="rating-score score-yellow">{sv}</div><div class="rating-label">估值安全</div></div>
    <div class="rating-box"><div class="rating-score">{st_}</div><div class="rating-label">股价趋势</div></div>
</div>
<div style="text-align:center; font-size:12px; color:#999; margin:10px 0;">(基础评分免费查看，详细解读需解锁)</div>
""", unsafe_allow_html=True)

# 2. 权限判断
has_access = check_is_vip(st.session_state["user"]) or (st.session_state.paid_code == st.session_state.code)

if not has_access:
    # --- 锁定状态展示 ---
    st.markdown("---")
    # 模糊的图表 (Simple Mode)
    st.caption("🔒 基础K线 (指标已隐藏)")
    plot_chart(df.tail(60), name, {}, 5, 20, is_simple=True)
    
    # 遮罩层
    st.markdown("""
    <div class="lock-overlay">
        <h3>🔒 解锁深度透视 (Deep Dive)</h3>
        <p>查看 <b>买卖点位</b> | <b>AI 投顾助理</b> | <b>缠论结构图</b> | <b>完整研报</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    c_btn1, c_btn2, c_btn3 = st.columns([1,2,1])
    with c_btn2:
        if st.button("🔓 支付 1 积分解锁本次 (或开通VIP)", type="primary"):
            if consume_quota(st.session_state["user"]):
                st.session_state.paid_code = st.session_state.code
                st.rerun()
            else:
                st.error("积分不足，请在左侧侧边栏充值！")
    
else:
    # --- 解锁后展示 (原有所有高级功能) ---
    
    # AI 助理
    ai_text, ai_mood = generate_ai_copilot_text(df, name)
    ai_icon = "🤖" if ai_mood == "neutral" else "😊" if ai_mood == "happy" else "😰"
    st.markdown(f"""
    <div class="ai-chat-box">
        <div class="ai-avatar">{ai_icon}</div>
        <div class="ai-content"><span style="font-weight:bold; color:#2962ff;">AI 投顾助理：</span>{ai_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # 完整图表
    st.caption("📊 专家级图表 (MA + BOLL + 缠论 + MACD)")
    plot_chart(df.tail(days), name, flags, ma_s, ma_l, is_simple=False)

    # 详细策略建议
    sc, act, col, sl, tp, pos, sup, res, reasons = analyze_score(df)
    reason_html = "".join([f"<div>• {r}</div>" for r in reasons])
    st.markdown(f"""
    <div class="strategy-card">
        <div class="strategy-title">🤖 最终操作建议：{act}</div>
        <div class="strategy-grid">
            <div class="strategy-col"><span class="st-lbl">仓位</span><span class="st-val">{pos}</span></div>
            <div class="strategy-col"><span class="st-lbl">止盈</span><span class="st-val" style="color:#ff3b30">{tp:.2f}</span></div>
            <div class="strategy-col"><span class="st-lbl">止损</span><span class="st-val" style="color:#00c853">{sl:.2f}</span></div>
        </div>
        <div class="support-line">
            <span>📍 支撑位：<span style="color:#00c853; font-weight:bold;">{sup:.2f}</span></span>
            <span>⚡ 压力位：<span style="color:#ff3b30; font-weight:bold;">{res:.2f}</span></span>
        </div>
        <div class="reason-box"><div class="reason-title">💡 决策依据</div>{reason_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # 深度研报 & 导出
    st.markdown(generate_deep_report(df, name), unsafe_allow_html=True)
    st.markdown(create_download_link(df, name), unsafe_allow_html=True)