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
# 1. 核心配置 & CSS
# ==========================================
st.set_page_config(
    page_title="阿尔法量研 Pro V68 (VIP Biz)",
    layout="wide",
    page_icon="🔥",
    initial_sidebar_state="expanded"
)

# 初始化 Session
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if "code" not in st.session_state: st.session_state.code = "600519"
if "paid_code" not in st.session_state: st.session_state.paid_code = "" # 单次付费记录
if "paper_holdings" not in st.session_state: st.session_state.paper_holdings = {}

# 全局变量兜底
ma_s = 5
ma_l = 20
flags = {
    'ma': True, 'boll': True, 'vol': True, 'macd': True, 
    'kdj': True, 'gann': False, 'fib': True, 'chan': True
}
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"
DB_FILE = "users_v68.csv" # 升级文件名以防混淆，自动迁移
KEYS_FILE = "card_keys.csv"

# Optional deps
ts = None; bs = None
try: import tushare as ts
except: pass
try: import baostock as bs
except: pass

# 🔥 V68.0 CSS：新增锁屏遮罩样式
ui_css = """
<style>
    /* 继承 V67 样式 */
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

    /* 果冻按钮 */
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
    div.stButton > button[kind="primary"] { background: linear-gradient(145deg, #2962ff 0%, #0039cb 100%); color: white; border: none; box-shadow: 0 4px 10px rgba(41, 98, 255, 0.3); }

    /* 通用容器 */
    .app-card { background-color: #ffffff; border-radius: 12px; padding: 16px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.02); }
    .section-header { display: flex; align-items: center; margin-bottom: 12px; margin-top: 8px; }
    .section-title { font-size: 17px; font-weight: 900; color: #333; margin-right: 5px; }
    .vip-badge { background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 99%); color: #d32f2f; font-size: 10px; font-weight: 800; padding: 2px 8px; border-radius: 10px; font-style: italic; }

    /* AI 框 */
    .ai-chat-box { background: #f0f7ff; border-radius: 12px; padding: 15px; margin-bottom: 20px; border-left: 5px solid #2962ff; box-shadow: 0 4px 12px rgba(41, 98, 255, 0.1); }
    .ai-avatar { font-size: 24px; margin-right: 10px; float: left; }
    .ai-content { overflow: hidden; font-size: 15px; line-height: 1.6; color: #2c3e50; }

    /* 红绿灯 */
    .market-status-box { padding: 12px 20px; border-radius: 12px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.05); }
    .status-green { background: #e8f5e9; color: #2e7d32; border-left: 5px solid #2e7d32; }
    .status-red { background: #ffebee; color: #c62828; border-left: 5px solid #c62828; }
    .status-yellow { background: #fffde7; color: #f9a825; border-left: 5px solid #f9a825; }
    .status-icon { font-size: 24px; margin-right: 12px; }
    .status-text { font-weight: 800; font-size: 16px; }

    /* 股价大字 */
    .big-price-box { text-align: center; margin-bottom: 20px; }
    .price-main { font-size: 48px; font-weight: 900; line-height: 1; letter-spacing: -1.5px; }
    .price-sub { font-size: 16px; font-weight: 600; margin-left: 8px; padding: 2px 6px; border-radius: 4px; }
    .param-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 15px; }
    .param-item { background: #f9fafe; border-radius: 10px; padding: 10px; text-align: center; border: 1px solid #edf2f7; }
    .param-val { font-size: 20px; font-weight: 800; color: #2c3e50; }
    .param-lbl { font-size: 12px; color: #95a5a6; margin-top: 2px; }

    /* 评级 */
    .rating-container { display: flex; justify-content: space-between; gap: 8px; }
    .rating-box { flex: 1; background: #fff; border: 1px solid #f0f0f0; border-radius: 12px; text-align: center; padding: 15px 2px; box-shadow: 0 4px 10px rgba(0,0,0,0.02); }
    .rating-score { font-size: 28px; font-weight: 900; color: #ff3b30; line-height: 1; margin-bottom: 5px; }
    .rating-label { font-size: 12px; color: #666; font-weight: 500; }
    .score-yellow { color: #ff9800 !important; }

    /* 策略卡片 & 解释性AI */
    .strategy-card { background: #fcfcfc; border: 1px solid #eee; border-left: 4px solid #ffca28; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.02); }
    .strategy-title { font-size: 18px; font-weight: 800; color: #333; margin-bottom: 10px; }
    .strategy-grid { display: flex; justify-content: space-between; margin-bottom: 10px; }
    .support-line { border-top: 1px dashed #eee; margin-top: 10px; padding-top: 10px; font-size: 12px; color: #888; display: flex; justify-content: space-between; }
    .reason-box { background: #f8f9fa; border-radius: 8px; padding: 10px; margin-top: 8px; font-size: 13px; color: #555; }
    .reason-title { font-weight: 700; color: #333; margin-bottom: 4px; display: flex; align-items: center; }
    
    /* 风险雷达 */
    .risk-header { display: flex; justify-content: space-between; font-size: 12px; color: #666; margin-bottom: 5px; }
    .risk-bar-bg { height: 6px; background: #eee; border-radius: 3px; overflow: hidden; }
    .risk-bar-fill { height: 100%; border-radius: 3px; }

    /* 侧边栏 */
    .brand-title { font-size: 22px; font-weight: 900; color: #333; margin-bottom: 2px; }
    .brand-slogan { font-size: 12px; color: #999; margin-bottom: 20px; }
    
    /* ✅ NEW: 模糊锁定遮罩 */
    .blur-lock { 
        filter: blur(6px); opacity: 0.6; pointer-events: none; user-select: none; 
        transition: all 0.5s;
    }
    .lock-overlay {
        position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
        z-index: 10; text-align: center; width: 80%;
    }
    .lock-card {
        background: rgba(255, 255, 255, 0.95); padding: 20px; border-radius: 16px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12); border: 1px solid #fff; backdrop-filter: blur(10px);
    }
    
    [data-testid="metric-container"] { display: none; }
</style>
"""
st.markdown(ui_css, unsafe_allow_html=True)

# ==========================================
# 2. 数据库与工具 (增强版)
# ==========================================
def init_db():
    # 自动迁移旧数据库
    if not os.path.exists(DB_FILE):
        if os.path.exists("users_v61.csv"):
            try:
                df_old = pd.read_csv("users_v61.csv")
                df_old['vip_expiry'] = "1970-01-01" # 默认非VIP
                df_old.to_csv(DB_FILE, index=False)
            except:
                df = pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota", "vip_expiry"])
                df.to_csv(DB_FILE, index=False)
        else:
            df = pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota", "vip_expiry"])
            df.to_csv(DB_FILE, index=False)
    
    if not os.path.exists(KEYS_FILE):
        df_keys = pd.DataFrame(columns=["key", "points", "status", "created_at"])
        df_keys.to_csv(KEYS_FILE, index=False)

def safe_fmt(value, fmt="{:.2f}", default="-", suffix=""):
    try:
        if value is None: return default
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
        # 鲁棒性检查：如果旧版没有 vip_expiry，自动补上
        if "vip_expiry" not in df.columns:
            df["vip_expiry"] = "1970-01-01"
            save_users(df)
        return df
    except: return pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota", "vip_expiry"])

def save_users(df): df.to_csv(DB_FILE, index=False)
def load_keys():
    try: return pd.read_csv(KEYS_FILE)
    except: return pd.DataFrame(columns=["key", "points", "status", "created_at"])
def save_keys(df): df.to_csv(KEYS_FILE, index=False)

def check_vip_status(username):
    """返回 (是否VIP, 到期日期字符串)"""
    if username == ADMIN_USER: return True, "2099-12-31"
    df = load_users()
    row = df[df["username"] == username]
    if row.empty: return False, "1970-01-01"
    expiry = str(row.iloc[0]["vip_expiry"])
    try:
        exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        is_vip = exp_date >= datetime.now().date()
        return is_vip, expiry
    except:
        return False, "1970-01-01"

def extend_vip(username, days):
    df = load_users()
    idx = df[df["username"] == username].index[0]
    current_exp = str(df.loc[idx, "vip_expiry"])
    try:
        curr_date = datetime.strptime(current_exp, "%Y-%m-%d").date()
    except:
        curr_date = datetime.now().date() - timedelta(days=1)
    
    # 如果已经过期，从今天开始算；如果没过期，在原基础上加
    base_date = max(curr_date, datetime.now().date())
    new_date = base_date + timedelta(days=days)
    df.loc[idx, "vip_expiry"] = new_date.strftime("%Y-%m-%d")
    save_users(df)
    return new_date.strftime("%Y-%m-%d")

# 充值与消费
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

def consume_quota(u, amount=1):
    if u == ADMIN_USER: return True
    df = load_users()
    idx = df[df["username"] == u].index
    if len(idx) > 0 and df.loc[idx[0], "quota"] >= amount:
        df.loc[idx[0], "quota"] -= amount
        save_users(df)
        return True
    return False

# 用户管理
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
    new_row = {"username": u, "password_hash": hashed, "watchlist": "", "quota": 0, "vip_expiry": "1970-01-01"}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_users(df)
    return True, "注册成功"

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

def get_user_watchlist(username):
    df = load_users()
    if username == ADMIN_USER: return []
    row = df[df["username"] == username]
    if row.empty: return []
    wl_str = str(row.iloc[0]["watchlist"])
    if wl_str == "nan": return []
    return [c.strip() for c in wl_str.split(",") if c.strip()]

# ==========================================
# 3. 股票逻辑 (保持 V67 原样，无删减)
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
    # 优先尝试 yfinance
    try:
        yf_df = yf.download(code, period="2y", interval="1d", progress=False, auto_adjust=False)
        if not yf_df.empty:
            if isinstance(yf_df.columns, pd.MultiIndex): yf_df.columns = yf_df.columns.get_level_values(0)
            yf_df.columns = [str(c).lower().strip() for c in yf_df.columns]
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
    code = process_ticker(code)
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
    c = df['close']; h = df['high']; l = df['low']; v = df['volume']
    df['MA_Short'] = c.rolling(ma_s).mean()
    df['MA_Long'] = c.rolling(ma_l).mean()
    df['MA60'] = c.rolling(60).mean()

    # Ichimoku
    p_high = h.rolling(9).max(); p_low = l.rolling(9).min()
    df['Tenkan'] = (p_high + p_low) / 2
    p_high26 = h.rolling(26).max(); p_low26 = l.rolling(26).min()
    df['Kijun'] = (p_high26 + p_low26) / 2
    df['SpanA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['SpanB'] = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)

    # BOLL
    mid = c.rolling(20).mean(); std = c.rolling(20).std()
    df['Upper'] = mid + 2*std; df['Lower'] = mid - 2*std

    # MACD
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    df['DIF'] = e12 - e26; df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean(); df['HIST'] = 2 * (df['DIF'] - df['DEA'])

    # RSI
    delta = c.diff(); up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
    rs = up.rolling(14).mean()/(down.rolling(14).mean()+1e-9)
    df['RSI'] = 100 - (100/(1+rs))

    # KDJ
    low9 = l.rolling(9).min(); high9 = h.rolling(9).max()
    rsv = (c - low9)/(high9 - low9 + 1e-9) * 100
    df['K'] = rsv.ewm(com=2).mean(); df['D'] = df['K'].ewm(com=2).mean(); df['J'] = 3 * df['K'] - 2 * df['D']

    # ATR & ADX
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()
    dm_p = np.where((h.diff() > l.diff().abs()) & (h.diff()>0), h.diff(), 0)
    dm_m = np.where((l.diff().abs() > h.diff()) & (l.diff()<0), l.diff().abs(), 0)
    di_plus = 100 * pd.Series(dm_p).rolling(14).sum() / (tr.rolling(14).sum()+1e-9)
    di_minus = 100 * pd.Series(dm_m).rolling(14).sum() / (tr.rolling(14).sum()+1e-9)
    df['ADX'] = (abs(di_plus - di_minus)/(di_plus + di_minus + 1e-9) * 100).rolling(14).mean()
    df['VolRatio'] = v / (v.rolling(5).mean() + 1e-9)
    df = df.fillna(0) # 简单填充
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
# 4. 分析与生成逻辑 (功能完全保留)
# ==========================================
def check_market_status(df):
    if df is None or df.empty or len(df) < 60: return "neutral", "数据不足", "gray"
    curr = df.iloc[-1]
    if curr['close'] > curr['MA60']: return "green", "🚀 趋势向上 (可积极做多)", "status-green"
    elif curr['close'] < curr['MA60']: return "red", "🛑 趋势转弱 (建议空仓观望)", "status-red"
    else: return "yellow", "⚠️ 震荡整理 (轻仓操作)", "status-yellow"

def generate_ai_copilot_text(df, name):
    c = df.iloc[-1]
    openers = ["主人好！", "Hi~ 老板，", "数据汇报："]
    mood = "neutral"
    advice = ""
    if c['close'] > c['MA60']:
        if c['MA_Short'] > c['MA_Long']:
            advice = f"现在的 {name} 走势很漂亮，多头排列，你可以继续持有享受泡沫。"
            mood = "happy"
        else:
            advice = f"虽然还在牛熊线上方，但短期有回调压力，别追高哦。"
            mood = "neutral"
    else:
        advice = f"目前趋势偏弱，处于空头掌控中，建议多看少动，保住本金最重要。"
        mood = "worried"
    
    tech = ""
    if c['RSI'] < 30: tech = "不过我看 RSI 已经超卖了，短期随时可能反弹。"
    elif c['RSI'] > 75: tech = "而且 RSI 有点过热了，小心主力骗炮出货。"
    final_text = f"{random.choice(openers)} {advice} {tech} 切记，即使我看好，也要设好止损线 {c['close']*0.95:.2f} 保护自己。"
    return final_text, mood

def analyze_score(df):
    c = df.iloc[-1]; score=0; reasons=[]
    if c['MA_Short']>c['MA_Long']: score+=2; reasons.append("均线金叉 (短线看涨)")
    else: score-=2; reasons.append("均线死叉 (短线看跌)")
    if c['close']>c['MA_Long']: score+=1; reasons.append("站上长期生命线")
    else: reasons.append("跌破长期生命线")
    if c['DIF']>c['DEA']: score+=1; reasons.append("MACD 处于多头区域")
    
    action = "积极买入" if score>=4 else "持有/观望" if score>=0 else "减仓/卖出"
    color = "success" if score>=4 else "warning" if score>=0 else "error"
    pos_txt = "80%" if score>=4 else "50%" if score>=0 else "0%"
    atr = c['ATR14']
    stop_loss = c['close'] - 2*atr
    take_profit = c['close'] + 3*atr
    return score, action, color, stop_loss, take_profit, pos_txt, reasons

def plot_chart(df, name, flags, ma_s, ma_l, is_vip=False):
    # 根据权限决定显示内容的丰富度
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.55,0.1,0.15,0.2])
    fig.update_layout(dragmode=False, margin=dict(l=10, r=10, t=10, b=10))
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'), 1, 1)
    
    if flags.get('ma'):
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA_Short'], name=f'MA{ma_s}'), 1, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA_Long'], name=f'MA{ma_l}'), 1, 1)
    
    # VIP功能：BOLL, Gann, Fib, Chan 
    if is_vip:
        if flags.get('boll'):
            fig.add_trace(go.Scatter(x=df['date'], y=df['Upper'], line=dict(dash='dash', width=1), name='Upper'), 1, 1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['Lower'], line=dict(dash='dash', width=1), name='Lower'), 1, 1)
        ga, fi = get_drawing_lines(df)
        if flags.get('gann'):
            for k,v in ga.items(): fig.add_trace(go.Scatter(x=df['date'], y=v, mode='lines', line=dict(dash='dot', width=0.8), name=f'Gann {k}'), 1, 1)
        if flags.get('chan'):
            # 画笔
            chan_pts = []
            for i, row in df.iterrows():
                if row['F_Top']: chan_pts.append({'d': row['date'], 'v': row['high'], 't': 'top'})
                elif row['F_Bot']: chan_pts.append({'d': row['date'], 'v': row['low'], 't': 'bot'})
            if chan_pts:
                clean = [chan_pts[0]]
                for p in chan_pts[1:]:
                    if p['t'] != clean[-1]['t']: clean.append(p)
                    else:
                        if p['t']=='top' and p['v']>clean[-1]['v']: clean[-1]=p
                        elif p['t']=='bot' and p['v']<clean[-1]['v']: clean[-1]=p
                fig.add_trace(go.Scatter(x=[p['d'] for p in clean], y=[p['v'] for p in clean], mode='lines', line=dict(color='#2962ff', width=2), name='缠论笔'), 1, 1)

    # 副图
    if flags.get('vol'): fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name='Vol'), 2, 1)
    if flags.get('macd'): fig.add_trace(go.Bar(x=df['date'], y=df['HIST'], name='MACD'), 3, 1)
    if flags.get('kdj'): fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name='K'), 4, 1)

    fig.update_layout(height=600, xaxis_rangeslider_visible=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. 执行入口
# ==========================================
init_db()

# 登录逻辑
if not st.session_state.get('logged_in'):
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.markdown("<h1 style='text-align:center'>阿尔法量研 Pro V68</h1>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔑 登录", "📝 注册"])
        with tab1:
            u = st.text_input("账号")
            p = st.text_input("密码", type="password")
            if st.button("登录系统"):
                if verify_login(u.strip(), p): st.session_state["logged_in"] = True; st.session_state["user"] = u.strip(); st.rerun()
                else: st.error("账号或密码错误")
        with tab2:
            nu = st.text_input("新用户"); np1 = st.text_input("设置密码", type="password")
            if st.button("立即注册"):
                s, m = register_user(nu.strip(), np1)
                if s: st.success(m)
                else: st.error(m)
    st.stop()

# 用户状态检查
user = st.session_state["user"]
is_admin = (user == ADMIN_USER)
is_vip, vip_expiry = check_vip_status(user)
# 核心权限判定：如果是VIP，或者已经为该代码付费
has_access = is_vip or (st.session_state.code == st.session_state.paid_code)

# 侧边栏
with st.sidebar:
    st.markdown(f"""
    <div style='text-align: left; margin-bottom: 20px;'>
        <div class='brand-title'>阿尔法量研 <span style='color:#0071e3'>Pro</span></div>
        <div class='brand-slogan'>V68 商业化变现版</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 状态栏
    if is_vip:
        st.success(f"👑 尊贵VIP用户\n\n到期: {vip_expiry}")
    else:
        st.info("普通用户 (Free)")

    new_c = st.text_input("🔍 股票代码", st.session_state.code)
    if new_c != st.session_state.code: st.session_state.code = new_c; st.session_state.paid_code = ""; st.rerun()

    if not is_admin:
        with st.expander("💎 会员与充值", expanded=True):
            my_quota = load_users()[load_users()['username']==user]['quota'].iloc[0]
            st.write(f"当前积分: **{my_quota}**")
            
            tab_pay, tab_vip = st.tabs(["充值", "兑换VIP"])
            with tab_pay:
                st.write("1. 扫码(假) -> 2. 输入卡密")
                k_in = st.text_input("输入卡密")
                if st.button("充值"):
                    s, m = redeem_key(user, k_in)
                    if s: st.success(m); time.sleep(1); st.rerun()
                    else: st.error(m)
            with tab_vip:
                st.caption("✨ VIP特权：无限次查看所有深度数据")
                if st.button("🔥 100积分兑换30天VIP"):
                    if consume_quota(user, 100):
                        new_exp = extend_vip(user, 30)
                        st.balloons()
                        st.success(f"兑换成功！有效期至 {new_exp}")
                        time.sleep(2); st.rerun()
                    else:
                        st.error("积分不足！请先充值")

        with st.expander("⭐ 自选股"):
            current_wl = get_user_watchlist(user)
            for c in current_wl:
                if st.button(c, key=f"wl_{c}"): st.session_state.code = c; st.session_state.paid_code = ""; st.rerun()
            if st.button("❤️ 加入自选"): update_watchlist(user, st.session_state.code, "add"); st.rerun()
    
    if is_admin:
        with st.expander("👑 管理员控制台"):
            st.write("生成卡密")
            p_gen = st.number_input("面值", 10, 1000, 100)
            if st.button("生成"):
                k = generate_key(p_gen)
                st.code(k)

# --- 主界面 ---
name = get_name(st.session_state.code, "", None)
st.title(f"📈 {name} ({st.session_state.code})")

# 数据加载
with st.spinner("正在连接交易所数据..."):
    df = get_data_and_resample(st.session_state.code, "", "日线", "qfq")
    if df.empty:
        st.warning("⚠️ 数据获取失败，启用演示数据")
        df = generate_mock_data(300)

df = calc_full_indicators(df, ma_s, ma_l)
df = detect_patterns(df)
funda = get_fundamentals(st.session_state.code, "")

# === 区域 1：免费公开区 (Free Tier) ===
# 1. 红绿灯 (Always Free)
status, msg, css = check_market_status(df)
st.markdown(f"""
<div class="market-status-box {css}">
    <div style="display:flex; align-items:center;">
        <span class="status-icon">{'🟢' if status=='green' else '🔴' if status=='red' else '🟡'}</span>
        <div><div class="status-text">{msg}</div><div class="status-sub">基础趋势判断 (免费)</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. 核心价格 (Always Free)
l = df.iloc[-1]
color = "#ff3b30" if l['pct_change'] > 0 else "#00c853"
st.markdown(f"""
<div class="big-price-box">
    <span class="price-main" style="color:{color}">{l['close']:.2f}</span>
    <span class="price-sub" style="color:{color}">{l['pct_change']:.2f}%</span>
</div>
""", unsafe_allow_html=True)

# 3. 基础评分 (Always Free - 但隐藏具体理由)
sc, act, col, sl, tp, pos, reasons = analyze_score(df)
st.markdown(f"""
<div class="rating-box" style="margin-bottom:20px;">
    <div class="rating-score" style="color:{'#ff3b30' if sc>0 else '#00c853'}">{sc} <span style="font-size:14px">/10</span></div>
    <div class="rating-label">AI 综合打分 (6分以上推荐)</div>
</div>
""", unsafe_allow_html=True)

# === 区域 2：深度图表 (VIP/Paid 增强版) ===
# 如果没有权限，只显示基础MA，且不显示高级指标
st.markdown("### 📊 技术面透视")
plot_chart(df.tail(120), name, flags, ma_s, ma_l, is_vip=has_access)

# === 区域 3：付费锁定区 (The Gate) ===
if has_access:
    # ✅ 解锁状态：显示所有高级功能
    
    # 1. AI 投顾 (Unlocked)
    ai_text, ai_mood = generate_ai_copilot_text(df, name)
    st.markdown(f"""
    <div class="ai-chat-box">
        <div class="ai-content"><span style="font-weight:bold; color:#2962ff;">🤖 AI 深度解读：</span> {ai_text}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. 详细点位策略 (Unlocked)
    st.markdown(f"""
    <div class="strategy-card">
        <div class="strategy-title">🎯 机构级操作策略</div>
        <div class="strategy-grid">
            <div class="strategy-col"><span class="st-lbl">建议仓位</span><br><b>{pos}</b></div>
            <div class="strategy-col"><span class="st-lbl">止盈位</span><br><b style="color:#ff3b30">{tp:.2f}</b></div>
            <div class="strategy-col"><span class="st-lbl">止损位</span><br><b style="color:#00c853">{sl:.2f}</b></div>
        </div>
        <div class="reason-box">
            <b>💡 决策依据：</b><br>
            {'<br>'.join([f'• {r}' for r in reasons])}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.success("✅ 已解锁全部深度数据")

else:
    # 🔒 锁定状态：模糊遮罩 + 付费按钮
    st.markdown("""
    <div style="position:relative; height: 300px; overflow:hidden; border-radius:12px; border:1px solid #eee;">
        <div class="blur-lock">
            <div class="ai-chat-box" style="filter:blur(5px);">AI 正在分析...<br>内容已隐藏...</div>
            <div class="strategy-card" style="filter:blur(5px);">
                <div class="strategy-title">🎯 操作策略</div>
                <div>建议仓位: **%</div>
                <div>止盈位: ***.**</div>
            </div>
        </div>
        
        <div class="lock-overlay">
            <div class="lock-card">
                <h3>🔒 解锁深度研报</h3>
                <p style="color:#666; font-size:14px;">包含：AI解读、买卖点位、缠论结构、机构评级</p>
                <div style="margin-top:15px;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 解锁按钮逻辑
    c_pay1, c_pay2 = st.columns(2)
    with c_pay1:
        if st.button("🔓 支付 1 积分查看本股", type="primary", use_container_width=True):
            if consume_quota(user, 1):
                st.session_state.paid_code = st.session_state.code
                st.rerun()
            else:
                st.error("积分不足，请在左侧充值")
    with c_pay2:
        st.button("💎 开通 VIP 无限看", disabled=True, use_container_width=True, help="请在左侧侧边栏兑换")

st.divider()
st.caption("免责声明：本系统仅供量化研究，不构成投资建议。")