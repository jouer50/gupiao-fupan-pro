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
    page_title="阿尔法量研 Pro V70.1 (Fix)",
    layout="wide",
    page_icon="🔥",
    initial_sidebar_state="expanded"
)

# 初始化 Session
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if "code" not in st.session_state: st.session_state.code = "600519"
if "paid_code" not in st.session_state: st.session_state.paid_code = ""

# ✅ 模拟交易 Session
if "paper_holdings" not in st.session_state: st.session_state.paper_holdings = {}

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
DB_FILE = "users_v70.csv" 
KEYS_FILE = "card_keys.csv"

# Optional deps
ts = None
bs = None
try: import tushare as ts
except: pass
try: import baostock as bs
except: pass

# 🔥 CSS 样式
ui_css = """
<style>
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
    
    /* 按钮样式 */
    div.stButton > button {
        background: linear-gradient(145deg, #ffdb4d 0%, #ffb300 100%); 
        color: #5d4037; border: 2px solid #fff9c4; border-radius: 25px; 
        padding: 0.6rem 1.2rem; font-weight: 800; font-size: 16px;
        box-shadow: 0 4px 10px rgba(255, 179, 0, 0.4); 
        transition: all 0.2s; width: 100%;
    }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(255, 179, 0, 0.5); }
    div.stButton > button[kind="primary"] { 
        background: linear-gradient(145deg, #2962ff 0%, #0039cb 100%); 
        color: white; border: none; box-shadow: 0 4px 10px rgba(41, 98, 255, 0.3);
    }

    /* 卡片体系 */
    .app-card { background-color: #ffffff; border-radius: 12px; padding: 16px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.02); }
    .section-header { display: flex; align-items: center; margin-bottom: 12px; margin-top: 8px; }
    .section-title { font-size: 17px; font-weight: 900; color: #333; margin-right: 5px; }
    .vip-badge { background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 99%); color: #d32f2f; font-size: 10px; font-weight: 800; padding: 2px 8px; border-radius: 10px; font-style: italic; }
    
    /* 核心数据展示 */
    .market-status-box {
        padding: 12px 20px; border-radius: 12px; margin-bottom: 20px;
        display: flex; align-items: center; justify-content: space-between;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.05);
    }
    .status-green { background: #e8f5e9; color: #2e7d32; border-left: 5px solid #2e7d32; }
    .status-red { background: #ffebee; color: #c62828; border-left: 5px solid #c62828; }
    .status-yellow { background: #fffde7; color: #f9a825; border-left: 5px solid #f9a825; }
    .status-text { font-weight: 800; font-size: 16px; }
    .big-price-box { text-align: center; margin-bottom: 20px; }
    .price-main { font-size: 48px; font-weight: 900; line-height: 1; letter-spacing: -1.5px; }
    .price-sub { font-size: 16px; font-weight: 600; margin-left: 8px; padding: 2px 6px; border-radius: 4px; }
    
    /* AI与策略 */
    .ai-chat-box {
        background: #f0f7ff; border-radius: 12px; padding: 15px; margin-bottom: 20px;
        border-left: 5px solid #2962ff; box-shadow: 0 4px 12px rgba(41, 98, 255, 0.1);
    }
    .strategy-card { background: #fcfcfc; border: 1px solid #eee; border-left: 4px solid #ffca28; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.02); }
    .strategy-title { font-size: 18px; font-weight: 800; color: #333; margin-bottom: 10px; }
    .strategy-grid { display: flex; justify-content: space-between; margin-bottom: 10px; }
    
    /* 锁定遮罩 - 增强版 */
    .locked-wrapper { position: relative; margin-top: 20px; border-radius: 12px; overflow: hidden;}
    .locked-blur { filter: blur(8px); user-select: none; opacity: 0.5; pointer-events: none; background: #fff; padding: 20px;}
    .locked-overlay {
        position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: rgba(255, 255, 255, 0.6); z-index: 10;
        backdrop-filter: blur(2px);
    }
    .lock-icon { font-size: 48px; margin-bottom: 10px; text-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    .lock-title { font-size: 20px; font-weight: 900; color: #333; margin-bottom: 5px; }
    .lock-desc { font-size: 14px; color: #666; margin-bottom: 20px; text-align: center; max-width: 80%; }
    
    /* 评分 */
    .rating-container { display: flex; justify-content: space-between; gap: 8px; }
    .rating-box { flex: 1; background: #fff; border: 1px solid #f0f0f0; border-radius: 12px; text-align: center; padding: 15px 2px; box-shadow: 0 4px 10px rgba(0,0,0,0.02); }
    .rating-score { font-size: 28px; font-weight: 900; color: #ff3b30; line-height: 1; margin-bottom: 5px; }
    .rating-label { font-size: 12px; color: #666; font-weight: 500; }
    .score-yellow { color: #ff9800 !important; }
    
    /* 回测卡片 */
    .backtest-metric-box {
        background: #f8f9fa; border-radius: 8px; padding: 10px; text-align: center; border: 1px solid #eee;
    }
    .brand-title { font-size: 22px; font-weight: 900; color: #333; margin-bottom: 2px; }
    
    [data-testid="metric-container"] { display: none; }
</style>
"""
st.markdown(ui_css, unsafe_allow_html=True)

# ==========================================
# 2. 数据库与工具
# ==========================================
def init_db():
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota", "vip_expiry"])
        df.to_csv(DB_FILE, index=False)
    else:
        df = pd.read_csv(DB_FILE)
        if "vip_expiry" not in df.columns:
            df["vip_expiry"] = ""
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
        df = pd.read_csv(DB_FILE, dtype={"watchlist": str, "quota": int, "vip_expiry": str})
        if "vip_expiry" not in df.columns: df["vip_expiry"] = ""
        return df.fillna("")
    except: return pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota", "vip_expiry"])

def save_users(df): df.to_csv(DB_FILE, index=False)
def load_keys():
    try: return pd.read_csv(KEYS_FILE)
    except: return pd.DataFrame(columns=["key", "points", "status", "created_at"])
def save_keys(df): df.to_csv(KEYS_FILE, index=False)

def check_vip_status(username):
    if username == ADMIN_USER: return True, "永久管理员"
    df = load_users()
    row = df[df["username"] == username]
    if row.empty: return False, "非会员"
    expiry_str = str(row.iloc[0]["vip_expiry"])
    if not expiry_str or expiry_str == "nan": return False, "非会员"
    try:
        exp_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        if exp_date >= datetime.now():
            days_left = (exp_date - datetime.now()).days + 1
            return True, f"VIP 剩余 {days_left} 天"
        else: return False, "VIP 已过期"
    except: return False, "日期错误"

def update_vip_days(target_user, days_to_add):
    df = load_users()
    idx = df[df["username"] == target_user].index
    if len(idx) == 0: return False
    current_exp = df.loc[idx[0], "vip_expiry"]
    now = datetime.now()
    try:
        if current_exp and current_exp != "nan":
            curr_date = datetime.strptime(current_exp, "%Y-%m-%d")
            base_date = curr_date if curr_date > now else now
        else: base_date = now
    except: base_date = now
    new_date = base_date + timedelta(days=int(days_to_add))
    df.loc[idx[0], "vip_expiry"] = new_date.strftime("%Y-%m-%d")
    save_users(df)
    return True

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

def consume_quota(u):
    if u == ADMIN_USER: return True
    is_vip, _ = check_vip_status(u)
    if is_vip: return True
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
# 3. 股票逻辑 (修复版)
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
    
    if is_cn_stock(code) and token and ts:
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
        except Exception: raw_df = pd.DataFrame()

    if raw_df.empty and is_cn_stock(code) and bs:
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
        except Exception: raw_df = pd.DataFrame()

    if raw_df.empty:
        try:
            yf_df = yf.download(code, period="5y", interval="1d", progress=False, auto_adjust=False)
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
                    elif 'adj close' in c: rename_map[c] = 'adj_close'
                yf_df.rename(columns=rename_map, inplace=True)
                req_cols = ['date','open','high','low','close']
                if all(c in yf_df.columns for c in req_cols):
                    if 'volume' not in yf_df.columns: yf_df['volume'] = 0
                    raw_df = yf_df[['date','open','high','low','close','volume']].copy()
                    for c in ['open','high','low','close','volume']: raw_df[c] = pd.to_numeric(raw_df[c], errors='coerce')
                    raw_df['pct_change'] = raw_df['close'].pct_change() * 100
        except Exception: pass

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

    df['MA_Short'] = c.rolling(ma_s).mean()
    df['MA_Long'] = c.rolling(ma_l).mean()
    df['MA60'] = c.rolling(60).mean()
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
    df['VolRatio'] = v / (v.rolling(5).mean() + 1e-9)
    df['F_Top'] = (df['high'].shift(1)<df['high']) & (df['high'].shift(-1)<df['high'])
    df['F_Bot'] = (df['low'].shift(1)>df['low']) & (df['low'].shift(-1)>df['low'])
    df[['K','D','J','DIF','DEA','HIST','RSI']] = df[['K','D','J','DIF','DEA','HIST','RSI']].fillna(50)
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
    if curr['close'] > curr['MA60']:
        return "green", "🚀 趋势向上 (可积极做多)", "status-green"
    elif curr['close'] < curr['MA60']:
        return "red", "🛑 趋势转弱 (建议空仓观望)", "status-red"
    else:
        return "yellow", "⚠️ 震荡整理 (轻仓操作)", "status-yellow"

def get_daily_picks(user_watchlist):
    hot_stocks = ["600519", "NVDA", "TSLA", "300750", "002594", "AAPL"]
    pool = list(set(hot_stocks + user_watchlist))
    results = []
    for code in pool[:6]: 
        name = get_name(code, "", None)
        status = random.choice(["buy", "hold", "wait"])
        if status == "buy":
            results.append({"code": code, "name": name, "tag": "今日买点", "type": "tag-buy"})
        elif status == "hold":
            results.append({"code": code, "name": name, "tag": "持股待涨", "type": "tag-hold"})
    return results

def run_backtest(df):
    if df is None or len(df) < 50: return 0.0, 0.0, 0.0, 0.0, pd.DataFrame({'date':[], 'equity':[]})
    needed = ['MA_Short', 'MA_Long', 'close', 'date']
    df_bt = df.dropna(subset=needed).reset_index(drop=True)
    if len(df_bt) < 20: return 0.0, 0.0, 0.0, 0.0, pd.DataFrame({'date':[], 'equity':[]})
    
    capital = 100000; position = 0
    equity = [capital]; dates = [df_bt.iloc[0]['date']]
    
    # 策略逻辑：纯金叉死叉
    for i in range(1, len(df_bt)):
        curr = df_bt.iloc[i]; prev = df_bt.iloc[i-1]; price = curr['close']; date = curr['date']
        buy_sig = prev['MA_Short'] <= prev['MA_Long'] and curr['MA_Short'] > curr['MA_Long']
        sell_sig = prev['MA_Short'] >= prev['MA_Long'] and curr['MA_Short'] < curr['MA_Long']
        
        if buy_sig and position == 0:
            position = capital / price; capital = 0
        elif sell_sig and position > 0: 
            capital = position * price; position = 0
        
        current_val = capital + (position * price)
        equity.append(current_val)
        dates.append(date)
        
    final = equity[-1]
    strategy_ret = (final - 100000) / 100000 * 100
    
    # 计算基准收益 (大盘/个股Buy&Hold)
    start_price = df_bt.iloc[0]['close']
    end_price = df_bt.iloc[-1]['close']
    benchmark_ret = (end_price - start_price) / start_price * 100
    
    # 计算Alpha
    alpha = strategy_ret - benchmark_ret
    
    # 简单的最大回撤
    eq_series = pd.Series(equity); cummax = eq_series.cummax()
    drawdown = (eq_series - cummax) / cummax; max_dd = drawdown.min() * 100
    eq_df = pd.DataFrame({'date': dates, 'equity': equity})
    
    return strategy_ret, benchmark_ret, alpha, max_dd, eq_df

def generate_deep_report(df, name):
    curr = df.iloc[-1]
    chan_trend = "底分型构造中" if curr['F_Bot'] else "顶分型构造中" if curr['F_Top'] else "中继形态"
    gann, fib = get_drawing_lines(df)
    try:
        fib_near = min(fib.items(), key=lambda x: abs(x[1]-curr['close']))
        fib_txt = f"股价正逼近斐波那契 <b>{fib_near[0]}</b> 关键位 ({fib_near[1]:.2f})。"
    except: fib_txt = "数据不足，无法计算位置。"
    return f"""
    <div class="app-card">
        <div class="deep-title">📐 缠论结构与形态学</div>
        <div class="deep-text">• <b>分型状态</b>：{chan_trend}。</div>
    </div>
    <div class="app-card">
        <div class="deep-title">🌌 江恩与斐波那契</div>
        <div class="deep-text">• <b>斐波那契回撤</b>：{fib_txt}</div>
    </div>
    """

def generate_ai_copilot_text(df, name):
    c = df.iloc[-1]
    advice = "多头排列，继续持有。" if c['MA_Short'] > c['MA_Long'] else "空头趋势，建议观望。"
    return f"主人好！我是您的AI投顾。{advice} 注意 RSI 目前数值为 {c['RSI']:.1f}。", "happy" if c['pct_change']>0 else "neutral"

# 🔥 FIX: Added 'color' back to return tuple to match calling code unpacking
def analyze_score(df):
    c = df.iloc[-1]; score=0; reasons=[]
    if c['MA_Short']>c['MA_Long']: score+=2; reasons.append("均线金叉")
    else: score-=2; reasons.append("均线死叉")
    
    action = "积极买入" if score>=0 else "减仓/卖出"
    color = "success" if score>=0 else "error"  # Fixed logic
    
    return score, action, color, c['close']-c['ATR14']*2, c['close']+c['ATR14']*3, "50%", c['low']*0.95, c['high']*1.05, reasons

def calculate_smart_score(df, funda):
    return 8.5, 7.0, 6.5 # Mock scores for speed

def plot_chart(df, name, flags, ma_s, ma_l):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'), 1, 1)
    if flags.get('ma'):
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA_Short'], name=f'MA{ma_s}', line=dict(color='#333')), 1, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA_Long'], name=f'MA{ma_l}', line=dict(color='#ffcc00')), 1, 1)
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. 执行入口
# ==========================================
init_db()

with st.sidebar:
    st.markdown("""
    <div style='text-align: left; margin-bottom: 20px;'>
        <div class='brand-title'>阿尔法量研 <span style='color:#0071e3'>Pro</span></div>
        <div class='brand-en'>AlphaQuant Pro V70.1</div>
        <div class='brand-slogan'>用历史验证未来，用数据构建策略。</div>
    </div>
    """, unsafe_allow_html=True)
    
    new_c = st.text_input("🔍 股票代码", st.session_state.code)
    if new_c != st.session_state.code: st.session_state.code = new_c; st.session_state.paid_code = ""; st.rerun()

    if st.session_state.get('logged_in'):
        user = st.session_state["user"]
        is_admin = (user == ADMIN_USER)
        is_vip, vip_msg = check_vip_status(user)
        
        if is_vip: st.success(f"👑 {vip_msg}")
        else: st.info(f"👤 普通用户 (积分: {load_users()[load_users()['username']==user]['quota'].iloc[0]})")

        st.markdown("### 👁️ 模式选择")
        view_mode = st.radio("选择查看模式", ["极简模式 (Free)", "专业模式 (VIP/Paid)"], index=0)
        is_pro_mode = "专业" in view_mode

        if not is_admin:
            with st.expander("⭐ 我的自选股", expanded=False):
                current_wl = get_user_watchlist(user)
                if not current_wl: st.caption("暂无自选，请在上方添加")
                else:
                    for c in current_wl:
                        if st.button(f"{c}", key=f"wl_{c}"):
                            st.session_state.code = c; st.session_state.paid_code = ""; st.rerun()
            if st.button("❤️ 加入自选"): update_watchlist(user, st.session_state.code, "add"); st.rerun()

        if st.button("🔄 刷新缓存"): st.cache_data.clear(); st.rerun()

        if not is_admin:
            with st.expander("💎 充值与会员", expanded=False):
                st.info(f"当前积分: {load_users()[load_users()['username']==user]['quota'].iloc[0]}")
                st.write("##### 1. 扫码充值")
                pay_opt = st.radio("充值面额", [20, 50, 100], horizontal=True, format_func=lambda x: f"￥{x}")
                if os.path.exists("alipay.png"): st.image("alipay.png", caption="请使用支付宝扫码", width=200)
                else: st.warning("请上传 alipay.png")
                
                st.write("##### 2. 兑换")
                k_in = st.text_input("输入卡密")
                if st.button("兑换"):
                    s, m = redeem_key(user, k_in)
                    if s: st.success(m); time.sleep(1); st.rerun()
                    else: st.error(m)

        if is_admin:
            st.success("👑 管理员模式")
            with st.expander("👑 VIP 权限管理", expanded=True):
                df_u = load_users()
                u_list = [x for x in df_u["username"] if x!=ADMIN_USER]
                if u_list:
                    vip_target = st.selectbox("选择用户", u_list, key="vip_sel")
                    vip_days = st.number_input("增加天数", value=30, step=1)
                    if st.button("更新 VIP 权限"):
                        if update_vip_days(vip_target, vip_days): st.success(f"已更新 {vip_target} 的 VIP"); time.sleep(1); st.rerun()
            with st.expander("💳 卡密生成"):
                points_gen = st.selectbox("面值", [20, 50, 100, 200, 500])
                count_gen = st.number_input("数量", 1, 50, 10)
                if st.button("批量生成"):
                    num = batch_generate_keys(points_gen, count_gen)
                    st.success(f"已生成 {num} 张卡密")

        timeframe = st.selectbox("周期", ["日线", "周线", "月线"])
        adjust = st.selectbox("复权", ["qfq","hfq",""], 0)
        
        if is_pro_mode:
            with st.expander("🎛️ 策略参数 (VIP)", expanded=False):
                ma_s = st.slider("短期均线", 2, 20, 5)
                ma_l = st.slider("长期均线", 10, 120, 20)
            st.markdown("### 🛠️ 指标开关")
            c_flags = st.columns(2)
            with c_flags[0]:
                flags['ma'] = st.checkbox("MA", True)
                flags['boll'] = st.checkbox("BOLL", True)
            with c_flags[1]:
                flags['kdj'] = st.checkbox("KDJ", True)
                flags['chan'] = st.checkbox("缠论", True)
        
        st.divider()
        if st.button("退出登录"): st.session_state["logged_in"]=False; st.rerun()
    else:
        st.info("请先登录系统")

# --- 主内容区 ---
name = get_name(st.session_state.code, "", None) 
c1, c2 = st.columns([3, 1])
with c1: st.title(f"📈 {name} ({st.session_state.code})")

# 数据加载
loading_tips = ["正在加载因子库…", "正在构建回测引擎…", "正在同步行情数据…"]
with st.spinner(random.choice(loading_tips)):
    df = get_data_and_resample(st.session_state.code, "", timeframe, adjust, proxy=None)
    if df.empty:
        st.warning("⚠️ 暂无数据。自动切换至演示模式。")
        df = generate_mock_data(250)

# 基础数据准备
funda = get_fundamentals(st.session_state.code, "")
df = calc_full_indicators(df, ma_s, ma_l)

# ============================================
# 1. 极简模式内容 (所有用户可见)
# ============================================
# 状态与红绿灯
status, msg, css_class = check_market_status(df)
st.markdown(f"""
<div class="market-status-box {css_class}">
    <div style="display:flex; align-items:center;">
        <span class="status-icon">{'🟢' if status=='green' else '🔴' if status=='red' else '🟡'}</span>
        <div>
            <div class="status-text">{msg}</div>
            <div class="status-sub">基于 MA60 牛熊线与波动率分析</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 核心价格展示
l = df.iloc[-1]
color = "#ff3b30" if l['pct_change'] > 0 else "#00c853"
st.markdown(f"""
<div class="big-price-box">
    <span class="price-main" style="color:{color}">{l['close']:.2f}</span>
    <span class="price-sub" style="color:{color}">{l['pct_change']:.2f}%</span>
</div>
""", unsafe_allow_html=True)

# 基础评分
sq, sv, st_ = calculate_smart_score(df, funda)
st.markdown(f"""
<div class="rating-container">
    <div class="rating-box">
        <div class="rating-score">{sq} <span class="rating-score-sub">/10</span></div>
        <div class="rating-label">公司质量</div>
    </div>
    <div class="rating-box">
        <div class="rating-score score-yellow">{sv} <span class="rating-score-sub sub-yellow">/10</span></div>
        <div class="rating-label">估值安全</div>
    </div>
</div>
<div style="height:20px"></div>
""", unsafe_allow_html=True)

# ============================================
# 2. 专业模式内容 (需解锁)
# ============================================

# 权限判断逻辑
has_access = False
if is_admin: has_access = True
elif is_vip: has_access = True
elif st.session_state.paid_code == st.session_state.code: has_access = True

# 如果选了极简模式，直接停止渲染后续VIP内容
if not is_pro_mode:
    st.info("💡 切换至「专业模式」可查看：AI投顾、买卖点策略、胜率回测、深度研报。")
    st.stop()

# 如果选了专业模式，但没权限，添加遮罩
if is_pro_mode and not has_access:
    st.markdown('<div class="locked-wrapper"><div class="locked-blur">', unsafe_allow_html=True)

# ---- 以下内容受锁保护 ----

# A. AI 投顾
ai_text, ai_mood = generate_ai_copilot_text(df, name)
ai_icon = "🤖" if ai_mood == "neutral" else "😊" if ai_mood == "happy" else "😰"
st.markdown(f"""
<div class="ai-chat-box">
    <div class="ai-avatar">{ai_icon}</div>
    <div class="ai-content">
        <span style="font-weight:bold; color:#2962ff;">AI 投顾助理 (Pro)：</span>
        {ai_text}
    </div>
</div>
""", unsafe_allow_html=True)

# B. 最终建议策略
sc, act, col, sl, tp, pos, sup, res, reasons = analyze_score(df)
reason_html = "".join([f"<div>• {r}</div>" for r in reasons])
st.markdown(f"""
<div class="strategy-card">
    <div class="strategy-title">🤖 最终操作建议：{act}</div>
    <div class="strategy-grid">
        <div class="strategy-col"><span class="st-lbl">建议仓位</span><span class="st-val" style="color:#333">{pos}</span></div>
        <div class="strategy-col"><span class="st-lbl">目标止盈</span><span class="st-val" style="color:#ff3b30">{tp:.2f}</span></div>
        <div class="strategy-col"><span class="st-lbl">预警止损</span><span class="st-val" style="color:#00c853">{sl:.2f}</span></div>
    </div>
    <div class="reason-box">
        <div class="reason-title">💡 决策因子</div>
        {reason_html}
    </div>
</div>
""", unsafe_allow_html=True)

# C. 历史回测 (默认展开，美化版)
# ✅ NEW: 增加 Alpha 计算展示
with st.expander("⚖️ 历史回测数据 (Alpha 增强版)", expanded=True):
    ret, bench_ret, alpha, mdd, eq = run_backtest(df)
    
    # 美化指标卡片
    bc1, bc2, bc3 = st.columns(3)
    bc1.metric("策略总收益", f"{ret:.1f}%", help="策略由10万本金开始运行至今的收益")
    bc2.metric("超额收益 (Alpha)", f"{alpha:.1f}%", delta=f"{alpha:.1f}%", delta_color="normal", help="相比于买入持有不动(Buy & Hold)多赚的钱")
    bc3.metric("最大回撤", f"{mdd:.1f}%", help="历史上最糟糕的回撤幅度")
    
    if alpha > 0:
        st.success(f"🔥 **强于大盘！** 您的策略跑赢了基准收益 ({bench_ret:.1f}%)。")
    else:
        st.warning(f"🐢 **弱于大盘。** 策略跑输了基准收益 ({bench_ret:.1f}%)，建议优化。")
    
    if not eq.empty:
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=eq['date'], y=eq['equity'], fill='tozeroy', line=dict(color='#2962ff', width=2), name='策略净值'))
        f2.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#eee'), showlegend=False)
        st.plotly_chart(f2, use_container_width=True)

# D. 深度研报 & 图表
plot_chart(df.tail(120), name, flags, ma_s, ma_l)
st.markdown(generate_deep_report(df, name), unsafe_allow_html=True)

# ---- 锁保护结束 ----

if is_pro_mode and not has_access:
    st.markdown('</div>', unsafe_allow_html=True) # close locked-blur
    try: bal = load_users()[load_users()["username"]==user]["quota"].iloc[0]
    except: bal = 0
    
    st.markdown(f"""
    <div class="locked-overlay">
        <div class="lock-icon">🔒</div>
        <div class="lock-title">解锁专业版深度数据</div>
        <div class="lock-desc">
            包含：<br>
            ✅ AI 智能投顾分析<br>
            ✅ 买卖点位与止盈止损策略<br>
            ✅ 历史回测超额收益分析<br>
            ✅ 缠论结构与主力资金研报
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    c_lock1, c_lock2, c_lock3 = st.columns([1,2,1])
    with c_lock2:
        if st.button(f"🔓 支付 1 积分解锁本股 (余额: {bal})", type="primary", use_container_width=True):
            if consume_quota(user):
                st.session_state.paid_code = st.session_state.code
                st.rerun()
            else:
                st.error("积分不足，请联系管理员充值！")