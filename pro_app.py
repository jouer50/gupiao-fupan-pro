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
    page_title="阿尔法量研 Pro V78 (SimTrade)",
    layout="wide",
    page_icon="🔥",
    initial_sidebar_state="expanded"
)

# 初始化 Session
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if "code" not in st.session_state: st.session_state.code = "600519"
if "paid_code" not in st.session_state: st.session_state.paid_code = "" 

# ✅ 模拟交易数据结构初始化 (升级版)
# 结构: {'cash': 1000000, 'holdings': {}, 'history': []}
if "paper_account" not in st.session_state: 
    st.session_state.paper_account = {
        "cash": 1000000.0,  # 初始资金 100万
        "holdings": {},     # 持仓字典
        "history": []       # 交易记录
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

# 🔥 CSS 样式 (UI Fix & Optimization)
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
    .app-card { background-color: #ffffff; border-radius: 12px; padding: 16px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.02); }
    .vip-badge { background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 99%); color: #d32f2f; font-size: 10px; font-weight: 800; padding: 2px 8px; border-radius: 10px; font-style: italic; }
    .ai-chat-box {
        background: #f0f7ff; border-radius: 12px; padding: 15px; margin-bottom: 20px;
        border-left: 5px solid #2962ff; box-shadow: 0 4px 12px rgba(41, 98, 255, 0.1);
    }
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
    .rating-container { display: flex; justify-content: space-between; gap: 8px; }
    .rating-box { flex: 1; background: #fff; border: 1px solid #f0f0f0; border-radius: 12px; text-align: center; padding: 15px 2px; box-shadow: 0 4px 10px rgba(0,0,0,0.02); }
    .rating-score { font-size: 28px; font-weight: 900; color: #ff3b30; line-height: 1; margin-bottom: 5px; }
    .rating-label { font-size: 12px; color: #666; font-weight: 500; }
    .score-yellow { color: #ff9800 !important; }
    
    .brand-title { font-size: 22px; font-weight: 900; color: #333; margin-bottom: 2px; }
    
    /* 回测看板样式 */
    .bt-container { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.04); margin-bottom: 20px; border: 1px solid #f0f0f0; }
    .bt-header { font-size: 18px; font-weight: 800; color: #1d1d1f; margin-bottom: 15px; border-left: 4px solid #2962ff; padding-left: 10px; }
    .bt-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }
    .bt-card { background: #f9f9f9; padding: 15px; border-radius: 10px; text-align: center; transition: all 0.3s; }
    .bt-card:hover { transform: translateY(-3px); box-shadow: 0 5px 15px rgba(0,0,0,0.05); background: #fff; border: 1px solid #e0e0e0; }
    .bt-val { font-size: 24px; font-weight: 900; color: #333; }
    .bt-lbl { font-size: 12px; color: #666; margin-top: 5px; }
    .bt-pos { color: #d32f2f; }
    .bt-neu { color: #333; }
    .bt-neg { color: #2e7d32; }
    .bt-tag { display: inline-block; padding: 2px 8px; font-size: 10px; border-radius: 4px; margin-top: 2px; }
    .tag-alpha { background: rgba(255, 59, 48, 0.1); color: #ff3b30; }

    /* 锁定状态样式 */
    .locked-container { position: relative; overflow: hidden; }
    .locked-blur { filter: blur(6px); user-select: none; opacity: 0.6; pointer-events: none; }
    .locked-overlay {
        position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: rgba(255, 255, 255, 0.4); z-index: 10;
    }
    .lock-icon { font-size: 40px; margin-bottom: 10px; }
    .lock-title { font-size: 18px; font-weight: 900; color: #333; margin-bottom: 5px; }
    .lock-desc { font-size: 13px; color: #666; margin-bottom: 15px; }
    [data-testid="metric-container"] { display: none; }
    .deep-title { font-size: 16px; font-weight: 700; color: #1d1d1f; margin-bottom: 8px; border-left: 3px solid #ff9800; padding-left: 8px; }
    .deep-text { font-size: 13px; color: #444; line-height: 1.6; }
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
        cols_needed = ["vip_expiry", "paper_json"]
        updated = False
        for c in cols_needed:
            if c not in df.columns:
                df[c] = ""
                updated = True
        if updated:
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
        df = pd.read_csv(DB_FILE, dtype={"watchlist": str, "quota": int, "vip_expiry": str, "paper_json": str})
        return df.fillna("")
    except: return pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota", "vip_expiry", "paper_json"])

def save_users(df): df.to_csv(DB_FILE, index=False)

def save_user_holdings(username):
    if username == ADMIN_USER: return
    df = load_users()
    idx = df[df["username"] == username].index
    if len(idx) > 0:
        # 保存整个 paper_account 结构
        holdings_json = json.dumps(st.session_state.paper_account)
        df.loc[idx[0], "paper_json"] = holdings_json
        save_users(df)

def load_user_holdings(username):
    if username == ADMIN_USER: return
    df = load_users()
    row = df[df["username"] == username]
    if not row.empty:
        json_str = str(row.iloc[0]["paper_json"])
        if json_str and json_str != "nan":
            try:
                data = json.loads(json_str)
                # 兼容旧版本数据结构
                if "cash" not in data:
                    st.session_state.paper_account = {
                        "cash": 1000000.0,
                        "holdings": data, # 假设旧数据全是 holdings
                        "history": []
                    }
                else:
                    st.session_state.paper_account = data
            except:
                st.session_state.paper_account = {"cash": 1000000.0, "holdings": {}, "history": []}
    
    # 确保初始化
    if "cash" not in st.session_state.paper_account:
        st.session_state.paper_account["cash"] = 1000000.0

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
    
    if match.empty: return False, "❌ 无效卡密或已被使用"
    
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

def register_user(u, p, initial_quota=10):
    if u == ADMIN_USER: return False, "保留账号"
    df = load_users()
    if u in df["username"].values: return False, "用户已存在"
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(p.encode(), salt).decode()
    # 注册时初始化 paper_json 为新的结构
    init_paper = json.dumps({"cash": 1000000.0, "holdings": {}, "history": []})
    new_row = {"username": u, "password_hash": hashed, "watchlist": "", "quota": initial_quota, "vip_expiry": "", "paper_json": init_paper}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_users(df)
    return True, f"注册成功，已获赠 {initial_quota} 积分！"

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
        except Exception: 
            raw_df = pd.DataFrame() 
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
        except Exception:
            raw_df = pd.DataFrame()
    if raw_df.empty:
        try:
            yf_df = yf.download(code, period="5y", interval="1d", progress=False, auto_adjust=False)
            if not yf_df.empty:
                if isinstance(yf_df.columns, pd.MultiIndex):
                    yf_df.columns = yf_df.columns.get_level_values(0)
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
                    for c in ['open','high','low','close','volume']: 
                        raw_df[c] = pd.to_numeric(raw_df[c], errors='coerce')
                    raw_df['pct_change'] = raw_df['close'].pct_change() * 100
        except Exception:
            pass

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
    if df is None or len(df) < 50: return 0.0, 0.0, 0.0, [], [], pd.DataFrame({'date':[], 'equity':[]})
    needed = ['MA_Short', 'MA_Long', 'close', 'date']
    df_bt = df.dropna(subset=needed).reset_index(drop=True)
    if len(df_bt) < 20: return 0.0, 0.0, 0.0, [], [], pd.DataFrame({'date':[], 'equity':[]})
    
    capital = 100000; position = 0
    buy_signals = []; sell_signals = []; equity = [capital]; dates = [df_bt.iloc[0]['date']]
    
    trade_count = 0; wins = 0; entry_price = 0
    
    for i in range(1, len(df_bt)):
        curr = df_bt.iloc[i]; prev = df_bt.iloc[i-1]; price = curr['close']; date = curr['date']
        buy_sig = prev['MA_Short'] <= prev['MA_Long'] and curr['MA_Short'] > curr['MA_Long']
        sell_sig = prev['MA_Short'] >= prev['MA_Long'] and curr['MA_Short'] < curr['MA_Long']
        
        if buy_sig and position == 0:
            position = capital / price; capital = 0; buy_signals.append(date)
            entry_price = price
        elif sell_sig and position > 0: 
            capital = position * price; position = 0; sell_signals.append(date)
            trade_count += 1
            if price > entry_price: wins += 1
        
        current_val = capital + (position * price)
        equity.append(current_val)
        dates.append(date)
        
    final = equity[-1]; ret = (final - 100000) / 100000 * 100
    win_rate = (wins / trade_count * 100) if trade_count > 0 else 0.0
    eq_series = pd.Series(equity); cummax = eq_series.cummax()
    drawdown = (eq_series - cummax) / cummax; max_dd = drawdown.min() * 100
    first_price = df_bt.iloc[0]['close']
    bench_equity = [(p / first_price) * 100000 for p in df_bt['close']]
    
    eq_df = pd.DataFrame({
        'date': dates, 
        'equity': equity,
        'benchmark': bench_equity[:len(dates)] 
    })
    return ret, win_rate, max_dd, buy_signals, sell_signals, eq_df

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
    if c['RSI'] < 30: tech = "不过我看 RSI 已经超卖了，短期随时可能反弹，如果你是左侧交易者可以轻仓试错。"
    elif c['RSI'] > 75: tech = "而且 RSI 有点过热了，小心主力骗炮出货，记得推高止损。"
    if c['VolRatio'] > 1.8: tech += " 另外，今天量能放得很大，主力有动作！"
    final_text = f"{random.choice(openers)} {advice} {tech} 切记，即使我看好，也要设好止损线 {c['close']*0.95:.2f} 保护自己。"
    return final_text, mood

# ✅ 极致简化版策略卡片：使用纯 HTML 表格，拒绝花哨CSS
def generate_strategy_card(df, name):
    if df.empty: return ""
    c = df.iloc[-1]
    
    # 1. 核心数据计算
    support = df['low'].tail(20).min()
    resistance = df['high'].tail(20).max()
    stop_loss = c['close'] - 2.0 * c['ATR14']
    take_profit = c['close'] + 3.0 * c['ATR14']
    
    # 2. 策略逻辑判断
    action = "观望 Wait"
    position = "0成"
    
    if c['MA_Short'] > c['MA_Long'] and c['close'] > c['MA60']:
        action = "🟢 积极买入"
        position = "6-8成"
    elif c['MA_Short'] < c['MA_Long']:
        action = "🔴 减仓止盈"
        position = "0-3成"
    elif c['close'] < c['MA60']:
        action = "⚠️ 反弹减持"
        position = "2-4成"
        
    # 3. 纯表格输出，无CSS Grid
    html = f"""
    <div class="app-card">
        <h4 style="margin-top:0;">🛡️ 交易计划: {action} (仓位: {position})</h4>
        <table width="100%" border="1" cellspacing="0" cellpadding="8" style="text-align: center; border-collapse: collapse; border: 1px solid #ddd;">
            <tr>
                <td width="50%" style="background-color: #f9f9f9;">🎯 压力位 (Resistance)<br><b style="font-size:16px;">{resistance:.2f}</b></td>
                <td width="50%" style="background-color: #f9f9f9;">⚓ 支撑位 (Support)<br><b style="font-size:16px;">{support:.2f}</b></td>
            </tr>
            <tr>
                <td>💰 建议止盈 (Target)<br><b style="font-size:16px;">{take_profit:.2f}</b></td>
                <td>🛡️ 建议止损 (Stop)<br><b style="font-size:16px;">{stop_loss:.2f}</b></td>
            </tr>
        </table>
        <div style="font-size: 12px; color: gray; margin-top: 5px;">* 止损基于2倍ATR波动率，压力支撑基于20日极值</div>
    </div>
    """
    return html

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
        chan_pts = []
        for i, row in df.iterrows():
            if row['F_Top']: chan_pts.append({'d': row['date'], 'v': row['high'], 't': 'top'})
            elif row['F_Bot']: chan_pts.append({'d': row['date'], 'v': row['low'], 't': 'bot'})
        if chan_pts:
            clean_pts = [chan_pts[0]]
            for p in chan_pts[1:]:
                if p['t'] != clean_pts[-1]['t']: clean_pts.append(p)
                else:
                    if p['t'] == 'top' and p['v'] > clean_pts[-1]['v']: clean_pts[-1] = p
                    elif p['t'] == 'bot' and p['v'] < clean_pts[-1]['v']: clean_pts[-1] = p
            cx = [p['d'] for p in clean_pts]; cy = [p['v'] for p in clean_pts]
            fig.add_trace(go.Scatter(x=cx, y=cy, mode='lines', line=dict(color='#2962ff', width=2), name='缠论笔'), 1, 1)
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
# 5. 执行入口
# ==========================================
init_db()

with st.sidebar:
    st.markdown("""
    <div style='text-align: left; margin-bottom: 20px;'>
        <div class='brand-title'>阿尔法量研 <span style='color:#0071e3'>Pro</span></div>
        <div class='brand-en'>AlphaQuant Pro V78</div>
        <div class='brand-slogan'>用历史验证未来，用数据构建策略。</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.get('logged_in'):
        user = st.session_state["user"]
        is_admin = (user == ADMIN_USER)
        
        if not is_admin:
            with st.expander("💎 会员与充值中心", expanded=False):
                st.info(f"当前积分: {load_users()[load_users()['username']==user]['quota'].iloc[0]}")
                st.markdown("""
                **💰 充值说明 (1元 = 2积分)**
                * 充值时请备注您的用户名。
                * 有问题咨询微信公众号：`lubingxpiaoliuji`
                """)
                
                if os.path.exists("alipay.png"):
                    st.image("alipay.png", caption="请使用支付宝扫码 (备注用户名)", width=200)
                
                st.markdown("---")
                st.write("##### 卡密兑换")
                k_in = st.text_input("输入卡密")
                if st.button("兑换"):
                    s, m = redeem_key(user, k_in)
                    if s: st.success(m); time.sleep(1); st.rerun()
                    else: st.error(m)

    new_c = st.text_input("🔍 股票代码", st.session_state.code)
    if new_c != st.session_state.code: st.session_state.code = new_c; st.session_state.paid_code = ""; st.rerun()

    if st.session_state.get('logged_in'):
        is_vip, vip_msg = check_vip_status(user)
        
        # 加载用户持仓数据（含资金）
        load_user_holdings(user)
        
        if is_vip: st.success(f"👑 {vip_msg}")
        else: st.info(f"👤 普通用户")

        st.markdown("### 👁️ 视觉模式")
        view_mode = st.radio("显示模式", ["极简模式", "专业模式"], index=0, horizontal=True)
        
        is_unlocked = False
        if is_admin or is_vip or st.session_state.paid_code == st.session_state.code:
            is_unlocked = True

        if view_mode == "专业模式" and not is_unlocked:
            st.warning("🔒 专业模式需解锁 (1积分/次)")
            if st.button("🔓 立即解锁", key="sidebar_unlock", type="primary"):
                if consume_quota(user):
                    st.session_state.paid_code = st.session_state.code
                    st.success("已解锁！")
                    st.rerun()
                else:
                    st.error("积分不足，请充值")
            is_pro = False 
        else:
            is_pro = (view_mode == "专业模式")
        
        if not is_admin:
            st.markdown("### 🎯 每日精选策略")
            user_wl = get_user_watchlist(user)
            picks = get_daily_picks(user_wl)
            for pick in picks:
                if st.button(f"{pick['tag']} | {pick['name']}", key=f"pick_{pick['code']}"):
                    st.session_state.code = pick['code']
                    st.rerun()
            st.divider()
        
        # ✅✅✅✅✅✅✅✅✅ 重构后的模拟交易模块 ✅✅✅✅✅✅✅✅✅
        if not is_admin:
            with st.expander("🎮 模拟交易 (仿真账户)", expanded=True):
                # 1. 账户核心数据准备
                paper = st.session_state.paper_account
                cash = paper.get("cash", 1000000.0)
                holdings = paper.get("holdings", {})
                
                # 获取当前股价
                try:
                    curr_price = float(yf.Ticker(process_ticker(st.session_state.code)).fast_info.last_price)
                except: curr_price = 0
                
                # 计算总资产
                total_mkt_val = 0
                for c_code, c_data in holdings.items():
                    # 简单估算，如果不是当前代码，无法实时获取所有价格，这里简化处理：
                    # 仅实时更新当前选中的股票市值，其他按成本价估算（实际生产环境应批量获取价格）
                    if c_code == st.session_state.code and curr_price > 0:
                        total_mkt_val += curr_price * c_data['qty']
                    else:
                        total_mkt_val += c_data['cost'] * c_data['qty'] # 暂时用成本代替非当前股价格
                
                total_assets = cash + total_mkt_val
                total_profit = total_assets - 1000000.0
                p_color = "red" if total_profit >= 0 else "green"

                # 2. 账户概览面板
                st.markdown(f"""
                <div style="background:#f0f2f6; padding:10px; border-radius:8px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px; color:#666;">
                        <span>总资产(估)</span>
                        <span>可用资金</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:flex-end;">
                        <span style="font-size:18px; font-weight:bold; color:#333;">{total_assets:,.0f}</span>
                        <span style="font-size:14px; color:#333;">{cash:,.0f}</span>
                    </div>
                    <div style="font-size:12px; margin-top:5px;">
                        总盈亏: <span style="color:{p_color}; font-weight:bold;">{total_profit:+,.0f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 3. 功能标签页
                tab_pos, tab_trade, tab_his = st.tabs(["我的持仓", "下单交易", "交易记录"])
                
                # --- Tab 1: 持仓列表 ---
                with tab_pos:
                    if not holdings:
                        st.caption("暂无持仓，快去下单吧！")
                    else:
                        # 转换为DataFrame展示
                        pos_data = []
                        for h_code, h_val in holdings.items():
                            p_now = curr_price if h_code == st.session_state.code else h_val['cost'] # 简化
                            mkt = p_now * h_val['qty']
                            pnl = (p_now - h_val['cost']) * h_val['qty']
                            pnl_pct = (p_now - h_val['cost']) / h_val['cost'] * 100
                            pos_data.append({
                                "代码": h_code,
                                "名称": h_val.get('name', h_code),
                                "成本": f"{h_val['cost']:.2f}",
                                "现价": f"{p_now:.2f}",
                                "持仓": h_val['qty'],
                                "盈亏": f"{pnl:+.0f}",
                                "比例": f"{pnl_pct:+.2f}%"
                            })
                        st.dataframe(pd.DataFrame(pos_data), hide_index=True, use_container_width=True)

                # --- Tab 2: 交易面板 ---
                with tab_trade:
                    tr_action = st.radio("操作方向", ["买入", "卖出"], horizontal=True, label_visibility="collapsed")
                    
                    if curr_price <= 0:
                        st.error("当前无法获取实时价格，无法交易")
                    else:
                        # 交易表单
                        tr_price = st.number_input("委托价格", value=curr_price, step=0.01, format="%.2f")
                        
                        if tr_action == "买入":
                            max_buy = int(cash // (tr_price * 100)) # 手数
                            st.caption(f"最大可买: {max_buy} 手")
                            buy_vol = st.number_input("买入数量 (手)", min_value=1, max_value=max(1, max_buy), value=1, step=1)
                            
                            if st.button("🔴 立即买入", type="primary", use_container_width=True):
                                cost_amt = buy_vol * 100 * tr_price
                                if cost_amt > cash:
                                    st.error("资金不足！")
                                else:
                                    # 更新资金
                                    st.session_state.paper_account['cash'] -= cost_amt
                                    # 更新持仓
                                    if st.session_state.code in holdings:
                                        old = holdings[st.session_state.code]
                                        old_cost = old['cost']
                                        old_qty = old['qty']
                                        new_qty = old_qty + (buy_vol * 100)
                                        new_cost = (old_cost * old_qty + cost_amt) / new_qty
                                        holdings[st.session_state.code]['qty'] = new_qty
                                        holdings[st.session_state.code]['cost'] = new_cost
                                    else:
                                        holdings[st.session_state.code] = {
                                            'name': get_name(st.session_state.code, "", None),
                                            'qty': buy_vol * 100,
                                            'cost': tr_price
                                        }
                                    # 记日志
                                    st.session_state.paper_account['history'].append({
                                        "time": datetime.now().strftime("%m-%d %H:%M"),
                                        "code": st.session_state.code,
                                        "action": "买入",
                                        "price": tr_price,
                                        "qty": buy_vol * 100,
                                        "amt": -cost_amt
                                    })
                                    save_user_holdings(user)
                                    st.success("买入成交！")
                                    time.sleep(1)
                                    st.rerun()
                                    
                        else: # 卖出
                            curr_hold = holdings.get(st.session_state.code, None)
                            if not curr_hold:
                                st.warning("当前无持仓")
                            else:
                                max_sell = int(curr_hold['qty'] / 100)
                                st.caption(f"最大可卖: {max_sell} 手")
                                sell_vol = st.number_input("卖出数量 (手)", min_value=1, max_value=max(1, max_sell), value=1, step=1)
                                
                                if st.button("🟢 立即卖出", type="primary", use_container_width=True):
                                    get_amt = sell_vol * 100 * tr_price
                                    # 更新资金
                                    st.session_state.paper_account['cash'] += get_amt
                                    # 更新持仓
                                    left_qty = curr_hold['qty'] - (sell_vol * 100)
                                    if left_qty <= 0:
                                        del holdings[st.session_state.code]
                                    else:
                                        holdings[st.session_state.code]['qty'] = left_qty
                                    
                                    # 记日志
                                    st.session_state.paper_account['history'].append({
                                        "time": datetime.now().strftime("%m-%d %H:%M"),
                                        "code": st.session_state.code,
                                        "action": "卖出",
                                        "price": tr_price,
                                        "qty": sell_vol * 100,
                                        "amt": get_amt
                                    })
                                    save_user_holdings(user)
                                    st.success("卖出成交！")
                                    time.sleep(1)
                                    st.rerun()

                # --- Tab 3: 历史记录 ---
                with tab_his:
                    his = paper.get("history", [])
                    if not his:
                        st.caption("暂无交易记录")
                    else:
                        # 倒序展示
                        st.dataframe(pd.DataFrame(his[::-1]), hide_index=True, use_container_width=True)

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

        if is_admin:
            st.success("👑 管理员模式")
            with st.expander("👑 VIP 权限管理", expanded=True):
                df_u = load_users()
                u_list = [x for x in df_u["username"] if x!=ADMIN_USER]
                if u_list:
                    vip_target = st.selectbox("选择用户", u_list, key="vip_sel")
                    vip_days = st.number_input("增加天数", value=30, step=1)
                    if st.button("更新 VIP 权限"):
                        if update_vip_days(vip_target, vip_days):
                            st.success(f"已更新 {vip_target} 的 VIP 权限！")
                            time.sleep(1); st.rerun()
                        else: st.error("更新失败")
          
            with st.expander("💳 卡密库存管理 (Stock)", expanded=True):
                points_gen = st.selectbox("面值", [20, 50, 100, 200, 500])
                count_gen = st.number_input("数量", 1, 50, 10)
                if st.button("批量生成库存"):
                    num = batch_generate_keys(points_gen, count_gen)
                    st.success(f"已入库 {num} 张卡密 (面值{points_gen})")
                
                # Show stock stats
                try:
                    df_k = load_keys()
                    st.write("当前库存统计:")
                    st.dataframe(df_k[df_k['status']=='unused'].groupby('points').size().reset_index(name='count'), hide_index=True)
                except: pass
                    
            with st.expander("用户管理"):
                uploaded_file = st.file_uploader("📂 导入用户数据 (CSV)", type=['csv'])
                if uploaded_file is not None:
                    try:
                        new_data = pd.read_csv(uploaded_file)
                        current_data = load_users()
                        combined = pd.concat([current_data, new_data]).drop_duplicates(subset=['username'], keep='last')
                        save_users(combined)
                        st.success(f"成功导入！当前总用户数: {len(combined)}")
                    except Exception as e:
                        st.error(f"导入失败: {e}")

                df_u = load_users()
                st.dataframe(df_u[["username","quota", "vip_expiry", "paper_json"]], hide_index=True)
                csv = df_u.to_csv(index=False).encode('utf-8')
                st.download_button("备份数据 (含模拟持仓)", csv, "backup.csv", "text/csv")
                
                u_list = [x for x in df_u["username"] if x!=ADMIN_USER]
                if u_list:
                    target = st.selectbox("选择用户", u_list)
                    val = st.number_input("新积分", value=0, step=10)
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("更新积分"): update_user_quota(target, val); st.success("OK"); time.sleep(0.5); st.rerun()
                    with c2:
                        chk = st.checkbox("确认删除")
                        if st.button("删除") and chk: delete_user(target); st.success("Del"); time.sleep(0.5); st.rerun()

        timeframe = st.selectbox("周期", ["日线", "周线", "月线"])
        # ✅ 修改：增加7和10天选项，方便短线
        days = st.radio("范围", [7, 10, 30, 60, 120, 250], 2, horizontal=True)
        adjust = st.selectbox("复权", ["qfq","hfq",""], 0)
        
        st.divider()
        
        if is_pro:
            with st.expander("🎛️ 策略参数 (Pro)", expanded=True):
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

# 登录逻辑 (包含新增的注册逻辑)
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
            # 注册方式选择：微信在前，普通在后
            reg_type = st.radio("选择注册方式", 
                              ["微信公众号验证注册 (推荐)", "普通用户注册"], 
                              horizontal=False)
            
            nu = st.text_input("新用户名")
            np1 = st.text_input("设置密码", type="password")
            
            if "微信" in reg_type:
                st.markdown("""
                **1. 关注公众号 `lubingxingpiaoliuji`**<br>
                **2. 发送“注册”获取验证码**<br>
                <span style='color:#d32f2f; font-weight:bold'>🎁 成功注册即送 20 积分！</span>
                """, unsafe_allow_html=True)
                
                if os.path.exists("qrcode.png"):
                    st.image("qrcode.png", width=150)
                
                v_code = st.text_input("请输入验证码")
                if st.button("验证并注册"):
                    if v_code == WECHAT_VALID_CODE:
                        suc, msg = register_user(nu.strip(), np1, initial_quota=20)
                        if suc: st.success(msg)
                        else: st.error(msg)
                    else:
                        st.error("验证码错误，请检查公众号回复。")
            else:
                st.caption("⚠️ 普通注册不赠送积分。")
                if st.button("立即注册 (普通)"):
                    suc, msg = register_user(nu.strip(), np1, initial_quota=0)
                    if suc: st.success(msg)
                    else: st.error(msg)

    st.stop()

# --- 主内容区 ---
name = get_name(st.session_state.code, "", None) 
c1, c2 = st.columns([3, 1])
with c1: st.title(f"📈 {name} ({st.session_state.code})")

# 数据加载
is_demo = False
loading_tips = ["正在加载因子库…", "正在构建回测引擎…", "正在初始化模型框架…", "正在同步行情数据…"]
with st.spinner(random.choice(loading_tips)):
    df = get_data_and_resample(st.session_state.code, "", timeframe, adjust, proxy=None)
    if df.empty:
        st.warning("⚠️ 暂无数据 (可能因网络原因)。自动切换至演示模式。")
        df = generate_mock_data(days)
        is_demo = True

try:
    # 基础指标计算
    funda = get_fundamentals(st.session_state.code, "")
    df = calc_full_indicators(df, ma_s, ma_l)
    df = detect_patterns(df)
    
    # === 区域 1：基础行情 (免费) ===
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
    
    # AI 助理部分 (免费可见，作为引流)
    ai_text, ai_mood = generate_ai_copilot_text(df, name)
    ai_icon = "🤖" if ai_mood == "neutral" else "😊" if ai_mood == "happy" else "😰"
    
    st.markdown(f"""
    <div class="ai-chat-box">
        <div class="ai-avatar">{ai_icon}</div>
        <div class="ai-content">
            <span style="font-weight:bold; color:#2962ff;">AI 投顾助理：</span>
            {ai_text}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # === 区域 2：深度内容 (VIP/付费解锁) ===
    # 权限判断
    has_access = False
    if is_admin: has_access = True
    elif is_vip: has_access = True
    elif st.session_state.paid_code == st.session_state.code: has_access = True
    
    if not has_access:
        st.markdown('<div class="locked-container"><div class="locked-blur">', unsafe_allow_html=True)

    # 1. 绘图 (图表前置)
    plot_chart(df.tail(days), name, flags, ma_s, ma_l)
    
    # 2. 深度研报
    st.markdown(generate_deep_report(df, name), unsafe_allow_html=True)
    
    st.divider()

    # ✅ 新增：交易计划卡片 (纯Table实现)
    if is_pro:
        plan_html = generate_strategy_card(df, name)
        st.markdown(plan_html, unsafe_allow_html=True)
    else:
        st.info("🔒 开启 [专业模式] 可查看具体的买卖点位、止盈止损价格及仓位建议。")

    # 回测看板
    st.markdown("""<div class="bt-header">⚖️ 策略回测报告 (Strategy Backtest)</div>""", unsafe_allow_html=True)
    ret, win, mdd, buy_sigs, sell_sigs, eq = run_backtest(df)
    try:
        daily_returns = eq['equity'].pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    except: sharpe = 0

    st.markdown(f"""
    <div class="bt-container">
        <div class="bt-grid">
            <div class="bt-card">
                <div class="bt-val bt-pos">+{ret:.1f}%</div>
                <div class="bt-lbl">策略总回报</div>
                <div class="bt-tag tag-alpha">Alpha</div>
            </div>
            <div class="bt-card">
                <div class="bt-val bt-pos">{win:.1f}%</div>
                <div class="bt-lbl">实盘胜率</div>
            </div>
            <div class="bt-card">
                <div class="bt-val bt-neg">-{mdd:.1f}%</div>
                <div class="bt-lbl">最大回撤 (Risk)</div>
            </div>
            <div class="bt-card">
                <div class="bt-val bt-neu">{sharpe:.2f}</div>
                <div class="bt-lbl">夏普比率 (Sharpe)</div>
            </div>
        </div>
        <div style="font-size:12px; color:#888; text-align:right;">* 回测区间包含 {len(eq)} 个交易日，对比基准为“买入持有”策略</div>
    </div>
    """, unsafe_allow_html=True)

    # 交互式图表 (Plotly)
    if not eq.empty:
        bt_fig = make_subplots(rows=1, cols=1)
        bt_fig.add_trace(go.Scatter(x=eq['date'], y=eq['equity'], name='策略净值 (Strategy)', 
                                    line=dict(color='#2962ff', width=2), fill='tozeroy', fillcolor='rgba(41, 98, 255, 0.1)'))
        bt_fig.add_trace(go.Scatter(x=eq['date'], y=eq['benchmark'], name='基准 (Buy&Hold)', 
                                    line=dict(color='#9e9e9e', width=1.5, dash='dash')))
        if len(buy_sigs) > 0:
            buy_vals = eq[eq['date'].isin(buy_sigs)]['equity']
            bt_fig.add_trace(go.Scatter(x=buy_vals.index.map(lambda x: eq.loc[x, 'date']), y=buy_vals, mode='markers', 
                                        marker=dict(symbol='triangle-up', size=10, color='#d32f2f'), name='买入信号'))
        bt_fig.update_layout(height=350, margin=dict(l=10,r=10,t=40,b=10), legend=dict(orientation="h", y=1.1), yaxis_title="账户净值", hovermode="x unified")
        st.plotly_chart(bt_fig, use_container_width=True)

    if not has_access:
        st.markdown('</div>', unsafe_allow_html=True) # close blur
        try: bal = load_users()[load_users()["username"]==user]["quota"].iloc[0]
        except: bal = 0
        st.markdown(f"""
        <div class="locked-overlay">
            <div class="lock-icon">🔒</div>
            <div class="lock-title">深度策略已锁定</div>
            <div class="lock-desc">包含：AI解读、买卖点位、缠论结构、机构研报</div>
        </div>
        """, unsafe_allow_html=True)
        c_lock1, c_lock2, c_lock3 = st.columns([1,2,1])
        with c_lock2:
            if st.button(f"🔓 支付 1 积分解锁 (余额: {bal})", key="main_unlock", type="primary", use_container_width=True):
                if consume_quota(user):
                    st.session_state.paid_code = st.session_state.code
                    st.rerun()
                else: st.error("积分不足！")
        
except Exception as e:
    st.error(f"Error: {e}")
    st.error(traceback.format_exc())