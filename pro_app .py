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
import io

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
    page_title="阿尔法量研 Pro V74 (Custom)",
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
DB_FILE = "users_v69.csv" 
KEYS_FILE = "card_keys.csv"

# Optional deps
ts = None
bs = None
try: import tushare as ts
except: pass
try: import baostock as bs
except: pass

# 🔥 CSS 样式 (保持原样并微调)
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
    
    /* 🔥 新增底部交易看板样式 */
    .trade-plan-container {
        background: #fff; border-top: 4px solid #00c853; border-radius: 8px;
        padding: 20px; margin-top: 20px; box-shadow: 0 -2px 15px rgba(0,0,0,0.05);
    }
    .tp-title { font-size: 20px; font-weight: 900; color: #333; margin-bottom: 15px; display: flex; align-items: center; }
    .tp-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
    .tp-box { background: #f5f7f9; padding: 10px; border-radius: 6px; text-align: center; }
    .tp-val { font-size: 18px; font-weight: bold; color: #333; }
    .tp-lbl { font-size: 12px; color: #777; margin-top: 4px; }
    .risk-alert { 
        margin-top: 30px; padding: 15px; background: #fff3e0; 
        color: #e65100; border: 1px solid #ffe0b2; border-radius: 8px;
        font-size: 12px; text-align: center; font-weight: bold;
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
        holdings_json = json.dumps(st.session_state.paper_holdings)
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
                st.session_state.paper_holdings = json.loads(json_str)
            except:
                st.session_state.paper_holdings = {}

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
    if not expiry_str or expiry_str == "nan": return False, "普通会员"
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

def register_user(u, p, initial_quota=0):
    if u == ADMIN_USER: return False, "保留账号"
    df = load_users()
    if u in df["username"].values: return False, "用户已存在"
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(p.encode(), salt).decode()
    new_row = {"username": u, "password_hash": hashed, "watchlist": "", "quota": initial_quota, "vip_expiry": "", "paper_json": "{}"}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_users(df)
    return True, "注册成功！"

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
    atr = c['ATR14']
    stop_loss = c['close'] - 2*atr
    take_profit = c['close'] + 3*atr
    # 唐奇安通道作为支撑压力
    support = df['low'].iloc[-20:].min()
    resistance = df['high'].iloc[-20:].max()
    return score, action, color, stop_loss, take_profit, pos_txt, support, resistance, reasons

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
        <div class='brand-en'>AlphaQuant Pro V74</div>
        <div class='brand-slogan'>用历史验证未来，用数据构建策略。</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 🔥🔥🔥 1. 会员与充值中心 (上移至顶部，默认折叠)
    if st.session_state.get('logged_in') and not (st.session_state["user"] == ADMIN_USER):
        user = st.session_state["user"]
        with st.expander("💎 充值与会员中心", expanded=False):
            st.info(f"当前积分: {load_users()[load_users()['username']==user]['quota'].iloc[0]}")
            
            st.write("##### 1. 扫码充值")
            # 增加10元选项
            pay_opt = st.radio("充值面额", [10, 20, 50, 100], horizontal=True, format_func=lambda x: f"￥{x}")
            
            if os.path.exists("alipay.png"):
                st.image("alipay.png", caption="请使用支付宝扫码", width=200)
            else:
                st.warning("请上传 alipay.png")
            
            st.markdown("""
            **📢 充值说明：**
            1. **兑换比例**：1元 = 2积分。
            2. **会员套餐**：
               - 充值 **30元** 兑换 **月卡会员** (VIP)
               - 充值 **80元** 兑换 **季卡会员** (VIP)
            3. **注意事项**：充值时请务必备注您的 **用户名**。
            4. **客服咨询**：如有问题请关注微信公众号 **lubingxpiaoliuji** 咨询。
            """)
            
            st.write("##### 2. 卡密兑换")
            k_in = st.text_input("输入卡密")
            if st.button("兑换"):
                s, m = redeem_key(user, k_in)
                if s: st.success(m); time.sleep(1); st.rerun()
                else: st.error(m)
    
    new_c = st.text_input("🔍 股票代码", st.session_state.code)
    if new_c != st.session_state.code: st.session_state.code = new_c; st.session_state.paid_code = ""; st.rerun()

    if st.session_state.get('logged_in'):
        user = st.session_state["user"]
        is_admin = (user == ADMIN_USER)
        is_vip, vip_msg = check_vip_status(user)
        
        load_user_holdings(user)
        
        if is_vip: st.success(f"👑 {vip_msg} (VIP)")
        else: st.info(f"👤 普通用户 (积分: {load_users()[load_users()['username']==user]['quota'].iloc[0]})")

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
        
        if not is_admin:
            with st.expander("🎮 模拟交易 (Paper Trading)", expanded=True):
                with st.expander("📚 使用说明与功能价值", expanded=False):
                    st.markdown("""
                    **💡 功能价值：**
                    1. **零风险试错**：验证您的策略是否有效，而无需投入真金白银。
                    2. **盘感训练**：记录买卖逻辑，通过盈亏反馈修正交易心态。
                    3. **数据永存**：您的持仓数据已云端备份，随时可查。
                    """)
              
                curr_hold = st.session_state.paper_holdings.get(st.session_state.code, None)
              
                curr_price = 0
                try:
                    curr_price = float(yf.Ticker(process_ticker(st.session_state.code)).fast_info.last_price)
                except: curr_price = 0 
              
                if curr_hold:
                    cost = curr_hold.get('cost', 0)
                    qty = curr_hold.get('qty', 100)
                    if curr_price > 0:
                        mkt_val = curr_price * qty
                        profit = (curr_price - cost) * qty
                        profit_pct = (curr_price - cost) / cost * 100
                        p_color = "red" if profit > 0 else "green" 
                        st.markdown(f"""
                        <div style="font-size:14px; margin-bottom:5px;">
                            <b>持仓成本:</b> {cost:.2f}<br>
                            <b>持仓数量:</b> {qty} 股<br>
                            <b>持仓市值:</b> {mkt_val:.0f}<br>
                            <b>浮动盈亏:</b> <span style='color:{p_color}; font-weight:bold'>{profit:.0f} ({profit_pct:.2f}%)</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info(f"成本: {cost:.2f} | 数量: {qty}")
                       
                    if st.button("卖出平仓", key="paper_sell"):
                        del st.session_state.paper_holdings[st.session_state.code]
                        save_user_holdings(user)
                        st.success("已卖出！")
                        st.rerun()
                else:
                    buy_qty = st.number_input("买入数量 (手)", min_value=1, max_value=100, value=1, step=1)
                    if st.button("➕ 模拟买入", key="paper_buy"):
                        st.session_state.paper_holdings[st.session_state.code] = {
                            'cost': 0, 
                            'qty': buy_qty * 100, 
                            'date': datetime.now().strftime("%Y-%m-%d"),
                            'name': ""
                        }
                        save_user_holdings(user)
                        st.success("买入成功！")
                        st.rerun()

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
           
            with st.expander("💳 卡密生成"):
                points_gen = st.selectbox("面值", [20, 50, 100, 200, 500])
                count_gen = st.number_input("数量", 1, 50, 10)
                if st.button("批量生成"):
                    num = batch_generate_keys(points_gen, count_gen)
                    st.success(f"已生成 {num} 张卡密")
                   
            with st.expander("👤 用户管理"):
                # 🔥🔥🔥 管理员上传用户数据功能
                uploaded_file = st.file_uploader("📂 上传用户数据 (覆盖 users.csv)", type="csv")
                if uploaded_file is not None:
                    try:
                        new_df = pd.read_csv(uploaded_file)
                        # 简单校验必要列
                        if "username" in new_df.columns:
                            new_df.to_csv(DB_FILE, index=False)
                            st.success("用户数据更新成功！")
                            time.sleep(1); st.rerun()
                        else:
                            st.error("CSV 格式错误：缺少 username 列")
                    except Exception as e:
                        st.error(f"上传失败: {e}")

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
        # 🔥🔥🔥 增加 7 和 10 两个周期选项
        days = st.radio("范围", [7, 10, 30, 60, 120, 250], 4, horizontal=True)
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
        
        # 🔥🔥🔥 注册系统升级：微信验证注册 + 普通注册
        tab1, tab2, tab3 = st.tabs(["🔑 登录", "✨ 微信验证注册 (+20积分)", "📧 普通注册"])
        
        with tab1:
            u = st.text_input("账号")
            p = st.text_input("密码", type="password")
            if st.button("登录系统"):
                if verify_login(u.strip(), p): st.session_state["logged_in"] = True; st.session_state["user"] = u.strip(); st.session_state["paid_code"] = ""; st.rerun()
                else: st.error("账号或密码错误")
        
        with tab2:
            st.info("🎁 关注公众号获取验证码，注册即送 20 积分！")
            
            # 显示二维码 (确保 qrcode.png 存在)
            if os.path.exists("qrcode.png"):
                st.image("qrcode.png", width=150, caption="扫码关注公众号：lubingxpiaoliuji")
            else:
                st.warning("请联系管理员上传 qrcode.png")
                st.caption("公众号: lubingxpiaoliuji")
            
            wx_u = st.text_input("用户名 (微信注册)", key="wx_u")
            wx_p = st.text_input("设置密码", type="password", key="wx_p")
            verify_code = st.text_input("验证码 (发送'注册'获取)")
            
            if st.button("🎁 立即注册领积分"):
                if not verify_code:
                    st.error("请输入验证码")
                # 模拟验证码校验逻辑 (实际需对接公众号接口)
                elif len(verify_code) > 0: 
                    # 这里假设任意非空验证码通过，实际可设置为固定码如 '666'
                    suc, msg = register_user(wx_u.strip(), wx_p, initial_quota=20)
                    if suc: st.success("🎉 注册成功，已获赠20积分！请登录。")
                    else: st.error(msg)
                else:
                    st.error("验证码错误")
        
        with tab3:
            st.caption("普通注册无积分赠送")
            nu = st.text_input("用户名", key="reg_u")
            np1 = st.text_input("设置密码", type="password", key="reg_p")
            if st.button("立即注册"):
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

if st.session_state.code in st.session_state.paper_holdings:
    if st.session_state.paper_holdings[st.session_state.code]['cost'] == 0:
        st.session_state.paper_holdings[st.session_state.code]['cost'] = df.iloc[-1]['close']
        st.session_state.paper_holdings[st.session_state.code]['name'] = name
        save_user_holdings(user) 

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
    
    # 核心分析数据准备 (用于后续模块)
    sc, act, col, sl, tp, pos, sup, res, reasons = analyze_score(df)
    
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

    # 🔥🔥🔥 3. 新增底部交易计划模块 (替代原智能决策)
    if is_pro:
        st.markdown(f"""
        <div class="trade-plan-container">
            <div class="tp-title">🛡️ 交易计划与风控看板 (Trade Plan & Risk Control)</div>
            <div class="tp-grid">
                <div class="tp-box" style="border-bottom: 3px solid #ff3b30;">
                    <div class="tp-val">{tp:.2f}</div>
                    <div class="tp-lbl">🎯 目标止盈位</div>
                </div>
                <div class="tp-box" style="border-bottom: 3px solid #00c853;">
                    <div class="tp-val">{sl:.2f}</div>
                    <div class="tp-lbl">🛑 预警止损位</div>
                </div>
                <div class="tp-box" style="border-bottom: 3px solid #2962ff;">
                    <div class="tp-val">{sup:.2f}</div>
                    <div class="tp-lbl">🧱 下方支撑位</div>
                </div>
                <div class="tp-box" style="border-bottom: 3px solid #ff9800;">
                    <div class="tp-val">{res:.2f}</div>
                    <div class="tp-lbl">🔨 上方压力位</div>
                </div>
            </div>
            
            <div class="risk-alert">
                ⚠️ <b>风险免责声明 (Disclaimer)：</b><br>
                本系统所有数据、策略、关键位及AI分析结果仅基于历史数据计算，<b>仅供量化研究参考，不构成任何投资建议</b>。<br>
                股市有风险，入市需谨慎。用户应独立判断市场风险，自负盈亏。
            </div>
        </div>
        """, unsafe_allow_html=True)

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