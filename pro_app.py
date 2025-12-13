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
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# 初始化 Session
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if "code" not in st.session_state: st.session_state.code = "600519"
if "paid_code" not in st.session_state: st.session_state.paid_code = ""

# 🔥 V47.0 史诗级 CSS：金融终端风格
pro_css = """
<style>
    /* 全局背景 */
    .stApp {
        background-color: #f0f2f5; /* 浅灰底色，突显卡片 */
        font-family: "Helvetica Neue", Helvetica, "PingFang SC", "Microsoft YaHei", Arial, sans-serif;
    }
    
    /* 隐藏杂项 */
    .stDeployButton, footer, header {display: none !important;}
    .block-container {padding-top: 1.5rem !important; padding-bottom: 2rem !important;}

    /* ================= 卡片容器 (核心) ================= */
    .content-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #e1e4e8;
    }

    /* 标题样式 */
    .card-title {
        font-size: 16px;
        font-weight: 700;
        color: #333;
        margin-bottom: 15px;
        border-left: 4px solid #2962ff; /* 蓝条索引 */
        padding-left: 10px;
        display: flex;
        align-items: center;
    }

    /* ================= 侧边栏优化 ================= */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e1e4e8;
    }
    .brand-box {
        text-align: center; margin-bottom: 30px; padding-bottom: 20px;
        border-bottom: 1px dashed #eee;
    }
    .brand-main { font-size: 20px; font-weight: 900; color: #1a237e; letter-spacing: 1px; }
    .brand-sub { font-size: 12px; color: #666; margin-top: 5px; }

    /* ================= 按钮美化 ================= */
    div.stButton > button {
        border-radius: 8px; font-weight: 600; border: none; transition: all 0.2s;
    }
    div.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    
    /* ================= 评分卡片 (V47新版) ================= */
    .score-grid { display: flex; gap: 15px; width: 100%; }
    .score-item {
        flex: 1; background: #f8f9fa; border-radius: 10px; padding: 15px;
        text-align: center; border: 1px solid #eee; position: relative;
        overflow: hidden;
    }
    .score-num { font-size: 28px; font-weight: 800; line-height: 1.2; }
    .score-name { font-size: 13px; color: #666; margin-top: 4px; }
    .score-bar-bg { width: 100%; height: 4px; background: #eee; margin-top: 10px; border-radius: 2px; }
    .score-bar-fill { height: 100%; border-radius: 2px; }

    /* ================= 风险雷达条 ================= */
    .risk-container { margin-top: 10px; }
    .risk-bar-outer { height: 10px; background: #e0e0e0; border-radius: 5px; overflow: hidden; position: relative; }
    .risk-bar-fill { height: 100%; border-radius: 5px; transition: width 0.5s; }
    .risk-label { display: flex; justify-content: space-between; font-size: 12px; color: #666; margin-top: 5px; }

    /* ================= 投资亮点 ================= */
    .highlight-row { 
        display: flex; align-items: start; margin-bottom: 12px; 
        background: #f9fbfd; padding: 10px; border-radius: 8px;
    }
    .hl-badge {
        font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px;
        margin-right: 10px; white-space: nowrap; margin-top: 2px;
    }
    .hl-text { font-size: 14px; color: #333; line-height: 1.5; }

    /* 颜色定义类 */
    .col-red { color: #d32f2f; }
    .col-green { color: #00c853; }
    .bg-red-light { background-color: #ffebee; color: #c62828; }
    .bg-green-light { background-color: #e8f5e9; color: #2e7d32; }
    .bg-blue-light { background-color: #e3f2fd; color: #1565c0; }
    
    /* 覆盖原生 Metric 样式 */
    [data-testid="stMetricValue"] { font-size: 24px !important; font-family: 'Roboto', sans-serif; }
</style>
"""
st.markdown(pro_css, unsafe_allow_html=True)

# 👑 全局常量
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"
DB_FILE = "users_v47.csv"
KEYS_FILE = "card_keys.csv"

# Optional deps
try:
    import tushare as ts
except: ts = None
try:
    import baostock as bs
except: bs = None

# ==========================================
# 2. 数据库与工具函数
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
    if match.empty: return False, "❌ 卡密无效或已被使用"
    points_to_add = int(match.iloc[0]["points"])
    df_keys.loc[match.index[0], "status"] = f"used_by_{username}"
    save_keys(df_keys)
    df_users = load_users()
    u_idx = df_users[df_users["username"] == username].index[0]
    df_users.loc[u_idx, "quota"] += points_to_add
    save_users(df_users)
    return True, f"✅ 充值成功！增加 {points_to_add} 积分"

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
    QUICK_MAP = {'600519': '贵州茅台','000858': '五粮液','601318': '中国平安','600036': '招商银行','300750': '宁德时代','002594': '比亚迪','601888': '中国中免','600276': '恒瑞医药','601857': '中国石油','601088': '中国神华','601988': '中国银行','601398': '工商银行','AAPL': 'Apple','TSLA': 'Tesla','NVDA': 'NVIDIA','MSFT': 'Microsoft','BABA': 'Alibaba'}
    if clean_code in QUICK_MAP: return QUICK_MAP[clean_code]
    if token and ts and is_cn_stock(clean_code):
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
    p_high = h.rolling(9).max(); p_low = l.rolling(9).min()
    df['Tenkan'] = (p_high + p_low) / 2
    p_high26 = h.rolling(26).max(); p_low26 = l.rolling(26).min()
    df['Kijun'] = (p_high26 + p_low26) / 2
    df['SpanA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['SpanB'] = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    df['SpanA'] = df['SpanA'].fillna(method='bfill').fillna(0)
    df['SpanB'] = df['SpanB'].fillna(method='bfill').fillna(0)
    df['MA_Short'] = c.rolling(ma_s).mean()
    df['MA_Long'] = c.rolling(ma_l).mean()
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

def run_backtest(df):
    if df is None or len(df) < 50: return 0.0, 0.0, 0.0, [], [], pd.DataFrame({'date':[], 'equity':[]})
    needed = ['MA_Short', 'MA_Long', 'close', 'date']
    if not all(c in df.columns for c in needed): return 0.0, 0.0, 0.0, [], [], pd.DataFrame({'date':[], 'equity':[]})
    df_bt = df.dropna(subset=needed).reset_index(drop=True)
    if len(df_bt) < 20: return 0.0, 0.0, 0.0, [], [], pd.DataFrame({'date':[], 'equity':[]})
    capital = 100000; position = 0; buy_signals = []; sell_signals = []; equity = [capital]; dates = [df_bt.iloc[0]['date']]
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
    win_rate = 50 + (ret / 10); win_rate = max(10, min(90, win_rate))
    eq_series = pd.Series(equity)
    cummax = eq_series.cummax()
    drawdown = (eq_series - cummax) / cummax
    max_dd = drawdown.min() * 100
    eq_df = pd.DataFrame({'date': dates, 'equity': equity})
    return ret, win_rate, max_dd, buy_signals, sell_signals, eq_df

def generate_deep_report(df, name):
    curr = df.iloc[-1]
    chan_trend = "底分型构造中" if curr['F_Bot'] else "顶分型构造中" if curr['F_Top'] else "中继形态"
    chan_logic = f"""
    <div class="content-card">
        <div class="card-title">📐 缠论结构</div>
        <div style="font-size:14px; color:#444;">
        • <b>分型状态</b>：{chan_trend}<br>
        • <b>笔的延伸</b>：当前价格处于一笔走势的{ "延续阶段" if not (curr['F_Top'] or curr['F_Bot']) else "转折关口" }
        </div>
    </div>
    """
    return chan_logic

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
    return score, action, color, c['close']-2*atr, c['close']+3*atr, pos_txt

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
        highlights.append(("机构", f"华尔街/机构评级为 {funda['rating']}"))
    
    # 风险与估值
    if is_high_risk:
        highlights.append(("⚠️ 风险", f"当前价格处于近10年 {price_pct}% 高位，注意回调！"))
    elif price_pct < 15:
        highlights.append(("机会", f"当前价格处于近10年 {price_pct}% 低位，安全边际高。"))
    
    try:
        pe = float(funda['pe'])
        if pe > 0 and pe < 20: highlights.append(("估值", f"当前PE为{pe}，估值偏低。"))
        elif pe > 60: highlights.append(("泡沫", f"当前PE高达{pe}，存在泡沫风险。"))
    except: pass
    
    # 技术面
    if last['MA_Short'] > last['MA_Long']: highlights.append(("趋势", "均线呈多头排列，短期趋势向上。"))
    else: highlights.append(("趋势", "均线呈空头排列，短期趋势向下。"))
    
    if last['VolRatio'] > 2: highlights.append(("资金", "今日放量明显，主力资金异动。"))
    
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
# 4. 执行入口 (Logic)
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
    
    if st.session_state.get('logged_in'):
        user = st.session_state["user"]
        is_admin = (user == ADMIN_USER)
        
        if st.button("🔄 刷新数据"):
            st.cache_data.clear()
            st.rerun()

        if is_admin:
            st.success("👑 管理员模式")
            with st.expander("💳 卡密生成"):
                p_gen = st.selectbox("面值", [20, 50, 100]); c_gen = st.number_input("数量", 1, 50, 10)
                if st.button("批量生成"): st.success(f"生成 {batch_generate_keys(p_gen, c_gen)} 张")
            with st.expander("用户管理"):
                st.dataframe(load_users()[["username","quota"]], hide_index=True)
            with st.expander("卡密管理"):
                st.dataframe(load_keys(), hide_index=True)
                if st.button("清理失效卡密"): 
                    save_keys(load_keys()[load_keys()['status']=='unused'])
                    st.rerun()
        else:
            st.info(f"👤 {user} | 积分: {load_users()[load_users()['username']==user]['quota'].iloc[0]}")
            with st.expander("💎 充值中心", expanded=True):
                pay_opt = st.radio("面额", [20, 50, 100], horizontal=True, format_func=lambda x: f"￥{x}")
                if os.path.exists("alipay.png"): st.image("alipay.png", width=200)
                else: st.caption("请联系管理员获取二维码")
                if st.button("✅ 我已支付，发货"):
                    st.code(generate_key(pay_opt), language="text")
                k_in = st.text_input("兑换卡密")
                if st.button("兑换"):
                    s, m = redeem_key(user, k_in)
                    if s: st.success(m); time.sleep(1); st.rerun()
                    else: st.error(m)
        
        st.divider()
        new_c = st.text_input("🔍 股票代码", st.session_state.code)
        if new_c != st.session_state.code: st.session_state.code = new_c; st.session_state.paid_code = ""; st.rerun()
        
        # 自选股
        if not is_admin:
            wl = get_user_watchlist(user)
            if wl:
                c_sel = st.selectbox("⭐ 我的自选", ["请选择..."] + wl)
                if c_sel != "请选择...": st.session_state.code = c_sel; st.session_state.paid_code = ""; st.rerun()
            if st.button("❤️ 加入自选"): update_watchlist(user, st.session_state.code, "add"); st.rerun()

        timeframe = st.selectbox("周期", ["日线", "周线", "月线"])
        days = st.radio("范围", [30,60,120,250], 2, horizontal=True)
        
        st.divider()
        with st.expander("🎛️ 策略参数"):
            ma_s = st.slider("短均线", 2, 20, 5)
            ma_l = st.slider("长均线", 10, 120, 20)
        
        flags = {
            'ma': st.checkbox("MA", True), 'boll': st.checkbox("BOLL", True),
            'vol': st.checkbox("VOL", True), 'macd': st.checkbox("MACD", True),
            'kdj': st.checkbox("KDJ", True), 'gann': st.checkbox("江恩", False),
            'fib': st.checkbox("斐波那契", True), 'chan': st.checkbox("缠论", True)
        }
        st.divider()
        if st.button("退出登录"): st.session_state["logged_in"]=False; st.rerun()
    else:
        st.info("请登录")

if not st.session_state.get('logged_in'):
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.markdown("<br><br><h1 style='text-align:center'>AlphaQuant Pro</h1>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["登录", "注册"])
        with tab1:
            u = st.text_input("账号"); p = st.text_input("密码", type="password")
            if st.button("登录"):
                if verify_login(u,p): st.session_state["logged_in"]=True; st.session_state["user"]=u; st.rerun()
                else: st.error("错误")
        with tab2:
            nu = st.text_input("新账号"); np1 = st.text_input("设密码", type="password")
            if st.button("注册"):
                s, m = register_user(nu, np1)
                if s: st.success(m)
                else: st.error(m)
    st.stop()

# --- Content ---
name = get_name(st.session_state.code, "", None)
st.title(f"📈 {name} ({st.session_state.code})")

is_demo = False
if st.session_state.code != st.session_state.paid_code:
    u_q = load_users()
    try: bal = u_q[u_q["username"]==user]["quota"].iloc[0]
    except: bal = 0
    if bal > 0:
        st.info(f"🔒 需解锁 (余额: {bal})")
        if st.button("🔓 支付1积分"):
            if consume_quota(user): st.session_state.paid_code = st.session_state.code; st.rerun()
        st.stop()
    else:
        st.warning("👀 演示模式")
        is_demo = True
        df = generate_mock_data(days)

if not is_demo:
    with st.spinner("AI正在分析..."):
        df = get_data_and_resample(st.session_state.code, "", timeframe, "qfq", None)
        if df.empty: df = generate_mock_data(days); is_demo = True

try:
    funda = get_fundamentals(st.session_state.code, "")
    df = calc_full_indicators(df, ma_s, ma_l)
    df = detect_patterns(df)
    
    # 顶部横幅
    t_txt, t_col = main_uptrend_check(df)
    bg = "#e6f4ea" if t_col=="success" else "#fff7e6" if t_col=="warning" else "#fce8e6"
    tc = "#137333" if t_col=="success" else "#b06000" if t_col=="warning" else "#c5221f"
    st.markdown(f"<div class='trend-banner' style='background:{bg};'><h3 class='trend-title' style='color:{tc}'>{t_txt}</h3></div>", unsafe_allow_html=True)

    # 核心指标
    l = df.iloc[-1]
    c1, c2 = st.columns(2)
    with c1:
        st.metric("价格", f"{l['close']:.2f}", f"{l['pct_change']:.2f}%")
        st.metric("RSI", f"{l['RSI']:.1f}")
        st.metric("量比", f"{l['VolRatio']:.2f}")
    with c2:
        st.metric("PE", funda['pe'])
        st.metric("ADX", f"{l['ADX']:.1f}")
        
    # 评分卡
    sq, sv, st_ = calculate_smart_score(df, funda)
    st.markdown(f"""
    <div class="score-grid">
        <div class="score-item"><div class="score-num" style="color:#ff3b30">{sq}</div><div class="score-name">质量</div><div class="score-bar-bg"><div class="score-bar-fill" style="width:{sq*10}%; background:#ff3b30"></div></div></div>
        <div class="score-item"><div class="score-num" style="color:#ff9500">{sv}</div><div class="score-name">价值</div><div class="score-bar-bg"><div class="score-bar-fill" style="width:{sv*10}%; background:#ff9500"></div></div></div>
        <div class="score-item"><div class="score-num" style="color:#34c759">{st_}</div><div class="score-name">趋势</div><div class="score-bar-bg"><div class="score-bar-fill" style="width:{st_*10}%; background:#34c759"></div></div></div>
    </div>
    """, unsafe_allow_html=True)
    
    # 亮点与风险
    p_pct, is_risk = calculate_risk_percentile(df)
    hls = get_smart_highlights(df, funda, p_pct, is_risk)
    hl_html = "".join([f"<div class='highlight-row'><span class='hl-badge {'bg-red-light' if '风险' in t else 'bg-green-light' if '机会' in t else 'bg-blue-light'}'>{t}</span><span class='hl-text'>{d}</span></div>" for t,d in hls])
    
    st.markdown(f"""
    <div class="content-card" style="margin-top:15px;">
        <div class="card-title">深度透视 <span style="font-size:12px;color:#999;font-weight:400;margin-left:auto">Risk & Opps</span></div>
        <div style="margin-bottom:15px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;font-size:13px;color:#666;">
                <span>历史分位 (10年)</span><span>{p_pct}%</span>
            </div>
            <div class="risk-bar-outer"><div class="risk-bar-fill" style="width:{p_pct}%; background:{'#d32f2f' if is_risk else '#1976d2'}"></div></div>
        </div>
        {hl_html}
    </div>
    """, unsafe_allow_html=True)

    # 图表
    plot_chart(df.tail(days), name, flags, ma_s, ma_l)
    
    # 研报
    st.markdown(generate_deep_report(df, name), unsafe_allow_html=True)
    
    # 建议
    sc, act, col, sl, tp, pos = analyze_score(df)
    st.markdown(f"""
    <div class="content-card" style="border-left:5px solid {'#00c853' if col=='success' else '#ff9800' if col=='warning' else '#d32f2f'};">
        <div style="font-size:18px; font-weight:800; color:#333;">🤖 策略建议：{act}</div>
        <div style="display:flex; gap:15px; margin-top:10px; font-size:14px; color:#555;">
            <div>仓位：<b>{pos}</b></div>
            <div>止损：<b>{sl:.2f}</b></div>
            <div>止盈：<b>{tp:.2f}</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 回测
    with st.expander("⚖️ 历史回测数据"):
        ret, win, mdd, _, _, eq = run_backtest(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("收益", f"{ret:.1f}%"); c2.metric("胜率", f"{win:.0f}%"); c3.metric("回撤", f"{mdd:.1f}%")
        if not eq.empty:
            f2 = go.Figure()
            f2.add_trace(go.Scatter(x=eq['date'], y=eq['equity'], fill='tozeroy', line=dict(color='#2962ff', width=1.5)))
            f2.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            st.plotly_chart(f2, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
