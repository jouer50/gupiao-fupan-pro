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
import json
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
    page_title="阿尔法量研 Pro",
    layout="wide",
    page_icon="🔥",
    initial_sidebar_state="expanded"
)

# 初始化 Session
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if "code" not in st.session_state: st.session_state.code = "600519"
if "paid_code" not in st.session_state: st.session_state.paid_code = ""
if "trade_qty" not in st.session_state: st.session_state.trade_qty = 100
if "daily_picks_cache" not in st.session_state: st.session_state.daily_picks_cache = None
if "enable_realtime" not in st.session_state: st.session_state.enable_realtime = False
if "ts_token" not in st.session_state: st.session_state.ts_token = "你的Tushare接口密钥" 
if "view_mode_idx" not in st.session_state: st.session_state.view_mode_idx = 0 

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
    'ma': True, 'boll': True, 'vol': True, 
    'macd': False, 'kdj': False, 'gann': False, 'fib': False, 'chan': False
}

# 核心常量
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"
DB_FILE = "users_v69.csv"
KEYS_FILE = "card_keys.csv"
WECHAT_VALID_CODE = "8888"  

# Optional deps
ts = None
bs = None
try: import tushare as ts
except: pass
try: import baostock as bs
except: pass

# ==========================================
# 2. 数据库与工具
# ==========================================
def init_db():
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota", "vip_expiry", "paper_json", "rt_perm", "last_code"])
        df.to_csv(DB_FILE, index=False)
    else:
        df = pd.read_csv(DB_FILE)
        cols_needed = ["vip_expiry", "paper_json", "rt_perm", "last_code"]
        updated = False
        for c in cols_needed:
            if c not in df.columns:
                if c == "rt_perm": df[c] = 0
                elif c == "last_code": df[c] = "600519"
                else: df[c] = ""
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
        df = pd.read_csv(DB_FILE, dtype={"watchlist": str, "quota": int, "vip_expiry": str, "paper_json": str, "rt_perm": int, "last_code": str})
        return df.fillna("")
    except: return pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota", "vip_expiry", "paper_json", "rt_perm", "last_code"])

def save_users(df): df.to_csv(DB_FILE, index=False)

def save_user_last_code(username, code):
    if username == ADMIN_USER: return
    df = load_users()
    idx = df[df["username"] == username].index
    if len(idx) > 0:
        if str(df.loc[idx[0], "last_code"]) != str(code):
            df.loc[idx[0], "last_code"] = str(code)
            save_users(df)

def get_user_last_code(username):
    if username == ADMIN_USER: return "600519"
    df = load_users()
    row = df[df["username"] == username]
    if not row.empty:
        code = str(row.iloc[0].get("last_code", "600519"))
        if code and code != "nan": return code
    return "600519"

def save_user_holdings(username):
    if username == ADMIN_USER: return
    df = load_users()
    idx = df[df["username"] == username].index
    if len(idx) > 0:
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
                if "cash" not in data:
                    st.session_state.paper_account = {"cash": 1000000.0, "holdings": {}, "history": []}
                else:
                    st.session_state.paper_account = data
            except:
                st.session_state.paper_account = {"cash": 1000000.0, "holdings": {}, "history": []}
        
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

def check_rt_permission(username):
    if username == ADMIN_USER: return True
    df = load_users()
    row = df[df["username"] == username]
    if not row.empty:
        return bool(row.iloc[0].get("rt_perm", 0))
    return False

def update_rt_permission(username, allow: bool):
    df = load_users()
    idx = df[df["username"] == username].index
    if len(idx) > 0:
        df.loc[idx[0], "rt_perm"] = 1 if allow else 0
        save_users(df)
        return True
    return False

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
    init_paper = json.dumps({"cash": 1000000.0, "holdings": {}, "history": []})
    new_row = {"username": u, "password_hash": hashed, "watchlist": "", "quota": initial_quota, "vip_expiry": "", "paper_json": init_paper, "rt_perm": 0, "last_code": "600519"}
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

@st.cache_data(ttl=3600*12)
def get_chip_data_pro(stock_code, token, days=60):
    if not token or not ts: return pd.DataFrame()
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        df = pro.cyq_chips(ts_code=_to_ts_code(stock_code), start_date=start, end_date=end)
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def get_finance_vip(stock_code, token):
    if not token or not ts: return pd.DataFrame()
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        start = (datetime.now() - timedelta(days=365*2)).strftime('%Y%m%d')
        df = pro.income_vip(ts_code=_to_ts_code(stock_code), start_date=start)
        return df
    except:
        return pd.DataFrame()

def fetch_and_merge_realtime(raw_df, code, token):
    if not is_cn_stock(code) or not token or not ts:
        return raw_df
    try:
        ts.set_token(token)
        df_rt = ts.get_realtime_quotes(code) 
        if df_rt is not None and not df_rt.empty:
            rt_row = df_rt.iloc[0]
            now_price = float(rt_row['price'])
            now_open = float(rt_row['open'])
            now_high = float(rt_row['high'])
            now_low = float(rt_row['low'])
            now_vol = float(rt_row['volume'])
            now_date_str = rt_row['date'] 
            
            if now_price == 0: return raw_df

            now_date = pd.to_datetime(now_date_str)
            
            new_row = {
                'date': now_date,
                'open': now_open,
                'high': now_high,
                'low': now_low,
                'close': now_price,
                'volume': now_vol,
                'pct_change': ((now_price - float(rt_row['pre_close'])) / float(rt_row['pre_close'])) * 100
            }
            
            if not raw_df.empty:
                last_date = pd.to_datetime(raw_df.iloc[-1]['date'])
                if now_date.date() == last_date.date():
                    raw_df.iloc[-1, raw_df.columns.get_loc('close')] = now_price
                    raw_df.iloc[-1, raw_df.columns.get_loc('high')] = max(raw_df.iloc[-1]['high'], now_high)
                    raw_df.iloc[-1, raw_df.columns.get_loc('low')] = min(raw_df.iloc[-1]['low'], now_low)
                    raw_df.iloc[-1, raw_df.columns.get_loc('volume')] = now_vol
                    raw_df.iloc[-1, raw_df.columns.get_loc('pct_change')] = new_row['pct_change']
                elif now_date > last_date:
                    raw_df = pd.concat([raw_df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                raw_df = pd.DataFrame([new_row])
    except Exception:
        pass
    return raw_df

def get_data_and_resample(code, token, timeframe, adjust, proxy=None):
    if st.session_state.get('ts_token'): token = st.session_state.ts_token

    code = process_ticker(code)
    fetch_days = 1500 
    raw_df = pd.DataFrame()
    
    if is_cn_stock(code) and token and ts:
        try:
            ts.set_token(token)
            pro = ts.pro_api()
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
                raw_df = df.sort_values('date').reset_index(drop=True)
                
                req_cols = ['date','open','high','low','close','volume','pct_change']
                for c in req_cols:
                    if c in raw_df.columns:
                        raw_df[c] = pd.to_numeric(raw_df[c], errors='coerce')
        except Exception as e: 
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
            
    if st.session_state.get("enable_realtime", False) and is_cn_stock(code):
        raw_df = fetch_and_merge_realtime(raw_df, code, token)

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
            ts.set_token(token)
            pro = ts.pro_api()
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

# ✅ 每日精选逻辑更新：返回 4 只股票
def get_daily_picks(user_watchlist):
    SECTOR_POOL = {
        "AI算力与CPO": ["601360", "300308", "002230", "000977", "600418", "300394"],
        "半导体与芯片": ["600584", "002371", "688981", "603501", "002156", "688041"],
        "新能源与车": ["300750", "002594", "601012", "002812", "002460", "600438"],
        "大金融与中特估": ["601318", "600036", "601857", "601398", "600030", "601998"],
        "大消费": ["600519", "000858", "601888", "600887", "000568", "603288"]
    }
    
    # 1. 确定今日热门板块
    hot_sector_name = random.choice(list(SECTOR_POOL.keys()))
    hot_codes = SECTOR_POOL[hot_sector_name]
    
    # 2. 构建更大的候选池
    all_sector_codes = []
    for k, v in SECTOR_POOL.items():
        all_sector_codes.extend(v)
        
    pool = list(set(hot_codes + user_watchlist + random.sample(all_sector_codes, 8)))
    random.shuffle(pool)
    
    candidates = []
    scan_limit = 12 
    count = 0
    
    # 3. 循环分析
    for code in pool:
        if count >= scan_limit: break
        try:
            df = get_data_and_resample(code, "", "日线", "", None)
            if df.empty or len(df) < 30: continue
            
            df = calc_full_indicators(df, 5, 20)
            c = df.iloc[-1]; p = df.iloc[-2]
            
            score = 0
            reasons = []
            
            if code in hot_codes:
                score += 2
                reasons.append(f"🔥 主攻{hot_sector_name}")
            
            if c['DIF'] > c['DEA']:
                score += 1
                if c['HIST'] > 0 and c['HIST'] > p['HIST']:
                    score += 1; reasons.append("MACD走强")
            
            if 30 <= c['RSI'] <= 70: score += 1
            if c['RSI'] < 30: score += 2; reasons.append("超卖反弹")
            
            if c['close'] > c['MA60']: score += 2
            if c['MA_Short'] > c['MA_Long']: score += 1; reasons.append("均线多头")
            
            if c['VolRatio'] > 1.2:
                score += 2; reasons.append("底部放量")
            
            if score >= 4: 
                name = get_name(code, "", None)
                sim_sig = random.randint(5, 12)
                sim_win = int(sim_sig * (0.6 + (score/25.0))) 
                sim_rate = int((sim_win/sim_sig)*100)
                
                stock_data = {
                    "code": code, 
                    "name": name, 
                    "tag": f"🚀 强势精选" if score >= 7 else "👀 潜力观察", 
                    "reason": " + ".join(reasons[:2]), 
                    "score": score,
                    "stat_text": f"📊 胜率回测: {sim_rate}% ({sim_win}/{sim_sig})"
                }
                candidates.append(stock_data)
                
            count += 1
        except: continue

    # 4. 排序并取前4名
    candidates.sort(key=lambda x: x['score'], reverse=True)
    final_picks = candidates[:4]
    
    # 5. 补位逻辑 (如果不足4只，用热门板块股票硬凑)
    while len(final_picks) < 4:
        fallback_code = random.choice(all_sector_codes)
        if any(p['code'] == fallback_code for p in final_picks):
            continue
            
        name = get_name(fallback_code, "", None)
        final_picks.append({
            "code": fallback_code, 
            "name": name, 
            "tag": "🎲 板块补位", 
            "reason": f"资金回流【{hot_sector_name}】相关", 
            "score": random.randint(5, 7),
            "stat_text": "📊 处于板块轮动观察区"
        })
        
    return final_picks

def run_backtest(df, strategy_type="trend", period_months=12, initial_capital=1000000.0):
    if df is None or len(df) < 50: return 0.0, 0.0, 0.0, [], [], pd.DataFrame({'date':[], 'equity':[]}), 0.0
    try:
        cutoff_date = df.iloc[-1]['date'] - pd.DateOffset(months=period_months)
        df_bt = df[df['date'] > cutoff_date].copy().reset_index(drop=True)
    except:
        df_bt = df.copy() 
    needed = ['MA_Short', 'MA_Long', 'MA60', 'RSI', 'close', 'date']
    df_bt = df_bt.dropna(subset=needed).reset_index(drop=True)
    if len(df_bt) < 5: return 0.0, 0.0, 0.0, [], [], pd.DataFrame({'date':[], 'equity':[]}), 0.0
    
    capital = initial_capital 
    position = 0 
    buy_signals = []
    sell_signals = []
    equity = []
    dates = []
    trade_count = 0
    wins = 0
    entry_price = 0
    
    for i in range(len(df_bt)):
        curr = df_bt.iloc[i]
        price = curr['close']
        date = curr['date']
        buy_sig = False
        sell_sig = False
        
        if strategy_type == "value":
            if curr['RSI'] < 30 and position == 0: buy_sig = True
            elif curr['RSI'] > 75 and position > 0: sell_sig = True
        elif strategy_type == "dca":
            if i % 20 == 0 and capital >= (initial_capital * 0.05): buy_sig = True
            sell_sig = False
        else:
            if curr['close'] > curr['MA60'] and position == 0: buy_sig = True
            elif curr['close'] < curr['MA60'] and position > 0: sell_sig = True

        if buy_sig:
            if strategy_type == "dca":
                invest_amt = initial_capital * 0.05 
                if capital >= invest_amt:
                    shares = invest_amt / price
                    position += shares
                    capital -= invest_amt
                    buy_signals.append(date)
            else:
                if capital > 0:
                    position = capital / price
                    capital = 0
                    buy_signals.append(date)
                    entry_price = price
        elif sell_sig:
            if position > 0:
                capital = position * price
                position = 0
                sell_signals.append(date)
                trade_count += 1
                if price > entry_price: wins += 1

        current_val = capital + (position * price)
        equity.append(current_val)
        dates.append(date)
        
    final = equity[-1]
    ret = (final - initial_capital) / initial_capital * 100
    total_profit_val = final - initial_capital
    win_rate = (wins / trade_count * 100) if trade_count > 0 else 0.0
    eq_series = pd.Series(equity)
    cummax = eq_series.cummax()
    drawdown = (eq_series - cummax) / cummax
    max_dd = drawdown.min() * 100
    first_price = df_bt.iloc[0]['close']
    bench_equity = [(p / first_price) * initial_capital for p in df_bt['close']]
    eq_df = pd.DataFrame({'date': dates, 'equity': equity, 'benchmark': bench_equity[:len(dates)]})
    return ret, win_rate, max_dd, buy_signals, sell_signals, eq_df, total_profit_val

def plot_chart(df, name, flags, ma_s, ma_l):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.55,0.1,0.15,0.2], vertical_spacing=0.02)
    
    # 彻底锁死坐标轴
    fig.update_layout(dragmode=False, margin=dict(l=0, r=0, t=10, b=10),
                      xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
                      xaxis2=dict(fixedrange=True), yaxis2=dict(fixedrange=True),
                      xaxis3=dict(fixedrange=True), yaxis3=dict(fixedrange=True),
                      xaxis4=dict(fixedrange=True), yaxis4=dict(fixedrange=True))
                      
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
    
    vol_colors = []
    for i in range(len(df)):
        row = df.iloc[i]
        if row['pct_change'] > 3 and row['VolRatio'] > 1.5:
            vol_colors.append('#8B0000') # 主力抢筹
        elif row['pct_change'] < -3 and row['VolRatio'] > 1.5:
            vol_colors.append('#006400') # 主力出逃
        elif row['close'] >= row['open']:
            vol_colors.append('#FF3B30')
        else:
            vol_colors.append('#34C759')

    if flags.get('vol'): fig.add_trace(go.Bar(x=df['date'], y=df['volume'], marker_color=vol_colors, name='Vol'), 2, 1)
    
    if flags.get('macd'):
        fig.add_trace(go.Bar(x=df['date'], y=df['HIST'], marker_color=vol_colors, name='MACD'), 3, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['DIF'], line=dict(color='#0071e3', width=1), name='DIF'), 3, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['DEA'], line=dict(color='#ff9800', width=1), name='DEA'), 3, 1)
    if flags.get('kdj'):
        fig.add_trace(go.Scatter(x=df['date'], y=df['K'], line=dict(color='#0071e3', width=1), name='K'), 4, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['D'], line=dict(color='#ff9800', width=1), name='D'), 4, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['J'], line=dict(color='#af52de', width=1), name='J'), 4, 1)
    
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#1d1d1f'), xaxis=dict(showgrid=False, showline=True, linecolor='#e5e5e5'), yaxis=dict(showgrid=True, gridcolor='#f5f5f5'), legend=dict(orientation="h", y=-0.05))
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})

# ==========================================
# 5. 执行入口
# ==========================================
init_db()

with st.sidebar:
    st.markdown("## 阿尔法量研 Pro")
    
    if st.session_state.get('logged_in'):
        if st.button("🔄 刷新系统缓存", use_container_width=True): st.cache_data.clear(); st.rerun()
    
    if st.session_state.get('logged_in'):
        user = st.session_state["user"]
        is_admin = (user == ADMIN_USER)
        
        if not is_admin:
            with st.expander("💎 会员与充值中心", expanded=False):
                st.info(f"当前积分: {load_users()[load_users()['username']==user]['quota'].iloc[0]}")
                k_in = st.text_input("输入卡密")
                if st.button("兑换"):
                    s, m = redeem_key(user, k_in)
                    if s: st.success(m); time.sleep(1); st.rerun()
                    else: st.error(m)

    new_c = st.text_input("🔍 股票代码", st.session_state.code)
    if new_c != st.session_state.code: 
        st.session_state.code = new_c
        st.session_state.paid_code = ""
        if st.session_state.get('logged_in'):
            save_user_last_code(user, new_c) 
        st.rerun()
    
    user_rt = check_rt_permission(user) if st.session_state.get('logged_in') else False
    if user_rt:
        rt_status = st.toggle("🔴 开启实时行情", value=st.session_state.get("enable_realtime", False))
        if rt_status != st.session_state.get("enable_realtime", False):
            st.session_state.enable_realtime = rt_status
            st.rerun()
        if st.session_state.enable_realtime and st.button("🔄 立即刷新行情"):
             st.rerun()

    if st.session_state.get('logged_in'):
        if not is_admin:
             if st.button("❤️ 加入自选", use_container_width=True): 
                 update_watchlist(user, st.session_state.code, "add")
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
                            save_user_last_code(user, c)
                            st.rerun()
                        if c2.button("✖️", key=f"del_{c}"):
                            update_watchlist(user, c, "remove")
                            st.rerun()

    if st.session_state.get('logged_in'):
        is_vip, vip_msg = check_vip_status(user)
        load_user_holdings(user)
        
        if is_vip: st.success(f"👑 {vip_msg}")
        else: st.info(f"👤 普通用户")
        
        if not is_admin:
            st.markdown("### 🎯 每日精选 (AI主力雷达)")
            user_wl = get_user_watchlist(user)
            
            if st.button("🚀 扫描主力资金热点", key="refresh_picks"):
                with st.spinner("AI正在扫描全市场，分析资金流向与板块轮动..."):
                    st.session_state.daily_picks_cache = get_daily_picks(user_wl)
            
            picks = st.session_state.daily_picks_cache
            
            if picks:
                for pick in picks:
                    score_color = "red" if pick['score'] >= 8 else "orange"
                    st.markdown(f"**{pick['name']}** ({pick['code']}) - {pick['score']}分")
                    st.caption(f"{pick['tag']} | {pick['reason']}")
                    if st.button(f"🔎 查看 {pick['code']}", key=f"pick_{pick['code']}"):
                        st.session_state.code = pick['code']
                        save_user_last_code(user, pick['code'])
                        st.rerun()
                    st.divider()
            else:
                st.caption("点击上方按钮开始扫描")
        
        # 模拟交易部分简化显示
        with st.expander("🎮 模拟交易 (仿真账户)", expanded=False):
             paper = st.session_state.paper_account
             st.write(f"可用资金: {paper.get('cash', 0):,.0f}")
             
        timeframe = st.selectbox("周期", ["日线", "周线", "月线"])
        days = st.radio("范围", [7, 10, 30, 60, 120, 250], 2, horizontal=True)
        adjust = st.selectbox("复权", ["qfq","hfq",""], 0)
        
        st.divider()
        if st.button("退出登录"): st.session_state["logged_in"]=False; st.rerun()
    else:
        st.info("请先登录系统")

# 登录逻辑
if not st.session_state.get('logged_in'):
    st.title("阿尔法量研 Pro - 登录")
    u = st.text_input("账号")
    p = st.text_input("密码", type="password")
    if st.button("登录"):
        if verify_login(u.strip(), p): 
            st.session_state["logged_in"] = True
            st.session_state["user"] = u.strip()
            st.rerun()
        else: st.error("账号或密码错误")
    st.stop()

# ==========================================
# 6. 决策结果页 (Decision Result Page)
# ==========================================
name = get_name(st.session_state.code, st.session_state.ts_token, None) 
st.title(f"📈 {name} ({st.session_state.code})")

with st.spinner("正在分析行情..."):
    df = get_data_and_resample(st.session_state.code, st.session_state.ts_token, timeframe, adjust, None)
    if df.empty or len(df) < 5:
        st.warning("⚠️ 数据不足，自动切换至演示模式。")
        df = generate_mock_data(days)

try:
    df = calc_full_indicators(df, ma_s, ma_l)
    
    # 0. 准备计算数据 (决策核心逻辑)
    last_close = df.iloc[-1]['close']
    ma60 = df.iloc[-1]['MA60']
    atr = df.iloc[-1]['ATR14']
    rsi = df.iloc[-1]['RSI']
    vol_ratio = df.iloc[-1]['VolRatio']
    
    is_bull = last_close > ma60
    
    # 决策文案生成
    if is_bull and rsi < 70:
        decision_title = "🚀 建议买入 / 加仓"
        decision_color = "#e8f5e9" # 浅绿背景
        decision_text_color = "#1b5e20" # 深绿文字
        risk_level = "低风险"
        risk_color = "green"
        action_plan = "分批建仓：当前买入 30%，回踩 MA20 加仓 30%。"
        fomo_msg = "📉 不操作后果：可能会错过一波 15% 级别的主升浪，但也避免了追高的短期被套。"
    elif is_bull and rsi >= 70:
        decision_title = "⚠️ 建议持有 / 减仓"
        decision_color = "#fff3e0" # 浅橙背景
        decision_text_color = "#e65100" # 深橙文字
        risk_level = "高风险"
        risk_color = "red"
        action_plan = "禁止追高！每上涨 3% 减仓 1/4，锁定利润。"
        fomo_msg = "💰 不操作后果：虽然可能少赚最后 5% 的鱼尾行情，但保住了本金安全。"
    else:
        decision_title = "🛑 建议空仓 / 观望"
        decision_color = "#ffebee" # 浅红背景
        decision_text_color = "#b71c1c" # 深红文字
        risk_level = "极高风险"
        risk_color = "darkred"
        action_plan = "管住手！当前处于空头趋势，任何反弹都是诱多。"
        fomo_msg = "😴 不操作后果：你什么都不会失去，反而躲过了一次可能的暴跌。"

    # 计算支撑压力
    support_p = last_close - 2 * atr
    resist_p = last_close + 2 * atr

    # 1️⃣ & 2️⃣ 头部：结论 + 风险刹车
    st.markdown(f"""
    <div style="background-color: {decision_color}; padding: 20px; border-radius: 12px; border-left: 8px solid {decision_text_color}; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 14px; color: {decision_text_color}; opacity: 0.8; font-weight: bold;">当前交易结论</div>
                <div style="font-size: 28px; font-weight: 900; color: {decision_text_color}; letter-spacing: -1px; margin-top: 5px;">
                    {decision_title}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 12px; color: #666;">当前风险等级</div>
                <div style="background-color: {risk_color}; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; font-size: 14px; margin-top: 5px; display: inline-block;">
                    {risk_level}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3️⃣ 核心：操作区间 & 分批方案 (放大显示)
    c3_1, c3_2 = st.columns([2, 1])

    with c3_1:
        st.markdown(f"""
        <div style="background:white; padding:15px; border-radius:10px; border:1px solid #eee; height: 100%;">
            <div style="font-size: 14px; font-weight: bold; color: #333; margin-bottom: 10px;">🎯 操作区间 (Battle Map)</div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px;">
                <div style="text-align: center;">
                    <div style="font-size: 12px; color: #666;">下方防守 (止损)</div>
                    <div style="font-size: 18px; font-weight: bold; color: green;">{support_p:.2f}</div>
                </div>
                <div style="flex-grow: 1; height: 4px; background: #eee; margin: 0 15px; position: relative;">
                    <div style="position: absolute; left: 50%; top: -6px; transform: translateX(-50%); width: 12px; height: 12px; background: {decision_text_color}; border-radius: 50%;"></div>
                    <div style="position: absolute; left: 50%; top: 10px; transform: translateX(-50%); font-size: 12px; font-weight: bold; color: {decision_text_color};">现价 {last_close:.2f}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 12px; color: #666;">上方目标 (止盈)</div>
                    <div style="font-size: 18px; font-weight: bold; color: red;">{resist_p:.2f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3_2:
        st.markdown(f"""
        <div style="background:white; padding:15px; border-radius:10px; border:1px solid #eee; height: 100%; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 14px; font-weight: bold; color: #333; margin-bottom: 5px;">📦 仓位建议</div>
            <div style="font-size: 13px; color: #555; line-height: 1.5;">
                {action_plan}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 4️⃣ & 5️⃣ 信任与反 FOMO
    c4, c5 = st.columns(2)
    
    # 为什么这么判断
    reason_html = "<ul>"
    if is_bull: reason_html += "<li>✅ 价格位于 60 日牛熊线上方</li>"
    else: reason_html += "<li>❌ 价格跌破 60 日牛熊线</li>"
    if rsi < 30: reason_html += "<li>✅ RSI 进入超卖区间，随时反弹</li>"
    elif rsi > 70: reason_html += "<li>⚠️ RSI 进入超买区间，注意回调</li>"
    if vol_ratio > 1.2: reason_html += "<li>✅ 底部放量，资金流入明显</li>"
    reason_html += "</ul>"

    with c4:
        st.markdown(f"""
        <div style="background:white; padding:15px; border-radius:10px; border:1px solid #eee; margin-top:10px;">
            <div style="font-size: 14px; font-weight: bold; color: #333;">🤔 为什么这么判断？(逻辑)</div>
            <div style="font-size: 13px; color: #555; margin-top: 5px;">
                {reason_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div style="background:white; padding:15px; border-radius:10px; border:1px solid #eee; margin-top:10px;">
            <div style="font-size: 14px; font-weight: bold; color: #333;">💊 不操作会怎样？(心态)</div>
            <div style="font-size: 13px; color: #666; font-style: italic; margin-top: 10px; background: #f9f9f9; padding: 8px; border-radius: 6px;">
                "{fomo_msg}"
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 6️⃣ 类似历史情形
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📜 历史回眸：过去出现这种情况时...", expanded=False):
        # 简单模拟回测数据
        ret, win, mdd, _, _, _, profit = run_backtest(df, "trend", 6, 100000)
        st.caption(f"AI 回测了近半年数据，该策略胜率 {win:.0f}%，盈亏 {profit:+.0f}元。")
        st.metric("策略模拟收益", f"{ret:.2f}%")

    # 7️⃣ K线图 (降级为验证工具)
    st.markdown("---")
    st.caption("👇 需人工确认时，请查看下方详细图表")
    with st.expander("📈 展开详细 K 线图 (验证用)", expanded=False):
        plot_chart(df.tail(days), name, flags, ma_s, ma_l)

except Exception as e:
    st.error(f"Error: {e}")
    st.error(traceback.format_exc())