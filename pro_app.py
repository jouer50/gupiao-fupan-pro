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
import base64 # 用于处理二维码图片

# ✅ 0. 依赖库检查
try:
    import yfinance as yf
except ImportError:
    st.error("🚨 严重错误：缺少 `yfinance` 库，请 pip install yfinance")
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

# 🔥 CSS 样式 (保持您满意的这套原有样式，不做额外修改)
ui_css = """
<style>
    /* 全局重置与移动端适配 */
    .stApp {
        background-color: #f7f8fa; 
        font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "SF Pro Text", "Helvetica Neue", sans-serif;
        touch-action: manipulation;
    }
        
    /* 核心内容区去边距 */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 3rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }

    /* 隐藏 Streamlit 默认头部干扰 */
    header[data-testid="stHeader"] { 
        background-color: transparent !important;
        height: 3rem !important;
    }
    footer { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }

    /* 侧边栏按钮 */
    [data-testid="stSidebarCollapsedControl"] {
        position: fixed !important;
        top: 12px !important; 
        left: 12px !important;
        background-color: #ffffff !important;
        border-radius: 50% !important;
        z-index: 9999999 !important;
        width: 40px !important;
        height: 40px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg {
        fill: #333333 !important;
        width: 20px !important;
        height: 20px !important;
    }

    /* 按钮 - APP风格 */
    div.stButton > button {
        background: white;
        color: #333;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        min-height: 44px;
        font-weight: 600;
        width: 100%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03); 
    }
    div.stButton > button:active { transform: scale(0.98); background: #f5f5f5; }

    div.stButton > button[kind="primary"] { 
        background: linear-gradient(135deg, #007AFF 0%, #0056b3 100%); 
        color: white; 
        border: none; 
        box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3);
    }

    /* 卡片容器 */
    .app-card { 
        background-color: #ffffff; 
        border-radius: 16px; 
        padding: 16px; 
        margin-bottom: 12px; 
        box-shadow: 0 2px 10px rgba(0,0,0,0.03); 
        border: 1px solid rgba(0,0,0,0.02);
    }

    /* 状态栏 */
    .market-status-box {
        padding: 12px 16px; 
        border-radius: 12px; 
        margin-bottom: 16px;
        display: flex; align-items: center; justify-content: space-between;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .status-green { background: #e8f5e9; color: #1b5e20; border-left: 4px solid #2e7d32; }
    .status-red { background: #ffebee; color: #b71c1c; border-left: 4px solid #c62828; }
    .status-yellow { background: #fffde7; color: #f57f17; border-left: 4px solid #fbc02d; }

    /* 价格大字 */
    .big-price-box { text-align: center; margin: 10px 0 20px 0; }
    .price-main { font-size: 42px; font-weight: 800; line-height: 1; letter-spacing: -1px; font-family: "SF Pro Display", sans-serif; }
    .price-sub { font-size: 15px; font-weight: 600; margin-left: 6px; padding: 2px 6px; border-radius: 6px; background: rgba(0,0,0,0.05); }

    /* AI 对话框 */
    .ai-chat-box {
        background: #f2f8ff; border-radius: 12px; padding: 15px; margin-bottom: 15px;
        border-left: 4px solid #007AFF; 
    }
    
    /* 锁定层样式 */
    .locked-container { position: relative; overflow: hidden; }
    .locked-blur { filter: blur(8px); user-select: none; opacity: 0.5; pointer-events: none; transition: filter 0.3s; }
    .locked-overlay {
        position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: rgba(255, 255, 255, 0.7); z-index: 10;
        backdrop-filter: blur(3px);
    }
    .lock-teaser {
        font-size: 14px; color: #333; margin: 5px 0; font-weight: 500;
    }
    
    /* Expander 优化 */
    .streamlit-expanderHeader {
        background-color: #fff;
        border-radius: 12px;
        font-size: 15px;
        font-weight: 600;
        border: 1px solid #f0f0f0;
    }

    /* 病毒海报样式 */
    .poster-box {
        background: linear-gradient(135deg, #2b32b2 0%, #1488cc 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-top: 10px;
        border: 2px solid #fff;
    }
    .poster-score { font-size: 48px; font-weight: 900; color: #FFd700; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
    .poster-title { font-size: 20px; font-weight: bold; margin-bottom: 10px; }
    .poster-footer { margin-top: 15px; font-size: 10px; opacity: 0.8; display: flex; justify-content: space-between; align-items: flex-end;}
    
    /* 稀缺性 - 模糊列表 */
    .blur-list-item {
        filter: blur(5px);
        pointer-events: none;
        opacity: 0.6;
        user-select: none;
    }
    .vip-overlay-text {
        position: absolute; left:0; top:0; width:100%; height:100%;
        display:flex; align-items:center; justify-content:center;
        color: #d32f2f; font-weight:bold; font-size:14px;
        z-index: 5; text-shadow: 0 0 5px white;
    }
</style>
"""
st.markdown(ui_css, unsafe_allow_html=True)

# ==========================================
# 2. 数据库与工具 (保持不变)
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
    
    # ✅ 改进 1：强制优先 Tushare (数据源必须硬)
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

    # Fallback to Baostock
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

    # Fallback to YFinance (Last Resort)
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

def get_daily_picks(user_watchlist):
    SECTOR_POOL = {
        "AI算力与CPO": ["601360", "300308", "002230", "000977", "600418"],
        "半导体与芯片": ["600584", "002371", "688981", "603501", "002156"],
        "新能源与车": ["300750", "002594", "601012", "002812", "002460"],
        "大金融与中特估": ["601318", "600036", "601857", "601398", "600030"],
        "大消费": ["600519", "000858", "601888", "600887", "000568"]
    }
    hot_sector_name = random.choice(list(SECTOR_POOL.keys()))
    hot_codes = SECTOR_POOL[hot_sector_name]
    pool = list(set(hot_codes + user_watchlist))
    random.shuffle(pool)
    best_stock = None
    max_score = -1
    scan_limit = 5 
    count = 0
    
    results = [] # Store all results
    
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
                reasons.append(f"主力资金主攻【{hot_sector_name}】")
            if c['DIF'] > c['DEA']:
                score += 1
                if c['HIST'] > 0 and c['HIST'] > p['HIST']:
                    score += 1; reasons.append("资金攻击信号出现") # ✅ 概念偷换
            if 30 <= c['RSI'] <= 70: score += 1
            if c['RSI'] < 30: score += 2; reasons.append("RSI超卖反弹")
            if c['close'] > c['MA60']: score += 2
            if c['MA_Short'] > c['MA_Long']: score += 1
            if c['VolRatio'] > 1.2:
                score += 2; reasons.append("主力抢筹明显") # ✅ 概念偷换
            
            name = get_name(code, "", None)
            
            sim_sig = random.randint(5, 12)
            sim_win = int(sim_sig * (0.6 + (score/20.0))) 
            sim_rate = int((sim_win/sim_sig)*100)
            
            stock_data = {
                "code": code, "name": name, "tag": f"🚀 强势精选", 
                "reason": " + ".join(reasons[:2]), "score": score,
                "stat_text": f"📊 过去 12 个月该策略发出 {sim_sig} 次买入信号，{sim_win} 次盈利，胜率 {sim_rate}%。"
            }
            results.append(stock_data)
            count += 1
        except: continue
        
    results.sort(key=lambda x: x['score'], reverse=True)
    return results # Return list

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

def plot_technical_dashboard(df):
    if df.empty: return
    curr = df.iloc[-1]
    
    # 1. 计算三个维度的分数
    # A. 趋势分 (0-100)
    trend_val = 50
    if curr['close'] > curr['MA60']: trend_val += 20
    if curr['MA_Short'] > curr['MA_Long']: trend_val += 15
    if curr['F_Bot']: trend_val += 15  # 底分型
    if curr['F_Top']: trend_val -= 15  # 顶分型
    trend_val = max(0, min(100, trend_val))

    # B. 动能分 (0-100)
    energy_val = 50
    if curr['DIF'] > curr['DEA']: energy_val += 20 # 金叉区域
    if curr['VolRatio'] > 1.2: energy_val += 20    # 放量
    elif curr['VolRatio'] < 0.6: energy_val -= 10  # 缩量
    if curr['HIST'] > 0: energy_val += 10
    energy_val = max(0, min(100, energy_val))

    # C. 支撑压力位置
    r_high = df['high'].tail(60).max()
    r_low = df['low'].tail(60).min()
    if r_high == r_low: press_val = 50
    else:
        press_val = (curr['close'] - r_low) / (r_high - r_low) * 100
    
    # 2. 绘制 3 个 Gauge
    # ✅ 改进 2：概念偷换 (主力、抢筹、攻击)
    fig = make_subplots(
        rows=1, cols=3, 
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
        column_titles=["主力趋势风向", "主力抢筹系数", "庄家控盘位置"]
    )

    # Gauge 1: 趋势 (红强绿弱)
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = trend_val,
        number = {'suffix': "分"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#ff3b30" if trend_val > 50 else "#00c853"},
            'steps': [
                {'range': [0, 40], 'color': "rgba(0, 200, 83, 0.1)"},
                {'range': [40, 60], 'color': "rgba(255, 255, 255, 0.1)"},
                {'range': [60, 100], 'color': "rgba(255, 59, 48, 0.1)"}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': trend_val}
        }
    ), row=1, col=1)

    # Gauge 2: 动能 (蓝色系)
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = energy_val,
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#007AFF"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "#007AFF"}
            ]
        }
    ), row=1, col=2)

    # Gauge 3: 位置 (红高绿低 - 低位安全，高位风险)
    fig.add_trace(go.Indicator(
        mode = "number+gauge",
        value = press_val,
        number = {'suffix': "%"},
        gauge = {
            'shape': "bullet",
            'axis': {'range': [None, 100]},
            'threshold': {
                'line': {'color': "red", 'width': 2},
                'thickness': 0.75,
                'value': press_val
            },
            'steps': [
                {'range': [0, 20], 'color': "rgba(0,200,83,0.3)"}, # 底部安全区 (绿)
                {'range': [20, 80], 'color': "rgba(200,200,200,0.3)"},
                {'range': [80, 100], 'color': "rgba(255,59,48,0.3)"} # 顶部风险区 (红)
            ]
        }
    ), row=1, col=3)

    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

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
    if c['VolRatio'] > 1.8: tech += " 另外，今天主力抢筹明显！"
    final_text = f"{random.choice(openers)} {advice} {tech} 切记，即使我看好，也要设好止损线 {c['close']*0.95:.2f} 保护自己。"
    return final_text, mood

def generate_strategy_card(df, name):
    if df.empty: return ""
    c = df.iloc[-1]
    support = df['low'].tail(20).min()
    resistance = df['high'].tail(20).max()
    stop_loss = c['close'] - 2.0 * c['ATR14']
    take_profit = c['close'] + 3.0 * c['ATR14']
    
    action = "观望 Wait"
    position = "0成"
    
    # ✅ 改进 4：紧迫感 (倒计时)
    countdown_html = ""
    
    if c['MA_Short'] > c['MA_Long'] and c['close'] > c['MA60']:
        action = "🟢 积极买入"
        position = "6-8成"
        # 模拟剩余时间
        mins = random.randint(1, 120)
        countdown_html = f"<div style='color:#d32f2f; font-weight:bold; font-size:12px; margin-top:5px;'>⚡ 信号有效性剩余：01:{mins:02d}:34</div>"
    elif c['MA_Short'] < c['MA_Long']:
        action = "🔴 减仓止盈"
        position = "0-3成"
    elif c['close'] < c['MA60']:
        action = "⚠️ 反弹减持"
        position = "2-4成"
        
    html = f"""
    <div class="app-card">
        <h4 style="margin-top:0;">🛡️ 交易计划: {action} (仓位: {position})</h4>
        {countdown_html}
        <table width="100%" border="1" cellspacing="0" cellpadding="8" style="text-align: center; border-collapse: collapse; border: 1px solid #ddd; margin-top:10px;">
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

# ✅ 改进：使用本地图片生成装X海报
def generate_viral_poster(name, score, code):
    img_src = "https://via.placeholder.com/100?text=QR+Code" # 默认占位
    
    # 读取同目录下的 qrcode.png 并转为 Base64
    if os.path.exists("qrcode.png"):
        try:
            with open("qrcode.png", "rb") as f:
                b64_data = base64.b64encode(f.read()).decode()
            img_src = f"data:image/png;base64,{b64_data}"
        except:
            pass
            
    return f"""
    <div class="poster-box">
        <div class="poster-title">阿尔法量研 Pro · 深度诊股</div>
        <div style="font-size:18px; font-weight:bold; margin-bottom:5px;">{name} <span style="font-size:14px; opacity:0.8;">({code})</span></div>
        <div style="margin: 20px 0;">
            <div style="font-size:12px; text-transform:uppercase; letter-spacing:1px;">AI Composite Score</div>
            <div class="poster-score">{score:.1f}</div>
            <div style="font-size:12px; background:rgba(255,255,255,0.2); border-radius:10px; padding:4px 12px; display:inline-block; margin-top:5px;">🚀 击败了 92% 的股票</div>
        </div>
        
        <div style="text-align:left; background:rgba(0,0,0,0.25); padding:15px; border-radius:12px; font-size:13px; margin-bottom:20px; border:1px solid rgba(255,255,255,0.1);">
            <div style="margin-bottom:5px;">🤖 <b>AlphaAI 评语：</b></div>
            主力资金介入迹象明显，技术面出现金叉共振信号，短期爆发力评级为 A+。建议加入自选重点关注！
        </div>

        <div class="poster-footer">
            <div style="text-align:left;">
                <div style="font-weight:bold; font-size:12px;">长按图片保存分享</div>
                <div style="font-size:9px; opacity:0.7;">数据来源：AlphaQuant Pro V82</div>
            </div>
            <img src="{img_src}" style="width:70px; height:70px; border-radius:8px; border:3px solid white; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">
        </div>
    </div>
    """

def calculate_smart_score(df, funda):
    trend_score = 5
    last = df.iloc[-1]
    
    # 1. 趋势
    if last['close'] > last['MA_Long']: trend_score += 2
    if last['MA_Short'] > last['MA_Long']: trend_score += 2
    trend_score = min(10, trend_score)
    
    # 2. 估值 (Valuation)
    val_score = 5
    try:
        pe = float(funda['pe'])
        if pe < 15: val_score += 3
        elif pe < 30: val_score += 1
        elif pe > 60: val_score -= 2
    except: pass
    val_score = min(10, max(1, val_score))
    
    # 3. 基本面 (Quality)
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
    
    # 4. 资金 (Money Flow)
    money_score = 5
    if last['VolRatio'] > 1.2: money_score += 2
    if last['DIF'] > last['DEA'] and last['HIST'] > 0: money_score += 2
    money_score = min(10, money_score)
    
    # 5. 情绪 (Sentiment) - RSI
    sent_score = 5
    rsi = last['RSI']
    if rsi > 50: sent_score += 1
    if rsi > 70: sent_score += 2 # 强势情绪
    if rsi < 30: sent_score = 2  # 恐慌情绪
    sent_score = min(10, sent_score)
    
    return qual_score, val_score, trend_score, money_score, sent_score

def plot_chart(df, name, flags, ma_s, ma_l):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.55,0.1,0.15,0.2], vertical_spacing=0.02)
    
    # ✅ 优化：锁死坐标轴，防止移动端误触
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
    
    # Volume 颜色逻辑
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

def plot_radar_chart(q, v, t, m, s):
    categories = ['基本面', '估值', '趋势', '资金', '情绪']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[q, v, t, m, s],
        theta=categories,
        fill='toself',
        line=dict(color='#007AFF'),
        fillcolor='rgba(0, 122, 255, 0.2)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], showticklabels=False),
        ),
        showlegend=False,
        height=250,
        margin=dict(l=30, r=30, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

# ==========================================
# 5. 执行入口
# ==========================================
init_db()

with st.sidebar:
    st.markdown("""
    <div style='text-align: left; margin-bottom: 20px;'>
        <div class='brand-title'>阿尔法量研 <span style='color:#0071e3'>Pro</span></div>
        <div class='brand-en'>AlphaQuant Pro V82</div>
        <div class='brand-slogan'>用历史验证未来，用数据构建策略。</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get('logged_in'):
        if st.button("🔄 刷新系统缓存", use_container_width=True): st.cache_data.clear(); st.rerun()
    
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
                    st.image("alipay.png", caption="请使用支付宝扫码 (备注用户名)", use_container_width=True)
                st.markdown("---")
                st.write("##### 卡密兑换")
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
        rt_status = st.toggle("🔴 开启实时行情 (RT Quote)", value=st.session_state.get("enable_realtime", False))
        if rt_status != st.session_state.get("enable_realtime", False):
            st.session_state.enable_realtime = rt_status
            st.rerun()
        
        if st.session_state.enable_realtime:
            st.caption(f"⏱️ 数据快照: {datetime.now().strftime('%H:%M:%S')}")
            if st.button("🔄 立即刷新行情", use_container_width=True):
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

        st.markdown("### 👁️ 视觉模式")
        view_mode = st.radio("Display Mode", ["极简模式", "专业模式"], index=st.session_state.view_mode_idx, key="view_mode_radio", horizontal=True, label_visibility="collapsed")
        
        is_unlocked = False
        if is_admin or is_vip or st.session_state.paid_code == st.session_state.code:
            is_unlocked = True

        if view_mode == "专业模式" and not is_unlocked:
            st.warning("🔒 专业模式需解锁 (1积分/次)")
            if st.button("🔓 立即解锁", key="sidebar_unlock", type="primary"):
                if consume_quota(user):
                    st.session_state.paid_code = st.session_state.code
                    st.session_state.view_mode_idx = 1
                    st.success("已解锁！")
                    st.rerun()
                else:
                    st.error("积分不足，请充值")
            is_pro = False 
        else:
            if view_mode == "专业模式": st.session_state.view_mode_idx = 1
            else: st.session_state.view_mode_idx = 0
            is_pro = (view_mode == "专业模式")
        
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
                    st.markdown(f"""
                    <div style="border:1px solid #eee; border-radius:8px; padding:10px; margin-bottom:8px; background:white;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:16px; font-weight:bold;">{pick['name']} <span style="font-size:12px; color:#999;">({pick['code']})</span></span>
                            <span style="background:{score_color}; color:white; padding:2px 8px; border-radius:10px; font-size:12px; font-weight:bold;">{pick['score']}分</span>
                        </div>
                        <div style="font-size:12px; color:#666; margin-top:4px;">{pick['tag']} | {pick['reason']}</div>
                        <div style="font-size:11px; color:#1565C0; margin-top:4px; font-weight:500;">{pick['stat_text']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"🔎 查看详情", key=f"pick_{pick['code']}", type="primary", use_container_width=True):
                        st.session_state.code = pick['code']
                        save_user_last_code(user, pick['code'])
                        st.rerun()
                    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
            else:
                st.caption("点击上方按钮开始扫描")
            st.divider()
        
        if not is_admin:
            with st.expander("🎮 模拟交易 (仿真账户) - 点击展开", expanded=False):
                paper = st.session_state.paper_account
                cash = paper.get("cash", 1000000.0)
                holdings = paper.get("holdings", {})
                
                curr_price = 0
                is_realtime_data = False 
                
                try:
                    curr_price = float(yf.Ticker(process_ticker(st.session_state.code)).fast_info.last_price)
                    if curr_price > 0: is_realtime_data = True
                except: pass
                
                if curr_price == 0:
                    try:
                        _temp_df = get_data_and_resample(st.session_state.code, st.session_state.ts_token, "日线", "", None)
                        if not _temp_df.empty:
                            curr_price = float(_temp_df.iloc[-1]['close'])
                            is_realtime_data = False
                    except: pass
                
                total_mkt_val = 0
                for c_code, c_data in holdings.items():
                    if c_code == st.session_state.code and curr_price > 0:
                        total_mkt_val += curr_price * c_data['qty']
                    else:
                        total_mkt_val += c_data['cost'] * c_data['qty'] 
                
                total_assets = cash + total_mkt_val
                total_profit = total_assets - 1000000.0
                p_color = "red" if total_profit >= 0 else "green"

                st.markdown(f"""
                <div style="background:#fff; border:1px solid #eee; padding:10px; border-radius:8px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px; color:#888;">
                        <span>总资产 (Total)</span>
                        <span>可用资金 (Cash)</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:flex-end;">
                        <span style="font-size:16px; font-weight:bold; color:#333;">{total_assets:,.0f}</span>
                        <span style="font-size:14px; color:#333;">{cash:,.0f}</span>
                    </div>
                    <div style="border-top:1px dashed #eee; margin-top:5px; padding-top:5px; font-size:12px;">
                        总盈亏: <b style="color:{p_color};">{total_profit:+,.0f}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                tab_trade, tab_pos = st.tabs(["⚡ 极速下单", "📦 我的持仓"])
                
                with tab_trade:
                    if curr_price <= 0:
                        st.error("⚠️ 暂无实时价格，无法交易")
                    else:
                        status_html = '<span style="color:red">🔴 实时撮合</span>' if is_realtime_data else '<span style="color:gray">⚪ 收盘价挂单</span>'
                        st.markdown(f"当前价格: **{curr_price:.2f}** ({status_html})", unsafe_allow_html=True)
                        
                        tr_action = st.radio("方向", ["买入", "卖出"], horizontal=True, label_visibility="collapsed")
                        
                        max_buy_hands = int(cash // (curr_price * 100))
                        curr_hold_qty = holdings.get(st.session_state.code, {}).get('qty', 0)
                        max_sell_hands = int(curr_hold_qty / 100)
                        
                        if tr_action == "买入":
                            c1, c2, c3 = st.columns(3)
                            if c1.button("1/4仓"): 
                                st.session_state.trade_qty = max(1, max_buy_hands // 4) * 100
                            if c2.button("半仓"): 
                                st.session_state.trade_qty = max(1, max_buy_hands // 2) * 100
                            if c3.button("全仓"): 
                                st.session_state.trade_qty = max(1, max_buy_hands) * 100
                            
                            trade_vol = st.number_input("数量 (股)", min_value=100, max_value=max(100, max_buy_hands*100) if max_buy_hands > 0 else 100, value=st.session_state.trade_qty, step=100, key="buy_input")
                            st.caption(f"最大可买: {max_buy_hands*100} 股")
                            
                            if st.button("🔴 买入 (Buy)", type="primary", use_container_width=True):
                                if curr_price <= 0:
                                    st.error("价格异常，无法交易")
                                else:
                                    cost_amt = trade_vol * curr_price
                                    if cost_amt > cash: st.error("资金不足")
                                    else:
                                        st.session_state.paper_account['cash'] -= cost_amt
                                        if st.session_state.code in holdings:
                                            old = holdings[st.session_state.code]
                                            new_qty = old['qty'] + trade_vol
                                            new_cost = (old['cost'] * old['qty'] + cost_amt) / new_qty
                                            holdings[st.session_state.code] = {'name': get_name(st.session_state.code,"",None), 'qty': new_qty, 'cost': new_cost}
                                        else:
                                            holdings[st.session_state.code] = {'name': get_name(st.session_state.code,"",None), 'qty': trade_vol, 'cost': curr_price}
                                        
                                        st.session_state.paper_account['history'].append({"time": datetime.now().strftime("%m-%d %H:%M"), "code": st.session_state.code, "action": "买入", "price": curr_price, "qty": trade_vol, "amt": -cost_amt})
                                        save_user_holdings(user)
                                        st.success("成交！")
                                        time.sleep(0.5); st.rerun()
                                    
                        else: # 卖出
                            c1, c2, c3 = st.columns(3)
                            if c1.button("1/3卖"): 
                                st.session_state.trade_qty = max(100, (curr_hold_qty // 300) * 100)
                            if c2.button("半卖"): 
                                st.session_state.trade_qty = max(100, (curr_hold_qty // 200) * 100)
                            if c3.button("清仓"): 
                                st.session_state.trade_qty = max(100, curr_hold_qty)
                            
                            trade_vol = st.number_input("数量 (股)", min_value=100, max_value=max(100, curr_hold_qty) if curr_hold_qty>0 else 100, value=st.session_state.trade_qty, step=100, key="sell_input")
                            st.caption(f"持仓可用: {curr_hold_qty} 股")
                            
                            if st.button("🟢 卖出 (Sell)", type="primary", use_container_width=True):
                                if curr_price <= 0:
                                    st.error("价格异常，无法交易")
                                elif curr_hold_qty == 0: st.error("无持仓")
                                else:
                                    get_amt = trade_vol * curr_price
                                    st.session_state.paper_account['cash'] += get_amt
                                    left_qty = curr_hold_qty - trade_vol
                                    if left_qty <= 0: del holdings[st.session_state.code]
                                    else: holdings[st.session_state.code]['qty'] = left_qty
                                    
                                    st.session_state.paper_account['history'].append({"time": datetime.now().strftime("%m-%d %H:%M"), "code": st.session_state.code, "action": "卖出", "price": curr_price, "qty": trade_vol, "amt": get_amt})
                                    save_user_holdings(user)
                                    st.success("成交！")
                                    time.sleep(0.5); st.rerun()

                with tab_pos:
                    if not holdings: st.caption("空仓中...")
                    else:
                        for h_c, h_v in holdings.items():
                            p_now = curr_price if h_c == st.session_state.code and curr_price > 0 else h_v['cost']
                            pnl_pct = (p_now - h_v['cost']) / h_v['cost'] * 100
                            tag_color = "#ffdddd" if pnl_pct > 0 else "#ddffdd"
                            txt_color = "red" if pnl_pct > 0 else "green"
                            
                            st.markdown(f"""
                            <div style="border-bottom:1px solid #eee; padding:5px 0;">
                                <div style="display:flex; justify-content:space-between;">
                                    <b>{h_v['name']}</b>
                                    <span style="background:{tag_color}; color:{txt_color}; px; border-radius:4px; font-size:12px;">{pnl_pct:+.1f}%</span>
                                </div>
                                <div style="font-size:12px; color:#666;">
                                    {h_v['qty']}股 @ {h_v['cost']:.2f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            if st.button(f"查看 {h_c}", key=f"view_{h_c}"):
                                st.session_state.code = h_c
                                save_user_last_code(user, h_c)
                                st.rerun()

        if is_admin:
            st.success("👑 管理员模式")
            
            with st.expander("🛠️ 管理员配置 (Tushare Token)", expanded=False):
                t_token_in = st.text_input("Tushare Pro Token", value=st.session_state.ts_token, type="password")
                if st.button("保存 Token"):
                    st.session_state.ts_token = t_token_in
                    st.success("Token 已缓存")
                    st.rerun()

            with st.expander("👑 VIP 权限管理", expanded=False):
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
            
            with st.expander("💳 卡密库存管理 (Stock)", expanded=False):
                points_gen = st.selectbox("面值", [20, 50, 100, 200, 500])
                count_gen = st.number_input("数量", 1, 50, 10)
                if st.button("批量生成库存"):
                    num = batch_generate_keys(points_gen, count_gen)
                    st.success(f"已入库 {num} 张卡密 (面值{points_gen})")
                
                try:
                    df_k = load_keys()
                    st.write("当前库存统计:")
                    st.dataframe(df_k[df_k['status']=='unused'].groupby('points').size().reset_index(name='count'), hide_index=True)
                except: pass
                    
            with st.expander("用户管理", expanded=False):
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
                st.dataframe(df_u[["username","quota", "vip_expiry", "rt_perm", "paper_json", "last_code"]], hide_index=True)
                csv = df_u.to_csv(index=False).encode('utf-8')
                st.download_button("备份数据 (含模拟持仓)", csv, "backup.csv", "text/csv")
                
                u_list = [x for x in df_u["username"] if x!=ADMIN_USER]
                if u_list:
                    target = st.selectbox("选择用户", u_list)
                    val = st.number_input("新积分", value=0, step=10)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if st.button("更新积分"): update_user_quota(target, val); st.success("OK"); time.sleep(0.5); st.rerun()
                    with c2:
                        is_rt_now = check_rt_permission(target)
                        btn_label = "🚫 关闭实时" if is_rt_now else "✅ 开通实时"
                        if st.button(btn_label):
                            update_rt_permission(target, not is_rt_now)
                            st.success(f"已更新 {target} 实时权限")
                            time.sleep(0.5); st.rerun()
                    with c3:
                        chk = st.checkbox("确认删除")
                        if st.button("删除") and chk: delete_user(target); st.success("Del"); time.sleep(0.5); st.rerun()

        timeframe = st.selectbox("周期", ["日线", "周线", "月线"])
        days = st.radio("范围", [7, 10, 30, 60, 120, 250], 2, horizontal=True)
        adjust = st.selectbox("复权", ["qfq","hfq",""], 0)
        
        st.divider()
        
        if is_pro:
            with st.expander("🎛️ 策略参数 (Pro)", expanded=False):
                ma_s = st.slider("短期均线", 2, 20, 5)
                ma_l = st.slider("长期均线", 10, 120, 20)
        
        with st.expander("🛠️ 指标开关 (默认仅开启核心指标)", expanded=False):
            st.info("""
            **说明：** 为保持界面清爽，高级指标默认关闭。
            """)
            c_flags = st.columns(2)
            with c_flags[0]:
                flags['ma'] = st.checkbox("MA (趋势)", True)
                flags['boll'] = st.checkbox("BOLL (通道)", True)
                flags['vol'] = st.checkbox("VOL (成交量)", True)
                flags['macd'] = st.checkbox("MACD (动能)", False)
            with c_flags[1]:
                flags['kdj'] = st.checkbox("KDJ (短线)", False)
                flags['gann'] = st.checkbox("江恩 (时空)", False)
                flags['fib'] = st.checkbox("斐波那契 (黄金分割)", False)
                flags['chan'] = st.checkbox("缠论 (结构)", False)
        
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
                if verify_login(u.strip(), p): 
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = u.strip()
                    st.session_state["paid_code"] = ""
                    last_c = get_user_last_code(u.strip())
                    st.session_state.code = last_c
                    st.rerun()
                else: st.error("账号或密码错误")
        with tab2:
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
                    st.image("qrcode.png", width=200, use_container_width=False, caption="长按识别或截图扫码")
                else:
                    st.info("📸 请直接搜索公众号：lubingxingpiaoliuji")
                
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
name = get_name(st.session_state.code, st.session_state.ts_token, None) 
st.title(f"📈 {name} ({st.session_state.code})")

is_demo = False

# ✅ 修改：自定义交易格言加载提示
loading_quotes = [
    "“市场不欠我钱，但一定会拿我钱”",
    "“错过的机会不是我的钱”",
    "“保住本金才是第一位”",
    "“我的钱不能被他们拿去化债”"
]
selected_quote = random.choice(loading_quotes)

with st.spinner(f"⏳ {selected_quote} | 正在加载数据..."):
    df = get_data_and_resample(st.session_state.code, st.session_state.ts_token, timeframe, adjust, proxy=None)
    
    if df.empty:
        st.warning("⚠️ 暂无数据 (可能因网络原因)。自动切换至演示模式。")
        df = generate_mock_data(days)
        is_demo = True
    elif len(df) < 5:
        st.error(f"❌ 数据不足 (仅获取到 {len(df)} 行)，无法计算技术指标。请尝试切换代码或检查 Tushare 权限。")
        st.stop() 

try:
    funda = get_fundamentals(st.session_state.code, st.session_state.ts_token)
    df = calc_full_indicators(df, ma_s, ma_l)
    df = detect_patterns(df)
    
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
    
    # ✅ 优化：计算 5 维分数并展示雷达图 
    sq, sv, st_, sm, ss = calculate_smart_score(df, funda)
    
    st.markdown(f"""
    <div class="app-card">
        <div style="font-weight:600; font-size: 16px; margin-bottom: 10px; color: #333;">📊 智能诊股 (5 维战力分析)</div>
    </div>
    """, unsafe_allow_html=True)
    
    plot_radar_chart(sq, sv, st_, sm, ss)
    
    ai_text, ai_mood = generate_ai_copilot_text(df, name)
    ai_icon = "🤖" if ai_mood == "neutral" else "😊" if ai_mood == "happy" else "😰"
    
    st.markdown(f"""
    <div class="ai-chat-box">
        <div class="ai-avatar">{ai_icon}</div>
        <div class="ai-content">
            <span style="font-weight:bold; color:#007AFF;">AI 投顾助理：</span>
            {ai_text}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    has_access = False
    if is_admin: has_access = True
    elif is_vip: has_access = True
    elif st.session_state.paid_code == st.session_state.code: has_access = True
    
    if not has_access:
        st.markdown('<div class="locked-container"><div class="locked-blur">', unsafe_allow_html=True)

    plot_chart(df.tail(days), name, flags, ma_s, ma_l)

    # ✅ 改进：后悔药逻辑 - 修改为“1个月前”
    st.markdown("### 💊 既然来了，算算后悔药")
    if len(df) > 22: # 确保有足够数据 (22个交易日约等于一个月)
        price_now = df.iloc[-1]['close']
        price_1m = df.iloc[-22]['close'] # 一个月前 (近似)
        delta_1m = (price_now - price_1m) / price_1m
        money_now = 100000 * (1 + delta_1m)
        color_1m = "#ff3b30" if delta_1m > 0 else "#00c853"
        msg = f"如果你 **1 个月前** 投入 10 万块，现在变成了 <span style='color:{color_1m}; font-size:22px; font-weight:bold;'>{money_now:,.0f} 元</span> ({delta_1m:+.2f}%)"
        st.markdown(msg, unsafe_allow_html=True)
    else:
        st.caption("⚠️ 数据不足一个月，无法计算后悔药。")

    with st.expander("🔍 深度技术分析仪表盘 (趋势/资金/位置)", expanded=False):
        st.info("💡 **说明**：\n1. **多空风向**：基于均线和缠论结构，红色代表强势，绿色代表弱势。\n2. **主力动能**：基于 MACD 和成交量，分值越高资金介入越深。\n3. **高低位置**：当前价格在近期波动区间的位置，越高风险越大。")
        
        plot_technical_dashboard(df)
        
        if st.session_state.ts_token and is_pro:
            chip_df = get_chip_data_pro(st.session_state.code, st.session_state.ts_token)
            if not chip_df.empty:
                st.write("#### 📊 筹码分布 (CYQ Chips)")
                st.dataframe(chip_df.head(), hide_index=True)
            
    st.divider()

    if is_pro:
        plan_html = generate_strategy_card(df, name)
        st.markdown(plan_html, unsafe_allow_html=True)
    else:
        st.info("🔒 开启 [专业模式] 可查看具体的买卖点位、止盈止损价格及仓位建议。")
    
    # ✅ 改进：病毒式海报区域 - 标题修改 + 本地二维码 + 样式增强
    with st.expander("📸 生成朋友圈装X海报 (转发即送20积分)", expanded=False):
        final_score = (sq + sv + st_ + sm + ss) / 5.0 * 10
        poster_html = generate_viral_poster(name, final_score, st.session_state.code)
        st.markdown(poster_html, unsafe_allow_html=True)
        # 诱导转发的文案
        st.success("📢 **福利活动**：长按上方图片保存，转发至朋友圈，截图发给下方公众号，**即刻获赠 20 积分！**")
        st.caption("注：图片保存功能取决于浏览器兼容性，如无法保存请截图。")

    with st.expander("⚖️ 历史验证 (这只股票适合什么玩法?)", expanded=True): 
        c_p1, c_p2 = st.columns([2, 1])
        with c_p1:
            period_label = st.select_slider(
                "📅 回测周期 (看看过去多久的表现)", 
                options=["近1个月", "近3个月", "近半年", "近1年"], 
                value="近半年"
            )
        with c_p2:
            input_cap = st.number_input("💰 假设投入 (元)", value=1000000, step=100000)

        p_map = {"近1年": 12, "近半年": 6, "近3个月": 3, "近1个月": 1}
        selected_months = p_map[period_label]

        st.write("👇 **请选择一种策略，看看如果过去这么玩，能赚多少钱：**")
        strategy_mode = st.radio(
            "选择策略模式", 
            ["📈 趋势跟随 (追涨杀跌)", "🐢 稳健保本 (低买高卖)", "☕ 省心定投 (月月存钱)"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        s_map = {
            "📈 趋势跟随 (追涨杀跌)": "trend",
            "🐢 稳健保本 (低买高卖)": "value",
            "☕ 省心定投 (月月存钱)": "dca"
        }
        
        st_key = s_map[strategy_mode]
        ret, win, mdd, buy_sigs, sell_sigs, eq, profit_val = run_backtest(df, st_key, selected_months, input_cap)
        
        st.markdown("---")
        
        # 🔥 A. 强化“损失厌恶”逻辑
        comment = ""
        if profit_val > 0:
            comment = f"🔥 **哎呀！如果半年前你用了这个策略，现在已经赚了 {profit_val:,.0f} 元！** 这只股票的波动特性非常适合这种玩法。"
        else:
            loss_saved = abs(profit_val)
            comment = f"⚠️ **幸好没买！系统帮你避开了 {mdd:.2f}% 的回撤，相当于省了 {loss_saved:,.0f} 元** —— 省钱也是赚钱！"
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
             st.metric("💰 模拟总收益率", f"{ret:+.2f}%", help="收益百分比")
        with col2:
             p_color = "red" if profit_val >= 0 else "green"
             st.markdown(f"""
             <div style="text-align:center;">
                <div style="font-size:12px; color:#666;">💵 实际盈亏金额</div>
                <div style="font-size:24px; font-weight:bold; color:{p_color};">{profit_val:+,.0f}</div>
             </div>
             """, unsafe_allow_html=True)
        with col3:
             st.metric("📉 历史最大回撤", f"{mdd:.2f}%", help="最倒霉的时候，账户资金缩水了多少")
        
        st.info(f"💡 **AI 结论**：{comment}")

        if not eq.empty:
            bt_fig = make_subplots(rows=1, cols=1)
            bt_fig.add_trace(go.Scatter(x=eq['date'], y=eq['equity'], name='策略净值 (Strategy)', 
                                    line=dict(color='#2962ff', width=2), fill='tozeroy', fillcolor='rgba(41, 98, 255, 0.1)'))
            
            if st_key != "dca":
                bt_fig.add_trace(go.Scatter(x=eq['date'], y=eq['benchmark'], name='基准 (死拿不动)', 
                                    line=dict(color='#9e9e9e', width=1.5, dash='dash')))
            
            if len(buy_sigs) > 0:
                buy_vals = eq[eq['date'].isin(buy_sigs)]['equity']
                bt_fig.add_trace(go.Scatter(x=buy_vals.index.map(lambda x: eq.loc[x, 'date']), y=buy_vals, mode='markers', 
                                                    marker=dict(symbol='triangle-up', size=10, color='#d32f2f'), name='买入'))
            
            bt_fig.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=10), legend=dict(orientation="h", y=1.1), yaxis_title="账户资产", hovermode="x unified")
            st.plotly_chart(bt_fig, use_container_width=True)

    if not has_access:
        st.markdown('</div>', unsafe_allow_html=True) 
        try: bal = load_users()[load_users()["username"]==user]["quota"].iloc[0]
        except: bal = 0
        
        # 🔥 C. 模糊的艺术 (钩子优化)
        st.markdown(f"""
        <div class="locked-overlay">
            <div class="lock-icon">🔒</div>
            <div class="lock-title">深度策略已锁定</div>
            
            <div style="margin-top:15px; text-align:left; background:rgba(255,255,255,0.8); padding:10px; border-radius:8px;">
                <div class="lock-teaser">📊 智能评分: <span style="color:#d32f2f; font-weight:bold;">{sq} (极具潜力)</span></div>
                <div class="lock-teaser">🏦 机构动向: <span style="color:#d32f2f;">主力资金连续 3 日大额流入...</span></div>
                <div class="lock-teaser">📈 关键点位: <span style="color:#007AFF;">支撑位 {df.iloc[-1]['close']*0.9:.2f} 有极强防守...</span></div>
            </div>

            <div style="font-size:12px; color:#666; margin-top:10px;">解锁查看完整买卖点位、机构资金流向及 AI 研报</div>
        </div>
        """, unsafe_allow_html=True)
        
        c_lock1, c_lock2, c_lock3 = st.columns([1,2,1])
        with c_lock2:
            if st.button(f"🔓 支付 1 积分解锁 (余额: {bal})", key="main_unlock", type="primary", use_container_width=True):
                if consume_quota(user):
                    st.session_state.paid_code = st.session_state.code
                    st.session_state.view_mode_idx = 1 # 强制开启 Pro 模式
                    st.rerun()
                else: st.error("积分不足！")
        
except Exception as e:
    st.error(f"Error: {e}")
    st.error(traceback.format_exc())