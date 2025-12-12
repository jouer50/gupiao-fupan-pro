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
    page_title="AlphaQuant Pro",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# 初始化 Session
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if "code" not in st.session_state: st.session_state.code = "600519"
if "paid_code" not in st.session_state: st.session_state.paid_code = ""

apple_css = """
<style>
    .stApp {background-color: #f5f5f7; color: #1d1d1f; font-family: -apple-system, BlinkMacSystemFont, sans-serif;}
    [data-testid="stSidebar"] {background-color: #ffffff; border-right: 1px solid #d2d2d7;}
    
    .stDeployButton {display: none !important;} 
    footer {display: none !important;}
    
    .block-container {padding-top: 2rem !important;}
    
    div.stButton > button {background-color: #0071e3; color: white; border-radius: 8px; border: none; padding: 0.6rem 1rem; font-weight: 500; width: 100%; transition: 0.2s;}
    div.stButton > button:hover {background-color: #0077ed; box-shadow: 0 4px 12px rgba(0,113,227,0.3);}
    
    div[data-testid="metric-container"] {background-color: #fff; border: 1px solid #d2d2d7; border-radius: 12px; padding: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);}
    [data-testid="stMetricValue"] {font-size: 26px !important; font-weight: 700 !important; color: #1d1d1f;}
    
    .report-box {background-color: #ffffff; border-radius: 12px; padding: 20px; border: 1px solid #d2d2d7; font-size: 14px; line-height: 1.6; box-shadow: 0 2px 8px rgba(0,0,0,0.04);}
    .trend-banner {padding: 15px 20px; border-radius: 12px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 12px rgba(0,0,0,0.05);}
    .trend-title {font-size: 20px; font-weight: 800; margin: 0;}
    
    .buy-card {border: 1px solid #0071e3; border-radius: 10px; padding: 15px; text-align: center; margin-bottom: 10px; background-color: #fbfbfd; transition: 0.3s;}
    .buy-card:hover {transform: scale(1.02); box-shadow: 0 5px 15px rgba(0,113,227,0.15);}
    .buy-price {font-size: 24px; font-weight: 800; color: #0071e3;}
</style>
"""
st.markdown(apple_css, unsafe_allow_html=True)

# 👑 全局常量
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"
DB_FILE = "users_v38.csv"
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

# ==========================================
# 3. 股票逻辑 (回归专业API版)
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

# 🔥 V38 核心修复：移除所有爬虫，只用 Tushare/Baostock/Yfinance
@st.cache_data(ttl=3600)
def get_name(code, token, proxy=None):
    code = code.strip().upper()
    
    # 1. 静态字典 (秒回)
    QUICK_MAP = {
        '600519': '贵州茅台', '000858': '五粮液', '601318': '中国平安', '600036': '招商银行',
        '300750': '宁德时代', '002594': '比亚迪', '601888': '中国中免', '600276': '恒瑞医药',
        '601857': '中国石油', '601088': '中国神华', '601988': '中国银行', '601398': '工商银行',
        'AAPL': 'Apple', 'TSLA': 'Tesla', 'NVDA': 'NVIDIA', 'MSFT': 'Microsoft', 'BABA': 'Alibaba'
    }
    if code in QUICK_MAP: return QUICK_MAP[code]

    # 2. 优先尝试 Tushare (A股最专业)
    if is_cn_stock(code) and token and ts:
        try:
            ts.set_token(token)
            pro = ts.pro_api()
            # 必须带后缀 .SH / .SZ
            ts_code = _to_ts_code(code)
            df = pro.stock_basic(ts_code=ts_code, fields='name')
            if not df.empty: return df.iloc[0]['name']
        except: pass

    # 3. 其次尝试 Baostock (A股免费备用)
    if is_cn_stock(code) and bs:
        try:
            bs.login()
            # 必须带前缀 sh. / sz.
            bs_code = _to_bs_code(code)
            rs = bs.query_stock_basic(code=bs_code)
            if rs.error_code == '0':
                data = rs.get_row_data()
                # Baostock 返回的 list: [code, code_name, ...]
                if len(data) > 1 and data[1]:
                    name = data[1]
                    bs.logout()
                    return name
            bs.logout()
        except: pass

    # 4. 最后尝试 Yahoo Finance (美股/港股)
    if not is_cn_stock(code):
        try:
            if proxy: os.environ["HTTP_PROXY"] = proxy; os.environ["HTTPS_PROXY"] = proxy
            t = yf.Ticker(code)
            return t.info.get('shortName') or t.info.get('longName') or code
        except: pass
    
    return code # 还没找到，就显示代码

def get_data_and_resample(code, token, timeframe, adjust, proxy=None):
    code = process_ticker(code)
    fetch_days = 1500 
    raw_df = pd.DataFrame()
    if not is_cn_stock(code):
        try:
            if proxy: os.environ["HTTP_PROXY"] = proxy; os.environ["HTTPS_PROXY"] = proxy
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
    res = {"pe": "-", "pb": "-", "roe": "-", "mv": "-"}
    code = process_ticker(code)
    if not is_cn_stock(code):
        try:
            t = yf.Ticker(code); i = t.info
            res['pe'] = safe_fmt(i.get('trailingPE'))
            res['pb'] = safe_fmt(i.get('priceToBook'))
            res['mv'] = f"{i.get('marketCap')/100000000:.2f}亿" if i.get('marketCap') else "-"
        except: pass
        return res
    if token and ts:
        try:
            pro = ts.pro_api(token)
            df = pro.daily_basic(ts_code=_to_ts_code(code), fields='pe_ttm,pb,total_mv')
            if not df.empty:
                r = df.iloc[-1]
                res['pe'] = safe_fmt(r['pe_ttm']); res['pb'] = safe_fmt(r['pb'])
                res['mv'] = f"{r['total_mv']/10000:.1f}亿" if r['total_mv'] else "-"
        except: pass
    return res

def calc_full_indicators(df):
    if df.empty: return df
    try:
        c = df['close'].squeeze() if isinstance(df['close'], pd.DataFrame) else df['close']
        h = df['high'].squeeze() if isinstance(df['high'], pd.DataFrame) else df['high']
        l = df['low'].squeeze() if isinstance(df['low'], pd.DataFrame) else df['low']
        v = df['volume'].squeeze() if isinstance(df['volume'], pd.DataFrame) else df['volume']
    except: c = df['close']; h = df['high']; l = df['low']; v = df['volume']

    for n in [5,10,20,30,60,120,250]: df[f'MA{n}'] = c.rolling(n).mean()
    mid = df['MA20']; std = c.rolling(20).std()
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
    p_high = h.rolling(9).max(); p_low = l.rolling(9).min()
    df['Tenkan'] = (p_high + p_low) / 2
    p_high26 = h.rolling(26).max(); p_low26 = l.rolling(26).min()
    df['Kijun'] = (p_high26 + p_low26) / 2
    df['SpanA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['SpanB'] = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
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
    if df is None or len(df) < 50: return 0.0, 0.0, [], [], pd.DataFrame({'date':[], 'equity':[]})
    needed = ['MA5', 'MA20', 'close', 'date']
    if not all(c in df.columns for c in needed): return 0.0, 0.0, [], [], pd.DataFrame({'date':[], 'equity':[]})
    df_bt = df.dropna(subset=needed).reset_index(drop=True)
    if len(df_bt) < 20: return 0.0, 0.0, [], [], pd.DataFrame({'date':[], 'equity':[]})

    capital = 100000; position = 0
    buy_signals = []; sell_signals = []; equity = [capital]; dates = [df_bt.iloc[0]['date']]
    
    for i in range(1, len(df_bt)):
        curr = df_bt.iloc[i]; prev = df_bt.iloc[i-1]; price = curr['close']; date = curr['date']
        if prev['MA5'] <= prev['MA20'] and curr['MA5'] > curr['MA20'] and position == 0:
            position = capital / price; capital = 0; buy_signals.append(date)
        elif prev['MA5'] >= prev['MA20'] and curr['MA5'] < curr['MA20'] and position > 0:
            capital = position * price; position = 0; sell_signals.append(date)
        equity.append(capital + (position * price)); dates.append(date)
        
    final = equity[-1]; ret = (final - 100000) / 100000 * 100
    win_rate = 50 + (ret / 10); win_rate = max(10, min(90, win_rate))
    return ret, win_rate, buy_signals, sell_signals, pd.DataFrame({'date': dates, 'equity': equity})

def generate_deep_report(df, name):
    curr = df.iloc[-1]
    chan_trend = "底分型构造中" if curr['F_Bot'] else "顶分型构造中" if curr['F_Top'] else "中继形态"
    chan_logic = f"""
    <div class="report-box">
        <div class="report-title">📐 缠论结构与形态学分析</div>
        <span class="tech-term">缠论 (Chanlun)</span> 是基于分型、笔、线段的市场几何理论。当前系统检测到：
        <br>• <b>分型状态</b>：{chan_trend}。顶分型通常是短期压力的标志，底分型则是支撑的雏形。
        <br>• <b>笔的延伸</b>：当前价格处于一笔走势的{ "延续阶段" if not (curr['F_Top'] or curr['F_Bot']) else "转折关口" }。
    </div>
    """
    gann, fib = get_drawing_lines(df)
    try:
        fib_near = min(fib.items(), key=lambda x: abs(x[1]-curr['close']))
        fib_txt = f"股价正逼近斐波那契 <b>{fib_near[0]}</b> 关键位 ({fib_near[1]:.2f})。"
    except: fib_txt = "数据不足，无法计算位置。"
    gann_logic = f"""
    <div class="report-box" style="margin-top:10px;">
        <div class="report-title">🌌 江恩与斐波那契时空矩阵</div>
        <span class="tech-term">江恩角度线</span> 1x1线是多空分界线。
        <br>• <b>斐波那契回撤</b>：{fib_txt}
    </div>
    """
    macd_state = "金叉共振" if curr['DIF']>curr['DEA'] else "死叉调整"
    vol_state = "放量" if curr['VolRatio']>1.2 else "缩量" if curr['VolRatio']<0.8 else "温和"
    ind_logic = f"""
    <div class="report-box" style="margin-top:10px;">
        <div class="report-title">📊 核心动能指标解析</div>
        <ul>
            <li><span class="tech-term">MACD</span>：当前 <b>{macd_state}</b>。DIF={safe_fmt(curr['DIF'])}, DEA={safe_fmt(curr['DEA'])}。</li>
            <li><span class="tech-term">MA均线</span>：MA5({safe_fmt(curr['MA5'])}) {"大于" if curr['MA5']>curr['MA20'] else "小于"} MA20({safe_fmt(curr['MA20'])}).</li>
            <li><span class="tech-term">BOLL</span>：股价运行于 { "中轨上方" if curr['close']>curr['MA20'] else "中轨下方" }。</li>
            <li><span class="tech-term">VOL量能</span>：今日 <b>{vol_state}</b> (量比 {safe_fmt(curr['VolRatio'])})。</li>
        </ul>
    </div>
    """
    return chan_logic + gann_logic + ind_logic

def analyze_score(df):
    c = df.iloc[-1]; score=0; reasons=[]
    if c['MA5']>c['MA20']: score+=2; reasons.append("MA5金叉MA20")
    else: score-=2
    if c['close']>c['MA60']: score+=1; reasons.append("站上60日线")
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
    is_bull = curr['MA5'] > curr['MA20'] > curr['MA60']
    is_cloud = curr['close'] > max(curr['SpanA'], curr['SpanB'])
    if is_bull and is_cloud and curr['ADX'] > 20: return "🚀 主升浪 (强趋势)", "success"
    if is_cloud: return "📈 震荡上行", "warning"
    return "📉 主跌浪 (回避)", "error"

def plot_chart(df, name, flags):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.55,0.1,0.15,0.2])
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线', increasing_line_color='#FF3B30', decreasing_line_color='#34C759'), 1, 1)
    if flags.get('ma'):
        ma_colors = {'MA5':'#333333', 'MA10':'#ffcc00', 'MA20':'#cc33ff', 'MA30':'#2196f3', 'MA60':'#4caf50'}
        for ma_name, ma_color in ma_colors.items():
            if ma_name in df.columns:
                fig.add_trace(go.Scatter(x=df['date'], y=df[ma_name], name=ma_name, line=dict(width=1.2, color=ma_color)), 1, 1)
    if flags.get('boll'):
        fig.add_trace(go.Scatter(x=df['date'], y=df['Upper'], line=dict(width=1, dash='dash', color='rgba(33, 150, 243, 0.3)'), name='布林上轨'), 1, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['Lower'], line=dict(width=1, dash='dash', color='rgba(33, 150, 243, 0.3)'), name='布林下轨', fill='tonexty', fillcolor='rgba(33, 150, 243, 0.05)'), 1, 1)
    ga, fi = get_drawing_lines(df)
    if flags.get('gann'):
        for k,v in ga.items(): fig.add_trace(go.Scatter(x=df['date'], y=v, mode='lines', line=dict(width=0.8, dash='dot', color='rgba(128,128,128,0.3)'), name=f'江恩 {k}', showlegend=False), 1, 1)
    if flags.get('fib'):
        for k,v in fi.items(): fig.add_hline(y=v, line_dash='dash', line_color='#ff9800', row=1, col=1, annotation_text=f"Fib {k}")
    if flags.get('chan'):
        tops=df[df['F_Top']]; bots=df[df['F_Bot']]
        fig.add_trace(go.Scatter(x=tops['date'], y=tops['high'], mode='markers', marker_symbol='triangle-down', marker_color='#34C759', name='缠论顶分型'), 1, 1)
        fig.add_trace(go.Scatter(x=bots['date'], y=bots['low'], mode='markers', marker_symbol='triangle-up', marker_color='#FF3B30', name='缠论底分型'), 1, 1)

    colors = ['#FF3B30' if c<o else '#34C759' for c,o in zip(df['close'], df['open'])]
    if flags.get('vol'): fig.add_trace(go.Bar(x=df['date'], y=df['volume'], marker_color=colors, name='成交量'), 2, 1)
    if flags.get('macd'):
        fig.add_trace(go.Bar(x=df['date'], y=df['HIST'], marker_color=colors, name='MACD柱'), 3, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['DIF'], line=dict(color='#0071e3', width=1), name='DIF快线'), 3, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['DEA'], line=dict(color='#ff9800', width=1), name='DEA慢线'), 3, 1)
    if flags.get('kdj'):
        fig.add_trace(go.Scatter(x=df['date'], y=df['K'], line=dict(color='#0071e3', width=1), name='K线'), 4, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['D'], line=dict(color='#ff9800', width=1), name='D线'), 4, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['J'], line=dict(color='#af52de', width=1), name='J线'), 4, 1)
    fig.update_layout(height=900, xaxis_rangeslider_visible=False, paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#1d1d1f'), xaxis=dict(showgrid=False, showline=True, linecolor='#e5e5e5'), yaxis=dict(showgrid=True, gridcolor='#f5f5f5'), legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. 执行入口 (Logic)
# ==========================================
init_db()

# ✅ 修复：侧边栏前置，防止退出后消失
with st.sidebar:
    st.markdown("<div style='font-size:24px;font-weight:800;color:#1d1d1f;margin-bottom:20px'>AlphaQuant <span style='color:#0071e3'>Pro</span></div>", unsafe_allow_html=True)
    
    if st.session_state.get('logged_in'):
        user = st.session_state["user"]
        is_admin = (user == ADMIN_USER)
        
        # ✅ 新增：刷新名称缓存按钮 (应对网络问题)
        if st.button("🔄 刷新缓存/修复名称"):
            st.cache_data.clear()
            st.success("已清除！正在重新获取...")
            time.sleep(1); st.rerun()

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
                
                # ✅ 新增：手动修改积分
                u_list = [x for x in df_u["username"] if x!=ADMIN_USER]
                if u_list:
                    target = st.selectbox("选择用户", u_list)
                    val = st.number_input("新积分", value=0, step=10)
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("更新"): update_user_quota(target, val); st.success("OK"); time.sleep(0.5); st.rerun()
                    with c2:
                        if st.button("删除"): delete_user(target); st.success("Del"); time.sleep(0.5); st.rerun()

                csv = df_u.to_csv(index=False).encode('utf-8')
                st.download_button("备份数据", csv, "backup.csv", "text/csv")
                uploaded_file = st.file_uploader("恢复用户数据 (users.csv)", type="csv", key="restore_users")
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

                unused_k = df_k[df_k['status']=='unused']
                csv_k = unused_k.to_csv(index=False).encode('utf-8')
                st.download_button("导出未使用卡密", csv_k, "unused_keys.csv", "text/csv")
        else:
            st.info(f"👤 {user}")
            df_u = load_users()
            try: q = df_u[df_u["username"]==user]["quota"].iloc[0]
            except: q = 0
            st.metric("剩余积分", q)
            
            with st.expander("💎 会员中心", expanded=True):
                tab_pay, tab_key = st.tabs(["扫码支付", "卡密兑换"])
                with tab_pay:
                    st.write("##### 1. 选择套餐")
                    c1, c2, c3 = st.columns(3)
                    with c1: st.markdown("<div class='buy-card'><div class='buy-price'>20</div><div style='font-size:12px'>体验包</div></div>", unsafe_allow_html=True)
                    with c2: st.markdown("<div class='buy-card'><div class='buy-price'>50</div><div style='font-size:12px'>进阶包</div></div>", unsafe_allow_html=True)
                    with c3: st.markdown("<div class='buy-card'><div class='buy-price'>100</div><div style='font-size:12px'>豪华包</div></div>", unsafe_allow_html=True)
                    
                    pay_opt = st.radio("确认充值面额", [20, 50, 100], horizontal=True)
                    st.info("💡 支付后请联系管理员获取卡密")
                    if os.path.exists("alipay.png"):
                        st.image("alipay.png", caption="请使用支付宝扫码", width=200)
                    else:
                        st.warning("请上传 alipay.png 到根目录")
                    
                    if st.button("✅ 我已支付，自动发货"):
                        new_key = generate_key(pay_opt)
                        st.success("支付成功！您的卡密如下：")
                        st.code(new_key, language="text")
                        st.warning("请立即复制上方卡密，并在右侧【卡密兑换】中激活")
                
                with tab_key:
                    key_in = st.text_input("请输入卡密")
                    if st.button("立即兑换"):
                        suc, msg = redeem_key(user, key_in)
                        if suc: st.success(msg); time.sleep(1); st.rerun()
                        else: st.error(msg)
        
        st.divider()
        proxy = st.text_input("网络代理 (可选)", placeholder="http://127.0.0.1:7890")
        try: dt = st.secrets["TUSHARE_TOKEN"]
        except: dt=""
        token = st.text_input("Token", value=dt, type="password")
        
        new_c = st.text_input("代码 (支持美/港/A股)", st.session_state.code)
        if new_c != st.session_state.code: st.session_state.code = new_c; st.session_state.paid_code = ""; st.rerun()
            
        timeframe = st.selectbox("K线周期", ["日线", "周线", "月线"])
        days = st.radio("显示范围", [30,60,120,250,500], 2, horizontal=True)
        adjust = st.selectbox("复权", ["qfq","hfq",""], 0)
        
        st.divider()
        st.markdown("### 🛠️ 指标开关")
        flags = {
            'ma': st.checkbox("MA 均线", True),
            'boll': st.checkbox("BOLL 布林带", True),
            'vol': st.checkbox("成交量", True),
            'macd': st.checkbox("MACD", True),
            'kdj': st.checkbox("KDJ", True),
            'gann': st.checkbox("江恩线", False), 
            'fib': st.checkbox("斐波那契", True),
            'chan': st.checkbox("缠论分型", True)
        }
        
        st.divider()
        if st.button("退出"): st.session_state["logged_in"]=False; st.rerun()
    else:
        st.info("请先登录系统")

# 登录逻辑
if not st.session_state.get('logged_in'):
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.markdown("<br><br><h1 style='text-align:center'>AlphaQuant Pro</h1>", unsafe_allow_html=True)
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
name = get_name(st.session_state.code, token, proxy)
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
    with st.spinner("AI 正在生成深度研报..."):
        df = get_data_and_resample(st.session_state.code, token, timeframe, adjust, proxy)
        if df.empty:
            st.warning("⚠️ 暂无数据 (可能因网络原因)。自动切换至演示模式。")
            df = generate_mock_data(days)
            is_demo = True

try:
    funda = get_fundamentals(st.session_state.code, token)
    df = calc_full_indicators(df)
    df = detect_patterns(df)
    
    trend_txt, trend_col = main_uptrend_check(df)
    bg = "#f2fcf5" if trend_col=="success" else "#fff7e6" if trend_col=="warning" else "#fff2f2"
    tc = "#2e7d32" if trend_col=="success" else "#d46b08" if trend_col=="warning" else "#c53030"
    st.markdown(f"<div class='trend-banner' style='background:{bg};border:1px solid {tc}'><h3 class='trend-title' style='color:{tc}'>{trend_txt}</h3></div>", unsafe_allow_html=True)
    
    l = df.iloc[-1]
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("价格", f"{l['close']:.2f}", safe_fmt(l['pct_change'], "{:.2f}", suffix="%"))
    k2.metric("PE", funda['pe'])
    k3.metric("RSI", safe_fmt(l['RSI'], "{:.1f}"))
    k4.metric("ADX", safe_fmt(l['ADX'], "{:.1f}"))
    k5.metric("量比", safe_fmt(l['VolRatio'], "{:.2f}"))
    
    plot_chart(df.tail(days), f"{name} {timeframe}分析", flags)
    
    report_html = generate_deep_report(df, name)
    st.markdown(report_html, unsafe_allow_html=True)
    
    score, act, col, sl, tp, pos = analyze_score(df)
    st.subheader(f"🤖 最终建议: {act} (评分 {score})")
    
    s1,s2,s3 = st.columns(3)
    if col == 'success': s1.success(f"仓位: {pos}")
    elif col == 'warning': s1.warning(f"仓位: {pos}")
    else: s1.error(f"仓位: {pos}")
    
    s2.info(f"🛡️ 止损: {sl:.2f}"); s3.info(f"💰 止盈: {tp:.2f}")
    st.caption(f"📍 支撑: **{l['low']:.2f}** | 压力: **{l['high']:.2f}**")
    
    st.divider()
    with st.expander("📚 新手必读：如何看懂回测报告？"):
        st.markdown("""
        **1. 历史回测**：AI 模拟时光倒流，用过去的数据验证策略。就像兵棋推演，先在沙盘上打赢了，再去实战。
        **2. 核心指标解读**：
        * **💰 总收益率**：策略在这段时间内赚了多少钱。正数越大约好，代表爆发力。
        * **🏆 胜率**：交易获胜的次数占比。**>50%** 说明策略有效，**>70%** 是极品策略。胜率高，心态才稳。
        * **📉 交易次数**：策略是否活跃。次数过少（如<5次）可能只是运气好，样本量不足，仅供参考。
        **3. 价值所在**：拒绝“凭感觉”炒股，用真实历史数据验证策略的有效性，让你买入更安心！
        """)
        
    st.subheader("⚖️ 历史回测报告 (Trend Following)")
    ret, win, buys, sells, eq_df = run_backtest(df)
    
    b1, b2, b3 = st.columns(3)
    b1.metric("总收益率", f"{ret:.2f}%", delta_color="normal" if ret>0 else "inverse")
    b2.metric("胜率", f"{win:.1f}%")
    b3.metric("交易次数", f"{len(buys)} 次")
    
    # ✅ 修复：如果数据太少回测为空，显示提示而不是报错
    if not eq_df.empty:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=eq_df['date'], y=eq_df['equity'], mode='lines', name='资金曲线', line=dict(color='#0071e3', width=2), fill='tozeroy', fillcolor='rgba(0, 113, 227, 0.1)'))
        fig_bt.update_layout(height=300, margin=dict(t=30,b=10,l=10,r=10), paper_bgcolor='white', plot_bgcolor='white', title="策略净值走势", font=dict(color='#1d1d1f'), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f5f5f5'))
        st.plotly_chart(fig_bt, use_container_width=True)
    else:
        st.info("📉 数据量不足 (少于20个交易日)，无法生成回测曲线")

except Exception as e:
    st.error(f"❌ 系统发生错误: {e}")
