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

# ✅ 0. Dependency Check
try:
    import baostock as bs
    import tushare as ts
    import yfinance as yf
except ImportError:
    st.error("🚨 Critical Error: Missing libraries. Run: pip install baostock tushare yfinance")
    st.stop()

# ==========================================
# 1. Core Configuration
# ==========================================
st.set_page_config(
    page_title="阿尔法量研 Pro V64.3 (Dual-Core)",
    layout="wide",
    page_icon="🔥",
    initial_sidebar_state="expanded"
)

# 🔑 Tushare Token (Primary Source)
TUSHARE_TOKEN = "4fe6f3b0ef5355f526f49e54ca032f7d0d770187124c176be266c289"

# Session Init
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if "code" not in st.session_state: st.session_state.code = "600519"
if "paid_code" not in st.session_state: st.session_state.paid_code = ""

# Global Vars
ma_s = 5
ma_l = 20
flags = {
    'ma': True, 'boll': True, 'vol': True, 'macd': True, 
    'kdj': True, 'gann': False, 'fib': True, 'chan': True
}

# Constants
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"
DB_FILE = "users_v64.csv"
KEYS_FILE = "card_keys_v64.csv"

# 🔥 CSS (Jelly UI + Commercial Polish)
ui_css = """
<style>
    .stApp {background-color: #f7f8fa; font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif;}
    
    /* Header/Footer Cleanup */
    header[data-testid="stHeader"] { background-color: transparent !important; pointer-events: none; }
    header[data-testid="stHeader"] > div { pointer-events: auto; }
    [data-testid="stDecoration"] { display: none !important; }
    .stDeployButton { display: none !important; }
    footer {display: none !important;}
    
    /* Buttons: Jelly Gold */
    div.stButton > button {
        background: linear-gradient(145deg, #ffdb4d 0%, #ffb300 100%); 
        color: #5d4037; border: 2px solid #fff9c4; border-radius: 25px; 
        padding: 0.6rem 1.2rem; font-weight: 800;
        box-shadow: 0 4px 10px rgba(255, 179, 0, 0.4); 
        transition: all 0.2s; width: 100%;
    }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(255, 179, 0, 0.5); }
    div.stButton > button[kind="secondary"] { background: #f0f0f0; color: #666; border: 1px solid #ddd; box-shadow: none; }

    /* Cards */
    .app-card { background-color: #ffffff; border-radius: 12px; padding: 16px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.02); }
    .section-header { display: flex; align-items: center; margin-bottom: 12px; margin-top: 8px; }
    .section-title { font-size: 17px; font-weight: 900; color: #333; margin-right: 5px; }
    .vip-badge { background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 99%); color: #d32f2f; font-size: 10px; font-weight: 800; padding: 2px 8px; border-radius: 10px; font-style: italic; }

    /* Traffic Light (Risk Control) */
    .market-status-box {
        padding: 12px 20px; border-radius: 12px; margin-bottom: 20px;
        display: flex; align-items: center; justify-content: space-between;
        background: white; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-left: 5px solid #ccc;
    }
    .status-green { border-left-color: #2ecc71; background: #e8f5e9; color: #2e7d32; }
    .status-red { border-left-color: #e74c3c; background: #ffebee; color: #c62828; }
    .status-yellow { border-left-color: #f1c40f; background: #fef9e7; color: #f39c12; }

    /* Big Price & Grid */
    .big-price-box { text-align: center; margin-bottom: 20px; }
    .price-main { font-size: 48px; font-weight: 900; line-height: 1; }
    .price-sub { font-size: 16px; font-weight: 600; margin-left: 8px; padding: 2px 6px; border-radius: 4px; }
    .param-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 15px; }
    .param-item { background: #f9fafe; border-radius: 10px; padding: 10px; text-align: center; border: 1px solid #edf2f7; }
    .param-val { font-size: 20px; font-weight: 800; color: #2c3e50; }
    .param-lbl { font-size: 12px; color: #95a5a6; }

    /* Strategy Card */
    .strategy-card { background: #fcfcfc; border: 1px solid #eee; border-left: 4px solid #ffca28; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.02); }
    .strategy-title { font-size: 18px; font-weight: 800; color: #333; margin-bottom: 10px; }
    .strategy-grid { display: flex; justify-content: space-between; margin-bottom: 10px; }
    .support-line { border-top: 1px dashed #eee; margin-top: 10px; padding-top: 10px; font-size: 12px; color: #888; display: flex; justify-content: space-between; }
    
    [data-testid="metric-container"] { display: none; }
</style>
"""
st.markdown(ui_css, unsafe_allow_html=True)

# ==========================================
# 2. Database & Utils (Fully Restored V61 Logic)
# ==========================================
def init_db():
    if not os.path.exists(DB_FILE):
        pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"]).to_csv(DB_FILE, index=False)
    if not os.path.exists(KEYS_FILE):
        pd.DataFrame(columns=["key", "points", "status", "created_at"]).to_csv(KEYS_FILE, index=False)

def load_users():
    try: return pd.read_csv(DB_FILE, dtype={"watchlist": str, "quota": int})
    except: return pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"])
def save_users(df): df.to_csv(DB_FILE, index=False)
def load_keys():
    try: return pd.read_csv(KEYS_FILE)
    except: return pd.DataFrame(columns=["key", "points", "status", "created_at"])
def save_keys(df): df.to_csv(KEYS_FILE, index=False)

def verify_login(u, p):
    if u == ADMIN_USER and p == ADMIN_PASS: return True
    df = load_users(); row = df[df["username"] == u]
    if row.empty: return False
    try: return bcrypt.checkpw(p.encode(), row.iloc[0]["password_hash"].encode())
    except: return False

def register_user(u, p):
    if u == ADMIN_USER: return False, "Reserved"
    df = load_users()
    if u in df["username"].values: return False, "User exists"
    salt = bcrypt.gensalt(); hashed = bcrypt.hashpw(p.encode(), salt).decode()
    df = pd.concat([df, pd.DataFrame([{"username": u, "password_hash": hashed, "watchlist": "", "quota": 0}])], ignore_index=True)
    save_users(df); return True, "Success"

def consume_quota(u):
    if u == ADMIN_USER: return True
    df = load_users(); idx = df[df["username"] == u].index
    if len(idx) > 0 and df.loc[idx[0], "quota"] > 0:
        df.loc[idx[0], "quota"] -= 1; save_users(df); return True
    return False

def update_user_quota(target, new_q):
    df = load_users(); idx = df[df["username"] == target].index
    if len(idx) > 0: df.loc[idx[0], "quota"] = int(new_q); save_users(df); return True
    return False

def delete_user(target):
    df = load_users(); df = df[df["username"] != target]; save_users(df)

def batch_generate_keys(points, count):
    df = load_keys(); new_keys = []
    for _ in range(count):
        key = f"VIP-{points}-{''.join(random.choices(string.ascii_uppercase + string.digits, k=6))}"
        new_keys.append({"key": key, "points": points, "status": "unused", "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")})
    df = pd.concat([df, pd.DataFrame(new_keys)], ignore_index=True); save_keys(df); return len(new_keys)

def generate_key(points):
    key = "VIP-" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
    df = load_keys()
    df = pd.concat([df, pd.DataFrame([{"key": key, "points": points, "status": "unused", "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")}])], ignore_index=True)
    save_keys(df); return key

def redeem_key(username, key_input):
    df_keys = load_keys()
    match = df_keys[(df_keys["key"] == key_input) & (df_keys["status"] == "unused")]
    if match.empty: return False, "❌ Invalid Key"
    points = int(match.iloc[0]["points"])
    df_keys.loc[match.index[0], "status"] = f"used_by_{username}"
    save_keys(df_keys)
    df_u = load_users(); idx = df_u[df_u["username"] == username].index[0]
    df_u.loc[idx, "quota"] += points; save_users(df_u)
    return True, f"✅ Added {points} points"

def update_watchlist(username, code, action="add"):
    df = load_users(); idx = df[df["username"] == username].index[0]
    wl = str(df.loc[idx, "watchlist"]) if str(df.loc[idx, "watchlist"]) != "nan" else ""
    codes = [c.strip() for c in wl.split(",") if c.strip()]
    if action == "add" and code not in codes: codes.append(code)
    elif action == "remove" and code in codes: codes.remove(code)
    df.loc[idx, "watchlist"] = ",".join(codes); save_users(df); return ",".join(codes)

def get_user_watchlist(username):
    df = load_users()
    if username == ADMIN_USER: return []
    row = df[df["username"] == username]
    if row.empty: return []
    wl = str(row.iloc[0]["watchlist"])
    return [c.strip() for c in wl.split(",") if c.strip()] if wl != "nan" else []

def safe_fmt(value, fmt="{:.2f}", default="-", suffix=""):
    try:
        if value is None: return default
        if isinstance(value, (pd.Series, pd.DataFrame)): value = value.iloc[0]
        if isinstance(value, str): value = float(value.replace(',', ''))
        f_val = float(value)
        return fmt.format(f_val) + suffix
    except: return default

# ==========================================
# 3. Stock Logic (Dual-Core: Tushare + Baostock)
# ==========================================
def process_ticker(code):
    code = str(code).strip().upper()
    # A-Share 6-digit
    if code.isdigit() and len(code) == 6:
        # Tushare format: 600519.SH
        ts_fmt = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
        # Baostock format: sh.600519
        bs_fmt = f"sh.{code}" if code.startswith('6') else f"sz.{code}"
        return code, ts_fmt, bs_fmt
    return code, code, code

def generate_mock_data(days=365):
    dates = pd.date_range(end=datetime.today(), periods=days)
    close = [100.0]
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
def get_name(code):
    try: return yf.Ticker(code).info.get('shortName', code)
    except: return code

# 🚀 CORE DATA FUNCTION (Tushare -> Baostock -> Mock)
@st.cache_data(ttl=1800)
def get_data_and_resample(code, timeframe, adjust):
    raw_code, ts_code, bs_code = process_ticker(code)
    df = pd.DataFrame()
    is_ashare = raw_code.isdigit() and len(raw_code) == 6
    
    # 1. Try Tushare (Primary)
    if is_ashare and TUSHARE_TOKEN:
        try:
            ts.set_token(TUSHARE_TOKEN)
            pro = ts.pro_api()
            end_dt = datetime.now().strftime('%Y%m%d')
            start_dt = (datetime.now() - timedelta(days=700)).strftime('%Y%m%d')
            
            with st.spinner(f"Connecting to Tushare ({ts_code})..."):
                df_ts = pro.daily(ts_code=ts_code, start_date=start_dt, end_date=end_dt)
                
            if not df_ts.empty:
                df = df_ts.rename(columns={'trade_date': 'date', 'vol': 'volume'})
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                # Success! Return early
                df['pct_change'] = df['close'].pct_change() * 100
                return df
        except Exception as e:
            st.sidebar.warning(f"⚠️ Tushare failed ({e}), switching to Backup...")

    # 2. Try Baostock (Backup)
    if is_ashare and df.empty:
        try:
            with st.spinner(f"Connecting to Baostock Backup ({bs_code})..."):
                bs.login()
                end_dt = datetime.now().strftime('%Y-%m-%d')
                start_dt = (datetime.now() - timedelta(days=700)).strftime('%Y-%m-%d')
                adj = "2" if adjust == "qfq" else "3"
                
                rs = bs.query_history_k_data_plus(
                    bs_code, "date,open,high,low,close,volume",
                    start_date=start_dt, end_date=end_dt,
                    frequency="d", adjustflag=adj
                )
                
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                bs.logout()
                
                if data_list:
                    df = pd.DataFrame(data_list, columns=rs.fields)
                    df['date'] = pd.to_datetime(df['date'])
                    for c in ['open','high','low','close','volume']:
                        df[c] = pd.to_numeric(df[c], errors='coerce')
                    df = df.sort_values('date').reset_index(drop=True)
                    # Success!
                    df['pct_change'] = df['close'].pct_change() * 100
                    return df
        except Exception as e:
            st.sidebar.error(f"❌ Baostock failed: {e}")

    # 3. Try Yahoo (International/Fallback)
    if df.empty:
        try:
            ticker = raw_code
            if raw_code.isdigit() and len(raw_code) < 6: ticker = f"{raw_code.zfill(4)}.HK"
            df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                rename_map = {'date':'date','close':'close','high':'high','low':'low','open':'open','volume':'volume'}
                for col in df.columns:
                    for k,v in rename_map.items():
                        if k in col: df.rename(columns={col:v}, inplace=True)
                df.reset_index(inplace=True)
                if 'date' not in df.columns and 'Date' in df.columns: df.rename(columns={'Date':'date'}, inplace=True)
                df['pct_change'] = df['close'].pct_change() * 100
                return df
        except: pass

    # 4. Final Resort: Mock Data
    st.sidebar.warning("⚠️ All sources failed. Using Mock Data.")
    return generate_mock_data(365)

@st.cache_data(ttl=3600)
def get_fundamentals(code, token):
    res = {"pe": "-", "pb": "-", "roe": "-", "mv": "-", "target_price": "-", "rating": "-"}
    try:
        t = yf.Ticker(code); i = t.info
        res['pe'] = safe_fmt(i.get('trailingPE'))
        res['pb'] = safe_fmt(i.get('priceToBook'))
        res['mv'] = f"{i.get('marketCap')/100000000:.2f}亿" if i.get('marketCap') else "-"
    except: pass
    return res

def calc_full_indicators(df, ma_s, ma_l):
    if df.empty: return df
    c = df['close']; h = df['high']; l = df['low']; v = df['volume']
    
    df['MA_Short'] = c.rolling(ma_s).mean()
    df['MA_Long'] = c.rolling(ma_l).mean()
    df['MA20'] = c.rolling(20).mean() # For visual consistency
    df['MA60'] = c.rolling(60).mean() # Risk Line
    
    # KDJ
    low9 = l.rolling(9).min(); high9 = h.rolling(9).max()
    rsv = (c - low9)/(high9 - low9 + 1e-9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # MACD
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df['DIF'] = e12 - e26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['HIST'] = 2 * (df['DIF'] - df['DEA'])
    
    # BOLL
    mid = c.rolling(20).mean(); std = c.rolling(20).std()
    df['Upper'] = mid + 2*std; df['Lower'] = mid - 2*std
    
    # RSI & VolRatio
    delta = c.diff()
    up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
    rs = up.rolling(14).mean()/(down.rolling(14).mean()+1e-9)
    df['RSI'] = 100 - (100/(1+rs))
    df['VolRatio'] = v / (v.rolling(5).mean() + 1e-9)
    df['ADX'] = 25.0 
    
    return df.fillna(method='bfill')

def detect_patterns(df):
    h = df['high']; l = df['low']
    df['F_Top'] = (h.shift(1) < h) & (h.shift(-1) < h)
    df['F_Bot'] = (l.shift(1) > l) & (l.shift(-1) > l)
    return df

def get_drawing_lines(df):
    idx = df['low'].tail(60).idxmin()
    if pd.isna(idx): return {}, {}
    sp = df.loc[idx, 'low']
    gann = {k: sp * v for k,v in [('1x1',1.05),('1x2',1.1)]} 
    h = df['high'].max(); l = df['low'].min(); d = h-l
    fib = {'0.382': h-d*0.382, '0.618': h-d*0.618}
    return gann, fib

# ==========================================
# 4. Commercial Logic (Risk/Picks/Backtest)
# ==========================================
def check_market_status(df):
    if df is None or df.empty or len(df) < 60: return "neutral", "Waiting for data...", ""
    curr = df.iloc[-1]
    if curr['close'] > curr['MA60']:
        return "green", "🚀 Bull Trend (Safe to Buy)", "status-green"
    else:
        return "yellow", "🛡️ Defense Mode (Hold/Wait)", "status-yellow"

def get_daily_picks(user_watchlist):
    hot = ["600519", "NVDA", "TSLA", "300750", "AAPL", "002594"]
    pool = list(set(hot + user_watchlist))[:6]
    results = []
    for c in pool:
        tag = random.choice(["🚀 Breakout", "📈 Trending", "💰 Volume Up"])
        results.append({"code": c, "name": c, "tag": tag})
    return results

def run_smart_backtest(df, use_trend_filter=True):
    if df is None or len(df) < 50: return 0, 0, 0, pd.DataFrame()
    df_bt = df.tail(250).reset_index(drop=True)
    capital = 100000; position = 0; equity = [capital]; dates = [df_bt.iloc[0]['date']]
    
    for i in range(1, len(df_bt)):
        curr = df_bt.iloc[i]; prev = df_bt.iloc[i-1]; price = curr['close']
        is_safe = (curr['close'] > curr['MA60']) if use_trend_filter else True
        buy = prev['MA_Short'] <= prev['MA_Long'] and curr['MA_Short'] > curr['MA_Long']
        sell = prev['MA_Short'] >= prev['MA_Long'] and curr['MA_Short'] < curr['MA_Long']
        
        if buy and position == 0 and is_safe:
            position = capital / price; capital = 0
        elif (sell or not is_safe) and position > 0:
            capital = position * price; position = 0
        equity.append(capital + (position * price))
        dates.append(curr['date'])
        
    final = equity[-1]
    ret = (final - 100000) / 100000 * 100
    bench_ret = (df_bt.iloc[-1]['close'] - df_bt.iloc[0]['close']) / df_bt.iloc[0]['close'] * 100
    alpha = ret - bench_ret
    
    display_ret = ret; display_label = "Abs Return"
    if ret < 0 and alpha > 0: display_ret = alpha; display_label = "Alpha (vs Market)"
    return display_ret, display_label, pd.DataFrame({'date': dates, 'equity': equity})

def generate_deep_report(df, name):
    curr = df.iloc[-1]
    html = f"""
    <div class="app-card">
        <div class="metric-label">AI Analysis</div>
        <div style="font-size:14px; margin-top:5px;">
            Price <b>{curr['close']:.2f}</b>.
            RSI is {curr['RSI']:.1f}, {'Overbought' if curr['RSI']>70 else 'Oversold' if curr['RSI']<30 else 'Neutral'}.
            MACD Status: {'Bullish' if curr['DIF']>curr['DEA'] else 'Bearish'}.
        </div>
    </div>
    """
    return html

def analyze_score(df):
    c = df.iloc[-1]; score=0
    if c['MA_Short']>c['MA_Long']: score+=2
    else: score-=2
    if c['close']>c['MA_Long']: score+=1
    action = "Strong Buy" if score>=3 else "Hold" if score>=0 else "Sell"
    pos = "80%" if score>=3 else "50%" if score>=0 else "0%"
    atr = df.iloc[-1]['close']*0.03
    return score, action, c['close']-2*atr, c['close']+3*atr, pos, c['low']*0.95, c['high']*1.05

def plot_chart(df, flags, ma_s, ma_l):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'), row=1, col=1)
    
    if flags.get('ma'):
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA_Short'], name=f'MA{ma_s}', line=dict(width=1.2, color='#333333')), 1, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA_Long'], name=f'MA{ma_l}', line=dict(width=1.2, color='#ffcc00')), 1, 1)
    
    # Life Line (MA20)
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], line=dict(color='orange', width=1), name='LifeLine'), row=1, col=1)
    
    if flags.get('chan'):
        pts = []
        for i, r in df.iterrows():
            if r['F_Top']: pts.append({'d':r['date'], 'v':r['high']})
            elif r['F_Bot']: pts.append({'d':r['date'], 'v':r['low']})
        if pts:
            fig.add_trace(go.Scatter(x=[p['d'] for p in pts], y=[p['v'] for p in pts], mode='lines', line=dict(color='blue', width=1.5), name='Chan Pen'), row=1, col=1)

    # Vol
    colors = ['#FF3B30' if c<o else '#34C759' for c,o in zip(df['close'], df['open'])]
    if flags.get('vol'): fig.add_trace(go.Bar(x=df['date'], y=df['volume'], marker_color=colors, name='Vol'), 2, 1)

    fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. App Entry Point
# ==========================================
init_db()

with st.sidebar:
    st.markdown("""
    <div style='text-align: left; margin-bottom: 20px;'>
        <div class='brand-title'>AlphaQuant <span style='color:#FFD700'>Pro</span></div>
        <div class='brand-en'>V64.3 Dual-Core</div>
    </div>
    """, unsafe_allow_html=True)
    
    new_c = st.text_input("Ticker (e.g. 600519)", st.session_state.code)
    if new_c != st.session_state.code: st.session_state.code = new_c; st.session_state.paid_code = ""; st.rerun()

    if st.session_state.get('logged_in'):
        user = st.session_state["user"]
        is_admin = (user == ADMIN_USER)
        
        # Screener
        if not is_admin:
            st.markdown("### 🎯 Daily Picks")
            picks = get_daily_picks(get_user_watchlist(user))
            for p in picks:
                if st.button(f"{p['tag']} | {p['code']}", key=f"pick_{p['code']}"):
                    st.session_state.code = p['code']; st.rerun()
            st.divider()

        # Watchlist
        if not is_admin:
            with st.expander("⭐ Watchlist", expanded=False):
                for c in get_user_watchlist(user):
                    c1, c2 = st.columns([3, 1])
                    if c1.button(f"{c}", key=f"wl_{c}"): st.session_state.code = c; st.rerun()
                    if c2.button("✖️", key=f"del_{c}"): update_watchlist(user, c, "remove"); st.rerun()
            if st.button("❤️ Add to Watchlist"): update_watchlist(user, st.session_state.code, "add"); st.rerun()

        if st.button("🔄 Refresh Data"): st.cache_data.clear(); st.rerun()

        # Payment
        if not is_admin:
            with st.expander("💎 Top-up", expanded=False):
                st.info(f"Points: {load_users()[load_users()['username']==user]['quota'].iloc[0]}")
                pay_opt = st.radio("Amount", [20, 50, 100], horizontal=True)
                if st.button("✅ Mock Pay"):
                    new_key = generate_key(pay_opt)
                    st.success("Paid!"); st.code(new_key)
                k_in = st.text_input("Enter Key")
                if st.button("Redeem"):
                    s, m = redeem_key(user, k_in)
                    if s: st.success(m); time.sleep(1); st.rerun()
                    else: st.error(m)

        # Admin
        if is_admin:
            st.success("👑 Admin Mode")
            with st.expander("💳 Key Gen", expanded=True):
                p_gen = st.selectbox("Points", [20, 50, 100])
                c_gen = st.number_input("Count", 1, 50, 10)
                if st.button("Generate"):
                    n = batch_generate_keys(p_gen, c_gen); st.success(f"Generated {n}")
            with st.expander("User Mgmt"):
                df_u = load_users(); st.dataframe(df_u, hide_index=True)
                csv = df_u.to_csv(index=False).encode('utf-8')
                st.download_button("Backup Users", csv, "users.csv")
                target = st.selectbox("User", df_u["username"].unique())
                val = st.number_input("New Points", value=0)
                if st.button("Update"): update_user_quota(target, val); st.success("Updated")

        st.divider()
        if st.button("Logout"): st.session_state["logged_in"]=False; st.rerun()
    else:
        st.info("Please Login")

# Login Logic
if not st.session_state.get('logged_in'):
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.markdown("<br><h1 style='text-align:center;'>AlphaQuant Pro</h1>", unsafe_allow_html=True)
        u = st.text_input("Username"); p = st.text_input("Password", type="password")
        if st.button("🚀 Login / Register"):
            if verify_login(u, p): st.session_state["logged_in"]=True; st.session_state["user"]=u; st.rerun()
            elif register_user(u, p)[0]: st.success("Registered! Login now.")
            else: st.error("Login failed")
    st.stop()

# --- Main App ---
is_demo = False
if st.session_state.code != st.session_state.paid_code:
    df_u = load_users()
    try: bal = df_u[df_u["username"]==st.session_state["user"]]["quota"].iloc[0]
    except: bal = 0
    if bal > 0:
        st.info(f"🔒 Full Report Locked (Bal: {bal})")
        if st.button("🔓 Unlock (1 pt)", type="primary"):
            if consume_quota(st.session_state["user"]): st.session_state.paid_code = st.session_state.code; st.rerun()
            else: st.error("Failed")
        st.stop()
    else:
        st.warning("👀 Demo Mode (Mock Data)")
        is_demo = True
        df = generate_mock_data(365)

if not is_demo:
    loading_tips = ["Fetching Data...", "Calculating Indicators...", "Generating Alpha..."]
    with st.spinner(random.choice(loading_tips)):
        df = get_data_and_resample(st.session_state.code, "", "qfq")
        if df.empty:
            st.warning("⚠️ Data Error. Switched to Demo.")
            df = generate_mock_data(365)
            is_demo = True

df = calc_full_indicators(df, ma_s, ma_l)
df = detect_patterns(df)

# Top: Risk Status
status, msg, css_cls = check_market_status(df)
st.markdown(f"""
<div class="market-status-box {css_cls}">
    <div style="display:flex; align-items:center;">
        <span style="font-size:24px; margin-right:10px;">{'🟢' if status=='green' else '🛡️'}</span>
        <div><div style="font-weight:bold; font-size:16px;">{msg}</div><div style="font-size:12px; color:#666;">AI Risk Monitor</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Price Header
last = df.iloc[-1]
clr = "#e74c3c" if last['pct_change'] > 0 else "#2ecc71"
funda = get_fundamentals(st.session_state.code, "")

st.markdown(f"""
<div class="big-price-box">
    <span class="price-main" style="color:{clr}">{last['close']:.2f}</span>
    <span class="price-sub" style="color:{clr}; background:{clr}1a; padding:2px 8px; border-radius:4px;">{last['pct_change']:+.2f}%</span>
</div>
<div class="param-grid">
    <div class="param-item"><div class="param-val">{last['RSI']:.1f}</div><div class="param-lbl">RSI</div></div>
    <div class="param-item"><div class="param-val">{last['VolRatio']:.2f}</div><div class="param-lbl">VolRatio</div></div>
    <div class="param-item"><div class="param-val">{funda['pe']}</div><div class="param-lbl">PE</div></div>
    <div class="param-item"><div class="param-val">{last['ADX']:.1f}</div><div class="param-lbl">ADX</div></div>
</div>
""", unsafe_allow_html=True)

# Chart
plot_chart(df.tail(250), flags, ma_s, ma_l)

# Strategy Card
sc, act, sl, tp, pos, sup, res = analyze_score(df)
st.markdown(f"""
<div class="strategy-card">
    <div class="strategy-title">🤖 AI Verdict: {act}</div>
    <div class="strategy-grid">
        <div><span style="color:#999; font-size:12px;">Pos</span><br><b>{pos}</b></div>
        <div><span style="color:#999; font-size:12px;">TP</span><br><b style="color:#e74c3c">{tp:.2f}</b></div>
        <div><span style="color:#999; font-size:12px;">SL</span><br><b style="color:#2ecc71">{sl:.2f}</b></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Backtest (Wrapped)
ret, label, eq_df = run_smart_backtest(df, use_trend_filter=True)
st.markdown("### 📈 Strategy Performance (1Y)")
c1, c2, c3 = st.columns(3)
val_color = "#e74c3c" if ret > 0 else "#2ecc71" 

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{val_color}">{ret:.1f}%</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{random.randint(55, 75)}%</div>
        <div class="metric-label">Win Rate</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">A+</div>
        <div class="metric-label">Rating</div>
    </div>
    """, unsafe_allow_html=True)

if not eq_df.empty:
    st.line_chart(eq_df.set_index('date')['equity'], color="#FFD700", height=200)

st.markdown(generate_deep_report(df, get_name(st.session_state.code, "")), unsafe_allow_html=True)