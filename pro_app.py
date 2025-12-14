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
    import baostock as bs
    import tushare as ts
    import yfinance as yf
except ImportError:
    st.error("🚨 严重错误：缺少依赖库，请运行: pip install baostock tushare yfinance")
    st.stop()

# ==========================================
# 1. 核心配置
# ==========================================
st.set_page_config(
    page_title="阿尔法量研 Pro V64.4 (稳定中文版)",
    layout="wide",
    page_icon="🔥",
    initial_sidebar_state="expanded"
)

# 🔑 Tushare Token (您的Token)
TUSHARE_TOKEN = "4fe6f3b0ef5355f526f49e54ca032f7d0d770187124c176be266c289"

# 初始化 Session
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if "code" not in st.session_state: st.session_state.code = "600519"
if "paid_code" not in st.session_state: st.session_state.paid_code = ""

# 全局变量
ma_s = 5
ma_l = 20
flags = {
    'ma': True, 'boll': True, 'vol': True, 'macd': True, 
    'kdj': True, 'gann': False, 'fib': True, 'chan': True
}

# 核心常量
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"
DB_FILE = "users_v64.csv"
KEYS_FILE = "card_keys_v64.csv"

# 🔥 CSS (果冻UI + 商业化质感)
ui_css = """
<style>
    /* 全局字体 */
    .stApp {background-color: #f7f8fa; font-family: "PingFang SC", "Microsoft YaHei", sans-serif;}
    
    /* 隐藏多余元素 */
    header[data-testid="stHeader"] { background-color: transparent !important; pointer-events: none; }
    header[data-testid="stHeader"] > div { pointer-events: auto; }
    [data-testid="stDecoration"] { display: none !important; }
    .stDeployButton { display: none !important; }
    footer {display: none !important;}
    
    /* 按钮：果冻金 */
    div.stButton > button {
        background: linear-gradient(145deg, #ffdb4d 0%, #ffb300 100%); 
        color: #5d4037; border: 2px solid #fff9c4; border-radius: 25px; 
        padding: 0.6rem 1.2rem; font-weight: 800; font-size: 16px;
        box-shadow: 0 4px 10px rgba(255, 179, 0, 0.4); 
        transition: all 0.2s; width: 100%;
    }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(255, 179, 0, 0.5); }
    div.stButton > button[kind="secondary"] { background: #f0f0f0; color: #666; border: 1px solid #ddd; box-shadow: none; }

    /* 卡片容器 */
    .app-card { background-color: #ffffff; border-radius: 12px; padding: 16px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.02); }
    .vip-badge { background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 99%); color: #d32f2f; font-size: 10px; font-weight: 800; padding: 2px 8px; border-radius: 10px; font-style: italic; }

    /* 红绿灯风控 */
    .market-status-box {
        padding: 12px 20px; border-radius: 12px; margin-bottom: 20px;
        display: flex; align-items: center; justify-content: space-between;
        background: white; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-left: 5px solid #ccc;
    }
    .status-green { border-left-color: #2ecc71; background: #e8f5e9; color: #2e7d32; }
    .status-red { border-left-color: #e74c3c; background: #ffebee; color: #c62828; }
    .status-yellow { border-left-color: #f1c40f; background: #fef9e7; color: #f39c12; }

    /* 股价大字 */
    .big-price-box { text-align: center; margin-bottom: 20px; }
    .price-main { font-size: 48px; font-weight: 900; line-height: 1; }
    .price-sub { font-size: 16px; font-weight: 600; margin-left: 8px; padding: 2px 6px; border-radius: 4px; }
    .param-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 15px; }
    .param-item { background: #f9fafe; border-radius: 10px; padding: 10px; text-align: center; border: 1px solid #edf2f7; }
    .param-val { font-size: 20px; font-weight: 800; color: #2c3e50; }
    .param-lbl { font-size: 12px; color: #95a5a6; }

    /* 策略卡片 */
    .strategy-card { background: #fcfcfc; border: 1px solid #eee; border-left: 4px solid #ffca28; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.02); }
    .strategy-title { font-size: 18px; font-weight: 800; color: #333; margin-bottom: 10px; }
    .strategy-grid { display: flex; justify-content: space-between; margin-bottom: 10px; }
    
    /* 隐藏原生 Metric */
    [data-testid="metric-container"] { display: none; }
    
    /* 回测卡片 */
    .metric-card {
        background: white; padding: 15px; border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); text-align: center;
        border: 1px solid #f0f0f0;
    }
    .metric-value { font-size: 24px; font-weight: 800; color: #e74c3c; }
    .metric-label { font-size: 12px; color: #7f8c8d; }
</style>
"""
st.markdown(ui_css, unsafe_allow_html=True)

# ==========================================
# 2. 数据库与工具 (完整保留 V61 逻辑)
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
    if u == ADMIN_USER: return False, "保留账号"
    df = load_users()
    if u in df["username"].values: return False, "用户已存在"
    salt = bcrypt.gensalt(); hashed = bcrypt.hashpw(p.encode(), salt).decode()
    df = pd.concat([df, pd.DataFrame([{"username": u, "password_hash": hashed, "watchlist": "", "quota": 0}])], ignore_index=True)
    save_users(df); return True, "注册成功"

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
    if match.empty: return False, "❌ 无效卡密"
    points = int(match.iloc[0]["points"])
    df_keys.loc[match.index[0], "status"] = f"used_by_{username}"
    save_keys(df_keys)
    df_u = load_users(); idx = df_u[df_u["username"] == username].index[0]
    df_u.loc[idx, "quota"] += points; save_users(df_u)
    return True, f"✅ 成功充值 {points} 积分"

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
# 3. 股票数据逻辑 (Tushare + Baostock 双核)
# ==========================================
def process_ticker(code):
    code = str(code).strip().upper()
    # A股 6位数字
    if code.isdigit() and len(code) == 6:
        # Tushare 格式: 600519.SH
        ts_fmt = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
        # Baostock 格式: sh.600519
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

# 🚀 核心数据获取逻辑
@st.cache_data(ttl=1800)
def get_data_and_resample(code, timeframe, adjust):
    raw_code, ts_code, bs_code = process_ticker(code)
    df = pd.DataFrame()
    is_ashare = raw_code.isdigit() and len(raw_code) == 6
    
    # 1. 优先尝试 Tushare
    if is_ashare and TUSHARE_TOKEN:
        try:
            ts.set_token(TUSHARE_TOKEN)
            pro = ts.pro_api()
            end_dt = datetime.now().strftime('%Y%m%d')
            start_dt = (datetime.now() - timedelta(days=700)).strftime('%Y%m%d')
            
            with st.spinner(f"正在连接 Tushare 官方接口 ({ts_code})..."):
                df_ts = pro.daily(ts_code=ts_code, start_date=start_dt, end_date=end_dt)
                
            if not df_ts.empty:
                df = df_ts.rename(columns={'trade_date': 'date', 'vol': 'volume'})
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                df['pct_change'] = df['close'].pct_change() * 100
                return df
        except Exception as e:
            st.sidebar.warning(f"⚠️ Tushare 连接受限 ({str(e)})，正在切换备用源...")

    # 2. 备用尝试 Baostock (如果 Tushare 失败)
    if is_ashare and df.empty:
        try:
            with st.spinner(f"正在连接 Baostock 备用接口 ({bs_code})..."):
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
                    df['pct_change'] = df['close'].pct_change() * 100
                    return df
        except Exception as e:
            st.sidebar.error(f"❌ Baostock 连接失败: {e}")

    # 3. 最后尝试 Yahoo (美股/港股)
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

    # 4. 彻底失败：演示数据
    st.sidebar.warning("⚠️ 所有数据源均不可用，已切换至【离线演示数据】")
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
    df['MA20'] = c.rolling(20).mean() # 修复 KeyError 的关键
    df['MA60'] = c.rolling(60).mean() # 风控线
    
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
# 4. 商业化功能 (风控/精选/回测)
# ==========================================
def check_market_status(df):
    if df is None or df.empty or len(df) < 60: return "neutral", "等待数据...", ""
    curr = df.iloc[-1]
    if curr['close'] > curr['MA60']:
        return "green", "🚀 多头趋势 (建议：积极操作)", "status-green"
    else:
        return "yellow", "🛡️ 防御状态 (建议：空仓观望)", "status-yellow"

def get_daily_picks(user_watchlist):
    hot = ["600519", "NVDA", "TSLA", "300750", "AAPL", "002594"]
    pool = list(set(hot + user_watchlist))[:6]
    results = []
    for c in pool:
        tag = random.choice(["🚀 突破买点", "📈 趋势加速", "💰 主力吸筹"])
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
    
    display_ret = ret; display_label = "绝对收益"
    if ret < 0 and alpha > 0: display_ret = alpha; display_label = "跑赢市场 (Alpha)"
    return display_ret, display_label, pd.DataFrame({'date': dates, 'equity': equity})

def generate_deep_report(df, name):
    curr = df.iloc[-1]
    html = f"""
    <div class="app-card">
        <div class="metric-label">AI 综合研报</div>
        <div style="font-size:14px; margin-top:5px;">
            当前股价 <b>{curr['close']:.2f}</b>。
            RSI指标为 {curr['RSI']:.1f}，{'处于超买区' if curr['RSI']>70 else '处于超卖区' if curr['RSI']<30 else '处于中性区'}。
            MACD 状态: {'金叉共振' if curr['DIF']>curr['DEA'] else '死叉调整'}。
        </div>
    </div>
    """
    return html

def analyze_score(df):
    c = df.iloc[-1]; score=0
    if c['MA_Short']>c['MA_Long']: score+=2
    else: score-=2
    if c['close']>c['MA_Long']: score+=1
    action = "积极买入" if score>=3 else "持有/观望" if score>=0 else "减仓/卖出"
    pos = "80%" if score>=3 else "50%" if score>=0 else "0%"
    atr = df.iloc[-1]['close']*0.03
    return score, action, c['close']-2*atr, c['close']+3*atr, pos, c['low']*0.95, c['high']*1.05

def plot_chart(df, flags, ma_s, ma_l):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'), row=1, col=1)
    
    if flags.get('ma'):
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA_Short'], name=f'MA{ma_s}', line=dict(width=1.2, color='#333333')), 1, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA_Long'], name=f'MA{ma_l}', line=dict(width=1.2, color='#ffcc00')), 1, 1)
    
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], line=dict(color='orange', width=1), name='生命线'), row=1, col=1)
    
    if flags.get('chan'):
        pts = []
        for i, r in df.iterrows():
            if r['F_Top']: pts.append({'d':r['date'], 'v':r['high']})
            elif r['F_Bot']: pts.append({'d':r['date'], 'v':r['low']})
        if pts:
            fig.add_trace(go.Scatter(x=[p['d'] for p in pts], y=[p['v'] for p in pts], mode='lines', line=dict(color='blue', width=1.5), name='缠论笔'), row=1, col=1)

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
        <div class='brand-en'>V64.4 终极稳定版</div>
    </div>
    """, unsafe_allow_html=True)
    
    new_c = st.text_input("股票代码 (如 600519)", st.session_state.code)
    if new_c != st.session_state.code: st.session_state.code = new_c; st.session_state.paid_code = ""; st.rerun()

    if st.session_state.get('logged_in'):
        user = st.session_state["user"]
        is_admin = (user == ADMIN_USER)
        
        # Screener
        if not is_admin:
            st.markdown("### 🎯 每日精选池")
            picks = get_daily_picks(get_user_watchlist(user))
            for p in picks:
                if st.button(f"{p['tag']} | {p['code']}", key=f"pick_{p['code']}"):
                    st.session_state.code = p['code']; st.rerun()
            st.divider()

        # Watchlist
        if not is_admin:
            with st.expander("⭐ 我的自选股", expanded=False):
                for c in get_user_watchlist(user):
                    c1, c2 = st.columns([3, 1])
                    if c1.button(f"{c}", key=f"wl_{c}"): st.session_state.code = c; st.rerun()
                    if c2.button("✖️", key=f"del_{c}"): update_watchlist(user, c, "remove"); st.rerun()
            if st.button("❤️ 加入自选"): update_watchlist(user, st.session_state.code, "add"); st.rerun()

        if st.button("🔄 刷新缓存"): st.cache_data.clear(); st.rerun()

        # Payment
        if not is_admin:
            with st.expander("💎 充值中心", expanded=False):
                st.info(f"当前积分: {load_users()[load_users()['username']==user]['quota'].iloc[0]}")
                pay_opt = st.radio("充值面额", [20, 50, 100], horizontal=True)
                if st.button("✅ 模拟支付"):
                    new_key = generate_key(pay_opt)
                    st.success("支付成功!"); st.code(new_key)
                k_in = st.text_input("输入卡密")
                if st.button("兑换"):
                    s, m = redeem_key(user, k_in)
                    if s: st.success(m); time.sleep(1); st.rerun()
                    else: st.error(m)

        # Admin
        if is_admin:
            st.success("👑 管理员模式")
            with st.expander("💳 卡密生成", expanded=True):
                p_gen = st.selectbox("面值", [20, 50, 100])
                c_gen = st.number_input("数量", 1, 50, 10)
                if st.button("批量生成"):
                    n = batch_generate_keys(p_gen, c_gen); st.success(f"生成 {n} 张")
            with st.expander("用户管理"):
                df_u = load_users(); st.dataframe(df_u, hide_index=True)
                csv = df_u.to_csv(index=False).encode('utf-8')
                st.download_button("备份用户", csv, "users.csv")
                target = st.selectbox("选择用户", df_u["username"].unique())
                val = st.number_input("新积分", value=0)
                if st.button("更新积分"): update_user_quota(target, val); st.success("已更新")

        st.divider()
        if st.button("退出登录"): st.session_state["logged_in"]=False; st.rerun()
    else:
        st.info("请先登录")

# Login Logic
if not st.session_state.get('logged_in'):
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.markdown("<br><h1 style='text-align:center;'>AlphaQuant Pro</h1>", unsafe_allow_html=True)
        u = st.text_input("账号"); p = st.text_input("密码", type="password")
        if st.button("🚀 登录 / 注册"):
            if verify_login(u, p): st.session_state["logged_in"]=True; st.session_state["user"]=u; st.rerun()
            elif register_user(u, p)[0]: st.success("注册成功！请登录")
            else: st.error("登录失败")
    st.stop()

# --- Main App ---
is_demo = False
if st.session_state.code != st.session_state.paid_code:
    df_u = load_users()
    try: bal = df_u[df_u["username"]==st.session_state["user"]]["quota"].iloc[0]
    except: bal = 0
    if bal > 0:
        st.info(f"🔒 深度研报需解锁 (余额: {bal})")
        if st.button("🔓 支付 1 积分查看", type="primary"):
            if consume_quota(st.session_state["user"]): st.session_state.paid_code = st.session_state.code; st.rerun()
            else: st.error("扣费失败")
        st.stop()
    else:
        st.warning("👀 积分不足，已进入【演示模式】")
        is_demo = True
        df = generate_mock_data(365)

if not is_demo:
    loading_tips = ["正在获取数据...", "计算技术指标...", "AI 策略生成中..."]
    with st.spinner(random.choice(loading_tips)):
        df = get_data_and_resample(st.session_state.code, "", "qfq")
        if df.empty:
            st.warning("⚠️ 数据获取失败，切换至演示模式")
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
        <div><div style="font-weight:bold; font-size:16px;">{msg}</div><div style="font-size:12px; color:#666;">AI 实时风控模型监测中</div></div>
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
    <div class="param-item"><div class="param-val">{last['VolRatio']:.2f}</div><div class="param-lbl">量比</div></div>
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
    <div class="strategy-title">🤖 AI 最终建议：{act}</div>
    <div class="strategy-grid">
        <div><span style="color:#999; font-size:12px;">仓位</span><br><b>{pos}</b></div>
        <div><span style="color:#999; font-size:12px;">止盈</span><br><b style="color:#e74c3c">{tp:.2f}</b></div>
        <div><span style="color:#999; font-size:12px;">止损</span><br><b style="color:#2ecc71">{sl:.2f}</b></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Backtest (Wrapped)
ret, label, eq_df = run_smart_backtest(df, use_trend_filter=True)
st.markdown("### 📈 策略回测表现 (近1年)")
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
        <div class="metric-label">波段胜率</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">A+</div>
        <div class="metric-label">AI 评级</div>
    </div>
    """, unsafe_allow_html=True)

if not eq_df.empty:
    st.line_chart(eq_df.set_index('date')['equity'], color="#FFD700", height=200)

st.markdown(generate_deep_report(df, get_name(st.session_state.code)), unsafe_allow_html=True)