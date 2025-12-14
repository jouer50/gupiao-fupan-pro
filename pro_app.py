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
from PIL import Image

# ✅ 0. 依赖库检查与配置
try:
    import yfinance as yf
except ImportError:
    st.error("🚨 严重错误：缺少 `yfinance` 库，请 pip install yfinance")
    st.stop()

# ==========================================
# 1. 核心配置
# ==========================================
st.set_page_config(
    page_title="阿尔法量研 Pro V77 (稳定变现版)",
    layout="wide",
    page_icon="🔥",
    initial_sidebar_state="expanded"
)

# 初始化 Session
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if "code" not in st.session_state: st.session_state.code = "600519"
if "paid_code" not in st.session_state: st.session_state.paid_code = "" 
if "user" not in st.session_state: st.session_state.user = ""

# ✅ 模拟交易 Session
if "paper_holdings" not in st.session_state: st.session_state.paper_holdings = {}
# ✅ 刚刚购买的卡密缓存
if "last_purchased_key" not in st.session_state: st.session_state.last_purchased_key = ""

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

# 🔥 CSS 样式 (优化版：修复乱码，极简清爽)
ui_css = """
<style>
    .stApp {background-color: #f7f8fa; font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;}
    
    /* 侧边栏优化 */
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #f0f0f0; }
    
    /* 按钮样式 */
    div.stButton > button {
        background: #f0f2f5; color: #333; border: 1px solid #dcdfe6;
        border-radius: 8px; padding: 0.5rem 1rem; font-weight: 600;
        transition: all 0.2s; width: 100%;
    }
    div.stButton > button:hover { background: #e6e8eb; border-color: #c0c4cc; }
    div.stButton > button[kind="primary"] { 
        background: #2962ff; color: white; border: none;
    }
    div.stButton > button[kind="primary"]:hover { background: #0d47a1; }

    /* 通用卡片 */
    .app-card { background-color: #ffffff; border-radius: 12px; padding: 16px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.02); }
    
    /* 智能决策卡片 - 修复乱码版 (Simple & Clean) */
    .decision-card {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    .decision-header {
        background: linear-gradient(90deg, #2962ff, #2979ff);
        color: white;
        padding: 12px 20px;
        font-weight: 700;
        font-size: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .decision-body { padding: 20px; text-align: center; }
    
    .action-title { font-size: 36px; font-weight: 900; margin: 10px 0 20px 0; letter-spacing: 1px; }
    .action-buy { color: #d32f2f; }
    .action-sell { color: #2e7d32; }
    .action-hold { color: #ff9800; }
    
    .grid-3 { display: flex; justify-content: space-around; margin-bottom: 20px; border-bottom: 1px solid #f0f0f0; padding-bottom: 15px; }
    .grid-item { text-align: center; }
    .grid-val { font-size: 20px; font-weight: 800; color: #333; }
    .grid-lbl { font-size: 12px; color: #888; margin-top: 4px; }
    
    .support-box { background: #f9f9f9; border-radius: 8px; padding: 10px; display: flex; justify-content: space-around; margin-bottom: 15px; }
    .support-sub { text-align: center; }
    
    .reason-list { text-align: left; background: #f0f7ff; border-radius: 8px; padding: 12px; font-size: 13px; color: #444; }
    .reason-li { margin-bottom: 4px; display: flex; align-items: center; }
    .reason-li::before { content: "•"; color: #2962ff; font-weight: bold; margin-right: 6px; }

    /* 锁定遮罩 */
    .locked-container { position: relative; overflow: hidden; }
    .locked-blur { filter: blur(8px); pointer-events: none; user-select: none; opacity: 0.5; }
    .locked-overlay {
        position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: rgba(255,255,255,0.6); z-index: 10;
    }
    
    /* 其它微调 */
    .brand-title { font-size: 20px; font-weight: 800; color: #1a1a1a; }
    .vip-tag { background: #fff3e0; color: #ff6f00; font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: bold; border: 1px solid #ffe0b2; }
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
            if c not in df.columns: df[c] = ""; updated = True
        if updated: df.to_csv(DB_FILE, index=False)
            
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
    try: return pd.read_csv(DB_FILE, dtype={"watchlist": str, "quota": int, "vip_expiry": str, "paper_json": str}).fillna("")
    except: return pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota", "vip_expiry", "paper_json"])

def save_users(df): df.to_csv(DB_FILE, index=False)

def save_user_holdings(username):
    if username == ADMIN_USER: return
    df = load_users()
    idx = df[df["username"] == username].index
    if len(idx) > 0:
        df.loc[idx[0], "paper_json"] = json.dumps(st.session_state.paper_holdings)
        save_users(df)

def load_user_holdings(username):
    if username == ADMIN_USER: return
    df = load_users()
    row = df[df["username"] == username]
    if not row.empty:
        try: st.session_state.paper_holdings = json.loads(str(row.iloc[0]["paper_json"]))
        except: st.session_state.paper_holdings = {}

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
        base_date = datetime.strptime(current_exp, "%Y-%m-%d") if (current_exp and current_exp != "nan") else now
        if base_date < now: base_date = now
    except: base_date = now
    new_date = base_date + timedelta(days=int(days_to_add))
    df.loc[idx[0], "vip_expiry"] = new_date.strftime("%Y-%m-%d")
    save_users(df)
    return True

def batch_generate_keys(val, count, type="points"):
    df = load_keys()
    new_keys = []
    for _ in range(count):
        suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        key = f"VIP-D-{val}-{suffix}" if type == "days" else f"VIP-{val}-{suffix}"
        new_keys.append({"key": key, "points": val, "status": "unused", "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")})
    df = pd.concat([df, pd.DataFrame(new_keys)], ignore_index=True)
    save_keys(df)
    return [k['key'] for k in new_keys]

def redeem_key(username, key_input):
    key_input = key_input.strip()
    df_keys = load_keys()
    match = df_keys[(df_keys["key"] == key_input) & (df_keys["status"] == "unused")]
    if match.empty: return False, "❌ 无效卡密或已使用"
    val = int(match.iloc[0]["points"])
    is_day_card = "VIP-D-" in key_input
    df_keys.loc[match.index[0], "status"] = f"used_by_{username}"
    save_keys(df_keys)
    if is_day_card:
        update_vip_days(username, val)
        return True, f"✅ 成功激活 VIP {val} 天！"
    else:
        df = load_users()
        idx = df[df["username"] == username].index[0]
        df.loc[idx, "quota"] += val
        save_users(df)
        return True, f"✅ 成功充值 {val} 积分"

def verify_login(u, p):
    if u == ADMIN_USER and p == ADMIN_PASS: return True
    df = load_users()
    row = df[df["username"] == u]
    if row.empty: return False
    try: return bcrypt.checkpw(p.encode(), row.iloc[0]["password_hash"].encode())
    except: return False

def register_user(u, p, reg_type="normal", invite_code=""):
    if u == ADMIN_USER: return False, "保留账号"
    df = load_users()
    if u in df["username"].values: return False, "用户已存在"
    init_quota = 20 if reg_type == "wechat" and invite_code in ["666888", "8888", "alpha2025"] else 0
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(p.encode(), salt).decode()
    new_row = {"username": u, "password_hash": hashed, "watchlist": "", "quota": init_quota, "vip_expiry": "", "paper_json": "{}"}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_users(df)
    return True, f"注册成功！获赠 {init_quota} 积分"

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

def update_watchlist(username, code, action="add"):
    df = load_users()
    idx = df[df["username"] == username].index[0]
    current = str(df.loc[idx, "watchlist"])
    codes = [c.strip() for c in current.split(",") if c.strip()]
    if action == "add" and code not in codes: codes.append(code)
    elif action == "remove" and code in codes: codes.remove(code)
    df.loc[idx, "watchlist"] = ",".join(codes)
    save_users(df)

def get_user_watchlist(username):
    df = load_users()
    if username == ADMIN_USER: return []
    row = df[df["username"] == username]
    if row.empty: return []
    return [c.strip() for c in str(row.iloc[0]["watchlist"]).split(",") if c.strip()]

# ==========================================
# 3. 股票逻辑 (保持不变)
# ==========================================
def is_cn_stock(code): return code.isdigit() and len(code) == 6
def _to_ts_code(s): return f"{s}.SH" if s.startswith('6') else f"{s}.SZ" if s[0].isdigit() else s
def _to_bs_code(s): return f"sh.{s}" if s.startswith('6') else f"sz.{s}" if s[0].isdigit() else s
def process_ticker(code): return f"{code.zfill(4)}.HK" if code.isdigit() and len(code) < 6 else code.strip().upper()

def generate_mock_data(days=365):
    dates = pd.date_range(end=datetime.today(), periods=days)
    close = [150.0]; [close.append(max(10, close[-1] + np.random.normal(0.1, 3.0))) for _ in range(days-1)]
    df = pd.DataFrame({'date': dates, 'close': close})
    df['open'] = df['close'] * np.random.uniform(0.98, 1.02, days)
    df['high'] = df[['open', 'close']].max(axis=1) * np.random.uniform(1.0, 1.03, days)
    df['low'] = df[['open', 'close']].min(axis=1) * np.random.uniform(0.97, 1.0, days)
    df['volume'] = np.random.randint(1000000, 50000000, days)
    df['pct_change'] = df['close'].pct_change() * 100
    return df

@st.cache_data(ttl=3600)
def get_name(code, token, proxy=None):
    clean = code.strip().upper().replace('.SH','').replace('.SZ','')
    QUICK_MAP = {'600519':'贵州茅台','000858':'五粮液','300750':'宁德时代','NVDA':'NVIDIA','AAPL':'Apple','TSLA':'Tesla'}
    if clean in QUICK_MAP: return QUICK_MAP[clean]
    try: return yf.Ticker(code).info.get('shortName', code)
    except: return code

def get_data_and_resample(code, token, timeframe, adjust, proxy=None):
    code = process_ticker(code)
    try:
        yf_df = yf.download(code, period="2y", interval="1d", progress=False, auto_adjust=False)
        if yf_df.empty: return pd.DataFrame()
        if isinstance(yf_df.columns, pd.MultiIndex): yf_df.columns = yf_df.columns.get_level_values(0)
        yf_df.columns = [c.lower() for c in yf_df.columns]
        yf_df.reset_index(inplace=True)
        rename = {c: c for c in yf_df.columns}; rename.update({'date':'date','close':'close','volume':'volume'})
        yf_df.rename(columns=rename, inplace=True)
        if 'volume' not in yf_df.columns: yf_df['volume'] = 0
        df = yf_df[['date','open','high','low','close','volume']].copy()
        df['pct_change'] = df['close'].pct_change() * 100
        if timeframe != '日线':
            rule = 'W' if timeframe == '周线' else 'M'
            df.set_index('date', inplace=True)
            df = df.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
            df['pct_change'] = df['close'].pct_change() * 100
        return df
    except: return pd.DataFrame()

def calc_full_indicators(df, ma_s, ma_l):
    if df.empty: return df
    c = df['close']; h = df['high']; l = df['low']
    df['MA_Short'] = c.rolling(ma_s).mean()
    df['MA_Long'] = c.rolling(ma_l).mean()
    df['MA60'] = c.rolling(60).mean()
    mid = c.rolling(20).mean(); std = c.rolling(20).std()
    df['Upper'] = mid + 2*std; df['Lower'] = mid - 2*std
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    df['DIF'] = e12 - e26; df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean(); df['HIST'] = 2 * (df['DIF'] - df['DEA'])
    delta = c.diff(); up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
    df['RSI'] = 100 - (100 / (1 + up.rolling(14).mean() / (down.rolling(14).mean() + 1e-9)))
    rsv = (c - l.rolling(9).min()) / (h.rolling(9).max() - l.rolling(9).min() + 1e-9) * 100
    df['K'] = rsv.ewm(com=2).mean(); df['D'] = df['K'].ewm(com=2).mean(); df['J'] = 3 * df['K'] - 2 * df['D']
    df['VolRatio'] = df['volume'] / (df['volume'].rolling(5).mean() + 1e-9)
    df['ATR14'] = (h-l).rolling(14).mean()
    return df.fillna(method='bfill')

def detect_patterns(df):
    df['F_Top'] = (df['high'].shift(1)<df['high']) & (df['high'].shift(-1)<df['high'])
    df['F_Bot'] = (df['low'].shift(1)>df['low']) & (df['low'].shift(-1)>df['low'])
    return df

def get_drawing_lines(df):
    idx = df['low'].tail(60).idxmin()
    if pd.isna(idx): return {}, {}
    sp = df.loc[idx, 'low']
    rec = df.tail(120); h = rec['high'].max(); l = rec['low'].min(); d = h-l
    fib = {'0.382': h-d*0.382, '0.5': h-d*0.5, '0.618': h-d*0.618}
    return {}, fib

def analyze_score(df):
    c = df.iloc[-1]; score=0; reasons=[]
    if c['MA_Short']>c['MA_Long']: score+=2; reasons.append("均线金叉 (短线看涨)")
    else: score-=2; reasons.append("均线死叉 (短线看跌)")
    if c['close']>c['MA_Long']: score+=1; reasons.append("站上长期生命线")
    else: reasons.append("跌破长期生命线")
    if c['DIF']>c['DEA']: score+=1; reasons.append("MACD 处于多头区域")
    if c['RSI']<20: score+=2; reasons.append("RSI 进入超卖区 (反弹概率大)")
    if c['VolRatio']>1.5: score+=1; reasons.append("主力放量攻击")
    
    if score>=4: action="积极买入"; color="action-buy"; pos="80% (重仓)"
    elif score>=0: action="持有/观望"; color="action-hold"; pos="50% (中仓)"
    else: action="减仓/卖出"; color="action-sell"; pos="20% (底仓)"
    
    atr = c['ATR14'] if c['ATR14']>0 else c['close']*0.02
    stop_loss = c['close'] - 2*atr
    take_profit = c['close'] + 3*atr
    support = df['low'].iloc[-20:].min()
    resistance = df['high'].iloc[-20:].max()
    return score, action, color, stop_loss, take_profit, pos, support, resistance, reasons

def run_backtest(df):
    if df is None or len(df) < 50: return 0.0, 0.0, 0.0, [], [], pd.DataFrame({'date':[], 'equity':[]})
    df_bt = df.dropna().reset_index(drop=True)
    capital = 100000; position = 0; equity = [capital]; dates = [df_bt.iloc[0]['date']]
    buy_sigs = []; trade_count = 0; wins = 0; entry_price = 0
    
    for i in range(1, len(df_bt)):
        curr = df_bt.iloc[i]; prev = df_bt.iloc[i-1]; price = curr['close']
        buy_sig = prev['MA_Short'] <= prev['MA_Long'] and curr['MA_Short'] > curr['MA_Long']
        sell_sig = prev['MA_Short'] >= prev['MA_Long'] and curr['MA_Short'] < curr['MA_Long']
        
        if buy_sig and position == 0:
            position = capital / price; capital = 0; buy_sigs.append(curr['date']); entry_price = price
        elif sell_sig and position > 0:
            capital = position * price; position = 0; trade_count += 1
            if price > entry_price: wins += 1
        equity.append(capital + position * price); dates.append(curr['date'])
        
    final = equity[-1]; ret = (final - 100000) / 100000 * 100
    win_rate = (wins/trade_count*100) if trade_count>0 else 0
    eq_s = pd.Series(equity); max_dd = ((eq_s - eq_s.cummax()) / eq_s.cummax()).min() * 100
    return ret, win_rate, max_dd, buy_sigs, [], pd.DataFrame({'date':dates, 'equity':equity})

def plot_chart(df, name, flags, ma_s, ma_l):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'), 1, 1)
    if flags.get('ma'):
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA_Short'], name=f'MA{ma_s}', line=dict(color='black', width=1)), 1, 1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA_Long'], name=f'MA{ma_l}', line=dict(color='orange', width=1)), 1, 1)
    if flags.get('vol'):
        colors = ['red' if c>o else 'green' for c,o in zip(df['close'], df['open'])]
        fig.add_trace(go.Bar(x=df['date'], y=df['volume'], marker_color=colors, name='Vol'), 2, 1)
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. 执行入口
# ==========================================
init_db()

with st.sidebar:
    st.markdown("""<div style='text-align: left; margin-bottom: 20px;'><div class='brand-title'>阿尔法量研 <span style='color:#0071e3'>Pro</span></div></div>""", unsafe_allow_html=True)
    
    if st.session_state.get('logged_in'):
        user = st.session_state["user"]
        is_admin = (user == ADMIN_USER)
        is_vip, vip_msg = check_vip_status(user)
        load_user_holdings(user)
        
        # 用户状态
        if is_vip: st.success(f"👑 {vip_msg}")
        else: st.info(f"👤 积分: {load_users()[load_users()['username']==user]['quota'].iloc[0]}")

        # 🔥【修改点1】充值/会员模块移到搜索栏上方
        if not is_admin:
            with st.expander("💎 会员充值 (自动发货)", expanded=True):
                tab_pay, tab_key = st.tabs(["🛍️ 在线购买", "🔑 卡密兑换"])
                
                with tab_pay:
                    st.caption("模拟支付成功后，系统自动显示卡密")
                    buy_type = st.radio("选择套餐", ["20积分 (10元)", "VIP月卡 (30元)", "VIP季卡 (80元)"])
                    if st.button("💰 立即支付 (模拟)", type="primary"):
                        if "20积分" in buy_type:
                            k = batch_generate_keys(20, 1, "points")[0]
                        elif "月卡" in buy_type:
                            k = batch_generate_keys(30, 1, "days")[0]
                        else:
                            k = batch_generate_keys(90, 1, "days")[0]
                        st.session_state.last_purchased_key = k
                        st.balloons()
                    
                    if st.session_state.last_purchased_key:
                        st.success("支付成功！您的卡密如下：")
                        st.code(st.session_state.last_purchased_key, language="text")
                        st.caption("请复制上方卡密，点击右侧“卡密兑换”进行激活")

                with tab_key:
                    k_in = st.text_input("输入卡密", placeholder="粘贴卡密 VIP-...")
                    if st.button("🚀 立即激活"):
                        s, m = redeem_key(user, k_in)
                        if s: st.success(m); time.sleep(1); st.rerun()
                        else: st.error(m)

        # 股票搜索
        new_c = st.text_input("🔍 股票代码", st.session_state.code)
        if new_c != st.session_state.code: st.session_state.code = new_c; st.session_state.paid_code = ""; st.rerun()
        
        # 视觉模式
        view_mode = st.radio("模式", ["极简", "专业(需解锁)"], horizontal=True)
        is_pro = (view_mode == "专业(需解锁)" and (is_admin or is_vip or st.session_state.paid_code == st.session_state.code))
        if view_mode == "专业(需解锁)" and not is_pro:
            if st.button("🔓 解锁当前 (1积分)"):
                if consume_quota(user): st.session_state.paid_code = st.session_state.code; st.rerun()
                else: st.error("积分不足")

        # 模拟交易
        if not is_admin:
            with st.expander("🎮 模拟持仓", expanded=False):
                curr = st.session_state.paper_holdings.get(st.session_state.code, {})
                if curr:
                    st.write(f"成本: {curr.get('cost',0):.2f} | 数量: {curr.get('qty',0)}")
                    if st.button("卖出"): del st.session_state.paper_holdings[st.session_state.code]; save_user_holdings(user); st.rerun()
                else:
                    qty = st.number_input("股数", 100, 10000, 100, 100)
                    if st.button("买入"):
                        st.session_state.paper_holdings[st.session_state.code] = {'cost':0,'qty':qty} # cost 待更新
                        save_user_holdings(user); st.rerun()

        # 管理员面板
        if is_admin:
            st.divider()
            st.write("👑 **管理员控制台**")
            with st.expander("💳 卡密生成中心", expanded=True):
                # 🔥【修改点3】管理员新增VIP月卡生成
                gen_type = st.selectbox("卡密类型", ["积分卡 (Points)", "VIP月卡 (30天)", "VIP季卡 (90天)"])
                count_gen = st.number_input("数量", 1, 50, 5)
                
                if st.button("生成卡密"):
                    if "月卡" in gen_type:
                        keys = batch_generate_keys(30, count_gen, "days")
                    elif "季卡" in gen_type:
                        keys = batch_generate_keys(90, count_gen, "days")
                    else:
                        val = st.number_input("积分面值", 10, 1000, 20)
                        keys = batch_generate_keys(val, count_gen, "points")
                    st.success(f"已生成 {len(keys)} 张")
                    st.dataframe(keys)

        if st.button("退出登录"): st.session_state["logged_in"]=False; st.rerun()
    else:
        st.info("请先登录系统")

# 登录逻辑
if not st.session_state.get('logged_in'):
    st.title("AlphaQuant Pro 登录")
    tab1, tab2 = st.tabs(["登录", "注册"])
    with tab1:
        u = st.text_input("账号"); p = st.text_input("密码", type="password")
        if st.button("登录"):
            if verify_login(u, p): st.session_state["logged_in"]=True; st.session_state["user"]=u; st.rerun()
            else: st.error("失败")
    with tab2:
        nu = st.text_input("新账号"); np = st.text_input("新密码", type="password")
        if st.button("注册"):
            s, m = register_user(nu, np)
            if s: st.success(m)
            else: st.error(m)
    st.stop()

# --- 主内容 ---
st.title(f"📈 {get_name(st.session_state.code, '', None)} ({st.session_state.code})")

df = get_data_and_resample(st.session_state.code, "", "日线", "qfq")
if df.empty:
    st.warning("⚠️ 暂无实时数据，使用演示数据")
    df = generate_mock_data(120)

# 更新持仓成本
if st.session_state.code in st.session_state.paper_holdings:
    if st.session_state.paper_holdings[st.session_state.code]['cost'] == 0:
        st.session_state.paper_holdings[st.session_state.code]['cost'] = df.iloc[-1]['close']
        save_user_holdings(user)

df = calc_full_indicators(df, ma_s, ma_l)
df = detect_patterns(df)

# 基础图表
plot_chart(df.tail(120), "", flags, ma_s, ma_l)

# 智能决策部分
has_access = is_admin or is_vip or (st.session_state.paid_code == st.session_state.code)

if not has_access:
    st.markdown('<div class="locked-container"><div class="locked-blur">', unsafe_allow_html=True)

# 计算指标
sc, act, col_cls, sl, tp, pos, sup, res, reasons = analyze_score(df)
ret, win, mdd, _, _, _ = run_backtest(df)

# 🔥【修改点2】修复乱码，使用简化版HTML结构
# 这里移除了所有复杂的CSS定位，改用Flexbox布局，确保在任何分辨率下都不乱码
reason_html = "".join([f"<div class='reason-li'>{r}</div>" for r in reasons])

final_card_html = f"""
<div class="decision-card">
    <div class="decision-header">
        <span>🎯 智能决策系统 (Smart Decision)</span>
        <span class="vip-tag">Pro</span>
    </div>
    
    <div class="decision-body">
        <div class="action-title {col_cls}">{act}</div>
        
        <div class="grid-3">
            <div class="grid-item">
                <div class="grid-val">{pos}</div>
                <div class="grid-lbl">建议仓位</div>
            </div>
            <div class="grid-item">
                <div class="grid-val" style="color:#d32f2f">{tp:.2f}</div>
                <div class="grid-lbl">目标止盈</div>
            </div>
            <div class="grid-item">
                <div class="grid-val" style="color:#2e7d32">{sl:.2f}</div>
                <div class="grid-lbl">预警止损</div>
            </div>
        </div>
        
        <div class="support-box">
             <div class="support-sub">
                <div style="font-size:12px;color:#666;">🛡️ 下方支撑</div>
                <div style="font-weight:bold; color:#333;">{sup:.2f}</div>
             </div>
             <div class="support-sub">
                <div style="font-size:12px;color:#666;">⚔️ 上方压力</div>
                <div style="font-weight:bold; color:#333;">{res:.2f}</div>
             </div>
        </div>

        <div class="reason-list">
            <div style="font-weight:bold; margin-bottom:5px; color:#333;">💡 核心逻辑：</div>
            {reason_html}
        </div>
        
        <div style="margin-top:15px; font-size:10px; color:#999;">
            * 策略回测数据：总回报 {ret:.1f}% | 胜率 {win:.1f}% | 最大回撤 {mdd:.1f}%
        </div>
    </div>
</div>
"""

st.markdown(final_card_html, unsafe_allow_html=True)

if not has_access:
    st.markdown('</div>', unsafe_allow_html=True) # 结束模糊
    try: bal = load_users()[load_users()["username"]==user]["quota"].iloc[0]
    except: bal = 0
    st.markdown(f"""
    <div class="locked-overlay">
        <h3>🔒 深度决策已锁定</h3>
        <p>包含：买卖点位、仓位管理、机构逻辑</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button(f"🔓 解锁需 1 积分 (当前: {bal})", type="primary", use_container_width=True):
        if consume_quota(user): st.session_state.paid_code = st.session_state.code; st.rerun()
        else: st.error("积分不足，请在左侧充值")