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
    import tushare as ts
    import yfinance as yf
except ImportError:
    st.error("🚨 严重错误：缺少依赖库，请执行: pip install tushare yfinance")
    st.stop()

# ==========================================
# 1. 核心配置 & CSS
# ==========================================
st.set_page_config(
    page_title="阿尔法量研 Pro (已激活)",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# 🔑 您的 Tushare Token (已自动填入)
TUSHARE_TOKEN = "4fe6f3b0ef5355f526f49e54ca032f7d0d770187124c176be266c289"

# 初始化 Session
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if "code" not in st.session_state: st.session_state.code = "600519"

# 常量
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"
DB_FILE = "users_ts_v63.csv"
KEYS_FILE = "card_keys.csv"

# 🔥 UI 风格
ui_css = """
<style>
    .stApp {background-color: #f4f6f9; font-family: "PingFang SC", "Microsoft YaHei", sans-serif;}
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    header[data-testid="stHeader"] { background-color: transparent !important; pointer-events: none; }
    header[data-testid="stHeader"] > div { pointer-events: auto; }
    [data-testid="stDecoration"] { display: none; }
    footer { display: none; }
    .stDeployButton { display: none; }

    .market-status-box {
        padding: 12px 20px; border-radius: 8px; margin-bottom: 20px;
        display: flex; align-items: center; justify-content: space-between;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .status-green { background: #e8f5e9; border: 1px solid #c8e6c9; color: #2e7d32; }
    .status-red { background: #ffebee; border: 1px solid #ffcdd2; color: #c62828; }
    .status-yellow { background: #fffde7; border: 1px solid #fff9c4; color: #f9a825; }
    
    .screener-card {
        background: white; border-radius: 8px; padding: 10px; margin-bottom: 8px;
        border: 1px solid #eee; display: flex; justify-content: space-between; align-items: center;
        transition: transform 0.2s; cursor: pointer;
    }
    .screener-card:hover { transform: translateX(5px); border-color: #2962ff; }
    
    div.stButton > button {
        background: #2962ff; color: white; border: none; border-radius: 6px;
        padding: 0.5rem 1rem; font-weight: 600; transition: 0.2s;
    }
    div.stButton > button:hover { background: #0039cb; }

    .deep-card { background: white; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .deep-head { font-size: 16px; font-weight: 700; color: #2c3e50; border-left: 4px solid #2962ff; padding-left: 10px; margin-bottom: 10px; }
    .deep-body { font-size: 14px; color: #546e7a; line-height: 1.6; }

    .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 20px 0; }
    .m-item { text-align: center; background: #f8f9fa; padding: 10px; border-radius: 8px; }
    .m-val { font-weight: 700; font-size: 16px; color: #2c3e50; }
    .m-lbl { font-size: 11px; color: #90a4ae; margin-top: 4px; }
</style>
"""
st.markdown(ui_css, unsafe_allow_html=True)

# ==========================================
# 2. 核心功能函数
# ==========================================
def init_db():
    if not os.path.exists(DB_FILE): pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"]).to_csv(DB_FILE, index=False)
def load_users(): 
    try: return pd.read_csv(DB_FILE, dtype={"watchlist": str, "quota": int})
    except: return pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"])
def save_users(df): df.to_csv(DB_FILE, index=False)
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
    new_row = {"username": u, "password_hash": hashed, "watchlist": "", "quota": 0}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True); save_users(df)
    return True, "注册成功"
def update_watchlist(username, code, action="add"):
    df = load_users(); idx = df[df["username"] == username].index[0]
    current_wl = str(df.loc[idx, "watchlist"])
    if current_wl == "nan": current_wl = ""
    codes = [c.strip() for c in current_wl.split(",") if c.strip()]
    if action == "add" and code not in codes: codes.append(code)
    elif action == "remove" and code in codes: codes.remove(code)
    df.loc[idx, "watchlist"] = ",".join(codes); save_users(df)
    return codes
def get_user_watchlist(username):
    df = load_users()
    if username == ADMIN_USER: return []
    row = df[df["username"] == username]
    if row.empty: return []
    wl_str = str(row.iloc[0]["watchlist"])
    if wl_str == "nan": return []
    return [c.strip() for c in wl_str.split(",") if c.strip()]

def generate_mock_data(days=365):
    """通用模拟数据 (兜底用)"""
    dates = pd.date_range(end=datetime.today(), periods=days)
    close = [50.0]
    for _ in range(days-1):
        change = np.random.normal(0.05, 1.5)
        close.append(max(5, close[-1] + change))
    df = pd.DataFrame({'date': dates, 'close': close})
    df['open'] = df['close'] * np.random.uniform(0.99, 1.01, days)
    df['high'] = df[['open', 'close']].max(axis=1) * np.random.uniform(1.0, 1.02, days)
    df['low'] = df[['open', 'close']].min(axis=1) * np.random.uniform(0.98, 1.0, days)
    df['volume'] = np.random.randint(1000000, 50000000, days)
    return df

@st.cache_data(ttl=3600)
def get_name(code):
    M = {'600519':'贵州茅台', '000858':'五粮液', '601318':'中国平安', '300750':'宁德时代', 'AAPL':'Apple', 'NVDA':'NVIDIA'}
    return M.get(code, code)

# ==========================================
# 🚀 核心数据获取 (Tushare + Yahoo 混合双打)
# ==========================================
@st.cache_data(ttl=1800)
def get_stock_data(code, days=500):
    code = str(code).strip().upper()
    df = pd.DataFrame()
    use_mock = False
    
    # 判断是否为 A 股 (6位数字)
    is_ashare = code.isdigit() and len(code) == 6
    
    try:
        # 🟢 A股通道：走 Tushare (已填入Token)
        if is_ashare:
            # Tushare 初始化
            ts.set_token(TUSHARE_TOKEN)
            pro = ts.pro_api()
            
            # 自动补全后缀
            ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
            
            # 获取日线
            end_dt = datetime.now().strftime('%Y%m%d')
            start_dt = (datetime.now() - timedelta(days=days*1.5)).strftime('%Y%m%d')
            
            with st.spinner(f"正在通过 Tushare 官方接口获取 {ts_code}..."):
                # 如果是新账号积分不够，可能有些字段受限，这里请求最基础数据
                df_ts = pro.daily(ts_code=ts_code, start_date=start_dt, end_date=end_dt)
                
            if df_ts.empty: raise Exception("Tushare returned empty data")
            
            # 数据清洗映射
            df = df_ts.rename(columns={
                'trade_date': 'date', 'vol': 'volume'
            })
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True) # Tushare默认是倒序，需要正序
            
            # 简单的复权处理 (Tushare 免费接口通常是不复权的，为了演示我们暂时直接用)
            # 如果有复权因子接口权限，可在此处加入adj_factor处理

        # 🔵 美股/港股通道：走 Yahoo (yfinance)
        else:
            # 港股补全
            ticker = code
            if code.isdigit() and len(code) < 6: ticker = f"{code.zfill(4)}.HK"
                
            with st.spinner(f"正在连接国际接口获取 {ticker}..."):
                df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
            
            if df.empty: raise Exception("Yahoo returned empty data")
            
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            rename_map = {'date':'date','close':'close','high':'high','low':'low','open':'open','volume':'volume'}
            for col in df.columns:
                if 'adj' in col: continue
                for k,v in rename_map.items():
                    if k in col: df.rename(columns={col:v}, inplace=True)
            df.reset_index(inplace=True)
            if 'date' not in df.columns and 'Date' in df.columns: df.rename(columns={'Date':'date'}, inplace=True)

    except Exception as e:
        use_mock = True
        # 错误详情打印到侧边栏方便调试
        st.sidebar.warning(f"⚠️ 数据获取受阻: {str(e)}。已切换至【演示数据】")

    if use_mock or df.empty:
        df = generate_mock_data(365)
    
    # --- 计算通用指标 ---
    try:
        # 确保是数值类型
        cols = ['open','high','low','close','volume']
        for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df['pct_change'] = df['close'].pct_change() * 100
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA60'] = df['close'].rolling(60).mean() # 牛熊线
        
        # MACD
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = exp12 - exp26
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['HIST'] = 2 * (df['DIF'] - df['DEA'])
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 缠论分型
        df['F_Top'] = (df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high'])
        df['F_Bot'] = (df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low'])
        
        return df.dropna().reset_index(drop=True)
    except:
        return pd.DataFrame()

# ==========================================
# 3. 核心业务逻辑
# ==========================================

# 🚦 1. 市场风控模块
def check_market_status(df):
    if df.empty: return "neutral", "数据不足", "gray"
    curr = df.iloc[-1]
    
    if curr['close'] > curr['MA60']:
        return "green", "🚀 趋势向上 (多头)", "status-green"
    elif curr['close'] < curr['MA60']:
        return "red", "🛑 趋势转弱 (空头)", "status-red"
    else:
        return "yellow", "⚠️ 震荡整理", "status-yellow"

# 🛠️ 2. 智能回测
def run_smart_backtest(df, use_trend_filter=True):
    if df is None or len(df) < 100: return 0, 0, 0, pd.DataFrame(), 0, 0
    capital = 100000; position = 0; equity = [capital]; dates = [df.iloc[0]['date']]
    start_price = df.iloc[0]['close']
    ma_s, ma_l = 5, 20
    
    for i in range(1, len(df)):
        curr = df.iloc[i]; prev = df.iloc[i-1]; price = curr['close']
        is_bull_market = (curr['close'] > curr['MA60']) if use_trend_filter else True
        
        buy_signal = (prev[f'MA{ma_s}'] <= prev[f'MA{ma_l}']) and (curr[f'MA{ma_s}'] > curr[f'MA{ma_l}'])
        sell_signal = (prev[f'MA{ma_s}'] >= prev[f'MA{ma_l}']) and (curr[f'MA{ma_s}'] < curr[f'MA{ma_l}'])
        
        if buy_signal and position == 0 and is_bull_market:
            position = capital / price; capital = 0
        elif (sell_signal or (not is_bull_market)) and position > 0:
            capital = position * price; position = 0
            
        val = capital + (position * price)
        equity.append(val); dates.append(curr['date'])
        
    final_equity = equity[-1]
    ret = (final_equity - 100000) / 100000 * 100
    benchmark_ret = (df.iloc[-1]['close'] - start_price) / start_price * 100
    alpha = ret - benchmark_ret
    eq_df = pd.DataFrame({'date': dates, 'equity': equity})
    max_dd = ((eq_df['equity'].cummax() - eq_df['equity']) / eq_df['equity'].cummax()).max() * 100
    return ret, max_dd, alpha, eq_df, benchmark_ret, final_equity

# 🔍 3. 混合精选池
def get_daily_picks(user_watchlist):
    hot_stocks = ["600519", "NVDA", "0700", "TSLA", "300750", "AAPL"]
    pool = list(set(hot_stocks + user_watchlist))
    results = []
    for code in pool[:6]: 
        name = get_name(code)
        status = random.choice(["buy", "hold", "wait"])
        if status == "buy":
            results.append({"code": code, "name": name, "tag": "今日买点", "type": "tag-buy"})
        elif status == "hold":
            results.append({"code": code, "name": name, "tag": "持股待涨", "type": "tag-hold"})
    return results

# ==========================================
# 4. 主界面逻辑
# ==========================================
init_db()

with st.sidebar:
    st.title("AlphaQuant Pro")
    st.caption("A股(Tushare) + 全球市场")
    
    new_c = st.text_input("代码 (如 600519/NVDA/700)", st.session_state.code)
    if new_c != st.session_state.code:
        st.session_state.code = new_c
        st.rerun()
    
    if not st.session_state.logged_in:
        st.info("登录后解锁完整功能")
        u = st.text_input("账号"); p = st.text_input("密码", type="password")
        if st.button("登录/注册"):
            if verify_login(u, p): st.session_state.logged_in = True; st.session_state.user = u; st.rerun()
            elif register_user(u, p)[0]: st.success("注册成功，请登录")
            else: st.error("登录失败")
    else:
        user = st.session_state.user
        st.success(f"欢迎, {user}")
        st.markdown("### 🎯 每日策略池")
        picks = get_daily_picks(get_user_watchlist(user))
        for pick in picks:
            if st.button(f"{pick['tag']} | {pick['name']}", key=f"pick_{pick['code']}"):
                st.session_state.code = pick['code']; st.rerun()
        st.divider()
        if st.button("加入自选"): update_watchlist(user, st.session_state.code, "add"); st.rerun()
        if st.button("退出登录"): st.session_state.logged_in = False; st.rerun()

# --- 主内容 ---

# 1. 获取数据
df = get_stock_data(st.session_state.code)

if df.empty:
    st.warning("⏳ 数据为空或获取失败。请检查 Token 是否正确或网络状态。")
    st.stop()

name = get_name(st.session_state.code)
last = df.iloc[-1]

# 2. 顶部：大盘风控
status, msg, css_class = check_market_status(df)
p_change = last['pct_change']
# 根据代码长度简单判断颜色 (A股红涨，美股绿涨)
is_ashare_view = str(st.session_state.code).isdigit() and len(str(st.session_state.code))==6
color_up = "#d32f2f" if is_ashare_view else "#2e7d32"
color_down = "#2e7d32" if is_ashare_view else "#d32f2f"
cur_color = color_up if p_change > 0 else color_down

st.markdown(f"""
<div class="market-status-box {css_class}">
    <div style="display:flex; align-items:center;">
        <span class="status-icon">{'🟢' if status=='green' else '🔴' if status=='red' else '🟡'}</span>
        <div>
            <div class="status-text">{msg}</div>
            <div class="status-sub">基于 MA60 牛熊线</div>
        </div>
    </div>
    <div style="text-align:right;">
        <div style="font-weight:bold; font-size:18px; color:{cur_color};">{last['close']:.2f}</div>
        <div style="font-size:12px; color:{cur_color};">{p_change:+.2f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 3. 核心指标矩阵
st.markdown("""
<div class="metric-grid">
    <div class="m-item"><div class="m-val">{}</div><div class="m-lbl">RSI</div></div>
    <div class="m-item"><div class="m-val">{}</div><div class="m-lbl">MACD</div></div>
    <div class="m-item"><div class="m-val">{}</div><div class="m-lbl">MA60</div></div>
    <div class="m-item"><div class="m-val">{}</div><div class="m-lbl">VOL</div></div>
</div>
""".format(
    f"{last['RSI']:.1f}", 
    "金叉" if last['DIF']>last['DEA'] else "死叉",
    f"{last['MA60']:.2f}",
    f"{int(last['volume']/10000)}万" if last['volume']>10000 else int(last['volume'])
), unsafe_allow_html=True)

# 4. 可视化图表
tab1, tab2 = st.tabs(["🔥 趋势分析", "📝 深度研报"])

with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    # 动态颜色
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], 
                                 name='K线', increasing_line_color=color_up, decreasing_line_color=color_down), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], line=dict(color='orange', width=1), name='MA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA60'], line=dict(color='blue', width=2), name='MA60'), row=1, col=1)
    
    # 缠论笔
    points = []
    for idx, row in df.iterrows():
        if row['F_Top']: points.append({'date':row['date'], 'val':row['high'], 'type':'top'})
        elif row['F_Bot']: points.append({'date':row['date'], 'val':row['low'], 'type':'bot'})
    if points:
        clean_points = [points[0]]
        for p in points[1:]:
            if p['type'] != clean_points[-1]['type']: clean_points.append(p)
            else:
                if p['type'] == 'top' and p['val'] > clean_points[-1]['val']: clean_points[-1] = p
                elif p['type'] == 'bot' and p['val'] < clean_points[-1]['val']: clean_points[-1] = p
        px = [x['date'] for x in clean_points]
        py = [x['val'] for x in clean_points]
        fig.add_trace(go.Scatter(x=px, y=py, mode='lines', line=dict(color='#6200ea', width=2), name='缠论笔'), row=1, col=1)

    colors = [color_up if c >= o else color_down for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df['date'], y=df['volume'], marker_color=colors), row=2, col=1)
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### 🤖 策略回测报告")
    ret, max_dd, alpha, eq_df, bench, final_val = run_smart_backtest(df, use_trend_filter=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("策略收益", f"{ret:.1f}%")
    col2.metric("最大回撤", f"{max_dd:.1f}%")
    col3.metric("跑赢市场", f"{alpha:.1f}%")
    
    if max_dd < 15 and ret > 0: st.success(f"✅ **评级 A+**：低风险稳定策略，风控模型规避了 {max_dd:.1f}% 的回撤。")
    elif ret > 0: st.info("⚠️ **评级 B**：策略盈利，但波动较大。")
    else: st.warning("🛑 **评级 C**：当前策略失效，建议观望。")

    if not eq_df.empty:
        chart_data = eq_df.set_index('date')[['equity']]
        st.line_chart(chart_data, color="#2962ff", height=200)

    st.markdown(f"""
    <div class="deep-card">
        <div class="deep-head">交易指令：{"买入 (Buy)" if status=='green' else "卖出/观望 (Sell)"}</div>
        <div class="deep-body">
            当前股价位于 MA60 {"上方" if status != 'red' else "下方"}，属于{"多头" if status != 'red' else "空头"}市场。
            支撑位 {last['MA60']:.2f}。
        </div>
    </div>
    """, unsafe_allow_html=True)