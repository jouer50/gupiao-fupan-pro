import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import bcrypt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. 核心配置 & 界面隐藏
# ==========================================
st.set_page_config(
    page_title="A股深度复盘系统 Pro",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# 🚫 隐藏菜单 CSS
hide_css = """
<style>
    header {visibility: hidden !important; height: 0px !important; padding: 0px !important; margin: 0px !important;}
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    .block-container {padding-top: 1rem !important;}
    .stDeployButton {display: none !important;}
</style>
"""
st.markdown(hide_css, unsafe_allow_html=True)

# 👑 管理员账号 (硬编码，绝对能进)
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"
DB_FILE = "users_v6_final.csv"

# Optional deps
try:
    import tushare as ts
except Exception:
    ts = None
try:
    import baostock as bs
except Exception:
    bs = None

# ==========================================
# 2. 数据库逻辑
# ==========================================
def init_db():
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"])
        df.to_csv(DB_FILE, index=False)

init_db()

def load_users():
    try:
        return pd.read_csv(DB_FILE, dtype={"watchlist": str, "quota": int})
    except:
        return pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"])

def save_users(df):
    df.to_csv(DB_FILE, index=False)

def verify_login(u, p):
    if u == ADMIN_USER and p == ADMIN_PASS: return True
    df = load_users()
    row = df[df["username"] == u]
    if row.empty: return False
    try: return bcrypt.checkpw(p.encode(), row.iloc[0]["password_hash"].encode())
    except: return False

def consume_quota(u):
    if u == ADMIN_USER: return True
    df = load_users()
    idx = df[df["username"] == u].index
    if len(idx) > 0 and df.loc[idx[0], "quota"] > 0:
        df.loc[idx[0], "quota"] -= 1
        save_users(df)
        return True
    return False

def register_user(u, p):
    if u == ADMIN_USER: return False, "无法注册管理员名字"
    df = load_users()
    if u in df["username"].values: return False, "用户已存在"
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(p.encode(), salt).decode()
    new_row = {"username": u, "password_hash": hashed, "watchlist": "", "quota": 20}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_users(df)
    return True, "注册成功"

# ==========================================
# 3. 股票数据逻辑 (增强版)
# ==========================================
def _to_ts_code(symbol):
    symbol = symbol.strip()
    if symbol.isdigit(): return f"{symbol}.SH" if symbol.startswith('6') else f"{symbol}.SZ"
    return symbol

def _to_bs_code(symbol):
    symbol = symbol.strip()
    if symbol.isdigit(): return f"sh.{symbol}" if symbol.startswith('6') else f"sz.{symbol}"
    return symbol

@st.cache_data(ttl=3600)
def get_name(code, token):
    if token and ts:
        try:
            pro = ts.pro_api(token)
            df = pro.stock_basic(ts_code=_to_ts_code(code), fields='name')
            if not df.empty: return df.iloc[0]['name']
        except: pass
    return code

@st.cache_data(ttl=3600)
def get_data(code, token, window_size, adjust):
    # 核心修改：无论用户选几天，强制多拉取 400 天数据
    # 这样可以保证 MA250, MA120, MACD 等指标在切片前就能算准
    fetch_days = max(400, window_size + 100)
    
    # Tushare
    if token and ts:
        try:
            pro = ts.pro_api(token)
            e = pd.Timestamp.today().strftime('%Y%m%d')
            s = (pd.Timestamp.today() - pd.Timedelta(days=fetch_days)).strftime('%Y%m%d')
            df = pro.daily(ts_code=_to_ts_code(code), start_date=s, end_date=e)
            if df is not None and not df.empty:
                if adjust in ['qfq', 'hfq']:
                    adj = pro.adj_factor(ts_code=_to_ts_code(code), start_date=s, end_date=e)
                    if not adj.empty:
                        adj = adj.rename(columns={'trade_date':'date','adj_factor':'factor'})
                        df = df.rename(columns={'trade_date':'date'})
                        df = df.merge(adj[['date','factor']], on='date', how='left').fillna(method='ffill')
                        f = df['factor']
                        ratio = f / f.iloc[-1] if adjust == 'qfq' else f / f.iloc[0]
                        for c in ['open','high','low','close']: df[c] = df[c] * ratio
                
                df = df.rename(columns={'trade_date':'date','vol':'volume','pct_chg':'pct_change'})
                df['date'] = pd.to_datetime(df['date'])
                cols = ['open','high','low','close','volume']
                for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
                return df.sort_values('date').reset_index(drop=True)
        except: pass
        
    # Baostock
    if bs:
        bs.login()
        e = pd.Timestamp.today().strftime('%Y-%m-%d')
        s = (pd.Timestamp.today() - pd.Timedelta(days=fetch_days)).strftime('%Y-%m-%d')
        flag = "2" if adjust == 'qfq' else "1" if adjust == 'hfq' else "3"
        rs = bs.query_history_k_data_plus(_to_bs_code(code),
            "date,open,high,low,close,volume,pctChg",
            start_date=s, end_date=e, frequency="d", adjustflag=flag)
        data = rs.get_data()
        bs.logout()
        if not data.empty:
            df = data.rename(columns={'pctChg':'pct_change'})
            df['date'] = pd.to_datetime(df['date'])
            cols = ['open','high','low','close','volume','pct_change']
            for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
            return df.sort_values('date').reset_index(drop=True)
            
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_fundamentals(code, token):
    res = {"pe": "N/A", "pb": "N/A", "roe": "N/A", "mv": "N/A"}
    if token and ts:
        try:
            pro = ts.pro_api(token)
            df = pro.daily_basic(ts_code=_to_ts_code(code), fields='pe_ttm,pb,total_mv')
            if not df.empty:
                r = df.iloc[-1]
                res['pe'] = f"{r['pe_ttm']:.2f}" if r['pe_ttm'] else "-"
                res['pb'] = f"{r['pb']:.2f}" if r['pb'] else "-"
                res['mv'] = f"{r['total_mv']/10000:.1f}亿" if r['total_mv'] else "-"
            df2 = pro.fina_indicator(ts_code=_to_ts_code(code), fields='roe')
            if not df2.empty: res['roe'] = f"{df2.iloc[0]['roe']:.2f}%"
        except: pass
    return res

def calc_full_indicators(df):
    if df.empty: return df
    # 强制类型转换，防止KeyError/乱码
    for c in ['close','high','low','volume']: df[c] = df[c].astype(float)
    
    close, high, low = df['close'], df['high'], df['low']
    
    for n in [5,10,20,60,120,250]: # 补齐年线
        df[f'MA{n}'] = close.rolling(n).mean()
    
    mid = df['MA20']
    std = close.rolling(20).std()
    df['Upper'] = mid + 2*std
    df['Lower'] = mid - 2*std
    
    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['HIST'] = 2 * (df['DIF'] - df['DEA'])
    
    # RSI
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    rs = up.rolling(14).mean() / (down.rolling(14).mean() + 1e-9)
    df['RSI'] = 100 - (100/(1+rs))
    
    # KDJ
    low9 = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv = (close - low9) / (high9 - low9 + 1e-9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # ATR & ADX
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()
    
    dm_p = np.where((high.diff() > low.diff().abs()) & (high.diff()>0), high.diff(), 0)
    dm_m = np.where((low.diff().abs() > high.diff()) & (low.diff()<0), low.diff().abs(), 0)
    tr14 = tr.rolling(14).sum()
    di_p = 100 * pd.Series(dm_p).rolling(14).sum() / (tr14+1e-9)
    di_m = 100 * pd.Series(dm_m).rolling(14).sum() / (tr14+1e-9)
    df['ADX'] = (abs(di_p - di_m)/(di_p + di_m + 1e-9) * 100).rolling(14).mean()
    
    # Cloud
    p_high = high.rolling(9).max()
    p_low = low.rolling(9).min()
    df['Tenkan'] = (p_high + p_low) / 2
    p_high26 = high.rolling(26).max()
    p_low26 = low.rolling(26).min()
    df['Kijun'] = (p_high26 + p_low26) / 2
    df['SpanA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['SpanB'] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    
    df['VolRatio'] = df['volume'] / (df['volume'].rolling(5).mean() + 1e-9)
    
    return df.fillna(0)

def detect_patterns(df):
    df['Fractal_Top'] = (df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high'])
    df['Fractal_Bot'] = (df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low'])
    return df

def get_drawing_lines(df):
    low_idx = df['low'].tail(60).idxmin()
    if pd.isna(low_idx): return {}, {}
    start_date = df.loc[low_idx, 'date']
    start_price = df.loc[low_idx, 'low']
    gann = {}
    days = (df['date'] - start_date).dt.days
    step = df['ATR14'].iloc[-1] * 0.5
    if step == 0: step = start_price * 0.01
    gann['1x1'] = start_price + days * step
    gann['1x2'] = start_price + days * step * 0.5
    gann['2x1'] = start_price + days * step * 2.0
    recent = df.tail(120)
    h = recent['high'].max()
    l = recent['low'].min()
    diff = h - l
    fib = {
        '0.236': h - diff * 0.236,
        '0.382': h - diff * 0.382,
        '0.5': h - diff * 0.5,
        '0.618': h - diff * 0.618
    }
    return gann, fib

def analyze_signals(df):
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    reasons = []
    
    if curr['MA5'] > curr['MA20']: score += 2; reasons.append("✅ 短线多头 (MA5>MA20)")
    else: score -= 2; reasons.append("❌ 短线空头 (MA5<MA20)")
    if curr['close'] > curr['MA60']: score += 1; reasons.append("✅ 站上生命线 (MA60)")
    
    if curr['DIF'] > curr['DEA'] and curr['HIST'] > prev['HIST']: score += 1; reasons.append("✅ MACD金叉增强")
    elif curr['DIF'] < curr['DEA']: score -= 1; reasons.append("❌ MACD死叉")
    
    if curr['RSI'] < 20: score += 2; reasons.append("📉 RSI极度超卖")
    elif curr['RSI'] > 80: score -= 2; reasons.append("📈 RSI极度超买")
    
    if curr['close'] > max(curr['SpanA'], curr['SpanB']): score += 1; reasons.append("☁️ 云层上方 (趋势多)")
    if curr['ADX'] > 25: reasons.append("🔥 趋势强劲 (ADX>25)")
    if curr['VolRatio'] > 1.5: score += 1; reasons.append("🌊 今日放量")
        
    if score >= 4: action, color = "🚀 强力买入", "success"
    elif score >= 1: action, color = "👀 逢低吸纳", "warning"
    elif score >= -2: action, color = "✋ 观望/持有", "secondary"
    else: action, color = "🏃 减仓/卖出", "error"
    
    support = df['low'].tail(20).min()
    resistance = df['high'].tail(20).max()
    atr = curr['ATR14']
    stop_loss = curr['close'] - 2 * atr
    take_profit = curr['close'] + 3 * atr
    
    return {"score": score, "action": action, "color": color, "reasons": reasons, "sup": support, "res": resistance, "sl": stop_loss, "tp": take_profit}

def main_uptrend_check(df):
    curr = df.iloc[-1]
    is_bull = curr['MA5'] > curr['MA20'] > curr['MA60']
    is_cloud = curr['close'] > max(curr['SpanA'], curr['SpanB'])
    if is_bull and is_cloud and curr['ADX'] > 20: return "🚀 确认：主升浪行情中", "success"
    if is_cloud: return "📈 趋势：震荡上行", "warning"
    return "📉 趋势：弱势调整 / 空头", "error"

def plot_full_chart(df, title, show_gann, show_fib, show_chanlun):
    if df.empty: return
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.15, 0.15, 0.2])
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'), row=1, col=1)
    for m in ['MA5','MA20','MA60']: fig.add_trace(go.Scatter(x=df['date'], y=df[m], name=m, line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['Lower'], fill='tonexty', fillcolor='rgba(0,0,255,0.05)', line=dict(width=0), name='BOLL'), row=1, col=1)
    
    gann, fib = get_drawing_lines(df)
    if show_gann:
        for k, v in gann.items(): fig.add_trace(go.Scatter(x=df['date'], y=v, mode='lines', line=dict(dash='dot', width=1), name=f'Gann {k}'), row=1, col=1)
    if show_fib:
        for k, v in fib.items(): fig.add_hline(y=v, line_dash="dash", line_color="orange", annotation_text=f"Fib {k}", row=1, col=1)
            
    if show_chanlun:
        tops, bots = df[df['Fractal_Top']], df[df['Fractal_Bot']]
        fig.add_trace(go.Scatter(x=tops['date'], y=tops['high'], mode='markers', marker_symbol='triangle-down', marker_color='green', name='顶分型'), row=1, col=1)
        fig.add_trace(go.Scatter(x=bots['date'], y=bots['low'], mode='markers', marker_symbol='triangle-up', marker_color='red', name='底分型'), row=1, col=1)

    colors = ['red' if c>=o else 'green' for c,o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df['date'], y=df['volume'], marker_color=colors, name='Vol'), row=2, col=1)
    fig.add_trace(go.Bar(x=df['date'], y=df['HIST'], name='MACD柱'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['DIF'], name='DIF'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['DEA'], name='DEA'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name='K'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name='D'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['J'], name='J'), row=4, col=1)
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=900, margin=dict(t=30, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. 主程序
# ==========================================
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    st.markdown("<br><br><h1 style='text-align:center'>🔐 A股深度复盘系统 Pro</h1>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        tab1, tab2 = st.tabs(["登录", "注册"])
        with tab1:
            u = st.text_input("账号")
            p = st.text_input("密码", type="password")
            if st.button("🚀 登录", type="primary", use_container_width=True):
                if verify_login(u.strip(), p):
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = u.strip()
                    st.rerun()
                else: st.error("账号或密码错误 (管理员: ZCX001 / 123456)")
        with tab2:
            nu = st.text_input("新账号")
            np1 = st.text_input("新密码", type="password")
            if st.button("📝 注册", use_container_width=True):
                suc, msg = register_user(nu.strip(), np1)
                if suc: st.success(msg)
                else: st.error(msg)
    st.stop()

user = st.session_state["user"]
is_admin = (user == ADMIN_USER)

with st.sidebar:
    st.header(f"👤 {user}")
    if is_admin:
        st.success("✅ 管理员后台已激活")
        with st.expander("👮‍♂️ 积分管理", expanded=True):
            df_u = load_users()
            st.dataframe(df_u[["username","quota"]], hide_index=True)
            u_list = [x for x in df_u["username"] if x != ADMIN_USER]
            if u_list:
                target = st.selectbox("修改用户", u_list)
                val = st.number_input("积分", value=50, step=10)
                if st.button("💾 保存"):
                    idx = df_u[df_u["username"]==target].index[0]
                    df_u.loc[idx, "quota"] = val
                    save_users(df_u)
                    st.success("成功")
    else:
        df_u = load_users()
        q = df_u[df_u["username"]==user]["quota"].iloc[0]
        st.metric("剩余积分", q)

    st.divider()
    try: def_tok = st.secrets["TUSHARE_TOKEN"]
    except: def_tok = ""
    token = st.text_input("Tushare Token", value=def_tok, type="password")
    
    if "code" not in st.session_state: st.session_state.code = "600519"
    new_code = st.text_input("股票代码", st.session_state.code)
    if new_code != st.session_state.code: st.session_state.code = new_code
    
    name = get_name(st.session_state.code, token)
    
    # ✅ 修复：找回所有时间窗口选项
    days = st.radio("窗口 (天)", [7, 30, 60, 90, 180, 250, 360], index=2, horizontal=True)
    adjust = st.selectbox("复权", ["qfq", "hfq", ""], 0)
    
    st.divider()
    show_gann = st.checkbox("江恩角度线", True)
    show_fib = st.checkbox("斐波那契回撤", True)
    show_chanlun = st.checkbox("缠论分型", True)
    st.divider()
    if st.button("🚪 退出登录"): st.session_state["logged_in"] = False; st.rerun()

c1, c2 = st.columns([3, 1])
with c1: st.title(f"📈 {name} ({st.session_state.code})")
with c2:
    if st.button("🔄 刷新数据 (消耗1积分)", type="primary"):
        if consume_quota(user): st.session_state["refresh"] = time.time(); st.rerun()
        else: st.error("积分不足")

with st.spinner("🚀 AI 正在深度分析..."):
    # ✅ 核心修复：后台强制多拉数据，前台只展示选定的天数
    # 如果用户选7天，我们后台仍然拉400天，算出指标后，再截取最后7天画图
    # 这样 MA250 就不会断了
    df = get_data(st.session_state.code, token, days, adjust) 
    funda = get_fundamentals(st.session_state.code, token)

if df.empty:
    st.warning("⚠️ 暂无数据，请检查代码或 Token")
else:
    # 1. 先在全量数据上算指标 (保证准确)
    df = calc_full_indicators(df)
    df = detect_patterns(df)
    
    # 2. 趋势判断 (用最新数据)
    trend_txt, trend_col = main_uptrend_check(df)
    if trend_col == "success": st.success(f"### {trend_txt}")
    elif trend_col == "warning": st.warning(f"### {trend_txt}")
    else: st.error(f"### {trend_txt}")
    
    # 3. 截取用户想看的时间段进行画图 (View Slice)
    plot_df = df.tail(days).copy() 
    
    # 4. 指标卡片
    latest = df.iloc[-1]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("价格", f"{latest['close']:.2f}", f"{latest['pct_change']:.2f}%")
    k2.metric("PE (市盈率)", funda['pe'])
    k3.metric("RSI (强弱)", f"{latest['RSI']:.1f}")
    k4.metric("ADX (力度)", f"{latest['ADX']:.1f}")
    k5.metric("量比", f"{latest['VolRatio']:.2f}")
    
    # 5. 画图 (用切片后的数据)
    plot_full_chart(plot_df, f"{name} 深度技术分析", show_gann, show_fib, show_chanlun)
    
    # 6. 信号分析 (用全量数据算出的结果)
    res = analyze_signals(df)
    st.subheader(f"🤖 AI 决策建议: {res['action']} (评分: {res['score']})")
    
    s1, s2, s3 = st.columns(3)
    if res['color'] == 'success': s1.success(f"建议仓位: 高 (50%~80%)")
    elif res['color'] == 'warning': s1.warning(f"建议仓位: 中 (20%~50%)")
    else: s1.error(f"建议仓位: 低/空仓 (0%~20%)")
    
    s2.info(f"🛡️ 止损位: {res['sl']:.2f} (2ATR)")
    s3.info(f"💰 止盈位: {res['tp']:.2f} (3ATR)")
    st.caption(f"📍 关键点位监测 | 支撑: **{res['sup']:.2f}** | 压力: **{res['res']:.2f}**")
    
    with st.expander("🔍 点击查看详细评分逻辑", expanded=True):
        for r in res['reasons']: st.write(r)
