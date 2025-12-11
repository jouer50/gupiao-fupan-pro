import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import bcrypt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. 核心配置 (修复版)
# ==========================================
st.set_page_config(
    page_title="AlphaQuant Pro",
    layout="wide",
    page_icon="🍎",
    initial_sidebar_state="expanded"
)

# 🎨 Apple Style CSS - 保持美观，但代码更稳健
apple_css = """
<style>
    /* 全局字体与背景 */
    .stApp {
        background-color: #ffffff;
        color: #1d1d1f;
    }
    
    /* 侧边栏 */
    [data-testid="stSidebar"] {
        background-color: #fbfbfb;
        border-right: 1px solid #e5e5e5;
    }

    /* 隐藏原生多余元素 */
    header {visibility: hidden !important; height: 0px !important; padding: 0px !important; margin: 0px !important;}
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    .stDeployButton {display: none !important;}
    
    /* 强制内容上移 */
    .block-container {
        padding-top: 1rem !important;
    }

    /* 按钮美化 */
    div.stButton > button {
        background-color: #0071e3;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        width: 100%; /* 强制全宽，替代 use_container_width */
    }
    div.stButton > button:hover {
        background-color: #0077ed;
        box-shadow: 0 2px 8px rgba(0, 113, 227, 0.2);
    }

    /* 指标卡片 */
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 600 !important;
        color: #1d1d1f !important;
    }
    
    /* Logo */
    .brand-logo {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1d1d1f;
        text-align: left;
        margin-bottom: 20px;
        padding-left: 5px;
    }
    .brand-logo span { color: #0071e3; }
</style>
"""
st.markdown(apple_css, unsafe_allow_html=True)

# 👑 管理员账号
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"
DB_FILE = "users.csv"

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

def register_user(u, p):
    if u == ADMIN_USER: return False, "保留账号无法注册"
    df = load_users()
    if u in df["username"].values: return False, "用户已存在"
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(p.encode(), salt).decode()
    new_row = {"username": u, "password_hash": hashed, "watchlist": "", "quota": 20}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_users(df)
    return True, "注册成功"

# ==========================================
# 3. 股票数据逻辑
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
    if bs:
        try:
            bs.login()
            rs = bs.query_stock_basic(code=_to_bs_code(code))
            if rs.error_code == '0':
                row = rs.get_row_data()
                if row and len(row) > 1:
                    name = row[1]
                    bs.logout()
                    return name
            bs.logout()
        except: pass
    return code

@st.cache_data(ttl=3600)
def get_data(code, token, window_size, adjust):
    fetch_days = max(400, window_size + 100)
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
    if res['pe'] == "N/A" and bs:
        try:
            bs.login()
            import datetime
            e = datetime.date.today().strftime("%Y-%m-%d")
            s = (datetime.date.today() - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
            rs = bs.query_history_k_data_plus(_to_bs_code(code), "date,peTTM,pbMRQ", start_date=s, end_date=e, frequency="d")
            rows = rs.get_data()
            if not rows.empty:
                last = rows.iloc[-1]
                res['pe'] = str(last['peTTM'])
                res['pb'] = str(last['pbMRQ'])
            bs.logout()
        except: pass
    return res

def calc_full_indicators(df):
    if df.empty: return df
    for c in ['close','high','low','volume']: df[c] = df[c].astype(float)
    close, high, low = df['close'], df['high'], df['low']
    
    for n in [5,10,20,60,120,250]: df[f'MA{n}'] = close.rolling(n).mean()
    mid = df['MA20']; std = close.rolling(20).std()
    df['Upper'] = mid + 2*std; df['Lower'] = mid - 2*std
    
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['HIST'] = 2 * (df['DIF'] - df['DEA'])
    
    delta = close.diff()
    up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
    rs = up.rolling(14).mean() / (down.rolling(14).mean() + 1e-9)
    df['RSI'] = 100 - (100/(1+rs))
    
    low9 = low.rolling(9).min(); high9 = high.rolling(9).max()
    rsv = (close - low9) / (high9 - low9 + 1e-9) * 100
    df['K'] = rsv.ewm(com=2).mean(); df['D'] = df['K'].ewm(com=2).mean(); df['J'] = 3 * df['K'] - 2 * df['D']
    
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()
    
    dm_p = np.where((high.diff() > low.diff().abs()) & (high.diff()>0), high.diff(), 0)
    dm_m = np.where((low.diff().abs() > high.diff()) & (low.diff()<0), low.diff().abs(), 0)
    tr14 = tr.rolling(14).sum()
    di_p = 100 * pd.Series(dm_p).rolling(14).sum() / (tr14+1e-9)
    di_m = 100 * pd.Series(dm_m).rolling(14).sum() / (tr14+1e-9)
    df['ADX'] = (abs(di_p - di_m)/(di_p + di_m + 1e-9) * 100).rolling(14).mean()
    
    p_high = high.rolling(9).max(); p_low = low.rolling(9).min()
    df['Tenkan'] = (p_high + p_low) / 2
    p_high26 = high.rolling(26).max(); p_low26 = low.rolling(26).min()
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
    start_date = df.loc[low_idx, 'date']; start_price = df.loc[low_idx, 'low']
    gann = {}
    days = (df['date'] - start_date).dt.days
    step = df['ATR14'].iloc[-1] * 0.5
    if step == 0: step = start_price * 0.01
    gann['1x1'] = start_price + days * step
    gann['1x2'] = start_price + days * step * 0.5
    gann['2x1'] = start_price + days * step * 2.0
    recent = df.tail(120)
    h = recent['high'].max(); l = recent['low'].min(); diff = h - l
    fib = {'0.236': h-diff*0.236, '0.382': h-diff*0.382, '0.5': h-diff*0.5, '0.618': h-diff*0.618}
    return gann, fib

def analyze_signals(df):
    curr = df.iloc[-1]; prev = df.iloc[-2]; score = 0; reasons = []
    if curr['MA5'] > curr['MA20']: score += 2; reasons.append("✅ 短线多头 (MA5>MA20)")
    else: score -= 2; reasons.append("❌ 短线空头 (MA5<MA20)")
    if curr['close'] > curr['MA60']: score += 1; reasons.append("✅ 站上生命线 (MA60)")
    if curr['DIF'] > curr['DEA'] and curr['HIST'] > prev['HIST']: score += 1; reasons.append("✅ MACD金叉增强")
    elif curr['DIF'] < curr['DEA']: score -= 1; reasons.append("❌ MACD死叉")
    if curr['RSI'] < 20: score += 2; reasons.append("📉 RSI极度超卖")
    elif curr['RSI'] > 80: score -= 2; reasons.append("📈 RSI极度超买")
    if curr['close'] > max(curr['SpanA'], curr['SpanB']): score += 1; reasons.append("☁️ 云层上方")
    if curr['ADX'] > 25: reasons.append("🔥 趋势强劲 (ADX>25)")
    if curr['VolRatio'] > 1.5: score += 1; reasons.append("🌊 今日放量")
    if score >= 4: action, color = "买入", "success"
    elif score >= 1: action, color = "逢低吸纳", "warning"
    elif score >= -2: action, color = "观望", "secondary"
    else: action, color = "卖出", "error"
    support = df['low'].tail(20).min(); resistance = df['high'].tail(20).max()
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
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.55, 0.1, 0.15, 0.2])
    
    # 修复点：K线配色 (红涨绿跌)
    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], 
        name='K线', increasing_line_color='#FF3B30', decreasing_line_color='#34C759'
    ), row=1, col=1)
    
    for m, c in zip(['MA5','MA20','MA60'], ['#007AFF', '#FF9500', '#AF52DE']):
        if m in df.columns: fig.add_trace(go.Scatter(x=df['date'], y=df[m], name=m, line=dict(width=1.5, color=c)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['date'], y=df['Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['Lower'], fill='tonexty', fillcolor='rgba(0,122,255,0.05)', line=dict(width=0), name='BOLL'), row=1, col=1)
    
    gann, fib = get_drawing_lines(df)
    if show_gann:
        for k, v in gann.items(): fig.add_trace(go.Scatter(x=df['date'], y=v, mode='lines', line=dict(dash='dot', width=1, color='gray'), name=f'Gann {k}'), row=1, col=1)
    if show_fib:
        for k, v in fib.items(): fig.add_hline(y=v, line_dash="dash", line_color="orange", annotation_text=f"Fib {k}", row=1, col=1)
    if show_chanlun:
        tops = df[df['Fractal_Top']]; bots = df[df['Fractal_Bot']]
        fig.add_trace(go.Scatter(x=tops['date'], y=tops['high'], mode='markers', marker_symbol='triangle-down', marker_color='#34C759', marker_size=8, name='顶'), row=1, col=1)
        fig.add_trace(go.Scatter(x=bots['date'], y=bots['low'], mode='markers', marker_symbol='triangle-up', marker_color='#FF3B30', marker_size=8, name='底'), row=1, col=1)

    colors = ['#FF3B30' if c>=o else '#34C759' for c,o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df['date'], y=df['volume'], marker_color=colors, name='Vol'), row=2, col=1)
    fig.add_trace(go.Bar(x=df['date'], y=df['HIST'], marker_color=colors, name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['DIF'], line=dict(color='#007AFF', width=1), name='DIF'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['DEA'], line=dict(color='#FF9500', width=1), name='DEA'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name='K', line=dict(color='#007AFF', width=1)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name='D', line=dict(color='#FF9500', width=1)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['J'], name='J', line=dict(color='#AF52DE', width=1)), row=4, col=1)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#1d1d1f')),
        xaxis_rangeslider_visible=False, 
        height=900, 
        margin=dict(t=40, l=10, r=10, b=10),
        paper_bgcolor='#ffffff', 
        plot_bgcolor='#ffffff',
        font=dict(color='#1d1d1f'),
        grid=dict(rows=1, columns=1),
        xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor='#e5e5e5'),
        yaxis=dict(showgrid=True, gridcolor='#f5f5f5', zeroline=False),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. 主程序
# ==========================================
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    st.markdown("<br><br><h1 style='text-align:center'>AlphaQuant Pro</h1>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        tab1, tab2 = st.tabs(["登录", "注册"])
        with tab1:
            u = st.text_input("账号")
            p = st.text_input("密码", type="password")
            # 修复：移除 kind="secondary" 等参数，使用默认按钮，100% 兼容
            if st.button("🚀 登录", use_container_width=True):
                if verify_login(u.strip(), p):
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = u.strip()
                    st.session_state.paid_code = ""
                    st.rerun()
                else: st.error("账号或密码错误")
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
    st.markdown("<div class='brand-logo'>AlphaQuant <span>Pro</span></div>", unsafe_allow_html=True)
    
    if is_admin:
        st.success(f"👑 管理员已登录")
        with st.expander("👮‍♂️ 用户数据管理", expanded=True):
            df_u = load_users()
            st.markdown("### 💿 数据备份")
            csv = df_u.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ 下载数据", data=csv, file_name="users.csv", mime="text/csv", use_container_width=True)
            
            uploaded_file = st.file_uploader("⬆️ 恢复数据", type="csv")
            if uploaded_file is not None:
                try:
                    df_new = pd.read_csv(uploaded_file)
                    df_new.to_csv(DB_FILE, index=False)
                    st.success("恢复成功！")
                    time.sleep(1); st.rerun()
                except: st.error("文件错误")

            st.divider()
            st.markdown("### 👥 用户列表")
            st.dataframe(df_u[["username","quota"]], hide_index=True, use_container_width=True)
            
            u_list = [x for x in df_u["username"] if x != ADMIN_USER]
            if u_list:
                target = st.selectbox("管理用户", u_list)
                val = st.number_input("积分", value=50, step=10)
                if st.button("保存设置", use_container_width=True):
                    update_user_quota(target, val); st.success("已保存"); time.sleep(0.5); st.rerun()
                if st.button("删除用户", use_container_width=True):
                    delete_user(target); st.success("已删除"); time.sleep(0.5); st.rerun()
    else:
        st.info(f"👤 交易员: {user}")
        df_u = load_users()
        try: q = df_u[df_u["username"]==user]["quota"].iloc[0]
        except: q = 0
        st.metric("剩余积分", q)

    st.divider()
    try: def_tok = st.secrets["TUSHARE_TOKEN"]
    except: def_tok = ""
    token = st.text_input("Tushare Token", value=def_tok, type="password")
    
    if "code" not in st.session_state: st.session_state.code = "600519"
    new_code = st.text_input("股票代码", st.session_state.code)
    
    if "paid_code" not in st.session_state: st.session_state.paid_code = ""
    if new_code != st.session_state.code:
        st.session_state.code = new_code
        st.session_state.paid_code = "" 
        st.rerun()
    
    name = get_name(st.session_state.code, token)
    
    days = st.radio("分析周期", [7, 30, 60, 90, 180, 250, 360], index=2, horizontal=True)
    adjust = st.selectbox("复权模式", ["qfq", "hfq", ""], 0)
    
    st.divider()
    st.markdown("### 视图选项")
    show_gann = st.checkbox("江恩角度线", True)
    show_fib = st.checkbox("斐波那契回撤", True)
    show_chanlun = st.checkbox("缠论结构", True)
    st.divider()
    if st.button("退出登录", use_container_width=True): st.session_state["logged_in"] = False; st.rerun()

c1, c2 = st.columns([3, 1])
with c1: st.title(f"📈 {name} ({st.session_state.code})")

is_paid = (st.session_state.code == st.session_state.paid_code)

if not is_paid:
    st.info("🔒 深度分析数据已锁定")
    if st.button(f"使用 1 积分解锁分析", type="primary", use_container_width=True):
        if consume_quota(user):
            st.session_state.paid_code = st.session_state.code
            with st.spinner("正在进行 AI 计算..."):
                time.sleep(0.8)
            st.rerun()
        else: st.error("积分不足，请联系管理员充值")
    st.stop()

with c2:
    if st.button("刷新数据", use_container_width=True): st.cache_data.clear(); st.rerun()

with st.spinner("正在加载最新数据..."):
    df = get_data(st.session_state.code, token, days, adjust) 
    funda = get_fundamentals(st.session_state.code, token)

if df.empty:
    st.error("无法获取数据，请检查代码或稍后再试。")
else:
    df = calc_full_indicators(df)
    df = detect_patterns(df)
    
    trend_txt, trend_col = main_uptrend_check(df)
    if trend_col == "success": st.success(f"### {trend_txt}")
    elif trend_col == "warning": st.warning(f"### {trend_txt}")
    else: st.error(f"### {trend_txt}")
    
    plot_df = df.tail(days).copy() 
    latest = df.iloc[-1]
    
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("现价", f"{latest['close']:.2f}", f"{latest['pct_change']:.2f}%")
    k2.metric("PE (TTM)", funda['pe'])
    k3.metric("RSI (14)", f"{latest['RSI']:.1f}")
    k4.metric("ADX 趋势", f"{latest['ADX']:.1f}")
    k5.metric("量比", f"{latest['VolRatio']:.2f}")
    
    plot_full_chart(plot_df, f"{name} 技术分析概览", show_gann, show_fib, show_chanlun)
    
    res = analyze_signals(df)
    st.subheader(f"🤖 AI 决策建议: {res['action']} (评分: {res['score']})")
    
    s1, s2, s3 = st.columns(3)
    if res['color'] == 'success': s1.success(f"建议仓位: 高 (50%~80%)")
    elif res['color'] == 'warning': s1.warning(f"建议仓位: 中 (20%~50%)")
    else: s1.error(f"建议仓位: 低/空仓 (0%~20%)")
    
    s2.info(f"🛡️ 止损参考: {res['sl']:.2f}")
    s3.info(f"💰 止盈参考: {res['tp']:.2f}")
    st.caption(f"📍 关键位 | 支撑: **{res['sup']:.2f}** | 压力: **{res['res']:.2f}**")
    
    with st.expander("查看详细分析逻辑", expanded=True):
        for r in res['reasons']: st.write(r)
