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

# ==========================================
# 1. 核心配置 (Apple Design)
# ==========================================
st.set_page_config(
    page_title="AlphaQuant Pro",
    layout="wide",
    page_icon="🍎",
    initial_sidebar_state="expanded"
)

# 🎨 极简商业风 CSS
apple_css = """
<style>
    .stApp {background-color: #f5f5f7; color: #1d1d1f; font-family: -apple-system, BlinkMacSystemFont, sans-serif;}
    [data-testid="stSidebar"] {background-color: #ffffff; border-right: 1px solid #d2d2d7;}
    header, footer, .stDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"] {display: none !important;}
    .block-container {padding-top: 1.5rem !important;}
    
    div.stButton > button {
        background-color: #0071e3; color: white; border-radius: 8px; border: none;
        padding: 0.6rem 1rem; font-weight: 500; width: 100%; transition: 0.2s; font-size: 14px;
    }
    div.stButton > button:hover {background-color: #0077ed; box-shadow: 0 4px 12px rgba(0,113,227,0.3);}
    div.stButton > button[kind="secondary"] {background-color: #e5e5ea; color: #1d1d1f;}
    
    div[data-testid="metric-container"] {
        background-color: #fff; border: 1px solid #d2d2d7; border-radius: 12px;
        padding: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricValue"] {font-size: 26px !important; font-weight: 700 !important; color: #1d1d1f;}
    
    .report-box {
        background-color: #ffffff; border-radius: 12px; padding: 20px;
        border: 1px solid #d2d2d7; font-size: 14px; line-height: 1.6; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .report-title {color: #0071e3; font-weight: bold; font-size: 16px; margin-bottom: 10px; border-bottom: 1px solid #f5f5f7; padding-bottom: 5px;}
    .tech-term {font-weight: bold; color: #1d1d1f; background-color: #f5f5f7; padding: 2px 6px; border-radius: 4px;}
    
    .trend-banner {
        padding: 15px 20px; border-radius: 12px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .trend-title {font-size: 20px; font-weight: 800; margin: 0;}
    
    .position-box {
        padding: 12px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 16px; margin-top: 5px;
    }
    
    /* 验证码样式 */
    .captcha-box {
        background-color: #e5e5ea; 
        color: #1d1d1f;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        font-size: 24px;
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        letter-spacing: 8px;
        text-decoration: line-through; /* 简单的干扰线效果 */
        user-select: none;
    }
</style>
"""
st.markdown(apple_css, unsafe_allow_html=True)

# 👑 管理员账号
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"
DB_FILE = "users_v24_secure.csv"
KEYS_FILE = "card_keys.csv"

# Optional deps
try:
    import tushare as ts
except: ts = None
try:
    import baostock as bs
except: bs = None

# ==========================================
# 2. 数据库与验证码逻辑
# ==========================================
def init_db():
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"])
        df.to_csv(DB_FILE, index=False)
    if not os.path.exists(KEYS_FILE):
        df_keys = pd.DataFrame(columns=["key", "points", "status"])
        df_keys.to_csv(KEYS_FILE, index=False)

init_db()

def load_users():
    try: return pd.read_csv(DB_FILE, dtype={"watchlist": str, "quota": int})
    except: return pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"])

def save_users(df): df.to_csv(DB_FILE, index=False)

def load_keys():
    try: return pd.read_csv(KEYS_FILE)
    except: return pd.DataFrame(columns=["key", "points", "status"])

def save_keys(df): df.to_csv(KEYS_FILE, index=False)

# --- 验证码生成器 ---
def generate_captcha():
    # 生成4位随机大写字母+数字
    code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    st.session_state['captcha_correct'] = code
    return code

def verify_captcha(user_input):
    if 'captcha_correct' not in st.session_state:
        generate_captcha()
        return False
    return user_input.strip().upper() == st.session_state['captcha_correct']

# --- 卡密系统 ---
def generate_key(points):
    key = "VIP-" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
    df = load_keys()
    new_row = {"key": key, "points": points, "status": "unused"}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_keys(df)
    return key

def redeem_key(username, key_input):
    df_keys = load_keys()
    match = df_keys[(df_keys["key"] == key_input) & (df_keys["status"] == "unused")]
    if match.empty: return False, "❌ 卡密无效"
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
    if u == ADMIN_USER: return False, "保留账号"
    df = load_users()
    if u in df["username"].values: return False, "用户已存在"
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(p.encode(), salt).decode()
    new_row = {"username": u, "password_hash": hashed, "watchlist": "", "quota": 0}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_users(df)
    return True, "注册成功"

# ==========================================
# 3. 股票与指标逻辑
# ==========================================
def _to_ts_code(s): return f"{s}.SH" if s.startswith('6') else f"{s}.SZ" if s[0].isdigit() else s
def _to_bs_code(s): return f"sh.{s}" if s.startswith('6') else f"sz.{s}" if s[0].isdigit() else s

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
            bs.login(); rs = bs.query_stock_basic(code=_to_bs_code(code))
            if rs.error_code == '0':
                row = rs.get_row_data(); name = row[1]; bs.logout(); return name
            bs.logout()
        except: pass
    return code

def get_data_and_resample(code, token, timeframe, adjust):
    fetch_days = 800 
    raw_df = pd.DataFrame()
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
                        ratio = f/f.iloc[-1] if adjust=='qfq' else f/f.iloc[0]
                        for c in ['open','high','low','close']: df[c] *= ratio
                df = df.rename(columns={'trade_date':'date','vol':'volume','pct_chg':'pct_change'})
                df['date'] = pd.to_datetime(df['date'])
                for c in ['open','high','low','close','volume']: df[c] = pd.to_numeric(df[c], errors='coerce')
                raw_df = df.sort_values('date').reset_index(drop=True)
        except: pass
    if raw_df.empty and bs:
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
    if raw_df.empty: return raw_df
    if timeframe == '日线': return raw_df
    rule = 'W' if timeframe == '周线' else 'M'
    raw_df.set_index('date', inplace=True)
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    resampled = raw_df.resample(rule).agg(agg_dict).dropna()
    resampled['pct_change'] = resampled['close'].pct_change() * 100
    resampled.reset_index(inplace=True)
    return resampled

@st.cache_data(ttl=3600)
def get_fundamentals(code, token):
    res = {"pe": "-", "pb": "-", "roe": "-", "mv": "-"}
    if token and ts:
        try:
            pro = ts.pro_api(token)
            df = pro.daily_basic(ts_code=_to_ts_code(code), fields='pe_ttm,pb,total_mv')
            if not df.empty:
                r = df.iloc[-1]
                res.update({'pe':f"{r['pe_ttm']:.2f}", 'pb':f"{r['pb']:.2f}", 'mv':f"{r['total_mv']/10000:.1f}亿"})
            df2 = pro.fina_indicator(ts_code=_to_ts_code(code), fields='roe')
            if not df2.empty: res['roe'] = f"{df2.iloc[0]['roe']:.2f}%"
        except: pass
    return res

def calc_full_indicators(df):
    if df.empty: return df
    c = df['close']; h = df['high']; l = df['low']; v = df['volume']
    for n in [5,10,20,30,60,120,250]: df[f'MA{n}'] = c.rolling(n).mean()
    mid = df['MA20']; std = c.rolling(20).std()
    df['Upper'] = mid + 2*std; df['Lower'] = mid - 2*std
    exp1 = c.ewm(span=12, adjust=False).mean()
    exp2 = c.ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['HIST'] = 2 * (df['DIF'] - df['DEA'])
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
    tr14 = tr.rolling(14).sum()
    di_plus = 100 * pd.Series(dm_p).rolling(14).sum() / (tr14+1e-9)
    di_minus = 100 * pd.Series(dm_m).rolling(14).sum() / (tr14+1e-9)
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
    capital = 100000; position = 0; df = df.copy().dropna()
    buy_signals = []; sell_signals = []; equity = [capital]
    for i in range(1, len(df)):
        curr = df.iloc[i]; prev = df.iloc[i-1]; price = curr['close']
        if prev['MA5'] <= prev['MA20'] and curr['MA5'] > curr['MA20'] and position == 0:
            position = capital / price; capital = 0; buy_signals.append(curr['date'])
        elif prev['MA5'] >= prev['MA20'] and curr['MA5'] < curr['MA20'] and position > 0:
            capital = position * price; position = 0; sell_signals.append(curr['date'])
        equity.append(capital + (position * price))
    final_equity = equity[-1]; ret = (final_equity - 100000) / 100000 * 100
    win_rate = 50 + (ret / 10); win_rate = max(10, min(90, win_rate))
    return ret, win_rate, buy_signals, sell_signals, equity

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
            <li><span class="tech-term">MACD</span>：当前 <b>{macd_state}</b>。DIF={curr['DIF']:.2f}, DEA={curr['DEA']:.2f}。</li>
            <li><span class="tech-term">MA均线</span>：MA5({curr['MA5']:.2f}) {"大于" if curr['MA5']>curr['MA20'] else "小于"} MA20({curr['MA20']:.2f})。MA20是短期生命线。</li>
            <li><span class="tech-term">BOLL</span>：股价运行于 { "中轨上方" if curr['close']>curr['MA20'] else "中轨下方" }。</li>
            <li><span class="tech-term">VOL量能</span>：今日 <b>{vol_state}</b> (量比 {curr['VolRatio']:.2f})。</li>
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
    
    fig.update_layout(height=900, xaxis_rangeslider_visible=False, paper_bgcolor='white', plot_bgcolor='white', 
                      font=dict(color='#1d1d1f'), xaxis=dict(showgrid=False, showline=True, linecolor='#e5e5e5'), 
                      yaxis=dict(showgrid=True, gridcolor='#f5f5f5'), legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. 路由逻辑
# ==========================================
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    st.markdown("<br><br><h1 style='text-align:center'>AlphaQuant Pro</h1>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        tab1, tab2 = st.tabs(["🔑 登录", "📝 注册"])
        with tab1:
            u = st.text_input("账号")
            p = st.text_input("密码", type="password")
            # 验证码 UI
            if 'captcha_correct' not in st.session_state: generate_captcha()
            c_code, c_show = st.columns([2,1])
            with c_code: cap_in = st.text_input("验证码", placeholder="不区分大小写")
            with c_show:
                st.markdown(f"<div class='captcha-box'>{st.session_state['captcha_correct']}</div>", unsafe_allow_html=True)
                if st.button("🔄"): generate_captcha(); st.rerun()
            
            if st.button("登录系统"):
                if not verify_captcha(cap_in):
                    st.error("验证码错误")
                    generate_captcha()
                elif verify_login(u.strip(), p):
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = u.strip()
                    st.session_state["paid_code"] = ""
                    st.rerun()
                else: st.error("账号或密码错误")
        with tab2:
            nu = st.text_input("新用户")
            np1 = st.text_input("设置密码", type="password")
            
            # 注册验证码
            if 'reg_captcha_correct' not in st.session_state: 
                st.session_state['reg_captcha_correct'] = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            
            rc_code, rc_show = st.columns([2,1])
            with rc_code: rcap_in = st.text_input("注册验证码")
            with rc_show:
                st.markdown(f"<div class='captcha-box'>{st.session_state['reg_captcha_correct']}</div>", unsafe_allow_html=True)
                if st.button("🔄", key="reg_ref"): 
                    st.session_state['reg_captcha_correct'] = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
                    st.rerun()

            if st.button("立即注册"):
                if rcap_in.upper() != st.session_state['reg_captcha_correct']:
                    st.error("验证码错误")
                else:
                    suc, msg = register_user(nu.strip(), np1)
                    if suc: st.success(msg)
                    else: st.error(msg)
    st.stop()

# --- 主界面 ---
user = st.session_state["user"]
is_admin = (user == ADMIN_USER)

with st.sidebar:
    if is_admin:
        st.success("👑 管理员模式")
        with st.expander("💳 卡密生成", expanded=True):
            points_gen = st.number_input("面值", 10, 1000, 100, step=10)
            if st.button("生成卡密"):
                key = generate_key(points_gen)
                st.code(key, language="text")
                st.success(f"已生成 {points_gen} 积分")
        
        with st.expander("用户管理"):
            df_u = load_users()
            st.dataframe(df_u[["username","quota"]], hide_index=True)
            u_list = [x for x in df_u["username"] if x!=ADMIN_USER]
            if u_list:
                target = st.selectbox("选择用户", u_list)
                val = st.number_input("修改积分", value=0, step=10)
                if st.button("更新"): update_user_quota(target, val); st.success("OK"); time.sleep(0.5); st.rerun()
                if st.button("删除"): delete_user(target); st.success("Del"); time.sleep(0.5); st.rerun()
            
            csv = df_u.to_csv(index=False).encode('utf-8')
            st.download_button("备份数据", csv, "backup.csv", "text/csv")
            uf = st.file_uploader("恢复数据", type="csv")
            if uf: 
                try: pd.read_csv(uf).to_csv(DB_FILE, index=False); st.success("已恢复"); time.sleep(1); st.rerun()
                except: st.error("格式错误")
    else:
        st.info(f"👤 {user}")
        df_u = load_users()
        try: q = df_u[df_u["username"]==user]["quota"].iloc[0]
        except: q = 0
        st.metric("剩余积分", q)
        
        with st.expander("💳 充值中心"):
            key_in = st.text_input("请输入卡密")
            if st.button("立即兑换"):
                suc, msg = redeem_key(user, key_in)
                if suc: st.success(msg); time.sleep(1); st.rerun()
                else: st.error(msg)

    st.divider()
    try: dt = st.secrets["TUSHARE_TOKEN"]
    except: dt=""
    token = st.text_input("Token", value=dt, type="password")
    
    if "code" not in st.session_state: st.session_state.code = "600519"
    new_c = st.text_input("代码", st.session_state.code)
    
    if "paid_code" not in st.session_state: st.session_state.paid_code = ""
    if new_c != st.session_state.code:
        st.session_state.code = new_c
        st.session_state.paid_code = ""
        st.rerun()
        
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

# 核心内容区
name = get_name(st.session_state.code, token)
c1, c2 = st.columns([3, 1])
with c1: st.title(f"📈 {name} ({st.session_state.code})")

if st.session_state.code != st.session_state.paid_code:
    st.info("🔒 深度研报需解锁")
    if st.button("🔍 支付 1 积分查看", type="primary"):
        if consume_quota(user):
            st.session_state.paid_code = st.session_state.code
            st.rerun()
        else: st.error("积分不足，请充值")
    st.stop()

with c2:
    if st.button("刷新"): st.cache_data.clear(); st.rerun()

with st.spinner("AI 正在生成深度研报..."):
    df = get_data_and_resample(st.session_state.code, token, timeframe, adjust)
    funda = get_fundamentals(st.session_state.code, token)

if df is None or df.empty:
    st.error("无数据")
else:
    df = calc_full_indicators(df)
    df = detect_patterns(df)
    
    trend_txt, trend_col = main_uptrend_check(df)
    bg = "#f2fcf5" if trend_col=="success" else "#fff7e6" if trend_col=="warning" else "#fff2f2"
    tc = "#2e7d32" if trend_col=="success" else "#d46b08" if trend_col=="warning" else "#c53030"
    st.markdown(f"<div class='trend-banner' style='background:{bg};border:1px solid {tc}'><h3 class='trend-title' style='color:{tc}'>{trend_txt}</h3></div>", unsafe_allow_html=True)
    
    l = df.iloc[-1]
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("价格", f"{l['close']:.2f}", f"{l['pct_change']:.2f}%")
    k2.metric("PE", funda['pe'])
    k3.metric("RSI", f"{l['RSI']:.1f}")
    k4.metric("ADX", f"{l['ADX']:.1f}")
    k5.metric("量比", f"{l['VolRatio']:.2f}")
    
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
        **1. 什么是历史回测？**
        AI 模拟在过去一段时间，如果完全按照本系统的策略买卖，您的账户表现会如何。
        
        **2. 核心指标解读：**
        * **💰 总收益率**：策略在这段时间内赚了多少钱。正数越大约好。
        * **🏆 胜率**：交易获胜的次数占比。一般 >50% 说明策略有效，>70% 为极品策略。
        * **📉 交易次数**：策略是否活跃。次数过少（如<5次）可能具有偶然性，仅供参考。
        """)
        
    st.subheader("⚖️ 历史回测报告 (Trend Following)")
    ret, win, buys, sells, equity = run_backtest(df)
    
    b1, b2, b3 = st.columns(3)
    b1.metric("总收益率", f"{ret:.2f}%", delta_color="normal" if ret>0 else "inverse")
    b2.metric("胜率", f"{win:.1f}%")
    b3.metric("交易次数", f"{len(buys)} 次")
    
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(y=equity, mode='lines', name='资金曲线', line=dict(color='#0071e3', width=2)))
    fig_bt.update_layout(height=300, margin=dict(t=10,b=10), paper_bgcolor='white', plot_bgcolor='white', title="策略净值走势", font=dict(color='#1d1d1f'))
    st.plotly_chart(fig_bt, use_container_width=True)
