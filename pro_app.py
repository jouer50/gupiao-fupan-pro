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

# ✅ 0. 依赖库检查
try:
    import yfinance as yf
except ImportError:
    st.error("🚨 严重错误：缺少 `yfinance` 库")
    st.info("请在 GitHub 仓库的 requirements.txt 文件中添加一行：yfinance")
    st.stop()

# ==========================================
# 1. 核心配置 (Apple Design)
# ==========================================
st.set_page_config(
    page_title="AlphaQuant Pro",
    layout="wide",
    page_icon="🍎",
    initial_sidebar_state="expanded"
)

apple_css = """
<style>
    .stApp {background-color: #f5f5f7; color: #1d1d1f; font-family: -apple-system, BlinkMacSystemFont, sans-serif;}
    [data-testid="stSidebar"] {background-color: #ffffff; border-right: 1px solid #d2d2d7;}
    header, footer, .stDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"] {display: none !important;}
    .block-container {padding-top: 1.5rem !important;}
    div.stButton > button {background-color: #0071e3; color: white; border-radius: 8px; border: none; padding: 0.6rem 1rem; font-weight: 500; width: 100%;}
    div.stButton > button:hover {background-color: #0077ed; box-shadow: 0 4px 12px rgba(0,113,227,0.3);}
    div[data-testid="metric-container"] {background-color: #fff; border: 1px solid #d2d2d7; border-radius: 12px; padding: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);}
    [data-testid="stMetricValue"] {font-size: 26px !important; font-weight: 700 !important; color: #1d1d1f;}
    .report-box {background-color: #ffffff; border-radius: 12px; padding: 20px; border: 1px solid #d2d2d7; font-size: 14px; line-height: 1.6; box-shadow: 0 2px 8px rgba(0,0,0,0.04);}
    .report-title {color: #0071e3; font-weight: bold; font-size: 16px; margin-bottom: 10px; border-bottom: 1px solid #f5f5f7; padding-bottom: 5px;}
    .tech-term {font-weight: bold; color: #1d1d1f; background-color: #f5f5f7; padding: 2px 6px; border-radius: 4px;}
    .trend-banner {padding: 15px 20px; border-radius: 12px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 12px rgba(0,0,0,0.05);}
    .trend-title {font-size: 20px; font-weight: 800; margin: 0;}
    .position-box {padding: 12px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 16px; margin-top: 5px;}
    .captcha-box {background-color: #e5e5ea; color: #1d1d1f; font-family: monospace; font-weight: bold; font-size: 24px; text-align: center; padding: 10px; border-radius: 8px; letter-spacing: 8px; text-decoration: line-through; user-select: none;}
</style>
"""
st.markdown(apple_css, unsafe_allow_html=True)

# 👑 全局常量
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"
DB_FILE = "users_v26_3.csv"
KEYS_FILE = "card_keys.csv"

# Optional deps
try:
    import tushare as ts
except: ts = None
try:
    import baostock as bs
except: bs = None

# ==========================================
# 2. 核心工具函数 (含安全格式化)
# ==========================================
def init_db():
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"])
        df.to_csv(DB_FILE, index=False)
    if not os.path.exists(KEYS_FILE):
        df_keys = pd.DataFrame(columns=["key", "points", "status"])
        df_keys.to_csv(KEYS_FILE, index=False)

# ✅ 核心修复：安全格式化函数，防止报错
def safe_fmt(value, fmt="{:.2f}", default="-"):
    try:
        if value is None or value == "" or value == "N/A":
            return default
        # 尝试转为浮点数
        f_val = float(value)
        if np.isnan(f_val):
            return default
        return fmt.format(f_val)
    except:
        return default

def generate_captcha():
    code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    st.session_state['captcha_correct'] = code
    return code

def verify_captcha(user_input):
    if 'captcha_correct' not in st.session_state: 
        generate_captcha()
        return False
    return user_input.strip().upper() == st.session_state['captcha_correct']

def load_users():
    try: return pd.read_csv(DB_FILE, dtype={"watchlist": str, "quota": int})
    except: return pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"])

def save_users(df): df.to_csv(DB_FILE, index=False)

def load_keys():
    try: return pd.read_csv(KEYS_FILE)
    except: return pd.DataFrame(columns=["key", "points", "status"])

def save_keys(df): df.to_csv(KEYS_FILE, index=False)

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
# 3. 智能股票逻辑
# ==========================================
def is_cn_stock(code):
    return code.isdigit() and len(code) == 6

def _to_ts_code(s): return f"{s}.SH" if s.startswith('6') else f"{s}.SZ" if s[0].isdigit() else s
def _to_bs_code(s): return f"sh.{s}" if s.startswith('6') else f"sz.{s}" if s[0].isdigit() else s

def process_ticker(code):
    code = code.strip().upper()
    if code.isdigit() and len(code) < 6: return f"{code.zfill(4)}.HK"
    return code

@st.cache_data(ttl=3600)
def get_name(code, token, proxy=None):
    code = process_ticker(code)
    # 美股/港股
    if not is_cn_stock(code):
        try:
            if proxy: 
                os.environ["HTTP_PROXY"] = proxy
                os.environ["HTTPS_PROXY"] = proxy
            t = yf.Ticker(code)
            return t.info.get('shortName') or t.info.get('longName') or code
        except: return code
    # A股
    if token and ts:
        try:
            ts.set_token(token)
            pro = ts.pro_api()
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

def get_data_and_resample(code, token, timeframe, adjust, proxy=None):
    code = process_ticker(code)
    fetch_days = 1500 
    raw_df = pd.DataFrame()
    
    if not is_cn_stock(code):
        try:
            if proxy: 
                os.environ["HTTP_PROXY"] = proxy
                os.environ["HTTPS_PROXY"] = proxy
            
            yf_df = yf.download(code, period="5y", interval="1d", progress=False, auto_adjust=False)
            
            if not yf_df.empty:
                if isinstance(yf_df.columns, pd.MultiIndex):
                    yf_df.columns = yf_df.columns.get_level_values(0)
                
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
                    for c in ['open','high','low','close','volume']:
                        raw_df[c] = pd.to_numeric(raw_df[c], errors='coerce')
                    raw_df['pct_change'] = raw_df['close'].pct_change() * 100
        except Exception as e:
            st.error(f"全球数据源错误: {e}")
            
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
            bs.login()
            e = pd.Timestamp.today().strftime('%Y-%m-%d')
            s = (pd.Timestamp.today() - pd.Timedelta(days=fetch_days)).strftime('%Y-%m-%d')
            flag = "2" if adjust=='qfq' else "1" if adjust=='hfq' else "3"
            rs = bs.query_history_k_data_plus(_to_bs_code(code), "date,open,high,low,close,volume,pctChg", start_date=s, end_date=e, frequency="d", adjustflag=flag)
            data = rs.get_data(); bs.logout()
            if not data.empty:
                df = data.rename(columns={'pctChg':'pct_change'})
                df['date'] = pd.to_datetime(df['date'])
                for c in ['open','high','low','close','volume']: df[c] = pd.to_numeric(df[c], errors='coerce')
                raw_df = df.sort_values('date').reset_index(drop=True)

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
    
    # ✅ 修复点：使用 safe_fmt 转换数据
    if not is_cn_stock(code):
        try:
            t = yf.Ticker(code)
            i = t.info
            pe = i.get('trailingPE')
            pb = i.get('priceToBook')
            mk = i.get('marketCap')
            
            res['pe'] = safe_fmt(pe)
            res['pb'] = safe_fmt(pb)
            res['mv'] = f"{mk/100000000:.2f}亿" if mk else "-"
        except: pass
        return res

    if token:
        try:
            ts.set_token(token)
            pro = ts.pro_api()
            df = pro.daily_basic(ts_code=_to_ts_code(code), fields='pe_ttm,pb,total_mv')
            if not df.empty:
                r = df.iloc[-1]
                res['pe'] = safe_fmt(r['pe_ttm'])
                res['pb'] = safe_fmt(r['pb'])
                res['mv'] = f"{r['total_mv']/10000:.1f}亿" if r['total_mv'] else "-"
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
    
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df['DIF'] = e12 - e26
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
    
    fig.update_layout(height=900, xaxis_rangeslider_visible=False, paper_bgcolor='white', plot_bgcolor='white', 
                      font=dict(color='#1d1d1f'), xaxis=dict(showgrid=False, showline=True, linecolor='#e5e5e5'), 
                      yaxis=dict(showgrid=True, gridcolor='#f5f5f5'), legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"❌ 系统发生错误: {e}")
    # st.code(traceback.format_exc()) # 调试用
