import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import bcrypt
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
    .stApp {background-color: #ffffff; color: #1d1d1f; font-family: -apple-system, sans-serif;}
    [data-testid="stSidebar"] {background-color: #f5f5f7; border-right: 1px solid #e5e5e5;}
    header, footer, .stDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"] {display: none !important;}
    .block-container {padding-top: 1rem !important;}
    
    div.stButton > button {
        background-color: #0071e3; color: white; border-radius: 8px; border: none;
        padding: 0.5rem 1rem; font-weight: 500; width: 100%; transition: 0.2s;
    }
    div.stButton > button:hover {background-color: #0077ed; box-shadow: 0 2px 8px rgba(0,113,227,0.2);}
    div.stButton > button[kind="secondary"] {background-color: #e5e5ea; color: #1d1d1f;}
    
    div[data-testid="metric-container"] {
        background-color: #fff; border: 1px solid #d2d2d7; border-radius: 10px;
        padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    [data-testid="stMetricValue"] {font-size: 24px !important; font-weight: 600 !important;}
    
    .report-box {
        background-color: #f5f5f7; border-radius: 12px; padding: 20px;
        border: 1px solid #e5e5e5; font-size: 14px; line-height: 1.6;
    }
    .report-title {color: #0071e3; font-weight: bold; font-size: 16px; margin-bottom: 8px;}
    .tech-term {font-weight: bold; color: #1d1d1f;}
</style>
"""
st.markdown(apple_css, unsafe_allow_html=True)

# 👑 管理员账号
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"
DB_FILE = "users_v13_clean.csv" # 再次升级文件名，强制刷新环境

# Optional deps
try:
    import tushare as ts
except: ts = None
try:
    import baostock as bs
except: bs = None

# ==========================================
# 2. 数据库逻辑
# ==========================================
def init_db():
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"])
        df.to_csv(DB_FILE, index=False)

init_db()

def load_users():
    try: return pd.read_csv(DB_FILE, dtype={"watchlist": str, "quota": int})
    except: return pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"])

def save_users(df): df.to_csv(DB_FILE, index=False)

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
    new_row = {"username": u, "password_hash": hashed, "watchlist": "", "quota": 0} # 默认0积分
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_users(df)
    return True, "注册成功，请联系管理员充值"

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

@st.cache_data(ttl=3600)
def get_data(code, token, days, adjust):
    fetch_days = max(400, days + 100)
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
                return df.sort_values('date').reset_index(drop=True)
        except: pass
    if bs:
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
            return df.sort_values('date').reset_index(drop=True)
    return pd.DataFrame()

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
    if res['pe']=="-" and bs:
        try:
            bs.login()
            import datetime
            e = datetime.date.today().strftime("%Y-%m-%d")
            s = (datetime.date.today()-datetime.timedelta(days=10)).strftime("%Y-%m-%d")
            rs = bs.query_history_k_data_plus(_to_bs_code(code), "date,peTTM,pbMRQ", start_date=s, end_date=e, frequency="d")
            rows = rs.get_data(); bs.logout()
            if not rows.empty:
                l = rows.iloc[-1]
                res['pe'] = str(l['peTTM']); res['pb'] = str(l['pbMRQ'])
        except: pass
    return res

def calc_full_indicators(df):
    if df.empty: return df
    c = df['close']; h = df['high']; l = df['low']; v = df['volume']
    for n in [5,10,20,60,120,250]: df[f'MA{n}'] = c.rolling(n).mean()
    mid = df['MA20']; std = c.rolling(20).std()
    df['Upper'] = mid + 2*std; df['Lower'] = mid - 2*std
    e12 = c.ewm(span=12).mean(); e26 = c.ewm(span=26).mean()
    df['DIF'] = e12 - e26; df['DEA'] = df['DIF'].ewm(span=9).mean(); df['HIST'] = 2*(df['DIF']-df['DEA'])
    delta = c.diff(); up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
    rs = up.rolling(14).mean()/(down.rolling(14).mean()+1e-9)
    df['RSI'] = 100 - (100/(1+rs))
    l9 = l.rolling(9).min(); h9 = h.rolling(9).max()
    rsv = (c - l9)/(h9 - l9 + 1e-9)*100
    df['K'] = rsv.ewm(com=2).mean(); df['D'] = df['K'].ewm(com=2).mean(); df['J'] = 3*df['K']-2*df['D']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()
    dp = np.where((h.diff()>l.diff().abs()) & (h.diff()>0), h.diff(), 0)
    dm = np.where((l.diff().abs()>h.diff()) & (l.diff()<0), l.diff().abs(), 0)
    tr14 = tr.rolling(14).sum()
    dip = 100*pd.Series(dp).rolling(14).sum()/(tr14+1e-9)
    dim = 100*pd.Series(dm).rolling(14).sum()/(tr14+1e-9)
    df['ADX'] = (abs(dip-dim)/(dip+dim+1e-9)*100).rolling(14).mean()
    ph = h.rolling(9).max(); pl = l.rolling(9).min()
    df['Tenkan'] = (ph+pl)/2
    ph26 = h.rolling(26).max(); pl26 = l.rolling(26).min()
    df['Kijun'] = (ph26+pl26)/2
    df['SpanA'] = ((df['Tenkan']+df['Kijun'])/2).shift(26)
    df['SpanB'] = ((h.rolling(52).max()+l.rolling(52).min())/2).shift(26)
    df['VolRatio'] = v / (v.rolling(5).mean()+1e-9)
    return df.fillna(0)

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
    fib = {k: h-d*v for k,v in {'0.236':0.236,'0.382':0.382,'0.5':0.5,'0.618':0.618}.items()}
    return gann, fib

def generate_deep_report(df, name):
    curr = df.iloc[-1]; prev = df.iloc[-2]
    
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
    except:
        fib_txt = "数据不足，无法计算斐波那契位置。"

    gann_logic = f"""
    <div class="report-box" style="margin-top:10px;">
        <div class="report-title">🌌 江恩与斐波那契时空矩阵</div>
        <span class="tech-term">江恩角度线 (Gann Fan)</span> 代表了时间与价格的平衡关系。1x1线是多空分界线。
        <br>• <b>斐波那契回撤</b>：{fib_txt}
    </div>
    """
    
    macd_state = "金叉共振" if curr['DIF']>curr['DEA'] else "死叉调整"
    vol_state = "放量" if curr['VolRatio']>1.2 else "缩量" if curr['VolRatio']<0.8 else "温和"
    
    ind_logic = f"""
    <div class="report-box" style="margin-top:10px;">
        <div class="report-title">📊 核心动能指标解析</div>
        <ul>
            <li><span class="tech-term">MACD</span>：当前 <b>{macd_state}</b>。红柱代表多头主导，绿柱代表空头主导。</li>
            <li><span class="tech-term">MA (均线)</span>：MA5({curr['MA5']:.2f}) {"大于" if curr['MA5']>curr['MA20'] else "小于"} MA20({curr['MA20']:.2f})。MA20是短期生命线。</li>
            <li><span class="tech-term">BOLL</span>：股价运行于 { "中轨上方" if curr['close']>curr['MA20'] else "中轨下方" }。</li>
            <li><span class="tech-term">VOL</span>：今日 <b>{vol_state}</b> (量比 {curr['VolRatio']:.2f})。放量上涨更可靠。</li>
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
    atr = c['ATR14']
    return score, action, color, c['close']-2*atr, c['close']+3*atr

def plot_chart(df, name, gann_show, fib_show, chan_show):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.55,0.1,0.15,0.2])
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K', increasing_line_color='#34c759', decreasing_line_color='#ff3b30'), 1, 1)
    for m, c in zip(['MA5','MA20','MA60'], ['#0071e3','#ff9500','#af52de']):
        if m in df.columns: fig.add_trace(go.Scatter(x=df['date'], y=df[m], name=m, line=dict(width=1, color=c)), 1, 1)
    
    ga, fi = get_drawing_lines(df)
    if gann_show:
        for k,v in ga.items(): fig.add_trace(go.Scatter(x=df['date'], y=v, mode='lines', line=dict(width=1, dash='dot', color='#86868b'), name=f'G{k}'), 1, 1)
    if fib_show:
        for k,v in fi.items(): fig.add_hline(y=v, line_dash='dash', line_color='#ff9500', row=1, col=1)
    if chan_show:
        tops=df[df['F_Top']]; bots=df[df['F_Bot']]
        fig.add_trace(go.Scatter(x=tops['date'], y=tops['high'], mode='markers', marker_symbol='triangle-down', marker_color='#34c759', name='Top'), 1, 1)
        fig.add_trace(go.Scatter(x=bots['date'], y=bots['low'], mode='markers', marker_symbol='triangle-up', marker_color='#ff3b30', name='Bot'), 1, 1)

    colors = ['#ff3b30' if c<o else '#34c759' for c,o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df['date'], y=df['volume'], marker_color=colors), 2, 1)
    fig.add_trace(go.Bar(x=df['date'], y=df['HIST'], marker_color=colors), 3, 1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['DIF'], line=dict(color='#0071e3', width=1)), 3, 1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['DEA'], line=dict(color='#ff9500', width=1)), 3, 1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['K'], line=dict(color='#0071e3', width=1)), 4, 1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['D'], line=dict(color='#ff9500', width=1)), 4, 1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['J'], line=dict(color='#af52de', width=1)), 4, 1)
    
    fig.update_layout(height=900, xaxis_rangeslider_visible=False, paper_bgcolor='white', plot_bgcolor='white', 
                      font=dict(color='#1d1d1f'), xaxis=dict(showgrid=False, showline=True, linecolor='#e5e5e5'), 
                      yaxis=dict(showgrid=True, gridcolor='#f5f5f5'))
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
            if st.button("登录系统"):
                if verify_login(u.strip(), p):
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = u.strip()
                    st.session_state["paid_code"] = ""
                    st.rerun()
                else: st.error("账号或密码错误")
        with tab2:
            nu = st.text_input("新用户")
            np1 = st.text_input("设置密码", type="password")
            if st.button("立即注册"):
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
        with st.expander("用户管理", expanded=True):
            df_u = load_users()
            st.dataframe(df_u[["username","quota"]], hide_index=True)
            u_list = [x for x in df_u["username"] if x!=ADMIN_USER]
            if u_list:
                target = st.selectbox("选择用户", u_list)
                val = st.number_input("新积分", value=0, step=10)
                if st.button("修改"): update_user_quota(target, val); st.success("OK"); time.sleep(0.5); st.rerun()
                if st.button("删除"): delete_user(target); st.success("Del"); time.sleep(0.5); st.rerun()
    else:
        st.info(f"👤 {user}")
        df_u = load_users()
        try: q = df_u[df_u["username"]==user]["quota"].iloc[0]
        except: q = 0
        st.metric("剩余积分", q)

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
        
    days = st.radio("周期", [7,30,60,120,250], 2, horizontal=True)
    adj = st.selectbox("复权", ["qfq","hfq",""], 0)
    
    st.divider()
    # 修复：初始化变量
    gann = st.checkbox("江恩", True)
    fib = st.checkbox("Fib", True)
    chan = st.checkbox("缠论", True)
    
    st.divider()
    if st.button("退出"): st.session_state["logged_in"]=False; st.rerun()

# 内容区
name = get_name(st.session_state.code, token)
st.title(f"{name} ({st.session_state.code})")

# 付费墙
if st.session_state.code != st.session_state.paid_code:
    st.info("🔒 深度研报需解锁")
    if st.button("🔍 支付 1 积分查看", type="primary"):
        if consume_quota(user):
            st.session_state.paid_code = st.session_state.code
            st.rerun()
        else: st.error("积分不足，请充值")
    st.stop()

if st.button("刷新"): st.cache_data.clear(); st.rerun()

with st.spinner("AI 正在生成深度研报..."):
    df = get_data(st.session_state.code, token, days, adjust)
    funda = get_fundamentals(st.session_state.code, token)

if df.empty:
    st.error("无数据")
else:
    df = calc_full_indicators(df)
    df = detect_patterns(df)
    
    # 1. 核心指标
    l = df.iloc[-1]
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("价格", f"{l['close']:.2f}", f"{l['pct_change']:.2f}%")
    c2.metric("PE", funda['pe'])
    c3.metric("RSI", f"{l['RSI']:.1f}")
    c4.metric("ADX", f"{l['ADX']:.1f}")
    c5.metric("量比", f"{l['VolRatio']:.2f}")
    
    # 2. 图表
    # 修复：传入正确变量
    plot_chart(df.tail(days), f"{name} 分析图", gann, fib, chan)
    
    # 3. 深度研报 (核心升级点)
    report_html = generate_deep_report(df, name)
    st.markdown(report_html, unsafe_allow_html=True)
    
    # 4. 结论
    score, act, col, sl, tp = analyze_score(df)
    st.subheader(f"🤖 最终建议: {act} (评分 {score})")
    
    s1,s2 = st.columns(2)
    s1.info(f"🛡️ 止损: {sl:.2f}"); s2.info(f"💰 止盈: {tp:.2f}")
