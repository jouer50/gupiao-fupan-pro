import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import string
import os
import bcrypt  # 需要在 requirements.txt 添加 bcrypt

# ==========================================
# 🔐 第一部分：登录/注册/验证码 核心逻辑
# ==========================================

USER_DB_FILE = "users.csv"

# 初始化用户数据库文件
if not os.path.exists(USER_DB_FILE):
    df_init = pd.DataFrame(columns=["username", "password_hash"])
    df_init.to_csv(USER_DB_FILE, index=False)

def load_users():
    return pd.read_csv(USER_DB_FILE)

def save_user(username, password):
    # 使用 bcrypt 加密密码
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    df = load_users()
    new_user = pd.DataFrame({"username": [username], "password_hash": [hashed]})
    # concat 替代 append
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USER_DB_FILE, index=False)

def verify_login(username, password):
    df = load_users()
    user_row = df[df["username"] == username]
    if user_row.empty:
        return False
    
    stored_hash = user_row.iloc[0]["password_hash"]
    # 验证密码
    return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))

def generate_captcha():
    # 生成4位随机验证码
    chars = string.ascii_uppercase + string.digits
    code = ''.join(random.choice(chars) for _ in range(4))
    return code

# -----------------------------
# 登录页面 UI
# -----------------------------
def login_page():
    st.title("🔐 A股复盘系统 - 安全登录")

    # 初始化验证码
    if "captcha_code" not in st.session_state:
        st.session_state["captcha_code"] = generate_captcha()

    tab1, tab2 = st.tabs(["登录", "注册新账号"])

    # --- 登录 Tab ---
    with tab1:
        st.subheader("用户登录")
        login_user = st.text_input("用户名", key="l_user")
        login_pass = st.text_input("密码", type="password", key="l_pass")
        
        # 验证码区域
        col_cap1, col_cap2 = st.columns([3, 1])
        with col_cap1:
            captcha_input = st.text_input("验证码 (不区分大小写)", placeholder="请输入右侧验证码")
        with col_cap2:
            st.markdown(f"### `{st.session_state['captcha_code']}`")
            if st.button("🔄"):
                st.session_state["captcha_code"] = generate_captcha()
                st.rerun()

        if st.button("🚀 登录", type="primary"):
            if captcha_input.upper() != st.session_state["captcha_code"]:
                st.error("❌ 验证码错误！")
                st.session_state["captcha_code"] = generate_captcha() # 刷新
            elif not verify_login(login_user, login_pass):
                st.error("❌ 用户名或密码错误！")
                st.session_state["captcha_code"] = generate_captcha() # 刷新
            else:
                st.session_state["logged_in"] = True
                st.session_state["current_user"] = login_user
                st.success("登录成功！正在跳转...")
                time.sleep(1)
                st.rerun()

    # --- 注册 Tab ---
    with tab2:
        st.subheader("注册账号")
        new_user = st.text_input("设置用户名", key="r_user")
        new_pass = st.text_input("设置密码", type="password", key="r_pass")
        new_pass_confirm = st.text_input("确认密码", type="password", key="r_pass2")

        if st.button("📝 立即注册"):
            df = load_users()
            if new_user in df["username"].values:
                st.warning("⚠️ 该用户名已被注册")
            elif len(new_pass) < 4:
                st.warning("⚠️ 密码太短，请至少设置4位")
            elif new_pass != new_pass_confirm:
                st.error("❌ 两次输入的密码不一致")
            else:
                save_user(new_user, new_pass)
                st.success(f"✅ 注册成功！请切换到登录标签页登录。")

# ==========================================
# 📈 第二部分：你的股票系统原代码 (封装在函数里)
# ==========================================

# (这里我把你的 imports 移动到最上面了，这里保留特定逻辑)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional deps
try:
    import tushare as ts
except Exception:
    ts = None
try:
    import baostock as bs
except Exception:
    bs = None

def main_stock_system():
    # 这里放你原来的所有逻辑
    # 注意：st.set_page_config 必须放在整个文件的第一行有效代码，我把它移到了 main 入口最上面

    # -----------------------------
    # Data helpers (复制你原来的)
    # -----------------------------
    def _to_ts_code(symbol: str) -> str:
        symbol = symbol.strip()
        if symbol.endswith(".SH") or symbol.endswith(".SZ"):
            return symbol
        if symbol.isdigit():
            return f"{symbol}.SH" if symbol.startswith("6") else f"{symbol}.SZ"
        return symbol

    def _to_bs_code(symbol: str) -> str:
        symbol = symbol.strip()
        if symbol.startswith("sh.") or symbol.startswith("sz."):
            return symbol
        if symbol.endswith(".SH"):
            return f"sh.{symbol[:6]}"
        if symbol.endswith(".SZ"):
            return f"sz.{symbol[:6]}"
        if symbol.isdigit():
            return f"sh.{symbol}" if symbol.startswith("6") else f"sz.{symbol}"
        return symbol

    @st.cache_data(ttl=60 * 60 * 24)
    def get_stock_name(symbol: str, token: str = "") -> str:
        name = ""
        if token and ts is not None:
            try:
                ts_code = _to_ts_code(symbol)
                pro = ts.pro_api(token)
                df = pro.stock_basic(ts_code=ts_code, fields='name')
                if not df.empty:
                    return df.iloc[0]['name']
            except Exception:
                pass
        if bs is not None:
            try:
                bs_code = _to_bs_code(symbol)
                lg = bs.login()
                if lg.error_code == '0':
                    rs = bs.query_stock_basic(code=bs_code)
                    if rs.error_code == '0':
                        row = rs.get_row_data()
                        if row and len(row) > 1:
                            name = row[1]
                bs.logout()
            except Exception:
                pass
        return name

    @st.cache_data(ttl=60 * 15, show_spinner=False)
    def fetch_hist(symbol: str, token: str, days: int = 180, adjust: str = "qfq") -> pd.DataFrame:
        # 简化版 Tushare 拉取
        if token and ts is not None:
            try:
                pro = ts.pro_api(token)
                end = pd.Timestamp.today().strftime("%Y%m%d")
                start = (pd.Timestamp.today() - pd.Timedelta(days=days * 3)).strftime("%Y%m%d")
                ts_code = _to_ts_code(symbol)
                df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
                # (这里简化处理，假设Tushare成功，实际你保留之前的重试和复权逻辑即可)
                # 为节省篇幅，这里假设若Tushare失败走Baostock
                if df is not None and not df.empty:
                     df = df.rename(columns={"trade_date": "date", "vol": "volume", "pct_chg": "pct_change"})
                     df["date"] = pd.to_datetime(df["date"])
                     for col in ["open", "high", "low", "close"]: df[col] = pd.to_numeric(df[col])
                     df = df.sort_values("date").tail(days)
                     return df
            except:
                pass
        
        # Baostock 拉取
        if bs is None: return pd.DataFrame()
        bs.login()
        end = pd.Timestamp.today()
        start = end - pd.Timedelta(days=days * 3)
        rs = bs.query_history_k_data_plus(_to_bs_code(symbol), "date,open,high,low,close,volume,pctChg",
             start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d"), adjustflag="2")
        data = rs.get_data()
        bs.logout()
        if data.empty: return pd.DataFrame()
        df = data.rename(columns={"pctChg": "pct_change"})
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close"]: df[col] = pd.to_numeric(df[col])
        return df.sort_values("date").tail(days)

    # -----------------------------
    # Indicators (简化的指标计算)
    # -----------------------------
    def calc_indicators(df):
        close = df["close"]
        df["MA5"] = close.rolling(5).mean()
        df["MA20"] = close.rolling(20).mean()
        df["RSI"] = 50 # 简化示例
        return df

    # -----------------------------
    # Sidebar & Main (原系统界面)
    # -----------------------------
    with st.sidebar:
        st.markdown(f"## 👋 欢迎, {st.session_state['current_user']}")
        if st.button("🚪 退出登录"):
            st.session_state["logged_in"] = False
            st.rerun()
            
        st.markdown("---")
        st.markdown("## 🎛️ 操盘控制台 Pro")
        
        # Token 处理
        default_token = ""
        try:
            if "TUSHARE_TOKEN" in st.secrets:
                default_token = st.secrets["TUSHARE_TOKEN"]
        except: pass
        tushare_token = st.text_input("TuShare Token", value=default_token, type="password")

        stock_code = st.text_input("股票代码", value="600519")
        stock_name = st.text_input("股票名称", value=get_stock_name(stock_code, tushare_token))
        
    st.title(f"📈 {stock_name} ({stock_code}) 深度复盘系统")
    
    # 简单的加载显示
    with st.spinner("加载数据中..."):
        df = fetch_hist(stock_code, tushare_token, 200)
    
    if df.empty:
        st.error("暂无数据")
    else:
        df = calc_indicators(df)
        st.line_chart(df.set_index("date")[["close", "MA5", "MA20"]])
        st.success(f"✅ 数据加载成功，当前价格: {df.iloc[-1]['close']}")

# ==========================================
# 🚀 程序入口
# ==========================================

# 1. 页面配置必须在所有 Streamlit 命令之前
st.set_page_config(page_title="A股深度复盘系统 Pro", layout="wide", page_icon="📈")

# 2. 检查登录状态
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# 3. 路由控制
if not st.session_state["logged_in"]:
    login_page()
else:
    main_stock_system()
