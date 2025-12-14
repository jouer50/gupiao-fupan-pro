# pro_app.py
# =========================================================
# 阿尔法量研 Pro - GitHub 可直接部署完整版
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import json
import bcrypt
import random
import string
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# 基础依赖检查
# -----------------------------
try:
    import yfinance as yf
except Exception:
    st.error("缺少 yfinance 依赖，请在 requirements.txt 中加入 yfinance")
    st.stop()

# -----------------------------
# 页面配置
# -----------------------------
st.set_page_config(
    page_title="阿尔法量研 Pro",
    layout="wide",
    page_icon="🔥",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Session 初始化
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "code" not in st.session_state:
    st.session_state.code = "600519"

# -----------------------------
# 常量
# -----------------------------
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"
DB_FILE = "users.csv"

MA_S = 5
MA_L = 20

FLAGS = {
    "ma": True,
    "boll": True,
    "macd": True,
    "vol": True
}

# -----------------------------
# 简化 UI CSS
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {background-color:#f7f8fa;}
    div.stButton>button {border-radius:20px;font-weight:700}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# 数据库
# =========================================================

def init_db():
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["username", "password"])
        df.to_csv(DB_FILE, index=False)


def load_users():
    try:
        return pd.read_csv(DB_FILE)
    except Exception:
        return pd.DataFrame(columns=["username", "password"])


def save_users(df):
    df.to_csv(DB_FILE, index=False)


def verify_login(u, p):
    if u == ADMIN_USER and p == ADMIN_PASS:
        return True
    df = load_users()
    row = df[df.username == u]
    if row.empty:
        return False
    return bcrypt.checkpw(p.encode(), row.iloc[0]["password"].encode())


def register_user(u, p):
    df = load_users()
    if u in df.username.values:
        return False
    hashed = bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()
    df = pd.concat([df, pd.DataFrame([[u, hashed]], columns=df.columns)])
    save_users(df)
    return True


init_db()

# =========================================================
# 行情与指标
# =========================================================

def get_data(code):
    df = yf.download(code, period="2y", progress=False)
    if df.empty:
        return df
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df['ma_s'] = df['close'].rolling(MA_S).mean()
    df['ma_l'] = df['close'].rolling(MA_L).mean()
    mid = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['upper'] = mid + 2 * std
    df['lower'] = mid - 2 * std
    return df


def plot_chart(df, code):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])

    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="K"
        ),
        row=1, col=1
    )

    fig.add_trace(go.Scatter(x=df['date'], y=df['ma_s'], name=f"MA{MA_S}"), 1, 1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['ma_l'], name=f"MA{MA_L}"), 1, 1)

    fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name="VOL"), 2, 1)

    fig.update_layout(height=600, showlegend=True)
    return fig

# =========================================================
# 页面逻辑
# =========================================================

def login_page():
    st.title("登录 / 注册")
    u = st.text_input("用户名")
    p = st.text_input("密码", type="password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("登录"):
            if verify_login(u, p):
                st.session_state.logged_in = True
                st.session_state.username = u
                st.experimental_rerun()
            else:
                st.error("登录失败")
    with col2:
        if st.button("注册"):
            if register_user(u, p):
                st.success("注册成功，请登录")
            else:
                st.error("用户已存在")


def main_page():
    st.sidebar.title("阿尔法量研 Pro")
    code = st.sidebar.text_input("股票代码", st.session_state.code)
    if st.sidebar.button("分析"):
        st.session_state.code = code

    st.sidebar.markdown(f"当前用户：**{st.session_state.username}**")
    if st.sidebar.button("退出"):
        st.session_state.logged_in = False
        st.experimental_rerun()

    st.title(f"📈 {st.session_state.code} 技术分析")
    df = get_data(st.session_state.code)
    if df.empty:
        st.warning("无数据")
        return

    fig = plot_chart(df, st.session_state.code)
    st.plotly_chart(fig, use_container_width=True)

    last = df.iloc[-1]
    st.metric("最新价", round(last['close'], 2))


# =========================================================
# 入口
# =========================================================

if not st.session_state.logged_in:
    login_page()
else:
    main_page()
