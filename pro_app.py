import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import bcrypt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. 核心配置 & 暴力隐藏 UI
# ==========================================
st.set_page_config(
    page_title="A股复盘系统(绝对管理员版)",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# 🚫 CSS 隐藏菜单
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

# 🔴 版本标记 (刷新网页如果没看到这个，说明没更新成功！)
st.error("🔴 当前版本: V11.0 (管理员逻辑分离版)")

# 👑 管理员账号 (硬编码，优先级最高)
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"

# 💾 普通用户数据库 (换个文件名，彻底隔离旧数据)
DB_FILE = "users_final_v1.csv"

# ==========================================
# 2. 数据库逻辑 (仅用于普通用户)
# ==========================================
def init_db():
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["username", "password_hash", "watchlist", "quota"])
        df.to_csv(DB_FILE, index=False)

init_db()

def load_users(): return pd.read_csv(DB_FILE, dtype={"watchlist": str, "quota": int})
def save_users(df): df.to_csv(DB_FILE, index=False)

# 核心修改：登录验证逻辑
def verify_login(u, p):
    # 1. 【优先】检查管理员 (直接比对字符串，不查数据库)
    # 只要输入的是 ZCX001 / 123456，直接放行，无视任何文件
    if u == ADMIN_USER and p == ADMIN_PASS:
        return True
        
    # 2. 【次要】检查普通用户 (查数据库)
    df = load_users()
    row = df[df["username"] == u]
    if row.empty: return False
    try: return bcrypt.checkpw(p.encode(), row.iloc[0]["password_hash"].encode())
    except: return False

# 核心修改：扣费逻辑
def consume_quota(u):
    # 1. 管理员无限
    if u == ADMIN_USER: return True
    
    # 2. 普通用户扣费
    df = load_users()
    idx = df[df["username"] == u].index
    if len(idx) > 0 and df.loc[idx[0], "quota"] > 0:
        df.loc[idx[0], "quota"] -= 1
        save_users(df)
        return True
    return False

def get_quota_display(u):
    if u == ADMIN_USER: return "♾️ 无限 (管理员)"
    df = load_users()
    row = df[df["username"] == u]
    if row.empty: return "0"
    return str(row.iloc[0]["quota"])

def register_normal_user(u, p):
    if u == ADMIN_USER: return False, "不能注册管理员账号"
    df = load_users()
    if u in df["username"].values: return False, "用户已存在"
    
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(p.encode(), salt).decode()
    new_row = {"username": u, "password_hash": hashed, "watchlist": "", "quota": 20}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_users(df)
    return True, "注册成功"

# ==========================================
# 3. 页面路由
# ==========================================
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False

# --- 登录页 ---
if not st.session_state["logged_in"]:
    st.markdown("<br><h1 style='text-align: center;'>🔐 A股系统登录</h1>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        tab1, tab2 = st.tabs(["登录", "注册"])
        with tab1:
            u = st.text_input("账号", key="l_u")
            p = st.text_input("密码", type="password", key="l_p")
            if st.button("🚀 登录", use_container_width=True):
                if verify_login(u.strip(), p):
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = u.strip()
                    st.rerun()
                else:
                    st.error("账号或密码错误")
        with tab2:
            nu = st.text_input("新账号", key="r_u")
            np1 = st.text_input("密码", type="password", key="r_p")
            if st.button("📝 注册", use_container_width=True):
                suc, msg = register_normal_user(nu.strip(), np1)
                if suc: st.success(msg)
                else: st.error(msg)
    st.stop()

# --- 主系统 ---
user = st.session_state["user"]

with st.sidebar:
    st.header(f"👤 {user}")
    st.info(f"剩余积分: {get_quota_display(user)}")
    
    # 🔴 管理员后台
    if user == ADMIN_USER:
        with st.expander("👮‍♂️ 管理员后台", expanded=True):
            df_users = load_users()
            all_users = df_users["username"].tolist()
            target = st.selectbox("管理用户", ["请选择"] + all_users)
            
            if target != "请选择":
                curr_q = df_users[df_users["username"]==target]["quota"].iloc[0]
                new_q = st.number_input("设置积分", value=int(curr_q), step=10)
                if st.button("💾 保存设置"):
                    df_users.loc[df_users["username"]==target, "quota"] = new_q
                    save_users(df_users)
                    st.success("已保存")
    
    if st.button("🚪 退出"):
        st.session_state["logged_in"] = False
        st.rerun()

st.title("📈 A股深度复盘 Pro")

# 模拟功能
if st.button("🔍 查询数据 (消耗1积分)", type="primary"):
    if consume_quota(user):
        st.success(f"查询成功！")
        st.line_chart(np.random.randn(20, 3), height=300)
    else:
        st.error("❌ 积分不足")
