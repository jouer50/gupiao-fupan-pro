import streamlit as st
import pandas as pd
import numpy as np
import time
import bcrypt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. 核心配置 & 暴力隐藏 UI
# ==========================================
st.set_page_config(
    page_title="A股复盘系统(最终版)",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# 🚫 暴力隐藏 CSS (针对2025新版)
hide_css = """
<style>
    /* 隐藏顶部 Header */
    header {visibility: hidden !important; height: 0px !important; padding: 0px !important; margin: 0px !important;}
    /* 隐藏右上角菜单 */
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    /* 隐藏顶部装饰条 */
    [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    /* 隐藏 Footer */
    footer {visibility: hidden !important; display: none !important;}
    /* 强制内容上移 */
    .block-container {padding-top: 1rem !important;}
    /* 隐藏部署按钮 */
    .stDeployButton {display: none !important;}
</style>
"""
st.markdown(hide_css, unsafe_allow_html=True)

# 🔴 版本验证水印 (如果不显示这行字，说明代码没更新成功！)
st.markdown("<h3 style='color: red; text-align: center;'>🔴 当前版本：V10.0 (最终修复版) - 代码已更新</h3>", unsafe_allow_html=True)

# 👑 管理员账号 (直接写死，绝对有效)
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"

# 模拟数据库 (内存版，重启会重置，但管理员永远存在)
if "users_db" not in st.session_state:
    st.session_state["users_db"] = {
        "test": {"pass": "123", "quota": 20, "watch": []} # 预设一个测试用户
    }

# ==========================================
# 2. 简化的权限逻辑
# ==========================================
def verify_login(u, p):
    # 1. 检查管理员
    if u == ADMIN_USER and p == ADMIN_PASS:
        return True
    # 2. 检查普通用户
    db = st.session_state["users_db"]
    if u in db and db[u]["pass"] == p:
        return True
    return False

def get_quota(u):
    if u == ADMIN_USER: return "♾️ 无限"
    return st.session_state["users_db"].get(u, {}).get("quota", 0)

def consume_quota(u):
    if u == ADMIN_USER: return True
    db = st.session_state["users_db"]
    if u in db and db[u]["quota"] > 0:
        db[u]["quota"] -= 1
        return True
    return False

# ==========================================
# 3. 页面路由
# ==========================================
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False

# --- 登录页 ---
if not st.session_state["logged_in"]:
    st.markdown("<br><h1 style='text-align: center;'>🔐 A股系统登录</h1>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.info(f"👉 管理员账号: **{ADMIN_USER}** / 密码: **{ADMIN_PASS}**")
        
        tab1, tab2 = st.tabs(["登录", "注册"])
        with tab1:
            u = st.text_input("账号", key="l_u")
            p = st.text_input("密码", type="password", key="l_p")
            if st.button("🚀 登录", use_container_width=True):
                if verify_login(u, p):
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = u
                    st.rerun()
                else:
                    st.error("账号或密码错误")
        
        with tab2:
            nu = st.text_input("新账号", key="r_u")
            np1 = st.text_input("密码", type="password", key="r_p")
            if st.button("📝 注册", use_container_width=True):
                if nu in st.session_state["users_db"]:
                    st.warning("用户已存在")
                else:
                    st.session_state["users_db"][nu] = {"pass": np1, "quota": 20, "watch": []}
                    st.success("注册成功！请登录")
    st.stop()

# --- 主界面 ---
user = st.session_state["user"]
quota = get_quota(user)

with st.sidebar:
    # 身份卡片
    if user == ADMIN_USER:
        st.success(f"👑 **管理员 {user}**\n\n积分：{quota}")
        st.divider()
        with st.expander("👮‍♂️ 管理员后台", expanded=True):
            st.write("用户列表：")
            st.json(st.session_state["users_db"])
            
            # 修改积分功能
            target = st.selectbox("选择用户", list(st.session_state["users_db"].keys()))
            new_q = st.number_input("设置积分", value=100)
            if st.button("💾 保存设置"):
                st.session_state["users_db"][target]["quota"] = new_q
                st.success("已修改")
    else:
        st.info(f"👤 **用户 {user}**\n\n积分：{quota}")
    
    st.divider()
    if st.button("🚪 退出登录"):
        st.session_state["logged_in"] = False
        st.rerun()

# 顶部标题
st.title("📈 A股深度复盘系统 Pro")

# 功能演示区
col1, col2 = st.columns([3, 1])
with col1:
    code = st.text_input("股票代码", "600519")
with col2:
    if st.button("🔍 查询数据 (消耗1积分)", type="primary", use_container_width=True):
        if consume_quota(user):
            st.success(f"✅ 查询成功！剩余积分: {get_quota(user)}")
            # 假装画个图证明功能在
            st.line_chart(np.random.randn(20, 3), height=300)
        else:
            st.error("❌ 积分不足！请联系管理员充值。")

if user == ADMIN_USER:
    st.info("💡 提示：您是管理员，所有查询不消耗积分。")
