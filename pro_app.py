import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os
import bcrypt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. 核心配置 & 界面暴力隐藏
# ==========================================
st.set_page_config(
    page_title="A股深度复盘系统",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# 🚫 针对 2025 新版 Streamlit 的暴力隐藏代码
hide_css = """
<style>
    /* 隐藏顶部 Header */
    header {visibility: hidden !important; height: 0px !important; padding: 0px !important; margin: 0px !important;}
    /* 隐藏右上角菜单和按钮 */
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    /* 隐藏顶部装饰条 */
    [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    /* 隐藏底部 Footer */
    footer {visibility: hidden !important; display: none !important;}
    /* 强制内容置顶 */
    .block-container {padding-top: 1rem !important;}
    /* 隐藏部署按钮 */
    .stDeployButton {display: none !important;}
    /* 隐藏所有链接按钮图标 */
    button[kind="header"] {display: none !important;}
</style>
"""
st.markdown(hide_css, unsafe_allow_html=True)

# 👑 管理员账号 (硬编码)
ADMIN_USER = "ZCX001"
# 💾 数据文件 (JSON格式)
DB_FILE = "users.json"

# ==========================================
# 2. 数据库逻辑 (JSON版)
# ==========================================
def init_db():
    # 如果文件不存在，创建一个空的列表
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, 'w') as f:
            json.dump([], f)

init_db()

def load_users():
    try:
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_users(users_list):
    with open(DB_FILE, 'w') as f:
        json.dump(users_list, f, indent=4)

def verify_login(u, p):
    users = load_users()
    for user in users:
        if user['username'] == u:
            stored_pw = user.get('password', '')
            
            # 💡 特殊逻辑：允许通过 JSON 直接配置明文密码 (以 PLAIN: 开头)
            if stored_pw.startswith("PLAIN:"):
                real_pw = stored_pw.split("PLAIN:")[1]
                return p == real_pw
            
            # 正常 bcrypt 验证
            try:
                return bcrypt.checkpw(p.encode(), stored_pw.encode())
            except:
                return False
    return False

def register_user(u, p):
    users = load_users()
    # 检查重名
    for user in users:
        if user['username'] == u:
            return False, "用户已存在"
            
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(p.encode(), salt).decode()
    
    new_user = {
        "username": u,
        "password": hashed,
        "quota": 20, # 新用户送20次
        "watchlist": ""
    }
    users.append(new_user)
    save_users(users)
    return True, "注册成功"

def consume_quota(u):
    if u == ADMIN_USER: return True
    users = load_users()
    for user in users:
        if user['username'] == u:
            if user['quota'] > 0:
                user['quota'] -= 1
                save_users(users)
                return True
            else:
                return False
    return False

def get_user_quota(u):
    if u == ADMIN_USER: return 999999
    users = load_users()
    for user in users:
        if user['username'] == u:
            return user.get('quota', 0)
    return 0

# ==========================================
# 3. 页面逻辑
# ==========================================

# 登录页
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    st.markdown("<br><br><h1 style='text-align: center;'>🔐 A股系统登录</h1>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        tab1, tab2 = st.tabs(["登录", "注册"])
        
        with tab1:
            u = st.text_input("账号", key="l_u")
            p = st.text_input("密码", type="password", key="l_p")
            if st.button("🚀 登录", type="primary", use_container_width=True):
                if verify_login(u, p):
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = u
                    st.rerun()
                else:
                    st.error("账号或密码错误")
        
        with tab2:
            nu = st.text_input("新账号", key="r_u")
            np1 = st.text_input("设置密码", type="password", key="r_p")
            if st.button("📝 注册", use_container_width=True):
                success, msg = register_user(nu, np1)
                if success: st.success(msg)
                else: st.error(msg)
                
    st.stop() 

# --- 进入主系统 ---
user = st.session_state["user"]
quota = get_user_quota(user)

# 侧边栏
with st.sidebar:
    # 身份卡片
    if user == ADMIN_USER:
        st.success(f"👑 管理员: {user}")
    else:
        st.info(f"👤 用户: {user}")
    
    st.metric("剩余积分", "无限" if user == ADMIN_USER else quota)
    
    if st.button("🚪 退出登录"):
        st.session_state["logged_in"] = False
        st.rerun()
    
    st.divider()
    
    # 🔴 管理员特权区
    if user == ADMIN_USER:
        with st.expander("👮‍♂️ 用户管理后台", expanded=True):
            users = load_users()
            # 简单的表格展示
            display_data = [{"用户": u['username'], "积分": u['quota']} for u in users]
            st.dataframe(display_data, hide_index=True)
            
            # 修改积分
            user_list = [u['username'] for u in users if u['username'] != ADMIN_USER]
            if user_list:
                target = st.selectbox("选择用户", user_list)
                new_q = st.number_input("设置积分", value=100, step=10)
                if st.button("💾 保存修改"):
                    for u in users:
                        if u['username'] == target:
                            u['quota'] = new_q
                    save_users(users)
                    st.success("已保存!")
                    time.sleep(0.5)
                    st.rerun()
    
    st.divider()
    st.caption("复盘工具箱")
    code = st.text_input("股票代码", "600519")

# 主界面
c_title, c_time = st.columns([3, 1])
with c_title:
    st.title("📈 A股深度复盘 Pro")
with c_time:
    st.caption(f"当前用户: {user}")

# 模拟查询功能
if st.button("🔍 查询数据 (消耗1积分)", type="primary", use_container_width=True):
    if consume_quota(user):
        st.success(f"✅ 查询成功！代码: {code}")
        # 这里模拟画图
        st.line_chart(np.random.randn(50, 3).cumsum(0), height=350)
        
        st.info("📊 智能分析：该股目前处于震荡上行趋势，建议关注支撑位。")
    else:
        st.error("❌ 积分不足！请联系管理员 ZCX001 充值。")

