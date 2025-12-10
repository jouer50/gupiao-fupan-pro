# pro_app.py
# Streamlit A股深度复盘系统（含用户/管理员管理）
# 直接覆盖原 pro_app.py 即可。请把项目 push 到 GitHub，然后 Streamlit Cloud 自动部署。

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
import secrets
import hashlib
import base64
from typing import Optional, Dict

# ===========================
# CONFIG
# ===========================
st.set_page_config(page_title="A股深度复盘系统", layout="wide", page_icon="📈", initial_sidebar_state="expanded")

# UI hide CSS (你原来的样式保留)
hide_css = """
<style>
    header {visibility: hidden !important; height: 0px !important; margin: 0px !important; padding: 0px !important;}
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    .block-container {padding-top: 1rem !important; margin-top: 0rem !important;}
    .stDeployButton {display: none !important;}
</style>
"""
st.markdown(hide_css, unsafe_allow_html=True)

USERS_FILE = "users.json"
ADMIN_USER = "ZCX001"
DEFAULT_ADMIN_PW = "123456"
ADMIN_INIT_TOKEN_FILE = "admin_init.token"  # optional local force-init (useful in local dev)

# ===========================
# PASSWORD HASHING (PBKDF2)
# ===========================
def make_salt(nbytes=16) -> str:
    return base64.b64encode(secrets.token_bytes(nbytes)).decode()

def hash_password(password: str, salt_b64: str, iterations: int = 120000) -> str:
    salt = base64.b64decode(salt_b64.encode())
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return base64.b64encode(dk).decode()

def verify_password(password: str, salt_b64: str, hash_b64: str, iterations: int = 120000) -> bool:
    return hash_password(password, salt_b64, iterations) == hash_b64

# ===========================
# USERS JSON IO
# Format:
# {
#   "users": [
#     {"username":"ZCX001", "salt":"...", "password_hash":"...", "is_admin":true, "quota":999999, "watchlist":""},
#     ...
#   ]
# }
# ===========================
def load_users() -> Dict:
    if not os.path.exists(USERS_FILE):
        return {"users": []}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "users" not in data:
            data = {"users": []}
        return data
    except Exception:
        return {"users": []}

def save_users(data: Dict):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def find_user(data: Dict, username: str) -> Optional[dict]:
    for u in data["users"]:
        if u["username"] == username:
            return u
    return None

# ===========================
# INIT (create users.json if not exists; do not overwrite existing admin)
# ===========================
def init_users_file():
    data = load_users()
    if not data["users"]:
        # empty file -> create default admin
        salt = make_salt()
        pwd_hash = hash_password(DEFAULT_ADMIN_PW, salt)
        admin_row = {
            "username": ADMIN_USER,
            "salt": salt,
            "password_hash": pwd_hash,
            "is_admin": True,
            "quota": 999999,
            "watchlist": ""
        }
        data["users"].append(admin_row)
        save_users(data)
        st.info(f"已创建默认管理员 `{ADMIN_USER}` / 密码 `{DEFAULT_ADMIN_PW}`。**建议部署后立即修改管理员密码**。")
        st.warning("注意：如果你希望把用户持久保存到 GitHub，请将生成的 users.json 提交到仓库。")
        return

    # if users exist but no admin, try to add admin (do not overwrite existing admin)
    if not any(u.get("is_admin") for u in data["users"]):
        # if ADMIN_INIT_TOKEN_FILE exists with content, force creation; otherwise create admin but warn
        salt = make_salt()
        pwd_hash = hash_password(DEFAULT_ADMIN_PW, salt)
        admin_row = {
            "username": ADMIN_USER,
            "salt": salt,
            "password_hash": pwd_hash,
            "is_admin": True,
            "quota": 999999,
            "watchlist": ""
        }
        data["users"].append(admin_row)
        save_users(data)
        st.warning(f"检测到用户数据无管理员，已创建管理员 `{ADMIN_USER}`，默认密码 `{DEFAULT_ADMIN_PW}`。请尽快修改并提交 users.json。")

# run init
init_users_file()

# ===========================
# AUTH / QUOTA FUNCTIONS
# ===========================
def verify_login(username: str, password: str) -> bool:
    data = load_users()
    user = find_user(data, username)
    if not user:
        return False
    try:
        return verify_password(password, user["salt"], user["password_hash"])
    except Exception:
        return False

def change_password(username: str, new_password: str) -> bool:
    data = load_users()
    user = find_user(data, username)
    if not user:
        return False
    salt = make_salt()
    user["salt"] = salt
    user["password_hash"] = hash_password(new_password, salt)
    save_users(data)
    return True

def consume_quota(username: str) -> bool:
    if username == ADMIN_USER:
        return True
    data = load_users()
    user = find_user(data, username)
    if not user:
        return False
    if user.get("quota", 0) > 0:
        user["quota"] = user.get("quota", 0) - 1
        save_users(data)
        return True
    return False

def admin_create_user(admin_username: str, new_username: str, new_password: str, is_admin: bool=False, quota: int=100) -> (bool,str):
    data = load_users()
    admin = find_user(data, admin_username)
    if not admin or not admin.get("is_admin", False):
        return False, "权限不足"
    if find_user(data, new_username):
        return False, "用户名已存在"
    salt = make_salt()
    pwd_hash = hash_password(new_password, salt)
    row = {"username": new_username, "salt": salt, "password_hash": pwd_hash, "is_admin": bool(is_admin), "quota": int(quota), "watchlist": ""}
    data["users"].append(row)
    save_users(data)
    return True, "创建成功"

def admin_delete_user(admin_username: str, target_username: str) -> (bool,str):
    data = load_users()
    admin = find_user(data, admin_username)
    if not admin or not admin.get("is_admin", False):
        return False, "权限不足"
    if target_username == ADMIN_USER:
        return False, "不能删除默认管理员"
    new_users = [u for u in data["users"] if u["username"] != target_username]
    if len(new_users) == len(data["users"]):
        return False, "用户不存在"
    data["users"] = new_users
    save_users(data)
    return True, "删除成功"

def admin_reset_password(admin_username: str, target_username: str) -> (bool, str):
    data = load_users()
    admin = find_user(data, admin_username)
    if not admin or not admin.get("is_admin", False):
        return False, "权限不足"
    user = find_user(data, target_username)
    if not user:
        return False, "用户不存在"
    temp = secrets.token_urlsafe(8)
    salt = make_salt()
    user["salt"] = salt
    user["password_hash"] = hash_password(temp, salt)
    save_users(data)
    return True, temp

def admin_set_quota(admin_username: str, target_username: str, new_quota: int) -> (bool, str):
    data = load_users()
    admin = find_user(data, admin_username)
    if not admin or not admin.get("is_admin", False):
        return False, "权限不足"
    user = find_user(data, target_username)
    if not user:
        return False, "用户不存在"
    user["quota"] = int(new_quota)
    save_users(data)
    return True, "已修改"

def admin_promote(admin_username: str, target_username: str, make_admin: bool) -> (bool,str):
    data = load_users()
    admin = find_user(data, admin_username)
    if not admin or not admin.get("is_admin", False):
        return False, "权限不足"
    user = find_user(data, target_username)
    if not user:
        return False, "用户不存在"
    user["is_admin"] = bool(make_admin)
    save_users(data)
    return True, "已修改"

# ===========================
# STREAMLIT UI: LOGIN
# ===========================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None

if not st.session_state["logged_in"]:
    st.markdown("<br><h1 style='text-align: center;'>🔐 A股系统登录</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.info(f"👉 管理员账号: **{ADMIN_USER}**（首次运行默认密码: **{DEFAULT_ADMIN_PW}**）")
        username = st.text_input("账号")
        password = st.text_input("密码", type="password")
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("登录", type="primary", use_container_width=True):
                if verify_login(username, password):
                    tmp = username
                    st.session_state.clear()
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = tmp
                    st.experimental_rerun()
                else:
                    st.error("账号或密码错误")
        with col2:
            if st.button("游客 试用 (临时用户)", use_container_width=True):
                # create a temp quick user and login (and save to file so token persists while runtime alive)
                tmp_user = f"guest_{int(time.time()%100000)}"
                tmp_pwd = "guest"
                ok, msg = admin_create_user(ADMIN_USER, tmp_user, tmp_pwd, is_admin=False, quota=10)
                # if admin_create_user fails because admin exists? Usually admin exists, so ok should be True.
                st.session_state.clear()
                st.session_state["logged_in"] = True
                st.session_state["user"] = tmp_user
                st.experimental_rerun()

        st.markdown("---")
        st.subheader("密码相关")
        target_reset = st.text_input("要修改的账号（留空表示你自己）", value=username)
        cur_pw = st.text_input("当前密码（若要修改自己的密码）", type="password")
        new_pw = st.text_input("新的密码", type="password")
        if st.button("修改/重置密码（需知道当前密码）"):
            target = target_reset if target_reset else username
            if not target or not cur_pw:
                st.error("请填写账号和当前密码")
            else:
                if verify_login(target, cur_pw):
                    change_password(target, new_pw)
                    st.success(f"{target} 的密码已修改，请重新登录。")
                else:
                    st.error("当前密码校验失败")

        st.markdown("---")
        st.caption("注意：若 users.json 是由 GitHub 管理，建议在本地修改并 push 到仓库以持久化变更（Streamlit Cloud 运行时的文件变更不会自动写回 GitHub）。")

    st.stop()

# ===========================
# MAIN APP after login
# ===========================
user = st.session_state["user"]
data = load_users()
user_row = find_user(data, user)
curr_quota = user_row.get("quota", 0) if user_row else 0
is_admin = bool(user_row.get("is_admin", False)) if user_row else False

# Sidebar
with st.sidebar:
    st.header(f"👤 {user}")
    st.write(f"管理员权限: {is_admin}")
    if is_admin:
        st.success("✅ 管理员模式已激活")
    else:
        st.metric("剩余积分", curr_quota)

    st.markdown("---")
    if st.button("🚪 退出登录"):
        st.session_state["logged_in"] = False
        st.session_state["user"] = None
        st.experimental_rerun()

    # 管理员区块
    if is_admin:
        st.subheader("👮‍♂️ 用户管理（管理员专用）")
        df = pd.DataFrame(data["users"])
        # show basic table
        if not df.empty:
            display_df = df[["username", "is_admin", "quota"]].copy()
            st.dataframe(display_df)
        st.markdown("**创建用户**")
        new_user = st.text_input("用户名", key="admin_new_user")
        new_pw = st.text_input("密码", key="admin_new_user_pw")
        new_quota = st.number_input("积分", value=100, key="admin_new_user_quota")
        new_is_admin = st.checkbox("设为管理员", key="admin_new_user_admin")
        if st.button("创建用户", key="admin_do_create"):
            if not new_user or not new_pw:
                st.error("用户名/密码不能为空")
            else:
                ok, msg = admin_create_user(user, new_user, new_pw, new_is_admin, int(new_quota))
                if ok:
                    st.success(msg + "（请将 users.json commit 到 GitHub 持久化）")
                    st.experimental_rerun()
                else:
                    st.error(msg)

        st.markdown("---")
        st.markdown("**对用户操作**")
        target = st.selectbox("选择用户", ["请选择"] + [u["username"] for u in data["users"]], key="admin_target_select")
        if target and target != "请选择":
            st.write(find_user(data, target))
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("🔑 重置密码为临时值", key="admin_reset_btn"):
                    ok, temp = admin_reset_password(user, target)
                    if ok:
                        st.success(f"已重置。临时密码：{temp} （只显示一次，建议记录）")
                        st.experimental_rerun()
                    else:
                        st.error(temp)
            with col_b:
                new_q_val = st.number_input("修改积分为", value=int(find_user(data,target).get("quota",100)), key="admin_mod_quota")
                if st.button("保存积分", key="admin_save_quota"):
                    ok, msg = admin_set_quota(user, target, int(new_q_val))
                    if ok:
                        st.success(msg)
                        st.experimental_rerun()
                    else:
                        st.error(msg)
            with col_c:
                if st.checkbox("设为管理员", value=bool(find_user(data,target).get("is_admin", False)), key="admin_promote_chk"):
                    if st.button("保存角色", key="admin_save_role"):
                        ok, msg = admin_promote(user, target, True)
                        if ok:
                            st.success(msg)
                            st.experimental_rerun()
                        else:
                            st.error(msg)
                if st.button("删除用户", key="admin_delete_btn"):
                    ok, msg = admin_delete_user(user, target)
                    if ok:
                        st.success(msg)
                        st.experimental_rerun()
                    else:
                        st.error(msg)

    # debug / CSV viewer
    st.markdown("---")
    if st.checkbox("显示调试信息 (session + users.json)", value=False):
        st.write("Session:", dict(st.session_state))
        st.write("users.json 内容：")
        st.json(load_users())

# 主界面标题
st.title("📈 A股深度复盘系统 Pro")

# 简单股票查询演示（占位）
col1, col2 = st.columns([3,1])
with col1:
    code = st.text_input("股票代码", "600519")
    st.write("（此为占位查询。你可以把真实行情 API 集成到此处）")
    # 如果你想要接入行情，把对应获取数据的函数放到这里，替换下面示例
    try:
        # 简单示例图（随机）
        st.line_chart(np.random.randn(60, 3).cumsum(0), height=350)
    except Exception as e:
        st.error(f"图表渲染异常：{e}")

with col2:
    if st.button("🔍 查询 (消耗1积分)", type="primary", use_container_width=True):
        if consume_quota(user):
            st.success("查询成功！数据已刷新（演示数据）")
            st.experimental_rerun()
        else:
            st.error("❌ 积分不足！请联系管理员充值。")

# 密码修改区
with st.expander("🔐 修改我的密码"):
    old = st.text_input("当前密码", type="password", key="chg_old")
    new = st.text_input("新密码", type="password", key="chg_new")
    if st.button("修改密码", key="chg_do"):
        if verify_login(user, old):
            change_password(user, new)
            st.success("密码已修改，请重新登录。")
            st.session_state["logged_in"] = False
            st.session_state["user"] = None
            st.experimental_rerun()
        else:
            st.error("当前密码错误")

st.markdown("---")
st.caption("说明：1) 若需持久化用户变更，请将本仓库根目录的 users.json 提交到 GitHub。2) 若想自动从行情接口取数据，我可以在此加入 Tushare/akshare 示例并帮你配置 Key。")