# -*- coding: utf-8 -*-
import os
import re
import csv
import math
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ===============================
# ✅ 可选：tushare（没有也能跑兜底）
# ===============================
TUSHARE_AVAILABLE = True
try:
    import tushare as ts
except Exception:
    TUSHARE_AVAILABLE = False

# =========================================================
# 0) 基础配置（移动端友好：不靠侧边栏、不跳页）
# =========================================================
st.set_page_config(
    page_title="交易裁决 · 刹车系统",
    layout="wide",
    page_icon="🛑",
    initial_sidebar_state="collapsed",
)

# =========================================================
# 1) 全局常量（单文件最小上线）
# =========================================================
APP_NAME = "🛑 交易裁决 · 刹车系统"
SLOGAN = "我们不帮你赚钱，只帮你在不该动的时候停下来。"

DISCIPLINE = [
    "市场不欠我钱，但一定会拿我的钱",
    "错过的机会不是我的钱",
    "保住本金才是第一位，我的钱不能让他们拿去化债",
]

# CSV 本地库
USER_DB_PATH = "users.csv"
PRO_CODES_PATH = "pro_codes.csv"
ORDERS_PATH = "orders.csv"

# ✅ 初始管理员（首次运行会自动写入 users.csv）
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123456"  # ⚠️上线后请立即改掉

# ✅ Pro 激活码默认天数
PRO_DAYS_DEFAULT = 30

# ✅ 兜底码（没有 pro_codes.csv 时也能临时用）
DEFAULT_PRO_CODES = {"VIP-8888", "PRO-2025", "BRAKE-99"}

# A股一手=100股
LOT_SIZE = 100


# =========================================================
# 2) ✅ Session State：只在这里初始化一次（任何 widget 之前）
# =========================================================
def init_state():
    ss = st.session_state
    if "state_inited" not in ss:
        ss.state_inited = True

        # 登录态
        ss.logged_in = False
        ss.user = None
        ss.role = "free"        # free / pro / admin
        ss.pro_expire = None    # isoformat str or None
        ss.pro_enabled = True   # pro 开关（pro/admin有效）

        # UI 状态
        ss.toast = ""
        ss.show_pro_box = False

        # 应用状态（避免用 page 这种 key）
        ss.app_code = "600519"

init_state()


# =========================================================
# 3) ✅ 用户库（CSV） + 轻量密码哈希
# =========================================================
def _hash_password(password: str) -> str:
    import hashlib
    salt = os.urandom(16).hex()
    h = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return f"sha256${salt}${h}"

def _verify_password(password: str, stored: str) -> bool:
    import hashlib
    try:
        algo, salt, h = stored.split("$", 2)
        if algo != "sha256":
            return False
        chk = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
        return chk == h
    except Exception:
        return False

def _ensure_user_db():
    """
    - 没有 users.csv：创建并写入 admin
    - 有 users.csv 但没有 admin：补一个 admin
    """
    cols = ["username", "pwd_hash", "role", "pro_expire", "created_at", "last_login_at", "note"]
    now = datetime.now().isoformat(timespec="seconds")

    if not os.path.exists(USER_DB_PATH):
        with open(USER_DB_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerow([
                DEFAULT_ADMIN_USERNAME,
                _hash_password(DEFAULT_ADMIN_PASSWORD),
                "admin",
                "",
                now,
                now,
                "default_admin"
            ])
        return

    # 文件存在，检查是否有 admin
    try:
        df = pd.read_csv(USER_DB_PATH, dtype=str).fillna("")
    except Exception:
        # 文件坏了就重建（保守做法：尽量不炸）
        with open(USER_DB_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerow([
                DEFAULT_ADMIN_USERNAME,
                _hash_password(DEFAULT_ADMIN_PASSWORD),
                "admin",
                "",
                now,
                now,
                "default_admin_rebuilt"
            ])
        return

    if "username" not in df.columns:
        df["username"] = ""
    has_admin = (df["username"].str.lower() == DEFAULT_ADMIN_USERNAME.lower()).any()
    if not has_admin:
        row = {
            "username": DEFAULT_ADMIN_USERNAME,
            "pwd_hash": _hash_password(DEFAULT_ADMIN_PASSWORD),
            "role": "admin",
            "pro_expire": "",
            "created_at": now,
            "last_login_at": now,
            "note": "default_admin_added",
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(USER_DB_PATH, index=False, encoding="utf-8")

def load_users() -> pd.DataFrame:
    _ensure_user_db()
    try:
        df = pd.read_csv(USER_DB_PATH, dtype=str).fillna("")
        # 补列
        for col in ["username", "pwd_hash", "role", "pro_expire", "created_at", "last_login_at", "note"]:
            if col not in df.columns:
                df[col] = ""
        return df
    except Exception:
        return pd.DataFrame(columns=["username", "pwd_hash", "role", "pro_expire", "created_at", "last_login_at", "note"])

def save_users(df: pd.DataFrame):
    df.to_csv(USER_DB_PATH, index=False, encoding="utf-8")

def find_user(df: pd.DataFrame, username: str):
    m = df["username"].str.lower() == username.lower()
    if m.any():
        return df[m].iloc[0]
    return None

def upsert_user(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    df = df.copy()
    m = df["username"].str.lower() == row["username"].lower()
    if m.any():
        idx = df[m].index[0]
        for k, v in row.items():
            df.loc[idx, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df

def parse_expire(s: str):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def is_admin(role: str) -> bool:
    return (role or "").strip().lower() == "admin"

def is_pro(role: str, pro_expire: str) -> bool:
    r = (role or "").strip().lower()
    if r == "admin":
        return True
    if r != "pro":
        return False
    exp = parse_expire(pro_expire)
    if exp is None:
        return False
    return datetime.now() <= exp

def set_login(username: str, role: str, pro_expire: str):
    st.session_state.logged_in = True
    st.session_state.user = username

    role_l = (role or "free").strip().lower()
    if role_l == "admin":
        st.session_state.role = "admin"
        st.session_state.pro_expire = None
        st.session_state.pro_enabled = True
    elif is_pro(role, pro_expire):
        st.session_state.role = "pro"
        st.session_state.pro_expire = pro_expire
        st.session_state.pro_enabled = True
    else:
        st.session_state.role = "free"
        st.session_state.pro_expire = pro_expire if pro_expire else None
        st.session_state.pro_enabled = False

    st.session_state.toast = f"✅ 登录成功：{username}"

def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.role = "free"
    st.session_state.pro_expire = None
    st.session_state.pro_enabled = True
    st.session_state.toast = "👋 已退出"


# =========================================================
# 4) 💰PRO码池 + 订单记录（订阅最小闭环）
# =========================================================
def ensure_pro_codes_db():
    if os.path.exists(PRO_CODES_PATH):
        return
    with open(PRO_CODES_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "days", "status", "created_at", "used_by", "used_at", "note"])
        # 预置几条默认码（方便你测试）
        now = datetime.now().isoformat(timespec="seconds")
        for c in sorted(DEFAULT_PRO_CODES):
            w.writerow([c, PRO_DAYS_DEFAULT, "active", now, "", "", "seed"])

def ensure_orders_db():
    if os.path.exists(ORDERS_PATH):
        return
    with open(ORDERS_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["order_id", "username", "code", "days", "old_expire", "new_expire", "created_at", "note"])

def load_pro_codes() -> pd.DataFrame:
    ensure_pro_codes_db()
    try:
        df = pd.read_csv(PRO_CODES_PATH, dtype=str).fillna("")
        for col in ["code", "days", "status", "created_at", "used_by", "used_at", "note"]:
            if col not in df.columns:
                df[col] = ""
        return df
    except Exception:
        return pd.DataFrame(columns=["code", "days", "status", "created_at", "used_by", "used_at", "note"])

def save_pro_codes(df: pd.DataFrame):
    df.to_csv(PRO_CODES_PATH, index=False, encoding="utf-8")

def load_orders() -> pd.DataFrame:
    ensure_orders_db()
    try:
        df = pd.read_csv(ORDERS_PATH, dtype=str).fillna("")
        return df
    except Exception:
        return pd.DataFrame(columns=["order_id", "username", "code", "days", "old_expire", "new_expire", "created_at", "note"])

def append_order(username: str, code: str, days: int, old_expire: str, new_expire: str, note: str = ""):
    ensure_orders_db()
    oid = f"ODR-{datetime.now().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(3).upper()}"
    now = datetime.now().isoformat(timespec="seconds")
    with open(ORDERS_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([oid, username, code, str(days), old_expire or "", new_expire or "", now, note])

def gen_codes(n: int = 10, days: int = 30, prefix: str = "VIP") -> pd.DataFrame:
    df = load_pro_codes()
    now = datetime.now().isoformat(timespec="seconds")
    new_rows = []
    for _ in range(int(n)):
        code = f"{prefix}-{secrets.token_hex(3).upper()}"
        new_rows.append({
            "code": code,
            "days": str(int(days)),
            "status": "active",
            "created_at": now,
            "used_by": "",
            "used_at": "",
            "note": "generated",
        })
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    save_pro_codes(df)
    return pd.DataFrame(new_rows)

def redeem_code_for_user(username: str, code_in: str) -> (bool, str):
    """
    兑换成功 -> 更新 users.csv 到期、更新 pro_codes.csv 状态、写入 orders.csv
    返回 (ok, msg)
    """
    code_in = (code_in or "").strip()
    if not code_in:
        return False, "请输入激活码。"

    # 游客不可兑换
    if username == "guest":
        return False, "游客无法兑换：请先注册/登录账号。"

    # 找码：优先 codes csv；若没有找到再允许 DEFAULT_PRO_CODES 兜底
    dfc = load_pro_codes()
    row = None
    if not dfc.empty:
        m = dfc["code"].astype(str).str.strip().str.upper() == code_in.upper()
        if m.any():
            row = dfc[m].iloc[0]

    if row is None:
        # 兜底码（不写入码池也允许激活，方便你早期测试）
        if code_in in DEFAULT_PRO_CODES:
            days = PRO_DAYS_DEFAULT
            status_ok = True
            code_source = "fallback"
        else:
            return False, "激活码无效。"
    else:
        status = (row["status"] or "").strip().lower()
        if status != "active":
            return False, "激活码已使用或已作废。"
        try:
            days = int(float(row["days"] or PRO_DAYS_DEFAULT))
        except Exception:
            days = PRO_DAYS_DEFAULT
        status_ok = True
        code_source = "pool"

    if not status_ok:
        return False, "激活码不可用。"

    # 更新用户到期
    dfu = load_users()
    urow = find_user(dfu, username)
    if urow is None:
        return False, "用户不存在（用户库异常）。"

    now = datetime.now()
    old_exp = (urow["pro_expire"] or "").strip()
    cur_exp = parse_expire(old_exp)
    base = cur_exp if (cur_exp and cur_exp > now) else now
    new_exp = base + timedelta(days=int(days))

    urow_dict = dict(urow)
    urow_dict["role"] = "pro"
    urow_dict["pro_expire"] = new_exp.isoformat(timespec="seconds")
    urow_dict["note"] = f"activated:{code_in}"
    dfu = upsert_user(dfu, urow_dict)
    save_users(dfu)

    # 更新码池状态（如果来自码池）
    if code_source == "pool":
        m = dfc["code"].astype(str).str.strip().str.upper() == code_in.upper()
        idx = dfc[m].index[0]
        dfc.loc[idx, "status"] = "used"
        dfc.loc[idx, "used_by"] = username
        dfc.loc[idx, "used_at"] = datetime.now().isoformat(timespec="seconds")
        save_pro_codes(dfc)

    # 写订单记录
    append_order(username, code_in, int(days), old_exp, new_exp.isoformat(timespec="seconds"), note=f"source:{code_source}")

    # 刷新 session 登录态
    set_login(username, "pro", new_exp.isoformat(timespec="seconds"))
    return True, f"✅ PRO 已开通，到期：{new_exp.strftime('%Y-%m-%d %H:%M')}"


# =========================================================
# 5) 行情 & 指标（无 tushare 也能跑）
# =========================================================
def format_code(code: str) -> str:
    code = (code or "").strip().upper().replace(" ", "")
    if "." in code:
        return code
    if re.match(r"^(6|9)\d{5}$", code):
        return code + ".SH"
    if re.match(r"^(0|3)\d{5}$", code):
        return code + ".SZ"
    return code

def get_tushare_pro():
    token = None
    try:
        token = st.secrets.get("TUSHARE_TOKEN", None)
    except Exception:
        token = None
    token = token or os.environ.get("TUSHARE_TOKEN", None)

    # ⚠️你原本写死的 token 这里保留兼容（线上建议用 secrets/env）
    token = token or "4fe6f3b0ef5355f526f49e54ca032f7d0d770187124c176be266c289"

    if not TUSHARE_AVAILABLE:
        return None
    try:
        ts.set_token(token)
        return ts.pro_api()
    except Exception:
        return None

def fallback_data(days: int = 220) -> pd.DataFrame:
    dates = pd.date_range(end=datetime.today(), periods=days, freq="B")
    close = np.cumsum(np.random.normal(0, 1.2, days)) + 100
    high = close * (1 + np.random.uniform(0.001, 0.03, days))
    low = close * (1 - np.random.uniform(0.001, 0.03, days))
    open_ = (high + low) / 2
    vol = np.random.randint(1000, 50000, days)
    return pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "vol": vol,
        "amount": vol * close,
    })

@st.cache_data(ttl=300, show_spinner=False)
def get_data_cached(ts_code: str, days: int = 220) -> pd.DataFrame:
    pro = get_tushare_pro()
    if pro is None:
        return fallback_data(days)

    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days * 3)).strftime("%Y%m%d")
    df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)

    if df is None or df.empty:
        return fallback_data(days)

    df = df.rename(columns={"trade_date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(days).reset_index(drop=True)

    for c in ["date", "open", "high", "low", "close"]:
        if c not in df.columns:
            return fallback_data(days)

    if "vol" not in df.columns:
        df["vol"] = np.nan
    if "amount" not in df.columns:
        df["amount"] = np.nan
    return df

def indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA5"] = df["close"].rolling(5).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA60"] = df["close"].rolling(60).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI14"] = 100 - (100 / (1 + rs))

    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ),
    )
    df["ATR14"] = tr.rolling(14).mean()

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD"] = (df["DIF"] - df["DEA"]) * 2

    return df

def make_verdict(df: pd.DataFrame):
    last = df.iloc[-1]
    close = float(last["close"])
    ma60 = float(last["MA60"]) if not np.isnan(last["MA60"]) else float(df["close"].tail(60).mean())
    rsi = float(last["RSI14"]) if not np.isnan(last["RSI14"]) else 50.0
    atr = float(last["ATR14"]) if not np.isnan(last["ATR14"]) else float((df["high"] - df["low"]).tail(14).mean())

    low20 = float(df["low"].tail(20).min())
    high20 = float(df["high"].tail(20).max())

    buy_low = low20 - 0.5 * atr
    buy_high = low20 + 0.5 * atr
    sell_low = high20 - 0.5 * atr
    sell_high = high20 + 0.5 * atr
    stop_line = low20 - 0.5 * atr

    zones = {
        "buy": f"{buy_low:.2f} ~ {buy_high:.2f}",
        "sell": f"{sell_low:.2f} ~ {sell_high:.2f}",
        "stop": f"{stop_line:.2f}",
        "buy_low": buy_low,
        "buy_high": buy_high,
        "sell_low": sell_low,
        "sell_high": sell_high,
        "stop_line": stop_line,
        "close": close,
        "ma60": ma60,
        "rsi": rsi,
        "atr": atr,
    }

    if close < ma60:
        return "🔴 不建议参与", "高", zones, "降低仓位 / 退出"
    if rsi >= 75:
        return "🟡 等待观察", "中", zones, "别追，等回踩"
    return "🟢 可以参与（有条件）", "低", zones, "分批参与"


# =========================================================
# 6) 执行方案：固定仓位法 + 风险法（自动算手数/金额）
# =========================================================
@dataclass
class PlanRow:
    trigger: str
    action: str
    delta_pos: float  # 仓位变化（%）用于固定仓位法
    note: str
    ref_price: str    # 计算用参考价：buy_low / buy_mid / sell_high / stop_line

DEFAULT_PLAN = [
    PlanRow("回踩介入区", "试探加仓", 10.0, "判断是否有承接", "buy_low"),
    PlanRow("回踩中位", "主加仓", 15.0, "只在这里赚风险的钱", "buy_mid"),
    PlanRow("突破压力区", "确认加仓", 10.0, "右侧跟随", "sell_high"),
    PlanRow("跌破防守线", "防守退出", -100.0, "保命第一（减到 0~10%）", "stop_line"),
]

def round_to_lot(shares: float, lot_size: int = LOT_SIZE) -> int:
    if shares <= 0:
        return 0
    return int(math.floor(shares / lot_size) * lot_size)

def calc_order_fixed(capital: float, price: float, add_pos_pct: float, lot_size: int = LOT_SIZE):
    if capital <= 0 or price <= 0:
        return 0, 0.0
    budget = capital * (add_pos_pct / 100.0)
    raw_shares = budget / price
    shares = round_to_lot(raw_shares, lot_size=lot_size)
    amount = shares * price
    return shares, amount

def calc_order_risk(capital: float, entry_price: float, stop_price: float, risk_pct: float, lot_size: int = LOT_SIZE):
    """
    风险法：每次最多亏 capital*risk_pct%
    shares = risk_budget / (entry - stop)
    """
    if capital <= 0 or entry_price <= 0 or stop_price <= 0:
        return 0, 0.0, 0.0
    risk_budget = capital * (risk_pct / 100.0)
    per_share_risk = max(entry_price - stop_price, 0.0)
    if per_share_risk <= 0:
        return 0, 0.0, 0.0
    raw_shares = risk_budget / per_share_risk
    shares = round_to_lot(raw_shares, lot_size=lot_size)
    amount = shares * entry_price
    real_risk = shares * per_share_risk
    return shares, amount, real_risk


# =========================================================
# 7) 登录 / 注册 / Pro 激活（移动端主屏）
# =========================================================
def render_auth_gate():
    st.markdown(f"## {APP_NAME}")
    st.caption(SLOGAN)

    st.markdown(
        f"""
        <div style="padding:12px;border-left:6px solid #d32f2f;background:#fff5f5;margin-bottom:16px;border-radius:10px">
        <b>{DISCIPLINE[0]}</b><br>
        <b>{DISCIPLINE[1]}</b><br>
        <b>{DISCIPLINE[2]}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    tabs_auth = st.tabs(["🔐 登录", "🆕 注册", "👀 游客模式"])

    # --- 登录 ---
    with tabs_auth[0]:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("用户名", key="login_username")
            password = st.text_input("密码", type="password", key="login_password")
            submit = st.form_submit_button("登录")

        if submit:
            dfu = load_users()
            row = find_user(dfu, (username or "").strip())
            if row is None:
                st.error("用户不存在。")
                st.stop()
            if not _verify_password(password or "", row["pwd_hash"]):
                st.error("密码错误。")
                st.stop()

            row_dict = dict(row)
            row_dict["last_login_at"] = datetime.now().isoformat(timespec="seconds")
            dfu = upsert_user(dfu, row_dict)
            save_users(dfu)

            set_login(row["username"], row["role"], row["pro_expire"])
            st.success(st.session_state.toast)
            st.rerun()

    # --- 注册 ---
    with tabs_auth[1]:
        with st.form("reg_form", clear_on_submit=True):
            new_user = st.text_input("用户名（字母/数字/下划线，3-20位）", key="reg_username")
            new_pwd = st.text_input("密码（至少6位）", type="password", key="reg_password")
            new_pwd2 = st.text_input("确认密码", type="password", key="reg_password2")
            submit_reg = st.form_submit_button("创建账号")

        if submit_reg:
            new_user = (new_user or "").strip()
            if not re.match(r"^[A-Za-z0-9_]{3,20}$", new_user):
                st.error("用户名不合规：只能字母/数字/下划线，3-20位。")
                st.stop()
            if len(new_pwd or "") < 6:
                st.error("密码至少 6 位。")
                st.stop()
            if (new_pwd or "") != (new_pwd2 or ""):
                st.error("两次密码不一致。")
                st.stop()

            dfu = load_users()
            if find_user(dfu, new_user) is not None:
                st.error("用户名已存在。")
                st.stop()

            now = datetime.now().isoformat(timespec="seconds")
            row = {
                "username": new_user,
                "pwd_hash": _hash_password(new_pwd),
                "role": "free",
                "pro_expire": "",
                "created_at": now,
                "last_login_at": now,
                "note": "",
            }
            dfu = upsert_user(dfu, row)
            save_users(dfu)

            set_login(new_user, "free", "")
            st.success("✅ 注册并登录成功。")
            st.rerun()

    # --- 游客 ---
    with tabs_auth[2]:
        st.info("游客模式 = 免费用户。数据不保存账号。适合手机临时用。")
        if st.button("以游客进入（free）", use_container_width=True):
            st.session_state.logged_in = True
            st.session_state.user = "guest"
            st.session_state.role = "free"
            st.session_state.pro_expire = None
            st.session_state.pro_enabled = False
            st.rerun()

def render_topbar():
    c1, c2, c3, c4 = st.columns([2.4, 1.4, 1.4, 1.0])

    user = st.session_state.user
    role = st.session_state.role
    exp = st.session_state.pro_expire

    if is_admin(role):
        c1.markdown(f"**👤 {user}** · **角色：ADMIN**")
    elif role == "pro" and exp:
        exp_dt = parse_expire(exp)
        exp_txt = exp_dt.strftime("%Y-%m-%d") if exp_dt else exp
        c1.markdown(f"**👤 {user}** · **角色：PRO** · 到期：`{exp_txt}`")
    else:
        c1.markdown(f"**👤 {user}** · **角色：FREE**")

    if role in ("pro", "admin"):
        st.session_state.pro_enabled = c2.toggle("🔓 专业功能开关", value=st.session_state.pro_enabled, key="pro_toggle")
    else:
        c2.markdown("")

    if c3.button("💰 激活/续费 PRO", use_container_width=True):
        st.session_state.show_pro_box = True

    if c4.button("退出", use_container_width=True):
        logout()
        st.rerun()

def render_pro_box():
    if not st.session_state.show_pro_box:
        return

    with st.expander("💰 PRO 激活/续费（最小订阅闭环）", expanded=True):
        st.write("输入激活码即可开通/续费（码来自 pro_codes.csv 或兜底码）。")

        code_in = st.text_input("激活码", placeholder="例如：VIP-8888", key="pro_code_input")

        colA, colB = st.columns(2)
        if colA.button("立即激活/续费", use_container_width=True):
            ok, msg = redeem_code_for_user(st.session_state.user, code_in)
            if ok:
                st.success(msg)
                st.session_state.show_pro_box = False
                st.rerun()
            else:
                st.error(msg)

        if colB.button("关闭", use_container_width=True):
            st.session_state.show_pro_box = False
            st.rerun()


# =========================================================
# 8) 🧰 管理员后台（只增不改：放在 Tab 里）
# =========================================================
def render_admin_panel():
    st.subheader("🧰 管理员后台")

    # --- 用户管理 ---
    st.markdown("### 👥 用户管理（手动开通/续费 PRO）")
    dfu = load_users().copy()
    st.dataframe(dfu[["username", "role", "pro_expire", "created_at", "last_login_at", "note"]], use_container_width=True, hide_index=True)

    with st.form("admin_grant_pro", clear_on_submit=False):
        u = st.text_input("要开通的用户名", key="admin_user_to_grant")
        days = st.number_input("开通天数", 1, 365, PRO_DAYS_DEFAULT, step=1, key="admin_grant_days")
        submit = st.form_submit_button("给 TA 开通/续费 PRO")

    if submit:
        u = (u or "").strip()
        row = find_user(dfu, u)
        if row is None:
            st.error("用户不存在。")
        else:
            now = datetime.now()
            old_exp = (row["pro_expire"] or "").strip()
            cur_exp = parse_expire(old_exp)
            base = cur_exp if (cur_exp and cur_exp > now) else now
            new_exp = base + timedelta(days=int(days))
            row_dict = dict(row)
            row_dict["role"] = "pro" if row_dict["role"].lower() != "admin" else "admin"
            row_dict["pro_expire"] = new_exp.isoformat(timespec="seconds")
            row_dict["note"] = f"admin_grant:{days}d"
            dfu2 = upsert_user(dfu, row_dict)
            save_users(dfu2)
            append_order(u, "ADMIN-GRANT", int(days), old_exp, row_dict["pro_expire"], note="admin_manual")
            st.success(f"✅ 已开通/续费：{u} 到期 {new_exp.strftime('%Y-%m-%d %H:%M')}")

    st.divider()

    # --- 激活码管理 ---
    st.markdown("### 🔑 激活码管理（生成/作废/查看）")
    dfc = load_pro_codes().copy()
    col1, col2, col3 = st.columns(3)
    n = col1.number_input("生成数量", 1, 200, 10, step=1)
    days = col2.number_input("每个码天数", 1, 365, PRO_DAYS_DEFAULT, step=1)
    prefix = col3.text_input("前缀", value="VIP")

    if st.button("生成新激活码", use_container_width=True):
        new_df = gen_codes(int(n), int(days), prefix=prefix.strip() or "VIP")
        st.success(f"✅ 已生成 {len(new_df)} 个新码")
        st.dataframe(new_df, use_container_width=True, hide_index=True)

    # 作废码
    with st.form("admin_void_code", clear_on_submit=False):
        void_code = st.text_input("要作废的 code", key="admin_void_code_input")
        submit_void = st.form_submit_button("作废（void）")

    if submit_void:
        void_code = (void_code or "").strip()
        m = dfc["code"].astype(str).str.upper() == void_code.upper()
        if not m.any():
            st.error("码不存在。")
        else:
            idx = dfc[m].index[0]
            dfc.loc[idx, "status"] = "void"
            dfc.loc[idx, "note"] = "void_by_admin"
            save_pro_codes(dfc)
            st.success("✅ 已作废")

    # 展示码池
    st.dataframe(dfc.sort_values(by=["status", "created_at"], ascending=[True, False]),
                 use_container_width=True, hide_index=True)

    st.divider()

    # --- 订单查看 ---
    st.markdown("### 🧾 激活/续费记录（orders.csv）")
    dfo = load_orders()
    st.dataframe(dfo.sort_values(by=["created_at"], ascending=False), use_container_width=True, hide_index=True)


# =========================================================
# 9) 主界面（Tabs 不跳页）
# =========================================================
def render_app():
    render_topbar()
    render_pro_box()

    st.markdown(f"# {APP_NAME}")
    st.caption(SLOGAN)

    st.markdown(
        f"""
        <div style="padding:12px;border-left:6px solid #d32f2f;background:#fff5f5;margin-bottom:16px;border-radius:10px">
        <b>{DISCIPLINE[0]}</b><br>
        <b>{DISCIPLINE[1]}</b><br>
        <b>{DISCIPLINE[2]}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 股票输入
    code = st.text_input("股票代码（如 600519 / 000001 / 600519.SH）", st.session_state.app_code, key="code_input")
    st.session_state.app_code = code
    ts_code = format_code(code)

    with st.spinner("加载行情中…（tushare 不可用会自动用兜底数据）"):
        df0 = get_data_cached(ts_code, days=220)
        df = indicators(df0)

    verdict, risk, zones, action = make_verdict(df)

    # 首屏 4 卡片
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🛑 今日裁决", verdict)
    c2.metric("⚠️ 风险等级", risk)
    c3.markdown(
        f"""
        📥 **介入区**  
        {zones['buy']}  

        📤 **压力区**  
        {zones['sell']}  

        🛑 **防守线**  
        {zones['stop']}
        """
    )
    c4.metric("👉 行动建议", action)

    st.divider()

    # Tabs（管理员多一个 Tab）
    base_tabs = ["📌 执行方案", "📈 行情图", "📡 每日精选", "❓ Q&A", "👥 99 人方案"]
    if is_admin(st.session_state.role):
        base_tabs.append("🧰 管理员")
    tabs = st.tabs(base_tabs)

    # ===== Tab 1 执行方案 =====
    with tabs[0]:
        st.subheader("📌 分批执行建议（固定仓位 + 风险法自动算手数）")

        last_price = float(df.iloc[-1]["close"])
        buy_mid = (zones["buy_low"] + zones["buy_high"]) / 2.0
        zones["buy_mid"] = buy_mid

        col1, col2, col3 = st.columns(3)
        cost = col1.number_input("你的成本价", value=float(last_price * 0.95), min_value=0.01, step=0.1)
        pos = col2.number_input("当前仓位（%）", 0.0, 100.0, 30.0, step=1.0)
        capital = col3.number_input("账户资金（元）", value=300000.0, min_value=0.0, step=1000.0)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("当前价", f"{zones['close']:.2f}")
        k2.metric("MA60", f"{zones['ma60']:.2f}")
        k3.metric("RSI14", f"{zones['rsi']:.1f}")
        k4.metric("ATR14", f"{zones['atr']:.2f}")

        st.markdown("### 🧠 风险法参数（自动算手数）")
        r1, r2, r3 = st.columns(3)
        risk_pct = r1.number_input("单笔最大风险（%）", 0.1, 10.0, 1.0, step=0.1)
        stop_price = r2.number_input("止损价（默认=防守线）", value=float(zones["stop_line"]), min_value=0.01, step=0.1)
        lot = r3.number_input("一手股数（默认100）", 1, 10000, LOT_SIZE, step=1)

        st.markdown("### ✅ 计划表（两种算法一起给你）")

        rows = []
        for p in DEFAULT_PLAN:
            # 参考入场价
            ref_key = p.ref_price
            entry = zones.get(ref_key, last_price)
            entry = float(entry)

            if p.delta_pos > 0:
                # 固定仓位法
                sh_f, amt_f = calc_order_fixed(capital, entry, p.delta_pos, int(lot))
                # 风险法
                sh_r, amt_r, risk_r = calc_order_risk(capital, entry, stop_price, risk_pct, int(lot))

                rows.append({
                    "触发条件": p.trigger,
                    "动作": p.action,
                    "参考价": f"{entry:.2f}",
                    "固定仓位法": f"{sh_f} 股（≈ {amt_f:,.0f}）",
                    "风险法": f"{sh_r} 股（≈ {amt_r:,.0f}，最大亏≈ {risk_r:,.0f}）",
                    "说明": p.note,
                })
            else:
                # 防守退出：估算需要卖多少股（按当前仓位 pos%）
                cur_value = capital * (pos / 100.0)
                cur_shares = round_to_lot(cur_value / max(last_price, 0.01), int(lot))
                keep_pct = 10.0
                keep_value = capital * (keep_pct / 100.0)
                keep_shares = round_to_lot(keep_value / max(last_price, 0.01), int(lot))
                sell_shares = max(cur_shares - keep_shares, 0)

                rows.append({
                    "触发条件": p.trigger,
                    "动作": p.action,
                    "参考价": f"{entry:.2f}",
                    "固定仓位法": f"卖出 {sell_shares} 股（保留 {keep_shares} 股）",
                    "风险法": "—（这是止损动作）",
                    "说明": p.note,
                })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.error(f"⚠️ 若收盘价跌破 **{zones['stop']}**，原判断失效：先减仓/退出，再重新评估。")

        # PRO：纪律提醒
        if st.session_state.role in ("pro", "admin") and st.session_state.pro_enabled:
            st.markdown("### 🔓 PRO：纪律提醒（少犯错比多赚钱重要）")
            warn = []
            if zones["close"] > zones["sell_low"]:
                warn.append("你现在靠近压力区，**最容易犯的错 = 追涨加仓**。")
            if zones["close"] < zones["ma60"]:
                warn.append("价格在 MA60 下方，**冲动交易更容易变成回撤**。")
            if zones["rsi"] > 70:
                warn.append("RSI 偏热，**别在兴奋里加仓**。")
            if not warn:
                warn.append("当前没有明显的‘必犯错’信号，但仍建议分批、别梭哈。")
            st.info(" \n\n".join([f"- {x}" for x in warn]))

    # ===== Tab 2 行情图 =====
    with tabs[1]:
        fig = go.Figure(data=[go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线"
        )])
        fig.add_trace(go.Scatter(x=df["date"], y=df["MA20"], name="MA20"))
        fig.add_trace(go.Scatter(x=df["date"], y=df["MA60"], name="MA60"))

        # PRO 才显示 MACD 线
        if st.session_state.role in ("pro", "admin") and st.session_state.pro_enabled:
            fig.add_trace(go.Scatter(x=df["date"], y=df["DIF"], name="DIF"))
            fig.add_trace(go.Scatter(x=df["date"], y=df["DEA"], name="DEA"))

        fig.update_layout(height=460, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ===== Tab 3 每日精选 =====
    with tabs[2]:
        st.markdown("**今日仅供观察标的：**")
        st.write("中际旭创 / 工业富联 / 东方财富 / 紫金矿业")
        st.caption("⚠️ 不推票，只给你更安全的选择。")

        if st.session_state.role in ("pro", "admin") and st.session_state.pro_enabled:
            st.markdown("### 🔓 PRO：今日观察逻辑（为什么是它们）")
            st.write(
                "- 不是因为‘会涨’，而是因为它们更常出现在主力流动性里。\n"
                "- 你要的不是命中率，是**犯错成本更低**。\n"
                "- 先用刹车系统过滤掉冲动交易，再谈进攻。"
            )
        else:
            st.info("想看‘为什么是它们’，需要 PRO（不是为了赚钱，是为了少踩坑）。")

    # ===== Tab 4 Q&A =====
    with tabs[3]:
        st.markdown("""
**Q：这是荐股吗？**  
A：不是，这是刹车系统。

**Q：为什么不给预测？**  
A：预测不负责，裁决才负责。

**Q：亏了怎么办？**  
A：只要你按系统执行，亏损属于市场，不属于情绪。
""")

    # ===== Tab 5 99 人方案 =====
    with tabs[4]:
        st.markdown("""
**99 人方案说明**

- 限量用户  
- 不追热点  
- 不迎合情绪  
- 只在关键时刻给结论  

这是一个替你承担交易心理压力的系统。
""")

        # ✅ PRO 锁屏卡：今日一句话结论
        st.markdown("### 🔒 今日一句话结论")
        if st.session_state.role in ("pro", "admin") and st.session_state.pro_enabled:
            # 你也可以后面做成“每日更新文件/管理员后台编辑”
            today_line = (
                f"今天的任务：**别在压力区加仓**；如果跌破 {zones['stop']}，"
                f"先活下来再谈翻身。"
            )
            st.success(today_line)
            st.caption("（这句话不是预测，是交易契约：你违约，就容易亏。）")
        else:
            st.warning("PRO 才能看到“今日一句话结论”。你需要它不是为了赚钱，是为了少犯错。")

    # ===== 管理员 Tab =====
    if is_admin(st.session_state.role):
        with tabs[5]:
            render_admin_panel()


# =========================================================
# 10) 入口
# =========================================================
if not st.session_state.logged_in:
    render_auth_gate()
else:
    render_app()
