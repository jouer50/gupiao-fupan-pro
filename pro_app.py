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
from datetime import datetime, timedelta
import json
import base64 # 用于处理二维码图片

# ✅ 0. 依赖库检查
try:
    import yfinance as yf
except ImportError:
    st.error("🚨 严重错误：缺少 `yfinance` 库，请 pip install yfinance")
    st.stop()

# ==========================================
# 1. 核心配置
# ==========================================
st.set_page_config(
    page_title="阿尔法量研 Pro",
    layout="wide",
    page_icon="🔥",
    initial_sidebar_state="expanded"
)

# 初始化 Session
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if "code" not in st.session_state: st.session_state.code = "600519"
if "paid_code" not in st.session_state: st.session_state.paid_code = ""
if "trade_qty" not in st.session_state: st.session_state.trade_qty = 100
if "daily_picks_cache" not in st.session_state: st.session_state.daily_picks_cache = None
if "enable_realtime" not in st.session_state: st.session_state.enable_realtime = False
if "ts_token" not in st.session_state: st.session_state.ts_token = "你的Tushare接口密钥" 
if "view_mode_idx" not in st.session_state: st.session_state.view_mode_idx = 0 

# ✅ 模拟交易数据结构初始化
if "paper_account" not in st.session_state:
    st.session_state.paper_account = {
        "cash": 1000000.0,
        "holdings": {},
        "history": []
    }

# ✅ 全局变量
ma_s = 5
ma_l = 20
flags = {
    'ma': True, 'boll': True, 'vol': True, 
    'macd': False, 'kdj': False, 'gann': False, 'fib': False, 'chan': False
}

# 核心常量
ADMIN_USER = "ZCX001"
ADMIN_PASS = "123456"
DB_FILE = "users_v69.csv"
KEYS_FILE = "card_keys.csv"
WECHAT_VALID_CODE = "8888"  

# Optional deps
ts = None
bs = None
try: import tushare as ts
except: pass
try: import baostock as bs
except: pass

# 🔥 CSS 样式 (保持原有样式)
ui_css = """
<style>
    /* 全局重置与移动端适配 */
    .stApp {
        background-color: #f7f8fa; 
        font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "SF Pro Text", "Helvetica Neue", sans-serif;
        touch-action: manipulation;
    }
        
    /* 核心内容区去边距 */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 3rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }

    /* 隐藏 Streamlit 默认头部干扰 */
    header[data-testid="stHeader"] { 
        background-color: transparent !important;
        height: 3rem !important;
    }
    footer { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }

    /* 侧边栏按钮 */
    [data-testid="stSidebarCollapsedControl"] {
        position: fixed !important;
        top: 12px !important; 
        left: 12px !important;
        background-color: #ffffff !important;
        border-radius: 50% !important;
        z-index: 9999999 !important;
        width: 40px !important;
        height: 40px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg {
        fill: #333333 !important;
        width: 20px !important;
        height: 20px !important;
    }

    /* 按钮 - APP风格 */
    div.stButton > button {
        background: white;
        color: #333;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        min-height: 44px;
        font-weight: 600;
        width: 100%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03); 
    }
    div.stButton > button:active { transform: scale(0.98); background: #f5f5f5; }

    div.stButton > button[kind="primary"] { 
        background: linear-gradient(135deg, #007AFF 0%, #0056b3 100%); 
        color: white; 
        border: none; 
        box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3);
    }

    /* 卡片容器 */
    .app-card { 
        background-color: #ffffff; 
        border-radius: 16px; 
        padding: 16px; 
        margin-bottom: 12px; 
        box-shadow: 0 2px 10px rgba(0,0,0,0.03); 
        border: 1px solid rgba(0,0,0,0.02);
    }

    /* 状态栏 */
    .market-status-box {
        padding: 12px 16px; 
        border-radius: 12px; 
        margin-bottom: 16px;
        display: flex; align-items: center; justify-content: space-between;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .status-green { background: #e8f5e9; color: #1b5e20; border-left: 4px solid #2e7d32; }
    .status-red { background: #ffebee; color: #b71c1c; border-left: 4px solid #c62828; }
    .status-yellow { background: #fffde7; color: #f57f17; border-left: 4px solid #fbc02d; }

    /* 价格大字 */
    .big-price-box { text-align: center; margin: 10px 0 20px 0; }
    .price-main { font-size: 42px; font-weight: 800; line-height: 1; letter-spacing: -1px; font-family: "SF Pro Display", sans-serif; }
    .price-sub { font-size: 15px; font-weight: 600; margin-left: 6px; padding: 2px 6px; border-radius: 6px; background: rgba(0,0,0,0.05); }

    /* AI 对话框 */
    .ai-chat-box {
        background: #f2f8ff; border-radius: 12px; padding: 15px; margin-bottom: 15px;
        border-left: 4px solid #007AFF; 
    }
    
    /* 锁定层样式 */
    .locked-container { position: relative; overflow: hidden; }
    .locked-blur { filter: blur(8px); user-select: none; opacity: 0.5; pointer-events: none; transition: filter 0.3s; }
    .locked-overlay {
        position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: rgba(255, 255, 255, 0.7); z-index: 10;
        backdrop-filter: blur(3px);
    }
    .lock-teaser {
        font-size: 14px; color: #333; margin: 5px 0; font-weight: 500;
    }
    
    /* Expander 优化 */
    .streamlit-expanderHeader {
        background-color: #fff;
        border-radius: 12px;
        font-size: 15px;
        font-weight: 600;
        border: 1px solid #f0f0f0;
    }

    /* 病毒海报样式 */
    .poster-box {
        background: linear-gradient(135deg, #2b32b2 0%, #14