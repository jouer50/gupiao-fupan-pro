# ✅ 修改后：仅显示关键点位 (去除了冗余的 AI 研报文字)
def generate_simple_levels(df):
    c = df.iloc[-1]
    
    # === 计算逻辑保持不变 ===
    # 支撑位：取过去20日最低价
    support = df['low'].tail(20).min()
    # 压力位：取过去20日最高价
    resistance = df['high'].tail(20).max()
    # ATR用于止损止盈
    atr = c['ATR14'] if c['ATR14'] > 0 else c['close'] * 0.02
    # 止损：当前价格向下2倍ATR
    stop_loss = c['close'] - 2.0 * atr
    # 止盈：当前价格向上3倍ATR
    take_profit = c['close'] + 3.0 * atr

    # === 构造极简 HTML ===
    html = f"""
    <div class="app-card" style="margin-top: 20px;">
        <div style="font-size:16px; font-weight:800; color:#333; margin-bottom:15px; border-left:4px solid #2962ff; padding-left:10px;">
            🎯 交易关键位测算 (Key Levels)
        </div>
        
        <div class="final-grid" style="border:none; padding-top:0; margin-top:0;">
            <div class="final-item">
                <div class="final-item-val" style="color:#2e7d32">{support:.2f}</div>
                <div class="final-item-lbl">📉 强支撑 (Support)</div>
            </div>
             <div class="final-item">
                <div class="final-item-val" style="color:#c62828">{resistance:.2f}</div>
                <div class="final-item-lbl">📈 强压力 (Resist)</div>
            </div>
            <div class="final-item">
                <div class="final-item-val" style="color:#ff9800">{take_profit:.2f}</div>
                <div class="final-item-lbl">💰 建议止盈 (Profit)</div>
            </div>
            <div class="final-item">
                <div class="final-item-val" style="color:#333">{stop_loss:.2f}</div>
                <div class="final-item-lbl">🛡️ 建议止损 (Stop)</div>
            </div>
        </div>
    </div>
    """
    return html