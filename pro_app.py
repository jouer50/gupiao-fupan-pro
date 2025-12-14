if not is_admin:
        with st.expander("💎 会员与充值", expanded=True):
            # ====== 修复开始：增加安全检查 ======
            df_u = load_users()
            u_row = df_u[df_u['username'] == user]
            
            if not u_row.empty:
                my_quota = u_row['quota'].iloc[0]
            else:
                # 异常情况：Session里有登录态，但数据库里没这个人（通常发生在数据库重置后）
                my_quota = 0 
                st.warning("⚠️ 用户数据不同步，请退出重登")
                if st.button("🔄 立即修复 (退出登录)"):
                    st.session_state.clear()
                    st.rerun()
            # ====== 修复结束 ======

            st.write(f"当前积分: **{my_quota}**")
            
            tab_pay, tab_vip = st.tabs(["充值", "兑换VIP"])
            # ... 下面的代码保持不变 ...