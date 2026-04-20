import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC, SVR
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

def show_phase6():
    st.title("🏆 Phase 6: Final Benchmark & Leaderboard")
    st.markdown("---")

    if 'model_data' not in st.session_state:
        st.error("⚠️ Vui lòng chạy Tiền xử lý ở **Phase 4** trước để tạo dữ liệu Train/Test.")
        return

    data = st.session_state.model_data
    X_train_s, X_test_s = data['X_train'], data['X_test']
    y_tier_train, y_tier_test = data['y_tier_train'], data['y_tier_test']
    y_train_rev_s, y_test_rev_s = data['y_rev_train'], data['y_rev_test']
    y_vote_train, y_vote_test = data['y_vote_train'], data['y_vote_test']
    scaler_y_rev = data['scaler_y_rev']
    
    # Giải mã Target doanh thu để đánh giá thực tế
    y_rev_test_real = np.expm1(scaler_y_rev.inverse_transform(y_test_rev_s.reshape(-1, 1)).flatten())

    st.info("💡 **Comprehensive Benchmark (Đánh giá Toàn diện):** Trạm này sẽ tự động huấn luyện, kiểm thử và xếp hạng TẤT CẢ các kiến trúc học máy bạn đã thiết kế. Các mô hình sẽ thi đấu công bằng trên cùng một tập Test ẩn để tìm ra 'Nhà vô địch' cho từng bài toán.")

    # =================================================================
    # HÀM CHẠY BENCHMARK TOÀN DIỆN (LÕI XỬ LÝ)
    # =================================================================
    def run_comprehensive_benchmark():
        results_rev = []
        results_vote = []
        
        # --- 1. CHUẨN BỊ LÕI SVC (CHO HYBRID) ---
        svc_model = SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=42)
        svc_model.fit(X_train_s, y_tier_train)
        train_probs = svc_model.predict_proba(X_train_s)
        test_probs = svc_model.predict_proba(X_test_s)

        # Danh sách cấu hình Doanh thu
        rev_configs = [
            {"name": "Hybrid (SVC + XGBoost)", "type": "hybrid", "model": "xgb"},
            {"name": "Hybrid (SVC + SVR)", "type": "hybrid", "model": "svr"},
            {"name": "Hybrid (SVC + Linear)", "type": "hybrid", "model": "linear"},
            {"name": "Standalone (XGBoost)", "type": "standalone", "model": "xgb"},
            {"name": "Standalone (SVR)", "type": "standalone", "model": "svr"},
            {"name": "Baseline (Linear Reg)", "type": "standalone", "model": "linear"}
        ]

        # --- 2. VÒNG LẶP HUẤN LUYỆN DOANH THU ---
        for cfg in rev_configs:
            if cfg["type"] == "hybrid":
                rev_experts = {}
                for t in [0, 1, 2]:
                    weights = np.ascontiguousarray(train_probs[:, t]) # Fix C-contiguous
                    if cfg["model"] == "xgb": reg = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
                    elif cfg["model"] == "svr": reg = SVR(kernel='rbf', C=2.0, epsilon=0.1)
                    else: reg = LinearRegression()
                    
                    reg.fit(X_train_s, y_train_rev_s, sample_weight=weights)
                    rev_experts[t] = reg
                
                y_rev_pred_log = np.zeros(len(X_test_s))
                for i in range(len(X_test_s)):
                    e_preds = [rev_experts[t].predict([X_test_s[i]])[0] for t in [0, 1, 2]]
                    y_rev_pred_log[i] = np.sum(test_probs[i] * np.array(e_preds))
                    
            else: # Standalone
                if cfg["model"] == "xgb": reg = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
                elif cfg["model"] == "svr": reg = SVR(kernel='rbf', C=2.0, epsilon=0.1)
                else: reg = LinearRegression()
                
                reg.fit(X_train_s, y_train_rev_s)
                y_rev_pred_log = reg.predict(X_test_s)

            # Giải mã & Tính Metrics
            y_rev_pred = np.maximum(np.expm1(scaler_y_rev.inverse_transform(y_rev_pred_log.reshape(-1, 1)).flatten()), 0)
            r2 = r2_score(y_rev_test_real, y_rev_pred)
            mae = mean_absolute_error(y_rev_test_real, y_rev_pred)
            rmse = np.sqrt(mean_squared_error(y_rev_test_real, y_rev_pred))
            
            results_rev.append({"Model": cfg["name"], "R² Score": r2, "MAE": mae, "RMSE": rmse, "preds": y_rev_pred})

        # --- 3. VÒNG LẶP HUẤN LUYỆN ĐIỂM SỐ ---
        vote_configs = [
            {"name": "XGBoost (Cây Quyết định)", "model": XGBRegressor(n_estimators=100, random_state=42)},
            {"name": "SVR (Ống Lọc Nhiễu)", "model": SVR(kernel='rbf', C=5.0, epsilon=0.2)},
            {"name": "Baseline (Linear Reg)", "model": LinearRegression()}
        ]

        for cfg in vote_configs:
            model = cfg["model"]
            model.fit(X_train_s, y_vote_train)
            y_vote_pred = np.clip(model.predict(X_test_s), 0, 10)
            
            r2 = r2_score(y_vote_test, y_vote_pred)
            mae = mean_absolute_error(y_vote_test, y_vote_pred)
            rmse = np.sqrt(mean_squared_error(y_vote_test, y_vote_pred))
            
            results_vote.append({"Model": cfg["name"], "R² Score": r2, "MAE": mae, "RMSE": rmse, "preds": y_vote_pred})

        return pd.DataFrame(results_rev), pd.DataFrame(results_vote)

    # =================================================================
    # HÀM TÍNH TOÁN DELTA (CHÊNH LỆCH SO VỚI BASELINE)
    # =================================================================
    def add_deltas(df, baseline_name):
        base_row = df[df['Model'] == baseline_name].iloc[0]
        df['Δ R²'] = df['R² Score'] - base_row['R² Score']
        df['Δ MAE'] = df['MAE'] - base_row['MAE']
        df['Δ RMSE'] = df['RMSE'] - base_row['RMSE']
        
        # Sắp xếp lại thứ tự cột cho đẹp
        cols = ['Model', 'R² Score', 'Δ R²', 'MAE', 'Δ MAE', 'RMSE', 'Δ RMSE', 'preds']
        return df[cols]

    # =================================================================
    # HÀM ĐỊNH DẠNG MÀU SẮC (PANDAS STYLER)
    # =================================================================
    def color_r2_delta(val):
        if val > 0.0001: return 'color: #00E676; font-weight: bold;' # Xanh lá (Tốt)
        elif val < -0.0001: return 'color: #FF1744; font-weight: bold;' # Đỏ (Xấu)
        return 'color: gray;'

    def color_error_delta(val):
        if val < -0.0001: return 'color: #00E676; font-weight: bold;' # Giảm sai số -> Xanh lá (Tốt)
        elif val > 0.0001: return 'color: #FF1744; font-weight: bold;' # Tăng sai số -> Đỏ (Xấu)
        return 'color: gray;'

    # =================================================================
    # NÚT BẤM KÍCH HOẠT BENCHMARK
    # =================================================================
    if st.button("🚀 KÍCH HOẠT BENCHMARK TOÀN DIỆN (TỰ ĐỘNG XẾP HẠNG)", use_container_width=True):
        with st.status("Đang chạy Giải đấu Machine Learning...", expanded=True) as status:
            st.write("⏳ Đang phân luồng dữ liệu và Huấn luyện...")
            df_rev, df_vote = run_comprehensive_benchmark()
            
            # Xử lý tính Delta
            df_rev = add_deltas(df_rev, "Baseline (Linear Reg)")
            df_vote = add_deltas(df_vote, "Baseline (Linear Reg)")
            
            # Sắp xếp mặc định theo R2 từ cao tới thấp
            st.session_state.df_rev = df_rev.sort_values(by='R² Score', ascending=False).reset_index(drop=True)
            st.session_state.df_vote = df_vote.sort_values(by='R² Score', ascending=False).reset_index(drop=True)
            
            status.update(label="✅ Đã hoàn thành Benchmark! Vui lòng xem Bảng xếp hạng bên dưới.", state="complete")

    # =================================================================
    # HIỂN THỊ KẾT QUẢ VÀ BIỂU ĐỒ (LEADERBOARD)
    # =================================================================
    if 'df_rev' in st.session_state:
        tab_board_rev, tab_board_vote, tab_cases = st.tabs(["💰 Bảng Vàng Doanh Thu", "⭐ Bảng Vàng Điểm Số", "🔎 10 Case Studies Nổi Bật"])

        # -----------------------------------------------------------
        # TAB 1: DOANH THU
        # -----------------------------------------------------------
        with tab_board_rev:
            df_r = st.session_state.df_rev.copy()
            st.subheader("Bảng Xếp Hạng & Mức Độ Đột Phá (Bài toán Doanh thu)")
            st.caption("🖱️ *Mẹo: Click vào tiêu đề cột (VD: R² Score, MAE) để tự động Sắp xếp (Sort) từ cao tới thấp hoặc ngược lại.*")
            
            # Áp dụng Format và CSS cho DataFrame
            format_dict_rev = {
                'R² Score': '{:.4f}', 'Δ R²': '{:+.4f}',
                'MAE': '${:,.0f}', 'Δ MAE': '${:+,.0f}',
                'RMSE': '${:,.0f}', 'Δ RMSE': '${:+,.0f}'
            }
            
            # Lọc bỏ cột 'preds' để không hiện ra bảng
            view_cols = [c for c in df_r.columns if c != 'preds']
            
            styled_df_r = (df_r[view_cols].style
                           .format(format_dict_rev)
                           .map(color_r2_delta, subset=['Δ R²'])
                           .map(color_error_delta, subset=['Δ MAE', 'Δ RMSE'])
                           .highlight_max(subset=['R² Score'], color='rgba(46, 204, 113, 0.2)')
                           .highlight_min(subset=['MAE', 'RMSE'], color='rgba(46, 204, 113, 0.2)'))
            
            st.dataframe(styled_df_r, use_container_width=True)

            st.markdown("### 📊 Sơ Đồ Phân Tích Lực Lượng")
            fig_r, axes_r = plt.subplots(1, 2, figsize=(15, 6))
            
            # Chart 1: R2 Score
            sns.barplot(ax=axes_r[0], data=df_r, x='Model', y='R² Score', palette='viridis')
            axes_r[0].set_xticklabels(axes_r[0].get_xticklabels(), rotation=45, ha='right')
            axes_r[0].set_title('ĐỘ CHÍNH XÁC R² (CÀNG CAO CÀNG TỐT)', fontweight='bold')
            for p in axes_r[0].patches:
                axes_r[0].annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

            # Chart 2: Errors (MAE & RMSE)
            df_r_melt = df_r.melt(id_vars='Model', value_vars=['MAE', 'RMSE'], var_name='Metric', value_name='Sai số (Triệu USD)')
            df_r_melt['Sai số (Triệu USD)'] = df_r_melt['Sai số (Triệu USD)'] / 1e6 # Đổi ra triệu USD cho biểu đồ gọn
            
            sns.barplot(ax=axes_r[1], data=df_r_melt, x='Model', y='Sai số (Triệu USD)', hue='Metric', palette='magma')
            axes_r[1].set_xticklabels(axes_r[1].get_xticklabels(), rotation=45, ha='right')
            axes_r[1].set_title('SAI SỐ BÌNH QUÂN (CÀNG THẤP CÀNG TỐT)', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig_r)

        # -----------------------------------------------------------
        # TAB 2: ĐIỂM SỐ
        # -----------------------------------------------------------
        with tab_board_vote:
            df_v = st.session_state.df_vote.copy()
            st.subheader("Bảng Xếp Hạng & Mức Độ Đột Phá (Bài toán Điểm số)")
            st.caption("🖱️ *Mẹo: Click vào tiêu đề cột để Sắp xếp.*")
            
            format_dict_vote = {
                'R² Score': '{:.4f}', 'Δ R²': '{:+.4f}',
                'MAE': '{:.4f}', 'Δ MAE': '{:+.4f}',
                'RMSE': '{:.4f}', 'Δ RMSE': '{:+.4f}'
            }
            
            view_cols_v = [c for c in df_v.columns if c != 'preds']
            
            styled_df_v = (df_v[view_cols_v].style
                           .format(format_dict_vote)
                           .map(color_r2_delta, subset=['Δ R²'])
                           .map(color_error_delta, subset=['Δ MAE', 'Δ RMSE'])
                           .highlight_max(subset=['R² Score'], color='rgba(52, 152, 219, 0.2)')
                           .highlight_min(subset=['MAE', 'RMSE'], color='rgba(52, 152, 219, 0.2)'))
            
            st.dataframe(styled_df_v, use_container_width=True)

            st.markdown("### 📊 Sơ Đồ Phân Tích Lực Lượng")
            fig_v, axes_v = plt.subplots(1, 2, figsize=(15, 5))
            
            sns.barplot(ax=axes_v[0], data=df_v, x='R² Score', y='Model', palette='cubehelix')
            axes_v[0].set_title('ĐỘ CHÍNH XÁC R² (CÀNG CAO CÀNG TỐT)', fontweight='bold')
            
            df_v_melt = df_v.melt(id_vars='Model', value_vars=['MAE', 'RMSE'], var_name='Metric', value_name='Sai lệch (Sao)')
            sns.barplot(ax=axes_v[1], data=df_v_melt, x='Model', y='Sai lệch (Sao)', hue='Metric', palette='crest')
            axes_v[1].set_title('SAI SỐ BÌNH QUÂN (CÀNG THẤP CÀNG TỐT)', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig_v)

        # -----------------------------------------------------------
        # TAB 3: CASE STUDIES (THỰC CHIẾN)
        # -----------------------------------------------------------
        with tab_cases:
            st.markdown("### 🔎 Kiểm định chéo trên 10 bộ phim ngẫu nhiên (Tập Test)")
            
            np.random.seed(42)
            sample_idx = np.random.choice(len(y_rev_test_real), 10, replace=False)
            
            # Lấy Model Vô Địch và Bét bảng Doanh thu
            df_r = st.session_state.df_rev
            top_rev_model_name = df_r.iloc[0]['Model']
            top_rev_preds = df_r.iloc[0]['preds']
            
            base_rev_model_name = df_r[df_r['Model'] == 'Baseline (Linear Reg)'].iloc[0]['Model']
            base_rev_preds = df_r[df_r['Model'] == 'Baseline (Linear Reg)'].iloc[0]['preds']
            
            df_cases = pd.DataFrame({
                'Mã Phim': [f"Movie-{i}" for i in range(1, 11)],
                'Doanh Thu Thực': y_rev_test_real[sample_idx],
                f'Top 1 AI ({top_rev_model_name})': top_rev_preds[sample_idx],
                f'Baseline ({base_rev_model_name})': base_rev_preds[sample_idx],
            })
            
            # Vẽ biểu đồ Line chart
            fig_c, ax_c = plt.subplots(figsize=(12, 5))
            ax_c.plot(df_cases['Mã Phim'], df_cases['Doanh Thu Thực']/1e6, marker='o', markersize=8, linewidth=3, label='Đường Doanh Thu Thực Tế', color='#2ecc71')
            ax_c.plot(df_cases['Mã Phim'], df_cases[f'Top 1 AI ({top_rev_model_name})']/1e6, marker='x', markersize=8, linestyle='--', linewidth=2, label=f'AI Vô Địch ({top_rev_model_name})', color='#e74c3c')
            ax_c.plot(df_cases['Mã Phim'], df_cases[f'Baseline ({base_rev_model_name})']/1e6, marker='s', linestyle=':', alpha=0.5, label='Mô hình Cơ sở (Linear)', color='#95a5a6')
            
            ax_c.set_ylabel("Triệu USD")
            ax_c.set_title("QUỸ ĐẠO BÁM SÁT THỰC TẾ CỦA AI VÔ ĐỊCH VS BASELINE", fontweight='bold')
            ax_c.legend()
            ax_c.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig_c)
            
            # Bảng chi tiết
            for col in df_cases.columns[1:]:
                df_cases[col] = df_cases[col].map('${:,.0f}'.format)
            st.dataframe(df_cases, use_container_width=True, hide_index=True)
            
            st.info("💡 **Giải thích Biểu đồ Line:** Đường đứt nét màu đỏ (AI Vô địch) có khả năng bám theo độ giật cục của đường xanh lá (Thực tế). Trong khi đó, mô hình Baseline màu xám dự báo rất 'bình bình', chứng tỏ Linear Regression bất lực trước các quy luật phi tuyến của thế giới điện ảnh.")