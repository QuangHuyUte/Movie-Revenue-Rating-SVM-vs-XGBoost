import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

def show_phase5():
    st.title("🤖 Phase 5: Multi-task Model Training & Ensemble")
    st.markdown("---")

    if 'model_data' not in st.session_state:
        st.warning("⚠️ Vui lòng quay lại **Phase 4**, chạy Tiền xử lý để tạo dữ liệu Train/Test trước khi huấn luyện mô hình.")
        return

    data = st.session_state.model_data
    
    # =====================================================================
    # 🛑 FIX LỖI C-CONTIGUOUS TẦNG 1: ÉP KHỐI DỮ LIỆU ĐẦU VÀO
    # =====================================================================
    X_train_s = np.ascontiguousarray(data['X_train'], dtype=np.float64)
    X_test_s = np.ascontiguousarray(data['X_test'], dtype=np.float64)
    
    # Classification targets nên để int32 cho an toàn
    y_tier_train = np.ascontiguousarray(data['y_tier_train'], dtype=np.int32)
    y_tier_test = np.ascontiguousarray(data['y_tier_test'], dtype=np.int32)
    
    y_train_rev_s = np.ascontiguousarray(data['y_rev_train'], dtype=np.float64)
    y_test_rev_s = np.ascontiguousarray(data['y_rev_test'], dtype=np.float64)
    y_vote_train = np.ascontiguousarray(data['y_vote_train'], dtype=np.float64)
    y_vote_test = np.ascontiguousarray(data['y_vote_test'], dtype=np.float64)
    scaler_y_rev = data['scaler_y_rev']
    # =====================================================================

    feature_names = data.get('feature_names', [f"Feature {i}" for i in range(X_train_s.shape[1])])

    tab_rev, tab_vote = st.tabs(["💰 Bài toán 1: Doanh Thu (Hybrid vs Standalone)", "⭐ Bài toán 2: Điểm Số (SVR vs XGBoost vs Linear)"])

    # =================================================================
    # TAB 1: BÀI TOÁN DOANH THU 
    # =================================================================
    with tab_rev:
        st.subheader("Kiến trúc Dự báo: Hệ thống Lai (Phân tầng) vs Mô hình Độc lập (Toàn cục)")
        
        col_m1, col_m2 = st.columns([1, 1.5])
        
        with col_m1:
            st.write("⚙️ **Cấu hình Đường ống (Pipeline)**")
            
            # MỞ RỘNG 5 LỰA CHỌN CHIẾN LƯỢC
            rev_model_choice = st.radio(
                "Chọn Cấu trúc Thuật toán:", 
                ["SVC + XGBoost", "SVC + SVR", "SVC + Linear Regression", "XGBoost (Độc lập)", "SVR (Độc lập)"]
            )
            
            if "SVC" in rev_model_choice:
                st.write("**1. Tham số Phân luồng (SVC)**")
                svc_c = st.slider("Hệ số phạt SVC (C)", 0.1, 10.0, 1.0, 0.1, help="(⭐ Tối ưu: 1.0)")
                st.write(f"**2. Tham số Hồi quy ({rev_model_choice.split(' + ')[1]})**")
            else:
                st.write(f"**1. Tham số Mô hình ({rev_model_choice.replace(' (Độc lập)', '')})**")
                svc_c = 1.0 # Mặc định để train ẩn cho UI Phase 7 không bị lỗi

            if "XGBoost" in rev_model_choice:
                reg_param1 = st.slider("Số cây (n_estimators)", 50, 300, 200, 50, help="(⭐ Tối ưu: 200)")
                reg_param2 = st.slider("Tốc độ học (learning_rate)", 0.01, 0.3, 0.05, 0.01, help="(⭐ Tối ưu: 0.05)")
            elif "SVR" in rev_model_choice:
                reg_param1 = st.slider("Hệ số phạt SVR (C)", 0.1, 10.0, 2.0, 0.1, help="(⭐ Tối ưu: 2.0)")
                reg_param2 = st.slider("Ống dung sai (Epsilon)", 0.01, 1.0, 0.1, 0.01, help="(⭐ Tối ưu: 0.1)")
            else:
                st.info("💡 Linear Regression làm Baseline so sánh, không có siêu tham số tinh chỉnh.")
            
            btn_train_rev = st.button("🚀 Bắt đầu Huấn luyện Doanh thu", key="btn_rev")

        with col_m2:
            st.info(f"🧠 **Giải mã Hộp đen: {rev_model_choice}**")
            if "Độc lập" in rev_model_choice:
                st.write("**▶ Kiến trúc: Mô hình Toàn cục (Global Single Model)**")
                st.write("* **Cơ chế:** Thuật toán nhắm mắt phớt lờ các phân tầng rủi ro. Nó nạp toàn bộ 100% dữ liệu vào một cỗ máy duy nhất để tìm ra quy luật quy đổi từ Đặc trưng sang Tiền.")
                st.write("* **Mục đích:** Đóng vai trò là đối trọng (Baseline nâng cao) để chứng minh xem việc tốn công thiết kế Hệ thống phân luồng SVC phía trên có thực sự mang lại hiệu quả vượt trội hay không.")
            else:
                st.write("**▶ Kiến trúc: Hệ thống Lai (Hybrid Soft-Weighting)**")
                st.latex(r"\min_{w,b,\xi} \frac{1}{2}||w||^2 + C \sum \xi_i")
                st.write("* **Giai đoạn 1:** SVC khoanh vùng phim thành 3 tầng: Lỗ, Lời, Bom tấn.")
                st.write("* **Giai đoạn 2:** Dùng xác suất của SVC làm **Trọng số mẫu (Sample Weight)** ép các Chuyên gia tập trung vào vùng chuyên môn của mình trên toàn bộ tập dữ liệu, chống rò rỉ Out-of-Distribution triệt để.")

        # --- XỬ LÝ HUẤN LUYỆN DOANH THU ---
        if btn_train_rev:
            with st.status(f"Đang huấn luyện {rev_model_choice}...", expanded=True) as status:
                try:
                    # 1. Luôn train SVC ngầm để giữ PCA UI và tương thích Phase 7
                    svc_model = SVC(kernel='rbf', C=svc_c, class_weight='balanced', probability=True, random_state=42)
                    svc_model.fit(X_train_s, y_tier_train)
                    train_probs = svc_model.predict_proba(X_train_s)

                    rev_experts = {}
                    
                    # 2. XỬ LÝ MÔ HÌNH ĐỘC LẬP
                    if "Độc lập" in rev_model_choice:
                        if "XGBoost" in rev_model_choice: 
                            global_reg = XGBRegressor(n_estimators=reg_param1, learning_rate=reg_param2, max_depth=6, random_state=42)
                        elif "SVR" in rev_model_choice: 
                            global_reg = SVR(kernel='rbf', C=reg_param1, epsilon=reg_param2)
                        
                        global_reg.fit(X_train_s, y_train_rev_s)
                        # Thủ thuật: Gán 1 mô hình cho cả 3 Tier. Toán học Soft-weighting ở Phase 7 sẽ tự triệt tiêu xác suất!
                        rev_experts = {0: global_reg, 1: global_reg, 2: global_reg}
                    
                    # 3. XỬ LÝ MÔ HÌNH LAI (HYBRID)
                    else:
                        for t in [0, 1, 2]:
                            # =====================================================================
                            # 🛑 FIX LỖI C-CONTIGUOUS TẦNG 2 (SVR SAMPLE_WEIGHT CRASH)
                            # Slice array làm vỡ RAM, phải đóng gói lại bằng np.ascontiguousarray
                            # =====================================================================
                            weights = np.ascontiguousarray(train_probs[:, t], dtype=np.float64)
                            
                            if "XGBoost" in rev_model_choice: 
                                reg = XGBRegressor(n_estimators=reg_param1, learning_rate=reg_param2, max_depth=6, random_state=42)
                            elif "SVR" in rev_model_choice: 
                                reg = SVR(kernel='rbf', C=reg_param1, epsilon=reg_param2)
                            else: 
                                reg = LinearRegression()
                            
                            reg.fit(X_train_s, y_train_rev_s, sample_weight=weights)
                            rev_experts[t] = reg

                    # ĐÁNH GIÁ TRÊN TẬP TEST
                    y_probs = svc_model.predict_proba(X_test_s)
                    y_test_pred_s = np.zeros(len(X_test_s))
                    
                    for i in range(len(X_test_s)):
                        # FIX LỖI TẦNG 3: Ép mảng 1D sang khối 2D liền mạch để predict không bị văng
                        x_i = np.ascontiguousarray(X_test_s[i].reshape(1, -1))
                        expert_preds = [rev_experts[t].predict(x_i)[0] for t in [0, 1, 2]]
                        y_test_pred_s[i] = np.sum(y_probs[i] * np.array(expert_preds))

                    # Giải mã
                    y_test_pred_usd = np.maximum(np.expm1(scaler_y_rev.inverse_transform(y_test_pred_s.reshape(-1, 1)).flatten()), 0)
                    y_test_true_usd = np.expm1(scaler_y_rev.inverse_transform(y_test_rev_s.reshape(-1, 1)).flatten())

                    r2_rev = r2_score(y_test_true_usd, y_test_pred_usd)
                    mae_rev = mean_absolute_error(y_test_true_usd, y_test_pred_usd)
                    
                    st.session_state.metrics_rev = f"💰 KẾT QUẢ TẬP TEST ({rev_model_choice}): R² = {r2_rev:.4f} | Sai số MAE = ${mae_rev:,.0f}"
                    st.session_state.rev_models = {'svc': svc_model, 'experts': rev_experts, 'pipeline': rev_model_choice, 'epsilon': reg_param2 if "SVR" in rev_model_choice else None}
                    
                    st.session_state.fitted_models = st.session_state.get('fitted_models', {})
                    st.session_state.fitted_models['svc'] = svc_model
                    st.session_state.fitted_models['revenue_experts'] = rev_experts

                    status.update(label="✅ Huấn luyện thành công! Xem kiểm định trên tập Test.", state="complete")
                except Exception as e: st.error(f"Lỗi: {e}")

        # --- VẼ BIỂU ĐỒ DOANH THU (CHỈ DÙNG TẬP TEST - CHỐNG VISUAL LEAK) ---
        if 'rev_models' in st.session_state:
            if st.session_state.rev_models['pipeline'] != rev_model_choice:
                st.warning("⚠️ Bạn vừa đổi sang mô hình khác. Hãy bấm **Huấn luyện** để xem kết quả mới!")
            else:
                st.markdown("---")
                st.success(f"**{st.session_state.metrics_rev}**")
                
                col_c1, col_c2 = st.columns(2)
                
                with col_c1:
                    fig_svc, ax_svc = plt.subplots(figsize=(7, 5))
                    pca_2d = PCA(n_components=2)
                    
                    # FIX LỖI TẦNG 4: PCA Fit
                    X_test_pca = np.ascontiguousarray(pca_2d.fit_transform(X_test_s))
                    svc_2d = SVC(kernel='rbf', C=svc_c if "SVC" in rev_model_choice else 1.0, class_weight='balanced').fit(X_test_pca, y_tier_test)
                    x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
                    y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
                    Z_svc = svc_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
                    
                    ax_svc.contourf(xx, yy, Z_svc, alpha=0.3, cmap='Set1')
                    scatter = ax_svc.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_tier_test, cmap='Set1', edgecolor='k', s=30, alpha=0.8)
                    if "Độc lập" in rev_model_choice:
                        ax_svc.set_title('PHÂN LỚP RỦI RO (BỊ MÔ HÌNH ĐỘC LẬP BỎ QUA)', fontweight='bold', color='gray')
                    else:
                        ax_svc.set_title('ĐỘ PHỦ BIÊN GIỚI SVC TRÊN TẬP TEST (PCA)', fontweight='bold')
                    ax_svc.legend(*scatter.legend_elements(), title="Tier (0:Lỗ, 1:Lời, 2:Hit)")
                    st.pyplot(fig_svc)

                with col_c2:
                    pipeline = st.session_state.rev_models['pipeline']
                    experts = st.session_state.rev_models['experts']
                    
                    # NẾU LÀ MÔ HÌNH ĐỘC LẬP -> VẼ 1 BIỂU ĐỒ CHUNG
                    if "Độc lập" in pipeline:
                        st.markdown(f"**KIỂM ĐỊNH SAI SỐ TOÀN CỤC (TRÊN TẬP TEST)**")
                        expert = experts[0] # Lấy bản gốc
                        if "XGBoost" in pipeline:
                            fig_xgb, ax_xgb = plt.subplots(figsize=(7, 4.5))
                            importances = expert.feature_importances_
                            indices = np.argsort(importances)[-10:]
                            ax_xgb.barh(range(10), importances[indices], color='#3498db')
                            feat_names = [str(feature_names[i])[:15] for i in indices]
                            ax_xgb.set_yticks(range(10))
                            ax_xgb.set_yticklabels(feat_names)
                            ax_xgb.set_xlabel('Trọng số F-Score (Global Model)')
                            st.pyplot(fig_xgb)
                        elif "SVR" in pipeline:
                            fig_svr, ax_svr = plt.subplots(figsize=(7, 4.5))
                            y_pred_z = expert.predict(X_test_s)
                            residuals = y_test_rev_s - y_pred_z
                            eps = st.session_state.rev_models['epsilon']
                            inside = np.abs(residuals) <= eps
                            outside = ~inside
                            
                            ax_svr.scatter(y_pred_z[outside], residuals[outside], color='#e74c3c', edgecolor='white', s=40, label='Ngoại lai')
                            ax_svr.scatter(y_pred_z[inside], residuals[inside], color='#bdc3c7', alpha=0.5, s=20, label='Trong ống (An toàn)')
                            ax_svr.axhline(y=eps, color='#e74c3c', linestyle='--', lw=2, label=f'± Epsilon ({eps})')
                            ax_svr.axhline(y=-eps, color='#e74c3c', linestyle='--', lw=2)
                            ax_svr.fill_between([min(y_pred_z), max(y_pred_z)], eps, -eps, color='#e74c3c', alpha=0.1)
                            ax_svr.axhline(y=0, color='#2c3e50', lw=2)
                            ax_svr.set_title(f'PHÂN TÁN SAI SỐ TEST (GLOBAL SVR)', fontweight='bold')
                            ax_svr.set_xlabel('Dự báo Z-Score')
                            ax_svr.set_ylabel('Sai số Residual')
                            ax_svr.legend(loc='upper right')
                            st.pyplot(fig_svr)

                    # NẾU LÀ HYBRID -> VẼ 3 TAB CHO 3 CHUYÊN GIA
                    else:
                        st.markdown(f"**KIỂM ĐỊNH SAI SỐ CHUYÊN GIA (TRÊN TẬP TEST)**")
                        t0, t1, t2 = st.tabs(["🔴 Tier 0 (Lỗ)", "🟡 Tier 1 (Lời nhẹ)", "🟢 Tier 2 (Bom tấn)"])
                        colors = ['#e74c3c', '#f1c40f', '#2ecc71']
                        
                        for idx_tier, (tab, color) in enumerate(zip([t0, t1, t2], colors)):
                            with tab:
                                expert = experts.get(idx_tier)
                                if expert is not None:
                                    if "XGBoost" in pipeline:
                                        fig_xgb, ax_xgb = plt.subplots(figsize=(7, 4.5))
                                        importances = expert.feature_importances_
                                        indices = np.argsort(importances)[-10:]
                                        ax_xgb.barh(range(10), importances[indices], color=color)
                                        feat_names = [str(feature_names[i])[:15] for i in indices]
                                        ax_xgb.set_yticks(range(10))
                                        ax_xgb.set_yticklabels(feat_names)
                                        ax_xgb.set_xlabel('Trọng số F-Score')
                                        st.pyplot(fig_xgb)
                                    
                                    else: 
                                        fig_lr, ax_lr = plt.subplots(figsize=(7, 4.5))
                                        idx_test = (y_tier_test == idx_tier)
                                        if idx_test.sum() > 0:
                                            # FIX LỖI TẦNG 5: Ép mảng vẽ biểu đồ
                                            X_tier_test = np.ascontiguousarray(X_test_s[idx_test])
                                            y_pred_z = expert.predict(X_tier_test)
                                            residuals = y_test_rev_s[idx_test] - y_pred_z
                                            
                                            if "SVR" in pipeline:
                                                eps = st.session_state.rev_models['epsilon']
                                                inside = np.abs(residuals) <= eps
                                                outside = ~inside
                                                
                                                ax_lr.scatter(y_pred_z[outside], residuals[outside], color=color, edgecolor='white', s=40, label='Ngoại lai')
                                                ax_lr.scatter(y_pred_z[inside], residuals[inside], color='#bdc3c7', alpha=0.5, s=20, label='Trong ống (An toàn)')
                                                ax_lr.axhline(y=eps, color='#e74c3c', linestyle='--', lw=2, label=f'± Epsilon ({eps})')
                                                ax_lr.axhline(y=-eps, color='#e74c3c', linestyle='--', lw=2)
                                                ax_lr.fill_between([min(y_pred_z), max(y_pred_z)], eps, -eps, color='#e74c3c', alpha=0.1)
                                                ax_lr.legend(loc='upper right')
                                            else:
                                                ax_lr.scatter(y_pred_z, residuals, color=color, edgecolor='white', s=35, alpha=0.7)
                                                
                                            ax_lr.axhline(y=0, color='#2c3e50', lw=2)
                                            ax_lr.set_title(f'PHÂN TÁN SAI SỐ TẬP TEST (TIER {idx_tier})', fontweight='bold')
                                            ax_lr.set_xlabel('Dự báo Z-Score')
                                            ax_lr.set_ylabel('Sai số Residual')
                                            st.pyplot(fig_lr)
                                        else:
                                            st.write("Tập Test không có đủ mẫu thuộc Tier này để vẽ biểu đồ.")

    # =================================================================
    # TAB 2: BÀI TOÁN ĐIỂM SỐ
    # =================================================================
    with tab_vote:
        st.subheader("Đánh giá Điểm số (IMDB): Toán học (SVR) vs Cây quyết định (XGBoost) vs Tuyến tính (Linear)")
        
        col_m3, col_m4 = st.columns([1, 1.5])
        with col_m3:
            st.write("⚙️ **Cấu hình Thuật toán**")
            vote_model_choice = st.radio(
                "Chọn thuật toán chấm điểm:", 
                ["SVR", "XGBoost", "Linear Regression"], 
                help="Thêm Linear Regression làm Baseline để đánh giá sức mạnh của SVR/XGBoost."
            )
            
            if vote_model_choice == "SVR":
                svr_c = st.slider("Hệ số phạt SVR (C)", 0.1, 10.0, 5.0, 0.1, help="(⭐ Tối ưu: 5.0)")
                svr_eps = st.slider("Ống dung sai SVR (Epsilon)", 0.01, 1.0, 0.2, 0.05, help="(⭐ Tối ưu: 0.2)")
            elif vote_model_choice == "XGBoost":
                tree_n = st.slider("Số cây (XGBoost)", 50, 300, 100, 50, help="(⭐ Tối ưu: 100)")
            else:
                st.info("💡 Linear Regression không có siêu tham số để tinh chỉnh.")
            
            btn_train_vote = st.button("🚀 Huấn luyện Bài toán Điểm số", key="btn_vote")

        with col_m4:
            st.info(f"🧠 **Giải mã Hộp đen: {vote_model_choice}**")
            st.write("**▶ Cơ chế hoạt động**")
            if vote_model_choice == "SVR":
                st.write("* **SVR:** Dùng ống Epsilon $\epsilon$ bỏ qua các review 1 sao cố tình (Review Bombing). Các điểm ảnh hưởng ít sẽ bị bỏ qua, chỉ học trên những đánh giá thực sự nổi bật.")
            elif vote_model_choice == "XGBoost":
                st.write("* **XGBoost:** Học hỏi từ sai lầm, cây sau sửa lỗi cây trước liên tục. XGBoost sẽ vét cạn các sai số nhỏ nhất để tối ưu điểm số dự đoán.")
            else:
                st.write("* **Linear Regression:** Tìm một siêu mặt phẳng tối ưu cắt qua trung tâm các điểm dữ liệu. Vì điểm số khán giả có những giới hạn bão hòa (phi tuyến), mô hình này dự kiến sẽ cho độ chính xác thấp hơn.")

        # --- XỬ LÝ HUẤN LUYỆN ĐIỂM SỐ ---
        if btn_train_vote:
            with st.status(f"Đang huấn luyện {vote_model_choice}...", expanded=True) as status:
                try:
                    if vote_model_choice == "SVR":
                        vote_model = SVR(kernel='rbf', C=svr_c, epsilon=svr_eps)
                    elif vote_model_choice == "XGBoost":
                        vote_model = XGBRegressor(n_estimators=tree_n, random_state=42)
                    else:
                        vote_model = LinearRegression() 

                    vote_model.fit(X_train_s, y_vote_train)

                    y_test_vote_pred = np.clip(vote_model.predict(X_test_s), 0, 10)
                    r2_vote = r2_score(y_vote_test, y_test_vote_pred)
                    mae_vote = mean_absolute_error(y_vote_test, y_test_vote_pred)
                    
                    st.session_state.metrics_vote = f"⭐ KẾT QUẢ TẬP TEST ({vote_model_choice}): R² = {r2_vote:.4f} | Sai số MAE = {mae_vote:.4f}"
                    st.session_state.vote_models = {
                        'model': vote_model, 
                        'models_list': [vote_model_choice], 
                        'svr_eps': svr_eps if vote_model_choice == "SVR" else None
                    }

                    st.session_state.fitted_models = st.session_state.get('fitted_models', {})
                    st.session_state.fitted_models['vote_stack'] = vote_model

                    status.update(label="✅ Hoàn tất! Xem biểu đồ bóc tách bên dưới.", state="complete")
                except Exception as e: st.error(f"Lỗi: {e}")

        # --- VẼ BIỂU ĐỒ GIAI ĐOẠN 2 (CHỈ DÙNG TẬP TEST) ---
        if 'vote_models' in st.session_state:
            if st.session_state.vote_models['models_list'][0] != vote_model_choice:
                st.warning("⚠️ Thuật toán đã thay đổi. Hãy bấm **Huấn luyện** để xem kết quả mới!")
            else:
                st.markdown("---")
                st.success(f"**{st.session_state.metrics_vote}**")
                
                col_v1, col_v2 = st.columns(2)
                v_data = st.session_state.vote_models
                
                with col_v1:
                    fig_w, ax_w = plt.subplots(figsize=(7, 5))
                    ax_w.text(0.5, 0.5, f"Mô hình Đơn lập\n{vote_model_choice}", ha='center', va='center', fontsize=18, color='#2c3e50', fontweight='bold')
                    ax_w.axis('off')
                    ax_w.set_title('CHIẾN LƯỢC SINGLE MODEL', fontweight='bold')
                    st.pyplot(fig_w)

                with col_v2:
                    st.markdown("**KIỂM TRA CƠ CHẾ SAI SỐ (TRÊN TẬP TEST)**")
                    fig_res, ax_res = plt.subplots(figsize=(7, 4.5))
                    
                    base_model = v_data['model']
                    y_pred = base_model.predict(X_test_s)
                    residuals = y_vote_test - y_pred
                    
                    if vote_model_choice == "SVR":
                        eps = v_data['svr_eps']
                        inside = np.abs(residuals) <= eps
                        outside = ~inside
                        
                        ax_res.scatter(y_pred[outside], residuals[outside], color='#8e44ad', edgecolor='white', s=30, label='Ngoại lai (Bị phạt)')
                        ax_res.scatter(y_pred[inside], residuals[inside], color='#bdc3c7', alpha=0.6, s=15, label='Trong ống (An toàn)')
                        ax_res.axhline(y=eps, color='#e74c3c', linestyle='--', lw=2)
                        ax_res.axhline(y=-eps, color='#e74c3c', linestyle='--', lw=2)
                        ax_res.fill_between([min(y_pred), max(y_pred)], eps, -eps, color='#e74c3c', alpha=0.1)
                        ax_res.set_title('CƠ CHẾ SVR KHỬ REVIEW RÁC (TEST DATA)', fontweight='bold')
                        ax_res.legend()
                        
                    else:
                        model_label = "XGBOOST" if vote_model_choice == "XGBoost" else "LINEAR REGRESSION"
                        plot_color = '#e67e22' if vote_model_choice == "XGBoost" else '#3498db'
                        
                        ax_res.scatter(y_pred, residuals, color=plot_color, edgecolor='white', s=35, alpha=0.7)
                        ax_res.set_title(f'SAI SỐ TẬP TEST CỦA {model_label}', fontweight='bold')

                    ax_res.axhline(y=0, color='#2c3e50', lw=2.5)
                    ax_res.set_xlabel('Điểm số AI Dự đoán')
                    ax_res.set_ylabel('Sai số Thực tế')
                    st.pyplot(fig_res)

if __name__ == "__main__":
    show_phase5()