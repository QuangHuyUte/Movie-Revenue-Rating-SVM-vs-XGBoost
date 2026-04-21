import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression

def show_phase4():
    st.title("⚙️ Phase 4: Data Preprocessing & Transformation")
    st.markdown("---")

    if 'df_clean' not in st.session_state:
        st.warning("⚠️ Vui lòng quay lại **Phase 3**, bấm nút 'Bắt đầu Khai thác Đặc trưng' trước khi sang Phase 4.")
        return

    df_clean = st.session_state.df_clean
    
    # Lấy bảng đã gộp mã Gen nếu người dùng đã bấm nút Thực thi (để hiện Tags lên biểu đồ)
    df_source = st.session_state.get('df_fusion_raw', df_clean)

    # =================================================================
    # 1. TRỰC QUAN HÓA VẤN ĐỀ CỦA DỮ LIỆU THÔ (BEFORE)
    # =================================================================
    st.subheader("1. Khám bệnh: Phân tích các vấn đề của Dữ liệu Thô")
    
    tab_prob_y, tab_prob_x, tab_nl_rev, tab_nl_vote = st.tabs([
        "🎯 Vấn đề Mục tiêu Doanh thu (Y)", 
        "🛠️ Vấn đề Đặc trưng Đầu vào (X)",
        "📈 Tính Phi tuyến (Doanh Thu)",
        "⭐ Tính Phi tuyến (Điểm Số)"
    ])
    
    with tab_prob_y:
        st.error("🚨 **Căn bệnh 1: Lệch Phải (Right-Skewed) & Ngoại lai cực đoan ở Doanh thu**")
        fig_raw, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        sns.histplot(df_clean['revenue'] / 1e6, bins=50, kde=True, color='#e74c3c', ax=axes[0])
        axes[0].set_title("HISTOGRAM: Hiện tượng Lệch Phải", fontweight='bold')
        axes[0].set_xlabel("Doanh thu (Triệu USD)")
        axes[0].set_ylabel("Số lượng phim")

        sns.boxplot(x=df_clean['revenue'] / 1e6, color='#f39c12', flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markersize': 4}, ax=axes[1])
        axes[1].set_title("BOX PLOT: Nhận diện Ngoại lai (Outliers)", fontweight='bold')
        axes[1].set_xlabel("Doanh thu (Triệu USD)")

        st.pyplot(fig_raw)
        st.write("""
        * **Độ lệch phân phối:** Đa số phim có doanh thu thấp tạo thành chóp nhọn bên trái, đuôi kéo dài tít tắp về bên phải. AI sẽ bị "mù" do cố gắng học theo cái đuôi này.
        * **Outliers:** Hàng loạt chấm đen ngoài Box Plot (các bom tấn tỷ đô) sẽ phá hỏng trọng số (weights) của mô hình nếu đưa thẳng vào huấn luyện.
        """)

    with tab_prob_x:
        col_x1, col_x2 = st.columns(2)
        with col_x1:
            st.warning("🕰️ **Căn bệnh 2: Định dạng Thời gian (release_date)**")
            st.write("""
            * **Vấn đề:** Máy học (Machine Learning) chỉ hiểu các con số toán học, nó không thể hiểu chuỗi văn bản (YYYY-MM-DD) hay khái niệm "năm 1994".
            * **Hệ lụy:** Nếu bỏ qua biến thời gian, ta sẽ mất đi yếu tố "lạm phát" và "sự phát triển của quy mô rạp chiếu", vốn dĩ ảnh hưởng cực lớn đến doanh thu. Phim thu 100 triệu USD năm 1980 giá trị hoàn toàn khác 100 triệu USD năm 2023.
            """)
            
        with col_x2:
            st.warning("🛡️ **Căn bệnh 3: Outliers ở các biến Đầu vào (X)**")
            st.write("""
            * **Vấn đề:** Không chỉ Doanh thu (Y), mà ngay cả Đầu vào (X) như `budget`, `popularity` cũng chứa vô số Outliers. Dưới đây là Box Plot của Ngân sách.
            """)
            fig_bx, ax_bx = plt.subplots(figsize=(8, 3))
            sns.boxplot(x=df_clean['budget'] / 1e6, color='#9b59b6', flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markersize': 4}, ax=ax_bx)
            ax_bx.set_title("BOX PLOT: Outliers ở Ngân sách (Budget)", fontweight='bold', fontsize=10)
            ax_bx.set_xlabel("Budget (Triệu USD)")
            st.pyplot(fig_bx)

    with tab_nl_rev:
        st.info("📊 **Đối chiếu Tuyến tính vs Phi tuyến (Mục tiêu Doanh thu)**")
        
        feats_rev = ['vote_count', 'budget', 'popularity', 'director_power', 'cast_power', 'runtime']
        feats_rev = [c for c in feats_rev if c in df_source.columns]
        
        selected_feat_rev = st.selectbox("🔍 Chọn Đặc trưng để phân tích (Doanh thu):", feats_rev, key="rev_nl_select")
        
        pearson = df_source[selected_feat_rev].corr(df_source['revenue'])
        spearman = df_source[selected_feat_rev].corr(df_source['revenue'], method='spearman')
        
        c1, c2 = st.columns(2)
        c1.metric("Hệ số Pearson (Tuyến tính)", f"{pearson:.3f}", help="Càng gần 1 thì đường thẳng càng chuẩn")
        c2.metric("Hệ số Spearman (Phi tuyến)", f"{spearman:.3f}", help="Càng lớn hơn Pearson thì tính phi tuyến càng mạnh")

        fig_nl_rev, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=df_source[selected_feat_rev], y=df_source['revenue']/1e6, alpha=0.3, color='#3498db', label='Dữ liệu thực', ax=ax)
        
        sns.regplot(x=df_source[selected_feat_rev], y=df_source['revenue']/1e6, scatter=False, order=1, 
                    color='#e74c3c', line_kws={'label': 'Tuyến tính (Linear)', 'linestyle': '--'}, ax=ax)
        
        sns.regplot(x=df_source[selected_feat_rev], y=df_source['revenue']/1e6, scatter=False, order=2, 
                    color='#2ecc71', line_kws={'label': 'Phi tuyến (Polynomial - Bậc 2)', 'linewidth': 3}, ax=ax)
        
        ax.set_title(f"PHÂN TÍCH QUY LUẬT: {selected_feat_rev.upper()} & DOANH THU", fontweight='bold', fontsize=14)
        ax.set_ylabel("Doanh thu (Triệu USD)")
        ax.legend()
        st.pyplot(fig_nl_rev)
        
        st.success(f"👉 **Giải thích:** Nếu đường màu xanh lá (Bậc 2) ôm sát các điểm dữ liệu hơn đường đỏ đứt nét (Tuyến tính), điều đó chứng minh quy luật của {selected_feat_rev} là phi tuyến.")

    with tab_nl_vote:
        st.info("📊 **Đối chiếu Tuyến tính vs Phi tuyến (Mục tiêu Điểm số)**")
        
        if 'df_fusion_raw' not in st.session_state:
            st.caption("*(💡 Lưu ý: Để hiển thị các mã Gen cảm xúc như `predictable`, bạn cần cuộn xuống dưới bấm nút **Thực thi Transformation** 1 lần để hệ thống gộp dữ liệu nhé)*")
            
        feats_vote = ['runtime', 'budget', 'popularity', 'vote_count', 
                      'thought-provoking', 'predictable', 'boring', 'atmospheric', 'drama', 'action']
        feats_vote = [c for c in feats_vote if c in df_source.columns]
        
        selected_feat_vote = st.selectbox("🔍 Chọn Đặc trưng để phân tích (Điểm số):", feats_vote, key="vote_nl_select")
        
        pearson_v = df_source[selected_feat_vote].corr(df_source['vote_average'])
        spearman_v = df_source[selected_feat_vote].corr(df_source['vote_average'], method='spearman')

        c1, c2 = st.columns(2)
        c1.metric("Hệ số Pearson (Tuyến tính)", f"{pearson_v:.3f}")
        c2.metric("Hệ số Spearman (Phi tuyến)", f"{spearman_v:.3f}")

        fig_nl_vote, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=df_source[selected_feat_vote], y=df_source['vote_average'], alpha=0.3, color='#9b59b6', label='Dữ liệu thực', ax=ax)
        
        sns.regplot(x=df_source[selected_feat_vote], y=df_source['vote_average'], scatter=False, order=1, 
                    color='#e74c3c', line_kws={'label': 'Tuyến tính (Linear)', 'linestyle': '--'}, ax=ax)
        
        sns.regplot(x=df_source[selected_feat_vote], y=df_source['vote_average'], scatter=False, order=3, 
                    color='#f1c40f', line_kws={'label': 'Phi tuyến (Polynomial - Bậc 3)', 'linewidth': 3}, ax=ax)
        
        ax.set_title(f"PHÂN TÍCH QUY LUẬT: {selected_feat_vote.upper()} & ĐIỂM SỐ", fontweight='bold', fontsize=14)
        ax.set_ylabel("Điểm IMDB/TMDB (0-10)")
        ax.legend()
        st.pyplot(fig_nl_vote)
        
        st.success("👉 **Nhận xét chuyên môn:** Với các thẻ cảm xúc (mã Gen), quy luật thường mang tính tuyến tính (ví dụ: phim càng dễ đoán thì điểm càng tuột thẳng đứng).")

    # =================================================================
    # 2. CƠ SỞ LÝ THUYẾT & GIẢI PHÁP TÍNH TOÁN
    # =================================================================
    st.markdown("---")
    st.subheader("2. Kê đơn: Các giải pháp Toán học & Tiền xử lý")
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.info("🧬 **1. Biến đổi Logarit (Trị Lệch Phải)**")
        st.latex(r"y' = \ln(1 + y)")
        st.markdown("""**Cơ chế:** Nén khoảng cách tuyệt đối lớn thành các bước tăng trưởng tuyến tính.""")
        
        st.success("📐 **2. Chuẩn hóa Phân vị Quantile (Trị Outliers của Y)**")
        st.latex(r"F(x) \approx \mathcal{N}(0, 1)")
        st.markdown("""**Cơ chế:** Ép phân phối về hình quả chuông (Gaussian) dựa trên thứ hạng (Rank) thay vì giá trị.""")

    with col_s2:
        st.info("⏱️ **3. Khởi tạo Movie Age & Xử lý Lạm Phát Ngầm**")
        st.latex(r"Age = 2026 - Release\_Year")
        st.markdown("""**Cơ chế:** Biến đại diện (Proxy Variable) cho lạm phát.""")

        st.success("🛡️ **4. Sử dụng RobustScaler (Trị Outliers của X)**")
        st.latex(r"X' = \frac{X - Median}{IQR}")
        st.markdown("""**Cơ chế:** Dùng Trung vị (Median) và Tứ phân vị (IQR) để chia tỷ lệ, miễn nhiễm với ngoại lai.""")

    # =================================================================
    # 3. THỰC THI (ACTION) & CHIA TẬP DỮ LIỆU
    # =================================================================
    st.markdown("---")
    if st.button("🗂️ Thực thi Transformation & Chia tập dữ liệu (Train/Test)"):
        with st.status("Đang chạy Pipeline Tiền xử lý, Lọc Mã Gen và Áp dụng F-Regression...", expanded=True) as status:
            try:
                scores = pd.read_csv('genome-scores.csv')
                tags = pd.read_csv('genome-tags.csv')
                links = pd.read_csv('links.csv')
                
                total_raw_tags = len(tags['tag'].unique())
                
                if 'movieId' not in df_clean.columns:
                    df_mapped = pd.merge(df_clean, links[['movieId', 'tmdbId']], left_on='id', right_on='tmdbId', how='inner')
                else: df_mapped = df_clean
                
                # ==============================================================
                # 🛑 BỘ LỌC DATA LEAKAGE CẤP CAO BẰNG REGEX
                # ==============================================================
                blacklist_regex = [
                    r'\bafi\b', r'\boscar\b', r'\baward\b', r'\bimdb\b', r'rotten tomatoes', 
                    r'\bcriterion\b', r'movielens', r'\bbafta\b',
                    r'\bbox office\b', r'\bflop\b', r'\bhit\b(?! m[ea]n)', r'\bgross\b(?!-out)', 
                    r'\btop 10\b', r'\btop 250\b',
                    r'\bmasterpiece\b', r'\bbest\b', r'\bworst\b', r'\bgreat\b', r'\bexcellent\b', 
                    r'\bperfect\b', r'\bawful\b', r'\bhorrible\b', r'\bterrible\b', r'\bamazing\b',
                    r'\bbrilliant\b', r'\bgenius\b', r'\bawesome\b', r'\boverrated\b', r'\bunderrated\b', 
                    r'\bpredictable\b', r'\bboring\b', r'\bcliche\b', r'plot hole', r'\bentertaining\b', 
                    r'\bdisappoint\w*', r'\bbad\b', r'(?<!feel[- ])\bgood\b(?! versus)', r'\bfunny\b', 
                    r'\bstupid\b', r'\bdumb\b', r'\bcrap\b', r'so bad', r'not as good',
                    r'\bclassic\b(?! car|al)', r'\bhighly\b', r'\bfavorite\b', r'\bessential\b'
                ]
                
                valid_tags = tags[~tags['tag'].str.contains('|'.join(blacklist_regex), case=False, na=False, regex=True)]
                dropped_by_regex = total_raw_tags - len(valid_tags)
                
                # BƯỚC 1: PIVOT TOÀN BỘ CÁC THẺ SẠCH 
                pivot_tags = scores[scores['tagId'].isin(valid_tags['tagId'])].merge(tags, on='tagId').pivot(index='movieId', columns='tag', values='relevance').reset_index()
                
                df_fusion = pd.merge(df_mapped, pivot_tags, on='movieId', how='inner').fillna(0)
                st.session_state.df_fusion_raw = df_fusion 
                
                meta_cols = ['budget', 'popularity', 'runtime', 'release_year', 'is_franchise', 'is_english', 'brand_power', 'director_power', 'cast_power']
                tag_cols = [c for c in pivot_tags.columns if c != 'movieId']
                
                df_fusion['movie_age'] = 2026 - df_fusion['release_year']
                
                all_possible_features = meta_cols + tag_cols + ['movie_age']
                X_full = df_fusion[all_possible_features].values
                
                y_rev_raw = df_fusion['revenue'].values
                y_vote = df_fusion['vote_average'].values
                budgets = df_fusion['budget'].values

                y_rev_log = np.log1p(y_rev_raw)
                
                y_tier = np.zeros_like(y_rev_raw)
                roi = y_rev_raw / (budgets + 1e-5)
                y_tier[roi <= 1.5] = 0
                y_tier[(roi > 1.5) & (roi <= 4)] = 1
                y_tier[roi > 4] = 2

                # BƯỚC 2: CHIA TẬP DATA NGAY LẬP TỨC 
                indices = df_fusion['id'].values
                
                (X_train_full, X_test_full, y_rev_train_log, y_rev_test_log, 
                 y_vote_train, y_vote_test, y_tier_train, y_tier_test, train_idx, test_idx) = train_test_split(
                    X_full, y_rev_log, y_vote, y_tier, indices,
                    test_size=0.2, random_state=42, stratify=y_tier
                )
                
                # BƯỚC 3: DÙNG F-REGRESSION CHẤM ĐIỂM TRÊN TẬP TRAIN VÀ CHỌN TOP 50
                X_train_full_num = np.nan_to_num(X_train_full)
                f_scores_full, _ = f_regression(X_train_full_num, y_rev_train_log)
                f_scores_full = np.nan_to_num(f_scores_full)
                
                df_fscores = pd.DataFrame({'Feature': all_possible_features, 'F_Score': f_scores_full})
                df_fscores = df_fscores.sort_values(by='F_Score', ascending=False)
                
                top_50_features = df_fscores.head(50)['Feature'].tolist()
                top_50_indices = [all_possible_features.index(f) for f in top_50_features]
                
                dropped_by_fscore = len(all_possible_features) - 50

                # BƯỚC 4: SLICE X_train VÀ X_test 
                X_train_50 = X_train_full[:, top_50_indices]
                X_test_50 = X_test_full[:, top_50_indices]

                # BƯỚC 5: ROBUST SCALING
                scaler_X = RobustScaler()
                X_train_s = scaler_X.fit_transform(X_train_50)
                X_test_s = scaler_X.transform(X_test_50)

                # BƯỚC 6: QUANTILE TRANSFORM (CHO DOANH THU)
                scaler_y_rev = QuantileTransformer(output_distribution='normal', random_state=42)
                y_train_rev_s = scaler_y_rev.fit_transform(y_rev_train_log.reshape(-1, 1)).flatten()
                y_test_rev_s = scaler_y_rev.transform(y_rev_test_log.reshape(-1, 1)).flatten()

                f_vote_top50, _ = f_regression(np.nan_to_num(X_train_s), y_vote_train)
                f_vote_top50 = np.nan_to_num(f_vote_top50)

                # ==============================================================
                # 🛑 BƯỚC 7: FIX LỖI MEMORY (C-CONTIGUOUS) TẬN GỐC CHO SVC / SVR
                # ==============================================================
                # Ép Python dọn dẹp RAM, nén thành các dải bộ nhớ liên tục (C-order)
                X_train_s = np.ascontiguousarray(X_train_s)
                X_test_s = np.ascontiguousarray(X_test_s)
                
                y_train_rev_s = np.ascontiguousarray(y_train_rev_s)
                y_test_rev_s = np.ascontiguousarray(y_test_rev_s)
                
                y_vote_train = np.ascontiguousarray(y_vote_train)
                y_vote_test = np.ascontiguousarray(y_vote_test)
                
                y_tier_train = np.ascontiguousarray(y_tier_train)
                y_tier_test = np.ascontiguousarray(y_tier_test)

                # ==============================================================
                # LƯU KẾT QUẢ VÀO SESSION STATE
                # ==============================================================
                st.session_state.split_info = {
                    'train_size': len(X_train_50), 
                    'test_size': len(X_test_50),
                    'total_initial': len(meta_cols) + total_raw_tags + 1,
                    'dropped_regex': dropped_by_regex,
                    'dropped_fscore': dropped_by_fscore,
                    'kept_features': 50
                }
                
                st.session_state.sim_data = {
                    'df_fusion': df_fusion,
                    'df_fscores': df_fscores,
                    'all_features': all_possible_features
                }

                st.session_state.model_data = {
                    'y_train_rev_s': y_train_rev_s, 'y_vote_full': y_vote, 'y_tier_full': y_tier,
                    'X_train': X_train_s, 'X_test': X_test_s, 'y_rev_train': y_train_rev_s, 'y_rev_test': y_test_rev_s,
                    'y_vote_train': y_vote_train, 'y_vote_test': y_vote_test, 'y_tier_train': y_tier_train, 'y_tier_test': y_tier_test,
                    'scaler_y_rev': scaler_y_rev, 'feature_names': top_50_features,
                    'f_rev': df_fscores.head(50)['F_Score'].values, 'f_vote': f_vote_top50, 
                    'test_idx': test_idx  
                }
                status.update(label="✅ Tiền xử lý hoàn tất! Cấu trúc dữ liệu đã được bảo mật chống rò rỉ và fix lỗi bộ nhớ.", state="complete")
            except Exception as e:
                st.error(f"Lỗi hệ thống: {e}")

    # =================================================================
    # BẢNG THÔNG BÁO TỈ LỆ VÀ SÀNG LỌC ĐẶC TRƯNG
    # =================================================================
    if 'split_info' in st.session_state:
        info = st.session_state.split_info
        st.success(f"""
        📊 **Kết quả Khởi tạo Không gian Toán học & Chia tập dữ liệu (80/20)**
        
        **1. Quá trình Vắt lọc Đặc trưng (Feature Selection):**
        - Kích thước không gian đa chiều ban đầu: **{info['total_initial']:,}** đặc trưng *(Bao gồm Metadata + toàn bộ Genome Tags)*.
        - Đã loại bỏ bằng Regex Blacklist (Chống rò rỉ kết quả): **{info['dropped_regex']:,}** thẻ rác/gian lận.
        - Đã loại bỏ bằng Thuật toán F-Regression (Chống nhiễu số liệu): **{info['dropped_fscore']:,}** thẻ không liên quan.
        - 👉 Số đặc trưng tinh hoa cuối cùng giữ lại để cho AI học: **{info['kept_features']}** đặc trưng cốt lõi.

        **2. Chia Tập Dữ Liệu:**
        - 📚 **Tập Huấn luyện (Train): `{info['train_size']:,}` phim (80%)** - Dùng để truyền vào thuật toán dạy AI.
        - 🎯 **Tập Kiểm thử (Test): `{info['test_size']:,}` phim (20%)** - Giữ kín hoàn toàn để chấm điểm công bằng ở Phase 6.
        """)

    # =================================================================
    # 4. TÁI KHÁM (AFTER) - CÁC TAB KẾT QUẢ
    # =================================================================
    if 'model_data' in st.session_state:
        st.markdown("---")
        st.subheader("3. Tái khám: Kết quả sau chuẩn hóa (Trị dứt điểm Outliers)")
        data = st.session_state.model_data
        
        tab_after_y, tab_after_x, tab_tier, tab_vote, tab_features, tab_sim = st.tabs([
            "✨ Mục tiêu Doanh thu (Y)", 
            "🛡️ Đặc trưng Đầu vào (X)", 
            "📊 Phân tầng ROI", 
            "⭐ Điểm số Vote",
            "🏆 Top 15 Đặc trưng (Đã Lọc)",
            "🔬 Tái Khám Phá (Mô phỏng)"
        ])

        with tab_after_y:
            st.success("✅ **Sự lột xác của Doanh thu (Y) sau khi kết hợp Logarit + Quantile Transformer**")
            col_ay_hist, col_ay_box = st.columns(2)
            with col_ay_hist:
                fig_af_hist, ax_af_hist = plt.subplots(figsize=(7, 4.5))
                sns.histplot(data['y_train_rev_s'], bins=50, kde=True, color='#2ecc71', ax=ax_af_hist)
                ax_af_hist.set_title("HISTOGRAM: Đã nắn thành Gaussian", fontweight='bold')
                st.pyplot(fig_af_hist)
            with col_ay_box:
                fig_af_box, ax_af_box = plt.subplots(figsize=(7, 4.5))
                sns.boxplot(x=data['y_train_rev_s'], color='#2ecc71', flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markersize': 4}, ax=ax_af_box)
                ax_af_box.set_title("BOX PLOT: Đã khử sạch Ngoại lai", fontweight='bold')
                st.pyplot(fig_af_box)

        with tab_after_x:
            st.success("✅ **Sự lột xác của Budget & Popularity (X) sau khi đi qua màng lọc RobustScaler**")
            cols_to_plot = {}
            if 'budget' in data['feature_names']:
                cols_to_plot['Budget (Đã nén)'] = data['X_train'][:, data['feature_names'].index('budget')]
            if 'popularity' in data['feature_names']:
                cols_to_plot['Popularity (Đã nén)'] = data['X_train'][:, data['feature_names'].index('popularity')]
                
            if cols_to_plot:
                df_x_scaled = pd.DataFrame(cols_to_plot)
                fig_x_after, ax_x_after = plt.subplots(figsize=(10, 4))
                sns.boxplot(data=df_x_scaled, orient='h', palette='Blues', flierprops={'marker': 'o', 'markerfacecolor': '#e74c3c', 'markersize': 5}, ax=ax_x_after)
                ax_x_after.set_title("BOX PLOT: Các biến đầu vào (Sau Robust Scaling)", fontweight='bold')
                st.pyplot(fig_x_after)
            else:
                st.info("Budget và Popularity không được F-Regression chọn vào top nên không hiển thị.")

        with tab_tier:
            c1, c2 = st.columns([1.5, 1])
            with c1:
                fig_tier, ax_tier = plt.subplots(figsize=(8, 5))
                tier_counts = pd.Series(data['y_tier_full']).value_counts().sort_index()
                colors = ['#e74c3c', '#f1c40f', '#2ecc71']
                bars = sns.barplot(x=tier_counts.index, y=tier_counts.values, palette=colors, ax=ax_tier)
                ax_tier.set_title('PHÂN BỔ 3 TẦNG DOANH THU (ROI TIERING)', fontweight='bold')
                ax_tier.set_xticks([0, 1, 2])
                ax_tier.set_xticklabels(['Tier 0: Flop\n(ROI <= 1.5x)', 'Tier 1: Hit\n(ROI 1.5x - 4x)', 'Tier 2: Blockbuster\n(ROI > 4x)'])
                for bar in bars.patches: ax_tier.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{int(bar.get_height())}', ha='center', fontweight='bold')
                st.pyplot(fig_tier)
            with c2:
                st.info("💡 **Tại sao phải phân lớp rủi ro?**\n\nTa dùng AI Phân lớp (Classification) để bắt nó đoán xem phim sẽ Lỗ, Lời, hay Đại thắng. Điều này sát với thực tế kinh doanh hơn là chỉ chăm chăm dự báo một con số USD.")

        with tab_vote:
            c1, c2 = st.columns([1.5, 1])
            with c1:
                fig_vote, ax_vote = plt.subplots(figsize=(8, 5))
                bins = np.arange(0, 10.5, 0.5)
                sns.histplot(data['y_vote_full'], bins=bins, color='#3498db', kde=True, ax=ax_vote)
                ax_vote.set_title('PHÂN BỔ ĐIỂM SỐ KHÁN GIẢ (VOTE AVERAGE)', fontweight='bold')
                ax_vote.set_xticks(np.arange(0, 10.5, 1))
                st.pyplot(fig_vote)
            with c2:
                st.success("💡 **Không cần biến đổi (No Transform)**\n\nĐiểm số Vote tự nhiên đã phân bổ rất cân xứng (mốc 6.0 - 7.0). Ta đưa thẳng biến này vào AI Hồi quy mà không sợ nhiễu.")

        with tab_features:
            st.success("✅ **Sức mạnh Đặc trưng (F-Regression) trên Tập Huấn Luyện Sạch**")
            st.markdown("""Biểu đồ này minh chứng cho sự minh bạch của hệ thống: Các thẻ mang tính gian lận và thông tin tương lai như **Oscar, AFI, IMDB, Box Office...** đã bị Bộ lọc Regex chặn đứng hoàn toàn. AI giờ đây buộc phải dùng tư duy để dự đoán dựa trên các **Mã gen Cảm xúc cốt lõi** thực sự của kịch bản!""")
            
            features = data['feature_names']
            df_rev = pd.DataFrame({'Feature': features, 'Score': data['f_rev']}).sort_values('Score', ascending=False).head(15)
            df_vote = pd.DataFrame({'Feature': features, 'Score': data['f_vote']}).sort_values('Score', ascending=False).head(15)
            
            fig_fi, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            sns.barplot(ax=axes[0], x='Score', y='Feature', data=df_rev, palette='mako')
            axes[0].set_title('TOP 15 ĐẶC TRƯNG - DOANH THU', fontweight='bold', fontsize=12)
            axes[0].set_xlabel('Mức độ ảnh hưởng (F-Score)')
            axes[0].set_ylabel('')
            
            sns.barplot(ax=axes[1], x='Score', y='Feature', data=df_vote, palette='flare')
            axes[1].set_title('TOP 15 ĐẶC TRƯNG - ĐIỂM SỐ', fontweight='bold', fontsize=12)
            axes[1].set_xlabel('Mức độ ảnh hưởng (F-Score)')
            axes[1].set_ylabel('')
            
            plt.tight_layout()
            st.pyplot(fig_fi)

        # =====================================================================
        # TAB 6: MÔ PHỎNG ĐẶC TRƯNG TỐT VS RÁC
        # =====================================================================
        with tab_sim:
            st.info("🔬 **Tái Khám Phá: Mô phỏng Đặc trưng Quan trọng vs Đặc trưng Rác**\n\nKiểm chứng tại sao hệ thống Toán học lại chọn (hoặc vứt bỏ) một đặc trưng. Hãy kéo thanh trượt để tự mình kiểm chứng lý thuyết tương quan!")
            
            sim_data = st.session_state.sim_data
            df_f = sim_data['df_fscores']
            df_full = sim_data['df_fusion']
            
            col_sim1, col_sim2 = st.columns(2)
            with col_sim1:
                target_sim = st.selectbox("🎯 Chọn Mục tiêu kiểm chứng:", ["Doanh thu (Revenue)", "Điểm số (Vote)"])
            with col_sim2:
                feat_group = st.radio("📂 Nhóm Đặc trưng:", ["🔥 Top Quan trọng nhất (Được giữ lại)", "🗑️ Đặc trưng Rác (Bị F-Regression loại)"])
                
            if feat_group.startswith("🔥"):
                feat_list = df_f.head(20)['Feature'].tolist()
            else:
                feat_list = df_f.tail(100)['Feature'].tolist()
                
            selected_feat = st.selectbox("🔍 Chọn đặc trưng để mô phỏng sự biến thiên:", feat_list)
            
            X_sim = df_full[selected_feat].values.reshape(-1, 1)
            y_sim = df_full['revenue'].values if target_sim == "Doanh thu (Revenue)" else df_full['vote_average'].values
            
            lr = LinearRegression()
            lr.fit(X_sim, y_sim)
            
            min_val = float(X_sim.min())
            max_val = float(X_sim.max())
            step_val = (max_val - min_val) / 100 if max_val > min_val else 0.1
            
            if min_val == max_val:
                st.warning("⚠️ Đặc trưng này có giá trị không đổi (Constant) trong toàn bộ bộ dữ liệu. Không thể mô phỏng toán học.")
            else:
                sim_val = st.slider(f"🎚️ Kéo để thay đổi tham số của '{selected_feat}':", min_value=min_val, max_value=max_val, value=min_val + (max_val - min_val)/2, step=step_val)
                pred_y = lr.predict([[sim_val]])[0]
                
                fig_sim, ax_sim = plt.subplots(figsize=(10, 5))
                
                sample_idx = np.random.choice(len(X_sim), min(1000, len(X_sim)), replace=False)
                ax_sim.scatter(X_sim[sample_idx], y_sim[sample_idx], color='#bdc3c7', alpha=0.5, label='Dữ liệu thực tế')
                
                x_trend = np.array([min_val, max_val]).reshape(-1, 1)
                y_trend = lr.predict(x_trend)
                ax_sim.plot(x_trend, y_trend, color='#e74c3c', linewidth=3, label='Đường xu hướng (F-Regression)')
                
                ax_sim.plot([sim_val], [pred_y], marker='*', color='#f1c40f', markersize=20, markeredgecolor='black', label='Điểm mô phỏng')
                
                ax_sim.set_xlabel(selected_feat.upper(), fontweight='bold')
                ax_sim.set_ylabel(target_sim, fontweight='bold')
                ax_sim.legend()
                st.pyplot(fig_sim)
                
                if target_sim == "Doanh thu (Revenue)":
                    st.success(f"💡 **Toán học dự báo:** Khi **{selected_feat}** tăng đạt mức {sim_val:,.2f}, hệ thống kỳ vọng Doanh thu dịch chuyển chạm mốc **${pred_y:,.0f}**.")
                else:
                    st.success(f"💡 **Toán học dự báo:** Khi **{selected_feat}** tăng đạt mức {sim_val:,.2f}, hệ thống kỳ vọng Điểm số dịch chuyển chạm mốc **{pred_y:.2f} / 10**.")
                    
                if feat_group.startswith("🗑️"):
                    st.markdown("> *Hãy quan sát đường xu hướng màu đỏ, nó gần như **nằm ngang**. Dù bạn có kéo thanh trượt mỏi tay, việc tăng hay giảm biến này không hề làm mục tiêu biến động. Đó là lý do F-Regression thẳng tay vứt bỏ nó để tiết kiệm tài nguyên hệ thống.*")
                else:
                    st.markdown("> *Đường xu hướng dốc lên/dốc xuống cực kỳ mạnh mẽ! Chỉ cần một sự biến thiên nhỏ của tham số này cũng làm thay đổi mục tiêu rõ rệt. Đó là lý do AI xếp nó vào hàng tinh hoa để học hỏi.*")