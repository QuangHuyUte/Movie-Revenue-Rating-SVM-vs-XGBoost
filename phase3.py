import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.feature_selection import f_regression # Thêm thư viện quan trọng của Huy

# =====================================================================
# 1. CÁC HÀM XỬ LÝ TEXT & JSON (LÀM SẠCH DỮ LIỆU)
# =====================================================================
def get_primary_brand(company_str):
    try:
        if pd.isna(company_str) or company_str == '[]': return "Independent"
        companies = json.loads(company_str)
        if len(companies) > 0: return companies[0]['name']
    except: pass
    return "Independent"

def get_director(crew_str):
    try:
        for c in json.loads(crew_str):
            if c['job'] == 'Director': return c['name']
    except: pass
    return "Unknown"

def get_top_cast(cast_str):
    try:
        cast_list = json.loads(cast_str)
        return [c['name'] for c in cast_list[:3]]
    except: return []

def check_franchise(keyword_str):
    if pd.isna(keyword_str): return 0
    k_lower = str(keyword_str).lower()
    if any(word in k_lower for word in ['sequel', 'marvel', 'dc comics', 'based on comic', 'spin off', 'universe', 'series']):
        return 1
    return 0

# =====================================================================
# 2. HÀM GIAO DIỆN CHÍNH (SHOW PHASE 3)
# =====================================================================
def show_phase3():
    st.title("⚙️ Phase 3: Feature Engineering & Multi-task Importance")
    st.markdown("---")
    st.write("Giai đoạn bóc tách JSON thành Siêu đặc trưng và đánh giá mức độ đóng góp (F-Regression).")

    # NÚT BẤM KÍCH HOẠT QUÁ TRÌNH TRÍCH XUẤT
    if st.button("🚀 Bắt đầu Khai thác Đặc trưng (Feature Extraction)"):
        with st.status("Đang chạy Pipeline Kỹ nghệ Đặc trưng...", expanded=True) as status:
            st.write("Đang tải và gộp dữ liệu Movies & Credits...")
            df = pd.read_csv("tmdb_5000_movies.csv")
            
            if os.path.exists("tmdb_5000_credits.csv"):
                df_credits = pd.read_csv("tmdb_5000_credits.csv")
                df = pd.merge(df, df_credits, left_on='id', right_on='movie_id')

            st.write("Đang bóc tách JSON (Brand, Director, Cast, Franchise)...")
            df['brand_name'] = df['production_companies'].apply(get_primary_brand)
            if 'crew' in df.columns and 'cast' in df.columns:
                df['director_name'] = df['crew'].apply(get_director)
                df['top_cast'] = df['cast'].apply(get_top_cast)

            df['is_franchise'] = df['keywords'].apply(check_franchise)
            df['is_english'] = (df['original_language'] == 'en').astype(int)
            df['is_us_produced'] = df['production_countries'].fillna('').apply(lambda x: 1 if 'United States' in str(x) else 0)
            df['has_homepage'] = df['homepage'].notna().astype(int)
            
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            df['release_year'] = df['release_date'].dt.year.fillna(df['release_date'].dt.year.median()).astype(int)

            st.write("Đang tính toán Target Encoding cho Thương hiệu / Đạo diễn / Diễn viên...")
            mask_clean = (df['revenue'] > 1000) & (df['budget'] > 1000)
            df_clean = df[mask_clean].copy()
            global_mean = df_clean['revenue'].mean()

            df_clean['brand_power'] = df_clean['brand_name'].map(df_clean.groupby('brand_name')['revenue'].mean()).fillna(global_mean)
            
            if 'director_name' in df_clean.columns:
                df_clean['director_power'] = df_clean['director_name'].map(df_clean.groupby('director_name')['revenue'].mean()).fillna(global_mean)
            
            if 'top_cast' in df_clean.columns:
                cast_exploded = df_clean[['id', 'top_cast', 'revenue']].explode('top_cast')
                actor_power_dict = cast_exploded.groupby('top_cast')['revenue'].mean().to_dict()
                
                def calc_cast_power(cast_list):
                    if not isinstance(cast_list, list) or len(cast_list) == 0: return global_mean
                    return np.mean([actor_power_dict.get(a, global_mean) for a in cast_list])
                
                df_clean['cast_power'] = df_clean['top_cast'].apply(calc_cast_power)

            st.session_state.df_clean = df_clean
            status.update(label=f"✅ Trích xuất thành công {len(df_clean)} phim huấn luyện!", state="complete")

    # =================================================================
    # 3. TRỰC QUAN HÓA (3 TABS)
    # =================================================================
    if 'df_clean' in st.session_state:
        df_clean = st.session_state.df_clean
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["💰 Tương quan Doanh Thu", "⭐ Tương quan Điểm Số", "📊 Top 15 Đặc trưng (F-Regression)"])

        # -------------------------------------------------------------
        # TAB 1: REVENUE ANALYSIS (GIỮ NGUYÊN HEATMAP CŨ ĐẸP MẮT)
        # -------------------------------------------------------------
        with tab1:
            st.subheader("Bản đồ Nhiệt: Các yếu tố tác động đến Doanh Thu (Revenue)")
            col_chart_rev, col_text_rev = st.columns([1.5, 1])
            with col_chart_rev:
                cols_rev = ['revenue', 'budget', 'popularity', 'director_power', 'cast_power', 'brand_power', 'is_franchise', 'is_english', 'has_homepage']
                cols_rev = [c for c in cols_rev if c in df_clean.columns]
                
                fig_rev, ax_rev = plt.subplots(figsize=(8, 6))
                sns.heatmap(df_clean[cols_rev].corr(), annot=True, cmap='RdBu_r', fmt=".2f", vmin=-1, vmax=1, ax=ax_rev)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig_rev)
            with col_text_rev:
                st.info("💡 **Nhận xét chuyên sâu (Business Insights):**")
                st.markdown("""
                * **Budget (Kinh phí) là vua ($r \\approx 0.73$):** Quy luật "tiền đẻ ra tiền" rất rõ ràng. Phim được đầu tư mạnh sẽ có ngân sách Marketing lớn và số rạp chiếu áp đảo.
                * **Hiệu ứng Nhượng quyền (is_franchise):** Tương quan dương. Khán giả ưu tiên trả tiền cho vũ trụ điện ảnh quen thuộc (Marvel, DC) hơn là kịch bản gốc.
                * **Độ phủ sóng (Popularity):** Tương quan rất cao. Tầm quan trọng của việc viral trên mạng xã hội là không thể bàn cãi.
                * **Quyền lực Nhân sự (Cast/Director):** Đóng vai trò bảo chứng phòng vé. Tuy nhiên, họ thường đi kèm với dự án Budget lớn, dễ tạo đa cộng tuyến.
                """)

        # -------------------------------------------------------------
        # TAB 2: VOTE AVERAGE ANALYSIS (GIỮ NGUYÊN HEATMAP CŨ)
        # -------------------------------------------------------------
        with tab2:
            st.subheader("Bản đồ Nhiệt: Các yếu tố tác động đến Điểm Số (Vote)")
            col_chart_vote, col_text_vote = st.columns([1.5, 1])
            with col_chart_vote:
                cols_vote = ['vote_average', 'revenue', 'budget', 'popularity', 'runtime', 'director_power', 'cast_power', 'brand_power', 'is_franchise']
                cols_vote = [c for c in cols_vote if c in df_clean.columns]

                fig_vote, ax_vote = plt.subplots(figsize=(8, 6))
                sns.heatmap(df_clean[cols_vote].corr(), annot=True, cmap='summer_r', fmt=".2f", vmin=-1, vmax=1, ax=ax_vote)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig_vote)
            with col_text_vote:
                st.success("💡 **Nhận xét chuyên sâu (Quality Insights):**")
                st.markdown("""
                * **Tiền không mua được điểm số:** `budget` và `revenue` có tương quan yếu với `vote_average` ($r < 0.2$). Một bom tấn tỷ đô vẫn có thể bị chấm điểm thấp.
                * **Runtime (Thời lượng) có ý nghĩa lớn:** Tương quan dương ($r \\approx 0.35$). Phim dài (sử thi, tâm lý) thường có chiều sâu kịch bản và điểm cao hơn phim giải trí 90 phút.
                * **Franchise (Nhượng quyền):** Thường có điểm số không đột phá do kịch bản an toàn, mang tính thị trường cao hơn nghệ thuật.
                * **🔥 Kết luận cho AI:** Để dự báo Điểm số, ta **không thể** chỉ dùng Budget. AI bắt buộc phải kết hợp **Genome Tags (Đặc tính Tâm lý)**.
                """)

        # -------------------------------------------------------------
        # TAB 3: ĐÁNH GIÁ ĐỘ ĐÓNG GÓP TỔNG HỢP (YÊU CẦU CỦA HUY)
        # -------------------------------------------------------------
        with tab3:
            st.subheader("📊 Đánh giá Độ đóng góp Đa nhiệm (F-Regression)")
            st.write("Sử dụng F-Regression để chấm điểm mức độ quan trọng của toàn bộ hệ thống đặc trưng (Metadata + Top Genome Tags) đối với 2 biến mục tiêu.")

            with st.spinner("Đang kết hợp dữ liệu (Feature Fusion) và tính toán F-Scores..."):
                try:
                    # 1. LOAD GENOME DATA & BRIDGE LINKS
                    scores = pd.read_csv('genome-scores.csv')
                    tags = pd.read_csv('genome-tags.csv')
                    links = pd.read_csv('links.csv')

                    # 2. XÂY DỰNG X_fusion (Kết hợp Metadata và Genome)
                    # Kết nối df_clean với movieId
                    if 'movieId' not in df_clean.columns:
                        df_mapped = pd.merge(df_clean, links[['movieId', 'tmdbId']], left_on='id', right_on='tmdbId', how='inner')
                    else:
                        df_mapped = df_clean

                    # Lấy Top 50 Genome Tags phổ biến nhất để Fusion (Tránh tràn RAM)
                    top_tags_ids = scores.groupby('tagId')['relevance'].mean().nlargest(50).index
                    pivot_tags = scores[scores['tagId'].isin(top_tags_ids)].merge(tags, on='tagId').pivot(index='movieId', columns='tag', values='relevance').reset_index()
                    
                    # Gộp toàn bộ lại thành X_fusion
                    df_fusion = pd.merge(df_mapped, pivot_tags, on='movieId', how='inner').fillna(0)

                    # Định nghĩa các đặc trưng đưa vào chấm điểm
                    meta_features = ['budget', 'popularity', 'runtime', 'release_year', 'is_franchise', 'brand_power', 'director_power', 'cast_power']
                    genome_features = [c for c in pivot_tags.columns if c != 'movieId']
                    all_feature_names = meta_features + genome_features

                    X_fusion = df_fusion[all_feature_names].values
                    y_revenue = df_fusion['revenue'].values
                    y_vote = df_fusion['vote_average'].values

                    # 3. TÍNH TOÁN F-SCORES
                    f_scores_rev, _ = f_regression(X_fusion, y_revenue)
                    importance_rev_df = pd.DataFrame({'Feature': all_feature_names, 'Score_Rev': f_scores_rev}).sort_values(by='Score_Rev', ascending=False)

                    f_scores_vote, _ = f_regression(X_fusion, y_vote)
                    importance_vote_df = pd.DataFrame({'Feature': all_feature_names, 'Score_Vote': f_scores_vote}).sort_values(by='Score_Vote', ascending=False)

                    # 4. TRỰC QUAN HÓA SONG SONG (CHUẨN CODE JUPYTER CỦA HUY)
                    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

                    # Biểu đồ 1: Doanh Thu
                    sns.barplot(ax=axes[0], x='Score_Rev', y='Feature', data=importance_rev_df.head(15), palette='rocket')
                    axes[0].set_title('🏆 TOP 15 ĐẶC TRƯNG QUYẾT ĐỊNH DOANH THU', fontweight='bold', fontsize=13)
                    axes[0].set_xlabel('Điểm số quan trọng (F-Value)', fontsize=11)
                    axes[0].set_ylabel('Tên đặc trưng', fontsize=11)
                    axes[0].grid(axis='x', linestyle='--', alpha=0.5)

                    # Biểu đồ 2: Điểm Số
                    sns.barplot(ax=axes[1], x='Score_Vote', y='Feature', data=importance_vote_df.head(15), palette='mako')
                    axes[1].set_title('⭐ TOP 15 ĐẶC TRƯNG QUYẾT ĐỊNH ĐIỂM SỐ', fontweight='bold', fontsize=13)
                    axes[1].set_xlabel('Điểm số quan trọng (F-Value)', fontsize=11)
                    axes[1].set_ylabel('', fontsize=11)
                    axes[1].grid(axis='x', linestyle='--', alpha=0.5)

                    plt.tight_layout()
                    st.pyplot(fig)

                    # 5. NHẬN XÉT DƯỚI BIỂU ĐỒ (Giữ nguyên phân tích cốt lõi)
                    st.warning("💡 **Kết luận từ Hệ thống Đặc trưng Đa nhiệm (Multi-task Fusion):**")
                    st.write("""
                    Nhìn vào 2 biểu đồ F-Regression trên, ta thấy rõ sự phân hóa lực lượng cực kỳ thú vị:
                    * **Bài toán Tiền bạc (Bên trái):** Gần như bị thống trị hoàn toàn bởi các chỉ số Metadata. `budget`, `popularity`, và quyền lực của con người (`cast_power`, `brand_power`) là những cỗ máy in tiền thực sự.
                    * **Bài toán Chất lượng (Bên phải):** Các thẻ Genome (Mã gen nội dung) nổi lên mạnh mẽ. Kịch bản có gốc rễ chắc chắn, thoại hay, nội dung có chiều sâu chính là thứ quyết định khán giả có chấm điểm cao hay không, vượt qua sức mạnh của kinh phí.
                    """)

                except Exception as e:
                    st.error(f"Lỗi khi xử lý Feature Fusion: {e}. Đảm bảo các file genome và links nằm cùng thư mục.")