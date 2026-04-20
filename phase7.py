import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from sklearn.metrics import r2_score, mean_absolute_error
import json
import urllib.parse

# =====================================================================
# 1. CÁC HÀM TIỆN ÍCH HIỂN THỊ (KẾ THỪA TỪ PHASE 2 & 3)
# =====================================================================
@st.cache_data(show_spinner=False)
def get_movie_poster_url(tmdb_id):
    try:
        url = f"https://www.themoviedb.org/movie/{int(tmdb_id)}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            meta = soup.find('meta', property='og:image')
            if meta: return meta['content']
    except: pass
    return "https://images.unsplash.com/photo-1485846234645-a62644f84728?w=500&q=80"

def get_names_from_json(json_str, limit=3):
    try:
        data = json.loads(json_str)
        return ", ".join([item['name'] for item in data[:limit]])
    except: return "N/A"

def get_historical_power_ui(name, current_year, is_director, movies_df, credits_df):
    if name == 'Unknown' or not name: return pd.DataFrame()
    credits_sub = credits_df[['movie_id', 'cast', 'crew']]
    merged = movies_df.merge(credits_sub, left_on='id', right_on='movie_id')
    col = 'crew' if is_director else 'cast'
    mask = (merged['release_year'] < current_year) & (merged[col].str.contains(name, case=False, regex=False, na=False)) & (merged['revenue'] > 0)
    history = merged[mask].sort_values(by='release_year', ascending=False)
    return history[['title', 'release_year', 'revenue']].head(5) if len(history) > 0 else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_raw_csvs():
    m = pd.read_csv('tmdb_5000_movies.csv')
    c = pd.read_csv('tmdb_5000_credits.csv')
    m['release_year'] = pd.to_datetime(m['release_date'], errors='coerce').dt.year
    return m, c

# =====================================================================
# 2. GIAO DIỆN CHÍNH PHASE 7 (INFERENCE STATION)
# =====================================================================
def show_phase7():
    st.markdown("""
        <style>
        .info-box { background-color: #1e272e; padding: 25px; border-radius: 12px; border-left: 5px solid #e74c3c; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
        .highlight { color: #f1c40f; font-weight: bold; }
        .poster-img { border-radius: 12px; box-shadow: 0 8px 20px rgba(0,0,0,0.6); width: 100%; border: 1px solid #485460;}
        </style>
    """, unsafe_allow_html=True)

    st.title("🔮 Phase 7: Inference Station (Trạm Thực Thi)")
    st.info("🚀 Đưa vào một bộ phim chưa từng thấy (Tập Test) để hệ thống AI tự động phân tích Mã Gen và dự báo Doanh thu / Điểm số theo thời gian thực.")

    # KIỂM TRA MÔ HÌNH ĐÃ LƯU
    if 'model_data' not in st.session_state or 'fitted_models' not in st.session_state:
        st.error("⚠️ Không tìm thấy bộ não AI! Vui lòng quay lại **Phase 5**, bấm nút Huấn luyện ở cả 2 Tab Doanh thu và Điểm số để hệ thống nạp siêu tham số.")
        return

    data = st.session_state.model_data
    models = st.session_state.fitted_models 
    
    X_test_s = data['X_test']
    y_test_rev_s = data['y_rev_test']
    y_test_vote = data['y_vote_test']
    scaler_y_rev = data['scaler_y_rev']
    df_clean = st.session_state.df_clean
    test_idx = data.get('test_idx')
    
    # Lấy thông tin pipeline từ Phase 5
    rev_pipeline = st.session_state.get('rev_models', {}).get('pipeline', 'Hybrid (Cũ)')
    vote_pipeline = st.session_state.get('vote_models', {}).get('models_list', ['Unknown'])[0]

    # [FIX QUAN TRỌNG]: Tìm phim trong df_clean dựa trên cột 'id'
    if test_idx is not None:
        df_clean_indexed = df_clean.set_index('id')
        valid_ids = [tid for tid in test_idx if tid in df_clean_indexed.index]
        test_df = df_clean_indexed.loc[valid_ids].reset_index().copy()
    else:
        test_df = df_clean.iloc[-len(X_test_s):].copy()

    feature_cols = [c for c in df_clean.columns if c not in ['id', 'title', 'revenue', 'vote_average', 'tier', 'roi']]
    movies_raw, credits_raw = load_raw_csvs()

    if len(test_df) != len(X_test_s): test_df = test_df.iloc[:len(X_test_s)]

    st.markdown("### 🎬 1. Lựa chọn Phim Thử nghiệm (Dữ liệu chưa từng học)")
    options = ["-- Mời chọn --"] + list(range(len(test_df)))
    
    def format_movie_pos(pos):
        if pos == "-- Mời chọn --": return pos
        orig_id = test_idx[pos]
        try:
            title = movies_raw[movies_raw['id'] == orig_id]['title'].iloc[0]
            return f"🎬 {title}"
        except: return f"Phim vị trí {pos}"

    selected_pos = st.selectbox("Chọn bộ phim cần AI thẩm định:", options, format_func=format_movie_pos)

    if selected_pos != "-- Mời chọn --":
        cm = test_df.iloc[selected_pos]
        X_vec = X_test_s[selected_pos].reshape(1, -1)
        orig_id = test_idx[selected_pos]
        
        try: 
            raw_movie = movies_raw[movies_raw['id'] == orig_id].iloc[0]
        except: raw_movie = pd.Series()

        # THÔNG TIN TRỰC QUAN & THỰC TẾ
        tmdb_id = int(raw_movie.get('id', cm.get('id', 0)))
        target_year = int(raw_movie.get('release_year', 2024))
        budget = raw_movie.get('budget', 0)
        revenue_real = raw_movie.get('revenue', 0)
        vote_real = raw_movie.get('vote_average', 0)
        studio = get_names_from_json(raw_movie.get('production_companies', '[]'), 2)
        
        roi_real = ((revenue_real - budget) / budget * 100) if budget > 0 else 0
        actual_tier = 0 if roi_real <= 150 else (1 if roi_real <= 400 else 2)
        t_labels = ["🔴 Tier 0 (Lỗ/Huề)", "🟡 Tier 1 (Hit - Lời)", "🟢 Tier 2 (Blockbuster)"]

        try:
            credit_row = credits_raw[credits_raw['movie_id'] == tmdb_id].iloc[0]
            crew, cast = json.loads(credit_row['crew']), json.loads(credit_row['cast'])
            dir_name = next((m['name'] for m in crew if m['job'] == 'Director'), 'Unknown')
            cast_name = cast[0]['name'] if cast else 'Unknown'
        except: dir_name, cast_name = 'Unknown', 'Unknown'

        # UI: BẢNG THÔNG TIN CƠ BẢN
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        col_info, col_poster = st.columns([2.2, 1])
        with col_poster: st.image(get_movie_poster_url(tmdb_id), use_container_width=True)
        with col_info:
            st.markdown(f"<h2 class='highlight'>{raw_movie.get('title', 'N/A').upper()}</h2>", unsafe_allow_html=True)
            st.write(f"**📅 Ngày phát hành:** {raw_movie.get('release_date', 'N/A')} | **⏱️ Thời lượng:** {raw_movie.get('runtime', 0)} phút")
            st.markdown(f"**🏢 Studio:** <span style='color:#e74c3c; font-weight:bold;'>{studio}</span>", unsafe_allow_html=True)
            st.write(f"🎭 **Thể loại:** {get_names_from_json(raw_movie.get('genres', '[]'), 5)}")
            st.markdown(f"**📝 Overview:** *{raw_movie.get('overview', 'Chưa có tóm tắt.')}*")
            st.markdown("---\n")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Kinh phí", f"${budget:,.0f}")
            c2.metric("Doanh thu thực", f"${revenue_real:,.0f}")
            c3.metric("ROI thực tế", f"{roi_real:,.1f}%")
            c4.metric("Tier thực tế", t_labels[actual_tier].split(" ")[1])
        st.markdown("</div>", unsafe_allow_html=True)

        # UI: LỊCH SỬ NHÂN SỰ
        st.markdown("### 👁️ Phân tích Lịch sử Nhân sự (Director & Cast Power)")
        col_d, col_c = st.columns(2)
        with col_d:
            st.markdown(f"**🎥 Đạo diễn:** {dir_name}")
            d_hist = get_historical_power_ui(dir_name, target_year, True, movies_raw, credits_raw)
            if not d_hist.empty: st.dataframe(d_hist.style.format({'revenue': '${:,.0f}'}), hide_index=True)
            else: st.write("Không có dữ liệu lịch sử trước dự án này.")
            st.info(f"**Quyền lực Đạo diễn nạp vào AI: {cm.get('director_power', 0.0):.3f}**")
        with col_c:
            st.markdown(f"**🎭 Diễn viên chính:** {cast_name}")
            c_hist = get_historical_power_ui(cast_name, target_year, False, movies_raw, credits_raw)
            if not c_hist.empty: st.dataframe(c_hist.style.format({'revenue': '${:,.0f}'}), hide_index=True)
            else: st.write("Không có dữ liệu lịch sử trước dự án này.")
            st.info(f"**Quyền lực Diễn viên nạp vào AI: {cm.get('cast_power', 0.0):.3f}**")

        # UI: GIẢI MÃ 60 ĐẶC TRƯNG (Bao gồm SVD)
        with st.expander("👁️ Xem chi tiết Bảng 60 Đặc trưng Vector nạp vào Máy học (Bao gồm SVD Tags)"):
            f_names = [f"📊 {col.upper()}" for col in feature_cols]
            svd_labels = {
                1: "🧬 SVD 1: Tính Thương mại (Kỹ xảo/Hành động vs Tâm lý)",
                2: "🧬 SVD 2: Không khí Phim (Đen tối/Kinh dị vs Tươi sáng/Gia đình)",
                3: "🧬 SVD 3: Chiều không gian (Viễn tưởng vs Đời thực)",
                4: "🧬 SVD 4: Nhịp độ (Kịch tính/Gay cấn vs Chậm rãi)",
                5: "🧬 SVD 5: Giới hạn độ tuổi (Bạo lực vs Phổ thông)"
            }
            for i in range(len(f_names), X_vec.shape[1]):
                if i == X_vec.shape[1] - 1:
                    f_names.append("⏳ AGE (Đại diện Lạm phát Thời gian)")
                else:
                    svd_idx = i - len(feature_cols) + 1
                    label = svd_labels.get(svd_idx, f"🧬 SVD / Tag DNA {svd_idx} (Đặc trưng Cảm xúc)")
                    f_names.append(label)
            
            # Đảm bảo số lượng cột khớp nhau để không báo lỗi
            display_names = f_names[:X_vec.shape[1]] 
            st.dataframe(pd.DataFrame({'Tên Đặc trưng': display_names, 'Giá trị đã Scale': X_vec[0]}), height=300, use_container_width=True)

        # KÍCH HOẠT DỰ BÁO
        st.markdown('### ⚙️ 2. Trạng thái Bộ não AI (Đang kích hoạt)')
        st.success(f"✅ **Nhánh Doanh thu:** Đang chạy bằng Pipeline **{rev_pipeline}**")
        st.success(f"✅ **Nhánh Điểm số:** Đang chạy bằng Mô hình **{vote_pipeline}**")

        if st.button("🚀 KÍCH HOẠT QUÁ TRÌNH DỰ BÁO", use_container_width=True):
            with st.spinner("Hệ thống đang chạy qua các mạng lưới Hồi quy..."):
                
                m_svc = models['svc']
                m_experts = models['revenue_experts']
                m_vote = models['vote_stack']

                # =========================================================
                # BƯỚC 1: DỰ BÁO CÁ NHÂN CHO BỘ PHIM ĐANG CHỌN
                # =========================================================
                
                # Tính Phân tầng & Doanh thu
                p_probs = m_svc.predict_proba(X_vec)[0]
                p_tier = np.argmax(p_probs)
                
                # CHỌN LOGIC TÍNH TOÁN DỰA THEO LOẠI MÔ HÌNH Ở PHASE 5
                if "Độc lập" in rev_pipeline:
                    p_log = m_experts[0].predict(X_vec)[0]
                else: # Hệ thống Lai (Soft-Weighting)
                    expert_preds = [m_experts[t].predict(X_vec)[0] if t in m_experts else 0 for t in [0, 1, 2]]
                    p_log = np.sum(p_probs * np.array(expert_preds))
                
                # Giải mã Logarit và Quantile
                p_rev_usd_raw = np.maximum(np.expm1(scaler_y_rev.inverse_transform([[p_log]])[0][0]), 0)

                # Thuật toán Soft-Bounding: Giữ doanh thu thực tế (Tùy chọn)
                if budget > 0:
                    max_mult = (p_probs[0] * 1.55) + (p_probs[1] * 4.1) + (p_probs[2] * 20.0)
                    min_mult = (p_probs[0] * 0.0) + (p_probs[1] * 1.5) + (p_probs[2] * 4.0)
                    p_rev_usd = np.clip(p_rev_usd_raw, budget * min_mult, budget * max_mult)
                else:
                    p_rev_usd = p_rev_usd_raw

                p_roi = ((p_rev_usd - budget) / budget * 100) if budget > 0 else 0
                
                # Tính Điểm số
                p_vote = np.clip(m_vote.predict(X_vec)[0], 0, 10)

                # =========================================================
                # BƯỚC 2: TÍNH LẠI METRICS TỔNG THỂ (CHO BÁO CÁO Ở DƯỚI CÙNG)
                # =========================================================
                y_probs_test = m_svc.predict_proba(X_test_s)
                test_rev_preds_log = np.zeros(len(X_test_s))
                
                if "Độc lập" in rev_pipeline:
                    test_rev_preds_log = m_experts[0].predict(X_test_s)
                else:
                    for i in range(len(X_test_s)):
                        e_preds = [m_experts[t].predict([X_test_s[i]])[0] if t in m_experts else 0 for t in [0, 1, 2]]
                        test_rev_preds_log[i] = np.sum(y_probs_test[i] * np.array(e_preds))
                    
                pred_rev_test_usd_pure = np.maximum(np.expm1(scaler_y_rev.inverse_transform(test_rev_preds_log.reshape(-1,1)).flatten()), 0)
                y_rev_true_test_usd = np.expm1(scaler_y_rev.inverse_transform(y_test_rev_s.reshape(-1,1)).flatten())

                r2_rev = r2_score(y_rev_true_test_usd, pred_rev_test_usd_pure)
                mae_rev = mean_absolute_error(y_rev_true_test_usd, pred_rev_test_usd_pure)
                
                pred_vote_test = np.clip(m_vote.predict(X_test_s), 0, 10)
                r2_vote = r2_score(y_test_vote, pred_vote_test)

                # =========================================================
                # BƯỚC 3: HIỂN THỊ KẾT QUẢ VÀ BIỂU ĐỒ
                # =========================================================
                st.markdown("---")
                st.header(f"🏆 Phiếu Kết quả AI: {raw_movie.get('title', 'Phim')}")
                
                res1, res2 = st.columns(2)
                with res1:
                    st.info("### 💰 TÀI CHÍNH")
                    if "Độc lập" not in rev_pipeline:
                        st.write(f"**SVC phán quyết rủi ro cao nhất:** {t_labels[p_tier]}")
                    else:
                        st.write(f"**Cấu trúc tính toán:** {rev_pipeline}")
                        
                    st.metric("Dự báo Doanh thu", f"${p_rev_usd:,.0f}", delta=f"${p_rev_usd - revenue_real:,.0f} so với Thực tế")
                    
                    if "Độc lập" not in rev_pipeline:
                        st.caption(f"*Tỉ trọng Soft-Weighting: Lỗ: {p_probs[0]:.1%} | Lời: {p_probs[1]:.1%} | Bom tấn: {p_probs[2]:.1%}*")
                
                with res2:
                    st.success("### ⭐ CHẤT LƯỢNG")
                    st.metric("Dự báo Điểm số IMDB", f"{p_vote:.2f}/10", delta=f"{p_vote - vote_real:.2f} so với Thực tế")
                    st.write(f"**Điểm thực tế:** {vote_real}/10")

                st.markdown("### 📊 Biểu đồ Đối chiếu (AI vs Thực tế)")
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    fig_rev, ax_rev = plt.subplots(figsize=(6, 4.2))
                    ax_rev.bar(['Kinh phí', 'Thực tế', 'AI Dự báo'], [budget, revenue_real, p_rev_usd], color=['#95a5a6', '#2ecc71', '#3498db'])
                    ax_rev.set_title("So sánh Doanh thu (Triệu USD)", fontweight='bold')
                    st.pyplot(fig_rev)
                    st.write(f"- **Tỷ suất ROI:** AI đoán **{p_roi:.1f}%** 🆚 Thực tế: **{roi_real:.1f}%**")

                with col_chart2:
                    fig_vote, ax_vote = plt.subplots(figsize=(6, 4.2))
                    ax_vote.bar(['Thực tế', 'AI Dự báo'], [vote_real, p_vote], color=['#2ecc71', '#f1c40f'], width=0.5)
                    ax_vote.set_ylim(0, 10); ax_vote.set_title("So sánh Điểm số", fontweight='bold')
                    st.pyplot(fig_vote)

                st.markdown("---")
                st.subheader(f"📈 Độ tin cậy chung của Pipeline hiện tại (Trên {len(X_test_s)} phim Tập Test)")
                mc1, mc2 = st.columns(2)
                mc1.metric("R² Doanh thu toàn tập", f"{r2_rev:.4f}", help="Chỉ số khớp 100% với báo cáo ở Phase 5")
                mc1.write(f"Sai số MAE Doanh thu: **${mae_rev:,.0f}**")
                
                mc2.metric("R² Điểm số toàn tập", f"{r2_vote:.4f}")