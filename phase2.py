import streamlit as st
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
import json

# --- HÀM LẤY POSTER DỌC TỪ TMDB ---
def get_movie_poster_url(tmdb_id):
    try:
        url = f"https://www.themoviedb.org/movie/{tmdb_id}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            meta = soup.find('meta', property='og:image')
            if meta: return meta['content']
    except: pass
    return f"https://picsum.photos/seed/{tmdb_id}/400/600"

# --- HÀM TRUY XUẤT TOP TAGS ---
@st.cache_data(show_spinner=False)
def get_top_tags(movie_id):
    try:
        scores = pd.read_csv('genome-scores.csv')
        tags = pd.read_csv('genome-tags.csv')
        movie_scores = scores[scores['movieId'] == movie_id].sort_values('relevance', ascending=False).head(5)
        return pd.merge(movie_scores, tags, on='tagId')
    except: return None

def show_phase2():
    st.title("Bridge Phase 2: Data Integration System")
    st.markdown("---")

    # KHỞI TẠO BỘ NHỚ TRẠNG THÁI
    if 'movie_pool' not in st.session_state:
        st.session_state.movie_pool = None
    if 'slide_idx' not in st.session_state:
        st.session_state.slide_idx = 0
    if 'selected_movie' not in st.session_state:
        st.session_state.selected_movie = None

    # 1. CHIẾN LƯỢC INTEGRATION
    st.subheader("1. Tại sao cần Integration?")
    st.info("Mục tiêu: Kết hợp dữ liệu Tài chính (TMDB) và Đặc tính tâm lý (MovieLens) để AI có cái nhìn đa chiều.")

    if st.button("🚀 Thực hiện Inner Join & Pre-load Poster"):
        with st.status("Đang kết nối các hệ thống dữ liệu...", expanded=True) as status:
            if os.path.exists("tmdb_5000_movies.csv"):
                df_tmdb = pd.read_csv("tmdb_5000_movies.csv")
                df_links = pd.read_csv("links.csv")
                df_movies = pd.read_csv("movies.csv")

                df_links = df_links.dropna(subset=['tmdbId']).copy()
                df_links['tmdbId'] = df_links['tmdbId'].astype(int)
                merged = pd.merge(df_tmdb, df_links, left_on='id', right_on='tmdbId', how='inner')
                final_df = pd.merge(merged, df_movies, on='movieId', how='inner')

                sampled = final_df.sample(10).to_dict('records')
                # FIX: Ghi đè dòng loading để không bị tràn
                msg = st.empty()
                for i, m in enumerate(sampled):
                    msg.markdown(f"⏳ **Đang tải poster phim {i+1}/10:** `{m['title_x']}`")
                    sampled[i]['poster_url'] = get_movie_poster_url(m['tmdbId'])
                msg.empty()

                st.session_state.movie_pool = sampled
                st.session_state.slide_idx = 0
                status.update(label="✅ Tải hoàn tất!", state="complete")
            else: st.error("Thiếu file CSV!")

    # 2. GALLERY VÀ CLICK POSTER
    if st.session_state.movie_pool:
        st.markdown("---")
        st.subheader("📽️ Cinematic Gallery (Click vào ảnh để xem chi tiết)")

        # CSS CHO POSTER LÀM NÚT BẤM
        st.markdown("""
        <style>
        div[data-testid="column"] button[kind="secondary"] {
            border: none !important;
            padding: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
        }
        .poster-frame {
            width: 100%; height: 450px;
            border-radius: 15px;
            object-fit: cover;
            transition: 0.4s;
            border: 2px solid rgba(0, 255, 255, 0.2);
        }
        .poster-frame:hover {
            transform: scale(1.05);
            border: 3px solid #00FFFF;
            box-shadow: 0 0 30px #00FFFF;
        }
        /* Style mũi tên giữa dọc */
        .arrow-btn button {
            font-size: 4rem !important;
            color: white !important;
            text-shadow: 0 0 15px #00FFFF !important;
            height: 450px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Bố cục 3 phim với mũi tên 2 bên
        col_L, col_1, col_2, col_3, col_R = st.columns([1, 4, 4, 4, 1], vertical_alignment="center")

        with col_L:
            if st.button("〈", key="prev"):
                st.session_state.slide_idx = max(0, st.session_state.slide_idx - 3)
        with col_R:
            if st.button("〉", key="next"):
                st.session_state.slide_idx = min(7, st.session_state.slide_idx + 3)

        # Hiển thị 3 poster
        start = st.session_state.slide_idx
        movies = st.session_state.movie_pool[start : start + 3]
        display_cols = [col_1, col_2, col_3]

        for i, m in enumerate(movies):
            with display_cols[i]:
                # Sử dụng markdown để hiển thị ảnh poster
                st.markdown(f'<img src="{m["poster_url"]}" class="poster-frame">', unsafe_allow_html=True)
                # Nút chọn phim đặt ngay dưới ảnh (hoặc dùng nút trong suốt đè lên nếu thích phức tạp hơn)
                if st.button(f"Chọn: {str(m['title_x'])[:15]}...", key=f"sel_{m['tmdbId']}", use_container_width=True):
                    st.session_state.selected_movie = m

        # =====================================================================
        # 3. DASHBOARD PHÂN TÍCH (GIỮ NGUYÊN KHI NEXT/PREV)
        # =====================================================================
        if st.session_state.selected_movie:
            sm = st.session_state.selected_movie
            st.markdown("---")
            st.markdown(f"## 📊 Deep Dive Analysis: **{sm['title_x']}**")
            
            # Phân bổ thông tin tích hợp
            d_col1, d_col2, d_col3 = st.columns([1.5, 1, 1.5])

            with d_col1:
                st.success("🎬 Metadata (Từ TMDB)")
                st.write(f"**📅 Release:** {sm['release_date']}")
                st.write(f"**⏱️ Runtime:** {sm['runtime']} mins")
                st.write(f"**⭐ TMDB Vote:** {sm['vote_average']}/10")
                
                # Xử lý JSON Production
                try:
                    prods = json.loads(sm['production_companies'])
                    st.write(f"**🏢 Studio:** {prods[0]['name'] if prods else 'N/A'}")
                except: pass
                
                st.write("**📝 Overview:**")
                st.caption(sm['overview'])

            with d_col2:
                st.warning("💰 Financial Data")
                st.metric("Budget", f"${sm['budget']:,.0f}")
                st.metric("Revenue", f"${sm['revenue']:,.0f}")
                # Tính ROI vui vẻ
                roi = ((sm['revenue'] - sm['budget']) / sm['budget'] * 100) if sm['budget'] > 0 else 0
                st.metric("Estimated ROI", f"{roi:.1f}%")

            with d_col3:
                st.info("🧬 Movie Genome DNA (Từ MovieLens)")
                tags_df = get_top_tags(sm['movieId'])
                if tags_df is not None:
                    for _, row in tags_df.iterrows():
                        val = float(row['relevance']) * 100
                        st.write(f"**{str(row['tag']).upper()}**")
                        st.progress(int(val))
                        st.caption(f"Relevance: {val:.1f}%")