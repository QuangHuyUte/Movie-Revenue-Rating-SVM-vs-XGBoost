import streamlit as st
import pandas as pd
import os

def show_phase1(csv_files):
    st.title("🔍 Phase 1: Data Inspection (Khám phá dữ liệu thô)")
    st.write("Giai đoạn này cung cấp cái nhìn chi tiết về định nghĩa và cấu trúc của từng tập dữ liệu đầu vào.")

    # 1. TỪ ĐIỂM ĐỊNH NGHĨA CỘT
    col_desc = {
        "tmdb_5000_movies.csv": {
            "budget": "Kinh phí sản xuất (USD).",
            "revenue": "Doanh thu toàn cầu (USD).",
            "id": "Mã định danh TMDB.",
            "release_date": "Ngày phát hành.",
            "vote_average": "Điểm đánh giá trung bình (0-10).",
            "genres": "Thể loại phim (dạng JSON).",
            "overview": "Tóm tắt cốt truyện."
        },
        "links.csv": {
            "movieId": "ID hệ thống MovieLens.",
            "tmdbId": "ID hệ thống TMDB (Key chính để Join).",
            "imdbId": "ID hệ thống IMDB."
        },
        "movies.csv": {
            "movieId": "ID MovieLens.",
            "title": "Tên phim + năm sản xuất.",
            "genres": "Thể loại phim (phân cách bởi |)."
        },
        "genome-tags.csv": {
            "tagId": "ID thẻ mô tả.",
            "tag": "Tên thẻ (vd: 'sci-fi', 'emotional')."
        },
        "genome-scores.csv": {
            "movieId": "ID phim (khớp với movies.csv).",
            "tagId": "ID thẻ (khớp với genome-tags.csv).",
            "relevance": "Mức độ liên quan của thẻ đối với phim (0 đến 1)."
        }
    }

    selected_label = st.selectbox("👉 Chọn bảng dữ liệu để kiểm tra:", list(csv_files.keys()))
    file_path = csv_files[selected_label]
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Tổng số dòng", f"{len(df):,}")
        c2.metric("Số lượng cột", len(df.columns))
        c3.metric("Dữ liệu thiếu (Null)", df.isnull().sum().sum())

        st.subheader(f"📊 Preview 10 dòng đầu: {file_path}")
        
        # FIXED: Thay use_container_width bằng width='stretch' theo bản cập nhật Streamlit 2026
        st.dataframe(df.head(10), width='stretch')

        # -----------------------------------------------------------------
        # 💡 PHẦN GIẢI THÍCH ĐẶC BIỆT CHO GENOME SCORES (HUY THUYẾT TRÌNH TẠI ĐÂY)
        # -----------------------------------------------------------------
        if "genome-scores" in file_path:
            st.markdown("---")
            st.success("🎯 **VAI TRÒ CỐT LÕI CỦA GENOME SCORES:**")
            st.write("""
            Đây là **'Bản đồ mã Gen'** của bộ phim. Trong khi TMDB chỉ cho ta biết phim đó bao nhiêu tiền, thì bảng này cho ta biết **bản chất** của phim đó:
            * **Định lượng hóa cảm xúc:** Biến các thẻ mô tả (như 'u tối', 'hồi hộp') thành con số `relevance` từ 0 đến 1.
            * **Dữ liệu đa chiều:** Khi xử lý xong, mỗi bộ phim sẽ có 1,128 đặc trưng tâm lý khác nhau.
            * **Giá trị dự báo:** Đây là nguồn dữ liệu chính để AI phân biệt sự khác biệt giữa các phim cùng thể loại, giúp tăng độ chính xác của dự báo Doanh thu và Điểm số.
            """)
        # -----------------------------------------------------------------

        st.markdown("---")
        st.subheader(f"💡 Ý nghĩa các cột quan trọng")
        
        current_descriptions = col_desc.get(file_path, {})
        if current_descriptions:
            col_a, col_b = st.columns(2)
            items = list(current_descriptions.items())
            mid = len(items) // 2 + len(items) % 2
            
            with col_a:
                for col_name, desc in items[:mid]:
                    st.markdown(f"**`{col_name}`**: {desc}")
            with col_b:
                for col_name, desc in items[mid:]:
                    st.markdown(f"**`{col_name}`**: {desc}")
    else:
        st.error(f"❌ Không tìm thấy file `{file_path}`.")