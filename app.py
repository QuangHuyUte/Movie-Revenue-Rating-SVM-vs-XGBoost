import streamlit as st
import phase0  
import phase1
import phase2
import phase3
import phase4
import phase5  
import phase6
import phase7

# =====================================================================
# 1. CẤU HÌNH TRANG (Phải đặt trên cùng)
# =====================================================================
st.set_page_config(
    page_title="Movie AI Dashboard - HCMUTE", 
    page_icon="🎬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# 2. SIDEBAR ĐIỀU HƯỚNG
# =====================================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2503/2503508.png", width=100)
st.sidebar.title("🎬 Movie AI System")
st.sidebar.markdown("**AI Engineer:** Huy")
st.sidebar.markdown("**University:** HCMUTE")
st.sidebar.markdown("---")

st.sidebar.subheader("📍 Navigation")
choice = st.sidebar.radio(
    "Chọn giai đoạn thực hiện:", 
    [
        "Phase 0: XGBoost Simulation", # THÊM PHASE 0 VÀO ĐẦU DANH SÁCH
        "Phase 1: Data Inspection", 
        "Phase 2: Data Integration",
        "Phase 3: Feature Engineering",
        "Phase 4: Preprocessing & Scaling",
        "Phase 5: Model Training & Ensemble",
        "Phase 6: Final Evaluation",
        "Phase 7: Real-world Inference" 
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Hệ thống Modular:** Mỗi Phase được quản lý độc lập. "
    "Dữ liệu được làm sạch và chuyển tiếp qua Session State."
)

# =====================================================================
# 3. DANH SÁCH FILE DATA (Dùng chung cho Phase 1)
# =====================================================================
csv_files = {
    "TMDB 5000 Movies": "tmdb_5000_movies.csv",
    "Links (Bridge)": "links.csv",
    "MovieLens Movies": "movies.csv",
    "Genome Tags": "genome-tags.csv",
    "Genome Scores": "genome-scores.csv"
}

# =====================================================================
# 4. TRÌNH ĐIỀU HƯỚNG (ROUTER)
# =====================================================================
if choice == "Phase 0: XGBoost Simulation":  # THÊM ROUTER PHASE 0
    phase0.show_phase0()

elif choice == "Phase 1: Data Inspection":
    phase1.show_phase1(csv_files)

elif choice == "Phase 2: Data Integration":
    phase2.show_phase2()

elif choice == "Phase 3: Feature Engineering":
    phase3.show_phase3()

elif choice == "Phase 4: Preprocessing & Scaling":
    phase4.show_phase4()

elif choice == "Phase 5: Model Training & Ensemble":  
    phase5.show_phase5()

elif choice == "Phase 6: Final Evaluation":
    phase6.show_phase6()

elif choice == "Phase 7: Real-world Inference":
    phase7.show_phase7()