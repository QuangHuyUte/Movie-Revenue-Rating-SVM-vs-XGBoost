# 🎬 Predict A Film's Score and Revenue: A Hybrid Machine Learning Architecture

<div align="center">
  <a href="https://www.canva.com/design/your-link-here">
    <img src="Images/Canva.png" width="100%" alt="Banner Dự Án">
  </a>
  <p><i>Hệ thống dự báo đa nhiệm (Multi-task) ứng dụng kiến trúc học máy lai (Hybrid) cho ngành công nghiệp điện ảnh.</i></p>
</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/) 
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://streamlit.io/) 
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg?style=for-the-badge&logo=scikitlearn)](https://scikit-learn.org/) 
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-000000.svg?style=for-the-badge&logo=xgboost)](https://xgboost.readthedocs.io/)

</div>

---

## 🎓 0. Thông Tin Đồ Án & Đội Ngũ

- **Đơn vị:** Đại học Sư phạm Kỹ thuật TP.HCM (**HCMUTE**) - Khoa Đào tạo Chất lượng cao.
- **Giảng viên hướng dẫn:** Thầy **Bùi Mạnh Quân** - Chuyên gia trong lĩnh vực AI.
- **Nhóm sinh viên thực hiện:** 1. **Bùi Quang Huy** (Lớp AI - HCMUTE)
  2. **Nguyễn Tài Huy** (Lớp AI - HCMUTE)
- **Mục tiêu dự án:** Xây dựng một framework dự báo thông minh giúp các nhà sản xuất phim đánh giá rủi ro tài chính và chất lượng nghệ thuật (điểm IMDB) trước khi dự án ra mắt dựa trên dữ liệu lịch sử và metadata của phim.

---

## 📖 1. Cơ Sở Toán Học & Kiến Trúc Mô Hình Chi Tiết

Dự án này không sử dụng các mô hình hồi quy đơn thuần mà tập trung vào **Hybrid Soft-Weighting Architecture** để xử lý tính bất định của doanh thu điện ảnh.

### A. Công thức Dự báo Lai (Hybrid Soft-Weighting)
Thay vì dùng một mô hình duy nhất cho mọi loại phim, chúng tôi sử dụng mô hình **SVC (Support Vector Classifier)** làm bộ điều phối (Router). SVC sẽ phân loại phim vào các tầng doanh thu tiềm năng, sau đó trọng số dự báo sẽ được tính theo công thức:

$$
\hat{y} = \sum_{k=0}^{2} P(C_k | X) \cdot f_k(X)
$$

**Giải thích biến số:**
- $P(C_k | X)$: Xác suất bộ phim rơi vào phân khúc $k$ (Tier 0: Flop, Tier 1: Hit, Tier 2: Blockbuster).
- $f_k(X)$: Giá trị dự báo từ các mô hình chuyên biệt (**XGBoost** cho doanh thu và **SVR** cho điểm số) được tối ưu cho từng phân đoạn dữ liệu.

### B. Xử lý Dữ liệu Phi Tuyến (Data Transformation)
Dữ liệu doanh thu phim thường tuân theo **Power-law Distribution** (phân phối đuôi dài). Để đưa về phân phối chuẩn (Normal Distribution) giúp mô hình học tốt hơn, chúng tôi áp dụng:

1.  **Log-Transform:** Ổn định phương sai, thu hẹp khoảng cách giữa phim độc lập và phim bom tấn.
    $$y' = \ln(1 + y)$$
2.  **Robust Scaling:** Chống lại ảnh hưởng của các Outliers (phim có doanh thu đột biến quá cao hoặc quá thấp).
    $$X' = \frac{X - \text{median}(X)}{\text{IQR}(X)}$$

### C. Các Thước Đo Hiệu Năng (Performance Metrics)
Chúng tôi sử dụng bộ 3 chỉ số để đánh giá sự hội tụ của mô hình:
- **R² Score:** Đo lường tỷ lệ sự biến thiên của doanh thu được giải thích bởi mô hình.
- **MAE (Mean Absolute Error):** Sai số thực tế (tính bằng triệu USD) để dễ hình dung mức độ rủi ro.
- **RMSE (Root Mean Squared Error):** Trừng phạt các lỗi dự báo sai lệch lớn (ví dụ dự báo phim bom tấn thành phim xịt).

---

## ⚙️ 2. Quy Trình Thực Hiện 8 Giai Đoạn (Detailed Pipeline)

### 🔍 Phase 0: White-box Explainable AI (Giải thích mô hình)
Chúng tôi ưu tiên tính minh bạch của AI. Bằng cách sử dụng **Graphviz**, hệ thống trực quan hóa toàn bộ "cây tư duy" của XGBoost, cho phép nhà sản xuất biết chính xác tại sao mô hình lại đưa ra con số đó.

<div align="center">
  <img src="Images/Phase0.png" width="90%">
  <p><i>Hình 0: Sơ đồ rẽ nhánh điều kiện của XGBoost khi phân tích trọng số các đặc trưng phim.</i></p>
</div>

---

### 📊 Phase 1: Data Inspection (Khám phá dữ liệu thô)
Phân tích 5,000+ bản ghi từ **TMDB** và **MovieLens**. Tại đây, chúng tôi nhận diện sự mất cân bằng nghiêm trọng trong dữ liệu ngành phim: 80% lợi nhuận chỉ tập trung vào 20% số lượng phim (Nguyên lý Pareto).

<div align="center">
  <img src="Images/Phase1.1.png" width="48%">
  <img src="Images/Phase1.2.png" width="48%">
  <br>
  <p><i>Hình 1.1: Biểu đồ mật độ doanh thu lệch phải | Hình 1.2: Cấu trúc Metadata và các biến số gốc.</i></p>
</div>

---

### 🧠 Phase 2: Feature Engineering (Kỹ thuật đặc trưng nhân sự)
Đây là phần cốt lõi của dự án: Chuyển đổi tên Đạo diễn và Diễn viên thành chỉ số **Power Score**. Chỉ số này tính toán dựa trên ROI trung bình và điểm IMDB của các tác phẩm họ tham gia *trước ngày phim hiện tại phát hành*.

<div align="center">
  <img src="Images/Phase2.png" width="90%">
  <p><i>Hình 2: Thuật toán trích xuất lịch sử và gán trọng số quyền lực cho dàn ekip làm phim.</i></p>
</div>

---

### 🧹 Phase 3: Feature Selection (Lọc đặc trưng & Chặn rò rỉ)
Chúng tôi phân tích 1,128 thẻ "Genome Tags". Để đảm bảo mô hình không "biết trước tương lai", hệ thống sử dụng **Regex Blacklist** để xóa bỏ các thẻ như *'Oscar Winner'*, *'Top 250'*, *'Blockbuster'*.

<div align="center">
  <img src="Images/Phase3.1.png" width="32%">
  <img src="Images/Phase3.2.png" width="32%">
  <img src="Images/Phase3.3.png" width="32%">
  <br>
  <p><i>Hình 3.1: Ma trận tương quan 3D | Hình 3.2: Lọc f_regression | Hình 3.3: Kết quả sau khi chặn Data Leakage.</i></p>
</div>

---

### 🔄 Phase 4: Preprocessing (Tiền xử lý toán học)
Sử dụng **Quantile Transformer** để "nắn" các đặc trưng có phân phối phức tạp về dạng Gaussian. Ngoài ra, chúng tôi tính toán thêm biến **Movie Age** để xử lý yếu tố lạm phát và giá trị thời gian của dòng tiền.

<div align="center">
  <img src="Images/Phase4.png" width="90%">
  <p><i>Hình 4: Đối chiếu phân phối dữ liệu trước và sau khi chuẩn hóa bằng Quantile Transformer.</i></p>
</div>

---

### 🤖 Phase 5: Hybrid Model Training (Huấn luyện đa nhiệm)
Huấn luyện song song mô hình hồi quy doanh thu (XGBoost) và điểm số (SVR). Đặc biệt, cơ chế **Sample Weighting** giúp mô hình tập trung học kỹ hơn ở những phim có ngân sách lớn, nơi sai số mang lại thiệt hại tài chính cao nhất.

<div align="center">
  <img src="Images/Phase5.1.png" width="48%">
  <img src="Images/Phase5.2.png" width="48%">
  <br>
  <p><i>Hình 5.1: Quá trình tối ưu hóa Gradient Boosting | Hình 5.2: Không gian Epsilon-Tube trong huấn luyện SVR.</i></p>
</div>

---

### 🏆 Phase 6: Comprehensive Benchmark (Đánh giá & Đối soát)
Hệ thống thực hiện so sánh chéo giữa 9 kiến trúc (Linear, Lasso, Ridge, SVR, XGBoost, và các mô hình Hybrid). Chúng tôi đo lường **Delta (Δ)** - mức độ cải thiện so với Baseline để khẳng định tính hiệu quả của mô hình lai.

<div align="center">
  <img src="Images/Phase6.png" width="48%">
  <img src="Images/Phase6.1.png" width="48%">
</div>
<div align="center">
  <img src="Images/Phase6.2.png" width="32%">
  <img src="Images/Phase6.3.png" width="32%">
  <img src="Images/Phase6.4.png" width="32%">
  <br>
  <p><i>Hình 6: Bảng Leaderboard so sánh Delta Metrics và quỹ đạo dự báo trên 3 phân tầng phim (Flop/Hit/Blockbuster).</i></p>
</div>

---

### 🚀 Phase 7: Real-time Inference Station (Trạm thực thi)
Giao diện người dùng cuối được xây dựng bằng **Streamlit**. Khi người dùng nhập một kịch bản/ekip, AI sẽ giải mã vector DNA đặc trưng và xuất ra báo cáo dự báo toàn diện.

<div align="center">
  <img src="Images/Phase7.1.png" width="32%">
  <img src="Images/Phase7.2.png" width="32%">
  <img src="Images/Phase7.3.png" width="32%">
  <br>
  <p><i>Hình 7.1: Giải mã vector DNA | Hình 7.2: Phiếu dự báo doanh thu | Hình 7.3: Phân tích tỷ suất ROI dự kiến.</i></p>
</div>

---

## 🛠️ 3. Hướng dẫn Cài đặt & Khởi chạy (Dành cho Developer)

### 1. Clone Repository
Mở terminal và chạy lệnh:
```bash
git clone [https://github.com/QuangHuyUte/Movie-Revenue-Rating-SVM-vs-XGBoost.git](https://github.com/QuangHuyUte/Movie-Revenue-Rating-SVM-vs-XGBoost.git)
cd Movie-Revenue-Rating-SVM-vs-XGBoost
```

---

### 2. Cài thư viện

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit beautifulsoup4 requests
```

---

### 3. Chuẩn bị dữ liệu

Tải và đặt vào thư mục gốc:

**TMDB Dataset:**
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

**MovieLens Genome:**
- `genome-tags.csv`
- `genome-scores.csv`
- `links.csv`

---

### 4. Chạy app

```bash
streamlit run app.py
```

---

## 👨‍💻 Nhóm thực hiện

- **Bùi Quang Huy**  
- **Nguyễn Tài Huy**

**GVHD:** Thầy **Bùi Mạnh Quân**  
**Đơn vị:** HCMUTE