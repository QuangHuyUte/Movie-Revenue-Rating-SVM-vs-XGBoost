# 🎬 Predict A Film's Score and Revenue: A Hybrid Machine Learning Architecture

![Banner](images/Canva.jpg)
<p align="center"><i>Hình 1: Banner tổng quan dự án - Hệ thống dự báo đa nhiệm cho ngành công nghiệp điện ảnh.</i></p>

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-000000.svg)](https://xgboost.readthedocs.io/)

---

## 📑 1. Giới thiệu Dự án
Dự án tập trung vào việc xây dựng một hệ thống Trí tuệ nhân tạo có khả năng dự đoán song song **Doanh thu (Revenue)** và **Điểm số (IMDB Score)** của một bộ phim. Điểm đặc biệt của hệ thống nằm ở **Kiến trúc Hybrid (Lai)**: Sử dụng SVC để phân tầng rủi ro trước khi đưa vào các chuyên gia hồi quy (XGBoost/SVR) để tính toán con số cụ thể.

### 📊 Nguồn Dữ liệu
* **TMDB 5000 Movies:** Metadata về kinh phí, hãng sản xuất, diễn viên, đạo diễn.
* **MovieLens Genome Tags:** 1,128 mã gen cảm xúc (ví dụ: *thought-provoking, atmospheric*) giúp AI "đọc" được kịch bản phim.

---

## 📖 2. Cơ sở Toán học & Thước đo Đánh giá
Để đảm bảo tính minh bạch (White-box), hệ thống áp dụng các nền tảng toán học sau:

### A. Công thức Hợp nhất Hybrid (Soft-Weighting)
Dự báo cuối cùng không chỉ đến từ một mô hình, mà là tổng trọng số xác suất từ các "chuyên gia" của từng tầng rủi ro (Lỗ, Lời, Bom tấn):
$$\hat{Y}_{final} = \sum_{k=0}^{2} P(\text{Tier}_k | X) \cdot \hat{y}_{\text{expert\_k}}(X)$$
*Trong đó $P(\text{Tier}_k | X)$ là xác suất phim thuộc tầng $k$ do SVC dự báo.*

### B. Các chỉ số đo lường hiệu năng
| Chỉ số | Công thức toán học | Ý nghĩa |
| :--- | :--- | :--- |
| **R² Score** | $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$ | Độ khớp của mô hình so với dữ liệu thực tế. |
| **MAE** | $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$ | Sai số tuyệt đối trung bình (đơn vị: USD hoặc Sao). |
| **RMSE** | $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$ | Sai số bình phương trung bình (phạt nặng các lỗi dự báo lệch xa). |

---

## ⚙️ 3. Quy trình thực hiện (8-Phase Pipeline)

### Phase 0: White-box Explainable AI
![Phase 0](images/Phase0.png)
<p align="center"><i>Hình 2: Trực quan hóa tiến trình cộng dồn trọng số lá của 50 cây quyết định trong mô hình XGBoost.</i></p>

### Phase 1: Data Inspection (Khám phá dữ liệu thô)
![Phase 1.1](images/Phase1.1.jpg)
<p align="center"><i>Hình 3: Phân tích hiện tượng lệch phải (Skewness) và nhận diện Outliers của doanh thu.</i></p>

![Phase 1.2](images/Phase1.2.png)
<p align="center"><i>Hình 4: Định nghĩa cấu trúc các tệp tin dữ liệu từ TMDB và MovieLens.</i></p>

### Phase 2: Feature Engineering (Xử lý thực thể)
![Phase 2](images/Phase2.jpg)
<p align="center"><i>Hình 5: Thuật toán quy đổi quyền lực Đạo diễn và Diễn viên thành chỉ số Power Score.</i></p>

### Phase 3: EDA & Feature Selection
![Phase 3.1](images/Phase3.1.jpg)
<p align="center"><i>Hình 6: Phân tích sự tương quan giữa các biến số kinh tế và doanh thu.</i></p>

![Phase 3.2](images/Phase3.2.jpg)
<p align="center"><i>Hình 7: Biểu đồ 3D thể hiện tác động của Mã gen nội dung tới điểm số khán giả.</i></p>

![Phase 3.3](images/Phase3.3.png)
<p align="center"><i>Hình 8: Top 15 đặc trưng quan trọng nhất sau khi đã lọc sạch Blacklist (Data Leakage).</i></p>

### Phase 4: Preprocessing & Transformation
![Phase 4](images/Phase4.png)
<p align="center"><i>Hình 9: Đối chiếu tính Tuyến tính và Phi tuyến để lựa chọn phương pháp chuẩn hóa dữ liệu.</i></p>
Hệ thống sử dụng **Log-Transform** và **Quantile Transformer** để xử lý các bộ phim bom tấn (Outliers), đưa dữ liệu về phân phối chuẩn.

### Phase 5: Model Training & Hybrid Architecture
![Phase 5.1](images/Phase5.1.png)
<p align="center"><i>Hình 10: Giao diện cấu hình tham số cho hệ thống lai SVC + XGBoost/SVR.</i></p>

![Phase 5.2](images/Phase5.2.png)
<p align="center"><i>Hình 11: Minh họa cơ chế ống lọc nhiễu Epsilon của thuật toán SVR.</i></p>

### Phase 6: Comprehensive Benchmark (Bảng xếp hạng)
![Phase 6](images/Phase6.png)
<p align="center"><i>Hình 12: Hệ thống tự động thi đấu vòng tròn giữa 9 kiến trúc học máy khác nhau.</i></p>

![Phase 6.1](images/Phase6.1.png)
<p align="center"><i>Hình 13: Bảng so sánh Delta Metrics thể hiện mức độ cải thiện so với mô hình Baseline.</i></p>

![Phase 6.2](images/Phase6.2.png)
<p align="center"><i>Hình 14: Biểu đồ so sánh trực diện sai số MAE và RMSE giữa các mô hình.</i></p>

![Phase 6.3](images/Phase6.3.png)
<p align="center"><i>Hình 15: Biểu đồ đường chứng minh khả năng bám sát quỹ đạo thực tế của AI.</i></p>

![Phase 6.4](images/Phase6.4.png)
<p align="center"><i>Hình 16: Kết quả benchmark chi tiết cho bài toán dự báo Điểm số.</i></p>

### Phase 7: Inference Station (Trạm dự báo thực tế)
![Phase 7.1](images/Phase7.1.jpg)
<p align="center"><i>Hình 17: Giao diện thẩm định phim thời gian thực với đầy đủ poster và thông tin lịch sử.</i></p>

![Phase 7.2](images/Phase7.2.png)
<p align="center"><i>Hình 18: Bảng giải mã Vector đặc trưng nạp vào bộ não AI.</i></p>

![Phase 7.3](images/Phase7.3.png)
<p align="center"><i>Hình 19: Kết quả dự báo cuối cùng kèm theo phân tích tỷ suất ROI dự kiến.</i></p>

---

## 🛠️ 4. Hướng dẫn Cài đặt & Sử dụng

### Yêu cầu hệ thống
* Python 3.9 trở lên.
* Bộ nhớ RAM tối thiểu 4GB.

### Các bước khởi chạy
1.  **Clone dự án:**
    ```bash
    git clone [https://github.com/QuangHuyUte/Movie-Revenue-Rating-SVM-vs-XGBoost.git](https://github.com/QuangHuyUte/Movie-Revenue-Rating-SVM-vs-XGBoost.git)
    cd Movie-Revenue-Rating-SVM-vs-XGBoost
    ```

2.  **Cài đặt thư viện:**
    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit beautifulsoup4 requests
    ```

3.  **Tải Dữ liệu (Bắt buộc):**
    Do giới hạn dung lượng của GitHub, bạn cần tự tải 2 bộ dữ liệu sau và giải nén trực tiếp vào thư mục gốc của dự án:
    * **TMDB 5000 Movies:** Tải tại [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). Lấy 2 file `tmdb_5000_movies.csv` và `tmdb_5000_credits.csv`.
    * **MovieLens 20M:** Tải tại [GroupLens](https://grouplens.org/datasets/movielens/20m/). Lấy 3 file `genome-tags.csv`, `genome-scores.csv`, và `links.csv`.

4.  **Chạy ứng dụng:**
    ```bash
    streamlit run app.py
    ```

---

## 👤 5. Thông tin Nhóm thực hiện
* **Nhóm sinh viên:**
    1.  **Bùi Quang Huy**
    2.  **Nguyễn Tài Huy**
* **Giảng viên hướng dẫn:** Thầy **Bùi Mạnh Quân**
* **Đơn vị:** Đại học Công nghệ Kỹ thuật TP.HCM (HCMUTE)

---
*Dự án được phát triển nhằm mục đích nghiên cứu và ứng dụng Machine Learning trong lĩnh vực kinh tế điện ảnh.*