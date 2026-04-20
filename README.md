# 🎬 Predict A Film's Score and Revenue: A Hybrid Machine Learning Architecture

<div align="center">
  <img src="Images/Canva.jpg" width="100%" alt="Banner Dự Án">
  <p><i>Hệ thống dự báo đa nhiệm (Multi-task) ứng dụng kiến trúc học máy lai (Hybrid) cho ngành công nghiệp điện ảnh.</i></p>
</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/) 
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://streamlit.io/) 
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg?style=for-the-badge&logo=scikitlearn)](https://scikit-learn.org/) 
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-000000.svg?style=for-the-badge&logo=xgboost)](https://xgboost.readthedocs.io/)

</div>

---

## 🎓 Thông Tin Đồ Án

- **Giảng viên hướng dẫn:** Thầy **Bùi Mạnh Quân**  
- **Nhóm sinh viên thực hiện:**  
  1. **Bùi Quang Huy**  
  2. **Nguyễn Tài Huy**  
- **Đơn vị:** **Đại học Công nghệ Kỹ thuật (HCMUTE)**  
- **Chủ đề:** Dự đoán song song doanh thu phòng vé và điểm số đánh giá (IMDB), ứng dụng các kỹ thuật xử lý dữ liệu phi tuyến tính và kiến trúc Hybrid tinh vi.

---

## 📖 1. Cơ Sở Toán Học & Công Thức Mô Hình

### A. Công thức Dự báo Lai (Hybrid Soft-Weighting)

$$
\hat{y} = \sum_{k=0}^{2} P(C_k | X) \cdot f_k(X)
$$

**Trong đó:**
- \(P(C_k | X)\): Xác suất phân tầng (Flop / Hit / Blockbuster) từ SVC  
- \(f_k(X)\): Dự báo từ mô hình hồi quy tương ứng  

---

### B. Biến đổi Dữ liệu (Data Transformation)

**Log Transform:**

$$
y' = \ln(1 + y)
$$

**Robust Scaling:**

$$
X' = \frac{X - \text{median}(X)}{\text{IQR}(X)}
$$

---

### C. Thước Đo Đánh giá

**R² Score:**

$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

**MAE:**

$$
MAE = \frac{1}{n} \sum |y_i - \hat{y}_i|
$$

**RMSE:**

$$
RMSE = \sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}
$$

---

## ⚙️ 2. Quy Trình 8 Phase

### 🔍 Phase 0: White-box Explainable AI

- Visualize cây quyết định bằng Graphviz  
- Hiểu cách XGBoost ra quyết định  

<div align="center">
  <img src="Images/Phase0.png" width="90%">
</div>

---

### 📊 Phase 1: Data Inspection

- Phân tích dữ liệu từ TMDB & MovieLens  
- Phát hiện Long-tail Distribution  

<div align="center">
  <img src="Images/Phase1.1.jpg" width="48%">
  <img src="Images/Phase1.2.png" width="48%">
</div>

---

### 🧠 Phase 2: Feature Engineering

- Tạo **Power Score** cho Actor / Director  
- Dựa trên lịch sử trước khi phim phát hành  

<div align="center">
  <img src="Images/Phase2.jpg" width="90%">
</div>

---

### 🧹 Phase 3: Feature Selection

- Dùng `f_regression`  
- Loại bỏ Data Leakage bằng Regex Blacklist  

<div align="center">
  <img src="Images/Phase3.1.jpg" width="31%">
  <img src="Images/Phase3.2.jpg" width="31%">
  <img src="Images/Phase3.3.png" width="31%">
</div>

---

### 🔄 Phase 4: Preprocessing

- Quantile Transformer → Gaussian  
- Feature: Movie Age  

<div align="center">
  <img src="Images/Phase4.png" width="90%">
</div>

---

### 🤖 Phase 5: Hybrid Training

- SVC → phân tầng  
- XGBoost + SVR → hồi quy  
- Sample Weighting  

<div align="center">
  <img src="Images/Phase5.1.png" width="48%">
  <img src="Images/Phase5.2.png" width="48%">
</div>

---

### 🏆 Phase 6: Benchmark

- So sánh 9 mô hình  
- Delta so với Linear Regression  

<div align="center">
  <img src="Images/Phase6.png" width="48%">
  <img src="Images/Phase6.1.png" width="48%">
</div>

---

### 🚀 Phase 7: Inference

- Dự đoán real-time  
- Output: Revenue + IMDB + ROI  

<div align="center">
  <img src="Images/Phase7.1.jpg" width="32%">
  <img src="Images/Phase7.2.png" width="32%">
  <img src="Images/Phase7.3.png" width="32%">
</div>

---

## 🛠️ 3. Cài đặt & Chạy

### 1. Clone Repo

```bash
git clone https://github.com/QuangHuyUte/Movie-Revenue-Rating-SVM-vs-XGBoost.git
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