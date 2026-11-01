# pima-indians-diabetes
# Báo Cáo Học Thuật: Ứng Dụng Thuật Toán KNN Phân Loại Bệnh Tiểu Đường

## 1. Bản Báo Cáo Tổng Hợp Hoàn Chỉnh (Master Report)
Bản báo cáo đầy đủ, sẵn sàng để nộp hoặc in ấn, bao gồm các phần chính:

* **Trang bìa học thuật:** Đầy đủ khung tiêu đề, tên đề tài, thông tin nhóm sinh viên, giảng viên hướng dẫn và học phần.
* **Mục lục báo cáo:** Phân cấp nội dung rõ ràng từ Chương 1 đến Chương 4 và Tài liệu tham khảo.
* **Chương 1: Giới thiệu bài toán và dữ liệu**
  * Bối cảnh y tế.
  * Phân tích kích thước mẫu ($n = 768 > 30$).
  * Bảng từ điển dữ liệu chi tiết.
* **Chương 2: Cơ sở lý thuyết về thuật toán KNN**
  * Không gian metric, công thức khoảng cách Euclidean và Manhattan.
  * Quy trình thực hiện 4 bước.
  * Phân tích Bias-Variance Tradeoff.
  * Bảng so sánh ưu, nhược điểm của thuật toán.
* **Chương 3: Ứng dụng thực nghiệm 6 bước**
  * 3 prompt AI phân cấp mẫu.
  * Mã nguồn Python minh họa (trình bày bằng font Consolas).
  * Bảng kết quả 6 chỉ số y tế.
  * 3 đồ thị phân giải cao (300 DPI) được nhúng trực tiếp.
* **Chương 4: Kết luận và hướng phát triển**
  * Luận giải sâu sắc nguy cơ của Âm tính giả (False Negative - FN) trong bối cảnh y khoa.
  * Đề xuất cải tiến: Sử dụng Random Forest/XGBoost, kỹ thuật SMOTE và bổ sung chỉ số xét nghiệm HbA1c.
* **Tài liệu tham khảo:** Trích dẫn 5 nguồn tài liệu theo chuẩn APA 7th Edition.

---

## 2. File Báo Cáo & Đường Dẫn Truy Cập

* 📄 **File Word hoàn chỉnh (dùng để nộp):** `Bao_Cao_Hoc_Thuat_Hoan_Chinh_KNN_Tieu_Duong.docx`
  > **Quy cách định dạng:** Khổ giấy A4, lề Trái 3.5 cm, Phải 2.0 cm, Trên 2.5 cm, Dưới 2.5 cm. Phông chữ Times New Roman 13 pt, giãn dòng 1.5 lines, căn đều hai bên (Justify).
* 📝 **File Markdown hoàn chỉnh:** `Bao_Cao_Hoc_Thuat_Hoan_Chinh_KNN_Tieu_Duong.md`

---

## 3. Các Tài Nguyên Thực Thi Đi Kèm

* 💻 **Mã nguồn Python thực nghiệm:** 
  * `pima_knn_analysis.py`
* 📊 **Đồ thị khoa học (Chất lượng 300 DPI):**
  * `confusion_matrix.png` (Ma trận nhầm lẫn)
  * `roc_curve.png` (Đường cong ROC, AUC = 0.9124)
  * `k_tuning_curve.png` (Đường cong biểu diễn tối ưu hóa siêu tham số K)
