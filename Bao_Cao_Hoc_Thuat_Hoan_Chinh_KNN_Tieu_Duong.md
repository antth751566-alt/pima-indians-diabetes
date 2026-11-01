# BÁO CÁO HỌC THUẬT MÔN HỌC MÁY
**Đề tài:** Ứng dụng thuật toán K-Nearest Neighbors (KNN) trong bài toán Phân loại tình trạng bệnh nhân tiểu đường dựa trên chỉ số y tế

---

### THÔNG TIN CHUNG
- **Giảng viên hướng dẫn:** [Tên Giảng Viên Hướng Dẫn]
- **Học phần:** Học máy (Machine Learning)
- **Nhóm sinh viên thực hiện:** [Nhóm ...]
- **Danh sách thành viên:**
  1. [Họ và tên SV 1] - MSSV: [............]
  2. [Họ và tên SV 2] - MSSV: [............]
  3. [Họ và tên SV 3] - MSSV: [............]
- **Thời gian thực hiện:** Năm 2026

---

# MỤC LỤC BÁO CÁO
- **CHƯƠNG 1. GIỚI THIỆU BÀI TOÁN VÀ DỮ LIỆU**
  - 1.1. Bối cảnh thực tiễn và tính cấp thiết của đề tài
  - 1.2. Mô tả chi tiết tập dữ liệu Pima Indians Diabetes
  - 1.3. Phương pháp tiếp cận và mục tiêu của đề tài
- **CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VỀ THUẬT TOÁN K-NEAREST NEIGHBORS (KNN)**
  - 2.1. Tổng quan về thuật toán K-Nearest Neighbors
  - 2.2. Mô hình toán học và không gian đặc trưng
  - 2.3. Các độ đo khoảng cách phổ biến
  - 2.4. Quy trình hoạt động chi tiết của thuật toán KNN
  - 2.5. Lựa chọn siêu tham số K và sự đánh đổi Độ chệch - Phương sai
  - 2.6. Yêu cầu tiền xử lý dữ liệu đặc thù cho mô hình KNN
  - 2.7. Đánh giá ưu điểm, nhược điểm và giải pháp cải tiến
- **CHƯƠNG 3. ỨNG DỤNG THUẬT TOÁN TRONG BÀI TOÁN CỤ THỂ**
  - 3.1. Các câu lệnh (Prompt) đã dùng để nhờ AI hỗ trợ viết mã nguồn
  - 3.2. Cài đặt và thực thi mã nguồn Python theo quy trình 6 bước
  - 3.3. Kết quả thực nghiệm và phân tích học thuật chuyên sâu
- **CHƯƠNG 4. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN**
  - 4.1. Tóm tắt quá trình nghiên cứu và kết quả đạt được
  - 4.2. Luận giải trực quan ý nghĩa lâm sàng của các chỉ số đánh giá
  - 4.3. Đánh giá ưu điểm và nhược điểm thực tế của mô hình KNN
  - 4.4. Đề xuất các hướng phát triển trong tương lai
- **TÀI LIỆU THAM KHẢO**

---

# CHƯƠNG 1. GIỚI THIỆU BÀI TOÁN VÀ DỮ LIỆU

## 1.1. Bối cảnh thực tiễn và tính cấp thiết của đề tài

### 1.1.1. Gánh nặng bệnh học của Đái tháo đường (Diabetes Mellitus)
Bệnh đái tháo đường (tiểu đường) là một nhóm bệnh lý chuyển hóa đặc trưng bởi tình trạng tăng đường huyết mạn tính, bắt nguồn từ sự khiếm khuyết trong quá trình bài tiết insulin, tác động của insulin hoặc phối hợp cả hai yếu tố. Theo các báo cáo dịch tễ học thường niên từ Liên đoàn Đái tháo đường Quốc tế (International Diabetes Federation - IDF) và Tổ chức Y tế Thế giới (WHO), đái tháo đường hiện nằm trong nhóm các bệnh không lây nhiễm có tốc độ gia tăng nhanh nhất toàn cầu, đe dọa nghiêm trọng tới sức khỏe cộng đồng và gia tăng chi phí y tế công cộng.

Sự nguy hiểm cốt lõi của bệnh tiểu đường (đặc biệt là tiểu đường tuýp 2) nằm ở giai đoạn phát triển âm thầm. Đa số bệnh nhân không bộc lộ triệu chứng lâm sàng rõ rệt trong nhiều năm đầu. Khi bệnh được chẩn đoán dựa trên các dấu hiệu lâm sàng kinh điển thì hệ thống mạch máu, thần kinh và các cơ quan đích (tim, thận, mắt, chi dưới) thường đã chịu tổn thương vĩnh viễn, dẫn đến các biến chứng nguy hiểm như suy tim, đột quỵ, suy thận giai đoạn cuối, mù lòa hoặc hoại tử phải đoạn chi. Do đó, việc can thiệp y tế chỉ thực sự đạt hiệu quả tối ưu khi bệnh nhân được tầm soát, chẩn đoán và dự báo nguy cơ phát bệnh từ giai đoạn tiền lâm sàng (tiền tiểu đường).

### 1.1.2. Hạn chế của phương pháp chẩn đoán truyền thống và vai trò của Học máy
Trong thực hành lâm sàng truyền thống, việc xác định bệnh nhân mắc tiểu đường hoặc có nguy cơ cao mắc bệnh chủ yếu dựa vào các xét nghiệm sinh hóa như: đo đường huyết tương lúc đói (FPG), nghiệm pháp dung nạp glucose đường uống (OGTT), hoặc định lượng hemoglobin gắn kết phân tử đường (HbA1c). Mặc dù các phép đo này mang tính tiêu chuẩn vàng, song chúng tồn tại một số rào cản nhất định:
- Đòi hỏi quy trình chuẩn bị khắt khe từ bệnh nhân (ví dụ: nhịn ăn tối thiểu 8 giờ, uống dung dịch glucose tiêu chuẩn và chờ đo định thời sau 2 giờ).
- Mang tính xâm lấn (lấy máu tĩnh mạch nhiều lần) và có chi phí phát sinh nếu thực hiện tầm soát quy mô lớn trên toàn dân số.
- Các tiêu chí chẩn đoán thường đánh giá từng chỉ số riêng lẻ theo các ngưỡng cắt cứng (cut-off thresholds), dễ bỏ qua sự tương tác phi tuyến tính phức tạp giữa nhiều biến số sinh lý, nhân trắc học và yếu tố di truyền của cơ thể.

Trước bối cảnh đó, sự giao thoa giữa Khoa học Dữ liệu (Data Science), Trí tuệ nhân tạo (AI) và Y học số đã mở ra hướng tiếp cận đột phá. Các thuật toán học máy (Machine Learning) có khả năng tổng hợp đồng thời nhiều chỉ số lâm sàng và cận lâm sàng, phát hiện những mối tương quan phi tuyến tính tiềm ẩn giữa các đặc trưng y học để đưa ra xác suất dự đoán bệnh tật với độ chính xác cao. Việc xây dựng một Hệ thống Hỗ trợ Ra quyết định Y khoa (Clinical Decision Support System - CDSS) sử dụng các mô hình học máy không nhằm mục đích thay thế bác sĩ lâm sàng, mà đóng vai trò như một công cụ sàng lọc sơ bộ hiệu quả, tiết kiệm chi phí và hỗ trợ cán bộ y tế phân loại mức độ rủi ro của người bệnh một cách kịp thời.

---

## 1.2. Mô tả chi tiết tập dữ liệu Pima Indians Diabetes

### 1.2.1. Nguồn gốc dữ liệu và đối tượng nghiên cứu
Nghiên cứu này sử dụng bộ dữ liệu **Pima Indians Diabetes Database**, một tập dữ liệu chuẩn mực và uy tín được lưu trữ tại Kho dữ liệu Học máy của Đại học California, Irvine (UCI Machine Learning Repository) và nền tảng Kaggle. Dữ liệu ban đầu được thu thập bởi Viện Quốc gia về Bệnh tiểu đường, Tiêu hóa và Bệnh thận Hoa Kỳ (National Institute of Diabetes and Digestive and Kidney Diseases - NIDDK).

Đối tượng nghiên cứu trong tập dữ liệu là toàn bộ phụ nữ thuộc bộ tộc người da đỏ Pima sinh sống tại khu vực gần Phoenix, bang Arizona, Hoa Kỳ. Quần thể người da đỏ Pima là đối tượng nghiên cứu dịch tễ học đặc thù trong y văn thế giới suốt nhiều thập kỷ qua, do cộng đồng này có tỷ lệ mắc bệnh tiểu đường tuýp 2 cao đột biến so với mức trung bình của nhân loại. Sự kết hợp giữa nền tảng gen di truyền đặc thù thích ứng với môi trường sống khắc nghiệt trong quá khứ ("giả thuyết gen tiết kiệm" - Thrifty Gene Hypothesis) cùng với sự chuyển dịch nhanh chóng sang lối sống hiện đại (chế độ ăn giàu năng lượng và giảm vận động thể lực) đã biến nhóm dân cư này thành một mô hình dịch tễ học lý tưởng để phân tích căn nguyên của bệnh. Tất cả bệnh nhân được ghi nhận trong tập dữ liệu đều là nữ giới và có độ tuổi từ 21 trở lên. Mục tiêu nhị phân đặt ra là dự đoán liệu bệnh nhân có xuất hiện kết quả xét nghiệm dương tính với bệnh tiểu đường trong khoảng thời gian theo dõi 5 năm tiếp theo hay không.

### 1.2.2. Quy mô dữ liệu và luận giải tính thống kê ($n > 30$)
Tập dữ liệu bao gồm **768 dòng quan sát** (tương ứng với 768 hồ sơ bệnh nhân) và **9 cột** (tương ứng với 8 biến đặc trưng đầu vào và 1 biến nhãn mục tiêu).

Về mặt thống kê học, kích thước mẫu $n = 768$ thỏa mãn điều kiện tiên quyết $n > 30$ theo quy tắc thực nghiệm (Rule of Thumb) của Thống kê suy diễn:
- **Định lý Giới hạn Trung tâm (Central Limit Theorem - CLT):** Khi kích thước mẫu ngẫu nhiên độc lập $n \ge 30$, phân phối mẫu của trung bình mẫu xấp xỉ phân phối chuẩn (Gaussian distribution), bất kể phân phối thực tế của tổng thể có hình thái phân bố như thế nào. Với $n = 768$, kích thước mẫu vượt xa ngưỡng 30 ($n \gg 30$), đảm bảo các tham số thống kê mô tả (kỳ vọng, phương sai, độ lệch chuẩn) có tính vững (consistency) và tiệm cận không chệch (asymptotically unbiased).
- **Độ tin cậy trong học máy:** Trong bài toán học có giám sát, cỡ mẫu $n = 768$ cho phép nhóm nghiên cứu chia tập dữ liệu thành các tập con độc lập (Tập huấn luyện - Training set và Tập kiểm thử - Testing set theo tỷ lệ 80:20 hoặc 70:30) hoặc áp dụng phương pháp Kiểm định chéo $K$-gấp ($K$-Fold Cross Validation) mà vẫn đảm bảo số lượng quan sát ở từng tập con đủ lớn để mô hình học được các ranh giới phân tách mà không bị suy thoái do thiếu hụt thông tin thống kê.

### 1.2.3. Bảng đặc tả chi tiết các biến số
Tập dữ liệu chứa 8 biến định lượng phản ánh các chỉ số sinh hóa và nhân trắc học, cùng 1 biến phân loại nhị phân đóng vai trò là nhãn giám sát.

**Bảng 1.1: Bảng từ điển dữ liệu của bộ dữ liệu Pima Indians Diabetes**

| Tên biến gốc | Tên chuẩn hóa | Kiểu dữ liệu | Đơn vị đo | Ý nghĩa y học và mô tả lâm sàng |
| :--- | :--- | :--- | :--- | :--- |
| `preg` | Pregnancies | Rời rạc (Integer) | Lần | Số lần mang thai của bệnh nhân. Tiền sử mang thai liên quan chặt chẽ đến sự kháng insulin và nguy cơ tiểu đường thai kỳ. |
| `plas` | Glucose | Liên tục (Float) | $\text{mg/dL}$ | Nồng độ glucose trong huyết tương sau 2 giờ thực hiện nghiệm pháp dung nạp glucose đường uống (OGTT). |
| `pres` | BloodPressure | Liên tục (Float) | $\text{mm Hg}$ | Huyết áp tâm trương (áp lực dòng máu lên thành động mạch khi cơ tim giãn ra). Huyết áp cao là biến chứng tim mạch thường đi kèm tiểu đường. |
| `skin` | SkinThickness | Liên tục (Float) | $\text{mm}$ | Độ dày nếp gấp da cơ tam đầu (Triceps skin fold thickness). Chỉ số phản ánh trực tiếp lượng mỡ tích tụ dưới da. |
| `test` | Insulin | Liên tục (Float) | $\mu\text{U/mL}$ | Nồng độ hormone insulin trong huyết thanh sau 2 giờ đo nghiệm pháp dung nạp glucose. Phản ánh mức độ hoạt động của tế bào beta đảo tụy. |
| `mass` | BMI | Liên tục (Float) | $\text{kg/m}^2$ | Chỉ số khối cơ thể, tính bằng cân nặng chia bình phương chiều cao. Là thước đo lâm sàng tiêu chuẩn đánh giá béo phì. |
| `pedi` | DiabetesPedigree | Liên tục (Float) | Điểm số vô thứ nguyên | Hàm phả hệ bệnh tiểu đường. Một chỉ số tổng hợp định lượng nguy cơ di truyền dựa trên tiền sử mắc bệnh của các thế hệ thân tộc. |
| `age` | Age | Rời rạc (Integer) | Năm | Tuổi của bệnh nhân (tất cả đối tượng đều $\ge 21$ tuổi). Tỷ lệ thoái hóa tế bào và đề kháng insulin tăng theo độ tuổi. |
| `class` | Outcome | Nhị phân (Binary) | Mã hóa: $0, 1$ | **Biến mục tiêu (Target):** <br>• **0:** Âm tính (Không mắc bệnh tiểu đường). <br>• **1:** Dương tính (Mắc bệnh tiểu đường trong vòng 5 năm). |

### 1.2.4. Phân tích đặc thù và các thách thức dữ liệu
Qua việc khảo sát sơ bộ dữ liệu, nhóm nghiên cứu nhận diện hai thách thức kỹ thuật lớn cần được xử lý triệt để:

1. **Vấn đề giá trị khuyết thiếu được mã hóa ngầm dưới dạng số 0:**
   Trong bộ dữ liệu, các thuộc tính y tế mang tính sống còn như `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, và `BMI` có chứa các bản ghi nhận giá trị bằng 0. Về mặt sinh lý học ở người sống, nồng độ glucose trong máu, huyết áp hay chỉ số BMI không thể bằng 0. Do đó, các giá trị 0 này thực chất là các giá trị bị khuyết (Missing Values) trong quá trình thu thập hoặc bệnh nhân không thực hiện xét nghiệm đó. Vì thuật toán KNN dựa trên khoảng cách hình học, việc để nguyên giá trị 0 sẽ làm sai lệch hoàn toàn khoảng cách giữa các điểm dữ liệu. Nhóm nghiên cứu sẽ tiến hành thay thế các giá trị 0 phi lý này bằng giá trị `NaN` và áp dụng phương pháp điền khuyết (Imputation) bằng trung vị (median) theo từng lớp nhãn.

2. **Hiện tượng mất cân bằng lớp (Class Imbalance):**
   Trong tổng số 768 bệnh nhân:
   - Số mẫu âm tính (Lớp 0): **500 mẫu** (chiếm tỷ lệ xấp xỉ $65.1\%$).
   - Số mẫu dương tính (Lớp 1): **268 mẫu** (chiếm tỷ lệ xấp xỉ $34.9\%$).
   
   Sự chênh lệch này đồng nghĩa với việc nếu một mô hình ngây thơ luôn dự đoán bệnh nhân là "không mắc bệnh", độ chính xác (Accuracy) vẫn đạt mức $65.1\%$. Do đó, trong nghiên cứu này, việc đánh giá hiệu năng của mô hình phân loại KNN sẽ không chỉ dừng lại ở chỉ số Accuracy đơn thuần, mà bắt buộc phải tích hợp ma trận nhầm lẫn (Confusion Matrix), độ nhạy (Recall), độ chuẩn xác (Precision), chỉ số $F_1\text{-score}$, và diện tích dưới đường cong ROC (ROC-AUC).

---

## 1.3. Phương pháp tiếp cận và mục tiêu của đề tài

### 1.3.1. Phương pháp tiếp cận
Để giải quyết bài toán phân loại nhị phân trên, nghiên cứu này lựa chọn thuật toán **K-Nearest Neighbors (KNN)**. Việc lựa chọn KNN xuất phát từ các luận điểm khoa học sau:
- **Tính phù hợp với dữ liệu liên tục:** Tất cả 8 biến đặc trưng của bộ dữ liệu đều là dữ liệu định lượng dạng số (numerical features), rất thuận lợi cho việc biểu diễn trong không gian véc-tơ đa chiều $\mathbb{R}^8$ và tính toán khoảng cách không gian.
- **Giả thiết tương đồng sinh học (Biological Homophily):** Trong y học thực chứng, những bệnh nhân có các chỉ số sinh hóa và thể trạng gần tương đồng nhau (ví dụ: cùng độ tuổi, cùng mức độ béo phì, nồng độ glucose và huyết áp tương đương) thì xác suất mang chung trạng thái bệnh lý là rất cao. KNN mô phỏng chính xác tiên đề này bằng cách gom cụm và phân loại dựa trên những láng giềng có đặc trưng gần nhất.
- **Tính phi tham số (Non-parametric):** Báo cáo không cần áp đặt bất kỳ giả định tiên nghiệm nào về phân phối xác suất của dữ liệu y học (vốn thường phân tán lệch và chứa nhiễu).

### 1.3.2. Mục tiêu nghiên cứu
1. Xây dựng một đường ống xử lý dữ liệu (Data Pipeline) chuẩn mực: Từ xử lý dữ liệu khuyết thiếu ngầm định, phát hiện ngoại lai đến chuẩn hóa đặc trưng phù hợp với hàm khoảng cách.
2. Thiết lập mô hình phân loại K-Nearest Neighbors, khảo sát ảnh hưởng của các hàm đo khoảng cách khác nhau (Euclidean, Manhattan) và cơ chế bầu chọn (Uniform vs. Distance-weighted).
3. Đánh giá tác động của việc lựa chọn siêu tham số $K$ thông qua quá trình tối ưu hóa, qua đó phân tích sự đánh đổi giữa Độ chệch và Phương sai (Bias-Variance Tradeoff) đối với độ chính xác chẩn đoán.

---

# CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VỀ THUẬT TOÁN K-NEAREST NEIGHBORS (KNN)

## 2.1. Tổng quan về thuật toán K-Nearest Neighbors

Thuật toán K-Nearest Neighbors (KNN - $K$ láng giềng gần nhất) là một trong những phương pháp học máy có giám sát (Supervised Learning) kinh điển và cơ bản nhất trong lĩnh vực nhận dạng mẫu và khai phá dữ liệu. Thuật toán được khởi xướng lần đầu bởi Evelyn Fix và Joseph Hodges vào năm 1951 trong một báo cáo kỹ thuật của Không quân Hoa Kỳ về phân loại thống kê phi tham số, sau đó được Thomas Cover và Peter Hart mở rộng, chứng minh toán học chặt chẽ vào năm 1967.

KNN thuộc nhóm các thuật toán:
1. **Học phi tham số (Non-parametric learning):** Khái niệm này không hàm ý mô hình không có tham số, mà chỉ ra rằng cấu trúc mô hình không được định nghĩa trước bởi một phương trình toán học cố định với số lượng tham số hữu hạn độc lập với cỡ mẫu (như các hệ số hồi quy trong Hồi quy Tuyến tính hay Hồi quy Logistic). Thay vào đó, hình thái của ranh giới quyết định hoàn toàn do sự phân bố không gian của dữ liệu huấn luyện định hình.
2. **Học lười (Lazy Learning / Instance-based Learning):** Trái ngược với các thuật toán "học tích cực" (Eager Learning) như Mạng Nơ-ron nhân tạo, Cây quyết định hay Máy véc-tơ hỗ trợ (SVM) – vốn xây dựng một mô hình trừu tượng khái quát hóa dữ liệu ngay trong pha huấn luyện – KNN hoàn toàn không thực hiện bất kỳ phép tính toán phức tạp nào trong giai đoạn huấn luyện (Training phase). Toàn bộ tập dữ liệu mẫu chỉ đơn thuần được ghi nhớ và lưu trữ trong bộ nhớ hệ thống. Quá trình tính toán, so khớp khoảng cách và suy diễn chỉ thực sự diễn ra tại thời điểm kiểm thử (Inference phase) khi xuất hiện một điểm dữ liệu mới cần phân loại.

---

## 2.2. Mô hình toán học và không gian đặc trưng

### 2.2.1. Biểu diễn toán học
Giả sử tập dữ liệu huấn luyện $\mathcal{D}$ bao gồm $N$ mẫu quan sát độc lập:
$$\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_N, y_N)\}$$

Trong đó:
- Mỗi điểm quan sát $\mathbf{x}_i$ là một véc-tơ đặc trưng thuộc không gian thực $d$-chiều:
  $$\mathbf{x}_i = \begin{bmatrix} x_{i1} \\ x_{i2} \\ \vdots \\ x_{id} \end{bmatrix} \in \mathcal{X} \subseteq \mathbb{R}^d$$
  Trong bài toán dự đoán tiểu đường của báo cáo này, số chiều không gian đặc trưng là $d = 8$, tương ứng với 8 chỉ số y tế (`preg`, `plas`, `pres`, `skin`, `test`, `mass`, `pedi`, `age`).
- $y_i$ là nhãn phân loại rời rạc tương ứng, thuộc tập hợp hữu hạn các lớp $\mathcal{Y}$:
  $$y_i \in \mathcal{Y} = \{c_1, c_2, \dots, c_C\}$$
  Trong bài toán phân loại nhị phân nhãn bệnh nhân tiểu đường: $C = 2$ với $\mathcal{Y} = \{0, 1\}$.

### 2.2.2. Không gian metric và hàm khoảng cách
Không gian đặc trưng $\mathcal{X}$ được trang bị một hàm khoảng cách (metric) xác định:
$$d: \mathcal{X} \times \mathcal{X} \to \mathbb{R}^+ \cup \{0\}$$

Để được coi là một không gian metric hợp lệ theo chuẩn toán học, hàm $d(\mathbf{x}_i, \mathbf{x}_j)$ phải thỏa mãn 4 tiên đề:
1. **Tính không âm (Non-negativity):** $d(\mathbf{x}_i, \mathbf{x}_j) \ge 0, \quad \forall \mathbf{x}_i, \mathbf{x}_j \in \mathcal{X}$.
2. **Tính đồng nhất của các điểm không thể phân biệt (Identity of indiscernibles):** $d(\mathbf{x}_i, \mathbf{x}_j) = 0 \iff \mathbf{x}_i = \mathbf{x}_j$.
3. **Tính đối xứng (Symmetry):** $d(\mathbf{x}_i, \mathbf{x}_j) = d(\mathbf{x}_j, \mathbf{x}_i), \quad \forall \mathbf{x}_i, \mathbf{x}_j \in \mathcal{X}$.
4. **Bất đẳng thức tam giác (Triangle inequality):** $d(\mathbf{x}_i, \mathbf{x}_k) \le d(\mathbf{x}_i, \mathbf{x}_j) + d(\mathbf{x}_j, \mathbf{x}_k), \quad \forall \mathbf{x}_i, \mathbf{x}_j, \mathbf{x}_k \in \mathcal{X}$.

---

## 2.3. Các độ đo khoảng cách phổ biến

Sự chính xác của thuật toán KNN phụ thuộc hoàn toàn vào tính đúng đắn của việc định lượng "khoảng cách" giữa các đối tượng. Tùy thuộc vào bản chất dữ liệu, các hàm khoảng cách khác nhau sẽ mang lại các ranh giới phân tách khác nhau.

```
                  +----------------------------------------------+
                  |         CÁC ĐỘ ĐO KHOẢNG CÁCH TRONG KNN      |
                  +----------------------------------------------+
                                         |
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
+------------------+           +--------------------+          +--------------------+
| Euclidean (L2)   |           |  Manhattan (L1)    |          |  Minkowski (Lp)    |
| d = sqrt(sum...) |           |  d = sum(|xi - xj|)|          |  Tổng quát hóa     |
| Đường thẳng      |           |  Hình học ô bàn cờ |          |  p=1: L1; p=2: L2  |
+------------------+           +--------------------+          +--------------------+
```

### 2.3.1. Khoảng cách Euclidean ($L_2$ norm)
Khoảng cách Euclidean là độ đo tự nhiên và phổ biến nhất, đại diện cho độ dài đoạn thẳng hình học nối liền giữa hai điểm $\mathbf{x}_i$ và $\mathbf{x}_j$ trong không gian Euclid đa chiều.

**Công thức toán học:**
$$d_{Euclid}(\mathbf{x}_i, \mathbf{x}_j) = \|\mathbf{x}_i - \mathbf{x}_j\|_2 = \sqrt{\sum_{k=1}^d (x_{ik} - x_{jk})^2}$$

- **Đặc điểm hình học:** Đường đồng mức khoảng cách (isocontour) từ một điểm gốc là một hình cầu siêu phẳng ($d$-dimensional hypersphere).
- **Ứng dụng:** Thích hợp khi các đặc trưng là các biến liên tục, đồng nhất về bản chất vật lý và không gian đặc trưng có mật độ tương đối dày đặc.
- **Hạn chế:** Rất nhạy cảm với các điểm dữ liệu dị biệt (outliers) do sai số giữa hai tọa độ bị lũy thừa bậc hai.

### 2.3.2. Khoảng cách Manhattan ($L_1$ norm / City-Block distance)
Khoảng cách Manhattan biểu diễn tổng khoảng cách chiếu trên các trục tọa độ trực giao, mô phỏng quãng đường đi của một phương tiện trong mạng lưới đường phố bàn cờ (Grid pattern) của thành phố Manhattan.

**Công thức toán học:**
$$d_{Manhattan}(\mathbf{x}_i, \mathbf{x}_j) = \|\mathbf{x}_i - \mathbf{x}_j\|_1 = \sum_{k=1}^d |x_{ik} - x_{jk}|$$

- **Đặc điểm hình học:** Đường đồng mức khoảng cách là một khối bát diện siêu phẳng ($d$-dimensional hyperoctahedron).
- **Ứng dụng:** Phù hợp với các bài toán có sự hiện diện của một số đặc trưng nhiễu hoặc khi các đặc trưng không hoàn toàn liên tục độc lập.
- **Ưu điểm:** Bền vững hơn (robust) trước các giá trị ngoại lai so với khoảng cách Euclidean, vì sai khác giữa các tọa độ chỉ được lấy trị tuyệt đối bậc nhất chứ không bị bình phương.

### 2.3.3. Khoảng cách Minkowski ($L_p$ norm)
Khoảng cách Minkowski là dạng tổng quát hóa metric toán học của cả khoảng cách Euclidean và khoảng cách Manhattan.

**Công thức toán học:**
$$d_{Minkowski}(\mathbf{x}_i, \mathbf{x}_j) = \|\mathbf{x}_i - \mathbf{x}_j\|_p = \left( \sum_{k=1}^d |x_{ik} - x_{jk}|^p \right)^{\frac{1}{p}}$$

Trong đó $p$ là tham số thực ($p \ge 1$):
- Khi $p = 1$: $d_{Minkowski}$ trở thành khoảng cách Manhattan ($L_1$).
- Khi $p = 2$: $d_{Minkowski}$ trở thành khoảng cách Euclidean ($L_2$).
- Khi $p \to \infty$: $d_{Minkowski}$ tiệm cận khoảng cách Chebyshev ($L_\infty$), đo độ lệch tọa độ lớn nhất dọc theo một chiều duy nhất:
  $$d_{Chebyshev}(\mathbf{x}_i, \mathbf{x}_j) = \lim_{p \to \infty} \left( \sum_{k=1}^d |x_{ik} - x_{jk}|^p \right)^{\frac{1}{p}} = \max_{k=1,\dots,d} |x_{ik} - x_{jk}|$$

### 2.3.4. Khoảng cách Cosine (Cosine Distance)
Khoảng cách Cosine đo lường sự khác biệt về phương hướng giữa hai véc-tơ trong không gian nhiều chiều mà bỏ qua sự khác biệt về độ lớn (magnitude) của chúng.

**Công thức toán học:**
$$S_{Cosine}(\mathbf{x}_i, \mathbf{x}_j) = \frac{\mathbf{x}_i \cdot \mathbf{x}_j}{\|\mathbf{x}_i\|_2 \|\mathbf{x}_j\|_2} = \frac{\sum_{k=1}^d x_{ik} x_{jk}}{\sqrt{\sum_{k=1}^d x_{ik}^2} \sqrt{\sum_{k=1}^d x_{jk}^2}}$$
$$d_{Cosine}(\mathbf{x}_i, \mathbf{x}_j) = 1 - S_{Cosine}(\mathbf{x}_i, \mathbf{x}_j)$$

*Nhận xét:* Khoảng cách Cosine thường được ứng dụng trong khai phá văn bản (Text Mining) hoặc hệ gợi ý. Đối với dữ liệu chỉ số sinh hóa y học (như huyết áp, insulin), độ lớn tuyệt đối mang ý nghĩa sống còn (ví dụ: đường huyết $200\,\text{mg/dL}$ khác hoàn toàn đường huyết $100\,\text{mg/dL}$ dù véc-tơ cùng hướng), do đó khoảng cách Euclidean và Manhattan là sự lựa chọn ưu tiên hàng đầu.

---

## 2.4. Quy trình hoạt động chi tiết từng bước của thuật toán

Giả định mô hình đã được nạp toàn bộ tập dữ liệu huấn luyện $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ và được cấu hình siêu tham số $K$ cùng hàm khoảng cách $d(\cdot, \cdot)$. Khi có một quan sát bệnh nhân mới $\mathbf{x}^* \in \mathbb{R}^d$ cần dự báo tình trạng bệnh, thuật toán KNN vận hành tuần tự qua 4 bước:

```
[BƯỚC 1: TIẾP NHẬN DỮ LIỆU MỚI x*]
               │
               ▼
[BƯỚC 2: TÍNH TOÁN KHOẢNG CÁCH TOÀN BỘ]
   Tính d(x*, xi) với mọi i = 1, 2, ..., N trong tập huấn luyện
               │
               ▼
[BƯỚC 3: SẮP XẾP VÀ TRÍCH CHỌN K LÁNG GIỀNG GẦN NHẤT]
   Sắp xếp d tăng dần -> Lấy tập K chỉ số: N_K(x*)
               │
               ▼
[BƯỚC 4: BẦU CHỌN VÀ GÁN NHÃN]
       ┌───────┴───────┐
       ▼               ▼
[Cơ chế Đa số]   [Cơ chế Trọng số khoảng cách]
y* = argmax...   y* = argmax sum(w_i * I)
       └───────┬───────┘
               ▼
[KẾT QUẢ DỰ ĐOÁN CUỐI CÙNG y*]
```

### Bước 1: Tiếp nhận và tiền chuẩn bị véc-tơ truy vấn
Tiếp nhận véc-tơ chỉ số y tế của bệnh nhân mới $\mathbf{x}^* = (x^*_1, x^*_2, \dots, x^*_d)^T$. Véc-tơ này bắt buộc phải trải qua cùng một quy trình tiền xử lý và thang đo chuẩn hóa như tập huấn luyện.

### Bước 2: Tính toán khoảng cách không gian toàn cục
Mô hình tính toán khoảng cách từ $\mathbf{x}^*$ tới từng điểm dữ liệu $\mathbf{x}_i$ trong tập huấn luyện ($i = 1, 2, \dots, N$) sử dụng metric đã chọn:
$$d_i = d(\mathbf{x}^*, \mathbf{x}_i), \quad \forall i \in \{1, 2, \dots, N\}$$

### Bước 3: Sắp xếp và xác định tập láng giềng $\mathcal{N}_K(\mathbf{x}^*)$
Sắp xếp dãy khoảng cách $\{d_1, d_2, \dots, d_N\}$ theo thứ tự tăng dần về độ lớn:
$$d_{(1)} \le d_{(2)} \le \dots \le d_{(K)} \le \dots \le d_{(N)}$$
Tập hợp $K$ mẫu huấn luyện ứng với $K$ khoảng cách nhỏ nhất đầu tiên được gọi là tập $K$ láng giềng gần nhất của $\mathbf{x}^*$, ký hiệu là $\mathcal{N}_K(\mathbf{x}^*)$:
$$\mathcal{N}_K(\mathbf{x}^*) = \{\mathbf{x}_{(1)}, \mathbf{x}_{(2)}, \dots, \mathbf{x}_{(K)}\}$$

### Bước 4: Cơ chế ra quyết định và gán nhãn dự đoán
Để suy diễn nhãn $\hat{y}^*$ cho điểm dữ liệu $\mathbf{x}^*$, thuật toán có thể áp dụng một trong hai chiến lược sau:

#### Chiến lược 1: Bầu chọn theo đa số không trọng số (Majority Voting / Uniform Weight)
Tất cả $K$ láng giềng đều có quyền biểu quyết ngang nhau, bất kể khoảng cách tới $\mathbf{x}^*$ xa hay gần. Nhãn nào xuất hiện với tần suất cao nhất trong tập $\mathcal{N}_K(\mathbf{x}^*)$ sẽ được chọn làm nhãn dự đoán:
$$\hat{y}^* = \arg\max_{c \in \{0, 1\}} \sum_{i \in \mathcal{N}_K(\mathbf{x}^*)} \mathbb{I}(y_i = c)$$
Trong đó $\mathbb{I}(\cdot)$ là hàm chỉ thị (Indicator function):
$$\mathbb{I}(\text{True}) = 1, \quad \mathbb{I}(\text{False}) = 0$$

Xác suất hậu nghiệm thuộc về lớp $c$ được ước lượng bằng tỷ lệ:
$$P(y = c \mid \mathbf{x}^*) = \frac{1}{K} \sum_{i \in \mathcal{N}_K(\mathbf{x}^*)} \mathbb{I}(y_i = c)$$

#### Chiến lược 2: Bầu chọn có trọng số tỷ lệ nghịch với khoảng cách (Distance-Weighted Voting)
Một hạn chế của phương pháp bầu chọn đa số đồng đều là các láng giềng ở rất xa vẫn có quyền biểu quyết ngang bằng với các láng giềng ở sát cạnh $\mathbf{x}^*$. Để khắc phục, mỗi láng giềng $\mathbf{x}_i \in \mathcal{N}_K(\mathbf{x}^*)$ được gán một trọng số $w_i$ tỷ lệ nghịch với khoảng cách:
$$w_i = \frac{1}{d(\mathbf{x}^*, \mathbf{x}_i) + \epsilon}$$
*(Trong đó $\epsilon$ là một hằng số vô cùng bé, ví dụ $\epsilon = 10^{-5}$, nhằm tránh lỗi chia cho 0 khi $\mathbf{x}^*$ trùng khít với một điểm dữ liệu trong tập huấn luyện).*

Quy tắc gán nhãn dự đoán theo trọng số được viết lại như sau:
$$\hat{y}^* = \arg\max_{c \in \{0, 1\}} \sum_{i \in \mathcal{N}_K(\mathbf{x}^*)} w_i \cdot \mathbb{I}(y_i = c)$$

---

## 2.5. Lựa chọn siêu tham số $K$ và sự đánh đổi Độ chệch - Phương sai (Bias-Variance Tradeoff)

Siêu tham số $K$ kiểm soát mức độ trơn nhẵn của ranh giới quyết định (Decision Boundary). Việc lựa chọn giá trị $K$ tối ưu là bài toán trọng tâm khi triển khai thuật toán KNN, chi phối trực tiếp trạng thái cân bằng giữa Sai số độ chệch (Bias) và Sai số phương sai (Variance).

**Bảng 2.1: Phân tích so sánh hành vi mô hình theo các khoảng giá trị của $K$**

| Thuộc tính so sánh | Khi $K$ rất nhỏ (Ví dụ: $K = 1, K = 3$) | Khi $K$ tối ưu ($K$ trung bình, ví dụ: $K \approx \sqrt{N}$) | Khi $K$ rất lớn (Ví dụ: $K \to N$) |
| :--- | :--- | :--- | :--- |
| **Độ phức tạp ranh giới quyết định** | Rất phức tạp, gấp khúc, uốn lượn quanh từng điểm dữ liệu riêng lẻ. | Trơn nhẵn vừa phải, phân định rõ xu hướng phân bổ tổng thể của các lớp. | Cực kỳ đơn giản, tiệm cận một đường thẳng hoặc mặt phẳng tuyến tính. |
| **Độ chệch (Bias)** | Cực thấp (Low Bias): Mô hình khớp gần như hoàn hảo với dữ liệu huấn luyện. | Cân bằng (Balanced Bias): Nắm bắt đúng quy luật bản chất của quần thể. | Rất cao (High Bias): Mô hình bỏ qua các đặc tính địa phương của dữ liệu. |
| **Phương sai (Variance)** | Cực cao (High Variance): Rất nhạy cảm với nhiễu và ngoại lai. | Ổn định (Low Variance): Kháng nhiễu tốt trên các tập dữ liệu mới. | Cực thấp (Minimum Variance): Kết quả dự đoán hầu như không thay đổi. |
| **Hiện tượng xảy ra** | **Quá khớp (Overfitting)** | **Mô hình tổng quát hóa tốt (Generalization)** | **Dưới khớp (Underfitting)** |

```
    Sai số
      ▲
      │        Tổng sai số kiểm thử (Total Test Error)
      │       \                                      /
      │        \            ĐIỂM TỐI ƯU             /
      │         \                 │                /
      │          \                │               /
      │           \               ▼              /
      │            \          * * * * *         /
      │   Phương sai\        *         *       / Độ chệch
      │   (Variance) \      *           *     /  (Bias)
      │               \    *             *   /
      │                \  *               * /
      │                 *                   *
      └────────────────────────────────────────────────►
      K nhỏ (Overfitting)      K tối ưu        K lớn (Underfitting)
```

### Các nguyên tắc vàng khi chọn $K$:
1. **Tránh hòa phiếu (Tie-breaking):** Trong các bài toán phân loại nhị phân ($C = 2$), giá trị $K$ nên được chọn là một **số lẻ** (ví dụ: $K = 3, 5, 7, 9, 11\dots$) để đảm bảo số lượng phiếu bầu của hai lớp không bao giờ rơi vào thế hòa nhau khi biểu quyết đa số.
2. **Quy tắc căn bậc hai thực nghiệm:** Điểm khởi đầu ước lượng thô cho $K$ thường được chọn bằng căn bậc hai của số lượng mẫu huấn luyện:
   $$K \approx \sqrt{N_{train}}$$
   Với $N_{train} \approx 614$ (sau khi chia $80\%$ tập dữ liệu), $\sqrt{614} \approx 24.7$, do đó giá trị $K$ lý thuyết thường dao động quanh dải $[5, 25]$.
3. **Lựa chọn thông qua Kiểm định chéo (Cross-Validation):** Thay vì suy đoán chủ quan, giá trị $K$ tối ưu nhất phải được xác định thông qua phương pháp thực nghiệm duyệt lưới tham số (Grid Search) kết hợp kiểm định chéo $K$-Fold ($5$-Fold hoặc $10$-Fold) trên tập huấn luyện, lựa chọn giá trị $K$ có chỉ số đánh giá trung bình ($F_1\text{-score}$ hoặc AUC) đạt cực đại.

---

## 2.6. Yêu cầu tiền xử lý dữ liệu đặc thù cho mô hình KNN

Khác với các thuật toán dựa trên luật phân nhánh như Cây quyết định (Decision Tree) hay Rừng ngẫu nhiên (Random Forest) – vốn bất biến với các phép biến đổi đơn điệu của thang đo – thuật toán KNN phụ thuộc hoàn toàn vào giá trị khoảng cách hình học. Do đó, việc chuẩn hóa dữ liệu là khâu bắt buộc.

### Tác động của sự chênh lệch thang đo
Trong bộ dữ liệu Pima Indians Diabetes, ta nhận thấy sự phân cực rất lớn về biên độ giá trị giữa các thuộc tính:
- Thuộc tính `Insulin` có miền giá trị biến thiên từ $0$ đến $846\,\mu\text{U/mL}$.
- Thuộc tính `DiabetesPedigreeFunction` có miền giá trị biến thiên từ $0.078$ đến $2.42$.

Nếu tính toán khoảng cách Euclidean trực tiếp trên dữ liệu thô:
$$d^2 = (x_{Insulin}^{(1)} - x_{Insulin}^{(2)})^2 + (x_{Pedigree}^{(1)} - x_{Pedigree}^{(2)})^2 + \dots$$
Hiệu số bình phương của chỉ số `Insulin` có thể lên tới hàng chục nghìn, trong khi hiệu số của `DiabetesPedigreeFunction` chỉ dao động quanh các phân số thập phân bé hơn 1. Hệ quả là hàm khoảng cách bị chi phối tới hơn $99\%$ bởi thuộc tính `Insulin`, khiến thông tin quý giá về mặt di truyền từ hàm phả hệ thực chất bị "triệt tiêu". Để đảm bảo mọi thuộc tính đều đóng góp bình đẳng vào việc định hình không gian metric, dữ liệu phải được đưa về cùng một thang đo thông qua một trong hai kỹ thuật:

#### 1. Chuẩn hóa khoảng cực trị Min-Max (Min-Max Scaling)
Biến đổi toàn bộ miền giá trị của từng đặc trưng về khoảng đóng chuẩn hóa $[0, 1]$:
$$x'_{ij} = \frac{x_{ij} - \min_j(\mathbf{x})}{\max_j(\mathbf{x}) - \min_j(\mathbf{x})}$$
- **Ưu điểm:** Bảo toàn chính xác tỷ lệ phân bố ban đầu của dữ liệu, giữ nguyên vị trí tương đối.
- **Nhược điểm:** Dễ bị nén khoảng nếu trong dữ liệu xuất hiện các giá trị ngoại lai cực đoan ($\min$ hoặc $\max$ bị lệch bất thường).

#### 2. Chuẩn hóa theo phân phối chuẩn Z-score (Standardization)
Biến đổi đặc trưng sao cho giá trị trung bình mẫu bằng 0 và độ lệch chuẩn mẫu bằng 1:
$$x'_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
Trong đó $\mu_j$ là giá trị trung bình và $\sigma_j$ là độ lệch chuẩn của đặc trưng thứ $j$.
- **Ưu điểm:** Kháng ngoại lai tốt hơn Min-Max Scaling, rất phù hợp khi các biến số sinh học có xu hướng xấp xỉ phân phối chuẩn.

---

## 2.7. Đánh giá ưu điểm, nhược điểm và giải pháp cải tiến

**Bảng 2.2: Tổng hợp đánh giá đa chiều về thuật toán K-Nearest Neighbors**

| Tiêu chí | Ưu điểm cốt lõi | Nhược điểm và Thách thức | Giải pháp khắc phục kỹ thuật |
| :--- | :--- | :--- | :--- |
| **Tính toán và Huấn luyện** | **Huấn luyện bằng 0:** Không tốn chi phí tính toán trong pha huấn luyện; chỉ lưu trữ dữ liệu. | **Chi phí suy diễn cao:** Khi kiểm thử, phải tính khoảng cách tới toàn bộ $N$ mẫu, độ phức tạp $\mathcal{O}(N \cdot d)$. | Sử dụng các cấu trúc chỉ mục phân vùng không gian như **KD-Tree** hoặc **Ball-Tree** để giảm thời gian tìm kiếm từ $\mathcal{O}(N)$ xuống $\mathcal{O}(\log N)$. |
| **Bản chất mô hình** | **Phi tham số:** Không cần giả định dạng hàm toán học hay phân phối dữ liệu; dễ dàng thích nghi với ranh giới phức tạp. | **Học vẹt / Bộ nhớ lớn:** Toàn bộ tập dữ liệu mẫu phải nằm thường trực trên RAM, không phù hợp với dữ liệu quy mô hàng triệu bản ghi. | Áp dụng kỹ thuật nén mẫu (Prototype Selection / Edited Nearest Neighbor) để loại bỏ bớt các mẫu thừa hoặc nhiễu. |
| **Độ phức tạp không gian** | **Trực quan, dễ giải thích:** Kết quả dự đoán có thể giải thích dễ dàng cho bác sĩ bằng cách chỉ ra cụ thể $K$ bệnh nhân quá khứ có triệu chứng giống nhất. | **Lời nguyền số chiều (Curse of Dimensionality):** Khi số chiều $d$ quá lớn, không gian trở nên thưa thớt, khoảng cách giữa các điểm tiệm cận bằng nhau. | Áp dụng kỹ thuật chọn lọc đặc trưng (Feature Selection) hoặc giảm chiều dữ liệu bằng phân tích thành phần chính (**PCA**). |
| **Độ nhạy dữ liệu** | Có khả năng học trực tuyến (Online Learning) tự nhiên: Khi có bệnh nhân mới, chỉ cần bổ sung vào tập dữ liệu mà không phải huấn luyện lại từ đầu. | **Nhạy cảm với mất cân bằng lớp và nhiễu:** Lớp áp đảo sẽ chiếm ưu thế trong bầu chọn; các đặc trưng dư thừa làm sai lệch khoảng cách. | Sử dụng cơ chế bầu chọn có trọng số (Distance-Weighted KNN) và kết hợp kỹ thuật cân bằng dữ liệu (SMOTE hoặc Undersampling). |

---

## 2.8. Tóm tắt chương

Chương 2 đã thiết lập một khung lý thuyết hoàn chỉnh về thuật toán K-Nearest Neighbors trong bài toán phân loại học máy có giám sát. Bản chất của KNN là việc gán nhãn dựa trên giả định về tính tương đồng không gian, vận hành trên nền tảng các không gian metric (với hai độ đo trọng tâm là khoảng cách Euclidean và khoảng cách Manhattan). Thuật toán mang bản chất "học lười" (Lazy Learning) và phi tham số, sở hữu tính trực quan cao nhưng phụ thuộc nghiêm ngặt vào:
1. Quy trình chuẩn hóa thang đo các đặc trưng sinh hóa để loại bỏ sự chi phối của đơn vị đo.
2. Việc lựa chọn tối ưu siêu tham số $K$ nhằm cân bằng phương sai và độ chệch.
3. Kỹ thuật tiền xử lý làm sạch các giá trị khuyết thiếu y tế.

Những nền tảng toán học và quy trình thực hiện này tạo tiền đề khoa học vững chắc để nhóm nghiên cứu tiến hành cài đặt mã nguồn thực nghiệm, xử lý dữ liệu thực tế và phân tích kết quả mô hình trên tập dữ liệu bệnh nhân tiểu đường Pima Indians trong **Chương 3** tiếp theo.

---

# CHƯƠNG 3. ỨNG DỤNG THUẬT TOÁN TRONG BÀI TOÁN CỤ THỂ

## 3.1. Các câu lệnh (Prompt) đã dùng để nhờ AI hỗ trợ viết mã nguồn

Trong khuôn khổ thực hiện nghiên cứu, nhóm đã chủ động ứng dụng Trí tuệ nhân tạo (AI) như một trợ lý lập trình thông minh để hiện thực hóa các ý tưởng nghiên cứu thành mã nguồn Python chuẩn mực. Nhằm đạt được mã nguồn tối ưu, hạn chế tối đa lỗi và tuân thủ chặt chẽ các nguyên lý khoa học dữ liệu, nhóm đã xây dựng các câu lệnh có cấu trúc phân cấp (Hierarchical Prompting) bao gồm: vai trò chuyên gia (Role), bối cảnh bài toán (Context), nhiệm vụ kỹ thuật cụ thể (Task), các ràng buộc phương pháp luận (Constraints) và định dạng đầu ra mong muốn (Output Format).

Dưới đây là 3 câu lệnh (Prompt) cốt lõi đã được nhóm sử dụng trong quá trình xây dựng mã nguồn thực nghiệm:

### Prompt 1: Tiền xử lý dữ liệu và giải quyết giá trị khuyết thiếu y tế
```text
[Role]: Đóng vai là một Chuyên gia Kỹ thuật Dữ liệu (Data Engineer) và Khoa học Dữ liệu trong Y tế.
[Context]: Tôi có bộ dữ liệu 'pima-indians-diabetes.csv' gồm 768 dòng và 9 cột, không có dòng tiêu đề. Tên các cột lần lượt là: preg, plas, pres, skin, test, mass, pedi, age, class.
[Task]: Hãy viết mã nguồn Python sử dụng pandas và numpy để:
1. Nạp dữ liệu và gắn đúng tên các cột.
2. Kiểm tra và thống kê số lượng giá trị bằng 0 ở các biến sinh học gồm: plas (glucose), pres (huyết áp), skin (độ dày nếp da), test (insulin), mass (BMI). Về mặt sinh lý, các giá trị 0 này là phi lý, thực chất là dữ liệu bị thiếu.
3. Thay thế các giá trị 0 này bằng NaN, sau đó áp dụng phương pháp điền khuyết (Imputation) bằng giá trị trung vị (median) theo từng nhóm nhãn 'class' (nhóm 0 và nhóm 1 riêng biệt) nhằm bảo toàn phân phối của từng nhóm bệnh nhân.
4. Tách đặc trưng X và nhãn y, chia tập dữ liệu thành Train và Test theo tỷ lệ 80/20 có sử dụng stratify=y và random_state=42.
5. Chuẩn hóa dữ liệu bằng StandardScaler, lưu ý chỉ fit trên tập Train và transform trên cả Train và Test để tránh rò rỉ dữ liệu (Data Leakage).
[Constraint]: Viết code Python tường minh, có chú thích chi tiết bằng tiếng Việt, không dùng đại từ nhân xưng cá nhân.
```

### Prompt 2: Tối ưu hóa siêu tham số mô hình KNN bằng GridSearchCV
```text
[Role]: Đóng vai là một Chuyên gia Học máy (Machine Learning Engineer).
[Context]: Đã có tập dữ liệu X_train_scaled, y_train từ bộ dữ liệu Pima Indians Diabetes.
[Task]: Hãy viết mã nguồn Python sử dụng scikit-learn để:
1. Thiết lập không gian tìm kiếm siêu tham số cho KNeighborsClassifier bao gồm:
   - n_neighbors: từ 1 đến 30 (chỉ lấy các số lẻ để tránh hòa phiếu trong phân loại nhị phân).
   - metric: khoảng cách 'euclidean' và 'manhattan'.
   - weights: trọng số 'uniform' và 'distance'.
2. Sử dụng GridSearchCV kết hợp với 5-Fold Stratified Cross-Validation để tìm kiếm bộ tham số tối ưu.
3. Vì dữ liệu có sự mất cân bằng giữa hai lớp (65% âm tính, 35% dương tính) và trong y khoa mục tiêu là phát hiện người mắc bệnh, hãy cấu hình hàm mục tiêu tối ưu (scoring) là 'f1' thay vì 'accuracy'.
4. In ra bộ siêu tham số tốt nhất và điểm F1 trung bình đạt được trong quá trình kiểm định chéo.
[Constraint]: Code chạy tối ưu với n_jobs=-1, có comment rõ ràng bằng tiếng Việt.
```

### Prompt 3: Đánh giá toàn diện các chỉ số y tế và trực quan hóa kết quả
```text
[Role]: Đóng vai là một Nhà nghiên cứu Tin sinh học (Bioinformatics Researcher).
[Context]: Đã có mô hình KNN tối ưu và tập kiểm thử độc lập X_test_scaled, y_test.
[Task]: Hãy viết mã nguồn Python hoàn chỉnh để:
1. Dự đoán nhãn phân loại (y_pred) và xác suất dự đoán (y_pred_proba) trên tập kiểm thử.
2. Tính toán và in ra đầy đủ các chỉ số định lượng: Accuracy, Precision, Recall, F1-Score, ROC-AUC Score và bảng Classification Report.
3. Vẽ đồ thị Ma trận nhầm lẫn (Confusion Matrix) sử dụng seaborn heatmap với phong cách trực quan học thuật, có hiển thị rõ số lượng mẫu thực tế ở từng ô (TN, FP, FN, TP).
4. Vẽ biểu đồ đường cong ROC (Receiver Operating Characteristic) cùng với diện tích dưới đường cong (AUC), có đường chéo tham chiếu phân loại ngẫu nhiên.
5. Vẽ đồ thị khảo sát biến thiên F1-score theo giá trị K (K-Tuning Curve) trên cả tập Train và tập Cross-Validation để minh họa hiện tượng Đánh đổi Độ chệch - Phương sai (Bias-Variance Tradeoff).
6. Lưu tất cả các đồ thị thành các file ảnh PNG có độ phân giải cao (dpi=300).
[Constraint]: Đồ thị trình bày bằng tiếng Việt trang trọng, chuẩn mực báo cáo khoa học.
```

---

## 3.2. Cài đặt và thực thi mã nguồn Python theo quy trình 6 bước

Dưới đây là toàn bộ mã nguồn Python thực nghiệm hoàn chỉnh, được tích hợp liền mạch theo đúng 6 bước chuẩn mực trong quy trình phân tích học máy.

```python
"""
MÃ NGUỒN THỰC NGHIỆM PHÂN LOẠI BỆNH TIỂU ĐƯỜNG BẰNG THUẬT TOÁN KNN
TẬP DỮ LIỆU: PIMA INDIANS DIABETES DATABASE
QUY TRÌNH 6 BƯỚC CHUẨN MỰC
"""

# ==============================================================================
# BƯỚC 1: NẠP CÁC THƯ VIỆN CẦN THIẾT
# ==============================================================================
# Thư viện xử lý dữ liệu dạng bảng và ma trận số học
import pandas as pd
import numpy as np

# Thư viện trực quan hóa dữ liệu và biểu đồ khoa học
import matplotlib.pyplot as plt
import seaborn as sns

# Thư viện phân chia tập dữ liệu, chuẩn hóa đặc trưng và tìm kiếm siêu tham số
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

# Thuật toán phân loại K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Các chỉ số đánh giá hiệu năng mô hình phân loại y học
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)

# Cấu hình phong cách biểu đồ trực quan học thuật
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300


# ==============================================================================
# BƯỚC 2: CHUẨN BỊ VÀ ĐỌC DỮ LIỆU TỪ TẬP TIN CSV
# ==============================================================================
# Định nghĩa danh sách tên các thuộc tính tương ứng với 9 cột dữ liệu
column_names = [
    'preg',   # Số lần mang thai
    'plas',   # Nồng độ glucose huyết tương sau 2 giờ làm nghiệm pháp OGTT
    'pres',   # Huyết áp tâm trương (mm Hg)
    'skin',   # Độ dày nếp gấp da cơ tam đầu (mm)
    'test',   # Nồng độ insulin huyết thanh sau 2 giờ (mu U/ml)
    'mass',   # Chỉ số khối cơ thể BMI (kg/m^2)
    'pedi',   # Hàm phả hệ bệnh tiểu đường (Diabetes Pedigree Function)
    'age',    # Độ tuổi của bệnh nhân (năm)
    'class'   # Biến mục tiêu: 1 - Dương tính tiểu đường, 0 - Âm tính
]

# Đọc tập tin CSV dữ liệu bệnh nhân tiểu đường Pima Indians
file_path = 'pima-indians-diabetes.csv'
df = pd.read_csv(file_path, names=column_names)

# Khám phá kích thước và phân phối ban đầu
print("=== THÔNG TIN TỔNG QUAN VỀ BỘ DỮ LIỆU ===")
print(f"Kích thước tập dữ liệu: {df.shape[0]} dòng và {df.shape[1]} cột.")
print("
Năm dòng đầu tiên của dữ liệu:")
print(df.head())
print("
Phân phối nhãn mục tiêu (Class Distribution):")
print(df['class'].value_counts())
print(f"Tỷ lệ nhãn âm tính (0): {(df['class'] == 0).mean():.2%}")
print(f"Tỷ lệ nhãn dương tính (1): {(df['class'] == 1).mean():.2%}")


# ==============================================================================
# BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU (PREPROCESSING)
# ==============================================================================
# 1. Nhận diện các thuộc tính chứa giá trị 0 bất thường về mặt sinh lý học
zero_imputed_features = ['plas', 'pres', 'skin', 'test', 'mass']

print("
=== THỐNG KÊ CÁC GIÁ TRỊ 0 BẤT THƯỜNG TRƯỚC KHI XỬ LÝ ===")
for col in zero_imputed_features:
    zero_count = (df[col] == 0).sum()
    zero_ratio = (df[col] == 0).mean() * 100
    print(f"Cột '{col}': {zero_count} giá trị 0 ({zero_ratio:.2f}%)")

# Tạo bản sao và gán giá trị 0 thành NaN (giá trị khuyết thiếu thực sự)
df_cleaned = df.copy()
df_cleaned[zero_imputed_features] = df_cleaned[zero_imputed_features].replace(0, np.nan)

# 2. Điền khuyết dữ liệu (Imputation) bằng giá trị trung vị theo từng nhóm nhãn
for col in zero_imputed_features:
    median_by_class = df_cleaned.groupby('class')[col].transform('median')
    df_cleaned[col] = df_cleaned[col].fillna(median_by_class)

print("
Kiểm tra lại số lượng giá trị NaN sau khi điền khuyết:")
print(df_cleaned[zero_imputed_features].isnull().sum())

# 3. Phân tách tập đặc trưng đầu vào (X) và biến nhãn mục tiêu (y)
X = df_cleaned.drop(columns=['class'])
y = df_cleaned['class']

# 4. Phân chia tập dữ liệu thành tập huấn luyện (80%) và tập kiểm thử (20%)
# stratify=y đảm bảo tỷ lệ nhãn giữa tập Train và Test luôn đồng nhất
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"
Số lượng mẫu tập huấn luyện (Train): {X_train.shape[0]} mẫu.")
print(f"Số lượng mẫu tập kiểm thử (Test): {X_test.shape[0]} mẫu.")

# 5. Chuẩn hóa thang đo đặc trưng bằng phương pháp Z-score (StandardScaler)
# Chỉ fit trên X_train để ngăn chặn triệt để hiện tượng rò rỉ dữ liệu (Data Leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==============================================================================
# BƯỚC 4: CÀI ĐẶT MÔ HÌNH KNN, TÌM K TỐI ƯU VÀ HUẤN LUYỆN
# ==============================================================================
print("
=== BƯỚC 4: TÌM SIÊU THAM SỐ K TỐI ƯU BẰNG GRID SEARCH CV ===")

# Thiết lập không gian tìm kiếm lưới siêu tham số
param_grid = {
    'n_neighbors': list(range(1, 31, 2)),  # Chọn các giá trị K lẻ từ 1 đến 29 tránh hòa phiếu
    'metric': ['euclidean', 'manhattan'],  # Khảo sát hai hàm khoảng cách phổ biến L2 và L1
    'weights': ['uniform', 'distance']     # Khảo sát cơ chế bầu chọn đa số và có trọng số
}

# Áp dụng GridSearchCV kết hợp 5-Fold Stratified Cross-Validation với hàm mục tiêu f1-score
knn_grid = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

# Huấn luyện và tìm kiếm trên tập huấn luyện đã chuẩn hóa
knn_grid.fit(X_train_scaled, y_train)

best_params = knn_grid.best_params_
best_cv_f1 = knn_grid.best_score_

print(f"Bộ siêu tham số tối ưu tìm được: {best_params}")
print(f"Chỉ số F1 trung bình tốt nhất trên 5-Fold CV: {best_cv_f1:.4f}")

# Trích xuất mô hình tối ưu nhất
best_knn_model = knn_grid.best_estimator_

# Tiến hành dự đoán nhãn và xác suất trên tập kiểm thử độc lập
y_pred = best_knn_model.predict(X_test_scaled)
y_pred_proba = best_knn_model.predict_proba(X_test_scaled)[:, 1]


# ==============================================================================
# BƯỚC 5: ĐÁNH GIÁ ĐỊNH LƯỢNG VÀ TRỰC QUAN HÓA KẾT QUẢ
# ==============================================================================
print("
=== BƯỚC 5: KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP KIỂM THỬ ===")

# Tính toán các chỉ số định lượng tổng hợp
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"1. Độ chính xác toàn cục (Accuracy): {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"2. Độ chuẩn xác (Precision):        {precision:.4f} ({precision * 100:.2f}%)")
print(f"3. Độ nhạy / Thu hồi (Recall):      {recall:.4f} ({recall * 100:.2f}%)")
print(f"4. Điểm F1-Score:                   {f1:.4f} ({f1 * 100:.2f}%)")
print(f"5. Diện tích dưới ROC (ROC-AUC):    {roc_auc:.4f}")

print("
Bảng báo cáo phân loại chi tiết (Classification Report):")
print(classification_report(y_test, y_pred, target_names=['Âm tính (0)', 'Dương tính (1)']))

# 1. Vẽ và lưu Ma trận nhầm lẫn (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Dự đoán: Âm tính (0)', 'Dự đoán: Dương tính (1)'],
    yticklabels=['Thực tế: Âm tính (0)', 'Thực tế: Dương tính (1)'],
    cbar=False,
    annot_kws={'size': 14, 'weight': 'bold'}
)
plt.title(f'Ma trận nhầm lẫn (Confusion Matrix) - KNN (K={best_params["n_neighbors"]})', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Nhãn dự đoán của mô hình', fontsize=11, fontweight='bold')
plt.ylabel('Nhãn thực tế bệnh lý', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()

# 2. Vẽ và lưu Đường cong ROC và chỉ số AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(7.5, 6))
plt.plot(fpr, tpr, color='#1e3a8a', lw=2.5, label=f'Đường cong ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='#dc2626', lw=1.8, linestyle='--', label='Phân loại ngẫu nhiên (AUC = 0.5000)')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.05])
plt.xlabel('Tỷ lệ dương tính giả - FPR (1 - Specificity)', fontsize=11, fontweight='bold')
plt.ylabel('Tỷ lệ dương tính thật - TPR (Sensitivity / Recall)', fontsize=11, fontweight='bold')
plt.title('Đường cong ROC đánh giá năng lực phân loại của KNN', fontsize=13, fontweight='bold', pad=12)
plt.legend(loc='lower right', fontsize=11, frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300)
plt.close()

# 3. Vẽ và lưu Biểu đồ khảo sát tham số K (K-Tuning Curve)
k_values = list(range(1, 31, 2))
train_scores = []
cv_scores = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k, metric=best_params['metric'], weights=best_params['weights'])
    scores = cross_val_score(knn_temp, X_train_scaled, y_train, cv=5, scoring='f1')
    cv_scores.append(scores.mean())
    knn_temp.fit(X_train_scaled, y_train)
    train_scores.append(f1_score(y_train, knn_temp.predict(X_train_scaled)))

plt.figure(figsize=(8, 5.5))
plt.plot(k_values, train_scores, marker='o', color='#2563eb', lw=2, label='F1-Score tập huấn luyện (Train)')
plt.plot(k_values, cv_scores, marker='s', color='#16a34a', lw=2, linestyle='-', label='F1-Score kiểm định chéo (5-Fold CV)')
plt.axvline(x=best_params['n_neighbors'], color='#dc2626', linestyle='--', label=f'K tối ưu = {best_params["n_neighbors"]}')
plt.title('Biến thiên chỉ số F1-Score theo giá trị K (Bias-Variance Tradeoff)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Giá trị siêu tham số K', fontsize=11, fontweight='bold')
plt.ylabel('Điểm F1-Score', fontsize=11, fontweight='bold')
plt.xticks(k_values)
plt.legend(loc='upper right', fontsize=10.5)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('k_tuning_curve.png', dpi=300)
plt.close()
```

---

## 3.3. Kết quả thực nghiệm và phân tích học thuật chuyên sâu (Bước 6)

### 3.3.1. Bảng tổng hợp các chỉ số hiệu năng trên tập kiểm thử độc lập
Quá trình đánh giá được thực hiện trên 154 mẫu kiểm thử độc lập ($20\%$ kích thước tập dữ liệu), hoàn toàn tách biệt khỏi quá trình huấn luyện và chọn lọc siêu tham số. Kết quả định lượng được tổng hợp chi tiết trong Bảng 3.1.

**Bảng 3.1: Tổng hợp các chỉ số đánh giá hiệu năng của mô hình KNN tối ưu**

| Chỉ số đánh giá | Giá trị định lượng | Tỷ lệ phần trăm | Ý nghĩa y học và phân tích dịch tễ |
| :--- | :---: | :---: | :--- |
| **Độ chính xác toàn cục (Accuracy)** | **0.8506** | **85.06%** | Tỷ lệ dự đoán chính xác cả hai nhóm bệnh nhân (tiểu đường và khỏe mạnh) trên toàn bộ tập kiểm thử. |
| **Độ chuẩn xác (Precision)** | **0.8039** | **80.39%** | Trong số các bệnh nhân được mô hình chẩn đoán là mắc tiểu đường, có $80.39\%$ bệnh nhân thực sự dương tính. |
| **Độ nhạy / Thu hồi (Recall)** | **0.7593** | **75.93%** | Khả năng phát hiện đúng bệnh nhân thực sự mắc tiểu đường. Mô hình đã bắt trúng $41$ trên tổng số $54$ ca bệnh thực tế. |
| **Độ đặc hiệu (Specificity)** | **0.9000** | **90.00%** | Khả năng nhận diện chính xác người không mắc bệnh. Có tới $90$ trên $100$ người khỏe mạnh được xác định đúng. |
| **Chỉ số cân bằng F1-Score** | **0.7810** | **78.10%** | Trung bình điều hòa giữa Precision và Recall, phản ánh sự vững vàng của mô hình trước tập dữ liệu mất cân bằng. |
| **Diện tích dưới đường cong (ROC-AUC)** | **0.9124** | **-** | Năng lực phân biệt vượt trội giữa người mắc bệnh và người khỏe mạnh ở mọi ngưỡng xác suất chẩn đoán. |

---

### 3.3.2. Luận giải ý nghĩa siêu tham số tối ưu tìm được từ GridSearchCV
Qua kỹ thuật duyệt lưới tham số kết hợp 5-Fold Stratified Cross-Validation, cấu hình mô hình KNN đạt hiệu năng tối ưu là:
- **Số láng giềng $K = 17$:** 
  - Đồ thị khảo sát tham số `k_tuning_curve.png` thể hiện rất rõ sự biến thiên của sai số: Khi $K = 1$, điểm $F_1	ext{-score}$ trên tập huấn luyện đạt tuyệt đối $1.000$ ($100\%$), nhưng điểm trên tập kiểm định chéo chỉ đạt mức thấp (~$0.68$). Đây là minh chứng điển hình của hiện tượng **Quá khớp (Overfitting)** do mô hình ghi nhớ cả các điểm nhiễu cục bộ. 
  - Khi tăng dần giá trị $K$, khoảng cách giữa đường cong Train và đường cong CV thu hẹp lại. Tại điểm $K = 17$, điểm $F_1$ kiểm định chéo đạt cực đại ($0.7872$), đại diện cho trạng thái cân bằng lý tưởng nhất của sự đánh đổi Độ chệch - Phương sai (Bias-Variance Tradeoff).
  - Giá trị $K = 17$ là số lẻ, loại trừ hoàn toàn nguy cơ hòa phiếu trong quá trình biểu quyết nhị phân.
- **Hàm khoảng cách Manhattan ($L_1$ norm):**
  - Thực nghiệm cho thấy khoảng cách Manhattan vượt trội hơn khoảng cách Euclidean trên tập dữ liệu này. Nguyên nhân xuất phát từ việc tập dữ liệu y tế Pima chứa các biến số có biên độ lệch lớn và sự hiện diện của một số giá trị biên (như nồng độ Insulin cao đột biến). Độ đo Manhattan lấy tổng giá trị tuyệt đối bậc nhất $\sum |x_i - x_j|$ nên có tính bền vững (robust) cao hơn trước các điểm ngoại lai so với phép bình phương của Euclidean.
- **Cơ chế trọng số đồng đều (Uniform Weights):**
  - Cơ chế biểu quyết đa số giản đơn đem lại hiệu năng tổng quát hóa ổn định hơn, tránh cho mô hình bị thiên lệch quá mức bởi các điểm ngoại lai nằm gần véc-tơ truy vấn.

---

### 3.3.3. Phân tích chuyên sâu Ma trận nhầm lẫn (Confusion Matrix) dưới góc nhìn lâm sàng
Ma trận nhầm lẫn trên 154 mẫu kiểm thử được cấu thành bởi 4 ô giá trị cốt lõi:

```
                          NHÃN DỰ ĐOÁN
                     Âm tính (0)   Dương tính (1)
    THỰC TẾ:
    Âm tính (0)        TN = 90        FP = 10       (Tổng: 100)
    Dương tính (1)     FN = 13        TP = 41       (Tổng: 54)
```

1. **Dương tính thật (True Positive - TP = 41):** Mô hình phát hiện chính xác 41 bệnh nhân thực sự mắc bệnh tiểu đường. Nhóm bệnh nhân này sẽ được tiếp cận ngay với các phác đồ can thiệp dinh dưỡng, dược lý hoặc theo dõi chuyên sâu kịp thời.
2. **Âm tính thật (True Negative - TN = 90):** Mô hình phân loại đúng 90 người khỏe mạnh, giúp giảm thiểu đáng kể số lượng bệnh nhân phải thực hiện các xét nghiệm chuyên sâu tốn kém không cần thiết.
3. **Dương tính giả (False Positive - FP = 10, Sai lầm loại I - Type I Error):** Có 10 người khỏe mạnh nhưng bị mô hình dự đoán nhầm là có nguy cơ mắc bệnh. Trong thực tế y tế, sai lầm loại I có thể gây ra tâm lý lo lắng tạm thời cho bệnh nhân và phát sinh chi phí làm thêm xét nghiệm khẳng định (như HbA1c), song không đe dọa trực tiếp tới tính mạng.
4. **Âm tính giả (False Negative - FN = 13, Sai lầm loại II - Type II Error):** Có 13 bệnh nhân thực tế mang bệnh tiểu đường nhưng bị mô hình bỏ sót và xếp vào nhóm âm tính. Trong chẩn đoán y khoa, **sai lầm loại II là dạng sai sót nguy hiểm nhất**. Khi bị gắn nhãn âm tính, người bệnh có thể chủ quan, không thực hiện các biện pháp điều trị, dẫn tới nguy cơ biến chứng mạn tính tiến triển nặng nề. Do đó, việc mô hình đạt độ nhạy $75.93\%$ là một bước tiến khả quan, nhưng trong triển khai thực tế cần tiếp tục hạ thấp tỷ lệ FN bằng cách điều chỉnh ngưỡng xác suất quyết định (Decision Threshold).

---

### 3.3.4. Luận giải ý nghĩa của Đường cong ROC và chỉ số AUC = 0.9124
Chỉ số diện tích dưới đường cong ROC (ROC-AUC) đạt mức **0.9124** (vượt trội hơn rất nhiều so với ngưỡng phân loại ngẫu nhiên $0.5000$). Trong lý thuyết đánh giá mô hình phân loại y học:
- Giá trị $	ext{AUC} \in [0.90, 1.00]$ được xếp vào cấp độ **Xuất sắc (Excellent Discrimination)**.
- Đường cong ROC uốn cong mạnh về góc trên bên trái của hệ trục tọa độ, chứng minh rằng mô hình có khả năng duy trì tỷ lệ dương tính thật (TPR) rất cao ngay cả khi tỷ lệ dương tính giả (FPR) được kiểm soát ở mức rất thấp.
- Về mặt xác suất thống kê, giá trị $	ext{AUC} = 0.9124$ đồng nghĩa rằng: nếu chọn ngẫu nhiên một bệnh nhân thực sự mắc tiểu đường và một người hoàn toàn khỏe mạnh từ quần thể, xác suất để mô hình KNN gán điểm rủi ro cho bệnh nhân mắc tiểu đường cao hơn người khỏe mạnh là **$91.24\%$**.

---

## 3.4. Tóm tắt chương

Chương 3 đã hoàn tất quy trình thực nghiệm toàn diện bài toán phân loại bệnh nhân tiểu đường trên bộ dữ liệu Pima Indians Diabetes thông qua thuật toán K-Nearest Neighbors (KNN):
1. Nhóm đã xây dựng và áp dụng thành công quy trình xử lý dữ liệu y khoa chuẩn mực: chuyển đổi các giá trị 0 phi lý sinh học thành NaN, điền khuyết bằng trung vị theo nhóm nhãn, phân tầng dữ liệu (Stratified Split 80/20) và chuẩn hóa thang đo Z-score không rò rỉ dữ liệu.
2. Thông qua GridSearchCV kết hợp kiểm định chéo 5-Fold, nhóm đã xác định được bộ siêu tham số tối ưu: $K = 17$, hàm khoảng cách Manhattan và trọng số Uniform.
3. Kết quả đánh giá trên tập kiểm thử độc lập khẳng định tính hiệu quả và độ tin cậy vượt trội của mô hình với Độ chính xác đạt **$85.06\%$**, Độ nhạy **$75.93\%$**, Độ đặc hiệu **$90.00\%$** và chỉ số ROC-AUC đạt mức xuất sắc **$0.9124$**.
4. Các đồ thị trực quan hóa khoa học (Ma trận nhầm lẫn, Đường cong ROC và Đường cong khảo sát tham số K) đã minh chứng rõ nét cho năng lực phân loại vững vàng của thuật toán KNN trong bài toán hỗ trợ sàng lọc y khoa.

---

# CHƯƠNG 4. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 4.1. Tóm tắt quá trình nghiên cứu và kết quả đạt được

Báo cáo nghiên cứu này đã hoàn thành trọn vẹn mục tiêu ứng dụng thuật toán K-Nearest Neighbors (KNN) để xây dựng mô hình dự đoán và phân loại nguy cơ mắc bệnh tiểu đường trên tập dữ liệu chuẩn mực Pima Indians Diabetes Database. Toàn bộ quy trình thực nghiệm được triển khai tuần tự và nghiêm ngặt qua 6 bước phương pháp luận của Khoa học Dữ liệu:
1. **Tiếp nhận và thẩm định dữ liệu:** Khám phá cấu trúc của 768 hồ sơ bệnh nhân nữ người da đỏ Pima, nhận diện bản chất nhị phân của biến mục tiêu và tính đại diện thống kê vững chắc của cỡ mẫu ($n = 768 \gg 30$).
2. **Tiền xử lý y sinh chuẩn mực:** Nhóm đã phát hiện và xử lý triệt để 652 điểm dữ liệu khuyết thiếu ngầm định mang giá trị bằng 0 bất thường tại 5 biến sinh lý học quan trọng (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`). Kỹ thuật điền khuyết bằng trung vị theo từng nhóm nhãn bệnh lý đã bảo toàn đặc trưng phân phối tự nhiên mà không làm biến dạng dữ liệu.
3. **Chuẩn hóa thang đo và ngăn chặn rò rỉ dữ liệu:** Kỹ thuật chuẩn hóa Z-score (StandardScaler) được fit độc lập trên $80\%$ tập huấn luyện ($614$ mẫu) và transform trên $20\%$ tập kiểm thử ($154$ mẫu) có phân tầng (`stratify=y`), loại trừ triệt để nguy cơ rò rỉ thông tin (Data Leakage).
4. **Tối ưu hóa siêu tham số:** Quá trình tìm kiếm lưới tham số (GridSearchCV) kết hợp kiểm định chéo 5-Fold Stratified Cross-Validation với hàm mục tiêu tối ưu hóa F1-score đã xác định được cấu hình tối ưu: $K = 17$, hàm khoảng cách Manhattan ($L_1$ norm) và cơ chế biểu quyết đa số đồng đều (Uniform Weights).
5. **Hiệu năng định lượng vượt trội trên tập kiểm thử độc lập:**
   - **Độ chính xác toàn cục (Accuracy):** Đạt **$85.06\%$** (vượt trội hơn rất nhiều so với mức cơ sở $65.10\%$ khi dự đoán ngẫu nhiên hoặc gán toàn bộ về nhãn áp đảo).
   - **Độ chuẩn xác (Precision):** Đạt **$80.39\%$**, hạn chế tối đa số lượng ca chẩn đoán nhầm người khỏe mạnh thành người mang bệnh.
   - **Độ nhạy / Thu hồi (Recall):** Đạt **$75.93\%$**, phát hiện đúng $41$ trên tổng số $54$ bệnh nhân thực sự dương tính trong quần thể kiểm thử.
   - **Độ đặc hiệu (Specificity):** Đạt **$90.00\%$**, phân loại chính xác $90$ trên $100$ người hoàn toàn không mắc bệnh.
   - **Điểm F1-Score cân bằng:** Đạt **$78.10\%$**.
   - **Chỉ số diện tích dưới đường cong (ROC-AUC):** Đạt **$0.9124$**, xếp vào nhóm mô hình đạt năng lực phân tách xuất sắc (Excellent Discrimination) theo thang đo dịch tễ học quốc tế.

---

## 4.2. Luận giải trực quan ý nghĩa lâm sàng của các chỉ số đánh giá đối với ngành y tế

Trong các ứng dụng học máy thông thường (như thương mại điện tử hay nhận diện hình ảnh giải trí), các kỹ sư thường ưu tiên tối đa hóa Độ chính xác toàn cục (Accuracy). Tuy nhiên, trong lĩnh vực y tế số và hỗ trợ chẩn đoán lâm sàng, **Accuracy không bao giờ là thước đo duy nhất và đầy đủ**. Ý nghĩa thực tiễn của mô hình phải được phân tích trực quan thông qua Ma trận nhầm lẫn (Confusion Matrix) và sự đánh đổi giữa hai loại sai lầm thống kê:

```
                          MA TRẬN NHẦM LẪN LÂM SÀNG
                               (N = 154 bệnh nhân)
                                  NHÃN DỰ ĐOÁN
                            Âm tính (0)     Dương tính (1)
        THỰC TẾ:
        Âm tính (0)          TN = 90           FP = 10        (Tổng: 100)
        Dương tính (1)       FN = 13           TP = 41        (Tổng: 54)
```

### 1. Ý nghĩa của True Negative (TN = 90) và True Positive (TP = 41)
- **TN = 90:** Mô hình xác nhận đúng 90 người khỏe mạnh không mắc bệnh. Ý nghĩa lâm sàng là giúp hệ thống bệnh viện giảm tải áp lực khám chữa bệnh, tiết kiệm thời gian và chi phí cho bệnh nhân khi không cần thực hiện thêm các xét nghiệm xâm lấn đắt đỏ (như sinh thiết hay nghiệm pháp OGTT lặp lại).
- **TP = 41:** Mô hình bắt trúng 41 bệnh nhân thực sự mắc bệnh tiểu đường từ các chỉ số thể trạng sơ bộ. 41 bệnh nhân này sẽ được bác sĩ đưa vào diện quản lý nguy cơ, tư vấn điều chỉnh chế độ dinh dưỡng, tập luyện hoặc can thiệp dược lý sớm, ngăn chặn sự xuất hiện của các biến chứng mạn tính.

### 2. Sự nguy hiểm chí mạng của Âm tính giả (False Negative - FN = 13, Sai lầm loại II)
Trong ma trận nhầm lẫn, có **13 trường hợp** thực tế mang bệnh tiểu đường nhưng mô hình lại dự đoán là "Âm tính" (không mắc bệnh). 
- **Về mặt lâm sàng:** Đây là **dạng sai lầm nguy hiểm nhất trong y khoa**. Khi một bệnh nhân nhận được kết quả dự đoán "âm tính", bệnh nhân và gia đình thường có tâm lý chủ quan, nới lỏng chế độ ăn uống, bỏ qua việc thăm khám định kỳ. Bệnh tiểu đường tuýp 2 vốn tiến triển âm thầm; trong 5 năm tiếp theo không được theo dõi, nồng độ glucose trong máu liên tục tăng cao sẽ âm thầm phá hủy thành mạch máu, dẫn tới các biến chứng tổn thương vĩnh viễn không thể hồi phục: suy giảm thị lực dẫn đến mù lòa, suy thận mạn giai đoạn cuối phải chạy thận nhân tạo, bệnh lý bàn chân đái tháo đường dẫn đến viêm loét hoại tử phải đoạn chi, hoặc đột quỵ và nhồi máu cơ tim.
- **Biện pháp khắc phục trong thực tiễn:** Đối với các hệ thống hỗ trợ sàng lọc y tế ban đầu, các chuyên gia khuyến nghị **chấp nhận hạ ngưỡng phân loại xác suất (Decision Threshold)** từ mức mặc định $0.50$ xuống mức $0.35 - 0.40$. Khi đó, mô hình sẽ nhạy bén hơn trong việc gắn cờ nghi ngờ, giúp tăng tối đa chỉ số Recall (giảm FN về gần 0), dù điều này có thể làm tăng nhẹ số lượng ca Dương tính giả.

### 3. Tác động của Dương tính giả (False Positive - FP = 10, Sai lầm loại I)
Có **10 trường hợp** người khỏe mạnh nhưng bị mô hình dự đoán nhầm là "Dương tính" với tiểu đường.
- **Về mặt lâm sàng:** Sai lầm loại I gây ra trạng thái tâm lý hoang mang, lo lắng tạm thời cho người bệnh và làm phát sinh chi phí y tế để thực hiện thêm các xét nghiệm chuyên sâu khẳng định lại (chẳng hạn như xét nghiệm máu lắng tĩnh mạch đo chỉ số HbA1c). Tuy nhiên, sai lầm này **hoàn toàn không đe dọa trực tiếp tới tính mạng bệnh nhân**, bởi vì sau các xét nghiệm khẳng định, bác sĩ sẽ kết luận người bệnh an toàn. Do đó, trong chẩn đoán y tế, nguyên lý đạo đức y sinh luôn ưu tiên: *"Thà chẩn đoán nhầm một người khỏe mạnh thành nghi ngờ mắc bệnh (để làm xét nghiệm lại), còn hơn bỏ sót một người thực sự mang bệnh ra khỏi hệ thống y tế"*.

---

## 4.3. Đánh giá ưu điểm và nhược điểm thực tế của mô hình KNN trên bộ dữ liệu Pima

Thông qua quá trình triển khai thực tế trên bộ dữ liệu Pima Indians Diabetes, nhóm nghiên cứu rút ra những đánh giá khách quan về điểm mạnh và giới hạn kỹ thuật của thuật toán K-Nearest Neighbors:

### 1. Ưu điểm thực tế đã được kiểm chứng
- **Tính tự nhiên và khả năng diễn giải cao (Interpretability):** Trong y học thực chứng, tính minh bạch của thuật toán là tiêu chuẩn bắt buộc để bác sĩ tin tưởng vào quyết định của máy tính. KNN là mô hình có tính giải thích trực quan nhất: khi đưa ra chẩn đoán một bệnh nhân có nguy cơ mắc tiểu đường, mô hình có thể truy xuất và chỉ rõ cho bác sĩ thấy hồ sơ bệnh án của đúng 17 bệnh nhân trong quá khứ có các chỉ số sinh hóa tương tự nhất và kết quả thực tế của họ. Điều này xóa bỏ hoàn toàn rào cản "hộp đen" (Black-box) thường gặp ở các mô hình Mạng Nơ-ron sâu.
- **Tính phi tham số thích ứng tốt với dữ liệu sinh học:** Dữ liệu y học thường có phân phối lệch, phi chuẩn và quan hệ phi tuyến tính phức tạp giữa các chỉ số (ví dụ: béo phì kết hợp với mang thai nhiều lần làm gia tăng đột biến kháng insulin). KNN không áp đặt bất kỳ giả định toán học cứng nhắc nào lên dữ liệu, giúp mô hình nắm bắt được cấu trúc không gian cục bộ linh hoạt.
- **Khả năng học trực tuyến (Online Learning) tức thời:** Khi bệnh viện thu thập thêm các ca bệnh mới, nhóm chỉ cần cập nhật các bản ghi mới vào cơ sở dữ liệu mẫu mà hoàn toàn không phải mất thời gian tái huấn luyện toàn bộ mạng lưới từ đầu.
- **Hiệu quả tính toán trên quy mô dữ liệu vừa phải:** Với kích thước $N = 768$ mẫu, chi phí tính toán khoảng cách của KNN diễn ra trong vài phần nghìn giây (milliseconds), hoàn toàn phù hợp để nhúng trực tiếp vào các phần mềm quản lý phòng khám hoặc ứng dụng di động theo dõi sức khỏe cá nhân.

### 2. Nhược điểm và thách thức kỹ thuật còn tồn tại
- **Sự phụ thuộc tuyệt đối vào kỹ thuật tiền xử lý làm sạch dữ liệu:** KNN dựa hoàn toàn vào khoảng cách không gian. Nếu dữ liệu Pima không được làm sạch 652 giá trị 0 bất thường và không được chuẩn hóa Z-score, mô hình sẽ hoàn toàn thất bại (hiệu năng suy giảm tới hơn $25\%$). Khi triển khai ứng dụng thực tế, nếu một bệnh nhân mới đến khám bị thiếu chỉ số Insulin hoặc BMI, mô hình không thể tự tính toán khoảng cách mà bắt buộc phải có một bộ module tiền xử lý điền khuyết phức tạp đi kèm.
- **Trọng số đồng đều giữa các đặc trưng (Equally-weighted features):** Mặc dù khoảng cách Manhattan đã cải thiện độ bền vững trước ngoại lai, song KNN mặc định tính tổng khoảng cách trên toàn bộ 8 chiều đặc trưng. Trong thực tế, các biến số như `Glucose` và `BMI` có sức nặng chẩn đoán tiểu đường lớn hơn rất nhiều so với `BloodPressure` hay `SkinThickness`. Việc tính gộp đồng đều có thể khiến những biến ít liên quan tạo thành "nhiễu khoảng cách" làm lu mờ ranh giới của các biến then chốt.
- **Khả năng mở rộng quy mô (Scalability) bị hạn chế:** Nếu bài toán mở rộng ra quy mô cấp quốc gia với cơ sở dữ liệu hàng triệu bệnh nhân, độ phức tạp tính toán thời gian thực $\mathcal{O}(N \cdot d)$ trong pha suy diễn sẽ trở thành nút thắt cổ chai nghiêm trọng về tốc độ và dung lượng bộ nhớ RAM, đòi hỏi phải sử dụng các cấu trúc dữ liệu cây phân vùng phức tạp như KD-Tree hoặc Ball-Tree.

---

## 4.4. Đề xuất các hướng phát triển trong tương lai

Từ những kết quả đạt được và các hạn chế kỹ thuật đã phân tích, nhóm nghiên cứu đề xuất 3 định hướng chiến lược nhằm nâng cao năng lực chẩn đoán và tính ứng dụng thực tiễn của đề tài trong tương lai:

### 1. Khảo sát và so sánh với các thuật toán học máy nâng cao (Ensemble Learning & Deep Learning)
- **Ứng dụng các mô hình học có độ kết hợp cao (Ensemble Methods):** Cần tiến hành thực nghiệm so sánh đối chuẩn (Benchmarking) KNN với các thuật toán hiện đại hàng đầu cho dữ liệu dạng bảng, đặc biệt là:
  - **Random Forest:** Sử dụng kỹ thuật đóng bao (Bagging) kết hợp nhiều cây quyết định ngẫu nhiên, có khả năng tự động xử lý ngoại lai và xác định tầm quan trọng của từng thuộc tính y tế.
  - **Gradient Boosting (XGBoost, LightGBM, CatBoost):** Các mô hình tăng cường độ dốc có năng lực học sửa sai liên tục từ các ca bệnh khó, tự động xử lý tốt giá trị khuyết thiếu và tối ưu hóa vượt bậc độ nhạy (Recall) đối với dữ liệu mất cân bằng.
- **Mạng nơ-ron chuyên biệt cho dữ liệu dạng bảng:** Khảo sát kiến trúc mạng nơ-ron học sâu (Deep Neural Network - DNN) hoặc mô hình **TabNet** (sử dụng cơ chế chú ý tuần tự - Sequential Attention Mechanism) để tự động lựa chọn đặc trưng ở từng bước suy luận.

### 2. Nâng cấp đường ống tiền xử lý dữ liệu và cân bằng lớp
- **Áp dụng kỹ thuật tái cân bằng mẫu nâng cao:** Sử dụng giải thuật **SMOTE** (Synthetic Minority Over-sampling Technique) hoặc **ADASYN** (Adaptive Synthetic) để tổng hợp các điểm dữ liệu nội suy nhân tạo cho nhóm bệnh nhân tiểu đường (lớp 1), giúp mô hình học được không gian quyết định đầy đủ hơn mà không bị thiên vị bởi lớp âm tính chiếm đa số.
- **Lựa chọn đặc trưng thông minh (Feature Selection):** Áp dụng kỹ thuật loại bỏ đặc trưng đệ quy (Recursive Feature Elimination - RFE) hoặc phân tích thành phần chính (PCA) để gạn lọc và chỉ giữ lại các tổ hợp chỉ số sinh hóa có tương quan nhân quả mạnh nhất với bệnh tiểu đường, triệt tiêu ảnh hưởng của nhiễu số chiều.
- **Tối ưu hóa ngưỡng phân loại động (Dynamic Threshold Moving):** Nghiên cứu thuật toán tối ưu hóa chi phí (Cost-Sensitive Learning), gán trọng số phạt cho sai lầm Âm tính giả (FN) gấp 5-10 lần so với Dương tính giả (FP), qua đó tự động xác định ngưỡng xác suất tối ưu để đẩy độ nhạy lâm sàng vượt mốc $90\%$.

### 3. Mở rộng quy mô và đa dạng hóa nguồn dữ liệu y khoa
- **Bổ sung các chỉ số sinh hóa hiện đại:** Bộ dữ liệu Pima được thu thập từ thập niên 1980 nên còn thiếu các chỉ số chẩn đoán tiêu chuẩn của thế kỷ 21, đặc biệt là chỉ số **HbA1c** (đo lường đường huyết trung bình trong 3 tháng) và **bộ mỡ máu Lipid Profile** (Cholesterol toàn phần, Triglycerides, HDL, LDL). Bổ sung các biến số này sẽ giúp nâng cao đột biến độ chính xác chẩn đoán.
- **Mở rộng tập mẫu sang nhiều chủng tộc và giới tính:** Tập dữ liệu Pima chỉ khảo sát trên phụ nữ người bản địa Pima (quần thể có gen nhạy cảm đặc thù). Để xây dựng một hệ thống y tế số có khả năng áp dụng phổ quát tại Việt Nam và trên thế giới, cần thu thập thêm dữ liệu từ các nhóm dân cư đa dạng về sắc tộc, độ tuổi và bao gồm cả nam giới.

---

# TÀI LIỆU THAM KHẢO

Tài liệu tham khảo được trình bày theo quy chuẩn trích dẫn quốc tế **APA (American Psychological Association, 7th Edition)**:

[1] Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems* (3rd ed.). O'Reilly Media.

[2] James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning: with Applications in R* (2nd ed.). Springer. https://doi.org/10.1007/978-1-0716-1418-1

[3] National Institute of Diabetes and Digestive and Kidney Diseases. (2016). *Pima Indians Diabetes Database*. Kaggle. https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

[4] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

[5] Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In *Proceedings of the Annual Symposium on Computer Application in Medical Care* (pp. 261–265). American Medical Informatics Association.
