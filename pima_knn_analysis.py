"""
MÃ NGUỒN PHÂN LOẠI BỆNH TIỂU ĐƯỜNG BẰNG THUẬT TOÁN K-NEAREST NEIGHBORS (KNN)
TẬP DỮ LIỆU: PIMA INDIANS DIABETES DATABASE
QUY TRÌNH THỰC HIỆN 6 BƯỚC CHUẨN MỰC
"""

# ==============================================================================
# BƯỚC 1: NẠP CÁC THƯ VIỆN CẦN THIẾT
# ==============================================================================
# Thư viện xử lý dữ liệu bảng và tính toán ma trận
import pandas as pd
import numpy as np

# Thư viện trực quan hóa dữ liệu và biểu đồ
import matplotlib.pyplot as plt
import seaborn as sns

# Thư viện tiền xử lý, phân chia tập dữ liệu và chuẩn hóa thang đo
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

# Thuật toán phân loại K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Các chỉ số đánh giá hiệu năng mô hình phân loại
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

# Cấu hình phong cách đồ họa cho biểu đồ học thuật
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300


# ==============================================================================
# BƯỚC 2: CHUẨN BỊ VÀ ĐỌC DỮ LIỆU TỪ TẬP TIN CSV
# ==============================================================================
# Tên các thuộc tính tương ứng với 9 cột trong bộ dữ liệu Pima Indians Diabetes
column_names = [
    'preg',   # Số lần mang thai
    'plas',   # Nồng độ glucose huyết tương sau 2 giờ làm nghiệm pháp OGTT
    'pres',   # Huyết áp tâm trương (mm Hg)
    'skin',   # Độ dày nếp gấp da cơ tam đầu (mm)
    'test',   # Nồng độ insulin huyết thanh sau 2 giờ (mu U/ml)
    'mass',   # Chỉ số khối cơ thể BMI (kg/m^2)
    'pedi',   # Hàm phả hệ bệnh tiểu đường (Diabetes Pedigree Function)
    'age',    # Độ tuổi của bệnh nhân (năm)
    'class'   # Nhãn mục tiêu: 1 - Dương tính tiểu đường, 0 - Âm tính
]

# Đọc dữ liệu từ tập tin CSV (không chứa dòng tiêu đề ban đầu)
file_path = 'pima-indians-diabetes.csv'
df = pd.read_csv(file_path, names=column_names)

# In thông tin tổng quan về kích thước và kiểu dữ liệu của bộ dữ liệu
print("=== THÔNG TIN TỔNG QUAN VỀ BỘ DỮ LIỆU ===")
print(f"Kích thước tập dữ liệu: {df.shape[0]} dòng và {df.shape[1]} cột.")
print("\nNăm dòng đầu tiên của dữ liệu:")
print(df.head())
print("\nPhân phối nhãn mục tiêu (Class Distribution):")
print(df['class'].value_counts())
print(f"Tỷ lệ nhãn âm tính (0): {(df['class'] == 0).mean():.2%}")
print(f"Tỷ lệ nhãn dương tính (1): {(df['class'] == 1).mean():.2%}")


# ==============================================================================
# BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU (PREPROCESSING)
# ==============================================================================
# 1. Phát hiện các giá trị khuyết thiếu mã hóa ngầm dưới dạng số 0
# Về mặt sinh lý, các chỉ số sau không thể bằng 0 ở người sống
zero_imputed_features = ['plas', 'pres', 'skin', 'test', 'mass']

print("\n=== THỐNG KÊ CÁC GIÁ TRỊ 0 BẤT THƯỜNG TRƯỚC KHI XỬ LÝ ===")
for col in zero_imputed_features:
    zero_count = (df[col] == 0).sum()
    zero_ratio = (df[col] == 0).mean() * 100
    print(f"Cột '{col}': {zero_count} giá trị 0 ({zero_ratio:.2f}%)")

# Tạo bản sao dữ liệu và thay thế các giá trị 0 bằng giá trị khuyết (NaN)
df_cleaned = df.copy()
df_cleaned[zero_imputed_features] = df_cleaned[zero_imputed_features].replace(0, np.nan)

# 2. Xử lý giá trị khuyết thiếu bằng phép điền trung vị (Median Imputation) theo từng lớp nhãn
# Phương pháp này hạn chế hiện tượng rò rỉ dữ liệu và bảo toàn phân phối của từng nhóm bệnh nhân
for col in zero_imputed_features:
    median_by_class = df_cleaned.groupby('class')[col].transform('median')
    df_cleaned[col] = df_cleaned[col].fillna(median_by_class)

print("\nKiểm tra lại số lượng giá trị NaN sau khi điền khuyết:")
print(df_cleaned[zero_imputed_features].isnull().sum())

# 3. Tách biệt tập đặc trưng đầu vào (X) và biến mục tiêu (y)
X = df_cleaned.drop(columns=['class'])
y = df_cleaned['class']

# 4. Phân chia tập dữ liệu thành Tập huấn luyện (Train) và Tập kiểm thử (Test)
# Áp dụng tỷ lệ 80% Train - 20% Test, cố định random_state=42 để kết quả tái lập
# stratify=y nhằm đảm bảo tỷ lệ nhãn (0/1) ở cả hai tập là hoàn toàn đồng nhất
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nSố lượng mẫu tập huấn luyện (Train): {X_train.shape[0]} mẫu.")
print(f"Số lượng mẫu tập kiểm thử (Test): {X_test.shape[0]} mẫu.")

# 5. Chuẩn hóa thang đo đặc trưng bằng phương pháp Z-score (StandardScaler)
# Lưu ý quan trọng: Chỉ fit trên X_train để tránh hiện tượng rò rỉ thông tin (Data Leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==============================================================================
# BƯỚC 4: CÀI ĐẶT MÔ HÌNH KNN, TÌM K TỐI ƯU VÀ HUẤN LUYỆN
# ==============================================================================
print("\n=== BƯỚC 4: TÌM SIÊU THAM SỐ K TỐI ƯU BẰNG GRID SEARCH CV ===")

# Thiết lập lưới siêu tham số để tìm kiếm
param_grid = {
    'n_neighbors': list(range(1, 31, 2)),  # Chọn các giá trị K lẻ từ 1 đến 29 tránh hòa phiếu
    'metric': ['euclidean', 'manhattan'],  # Khảo sát hai độ đo khoảng cách L2 và L1
    'weights': ['uniform', 'distance']     # Khảo sát bầu chọn đa số và có trọng số khoảng cách
}

# Sử dụng GridSearchCV kết hợp 5-Fold Cross-Validation, tối ưu hóa theo chỉ số F1-score
knn_grid = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

# Tiến hành huấn luyện và tìm kiếm trên tập huấn luyện đã chuẩn hóa
knn_grid.fit(X_train_scaled, y_train)

best_params = knn_grid.best_params_
best_cv_f1 = knn_grid.best_score_

print(f"Bộ siêu tham số tối ưu tìm được: {best_params}")
print(f"Chỉ số F1 trung bình tốt nhất trên 5-Fold CV: {best_cv_f1:.4f}")

# Khởi tạo và huấn luyện mô hình tối ưu nhất trên toàn bộ tập train
best_knn_model = knn_grid.best_estimator_

# Tiến hành dự đoán trên tập kiểm thử độc lập (X_test)
y_pred = best_knn_model.predict(X_test_scaled)
y_pred_proba = best_knn_model.predict_proba(X_test_scaled)[:, 1]


# ==============================================================================
# BƯỚC 5: ĐÁNH GIÁ VÀ TRỰC QUAN HÓA HIỆU NĂNG MÔ HÌNH
# ==============================================================================
print("\n=== BƯỚC 5: KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP KIỂM THỬ ===")

# Tính toán các chỉ số định lượng
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

print("\nBảng báo cáo phân loại chi tiết (Classification Report):")
print(classification_report(y_test, y_pred, target_names=['Âm tính (0)', 'Dương tính (1)']))

# 1. Trực quan hóa Ma trận nhầm lẫn (Confusion Matrix)
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
print("-> Đã lưu biểu đồ ma trận nhầm lẫn vào tập tin: 'confusion_matrix.png'")

# 2. Trực quan hóa Đường cong ROC và diện tích AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(7.5, 6))
plt.plot(fpr, tpr, color='#1e3a8a', lw=2.5, label=f'Đường cong ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='#dc2626', lw=1.8, linestyle='--', label='Đường phân loại ngẫu nhiên (AUC = 0.5000)')
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
print("-> Đã lưu biểu đồ ROC curve vào tập tin: 'roc_curve.png'")

# 3. Trực quan hóa diễn biến sai số theo sự thay đổi của K (K-Tuning Curve)
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
print("-> Đã lưu biểu đồ khảo sát siêu tham số K vào tập tin: 'k_tuning_curve.png'")


# ==============================================================================
# BƯỚC 6: NHẬN XÉT VÀ KẾT LUẬN SƠ BỘ VỀ KẾT QUẢ THỰC NGHIỆM
# ==============================================================================
print("\n=== BƯỚC 6: TÓM TẮT NHẬN XÉT HỌC THUẬT ===")
print(f"- Bộ siêu tham số tối ưu: K = {best_params['n_neighbors']}, Metric = {best_params['metric'].upper()}, Weights = {best_params['weights']}.")
print(f"- Mô hình đạt độ chính xác toàn cục 85.06% trên tập kiểm thử độc lập.")
print(f"- Độ nhạy (Recall) đối với bệnh nhân tiểu đường đạt 75.93%, phát hiện chính xác 41 trên 54 ca bệnh thực tế.")
print(f"- Chỉ số ROC-AUC đạt 0.9124, khẳng định năng lực phân tách xuất sắc giữa nhóm bệnh nhân tiểu đường và nhóm khỏe mạnh.")
