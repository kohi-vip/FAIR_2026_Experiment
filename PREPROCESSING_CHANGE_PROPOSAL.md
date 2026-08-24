# Đề xuất tiền xử lý Sparkov an toàn bộ nhớ trên Kaggle

## 1. Phạm vi

Tài liệu này chỉ áp dụng cho bộ dữ liệu **Sparkov / Credit Card Transactions Fraud Detection**.

Mục tiêu:

- giữ toàn bộ dữ liệu `fraudTrain.csv`;
- tiếp tục dùng Stratified 5-Fold;
- preprocessing chỉ fit trên training fold;
- SMOTE chỉ áp dụng trên training fold;
- validation giữ nguyên tỷ lệ gian lận tự nhiên;
- giúp một fold có thể chạy tuần tự qua 12 baseline và TabNet trên Kaggle;
- không trộn kết quả memory-safe với kết quả strict reproduction.

## 2. Dữ liệu và nguyên nhân tràn RAM

Sparkov có:

- 1.289.169 giao dịch hợp lệ;
- 7.506 giao dịch gian lận;
- tổng cộng 1.296.675 giao dịch;
- tỷ lệ gian lận khoảng 0,58%.

Số dòng lớn là một nguyên nhân quan trọng, nhưng không phải nguyên nhân duy nhất. Mức sử dụng RAM tăng theo tích:

```text
số dòng sau SMOTE × số feature sau One-Hot × số byte mỗi giá trị
```

Trong một training fold, SMOTE 1:1 làm số dòng sau resampling xấp xỉ hai lần số giao dịch hợp lệ trong training. Các cột như `merchant`, `city`, `job`, `dob`, `state` và các trường categorical khác có thể tạo hàng nghìn dummy feature.

Với dense matrix:

```text
RAM ma trận float32 ≈ rows × features × 4 bytes
RAM ma trận float64 ≈ rows × features × 8 bytes
```

Trong lúc chạy còn đồng thời tồn tại:

- dữ liệu raw;
- train và validation chưa transform;
- ma trận sau One-Hot;
- ma trận mới do SMOTE tạo;
- các bản sao tạm của pandas, scikit-learn và NumPy.

Do đó peak RAM có thể lớn gấp 2,5–3 lần kích thước ma trận train cuối cùng và vượt giới hạn RAM của Kaggle.

## 3. Đặc trưng Sparkov cần giữ khả năng diễn giải

Các feature có ý nghĩa trực tiếp nên được giữ:

| Nhóm | Ví dụ | Cách xử lý đề xuất |
|---|---|---|
| Giao dịch | `amt`, `category`, `merchant` | scale numeric; One-Hot categorical |
| Khách hàng | `gender`, `city`, `state`, `zip`, `city_pop`, `job` | One-Hot categorical hoặc scale numeric đúng kiểu |
| Địa lý | `lat`, `long`, `merch_lat`, `merch_long` | StandardScaler |
| Thời gian | `transaction_hour`, `transaction_day`, `transaction_month`, `transaction_weekday`, `is_weekend`, `unix_time` | tạo từ timestamp theo cấu hình reproduction hiện tại; scale numeric |
| Truy vết | `trans_num` | giữ riêng để truy vết, không dùng làm feature |
| Identifier | `cc_num` | loại khỏi feature matrix theo cấu hình hiện tại |
| Cá nhân cardinality cao | `first`, `last`, `street` | loại khỏi feature matrix theo cấu hình hiện tại |
| Target | `is_fraud` | giữ nguyên, không scale/encode |

Không giải thích các mẫu tổng hợp do SMOTE tạo. XAI chỉ sử dụng giao dịch thật từ validation fold.

## 4. Hai protocol phải tách biệt

### 4.1 Strict reproduction

```text
One-Hot toàn bộ category
→ dense matrix
→ SMOTE 1:1 trên training
→ 13 models
```

Protocol này gần mô tả gốc nhất nhưng có nguy cơ không khả thi với khoảng 29–30 GB RAM của Kaggle.

Tên cấu hình đề xuất:

```text
strict_reproduction
```

Nếu tràn RAM, phải ghi nhận `resource_failure`; không được coi memory-safe là cùng một protocol.

### 4.2 Full-row memory-safe

Giữ toàn bộ dòng, nhưng giới hạn số dummy feature của từng categorical column:

```python
OneHotEncoder(
    handle_unknown="infrequent_if_exist",
    max_categories=64,
    sparse_output=False,
    dtype=np.float32,
)
```

Tên cấu hình đề xuất:

```text
full_rows_memory_safe_ohe64
```

Đặc điểm:

- không sampling hoặc loại bớt giao dịch;
- category phổ biến vẫn có dummy feature riêng;
- category hiếm được gộp vào nhóm infrequent;
- giữ One-Hot, StandardScaler và SMOTE như pipeline chính;
- giảm mạnh số feature và lượng RAM;
- kết quả phải được ghi là technical adaptation, không phải strict reproduction.

Nếu 64 category mỗi cột vẫn vượt RAM, chỉ được chuyển sang `max_categories=32` trong một cấu hình riêng:

```text
full_rows_memory_safe_ohe32
```

Không tự thay đổi giữa 64 và 32 trong cùng một bảng kết quả.

## 5. Pipeline đề xuất

```text
fraudTrain.csv
    ↓
schema và target validation
    ↓
loại technical ID theo config đã chốt
    ↓
tạo time features theo config đã chốt
    ↓
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ↓
chọn đúng một fold
    ↓
fit imputer/scaler/OHE trên TRAIN fold
    ↓
transform validation bằng preprocessor của TRAIN
    ↓
ép float32
    ↓
giải phóng raw/intermediate không còn dùng
    ↓
SMOTE(k_neighbors=5, sampling_strategy=1.0, random_state=42) trên TRAIN
    ↓
validation không SMOTE
    ↓
validation gate
    ↓
huấn luyện tuần tự 13 models
    ↓
giải phóng model/RAM/VRAM sau từng model
```

## 6. Kiểm tra RAM trước SMOTE

Trước khi tạo ma trận sau SMOTE, phải ước lượng:

```python
majority_rows = int(y_train_raw.value_counts().max())
estimated_rows_after_smote = majority_rows * 2
estimated_features = len(feature_names)

estimated_final_gb = (
    estimated_rows_after_smote
    * estimated_features
    * np.dtype(np.float32).itemsize
    / 1024**3
)

estimated_peak_gb = estimated_final_gb * 3.0
```

Quy tắc:

- lưu lại `estimated_final_gb` và `estimated_peak_gb` trong output notebook;
- nếu peak ước lượng vượt khoảng 24 GB, không tiếp tục tạo dense SMOTE matrix;
- dừng với thông báo rõ thay vì để Kaggle kill kernel;
- chuyển từ OHE64 sang OHE32 chỉ khi người chạy chủ động chọn cấu hình đó.

Hệ số 3,0 là biên an toàn cho input, output và các bản sao tạm; đây là ước lượng kỹ thuật, không phải giới hạn chính xác của thư viện.

## 7. Quản lý bộ nhớ bắt buộc

1. Dùng `float32` ngay từ output của encoder/scaler.
2. Không tạo DataFrame dense chứa toàn bộ train sau SMOTE.
3. Không lưu train/validation processed ra CSV.
4. Chỉ lưu `model_metrics.csv` và `summary_5fold.csv`.
5. Huấn luyện model tuần tự, không song song 13 model.
6. Sau mỗi model:

```python
del model
gc.collect()
torch.cuda.empty_cache()
```

7. Trước SMOTE, giải phóng các biến raw/intermediate không còn cần.
8. Chạy Sparkov trong một Kaggle session riêng.
9. Không kỳ vọng hai notebook chia sẻ `/kaggle/working`; kết quả cần được tải hoặc publish riêng trước khi tổng hợp.

## 8. Không tự áp dụng các thay đổi sau

Các thay đổi dưới đây làm thay đổi phương pháp mạnh hơn và không thuộc đề xuất mặc định:

- lấy mẫu bớt số dòng;
- RandomUnderSampler;
- thay SMOTE bằng class weights;
- SMOTENC;
- Target Encoding;
- Feature Hashing;
- thay One-Hot bằng Ordinal Encoding;
- bỏ thêm feature có khả năng diễn giải chỉ để giảm RAM;
- chạy XAI trên synthetic samples.

Nếu cần một trong các phương án này, phải tạo ablation/config riêng.

## 9. Cấu hình đề xuất

```python
SPARKOV_MEMORY_CONFIG = {
    "protocol": "full_rows_memory_safe_ohe64",
    "keep_all_rows": True,
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
    "one_hot": True,
    "one_hot_handle_unknown": "infrequent_if_exist",
    "one_hot_max_categories": 64,
    "matrix_dtype": "float32",
    "smote": True,
    "smote_k_neighbors": 5,
    "smote_sampling_strategy": 1.0,
    "smote_train_only": True,
    "validation_resampling": False,
    "estimated_peak_multiplier": 3.0,
    "estimated_peak_limit_gb": 24.0,
}
```

## 10. Điều kiện hoàn thành

Một fold chỉ được đưa sang model layer khi tất cả điều kiện sau đạt:

- dùng toàn bộ indices của fold đã chọn;
- train và validation không giao nhau;
- encoder/scaler chỉ fit trên train;
- validation không SMOTE;
- train sau SMOTE cân bằng 1:1;
- không còn NaN hoặc Inf;
- train và validation có cùng số feature;
- ma trận sử dụng `float32`;
- peak RAM ước lượng không vượt ngưỡng cấu hình;
- 13 models dùng cùng một fold và cùng một processed representation;
- metrics chỉ được ghi vào fold tương ứng.

## 11. Ảnh hưởng tới bài nghiên cứu

Kết quả `full_rows_memory_safe_ohe64` có thể được dùng làm thực nghiệm kỹ thuật đầy đủ trên Kaggle, nhưng bảng và phần phương pháp phải ghi rõ:

> Các category hiếm của Sparkov được gộp bằng giới hạn tối đa 64 category cho mỗi feature nhằm giữ toàn bộ giao dịch trong giới hạn bộ nhớ. Stratified 5-Fold, train-only preprocessing và SMOTE 1:1 không thay đổi.

Không được gắn nhãn kết quả này là exact reproduction nếu báo cáo cũ không sử dụng cơ chế gộp category hiếm.

## 12. Trạng thái

```text
STATUS: WAITING_FOR_USER_APPROVAL
```

Sau khi được phê duyệt, thay đổi chỉ nên áp dụng cho nhánh Sparkov trong `Main_Testing.ipynb`; MLG-ULB và IEEE-CIS giữ nguyên pipeline hiện tại.
