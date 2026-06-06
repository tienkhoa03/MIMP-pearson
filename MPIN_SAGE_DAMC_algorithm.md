# Giải thuật Bổ Khuyết MPIN SAGE++DAMC với Pearson-filtered Graph

## Tổng quan

MPIN (Missing data imPutation with Incremental Networks) dùng mạng GNN để bổ khuyết dữ liệu cảm biến bị thiếu. Phiên bản dùng ở đây là **SAGE++DAMC** (GraphSAGE với Dual Aggregation + Multi-hop + multi-scale Concat) kết hợp **lọc đồ thị bằng tương quan Pearson giữa các feature**.

Ý tưởng cốt lõi:
- Mỗi **timestep** (dòng dữ liệu) được coi là một **node** trong đồ thị.
- Các node gần nhau trong không gian feature được nối cạnh — nhờ đó node bị thiếu dữ liệu có thể "hỏi" thông tin từ các timestep lân cận.
- Pearson filter quyết định: chỉ dùng những feature thực sự tương quan với nhau làm cơ sở tính khoảng cách KNN, thay vì dùng toàn bộ feature.

---

## Dữ liệu mẫu minh hoạ (Labsensor thu nhỏ)

Labsensor (IBRL) có 3 feature: **Nhiệt độ**, **Độ ẩm**, **Ánh sáng**, đo theo từng giây. Ở đây dùng 6 timestep để minh hoạ:

```
Timestep | Nhiệt độ (°C) | Độ ẩm (%) | Ánh sáng (lux) | Mask
---------|---------------|-----------|----------------|-----
t=0      |  23.1         |  45.2     |  120           | [1, 1, 1]
t=1      |  23.3         |  45.5     |  ???           | [1, 1, 0]  ← thiếu Ánh sáng
t=2      |  ???          |  46.1     |  125           | [0, 1, 1]  ← thiếu Nhiệt độ
t=3      |  23.8         |  ???      |  130           | [1, 0, 1]  ← thiếu Độ ẩm
t=4      |  24.0         |  47.2     |  132           | [1, 1, 1]
t=5      |  24.1         |  47.5     |  135           | [1, 1, 1]
```

Mask = 1 nghĩa là giá trị đó **quan sát được**, Mask = 0 là **bị thiếu** (cần bổ khuyết).

Trong thực tế Labsensor với `stream=0.01` có ~34 885 dòng × 3 feature, nhưng logic hoàn toàn giống ví dụ 6 dòng trên.

---

## Bước 1 — Tiền xử lý dữ liệu

### 1.1 Nạp dữ liệu và tính mask gốc

Đọc toàn bộ file Labsensor. Với mỗi ô, nếu giá trị là NaN thì mask = 0, ngược lại mask = 1.

> **Ví dụ:** t=1 có Ánh sáng = NaN → mask[1, 2] = 0.

### 1.2 Subsampling theo stream

Nếu `stream=0.01`, chỉ giữ lại 1% dòng đầu tiên để giảm thời gian tính toán.

> **Ví dụ:** Tập gốc 3 488 540 dòng → giữ 34 885 dòng đầu.

### 1.3 Điền NaN toàn cục bằng mean cột

Trước khi đưa vào model, điền tạm các ô NaN bằng **trung bình cột** (tính trên toàn bộ dữ liệu đã load, bỏ qua NaN):

```
Nhiệt độ mean = (23.1 + 23.3 + 23.8 + 24.0 + 24.1) / 5 = 23.66
Độ ẩm mean   = (45.2 + 45.5 + 46.1 + 47.2 + 47.5) / 5 = 46.3
Ánh sáng mean = (120 + 125 + 130 + 132 + 135) / 5 = 128.4

→ Sau khi điền:
t=1: [23.3,  45.5,  128.4]   (Ánh sáng điền = 128.4)
t=2: [23.66, 46.1,  125  ]   (Nhiệt độ điền = 23.66)
t=3: [23.8,  46.3,  130  ]   (Độ ẩm điền = 46.3)
```

Mask giữ nguyên — nó vẫn nhớ ô nào thực sự quan sát được.

### 1.4 Chuẩn hoá (Standardization)

Tính mean và std toàn bộ (trên X sau khi đã điền NaN), sau đó chuẩn hoá:

```
X_std = (X - mean_global) / std_global
```

> **Ví dụ (giản lược):** mean_global ≈ 67.9, std_global ≈ 90.3 cho Labsensor thực.  
> Sau chuẩn hoá, tất cả giá trị nằm quanh 0, giúp model học ổn định hơn.

---

## Bước 2 — Tách tập đánh giá (data_transform)

### Mục đích

Để đánh giá model, ta cần biết **giá trị thật** tại một số vị trí. Vì vậy, ta **giả vờ làm mất** thêm một phần nhỏ các giá trị vốn đã quan sát được — model phải bổ khuyết chúng, rồi ta so sánh kết quả với giá trị thật.

### Cách thực hiện

Trong số tất cả vị trí có mask = 1, chọn ngẫu nhiên `eval_ratio` phần (ví dụ 10%):
- Đặt giá trị X tại những vị trí đó về 0.
- Đặt mask tại đó về 0 (model không "thấy" chúng khi train).
- Ghi lại chúng trong `eval_mask = 1` (dùng để đánh giá sau).

> **Ví dụ với eval_ratio = 0.1:**  
> Tổng số ô quan sát được = 15 (trong 6 × 3 = 18 ô, 3 ô bị thiếu gốc).  
> Số ô chọn làm eval = floor(0.1 × 15) = 1 ô.  
> Giả sử chọn ngẫu nhiên ra ô [t=0, Độ ẩm]:
>
> ```
> Trước:  X[0,1] = −1.12 (đã chuẩn hoá),  mask[0,1] = 1
> Sau:    X[0,1] = 0,                       mask[0,1] = 0,  eval_mask[0,1] = 1
> ```
>
> Model sẽ học mà không biết giá trị thật ở [t=0, Độ ẩm]. Cuối mỗi epoch, ta đo xem model dự đoán giá trị đó gần bao nhiêu so với −1.12.

---

## Bước 3 — Điền giá trị khuyết cục bộ trong window

Sau khi tách eval, ta có thêm các ô bị "ẩn" (mask = 0). Tất cả ô có mask = 0 (bao gồm cả ô thiếu gốc lẫn ô eval mới ẩn) được **điền bằng mean của các giá trị còn quan sát được trong window**.

Mục đích: cung cấp một giá trị tạm hợp lý để build đồ thị KNN (vì KNN cần vector đầy đủ, không có NaN).

> **Ví dụ:**  
> Sau data_transform, các ô mask = 0:  
> — [t=1, Ánh sáng], [t=2, Nhiệt độ], [t=3, Độ ẩm] (thiếu gốc)  
> — [t=0, Độ ẩm] (eval ẩn mới)
>
> Mean Độ ẩm từ các ô còn quan sát (mask=1): tính từ t=1,4,5 → mean ≈ 0.14 (sau chuẩn hoá)  
> → Điền X[0,1] = 0.14, X[3,1] = 0.14  
> Mean Ánh sáng (mask=1): tính từ t=2,4,5 → mean ≈ 0.03  
> → Điền X[1,2] = 0.03  
> Mean Nhiệt độ (mask=1): tính từ t=0,1,3,4,5 → mean ≈ −0.04  
> → Điền X[2,0] = −0.04

Ma trận X lúc này **không còn ô NaN hay ô 0 bất hợp lý** — mỗi ô đều có giá trị ước lượng sơ bộ. Gọi bản sao của X ở bước này là **X_knn** — chỉ dùng để build đồ thị, không dùng trực tiếp để train.

---

## Bước 4 — Xây dựng đồ thị bằng Pearson-filtered KNN

### 4.1 Tính ma trận Pearson giữa các stream (sample)

X_knn được chia thành các **stream** — mỗi stream là một đoạn `window_length` timestep liên tiếp. Mỗi stream được flatten thành một vector rồi tính **hệ số tương quan Pearson** giữa từng cặp stream:

```
pearson[i, j] = corr(flatten(stream_i), flatten(stream_j))
```

Kết quả là ma trận S × S (S = num_nodes // window_length):

> **Ví dụ với 6 timestep, window_length=2 → 3 streams:**
>
> ```
> stream_0 = [t=0, t=1]  → flatten → [−1.10, −1.12, −1.08, −0.80, −0.80, −0.03]
> stream_1 = [t=2, t=3]  → flatten → [−0.04, −0.22, −0.36, +0.19, +0.14, +0.32]
> stream_2 = [t=4, t=5]  → flatten → [+0.75, +0.87, +0.55, +1.00, +1.13, +0.60]
> ```
>
> ```
>            stream_0   stream_1   stream_2
> stream_0  [  1.00       0.96       0.94  ]
> stream_1  [  0.96       1.00       0.99  ]
> stream_2  [  0.94       0.99       1.00  ]
> ```
>
> Các stream trong cùng một chuỗi thời gian có xu hướng tương quan cao.

Đường chéo (tự tương quan) bị bỏ qua (= False).

### 4.2 Lọc stream theo ngưỡng delta

Áp dụng ngưỡng `delta`: một stream được phép nối cạnh với stream khác chỉ khi `|pearson[i, j]| >= delta`.

> **Ví dụ:**
>
> | delta | Cặp stream vượt ngưỡng | Kết quả |
> |-------|------------------------|---------|
> | 0.90  | (0,1)=0.96, (0,2)=0.94, (1,2)=0.99 — tất cả | Tất cả stream đều có thể nối nhau |
> | 0.95  | (0,1)=0.96, (1,2)=0.99 vượt; (0,2)=0.94 không | stream_0 chỉ nối được stream_1 |
> | 0.99  | Không cặp nào vượt | → Đồ thị rỗng (xem mục fallback) |

### 4.3 Xây dựng KNN graph trên các node của stream tương quan

Với mỗi stream `i`, chỉ tìm k láng giềng gần nhất của các node trong `stream_i` **từ các stream `j` mà `|pearson[i,j]| >= delta`**. Khoảng cách tính theo Euclidean giữa các vector feature của node.

> **Ví dụ (k=2, delta=0.90):**
>
> stream_1 (gồm t=2, t=3) có thể nối sang stream_0 và stream_2.
> Candidate nodes: {t=0, t=1} ∪ {t=4, t=5}.
>
> ```
> dist(t=2, t=0) = 1.55   dist(t=2, t=1) = 0.84 ← gần nhất
> dist(t=2, t=4) = 1.35   dist(t=2, t=5) = 1.70
> → t=2 nối với t=1 và t=4
>
> dist(t=3, t=0) = 1.73   dist(t=3, t=1) = 0.60 ← gần nhất
> dist(t=3, t=4) = 0.84   dist(t=3, t=5) = 1.09
> → t=3 nối với t=1 và t=4
> ```

Làm tương tự cho tất cả stream. Kết quả là **edge_index**: danh sách các cặp (node_src, node_dst).

> **Ý nghĩa:** Cạnh chỉ nối các node thuộc các đoạn thời gian có **pattern tương tự nhau** — không nối bừa với các stream có pattern khác biệt.

### 4.4 Trường hợp fallback — Không có cặp stream nào vượt ngưỡng delta

Xảy ra khi delta quá cao hoặc dữ liệu có tính tương quan thấp.

```
Kiểm tra edge_index.numel() == 0 → True
  → Fallback: build KNN thông thường trên toàn bộ feature
              (không lọc Pearson, dùng tất cả cột của X_knn)
              → Model vẫn có đồ thị, aggregate được thông tin láng giềng
```

Nếu quá trình build đồ thị gặp lỗi bất kỳ, fallback cũng được kích hoạt.

```
try:
    get_similarity_graph(...)   ← build Pearson-filtered graph
except Exception as exc:
    → fallback KNN toàn feature (an toàn)
```

---

## Bước 5 — Mô hình SAGE++DAMC

Kiến trúc gồm **2 lớp GNN song song** (gnn và gnn2) + 1 regressor MLP. Mỗi lớp GNN là `StaticGraphSAGEPlusDAMC`. Đồ thị được build một lần (static) và dùng lại cho tất cả các epoch.

### 5.1 Hop 1 — Dual Aggregation lần 1

Với mỗi node v, nhìn vào tập láng giềng N(v) (các node nối cạnh với v):

**Mean aggregation:**  
Lấy trung bình vector feature của tất cả láng giềng, kết hợp với feature của v qua phép chiếu tuyến tính.

**Max aggregation:**  
Lấy giá trị lớn nhất theo từng chiều trong số các láng giềng, kết hợp tương tự.

Hai kết quả được ghép lại thành embedding hop-1:

```
x1_mean[v] = ReLU( W_mean1 · MEAN({x_u : u ∈ N(v)} ∪ {x_v}) )
x1_max[v]  = ReLU( W_max1  ·  MAX({x_u : u ∈ N(v)} ∪ {x_v}) )
x1[v]      = concat(x1_mean[v], x1_max[v])
```

> **Ví dụ tại t=1** (láng giềng: t=2 và t=3, k=2):
>
> Feature ban đầu (sau khi điền, chuẩn hoá):
> ```
> x[t=1] = [−0.80, −0.80, −0.03]
> x[t=2] = [−0.04, −0.22, −0.36]
> x[t=3] = [+0.19, +0.14, +0.32]
> ```
>
> Mean láng giềng = [(−0.04+0.19)/2, (−0.22+0.14)/2, (−0.36+0.32)/2]
>                 = [+0.075, −0.04, −0.02]
>
> Max láng giềng  = [max(−0.04, 0.19), max(−0.22, 0.14), max(−0.36, 0.32)]
>                 = [+0.19, +0.14, +0.32]
>
> → x1_mean[t=1] = ReLU(W_mean1 · concat(x[t=1], mean_neighbor))  (projected to hidden_dim)
> → x1_max[t=1]  = ReLU(W_max1  · concat(x[t=1], max_neighbor))
> → x1[t=1]      = concat(x1_mean[t=1], x1_max[t=1])   ← vector 2×hidden_dim chiều

SAGEConv trong PyG thực chất concat self-feature và aggregated-neighbor-feature trước khi chiếu — nên **self-information luôn được giữ lại**, không bị "pha loãng" bởi láng giềng.

### 5.2 Hop 2 — Dual Aggregation lần 2

Làm lại aggregation, nhưng lần này đầu vào là **x1** (embedding hop-1), không phải x ban đầu. Dùng **cùng đồ thị cũ**:

```
x2_mean[v] = ReLU( W_mean2 · MEAN({x1_u : u ∈ N(v)} ∪ {x1_v}) )
x2_max[v]  = ReLU( W_max2  ·  MAX({x1_u : u ∈ N(v)} ∪ {x1_v}) )
x2[v]      = concat(x2_mean[v], x2_max[v])
```

Sau 2 hop, thông tin từ **láng giềng của láng giềng** (2-hop) đã được tổng hợp vào mỗi node. Với k=10, sau 2 hop mỗi node "nhìn thấy" thông tin từ tối đa 100 timestep xung quanh.

### 5.3 Multi-scale Concatenation và Projection

Ghép 3 mức thông tin: input gốc + hop-1 + hop-2:

```
x_multi[v] = concat( x_input[v], x1[v], x2[v] )
           → shape: (in_channels + 2×hidden_dim + 2×hidden_dim) = (3 + 4×hidden_dim)

x_out[v] = ReLU( W_proj · x_multi[v] )   → shape: (out_channels = 256)
```

> **Ý nghĩa của multi-scale:** Giữ thông tin ở nhiều "tầm nhìn" khác nhau:
> - x_input: thông tin tức thì tại timestep v.
> - x1: tổng hợp từ láng giềng trực tiếp (1 bước).
> - x2: tổng hợp từ láng giềng 2 bước (xa hơn, trừu tượng hơn).

### 5.4 Regressor — Ánh xạ embedding sang dự đoán

Embedding 256 chiều từ GNN được đưa vào **MLPNet** (2 layer fully-connected) để cho ra vector dự đoán cùng kích thước với input (3 chiều cho Labsensor):

```
pred[v] = MLP( x_out[v] )   → shape: (3,)
```

Mỗi phần tử trong pred[v] là giá trị bổ khuyết dự đoán cho feature tương ứng tại timestep v.

---

## Bước 6 — Vòng lặp huấn luyện (Training Loop)

Kiến trúc có **2 lớp GNN xếp chồng** (gnn + gnn2) xử lý tuần tự, tạo thành 2 lần "chỉnh sửa" dự đoán trước khi tính loss. Cả hai dùng chung một đồ thị, nhưng trọng số độc lập.

### Cấu trúc một epoch

**Vòng lặp imputation (2 lần):**

```
Lần 1 (gnn):
  X_emb1          = gnn(X_imputed, edge_index)   → embedding từ lớp GNN 1
  pred1            = regressor(X_emb1)            → dự đoán 1
  X_imputed        = X * mask + pred1 * (1−mask)  → giữ observed, điền missing bằng dự đoán
  loss            += MAE(X, pred1) trên các ô mask=1

Lần 2 (gnn2):
  X_emb2          = gnn2(X_imputed, edge_index)  → embedding từ lớp GNN 2 (đầu vào đã được cải thiện)
  pred2            = regressor(X_emb2)            → dự đoán 2
  X_imputed        = X * mask + pred2 * (1−mask)
  loss            += MAE(X, pred2) trên các ô mask=1
```

**Update:**
```
loss.backward()
optimizer.step()   (Adam, lr=0.01)
```

**Evaluation (không gradient):**

Sau mỗi epoch, tính MAE trên các ô `eval_mask=1` (ô ta đã giấu từ bước data_transform):

```
eval_MAE = mean( |X_imputed[eval_mask=1] - eval_X[eval_mask=1]| )
```

> **Ví dụ tại epoch 50:**
>
> ```
> Ô eval: [t=0, Độ ẩm], giá trị thật = −1.12
> Dự đoán của model (từ pred2) = −1.09
> MAE tại ô này = |−1.12 − (−1.09)| = 0.03
> ```
>
> Loss training giảm dần theo epoch vì model học được cách dùng thông tin từ láng giềng để ước lượng ô bị thiếu.

### Tracking kết quả

Sau mỗi epoch lưu lại MAE, MSE, MRE. Cuối cùng chọn **epoch có MAE nhỏ nhất** làm kết quả của window đó.

---

## Bước 7 — Lặp lại và tổng hợp (5 iteration)

Toàn bộ quy trình từ Bước 2 đến Bước 6 được **chạy lại 5 lần** (`num_of_iter=5`). Mỗi lần:
- `data_transform` chọn ngẫu nhiên bộ eval khác nhau (do seed ngẫu nhiên).
- Model huấn luyện từ đầu (không transfer).

Kết quả cuối cùng là **trung bình 5 lần** của (MAE, MSE, MRE, thời gian). Điều này giúp kết quả ổn định, không phụ thuộc vào một lần chọn eval ngẫu nhiên cụ thể.

---

## Trường hợp đồ thị rỗng — Model vẫn chạy được không?

**Có**, nhưng với năng lực yếu hơn. Khi `edge_index` rỗng:

```
SAGEConv(x, edge_index_rỗng):
  MEAN({}) = 0   →   x_out = W1 · x_v  (chỉ có self-feature)
  MAX({})  = 0   →   x_out = W1 · x_v
```

Toàn bộ mô hình trở thành **MLP nhiều lớp trên từng node độc lập**:

```
x1[v] = ReLU(W · x_input[v])      ← không dùng láng giềng nào
x2[v] = ReLU(W · x1[v])
x_multi[v] = concat(x_input[v], x1[v], x2[v])
x_out[v] = ReLU(W_proj · x_multi[v])
pred[v] = MLP(x_out[v])
```

> **Ý nghĩa:** Model vẫn bổ khuyết được nhờ **tương quan giữa các feature tại cùng timestep**. Ví dụ: tại t=2, Nhiệt độ bị thiếu nhưng Độ ẩm và Ánh sáng quan sát được. Model học rằng "Nhiệt độ thường xấp xỉ X khi Độ ẩm = Y và Ánh sáng = Z" — đây là cross-feature imputation.
>
> Cái mất đi: model không còn biết "hôm qua nhiệt độ là bao nhiêu" (temporal context từ láng giềng), nên chất lượng bổ khuyết thường kém hơn khi có đồ thị.

---

## Tóm tắt luồng chạy theo delta

```
delta=0.9  →  Một số cặp stream vượt ngưỡng nếu dữ liệu đủ tương quan
           →  Nếu không cặp nào vượt → edge_index rỗng → fallback KNN toàn feature

delta=0.5  →  Nhiều cặp stream vượt ngưỡng hơn
           →  Build KNN (k=10) chỉ nối node giữa các stream tương quan
           →  edge_index có cạnh → GNN aggregate được temporal context
           →  Ghi CSV kết quả

delta=0.3  →  Tương tự delta=0.5, thêm nhiều cặp stream được phép nối

delta=0.0  →  Tất cả cặp stream đều vượt ngưỡng
           →  Mỗi node có thể nối với bất kỳ node nào ở stream khác
           →  Tương đương KNN toàn tập không lọc Pearson
```

---

## Sơ đồ tổng quát một lần chạy

```
Dữ liệu thô (Labsensor)
        │
        ▼
[Bước 1] Nạp → Subsampling → Điền NaN global → Chuẩn hoá
        │
        ▼
[Bước 2] data_transform: ẩn eval_ratio ô quan sát → tạo eval set
        │
        ▼
[Bước 3] Điền NaN cục bộ trong window → X_knn
        │
        ▼
[Bước 4] Tính Pearson (F×F)
        │
        ├─ Có cặp stream vượt delta?
        │       │ Có → KNN(node của stream tương quan, k=10) → edge_index
        │       │ Không → edge_index rỗng
        │               │
        │               └─ fallback → KNN(X_knn toàn feature) → edge_index
        │
        ▼
[Bước 5-6] Training (200 epoch):
        │  ┌─ Epoch i ─────────────────────────────────┐
        │  │ GNN_1(X, edge_index) → pred → X_imputed   │
        │  │ GNN_2(X_imputed, edge_index) → pred       │
        │  │ Loss = MAE(pred, X) trên mask=1            │
        │  │ Backprop → Adam update                     │
        │  │ Eval MAE trên eval_mask                    │
        │  └────────────────────────────────────────────┘
        │  Chọn epoch có MAE nhỏ nhất
        │
        ▼
[Bước 7] Lặp 5 lần → Average MAE/MSE/MRE
        │
        ▼
Ghi CSV: exp_results/{prefix}_{dataset}_..._{delta}.csv
```
