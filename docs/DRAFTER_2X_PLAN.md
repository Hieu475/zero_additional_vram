# Hướng nâng cấp lên ~2x: bộ nháp học được cho giải mã suy đoán exact trên mô hình 4-bit

Ngày: 10/11/2026. Tài liệu làm việc, mô tả những gì đang làm, phương pháp, kết quả mong muốn, rủi ro và các bước tiếp theo. Mọi con số được ghi rõ là **đo thật** hay **mô phỏng**. Chỗ nào chưa biết ghi **CHƯA BIẾT**.

---

## 1. Bối cảnh và mục tiêu

Dự án hiện có bộ kernel 4-bit bất biến theo M (LM head, attention, RMSNorm, GEMM NF4) và vòng giải mã suy đoán exact bằng Prompt Lookup Decoding (PLD) trên CUDA graph. Kết quả đã có (Qwen2.5-3B NF4, RTX 4050 6 GB): tăng tốc 1.14–1.39x so với baseline M=1 bất biến, 0 lệch trên 1200 cặp prompt–mô hình.

Vấn đề: 1.2–1.4x là khiêm tốn cho ICML/MLSys. Mục tiêu của đợt này là tìm cách đạt **gần 2x**, và dừng sớm với mọi ý tưởng đo được là không hiệu quả.

## 2. Chẩn đoán: vì sao PLD chỉ cho ~1.3x

Tăng tốc của một vòng suy đoán exact xấp xỉ:

    tăng tốc = (số token trung bình mỗi bước) / (chi phí bước M hàng / chi phí bước M=1)

Chi phí bước đo thật trong CUDA graph (Qwen2.5-3B NF4, kernel bất biến, `results/kernels/wp5_block_cost.json`):

| M | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| ms/bước | 15.0 | 15.6 | 17.7 | 17.8 | 23.6 | 23.7 |

Hai hệ quả:
1. Chi phí có bậc thang: M=3 và M=4 giá như nhau. Khối 4 hàng (K=3 bản nháp) miễn phí so với khối 3 hàng.
2. Để đạt 2x ở M=4 (chi phí gấp ~1.19 lần M=1) cần ~2.4 token mỗi bước, tức khoảng 1.4 bản nháp được nhận mỗi bước. PLD chỉ cho 0.3–0.5 (khoảng 1.3–1.5 token mỗi bước).

Do đó điểm nghẽn là **chất lượng nguồn nháp**, không phải kernel. Phần kernel GEMM đã gần giới hạn (xem PLAN mục 17).

## 3. Những gì đã làm và kết quả

### 3.1. K=3 với đủ kích thước block (đo thật)

Thêm cờ `--all-m` vào [scripts/kernels/wp5_invariant_graph_pld.py](../scripts/wp5_invariant_graph_pld.py) để bắt graph cho mọi kích thước block 2..K+1. Chạy [scripts/kernels/run_kscan.sh](../scripts/run_kscan.sh): 100 prompt HumanEval, K=2 và K=3 xen kẽ 3 lượt, tất cả exact.

| Lượt | K=2 | K=3 |
|---|---|---|
| 1 | 1.304x | 1.356x |
| 2 | 1.276x | 1.341x |
| 3 | 1.279x | 1.378x |
| Trung bình | 1.286x | 1.358x |

K=3 hơn K=2 ở cả 3 lượt (+0.052, +0.065, +0.099); nhiễu giữa các lượt cùng cấu hình khoảng ±0.015. Đây là cải thiện miễn phí. Giới hạn: một mô hình, một tập; chưa chạy GSM8K/CNN-DM.

### 3.2. Mô phỏng ngoại tuyến (công cụ dừng sớm)

Vì số token nhận được mỗi khối chỉ phụ thuộc chuỗi tham chiếu và chiến lược nháp, có thể mô phỏng mà không chạy kernel:
- [scripts/drafter/dump_ref_tokens.py](../scripts/dump_ref_tokens.py): ghi chuỗi tham chiếu M=1 bất biến (`results/kernels/ref/`).
- [scripts/lib/sim_draft.py](../scripts/sim_draft.py): mô phỏng vòng giải mã, dùng bảng chi phí ở mục 2 (M≥7 là **ngoại suy, giả định**), có tham số chi phí bộ nháp mỗi bước.

Hiệu chuẩn (HumanEval): mô phỏng K=2 là 1.350x, K=3 là 1.422x; đo thật là 1.286x và 1.358x. Mô phỏng cao hơn ~5% (không tính chi phí host) nhưng chênh lệch K=3 so với K=2 khớp. Kết luận: tin được khi so sánh tương đối, phải trừ ~5% cho số tuyệt đối.

### 3.3. Họ n-gram bão hòa (mô phỏng, đã bỏ)

| Chiến lược | HumanEval | GSM8K |
|---|---|---|
| PLD K=2 | 1.35 | 1.21 |
| PLD K=3 | 1.42 | 1.25 |
| PLD K=5 | 1.36 | 1.17 |
| Cây 2–3 nhánh | 1.35–1.36 | 1.20 |
| Corpus từ đầu ra khác | 1.29 | 1.18 |
| PLD + corpus K=3 | 1.48 | 1.32 |

Không biến thể nào gần 2x. **Quyết định: bỏ hướng cây ứng viên và chính sách K thích ứng cho n-gram.**

### 3.4. Bộ nháp học được: kết quả đầu tiên

Thiết kế ([src/zassd/kernels/drafter.py](../src/zassd/kernels/drafter.py)), kiểu EAGLE-lite:
- Đầu vào: trạng thái ẩn cuối (sau norm) của vị trí được chấp nhận cuối cùng, và embedding của token pend (token mà mô hình chính vừa dự đoán).
- Mỗi bước nháp: `u = fc([s, emb(x)])`, `s' = u + MLP(rmsnorm(u))`, logits = `s' @ Wv^T`. Chạy tuần tự K bước, trạng thái tự truyền tiếp.
- **LM head thu gọn từ vựng** (16384 token thường gặp, khởi tạo từ embedding gốc): lý do là LM head đầy đủ của Qwen-3B (151936×2048 fp16, 622 MB) tốn ~3 ms mỗi lần đọc; EAGLE nguyên bản gọi nó một lần mỗi token nháp, quá đắt trên mô hình 4-bit này.
- Tính exact không phụ thuộc bộ nháp: bản nháp chỉ là đề xuất, kernel bất biến theo M xác minh, nên đầu ra vẫn bằng từng bit với giải mã tuần tự M=1.

Dữ liệu: tự chưng cất. Dùng chính mô hình đích sinh greedy trên prompt huấn luyện, ghi trạng thái ẩn ([scripts/drafter/drafter_gen_data.py](../scripts/drafter_gen_data.py)). Sinh theo lô bằng kernel gốc của bnb (số học không cần bit-exact khi chỉ để huấn luyện). Prompt huấn luyện lấy từ split train, tách biệt khỏi tập đánh giá.

Kết quả với bộ nháp d1 (K=3, chỉ 3000 chuỗi GSM8K, huấn luyện ~3 phút):

| | Đo trên | Kết quả |
|---|---|---|
| Độ chính xác top-1 từng vị trí (holdout) | 150 chuỗi giữ lại | 0.72 / 0.59 / 0.51 |
| Xác suất chuỗi nhận ≥1/2/3 token | holdout | 0.72 / 0.47 / 0.28 |
| **Tăng tốc dự đoán GSM8K test, chi phí bộ nháp 1 ms** | **mô phỏng** | **1.80x** (PLD: 1.25x) |
| Tăng tốc dự đoán HumanEval, chi phí 1 ms | mô phỏng | 0.96x (PLD: 1.42x), **không tổng quát** |

Hai điều rút ra: (a) hướng này là hướng duy nhất tiến gần 2x; (b) huấn luyện trên một miền không chuyển sang miền khác, nên cần dữ liệu đa miền. Còn giới hạn: 1.80x là **mô phỏng**, giả định chi phí bộ nháp 1 ms (chưa đo), và train loss giảm nhanh hơn holdout (overfit vì ít dữ liệu).

### 3.5. Dữ liệu đa miền đã sinh (`results/drafter/`)

| Tệp | Nguồn | Số chuỗi | Vị trí |
|---|---|---|---|
| gsm8k_train3000.npz | GSM8K train, 3000 prompt | 3000 | ~482k |
| gsm8k_train3000b.npz | GSM8K train, 3000 prompt khác | 3000 | ~480k |
| cnndm_train.npz | CNN/DM train, 2000 prompt, 128 token mới | 2000 | ~250k |
| code_train.npz | Bộ Python code instructions 18k, dựng stub hàm + docstring giống HumanEval | ~3000 | ~315k |

Giới hạn dữ liệu code: không phải bài toán HumanEval; định dạng được dựng lại bằng regex nên khác phân phối thật ở mức nào là **CHƯA BIẾT**.

### 3.6. Vòng giải mã tích hợp (đã viết, chưa chạy thử)

[scripts/drafter/wp9_drafter_graph.py](../scripts/wp9_drafter_graph.py): một bước nằm trọn trong một CUDA graph:
1. Xác minh khối `[pend, d1..dK]` bằng kernel bất biến.
2. Tính số bản nháp nhận được (khớp liên tiếp), chọn pend mới.
3. Ghi token ra bộ đệm đầu ra, cập nhật vị trí và độ dài KV (bằng tensor, không cần host).
4. Chạy bộ nháp từ hàng ẩn của vị trí được nhận cuối, sinh K bản nháp cho khối sau.

Host chỉ phát lệnh replay và kiểm tra số token đã sinh mỗi 4 bước. Baseline được xây tương tự (M=1, phản hồi token trong graph, không sync mỗi token) để so sánh công bằng, kèm baseline đồng bộ theo định nghĩa cũ. Kiểm tra exact: token và KV từng bit so với baseline, và so với tham chiếu eager M=1 trên mỗi prompt thứ `--ref-every`.

Điểm cần lưu ý: baseline async mạnh hơn baseline cũ (có sync mỗi token), nên tăng tốc báo cáo so với baseline async sẽ thấp hơn so với định nghĩa cũ. Phải báo cáo cả hai.

## 4. Trạng thái hiện tại và điểm nghẽn

Huấn luyện trên cả 4 tệp ([scripts/drafter/drafter_train.py](../scripts/drafter_train.py)) bị kill ba lần (exit 137). Nghi ngờ hết RAM host (khoảng 6–7 GB khả dụng, swap gần đầy, Brave chiếm nhiều). Sửa bằng memmap vẫn bị kill; **nguyên nhân thật CHƯA XÁC NHẬN**. Giả thuyết: `np.load(file.npz)["h"]` nạp trọn mảng 1–2 GB (npz không mmap được), và bước hợp nhất cùng giữ nhiều bản; cần đo RAM từng bước thay vì đoán.

## 5. Kế hoạch tiếp theo, theo thứ tự và có cổng dừng

### Bước A. Gỡ điểm nghẽn RAM (trước hết)
- Đóng Brave và ứng dụng nặng, cần ≥ 10 GB RAM trống.
- Chuyển từng `.npz` thành `h.npy` + `t.npy` + `off.npy` riêng (làm từng tệp, giải phóng sau mỗi tệp). Huấn luyện đọc bằng `np.load(mmap_mode="r")` cho từng tệp, không hợp nhất thành một mảng lớn.
- Chạy với `/usr/bin/time -v` hoặc theo dõi `free` để biết chính xác bước nào tràn.
- Phương án dự phòng: huấn luyện trên tập con (ví dụ 1.5 triệu vị trí) hoặc lưu đặc trưng fp16 trên đĩa và đọc theo khối.

### Bước B. Huấn luyện đa miền (K=5) và mô phỏng
- Chạy `drafter_train.py` với 4 tệp, K=5, 8 epoch. Mô hình K=5 dùng được cho K=3 hoặc 4 vì bước k chỉ phụ thuộc các bước trước.
- Chạy [scripts/drafter/drafter_eval_sim.py](../scripts/drafter_eval_sim.py) trên chuỗi tham chiếu GSM8K và HumanEval. CNN/DM cần tạo thêm tệp ref bằng `dump_ref_tokens.py`.
- **Cổng 1:** nếu mô phỏng (chi phí bộ nháp 1 ms) dưới 1.6x trên cả ba miền, bộ nháp này không đủ; chuyển sang phương án ở mục 6. Nếu GSM8K ≥ 1.8x và HumanEval ≥ PLD, đi tiếp.
- So sánh các cấu hình K (3, 4, 5) vì M=5 nhảy chi phí lên 23.6 ms; K=3 có thể vẫn tối ưu.
- Kết hợp với PLD (PLD-first, bộ nháp khi PLD không khớp) vì PLD rẻ và mạnh trên code; mô phỏng đã có sẵn dòng này.

### Bước C. Đo chi phí bộ nháp thật
- Giả định 1 ms trong mô phỏng là **ước tính**. Bộ nháp thực chạy K bước tuần tự, mỗi bước đọc ~32 MB trọng số (fc + MLP) cộng LM head thu gọn (16384×2048×2 = 67 MB); tổng ~300 MB cho K=3, tức ~1.5–2 ms ở băng thông ~190 GB/s. Có thể cao hơn 1 ms.
- Chạy mô phỏng với chi phí 1.0, 1.5, 2.0 ms để biết độ nhạy. **Cổng 2:** nếu ở 2 ms mô phỏng dưới 1.6x thì cần giảm chi phí (bước D).

### Bước D. Giảm chi phí bộ nháp nếu cần
- Thu nhỏ từ vựng (8192), giảm `ff` từ 4096 xuống 2048.
- Lượng tử hóa trọng số bộ nháp xuống int8 (bộ nháp không cần bit-exact).
- Chia sẻ lần đọc LM head thu gọn giữa các bước nếu có thể gộp (khó vì tuần tự).

### Bước E. Chạy vòng tích hợp trên kernel thật
- Chạy `wp9_drafter_graph.py` trên 5 prompt để gỡ lỗi (chưa từng chạy; rủi ro gồm cập nhật `cumulative_length` trong graph, `index_copy_` với tensor chỉ số, và mặt nạ nhân quả khi `cache_position` là tensor trong graph).
- **Cổng 3 (tính đúng):** token và KV bằng từng bit so với baseline, cộng khớp với tham chiếu eager trên các prompt được kiểm. Nếu không exact, dừng và sửa trước khi nhìn tốc độ. Bài học từ bug bias bf16/fp16 trước đây: "exact" phải đi kèm kiểm tra văn bản sinh ra.
- Sau đó chạy 100 prompt mỗi tập, 3 lượt xen kẽ với PLD K=3 để có khoảng tin cậy.
- **Cổng 4 (tốc độ):** so số đo thật với dự đoán mô phỏng; chênh quá ~10% nghĩa là mô hình chi phí thiếu thành phần.

### Bước F. Ổn định kết quả để viết bài
- Huấn luyện lại với nhiều seed hoặc nhiều dữ liệu hơn nếu overfit (mục tiêu 20–50 nghìn chuỗi như tài liệu EAGLE là ngoài tầm GPU này; ghi rõ giới hạn).
- Chạy thêm mô hình thứ hai (ví dụ Qwen2.5-1.5B hoặc Llama-3.2-3B; bộ nháp phải huấn luyện riêng cho mỗi mô hình, chi phí khoảng 1–2 giờ dữ liệu + vài phút huấn luyện).
- Cập nhật README, PLAN và đề cương bài báo.

## 6. Kết quả mong muốn và phương án nếu thất bại

| Kịch bản | Điều kiện | Hệ quả |
|---|---|---|
| Tốt | Đo thật ≥ 1.8x trên GSM8K và ≥ PLD trên HumanEval, exact | Bài có kết quả chính mới; câu chuyện "bộ nháp học được + kernel bất biến theo M" |
| Vừa | 1.5–1.8x trên GSM8K, không thắng PLD trên code | Dùng PLD-first + bộ nháp; bài vẫn mạnh hơn hiện tại, nhưng chưa đạt 2x |
| Xấu | Dưới 1.5x hoặc chi phí bộ nháp vượt dự kiến | Quay về PLD K=3; hướng bài chuyển sang phân tích (định nghĩa exact, mô hình chi phí) |

Mô hình đích 3B ở 4-bit đã rất rẻ, nên trần lý thuyết bị giới hạn bởi chi phí bộ nháp; trên mô hình lớn hơn (7B+) lợi ích có thể cao hơn nhưng chưa đo (VRAM không cho phép).

## 7. Rủi ro và giới hạn cần nói thẳng

1. **Mô phỏng chưa phải số đo.** 1.80x là dự đoán với giả định chi phí bộ nháp 1 ms; số thật có thể thấp hơn.
2. **Tổng quát hóa miền.** Đã thấy thất bại khi chỉ huấn luyện GSM8K. Cần xác nhận lại trên bộ nháp đa miền; code là rủi ro cao nhất vì dữ liệu huấn luyện không phải HumanEval.
3. **Baseline.** Phải báo cáo so với baseline M=1 async mạnh (không sync mỗi token), không chỉ baseline cũ.
4. **Một GPU, mô hình nhỏ.** Reviewer MLSys vẫn sẽ hỏi về GPU khác và mô hình lớn hơn. Không thay thế được bằng mô phỏng.
5. **Chi phí huấn luyện bộ nháp** (dữ liệu tự chưng cất cho từng mô hình) phải được tính vào câu chuyện "tăng tốc miễn phí": đây là chi phí một lần, cần báo cáo.
6. **Bất biến vẫn kiểm chứng thực nghiệm**, chưa chứng minh hình thức; phụ thuộc phiên bản Triton/nvcc/bnb.
7. **Tính mới.** Bộ nháp kiểu EAGLE/Medusa đã có; phần mới nằm ở việc kết hợp với xác minh exact từng bit trên mô hình 4-bit và xử lý chi phí LM head. Cần đối chiếu công trình đồng thời trước khi tuyên bố.

## 8. Danh sách tệp liên quan

| Tệp | Vai trò | Trạng thái |
|---|---|---|
| scripts/kernels/wp5_invariant_graph_pld.py | Vòng PLD exact (đã thêm `--all-m`) | Đã chạy |
| scripts/kernels/run_kscan.sh | Quét K=2/3 xen kẽ | Đã chạy |
| scripts/drafter/dump_ref_tokens.py | Ghi chuỗi tham chiếu | Đã chạy (GSM8K, HumanEval) |
| scripts/lib/sim_draft.py | Mô phỏng chiến lược nháp | Đã chạy |
| scripts/drafter/drafter_gen_data.py | Sinh dữ liệu tự chưng cất | Đã chạy (4 tệp) |
| src/zassd/kernels/drafter.py | Kiến trúc bộ nháp | Dùng được |
| scripts/drafter/drafter_train.py | Huấn luyện (đã đổi sang nạp theo khối) | Đã chạy, d2_k5.pt |
| scripts/drafter/drafter_eval_sim.py | Dự đoán tăng tốc của bộ nháp (thêm quét chi phí 0/1/1.5/2 ms) | Đã chạy với d2_k5 |
| scripts/drafter/wp9_drafter_graph.py | Vòng tích hợp trong CUDA graph (thêm `--k`) | Đã chạy, 100 prompt × 2 tập |
| results/drafter/d1.pt | Bộ nháp K=3, chỉ GSM8K | Đã xóa 11/11/2026 (đã bị d2_k5.pt thay thế; kết quả ở mục 3.4 giữ nguyên) |
| results/drafter/d2_k5.pt | Bộ nháp đa miền, K=5 huấn luyện (dùng K=3) | Có |
| results/drafter_runs/*.json | Kết quả đo thật wp9 | Có |

Chưa commit gì. Thư mục `results/drafter/` (vài GB) không nên đưa vào git.

---

## 9. Kết quả thực hiện (10/11/2026)

### Bước A: RAM (xong)
File `_merged_h.npy` đã đầy đủ (1 537 340 hàng, không NaN). `drafter_train.py` nay nạp theo khối ~150k vị trí gồm các chuỗi nguyên vẹn đã xáo trộn, đọc tuần tự từ memmap, không gather ngẫu nhiên trên 6.3 GB; lưu checkpoint mỗi epoch. Chạy trọn không bị kill (peakRSS 8.7 GB, phần lớn là trang file memmap có thể thu hồi). **Nguyên nhân các lần kill trước vẫn CHƯA XÁC NHẬN** (nhật ký chỉ có từ lần khởi động hôm nay; `systemd-oomd` đang bật nên áp lực bộ nhớ là nghi can chính).

### Bước B: huấn luyện đa miền và mô phỏng (xong, Cổng 1 đạt)
d2_k5: 4 tệp, K=5, 8 epoch, ~22 phút. Holdout hỗn hợp: P(nhận ≥1/2/3) = 0.70/0.47/0.32. Loss train 0.94 trong khi holdout tăng chậm: **overfit**.

| Mô phỏng | GSM8K | HumanEval |
|---|---|---|
| Bộ nháp K=3, chi phí 1 ms | 1.92x | 1.53x |
| Bộ nháp K=3, chi phí 2 ms | 1.83x | 1.46x |
| Bộ nháp K=3, chi phí 0 ms (trần) | 2.03x | 1.62x |
| PLD K=3 | 1.25x | 1.42x |

K=3 tối ưu; K=4,5 tụt mạnh vì M≥5 rơi vào bậc chi phí 23.6 ms. Dữ liệu đa miền đã sửa lỗi tổng quát hóa sang HumanEval (d1: 0.96x). PLD-first + bộ nháp K=5 ra 1.77x / 1.67x (1 ms), thấp hơn bộ nháp K=3 thuần nên không dùng.

### Bước C: chi phí bộ nháp thật (xong)
Chuỗi bộ nháp K=3 trong một CUDA graph: **2.0 ms** (bf16; fp16 2.2 ms), bị giới hạn băng thông (~350 MB trọng số). Giả định 1 ms trong kế hoạch **sai**; dùng 2 ms. Cổng 2 đạt (≥1.6x ở 2 ms trên GSM8K: 1.83x).

### Bước E: vòng tích hợp (xong, đo thật)
Sửa hai lỗi trước khi chạy được: lệch dtype bf16/fp16 khi gọi bộ nháp trong graph, và `cumulative_length` là tensor 0-chiều. Cổng 3 đạt: token và KV từng bit so với baseline async trên 200/200 prompt, token bằng tham chiếu eager M=1 trên 20/20 prompt được kiểm.

| Đo thật (100 prompt, 1 lượt, K=3) | Tăng tốc vs baseline async M=1 | Token/bước | Mô phỏng @2 ms |
|---|---|---|---|
| GSM8K | **1.713x** | 2.33 | 1.83x |
| HumanEval | **1.371x** | 1.86 | 1.46x |

Cổng 4: số đo thấp hơn mô phỏng ~6% ở cả hai tập, khớp mức hiệu chuẩn ~5% đã biết, khi dùng đúng chi phí bộ nháp 2 ms. Baseline async và sync cho cùng kết quả (63.8 so với 63.8 token/s) nên không có chênh lệch giữa hai định nghĩa baseline.

### Kết luận
Kịch bản rơi vào **"Vừa"**: 1.71x trên GSM8K (PLD mô phỏng 1.25x, PLD đo thật trên HumanEval 1.36x so với bộ nháp 1.37x tức hòa trên code). **Chưa đạt 2x và, theo mô hình đã hiệu chuẩn, không thể đạt 2x chỉ bằng cách giảm chi phí bộ nháp**: ngay cả ở 0 ms, trần mô phỏng GSM8K là 2.03x, tức ~1.9x đo thật. Muốn tiến xa hơn phải tăng độ chấp nhận (thêm dữ liệu, chống overfit) hoặc đổi mô hình/GPU.

### Chưa làm / giới hạn
- Mới 1 lượt cho mỗi tập; chưa chạy xen kẽ nhiều lượt với PLD K=3 trong cùng harness để có khoảng tin cậy (nhiễu giữa các lượt của PLD ≈ ±0.015). So sánh với PLD trên GSM8K hiện là số **mô phỏng**.
- Chưa chạy CNN/DM thật (chỉ có dữ liệu huấn luyện), chưa có mô hình thứ hai (Bước F).
- Bước D (giảm chi phí bộ nháp) chưa làm: mỗi 1 ms bớt đi chỉ đáng ~+0.09x trên GSM8K.
- Chưa commit gì.

---

## 10. Số liệu có khoảng tin cậy và bản lai PLD + bộ nháp (10/11/2026)

### Bước 1: lượt xen kẽ so với PLD K=3 cùng harness (đo thật, mọi lượt exact)
Tăng tốc so với baseline M=1 (graph), 100 prompt mỗi lượt; PLD chạy bằng `wp5_invariant_graph_pld.py --all-m --k 3`, bộ nháp bằng `wp9_drafter_graph.py --k 3`.

| Tập | PLD K=3 | Bộ nháp K=3 | Số lượt (PLD / bộ nháp) |
|---|---|---|---|
| GSM8K | 1.217x (1.212, 1.221, 1.218) | **1.696x** (1.713, 1.669, 1.706) | 3 / 3 |
| HumanEval | **1.389x** (1.388, 1.387, 1.391) | 1.370x (1.371, 1.371, 1.368) | 3 / 3 |
| CNN/DM | **1.250x** (1.251, 1.249) | 1.108x (1.108, 1.108) | 2 / 2 |

Bộ nháp học được chỉ thắng trên toán; hòa trên code, thua trên tóm tắt (PLD tận dụng việc chép từ bài gốc). Lượt 3 của CNN/DM bị bỏ vì lỗi thao tác (kill nhầm hàng đợi), nên chỉ có 2 lượt.

### Bản lai: tra cứu n-gram trong CUDA graph + bộ nháp (`scripts/drafter/wp10_hybrid_graph.py`)
Khi có khớp n-gram (độ dài 3 rồi 2, lần xuất hiện gần nhất, tìm trên prompt + đã sinh) thì PLD điền m≤K khe nháp đầu; bộ nháp điền phần còn lại, hoặc toàn bộ khi không khớp. Khối luôn có M=K+1=4 nên không có chi phí kích thước khối biến đổi. Tính exact không bị ảnh hưởng vì bản nháp chỉ là đề xuất.

| Tập | PLD | Bộ nháp | **Bản lai (n≥2)** | Bản lai (chỉ n=3) |
|---|---|---|---|---|
| GSM8K | 1.217x | 1.696x | 1.674x (1 lượt) | chưa đo |
| HumanEval | 1.389x | 1.370x | **1.613x** (1 lượt) | 1.603x |
| CNN/DM | 1.250x | 1.108x | **1.267x** (1 lượt) | 1.233x |

Tất cả exact (token và KV từng bit so với baseline, và khớp eager trên prompt được kiểm). Khớp bigram (n≥2) tốt hơn chỉ trigram ở cả hai tập đã so. PLD kích hoạt ở 37% bước trên GSM8K, 45% trên HumanEval, 34% trên CNN/DM.

Đọc kết quả: bản lai là cấu hình duy nhất không thua cấu hình nào trên cả ba tập (trong khoảng nhiễu): hòa bộ nháp trên GSM8K (kém 0.02, bằng nhiễu giữa các lượt ~0.02), **hơn PLD 16% trên code**, và nhỉnh hơn PLD 1.4% trên tóm tắt (chênh nhỏ, 1 lượt, chưa chắc). **Vẫn chưa đạt 2x.**

### Giới hạn
- Bản lai chỉ 1 lượt mỗi tập; chênh trên CNN/DM nằm gần mức nhiễu. Cần thêm lượt xen kẽ trước khi đưa vào bài.
- Chưa thử các biến thể khác: khe PLD một phần, ngưỡng độ dài tối thiểu, chọn PLD theo độ tin cậy.
- Các lượt ở mục này dùng cùng checkpoint d2_k5 (huấn luyện trên GSM8K train, CNN/DM train, code instructions).

---

## 11. Mô hình thứ hai: Qwen2.5-1.5B (10/11/2026)

Bộ nháp huấn luyện riêng cho 1.5B (d=1536, ff=4096, từ vựng thu gọn 16384): dữ liệu tự chưng cất cùng công thức với 3B (GSM8K ×2, CNN/DM, code; 1.57 triệu vị trí), K=5, 8 epoch, ~17 phút. Holdout hỗn hợp: P(nhận ≥1/2/3) = 0.66/0.43/0.29 (3B: 0.70/0.47/0.32). Không có mô phỏng cho 1.5B (bảng chi phí chỉ có cho 3B); đo thẳng trên kernel thật. Tất cả exact (token và KV từng bit so với baseline, khớp eager trên prompt được kiểm). Baseline M=1: ~101 token/s.

Tăng tốc đo thật, 100 prompt, **1 lượt mỗi ô**, K=3:

| Tập | PLD K=3 | Bộ nháp | Bản lai (n≥2) |
|---|---|---|---|
| GSM8K | 1.229x | **1.591x** | 1.578x |
| HumanEval | 1.529x | 1.348x | **1.715x** |
| CNN/DM | **1.551x** | 1.075x | 1.480x |

Tổng hợp hai mô hình (3B / 1.5B):
- Bộ nháp học được thắng PLD trên toán ở cả hai mô hình (1.70 so với 1.22; 1.59 so với 1.23), thua rõ trên tóm tắt (1.11 so với 1.25; 1.08 so với 1.55).
- Bản lai thắng cả hai trên code ở cả hai mô hình (1.61 so với 1.39 / 1.72 so với 1.53).
- Bản lai **không** thắng đều: trên CNN/DM 1.5B nó kém PLD thuần 0.07x (1.48 so với 1.55). Nguyên nhân khả dĩ (chưa kiểm chứng): PLD thuần dùng kích thước khối biến đổi (M=2..4 tùy số bản nháp tìm được), còn bản lai luôn trả chi phí khối M=4 kể cả khi chỉ có 1 token từ PLD. Chọn kích thước khối theo từng bước cần host biết số bản nháp trước, mâu thuẫn với thiết kế không đồng bộ trong graph.
- Chưa mô hình nào đạt 2x. Mức cao nhất đo được: 1.70x (3B, GSM8K, bộ nháp), 1.72x (1.5B, HumanEval, bản lai).

Giới hạn: 1 lượt mỗi ô ở 1.5B (nhiễu giữa các lượt ở 3B ≈ 0.002–0.02, chưa đo riêng cho 1.5B); chỉ hai mô hình cùng họ Qwen, cùng GPU; chưa thử Mistral-7B (bộ nháp d=4096 sẽ đắt hơn đáng kể, chưa rõ có lợi).

---

## 12. Khối biến đổi cho bản lai: kết quả âm (mô phỏng, 11/11/2026)

Giả thuyết: bản lai thua PLD trên CNN/DM 1.5B vì luôn trả chi phí khối M=4. Cách sửa là chọn M∈{1,2,4} từng bước theo độ tin cậy của bản nháp (softmax top-1 của bộ nháp; token do PLD đưa vào coi là tin cậy). Bảng chi phí chỉ có M=1,2 rẻ hơn M=4 (15.0, 15.6 so với 17.8 ms; M=3 bằng M=4), và mỗi bước phải đọc một giá trị từ host để chọn graph.

Mô phỏng trên CPU (`scripts/drafter/sim_adaptive_block.py`, 3B, chi phí bộ nháp 2 ms mỗi bước, số học bộ nháp fp32):

| Tập | Luôn M=4 | Trần oracle (biết trước, không tốn đồng bộ) | Cổng độ tin cậy tốt nhất, đồng bộ 0 ms | Tốt nhất, đồng bộ 0.3 ms |
|---|---|---|---|---|
| GSM8K | 1.796x | 1.885x | 1.803x (τ=0.4) | 1.776x |
| HumanEval | 1.715x | 1.821x | 1.727x (τ=0.3) | 1.701x |

Kết luận: ngay cả oracle chỉ hơn +5–6%; chính sách thực tế hơn +0.3–0.7% nếu đồng bộ miễn phí và **thua** luôn-M=4 (−1.0 đến −1.5%) với 0.3 ms đồng bộ mỗi bước (0.3 ms là giả định, chưa đo). **Không triển khai.** Theo quy tắc dừng sớm của kế hoạch, hướng này dừng ở đây.

Giới hạn: chỉ mô phỏng, chỉ 3B trên GSM8K và HumanEval. CNN/DM (nơi bản lai thua PLD) chưa kiểm tra vì chưa có chuỗi tham chiếu 3B cho tập đó; ở đó tỉ lệ bản nháp bị từ chối cao hơn nên trần oracle có thể lớn hơn, nhưng cổng độ tin cậy không cho thấy khả năng dự đoán đủ tốt trên hai tập đã thử. Nguyên nhân thật của khoảng cách với PLD trên CNN/DM 1.5B vẫn **chưa kiểm chứng**.

