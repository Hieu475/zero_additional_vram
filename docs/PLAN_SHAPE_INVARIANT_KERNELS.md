# Kế hoạch nghiên cứu duy nhất: kernel 4-bit bất biến theo hình dạng cho giải mã suy đoán exact

Ngày: 05/11/2026. Đây là hướng duy nhất để theo đến khi nộp bài. Mọi việc khác (xem mục 11) chỉ được làm nếu phục vụ trực tiếp hướng này. Mọi con số "đã đo" trỏ về `C_SHAPE_AUDIT_SUMMARY.md`, `RESEARCH_DIRECTIONS_A_C.md` và thư mục `experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/`. Chỗ nào chưa có bằng chứng được ghi **CHƯA BIẾT**.

---

## 1. Luận điểm và đóng góp dự kiến

**Vấn đề.** Giải mã suy đoán (speculative decoding) về lý thuyết không đổi kết quả. Trên mô hình 4-bit thực tế, khối K+1 token chạy qua kernel khác với từng token một, kết quả đổi từng bit ở một số toán tử, và chuỗi sinh ra lệch khỏi greedy tuần tự (đã đo: 22–82% chuỗi lệch với PLD ngây thơ; 34–66% với mô hình nháp thật).

**Cách sửa hiện có và giới hạn.** Chạy từng hàng những toán tử nhạy (đã đo: đưa lệch về 0/50–0/60 ở mọi ô). Nhưng mỗi hàng đọc lại toàn bộ trọng số của toán tử đó. Giải mã suy đoán có lợi chính vì đọc trọng số một lần cho nhiều token, nên chạy từng hàng làm mất đúng lợi ích đó. Kết quả hướng A cho thấy suy đoán exact trên nền CUDA graph chỉ thêm 1.009x.

**Đề xuất.** Bộ kernel 4-bit **bất biến theo số hàng M**: kết quả của hàng i bằng từng bit dù hàng đó được tính cùng 1, 2, ... hay 8 hàng khác, nhưng trọng số chỉ được đọc một lần cho cả khối. Dùng chung cho baseline (M=1) và bộ xác minh, nên "exact" đúng theo thiết kế, không chỉ theo quan sát.

**Đóng góp (theo thứ tự quan trọng):**
1. Bộ kernel (GEMM NF4, LM head, attention, RMSNorm) bất biến theo M, với chứng minh theo cấu trúc (thứ tự cộng gộp cố định, không phụ thuộc M) cộng kiểm tra từng bit thực nghiệm.
2. Khung giải mã suy đoán exact cho mô hình 4-bit dùng các kernel đó, với số đo tốc độ đầu-cuối so với (a) chạy từng hàng và (b) baseline cùng kernel.
3. Phân tích chi phí: bất biến theo M tốn bao nhiêu so với bnb/cuBLAS ở M=1, và lợi bao nhiêu ở M>1.
4. Kiểm toán có sẵn (`shape_audit.py`) trở thành công cụ kiểm chứng, không còn là đóng góp chính.

**Không tuyên bố:** "phát hiện hiện tượng mới" (đã có ở batch-invariant ops, LLM-42), "kernel 4-bit nhanh nhất", "exact cho mọi backend/GPU".

## 2. Định nghĩa "exact" (phải thống nhất trước khi viết kernel)

- **Tham chiếu mới** = greedy tuần tự M=1 chạy bằng **chính bộ kernel bất biến này**. Không cố tái tạo thứ tự cộng của cuBLAS/bnb (đóng nguồn một phần, đổi theo phiên bản).
- Hệ quả cần nói thẳng trong bài: tham chiếu này **không** bằng từng bit với đường HF+bnb mặc định. Phải đo và báo cáo độ khác biệt token giữa hai tham chiếu (kỳ vọng rất nhỏ nhưng **CHƯA BIẾT**), vì người đọc sẽ hỏi.
- Điều kiện exact: với mọi M trong [1, M_max], đầu ra hàng i của kernel(M) bằng đầu ra của kernel(1) trên hàng i, từng bit. Kiểm tra bằng thử nghiệm ngẫu nhiên lớn trên từng kernel và bằng quỹ đạo đầu-cuối.

## 3. Thiết kế kỹ thuật

Nguyên tắc chung: với mỗi phần tử đầu ra, thứ tự cộng gộp (cây rút gọn, kích thước khối, thứ tự split-K) là hằng số cố định, **không phụ thuộc M**. Mỗi chương trình (program/warp) tính một tập hàng, nhưng phép rút gọn của từng hàng chỉ phụ thuộc chiều K và cấu hình cố định. Không dùng split-K động, không dùng atomics, không để auto-tune chọn cấu hình khác nhau theo M.

| Toán tử | Phương án | Ghi chú rủi ro |
|---|---|---|
| LM head (fp16, vocab lớn) | Kernel Triton đọc trọng số một lần, tính M hàng, rút gọn fp32 cây cố định | Toán tử đắt nhất khi chạy từng hàng (Qwen 3B ≈ 0.6 GB đọc M lần). Làm đầu tiên |
| Attention | Split-KV với kích thước chia cố định, hợp nhất kiểu log-sum-exp theo thứ tự cố định | Phải đúng cho GQA (K/V gốc, không nhân bản; xem lỗi GQA đã gặp) và mặt nạ nhân quả theo hàng |
| RMSNorm | Một chương trình mỗi hàng, rút gọn cố định | Rẻ; hiện chỉ lệch hiếm |
| GEMM NF4 | Kernel Triton giải lượng tử NF4 (blocksize, absmax của bnb) và nhân, thứ tự K cố định | Chỉ cần cho M≥4 (bnb 0.50.0 đã exact đến M=3 trên projection). Cần đọc đúng quant state của bnb. Thư mục `three_row_kernel` có thể đã có phần nền; **cần kiểm tra trước khi viết lại** |

Việc gắn vào mô hình: thay các module bằng wrapper (cùng cơ chế `rowwise` đã dùng trong `audit_pld_trajectory_divergence.py`), nên khung D/PLD hiện có dùng lại được.

## 4. Các gói công việc, thứ tự và tiêu chí qua cổng

Mỗi gói có tiêu chí **đặt trước**. Không qua cổng thì báo kết quả âm, không đổi tiêu chí sau.

**WP1 — Kernel LM head bất biến (ước tính vài ngày).**
- Làm: kernel Triton; thử bằng nhau từng bit giữa M=1..8 trên đầu vào thật (lấy từ hook) và ngẫu nhiên (≥10^6 hàng).
- Đo: thời gian ở M=1 và M=3,5,8 so với cuBLAS và so với chạy M lần.
- Qua cổng nếu: bằng nhau từng bit ở mọi M; thời gian ở M=4 nhỏ hơn 4 lần thời gian M=1 của chính nó (tức thực sự dùng chung đọc trọng số), và chậm hơn cuBLAS M=1 không quá 1.5x.
- Nếu rớt: báo âm, dừng; hướng này không khả thi trên phần cứng này (xem mục 8).

**WP2 — Attention bất biến.** Qua cổng nếu bằng nhau từng bit M=1..8 trên các độ dài ngữ cảnh 128–3000, GQA và MHA; không chậm hơn SDPA mặc định quá 2x ở ngữ cảnh ≤1000.

**WP3 — RMSNorm và GEMM NF4 bất biến.** GEMM NF4 chỉ làm nếu muốn K≥3. Qua cổng nếu bằng nhau từng bit và chi phí ở M=1 không quá 1.3x so với bnb.

**WP4 — Tích hợp và kiểm tra quỹ đạo.** Giải mã suy đoán (PLD và mô hình nháp thật) dùng toàn bộ kernel. Qua cổng nếu 0 chuỗi lệch trên ≥500 prompt (cận trên Wilson 95% ≈ 0.8%) ở mọi ô chính.

**WP5 — Tốc độ đầu-cuối (kết quả quyết định).** Đo token/giây với cùng quy trình, cùng prompt, cùng nhiệt độ GPU, lặp ≥5 lần, báo trung vị và khoảng. So sánh: (a) baseline cùng kernel M=1, (b) suy đoán exact cùng kernel, (c) suy đoán chạy từng hàng, (d) suy đoán ngây thơ (không exact, làm trần).
- **Mục tiêu thành công tối thiểu:** (b) nhanh hơn (a) ít nhất 1.15x trên một cấu hình có mô hình nháp thật, và (b) nhanh hơn (c) ít nhất 1.3x.
- **Kết quả âm hợp lệ:** nếu (b) không vượt (a) ít nhất 1.05x, bài vẫn có thể là bài âm có giá trị, nhưng không nhắm ICML/MLSys; chuyển sang TMLR/workshop.

**WP6 — Tổng quát hóa (làm khi WP1–WP5 qua cổng).** Mô hình thứ hai và thứ ba (OLMo-2, Phi-3.5, Qwen3-4B); torchao INT4 nếu kernel áp dụng được; GPU thứ hai và mô hình 7B (bạn đã hoãn; **bắt buộc** nếu nhắm ICML/MLSys). Kiểm tra bnb 0.50.1/0.50.2.

**WP7 — Viết bài.** Dùng `PAPER_DRAFT_DIRECTION_C.md` làm nền cho phần kiểm toán/động cơ; đổi trọng tâm sang kernel và tốc độ.

## 5. Thực nghiệm sẽ đưa vào bài

1. Bảng bất biến: kernel × M × độ dài ngữ cảnh × mô hình, tỷ lệ phần tử khác bit (kỳ vọng 0 tuyệt đối).
2. Bảng chi phí kernel: thời gian ở M=1,2,4,8 so với bnb/cuBLAS/SDPA.
3. Bảng quỹ đạo: ngây thơ / từng hàng / kernel bất biến, ≥500 prompt, 3 tập dữ liệu, kèm khoảng Wilson.
4. Bảng tốc độ đầu-cuối (WP5) kèm phân rã thời gian: LM head, attention, projection, phần còn lại.
5. Khác biệt token giữa tham chiếu bất biến và HF+bnb mặc định (mục 2).
6. Phụ lục: kết quả kiểm toán hiện có, các kết quả âm (A, B, HQQ), tính không xác định của HQQ+GemLite.

## 6. Kết quả mong đợi (nói rõ độ chắc chắn)

- **Khá chắc:** kernel bất biến có thể đạt bằng nhau từng bit theo cấu trúc; quỹ đạo 0 lệch.
- **Không chắc (CHƯA BIẾT):** chi phí ở M=1 so với cuBLAS/bnb; mức tăng tốc đầu-cuối. Hướng A cho thấy phần tăng tốc exact còn rất mỏng, và mô hình ≤3B trên RTX 4050 6 GB bị giới hạn bởi overhead khởi chạy kernel, không chỉ băng thông. Đây là rủi ro lớn nhất.
- **Thực tế hóa kỳ vọng:** kết quả thành công hợp lý là 1.15–1.5x so với baseline exact cùng kernel, chưa phải mức tăng tốc cao của suy đoán không exact.

## 7. Tiêu chí dừng và quyết định (đặt trước)

- Sau WP1: nếu không qua cổng, dừng hướng này và quay lại bài kiểm toán thuần (TMLR/workshop), không tìm hướng mới tùy hứng.
- Sau WP5: nếu thành công tối thiểu đạt, mở rộng WP6 và nhắm hội nghị. Nếu không, viết bài âm/phương pháp cho TMLR.
- Không đổi tiêu chí sau khi thấy dữ liệu.

## 8. Rủi ro và cách giảm

| Rủi ro | Giảm |
|---|---|
| Kernel Triton chậm hơn cuBLAS nhiều | Đo ở WP1 trước khi làm các phần khác; cổng 1.5x |
| Overhead khởi chạy làm mất lợi ích | Kết hợp CUDA graph (đã có a2/a3); đo phân rã |
| Chỉ 6 GB VRAM, mô hình ≤3B | WP6; GPU/mô hình lớn là điều kiện cho ICML/MLSys |
| Tham chiếu mới khác HF+bnb | Báo cáo khác biệt token, nêu rõ định nghĩa |
| Bằng nhau từng bit chỉ trên GPU/phiên bản Triton này | Ghi phiên bản; thử ít nhất hai phiên bản Triton nếu có thể; không tuyên bố vượt GPU |
| Tính mới bị đồng thời công bố/đã có | Tra tài liệu lại ngay trước khi nộp (LLM-42 và batch-invariant ops là gần nhất; tôi mới đọc một phần) |

## 9. Nơi nộp (cần xác nhận từ trang chính thức, không dựa trí nhớ)

- Mục tiêu cao: ICML 2027 hoặc MLSys 2028 (hạn ICML 2027 mới có ước tính bên thứ ba; MLSys 2027 đã quá hạn).
- Nền an toàn: TMLR (nộp liên tục) hoặc workshop về suy luận hiệu quả.
- Quyết định chọn sau WP5, theo kết quả tốc độ và WP6.

## 10. Cách dùng kết quả đã có

- Kiểm toán từng toán tử: dùng làm động cơ và công cụ kiểm chứng.
- Kết quả D (mô hình nháp thật lệch 34–66%, từng hàng về 0): dùng làm bằng chứng vấn đề trên hệ thống thật và đường chuẩn "từng hàng".
- Kết quả A (1.009x): giữ làm phụ lục âm; chính nó là lý do hướng này cần kernel dùng chung đọc trọng số.
- Kết quả B (mô hình dự đoán không đạt R² ≥ 0.8): để ở phụ lục như kết quả âm, không phát triển thêm.

## 11. Ngoài phạm vi (không làm nếu không phục vụ hướng này)

Xác minh có chứng nhận theo margin; kernel NF4 nhanh hơn bnb nói chung; vLLM (hướng A, khó trên 6 GB); HQQ/GemLite (không xác định); thuật toán suy đoán mới. Nếu có ý tưởng mới trong lúc làm, ghi vào một danh sách riêng, không bắt đầu làm trước khi WP5 xong.

## 12. Việc đầu tiên khi bắt đầu

1. Đọc `three_row_kernel/` và các kernel Triton NF4 có sẵn để biết phần nào dùng lại được.
2. Viết kernel LM head bất biến + bộ thử bằng nhau từng bit (WP1).
3. Báo kết quả qua/không qua cổng WP1 trước khi sang WP2.

---

## 13. Kết quả WP1 (05/11/2026): kernel LM head bất biến theo M — QUA CỔNG

Mã: `src/zassd/kernels/lm_head_invariant.py`; thử nghiệm: `scripts/wp1_lm_head_invariant.py`; số liệu: `results/kernels/wp1_lm_head_*.json`.

Thiết kế cuối: mọi lời gọi (M = 1..16) đều chạy cùng một kernel Triton trên khối kích thước cố định 16 hàng (đệm số 0), `tl.dot` fp16→fp32, K chia khối 128 theo thứ tự cố định, không split-K, không atomics, không auto-tune. Bản đầu dùng `tl.sum` thay cho `tl.dot` chậm hơn cuBLAS 7.3x (24.5 ms so với 3.34 ms) và đã bị bỏ.

Kết quả (RTX 4050 Laptop, Triton 3.6.0, dữ liệu ngẫu nhiên cùng kích thước thật, 200 khối × 7 kích thước M × vị trí ngẫu nhiên):
- **Bất biến:** 0 hàng lệch trên 27 000 hàng, ở cả 5 hình dạng (V,K) = (151936,2048), (151936,1536), (151936,896), (32064,3072), (100352,2048).
- **Chi phí:** M=1: 3.42 ms so với cuBLAS 3.34 ms (1.02x), hằng số theo M=1..8 (cổng yêu cầu ≤ 1.5x và thời gian M=4 < 4 lần M=1: đạt). Chạy từng hàng 3 lần tốn 10.8 ms, kernel mới xử lý 3 hàng trong 3.42 ms (3.2x ít hơn).
- **Phát hiện phụ:** kernel này bằng từng bit với GEMM của cuBLAS ở M≥2 (0 phần tử khác), nhưng cuBLAS ở M=1 (đường gemv) khác 1.2% phần tử. Nghĩa là tham chiếu bất biến trùng với đường GEMM của cuBLAS, không trùng với M=1 mặc định của HF; độ khác biệt token đầu-cuối vẫn phải đo (mục 2).
- **Giới hạn:** dữ liệu ngẫu nhiên, một GPU, một phiên bản Triton; bất biến dựa trên tính chất của MMA và được kiểm chứng bằng thực nghiệm, chưa phải chứng minh. Tích hợp vào mô hình thật là WP4.

Bước tiếp theo theo kế hoạch: WP2 (attention bất biến).
