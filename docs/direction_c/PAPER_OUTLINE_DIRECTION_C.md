# Đề cương bài báo: kiểm toán độ bất biến theo hình dạng cho suy luận 4-bit (hướng C)

Ngày: 05/11/2026. Tài liệu này là đề cương làm việc cho tác giả, không phải bản thảo. Mọi con số trỏ về file bằng chứng trong `experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/` (gọi tắt `PROFILES/`) hoặc tài liệu `RESEARCH_DIRECTIONS_A_C.md` (gọi tắt `RDAC`). Chỗ nào còn thiếu bằng chứng được ghi rõ là **THIẾU**.

## 1. Luận điểm đề xuất

> Khi một mô hình 4-bit chạy khối K+1 token thay vì từng token, kết quả đổi từng bit ở một tập toán tử xác định được bằng đo đạc (attention, LM head, đôi khi RMSNorm và phép chiếu, tùy backend và phiên bản). Tập này quyết định giải mã suy đoán có còn bằng greedy tuần tự hay không. Đo từng toán tử rồi chạy riêng đúng tập đó từng hàng khi xác minh khôi phục tính exact.

Đóng góp cần nói thẳng, theo thứ tự đáng tin giảm dần:
1. **Một phương pháp và công cụ kiểm toán** (`scripts/audit/shape_audit.py`: `ops`, `plan`, `trajectory`, `summarize`) và bằng chứng rằng danh sách do `plan` sinh ra đưa lệch quỹ đạo về 0 trong mọi lượt chạy của mã đã sửa (3 tập dữ liệu, 6 cấu hình mô hình/backend; trên `cnndm_ext150` độc lập: 6 cấu hình, mỗi ô 0/60).
2. **Một bảng thực nghiệm** về toán tử nào nhạy theo hình dạng, trên trọng số 4-bit, theo M, ngữ cảnh, mô hình, backend và phiên bản bitsandbytes.
3. **Số đo hệ quả đầu-cuối:** PLD ngây thơ lệch 22–82% số chuỗi (tùy cấu hình), dù chỉ khoảng 0.4–1% hàng đổi argmax.
4. **Kết quả âm có giá trị:** giải mã suy đoán exact chỉ thêm 1.009x trên nền CUDA graph exact (hướng A).

Không nên tuyên bố: "phát hiện hiện tượng mới" (nguyên lý đã có ở batch-invariant ops và LLM-42), "đảm bảo exact" (chỉ quan sát trên mẫu), hay "tổng quát cho mọi backend 4-bit".

## 2. Vị trí so với công trình có sẵn (xem `LITERATURE_NOTES_DIRECTION_C.md`)

- Batch-invariant ops (Thinking Machines), SGLang/vLLM deterministic mode: nguyên lý và kernel cho bf16, không lượng tử hóa.
- LLM-42 (SOSP 2026): xác minh cửa sổ cố định; **không** xét lượng tử hóa trọng số; tự nhận **chưa** tích hợp giải mã suy đoán; yêu cầu nhất quán theo hình dạng cố định, không phải bằng greedy M=1.
- Integer Alibi (2608.13756) và bài kế tiếp: INT8 W8A8, hai kernel, không quét kích thước khối, 4-bit là hướng tương lai.
- Khoảng trống là "không thấy trong tập đã đọc", **không** phải "không tồn tại". **THIẾU:** Yuan et al. 2025 (arXiv:2506.09501) và "Lossless but Not Free" (2607.17283) chưa đọc; phần Evaluation của LLM-42 chưa đọc kỹ.

## 3. Cấu trúc đề xuất

| Mục | Nội dung | Bằng chứng |
|---|---|---|
| 1. Introduction | Giải mã suy đoán "lossless" lý thuyết, nhưng kernel 4-bit đổi chiến lược theo hình dạng; một câu hỏi cụ thể | — |
| 2. Setting & định nghĩa "exact" | Tham chiếu = greedy tuần tự M=1 (cùng cache, cùng mô hình); exact = token (và KV) bằng từng bit | README mục "What exact means"; lưu ý hướng A dùng tham chiếu `StaticCache` khác |
| 3. Phương pháp | (a) `ops`: gọi lại từng toán tử từng hàng trên cùng đầu vào thật, so từng bit; (b) `plan`: lớp nào có bất kỳ phần tử lệch; (c) `trajectory --rowwise`: kiểm tra đầu-cuối | `scripts/audit/audit_op_shape_sensitivity.py`, `shape_audit.py`, `audit_pld_trajectory_divergence.py` |
| 4. Kết quả A: toán tử nào lệch | Bảng theo mô hình × M × ngữ cảnh | `C_SHAPE_AUDIT_SUMMARY.md` (bảng 1) |
| 5. Kết quả B: hệ quả đầu-cuối | PLD ngây thơ lệch quỹ đạo, khoảng Wilson | `C_SHAPE_AUDIT_SUMMARY.md` (bảng 2) |
| 6. Kết quả C: bản sửa theo kiểm toán | 0 ca lệch, đối chứng âm (bỏ một toán tử), kiểm chứng độc lập | `fix_*`, `heldout_*`, `ext150_*` |
| 7. Phụ thuộc phiên bản/backend | bnb 0.49.2 so với 0.50.0; torchao; HQQ không xác định | `c10_*`, `c8_*`, `run_to_run_noise_v1.json` |
| 8. Kết quả âm: CUDA graph | baseline exact 1.40x, suy đoán thêm 1.009x | `a1_*`, `a2_*`, `a3_*` |
| 9. Threats to validity | xem mục 5 dưới đây | — |
| 10. Related work, Conclusion | — | `LITERATURE_NOTES_DIRECTION_C.md` |

## 4. Hình và bảng đề xuất

1. **Bảng toán tử × cấu hình:** tỷ lệ phần tử lệch (phép chiếu, RMSNorm, LM head, SDPA), M=3 và M=4, 8 cấu hình (bnb 0.50.0 ×5 mô hình, bnb 0.49.2, torchao). Nguồn: bảng 1 của `C_SHAPE_AUDIT_SUMMARY.md`.
2. **Hình: tỷ lệ chuỗi lệch quỹ đạo** của PLD ngây thơ so với có danh sách `plan`, theo mô hình và tập dữ liệu, kèm khoảng tin cậy. Nguồn: bảng 2.
3. **Hình: quét M** (2 đến 16) của lệch phép chiếu cho bnb 0.50.0 và 0.49.2 (một đường bậc thang tại M=4 và một đường phẳng ở 34%).
4. **Bảng đối chứng âm:** bỏ lần lượt từng toán tử khỏi danh sách (3B: bỏ RMSNorm còn 3/50; K=3 thiếu phép chiếu còn 18/50).
5. **Hộp "bài học":** lỗi GQA (K/V nhân bản so với K/V gốc) chỉ lộ ở ngữ cảnh dài.
6. **Bảng hướng A:** token/s eager, graph M=1, vòng suy đoán.

## 5. Threats to validity (nên đặt thành mục riêng)

- **Một GPU (RTX 4050), một phiên bản mỗi thư viện.** Ngưỡng M=4 là đặc thù bnb 0.50.0 (xem so sánh 0.49.2); heuristic điều phối còn đổi ở 0.50.1/0.50.2 và theo kiến trúc GPU. **THIẾU:** GPU/phiên bản khác (bạn sẽ làm sau).
- **Mô hình ≤ 3B**, K ≤ 3, greedy, batch 1.
- **Danh sách `plan` được chọn trên GSM8K 20 prompt** rồi thử ở HumanEval và CNN/DM. Bản sửa GQA được viết sau khi thấy lỗi trên CNN/DM; `cnndm_ext150` (60 prompt, 0/150 trùng với tập đã dùng) là kiểm chứng độc lập của nó.
- **"0 ca lệch" là quan sát trên mẫu** (khoảng Wilson 95% trên là 6–7% cho 50–60 prompt), không phải chứng minh exact.
- **Ngữ cảnh dài của kiểm toán** ghép prompt GSM8K làm đệm; CNN/DM (164–605 token) là văn bản tự nhiên nhưng chưa tới 3000 token.
- **Mỗi ô chạy một lần**, chưa lặp lại độc lập.
- **HQQ+GemLite bị loại vì không xác định;** hiện chỉ có hai backend hợp lệ (bnb, torchao), cùng họ INT4 chỉ trọng số. **THIẾU:** GPTQ/AWQ (cần biên dịch kernel).
- **Tham chiếu khác nhau** giữa hướng A (`StaticCache` M=1) và B/C (cache của dự án).

## 6. Việc còn thiếu trước khi nộp (không tính GPU/phiên bản khác)

1. Đọc Yuan et al. 2025, "Lossless but Not Free" và phần Evaluation của LLM-42, rồi cập nhật phần Related Work.
2. Nhân bản độc lập: đã có một lần (ext150 prompt 60–119: Qwen 3B 49/60 ngây thơ, 0/60 với danh sách; torchao 0/60). Chưa lặp lại cho các cấu hình khác.
3. Mô hình lớn hơn 3B nếu có cách chạy; Llama 3.2 (cần token HF).
4. Văn bản dài hơn 1000 token trong PLD: đã có 12 prompt (511–3072 token, Qwen 3B và OLMo-2: 0/12 với danh sách), nhưng mẫu nhỏ và là bài báo ghép; cần mẫu lớn hơn.
5. Viết lại phần hướng A sao cho nó là kết quả phụ, không phải đóng góp chính.
6. Chốt nơi nộp và hạn chót từ trang chính thức của hội nghị (tôi chưa tra; không nên dựa vào trí nhớ).
