# Đề cương bài báo (hướng kernel bất biến theo M cho giải mã suy đoán exact trên mô hình 4-bit)

Ngày: 05/11/2026. Đề cương làm việc, không phải bản thảo. Số liệu trỏ về `docs/PLAN_SHAPE_INVARIANT_KERNELS.md` (mục 13–17) và `results/kernels/`.

## Luận điểm
Giải mã suy đoán trên mô hình 4-bit mất tính "không đổi kết quả" vì kernel đổi chiến lược theo số hàng (đã đo: PLD ngây thơ lệch 26–82% chuỗi, mô hình nháp thật 34–66%). Chạy từng hàng khắc phục nhưng mất lợi ích băng thông. Bộ kernel 4-bit bất biến theo M (LM head, attention, RMSNorm, GEMM NF4) giữ exact từng bit và thực sự đọc trọng số một lần cho cả khối.

## Đóng góp (theo độ tin cậy giảm dần)
1. Bộ kernel bất biến theo M, có thiết kế "bất biến theo cấu trúc" (padding hàng cố định, thứ tự rút gọn cố định, không split-K/atomics) và kiểm chứng từng bit (≥ 27 000 hàng cho mỗi kernel, nhiều hình dạng/dtype); GEMM NF4 CUDA nhanh hơn kernel bnb ở M≥3 (17.7 so với 21.2 ms/bước ở M=3–4; 23.6 so với 31 ms ở M=5–6) và ngang ở M=1.
2. Vòng giải mã suy đoán exact trên CUDA graph: 700 lượt chạy, 4 mô hình, 3 tập, 0 lệch (token và KV từng bit), nhanh hơn baseline exact cùng kernel 1.14–1.32x và nhanh hơn vòng exact cũ (chạy từng hàng) 1.27–1.33x.
3. Công cụ kiểm toán theo toán tử và bảng thực nghiệm về toán tử nào nhạy theo hình dạng (động cơ).
4. Kết quả âm: mô hình nháp thần kinh không có lợi với mục tiêu 4-bit 3B (0.66x); tối ưu kernel còn ≤ 10%.
5. Phát hiện về định nghĩa: tham chiếu bất biến khác đường mặc định ở 56–70% chuỗi (độ chính xác GSM8K 72% so với 66%, n=50, trong nhiễu); mọi cài đặt hợp lệ khác thứ tự cộng đều tách nhau, nên exact chỉ có nghĩa so với tham chiếu xác định.

## Cấu trúc
1 Introduction · 2 Định nghĩa exact và tham chiếu · 3 Kiểm toán toán tử (động cơ) · 4 Thiết kế kernel bất biến · 5 Vòng giải mã exact trên CUDA graph · 6 Đánh giá (bất biến, chi phí kernel, tốc độ đầu-cuối, tham chiếu so với mặc định) · 7 Kết quả âm (mô hình nháp, tối ưu kernel) · 8 Threats to validity · 9 Related work (batch-invariant ops, LLM-42, Integer Alibi; xem `docs/direction_c/LITERATURE_NOTES_DIRECTION_C.md`)

## Điều còn thiếu trước khi nộp (nói thẳng)
- Một GPU (RTX 4050 6 GB), mô hình ≤ 3B: cần ít nhất một GPU khác và mô hình ≥ 7B (chỗ mô hình nháp có thể có lợi). Không có thì nhắm TMLR hoặc workshop, không nhắm ICML/MLSys.
- Phi-3.5 chưa hỗ trợ (head_dim 96); các kiến trúc khác chưa thử; bnb 0.50.1/0.50.2 chưa thử.
- Bất biến được kiểm chứng thực nghiệm và theo cấu trúc, chưa chứng minh hình thức; phụ thuộc phiên bản trình biên dịch (Triton 3.6.0, nvcc 12.8).
- Tăng tốc 1.14–1.32x là khiêm tốn và phụ thuộc mức PLD tìm được bản nháp (HumanEval cao nhất).
- Cần đọc kỹ phần Evaluation của LLM-42 và đối chiếu công trình đồng thời trước khi nộp.
