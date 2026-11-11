# Đề cương bài báo (hướng kernel bất biến theo M cho giải mã suy đoán exact trên mô hình 4-bit)

Ngày: 05/11/2026, cập nhật 11/11/2026 (thêm bộ nháp học được). Đề cương làm việc, không phải bản thảo. Số liệu trỏ về `docs/PLAN_SHAPE_INVARIANT_KERNELS.md` (mục 13–18), `docs/DRAFTER_2X_PLAN.md` và `results/kernels/`, `results/drafter_runs/`.

## Luận điểm
Giải mã suy đoán trên mô hình 4-bit mất tính "không đổi kết quả" vì kernel đổi chiến lược theo số hàng (đã đo: PLD ngây thơ lệch 26–82% chuỗi, mô hình nháp thật 34–66%). Chạy từng hàng khắc phục nhưng mất lợi ích băng thông. Bộ kernel 4-bit bất biến theo M (LM head, attention, RMSNorm, GEMM NF4) giữ exact từng bit và thực sự đọc trọng số một lần cho cả khối.

## Đóng góp (theo độ tin cậy giảm dần)
1. Bộ kernel bất biến theo M, có thiết kế "bất biến theo cấu trúc" (padding hàng cố định, thứ tự rút gọn cố định, không split-K/atomics) và kiểm chứng từng bit (≥ 27 000 hàng cho mỗi kernel, nhiều hình dạng/dtype); GEMM NF4 CUDA nhanh hơn kernel bnb ở M≥3 (17.7 so với 21.2 ms/bước ở M=3–4; 23.6 so với 31 ms ở M=5–6) và ngang ở M=1.
2. Vòng giải mã suy đoán exact trên CUDA graph: 700 lượt chạy, 4 mô hình, 3 tập, 0 lệch (token và KV từng bit), nhanh hơn baseline exact cùng kernel 1.14–1.32x và nhanh hơn vòng exact cũ (chạy từng hàng) 1.27–1.33x.
3. Công cụ kiểm toán theo toán tử và bảng thực nghiệm về toán tử nào nhạy theo hình dạng (động cơ).
4. Kết quả âm: mô hình ngôn ngữ nhỏ riêng làm bản nháp (0.5B cho mục tiêu 3B) không có lợi (0.66x); tối ưu kernel còn ≤ 10%; khối biến đổi theo độ tin cậy cho bản lai: trần oracle +5–6%, chính sách thực tế ≤ +0.7% và âm khi tính đồng bộ host (mô phỏng). Lưu ý: đây KHÁC với bộ nháp học được ở mục 6 (dùng trạng thái ẩn của chính mô hình đích, không phải một mô hình riêng); bài phải nói rõ hai thứ này để không mâu thuẫn.
5. Phát hiện về định nghĩa: tham chiếu bất biến khác đường mặc định ở 56–70% chuỗi (độ chính xác GSM8K 72% so với 66%, n=50, trong nhiễu); mọi cài đặt hợp lệ khác thứ tự cộng đều tách nhau, nên exact chỉ có nghĩa so với tham chiếu xác định.
6. Bộ nháp học được kiểu EAGLE (đầu vào là trạng thái ẩn cuối của mô hình đích; LM head thu gọn 16384 token vì LM head đầy đủ tốn ~3 ms/lần đọc) và bản lai PLD trong graph, chạy trong cùng CUDA graph, xác minh bằng kernel bất biến, exact trong cả 18 ô. Tăng tốc so với baseline M=1 exact (trung bình 2–3 lượt xen kẽ, 3B / 1.5B): GSM8K 1.70x / 1.58x (PLD 1.22x / 1.23x); HumanEval bản lai 1.58x / 1.70x (PLD 1.39x / 1.53x); CNN/DM PLD tốt nhất hoặc hòa (1.25x / 1.55x). Mô hình chi phí đã hiệu chuẩn (mô phỏng cao hơn đo ~6%) giải thích trần ~1.9x. Cao nhất 1.70x, chưa đạt 2x.

## Cấu trúc
1 Introduction · 2 Định nghĩa exact và tham chiếu · 3 Kiểm toán toán tử (động cơ) · 4 Thiết kế kernel bất biến · 5 Vòng giải mã exact trên CUDA graph (kèm 5b: bộ nháp học được và bản lai trong graph, mô hình chi phí) · 6 Đánh giá (bất biến, chi phí kernel, tốc độ đầu-cuối, tham chiếu so với mặc định) · 7 Kết quả âm (mô hình nháp, tối ưu kernel) · 8 Threats to validity · 9 Related work (batch-invariant ops, LLM-42, Integer Alibi; xem `docs/direction_c/LITERATURE_NOTES_DIRECTION_C.md`)

## Điều còn thiếu trước khi nộp (nói thẳng)
- Một GPU (RTX 4050 6 GB), hai kích thước Qwen (3B, 1.5B) cho bộ nháp học được: cần ít nhất một GPU khác và một họ mô hình khác (Mistral-7B chưa thử; bộ nháp d=4096 đắt hơn tương ứng nên chưa chắc có lợi hơn). Không có thì nhắm TMLR hoặc workshop, không nhắm ICML/MLSys.
- Bộ nháp học được phải huấn luyện riêng cho từng mô hình (khoảng 1–2 giờ sinh dữ liệu + 20 phút huấn luyện): phải tính vào câu chuyện "miễn phí". Còn overfit (loss train 0.9–1.6 trong khi holdout tăng chậm); tập huấn luyện ~10 nghìn chuỗi, nhỏ hơn nhiều so với tài liệu EAGLE.
- CNN/DM của 3B (PLD, bộ nháp thuần) chỉ có 2 lượt; khoảng cách bản lai với PLD trên CNN/DM 1.5B (thua ~0.08x) chưa có nguyên nhân được kiểm chứng.
- Phi-3.5 chưa hỗ trợ (head_dim 96); các kiến trúc khác chưa thử; bnb 0.50.1/0.50.2 chưa thử.
- Bất biến được kiểm chứng thực nghiệm và theo cấu trúc, chưa chứng minh hình thức; phụ thuộc phiên bản trình biên dịch (Triton 3.6.0, nvcc 12.8).
- Tăng tốc phụ thuộc miền: PLD 1.14–1.32x; bộ nháp học được đến 1.70x trên toán nhưng không thắng PLD trên code và tóm tắt; bản lai đến 1.70x trên code (1.5B). Chưa tập nào đạt 2x.
- Cần đọc kỹ phần Evaluation của LLM-42 và đối chiếu công trình đồng thời trước khi nộp.
