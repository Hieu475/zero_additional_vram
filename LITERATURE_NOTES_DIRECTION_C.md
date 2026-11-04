# Ghi chú khảo sát tài liệu cho hướng C (03/11/2026)

**Phạm vi và độ tin cậy:** mục 1–3 là khảo sát nhanh (khoảng 6 truy vấn web, đoạn trích). Mục 4 (bổ sung 04/11/2026) là lần đọc nội dung hai bài chính qua công cụ tải trang. Hạn chế của mục 4: với LLM-42, chỉ khoảng 39 nghìn ký tự đầu (đến mục 4.2) được trả về nguyên văn; mục 4.3 trở đi, phần Evaluation và Related Works chỉ có bản tóm tắt do mô hình trích, nên các ý đánh dấu [tóm tắt] cần kiểm tra lại với bản gốc. Chưa xem mã nguồn github.com/microsoft/llm-42 và `batch_invariant_ops`. Vì vậy vẫn không đủ để khẳng định "chưa ai làm".

## 1. Công trình gần nhất đã tìm thấy

| Công trình | Nội dung (theo đoạn trích) | Quan hệ với hướng C |
|---|---|---|
| Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" (2025) và thư viện `batch_invariant_ops`; SGLang (lmsys, 22/09/2025) và vLLM (`VLLM_BATCH_INVARIANT=1`, theo một bài blog, chưa kiểm tra tài liệu vLLM) | Nguồn gốc bất định là kernel không bất biến theo kích thước batch (RMSNorm, matmul, attention); giải pháp là kernel bất biến theo batch (chia split cố định trong attention, bỏ Split-K) | **Trùng ý tưởng cốt lõi** (độ nhạy theo hình dạng của RMSNorm/matmul/attention). Công trình đó nhắm mô hình không lượng tử hóa và độ chính xác bf16, không phải trọng số 4-bit hay khối xác minh của giải mã suy đoán |
| LLM-42: Enabling Determinism in LLM Inference with Verified Speculation (arXiv 2601.17768; SOSP 2026) | Đường nhanh sinh token, bộ xác minh phát lại dưới lịch rút gọn hình dạng cố định, giữ token nhất quán và quay lui phần còn lại; cài trên SGLang | **Gần nhất về cơ chế** (xác minh theo kiểu suy đoán + quay lui để giữ tính xác định). Đã được tài liệu hóa là công trình gần nhất ngay từ trong dự án. Chưa rõ họ có xét lượng tử hóa 4-bit hay không |
| "The Integer Alibi" (arXiv 2608.13756) và bài tiếp theo (arXiv 2609.00363) | Đổi kernel INT8 (CUTLASS so với Triton) trong vLLM; bộ tích lũy INT32 chính xác nên lệch nằm ở bước áp thang và làm tròn đầu ra; chênh tối đa một khoảng cách bfloat16 | **Gần về phương pháp** (định vị lệch giữa các kernel theo toán tử), nhưng là INT8 và so sánh giữa hai kernel, không phải theo hình dạng khối |
| bitsandbytes PR #1949 (hợp nhất tháng 05/2026) | Thêm kernel GEMM 4-bit hợp nhất thay đường GEMV batch 1 và đường dequantize + `F.linear` cho batch nhỏ đến vừa | **Rủi ro cho kết quả của chúng ta:** ngưỡng M=4 của bitsandbytes 0.50.0 có thể đã đổi ở phiên bản mới. Chưa kiểm tra bản 0.50.0 đã chứa PR này hay chưa |
| Tài liệu vLLM về giải mã suy đoán | Nêu giải mã suy đoán "lossless" chỉ tới giới hạn số học phần cứng; đổi kích thước batch có thể làm đổi logprob | Xác nhận vấn đề đã được biết ở mức tổng quát, không đo theo toán tử |

## 2. Đánh giá thẳng về tính mới

- **Ý tưởng "độ nhạy theo hình dạng của từng toán tử" không mới.** Đã có trong công trình batch-invariant và LLM-42.
- **Phần có thể còn mới (chưa xác nhận):** (i) kiểm toán theo toán tử trên trọng số 4-bit (bitsandbytes NF4, HQQ+GemLite), gồm phát hiện ngưỡng M=4 của phép chiếu NF4 trên nhiều họ mô hình; (ii) gắn độ lệch toán tử với hệ quả cuối-cuối của PLD (6–14/20 chuỗi lệch quỹ đạo); (iii) ghi nhận HQQ+GemLite lệch gần 50% phần tử ở phép chiếu nhưng ít đổi argmax; (iv) nền tảng khối xác minh nhiều hàng cho giải mã suy đoán chứ không chỉ batch động.
- **Rủi ro chính:** nếu LLM-42 hoặc các bài khác đã xét lượng tử hóa 4-bit, (i)–(iii) sẽ yếu đi nhiều. Cần đọc toàn văn LLM-42 (mục thực nghiệm và công trình liên quan) để biết.

## 3. Việc cần làm tiếp
1. Đọc toàn văn LLM-42 và "Defeating Nondeterminism": họ có thử mô hình lượng tử hóa không, và đo gì ở mức toán tử.
2. Tìm thêm: bitsandbytes/HQQ/GPTQ/AWQ trong ngữ cảnh xác định, và suy luận xác định với giải mã suy đoán trên mô hình lượng tử hóa.
3. Kiểm tra bitsandbytes 0.50.0 có chứa PR #1949 không; nếu không, chạy lại với phiên bản mới (đã nằm trong mục C7).
4. Chưa khảo sát: các bài trên arXiv và trang hội nghị khác về cùng chủ đề (tìm theo từ khóa khác, có thể bỏ sót).

## Nguồn
- [Defeating Nondeterminism in LLM Inference (bản đăng lại)](https://community.openai.com/t/defeating-nondeterminism-in-llm-inference/1358623)
- [SGLang deterministic inference](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)
- [LLM-42 (arXiv 2601.17768)](https://arxiv.org/abs/2601.17768v1)
- [The Integer Alibi (arXiv 2608.13756)](https://arxiv.org/pdf/2608.13756)
- [Deterministic LLM Inference Across GPU Kernels (arXiv 2609.00363)](https://arxiv.org/pdf/2609.00363)
- [bitsandbytes PR #1949](https://gh.hhj520.top/bitsandbytes-foundation/bitsandbytes/pull/1949)
- [vLLM speculative decoding docs](https://docs.vllm.ai/en/v0.21.0/features/speculative_decoding/)

## 4. Kết quả đọc nội dung hai bài chính (04/11/2026)

**LLM-42 (arXiv 2601.17768v2)**
- **Không đánh giá hay bàn về mô hình lượng tử hóa trọng số.** Chỉ trích dẫn một tài liệu về "w4a16 với split-K" [tóm tắt, chưa kiểm tra]. Mô hình chính Llama-3.1-8B-Instruct; một số mô hình Qwen chỉ để kiểm tra đúng đắn [tóm tắt]; 4 GPU H100 PCIe, SGLang v0.5.3rc0 [tóm tắt].
- **Không có phép quét độ lệch từng bit theo toán tử.** Bảng 2 chỉ là phân loại định tính (cuBLAS GEMM: không bất biến theo batch; FlashAttention-3 với num_splits=1: bất biến; RMSNorm mặc định của vLLM/SGLang: không bất biến theo batch). Quan sát O2 ("kernel nhất quán theo hình dạng") được khẳng định, không đo từng toán tử. Không có ngưỡng kích thước. LM head không được bàn.
- **Bộ xác minh** chạy trên số token cố định (ví dụ 10 token; chiều sâu 32–64 token ở Bảng 1) và đệm token giả khi thiếu. Cơ chế decode–verify–rollback lấy cảm hứng từ giải mã suy đoán, nhưng không phải giải mã suy đoán với mô hình nháp. Bài nói rõ nó "chưa tích hợp với suy luận dựa trên giải mã suy đoán" [tóm tắt] và chỉ khẳng định tính xác định, không khẳng định tương đương với một tham chiếu.

**Thinking Machines, "Defeating Nondeterminism in LLM Inference" (10/09/2025)**
- Không đề cập lượng tử hóa 4-bit/8-bit hay FP8; chỉ có ví dụ matmul bf16. Mô hình chính Qwen3-235B-A22B-Instruct-2507 (thí nghiệm bất định, 1000 lần × 1000 token: 80 chuỗi khác nhau; với kernel bất biến theo batch: 1000 chuỗi giống nhau) và Qwen-3-8B (đo hiệu năng); GPU không nêu tên.
- Ba toán tử cần xử lý: RMSNorm, matmul (split-K, lệnh tensor-core đổi theo kích thước), attention (split-KV/FlashDecoding, cần kích thước split cố định). Không nêu ngưỡng số.
- Nguyên tắc nêu rõ: thứ tự rút gọn của một token không được phụ thuộc vào số token khác của cùng chuỗi đang được xử lý đồng thời (ví dụ chunked prefill, prefix caching). Giải mã suy đoán và xác minh nhiều token không được nêu tên.

## 5. Hệ quả cho tính mới của hướng C (cập nhật)

- **Mạnh lên:** trong hai bài chính được đọc, không có bài nào có (a) đo từng bit theo toán tử có số liệu, (b) mô hình lượng tử hóa trọng số 4-bit, (c) quét kích thước khối M, (d) liên hệ tới giải mã suy đoán và độ lệch quỹ đạo. Đây là khoảng trống thật trong hai bài đó.
- **Vẫn là cảnh báo:** nguyên lý (độ nhạy theo hình dạng, kernel đổi chiến lược theo kích thước) đã được nêu và chấp nhận rộng rãi; đóng góp của ta là *đo đạc có hệ thống trên 4-bit* và liên hệ với suy đoán, không phải phát hiện ra hiện tượng. Cách định vị nên là "kiểm toán thực nghiệm", không phải "phát hiện mới".
- **Chưa loại trừ:** các bài khác về lượng tử hóa và xác định (đã thấy "The Integer Alibi" cho INT8, chưa đọc nội dung), mã nguồn LLM-42, và các công trình về suy luận xác định cho GPTQ/AWQ/HQQ. Cần tìm thêm trước khi nộp.
- **Việc còn lại:** đọc nguyên văn phần Evaluation/Related Works của LLM-42; xem mã `microsoft/llm-42` và `batch_invariant_ops`; đọc "The Integer Alibi"; kiểm tra PR #1949 của bitsandbytes.
