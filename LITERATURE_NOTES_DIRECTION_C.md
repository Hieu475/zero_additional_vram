# Ghi chú khảo sát tài liệu cho hướng C (03/11/2026)

**Phạm vi và độ tin cậy:** đây là khảo sát nhanh, khoảng 6 truy vấn web, chủ yếu đọc đoạn trích và tóm tắt, chưa đọc toàn văn bài nào. Vì vậy nó chỉ đủ để nói "đã thấy các công trình gần dưới đây", không đủ để khẳng định "chưa ai làm". Cần đọc toàn văn các bài ở mục 1 trước khi viết phần so sánh trong bài báo.

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
