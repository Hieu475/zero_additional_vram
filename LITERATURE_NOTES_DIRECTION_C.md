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

## 6. Đọc bổ sung (04/11/2026, đợt 2): LLM-42 phần sau, mã nguồn, Integer Alibi, bitsandbytes

Nhãn: [V] = nguyên văn trang; [S] = bản tóm tắt do công cụ tải trang sinh ra; [T] = đoạn trích kết quả tìm kiếm.

**LLM-42** — phần Evaluation/Related Works vẫn **chưa lấy được nguyên văn** (bản PDF trả về nhị phân; HTML chỉ nguyên văn đến mục 4.2).
- [V] Chỉ nguyên văn đến 4.2; không có từ khóa lượng tử hóa nào trong phần đó. [S] Các phần sau cũng không có.
- [V] "Batch-invariant GEMM kernels ... không dùng split-K [14, 26]"; [S] tài liệu [14] có tiêu đề "Accelerating a triton fused kernel for w4a16 quantized inference with splitk work decomposition" (chưa xác nhận với danh mục tham khảo).
- [S] Phần hạn chế: "chưa tích hợp với suy luận dựa trên giải mã suy đoán". [V] Giải mã suy đoán chỉ là phép loại suy (Abstract, mục 1, Bảng 1, mục 4.1).
- [V] Thí nghiệm lệch: 350 yêu cầu ShareGPT so với batch size 1, đo lệch mức token, không đo tỷ lệ lệch từng bit theo cửa sổ.
- [V] Repo `microsoft/llm-42` (README, cây thư mục): xây trên SGLang v0.5.3, 4×H100 PCIe, cờ `--llm42-window-size` (mặc định 64), `--llm42-verify-batch-size` (mặc định 8). Không có hỗ trợ hay nhắc đến lượng tử hóa; có thư mục `test_batch_invariance/` (chưa mở file kiểm thử nên không biết nó khẳng định gì).
- [V] `batch_invariant_ops` (Thinking Machines): chỉ `mm`, `addmm`, `log_softmax`, `mean`; không nhắc lượng tử hóa.

**"The Integer Alibi" (2608.13756) và 2609.00363** — đọc nguyên văn gần đầy đủ; bản tiền ấn phẩm một tác giả, chưa bình duyệt.
- [V] Chỉ W8A8 INT8, hai kernel (CUTLASS so với Triton) trong vLLM 0.27.1, một RTX 4090; Qwen3-1.7B/8B (14B chỉ end-to-end).
- [V] **Không** nghiên cứu 4-bit trọng số (NF4, GPTQ, AWQ, HQQ, bitsandbytes không được nhắc; 4-bit chỉ nằm ở "hướng tương lai"), **không** quét kích thước batch/khối (lô {1,4,16} được đăng ký trước nhưng "không được thực hiện"), **không** nghiên cứu giải mã suy đoán (chỉ trích dẫn LLM-42), ảnh chụp tầng chỉ ở M=512 kiểu prefill, **không** có ảnh chụp decode theo tầng.
- [V] Kết luận chính: tích lũy INT32 chính xác nên lệch nằm ở áp thang/làm tròn đầu ra, tối đa 1 khoảng bf16; với thang lũy thừa của 2 thì 196/196 tầng giống từng bit; FP8 lệch tăng theo độ sâu rút gọn K (8.2% → 52.7% phần tử).
- Hệ quả: cơ chế của họ (bộ tích lũy số nguyên chính xác) **không áp dụng** cho NF4/HQQ, nơi bộ tích lũy là dấu phẩy động. Vì vậy phần của ta không bị họ bao trùm, nhưng họ cho một mẫu phương pháp đáng học: đăng ký trước, thử tiêm lỗi để chứng minh kiểm tra thật sự "bắn", và báo cáo hạn chế thẳng thắn.

**bitsandbytes** [V]
- PR #1949 ("New 4bit GEMM kernels for inference", hợp nhất 21/05/2026) **nằm trong bản 0.50.0** (chính bản ta dùng). Nó thay đường GEMV batch 1 và đường dequantize + `F.linear` cho batch nhỏ đến vừa; ba tầng kernel (SIMT sm60+, MMA m16n8k8 sm75, MMA m16n8k16 sm80+), chọn lúc chạy theo hình dạng, kiến trúc GPU và số SM; lời của người bảo trì: trên 1536 token luôn dùng đường cũ, "đa số GPU và hình dạng > 128 token đủ để quay về đường cũ". Không có ngưỡng M duy nhất. PR không bàn về xác định hay batch-invariance.
- 0.50.1 và 0.50.2 đổi heuristic điều phối (cho GB10) → kết quả của ta phụ thuộc phiên bản và GPU, đúng như đã cảnh báo.
- **Hệ quả cho các kết luận C đã ghi:** ngưỡng "M=4 làm phép chiếu NF4 vỡ" là hiện tượng đo được trên *kernel mới 0.50.0, RTX 4050 (sm_89)*, không thể tổng quát hóa sang GPU/phiên bản khác. Cách diễn đạt trong bài phải là "với bitsandbytes 0.50.0 trên sm_89", kèm đề xuất lặp lại trên bản 0.49.2 (đường cũ) để so sánh trước/sau PR.

**Tìm kiếm [T] (chưa mở nguồn):**
- vLLM bỏ chuyển sang kernel Marlin khi bật chế độ batch-invariant với lý do "Marlin không bất biến theo batch" (tài liệu `awq_marlin`) → ghi nhận gián tiếp rằng kernel 4-bit nhanh không bất biến theo batch; chưa tìm thấy nghiên cứu đo lường.
- "Lossless but Not Free: An Empirical Anatomy of Speculative Decoding on Consumer Hardware" (alphaxiv 2607.17283): tiền ấn phẩm một tác giả về giải mã suy đoán trên phần cứng tiêu dùng, nêu runtime lượng tử hóa trên Metal chạy xác minh "song song" tuần tự. Gần với bối cảnh của ta (phần cứng tiêu dùng, lượng tử hóa); **cần đọc**.
- Tài liệu TensorRT-Edge-LLM nêu rằng nếu xác minh dùng đường tính toán khác với giải mã tự hồi quy thì làm tròn khác nhau và greedy có thể đổi token → xác nhận vấn đề ở mức tài liệu.

## 7. Kết luận cập nhật về tính mới

- **Khoảng trống được củng cố**: trong mọi nguồn đã đọc (nguyên văn hoặc tóm tắt), không thấy ai đo (a) lệch từng bit theo toán tử, (b) trên trọng số 4-bit, (c) theo kích thước khối M, (d) gắn với lệch quỹ đạo của giải mã suy đoán. Bài INT8 gần nhất xác nhận trực tiếp rằng 4-bit và batch/khối là chưa làm (hướng tương lai).
- **Vẫn chưa thể khẳng định "chưa ai làm"**: LLM-42 phần sau chỉ có tóm tắt; kho tìm kiếm của tôi hẹp; "Lossless but Not Free" chưa đọc; nhiều tiền ấn phẩm 2026 có thể bỏ sót.
- **Việc còn lại cho mục tính mới:** đọc "Lossless but Not Free"; lấy nguyên văn Evaluation/Related Works của LLM-42 (tải PDF về máy rồi `pdftotext`); thử bản bitsandbytes 0.49.2 (trước PR #1949) để thấy sự khác biệt.

## 8. LLM-42: đọc nguyên văn từ PDF (05/11/2026)

Nguồn: PDF arXiv 2601.17768v2 (15 trang) tải về, trích bằng `pdftotext` (đọc theo thứ tự cột). Các câu dưới đây là trích nguyên văn từ văn bản trích xuất; **chưa đọc kỹ** phần Evaluation (mục 5.1–5.3 gồm bảng và hình) nên không kết luận gì về số liệu hiệu năng của họ ngoài những gì đã nêu ở mục 4.

**Phạm vi và mô hình.** "While we report performance results only for Llama-3.1-8B-Instruct in this paper, we have evaluated the correctness of our approach across multiple additional models, including Qwen-4B-Instruct-2507, Qwen3-14B and Qwen3-30B-A3B-Instruct-2507, and across 1-4 GPUs with tensor-parallelism." Không có từ khóa lượng tử hóa trong thân bài; từ `quantized` chỉ xuất hiện duy nhất trong danh mục tham khảo, mục [15]: Hoque, Wright, Yang, Srivatsa, Ganti, "Accelerating a triton fused kernel for w4a16 quantized inference with splitk work decomposition" (2024), được trích như ví dụ của chiến lược split-K trong GEMM. Họ không đánh giá mô hình lượng tử hóa.

**Giả định then chốt (O2 và chú thích 3).** "Most GPU kernels use uniform, shape-consistent reductions ... the strategy remains fixed for all batches of the same shape, changing only when the shape changes." Chú thích 3: "such guarantees do not hold for kernels that implement reductions via atomic operations. Fortunately, kernels used in the LLM forward pass do not use atomic reductions." **Liên hệ với kết quả của ta:** kernel HQQ+GemLite cho kết quả khác nhau giữa hai lần gọi giống hệt nhau (64% phần tử ở M=1, `run_to_run_noise_v1.json`), tức là một kernel 4-bit thực tế vi phạm đúng giả định này. Chưa xác định nguyên nhân (chưa kiểm tra có dùng cộng nguyên tử hay không), nên không khẳng định đó là phản ví dụ của chú thích 3, chỉ nói rằng giả định "kernel trong forward không xác định" không đúng cho GemLite 0.6.0 trên máy này.

**Giải mã suy đoán.** Hạn chế, nguyên văn: "Our implementation also does not currently integrate with speculative decoding–based LLM inference." Họ dùng giải mã suy đoán như một cơ chế (decode–verify–rollback) để thực thi tính xác định, không phải để tăng tốc rồi đảm bảo exact. Điều này xác nhận khoảng trống: vấn đề đảm bảo exact cho *giải mã suy đoán thật* dưới kernel nhạy theo hình dạng chưa được họ xét.

**Cách họ đảm bảo xác định.** Bộ xác minh chạy trên cửa sổ có kích thước cố định để kernel luôn gặp cùng hình dạng (O2), tận dụng việc kết quả tại một vị trí token chỉ cần nhất quán qua các lần chạy (O3). Cách này khác cách của ta: ta so với tham chiếu M=1 (từng hàng); họ chỉ yêu cầu nhất quán giữa các lần chạy cùng hình dạng, không yêu cầu bằng kết quả M=1. Hai mục tiêu khác nhau (xác định theo hình dạng cố định so với bằng với greedy tuần tự).

**Related Works (đọc nguyên văn).** Nhắc: Orca, vLLM, Sarathi-Serve, DistServe, Splitwise, FlexGen, giải mã suy đoán [9, 22, 25]; về xác định: Song et al. [36] (bất định làm méo đánh giá), Yuan et al. [44] ("linked reproducibility failures to mixed-precision arithmetic and fused kernel inconsistencies"), Rainbird AI [5], Zhang et al. [46] (kernel bất biến theo kích thước tensor-parallel), và hướng batch-invariant (He et al.). **Không có công trình nào về lượng tử hóa trọng số 4-bit hay về tính exact của giải mã suy đoán được trích dẫn.**

**Hệ quả cho tính mới (cập nhật):** sau khi đọc nguyên văn các phần trên, khoảng trống của hướng C được củng cố thêm bằng ba điểm: (i) LLM-42 không xét lượng tử hóa trọng số, (ii) không tích hợp với giải mã suy đoán (tự nhận là hạn chế), (iii) Related Works của họ không trích công trình nào về hai chủ đề đó. Điều này không loại trừ công trình khác ngoài tập đã đọc (đặc biệt các tiền ấn phẩm 2026 chưa tìm tới). Tài liệu cần đọc tiếp nếu muốn chắc hơn: Yuan et al. 2025 (arXiv:2506.09501), "Lossless but Not Free" (alphaxiv 2607.17283).
