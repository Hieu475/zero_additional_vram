---
title: "Zero-Additional-VRAM Self-Speculative Decoding — Deep Research Report"
subtitle: "Đánh giá hiện trạng, bản đồ toàn cảnh literature, và lộ trình phát triển hướng tới công bố A*/Q1 và sản phẩm demo"
author: "Biên soạn bởi Claude (Anthropic) — dựa trên review trực tiếp mã nguồn, dữ liệu thực nghiệm và đề cương nghiên cứu tại github.com/Hieu475/zero_additional_vram"
date: "Tháng 9, 2026"
---

# Tóm tắt điều hành

Dự án **Zero-Additional-VRAM Self-Speculative Decoding** đặt mục tiêu tăng tốc suy luận LLM trên GPU laptop 6GB (RTX 4050) bằng cách dùng chính target model làm draft model (bỏ qua một số layer), kết hợp ba thành phần: lựa chọn layer dựa trên CKA, điều chỉnh động độ dài speculation K theo entropy, và một bộ điều khiển nhận biết phần cứng (HECC) tối ưu đồng thời throughput/VRAM/energy.

Báo cáo này tổng hợp ba việc: (1) review trực tiếp toàn bộ mã nguồn, dữ liệu thực nghiệm và đề cương 258 dòng đã có trong repo; (2) rà soát toàn cảnh literature liên quan tính đến hiện tại (khoảng 45 công trình, trải từ nền tảng speculative decoding 2018–2023 tới các paper mới nhất công bố đầu/giữa năm 2026); (3) một lộ trình cụ thể theo từng giai đoạn để đưa dự án từ trạng thái hiện tại (proof-of-concept có bug) tới mức đủ chất lượng cho hội nghị/tạp chí hạng A*/Q1 và một sản phẩm demo dùng được thật.

**Ba kết luận quan trọng nhất:**

1. Dữ liệu thực nghiệm hiện có trong `experiments/07_ablation/summary.json` tự nó cho thấy pipeline self-speculative đầy đủ đang chạy **chậm hơn vanilla 2–3.7 lần**, và tỷ lệ khớp chính xác với greedy decoding (exact-match) chỉ đạt 0.6, không đạt 1.0 như yêu cầu lý thuyết. Đây là vấn đề kỹ thuật cần sửa trước tiên, không phải giới hạn của phương pháp — nguyên nhân gần như chắc chắn nằm ở việc thiếu tái sử dụng KV cache (file `kv_cache.py` hiện chỉ là stub trống) khiến cả draft lẫn verify phải recompute toàn bộ chuỗi mỗi chu kỳ.
2. Phần literature review trong đề cương hiện có đã rất tốt và cập nhật, nhưng **KnapSpec (ICML 2026)** là công trình cạnh tranh trực tiếp và nguy hiểm nhất chưa được đối chiếu bằng thực nghiệm — nó đã làm gần như chính xác ý tưởng "hardware-aware layer selection" mà đề tài nhắm tới, kèm cả nền tảng lý thuyết (cosine similarity làm proxy cho acceptance rate) mà đề cương còn để ngỏ.
3. Có một mạch nghiên cứu 2025–2026 hoàn toàn mới mà đề cương **chưa đề cập**: vấn đề **non-determinism/exactness dưới quantization** (LLM-42, Component-Aware SSD, Batch Speculative Decoding Done Right). Đây trực tiếp giải thích một phần hiện tượng exact-match=0.6 trong dữ liệu của bạn, và là mảng bắt buộc phải đọc trước khi công bố bất kỳ con số "lossless" nào.

---

# Phần 1 — Hiện trạng dự án (từ review trực tiếp mã nguồn)

## 1.1. Những gì đã làm tốt

- Cấu trúc mã nguồn `src/zassd/` đầy đủ, tách module rõ ràng: `layer_selection/` (CKA, static, random), `controllers/` (adaptive_k, entropy, hardware_controller), `decoding/` (speculative, rejection_sampling, verification, vanilla), `profiling/` (energy, gpu, latency, memory).
- Thuật toán rejection sampling (`rejection_sampling.py`) và greedy verification (`verification.py`) được cài đặt đúng theo công thức chuẩn của Leviathan et al. (2023).
- Đo lường phần cứng dùng NVML thật (`pynvml`), tích phân hình thang cho năng lượng (`energy.py`) — đúng phương pháp luận chuẩn.
- Đã chạy thực nghiệm thật trên phần cứng thật (RTX 4050 Laptop, driver 595.84, 4-bit quantization, Qwen2.5-3B-Instruct), có warm-up, nhiều lần lặp, ghi độ lệch chuẩn — kỷ luật thực nghiệm tốt hơn phần lớn đồ án sinh viên.
- Đề cương nghiên cứu 258 dòng có chất lượng literature review và tư duy phản biện khoa học thuộc loại hiếm gặp ở giai đoạn đề cương: tự nhận diện đúng rằng self-speculative decoding không còn là gap, tự đặt câu hỏi "vì sao không dùng CKA đơn độc", có quy trình validate exactness rõ ràng (mục 19).

## 1.2. Vấn đề kỹ thuật cụ thể đã phát hiện (ưu tiên sửa theo thứ tự)

### Bug #1 (nghiêm trọng nhất) — Không có tái sử dụng KV cache → chi phí O(n²)

`src/zassd/cache/kv_cache.py` hiện chỉ là một TODO trống. Trong `decoding/speculative.py`, mỗi chu kỳ verification đặt lại `pkv_draft = None`, buộc pha draft phải forward lại **toàn bộ chuỗi đã sinh từ đầu**; pha verify gọi `model(input_ids=candidate_ids, use_cache=False)` — recompute **toàn bộ chuỗi từ token 0** mỗi chu kỳ, không cache gì cả. Trong khi đó baseline vanilla (`decoding/vanilla.py`) dùng `past_key_values` đúng cách, O(1)/token.

Hệ quả đo được trong `experiments/04_self_speculative/summary.json`: draft-trần (chỉ chạy mạng cắt layer, không có vòng lặp) đạt speedup 1.2–1.6× thật, nhưng pipeline đầy đủ lại **chậm hơn vanilla 2–4 lần**, và càng chậm hơn khi K tăng (K1: 0.49×, K8: 0.24×) — khớp chính xác với một hệ thống có độ phức tạp O(n²) thay vì O(n).

### Bug #2 — HECC dùng hằng số tốc độ đoán, không đo thực

`controllers/hardware_controller.py`, hàm `compute_utility`:
```python
draft_speed_mult = 1.7 if "50" in config_name else 1.25
```
Hệ số hard-code, không lấy từ `draft_ms`/`verify_ms` đo thực tế, và chọn cấu hình qua so khớp chuỗi `"50" in config_name` (rất giòn). Hệ quả đo được: khi bật `hw_feedback=True`, acceptance rate rơi từ 0.49 (CKA+AdaptiveK) xuống **0.032** — controller chọn action tệ vì utility function bị lệch khỏi thực tế đo được.

### Bug #3 — Exact-match chỉ đạt 0.6, không đạt 1.0 như lý thuyết yêu cầu

`verification.py` cài đúng thuật toán, nhưng vanilla dùng đường tính incremental (cache từng bước), còn verify hiện tại tính một lượt full-sequence không cache. Với model 4-bit, hai đường tính này có thể chọn kernel/đường dequantize khác nhau theo shape đầu vào → sai số floating-point cực nhỏ đủ làm lệch argmax ở logit sít sao. Phần 3.6 của báo cáo này tổng hợp một mạch nghiên cứu 2025–2026 xác nhận đây là hiện tượng có thật và phổ biến (không phải chỉ riêng bug của bạn) — nhưng mức 90–96% match "bình thường" trong các paper đó vẫn cao hơn nhiều so với 60% quan sát được ở đây, nghĩa là **vẫn còn ít nhất một bug thật cộng thêm vào phần nhiễu số học vốn có**.

### Vấn đề phụ — `LayerManager` không reindex `layer_idx`

`models/layer_manager.py` thay `nn.ModuleList` bằng danh sách con nhưng không cập nhật thuộc tính `layer_idx` nội bộ của từng layer (dùng để định vị slot trong `Cache` của HuggingFace). Hiện tại chưa gây lỗi vì draft cache bị hủy mỗi chu kỳ (Bug #1), nhưng **sẽ gây lỗi ngay khi bạn sửa Bug #1** nếu không sửa đồng thời.

### Vấn đề về kiểm thử

`tests/test_verification.py` chỉ test đơn vị hàm thuần túy (đúng), nhưng **không có test tích hợp end-to-end** kiểm tra `self_speculative_generate(...) == vanilla_generate(...)`. Đây là bất biến quan trọng nhất của toàn hệ thống; nếu có trong CI, Bug #3 đã bị chặn từ sớm.

---

# Phần 2 — Bản đồ toàn cảnh literature (đầy đủ theo từng nhánh)

Ký hiệu: 🔴 = phải đọc kỹ và implement làm baseline thật; 🟡 = nên đọc, có thể dùng làm ý tưởng mở rộng; 🟢 = tham khảo bối cảnh/phương pháp luận, không cần implement lại.

## 2.1. Nền tảng speculative decoding (bắt buộc hiểu sâu — cơ sở lý thuyết cho phần exactness)

| # | Công trình | Đóng góp cốt lõi | Mức độ |
|---|---|---|---|
| 1 | Stern, Shazeer, Uszkoreit (2018), *Blockwise Parallel Decoding for Deep Autoregressive Models* | Tiền thân: dự đoán nhiều vị trí tương lai rồi verify phần tiền tố dài nhất được chấp nhận | 🟢 |
| 2 | Leviathan, Kalman, Matias (2023), ICML, *Fast Inference from Transformers via Speculative Decoding* | Đặt nền tảng lý thuyết exact speculative decoding, công thức acceptance = min(1, p/q) | 🔴 |
| 3 | Chen et al. (DeepMind, 2023), *Accelerating Large Language Model Decoding with Speculative Sampling* | Phiên bản độc lập cùng thời điểm, cùng cơ chế rejection sampling | 🔴 |
| 4 | Xia et al. (2024), Findings ACL, *Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding* (arXiv:2401.07851) | Survey nền tảng, taxonomy đầy đủ — nên dùng làm khung cho phần Related Work | 🔴 |
| 5 | Ryu & Kim (2024), *A Closer Look at Efficient Inference Methods: A Survey of Speculative Decoding* (arXiv:2411.13157) | Survey bổ sung, góc nhìn hệ thống | 🟡 |
| 6 | Yan, Agarwal, Venkataraman (2025), NAACL, *Decoding Speculative Decoding* | Phân tích thực nghiệm các yếu tố ảnh hưởng tới speedup thật (không chỉ acceptance rate) — rất hữu ích cho phần decomposition draft/verify cost mà đề cương mục 14 đã dự tính | 🔴 |

## 2.2. Self-speculative decoding & layer-skipping (nhánh trung tâm của đề tài)

| # | Công trình | Đóng góp cốt lõi | Mức độ |
|---|---|---|---|
| 7 | Zhang et al. (2024), ACL, *Draft & Verify* (arXiv:2309.08168) | Người khởi xướng self-speculative bằng layer-skip training-free, chọn layer bằng Bayesian optimization offline | 🔴 |
| 8 | Elhoushi et al. (Meta, 2024), ACL, *LayerSkip* (arXiv:2404.16710) | Early-exit + layer dropout training, self-spec bằng early exit thay vì skip giữa | 🔴 |
| 9 | Liu et al. (Huawei, 2024), NeurIPS, *Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting* (arXiv:2404.18911) | Adapter nhỏ (1 attention + 2 norm) bắc cầu từ layer nông sang layer cuối; early-exit cả trong lúc draft để tránh lãng phí trên token khó | 🔴 |
| 10 | Chen et al. (2025), ACL, *CLaSp: In-Context Layer Skip* (arXiv:2505.24196) | Layer-skip chọn động bằng dynamic programming dựa trên hidden state của bước verify gần nhất | 🔴 |
| 11 | Cha, Kim, Han, Yang, Han (2026), ICML, *KnapSpec* (arXiv:2602.20217) | **Quan trọng nhất cần đối chiếu**: tách riêng Attention/MLP, mô hình hóa latency theo context length, giải bài toán chọn layer bằng knapsack 0/1 + DP song song; chứng minh cosine similarity là proxy toán học hợp lý cho acceptance rate | 🔴🔴 |
| 12 | Wen & Feng (2026), Findings ACL, *SpecBound* (arXiv:2604.12247) | Annealed confidence threshold để chống overconfidence ở layer nông; bound động độ dài speculation theo độ khó token; đạt 2.33× | 🔴 |
| 13 | *Component-Aware Self-Speculative Decoding in Hybrid Language Models* (2026, arXiv:2605.01106) | Đề cập thêm SWIFT (layer sparsity theo input) và ConfLayers (confidence-based selection) — 2 nhánh nhỏ hơn cùng họ | 🟡 |
| 14 | Huang & Wen (2026), *Quasar: Quantized Self-Speculative Acceleration* (arXiv:2603.01399) | Nhận diện đúng: sau khi giảm chi phí draft, **verify trở thành bottleneck bandwidth-bound** — dùng low-bit quantization riêng cho verify. Rất liên quan tới GPU 6GB của bạn | 🔴 |
| 15 | Miao et al. (2024), ASPLOS, *QSpec: Speculative Decoding with Complementary Quantization Schemes* (arXiv:2410.11305) | Chuyển đổi lossless giữa 2 chế độ lượng tử hóa (draft nhanh W4A4, verify chính xác W4A16) trên cùng một bộ trọng số | 🟡 |

## 2.3. Đo độ dư thừa/tương đồng giữa các layer (nền tảng cho phần CKA)

| # | Công trình | Đóng góp cốt lõi | Mức độ |
|---|---|---|---|
| 16 | Kornblith et al. (2019), ICML, *Similarity of Neural Network Representations Revisited* | Nguồn gốc công thức CKA — trích dẫn bắt buộc | 🔴 |
| 17 | Men et al. (2024), ACL, *ShortGPT* (arXiv:2403.03853) | Block Influence (BI) — độ đo thay thế CKA, đơn giản hơn, đã có nhiều baseline so sánh trong literature | 🔴 |
| 18 | *Sliding-Window Merging for Compacting Patch-Redundant Layers in LLMs* (2025, arXiv:2502.19159) | Dùng CKA trực tiếp, phát hiện hiện tượng dư thừa dạng "patch" giữa các layer liên tiếp trong reproducing kernel Hilbert space | 🔴 |

## 2.4. Điều chỉnh động độ dài speculation (K) và runtime adaptation

| # | Công trình | Đóng góp cốt lõi | Mức độ |
|---|---|---|---|
| 19 | Mamou et al. (2024), *Dynamic Speculation Lookahead* (arXiv:2405.04304) | Baseline heuristic đơn giản cho adaptive K — nên implement làm baseline tối thiểu | 🔴 |
| 20 | Liu et al. (2024/2025), *PEARL: Parallel Speculative Decoding with Adaptive Draft Length* (arXiv:2408.11850) | Draft length thích ứng dạng song song | 🟡 |
| 21 | *AdaptiveSD: Stability-Aware Runtime-Adaptive Speculative Decoding for CPU-Constrained Inference* (2026, arXiv:2607.03876) | Rất gần bối cảnh phần cứng consumer của bạn — multi-policy orchestration | 🔴 |
| 22 | *CoVSpec* (2026, arXiv:2605.02218) | Margin-based gating + EMA acceptance rate cho adaptive length trong bối cảnh device-edge | 🟡 |
| 23 | Hou et al. (2025), ICML, *BanditSpec* | Xem cấu hình speculative decoding như bài toán online learning/bandit | 🔴 |
| 24 | Liu, Park, Shen (2025), ACL, *A Drop-In Solution for On-the-Fly Adaptation of Speculative Decoding* | Nhấn mạnh ảnh hưởng đồng thời của task/hardware/model tới cấu hình tối ưu | 🟡 |

## 2.5. Hardware-aware, energy-aware, và triển khai trên thiết bị hạn chế tài nguyên

| # | Công trình | Đóng góp cốt lõi | Mức độ |
|---|---|---|---|
| 25 | *GELATO: Generative Entropy- and Lyapunov-based Adaptive Token Offloading* (2026, arXiv:2605.10124) | Dùng lý thuyết điều khiển Lyapunov cho device-edge speculative — kỹ thuật điều khiển đáng tham khảo cho HECC | 🟡 |
| 26 | *SLED: Speculative LLM Decoding for Efficient Edge Serving* (2025, arXiv:2506.09397) | Định hướng lại speculative decoding cho edge serving đa thiết bị | 🟡 |
| 27 | *Delay-Adaptive Speculation Control for Low-Latency Edge-Cloud LLM Inference* (2026, arXiv:2606.20591) | Phân tích closed-form trade-off độ dài draft + regret bound cho online learning dưới delay ngẫu nhiên | 🟡 |
| 28 | Xu et al. (2026), *CAS-Spec: Cascade Adaptive Self-Speculative Decoding* (NeurIPS 2025) | Self-spec dạng cascade, chiến lược suy luận chuyển đổi động | 🟡 |
| 29 | *Nightjar: Dynamic Adaptive Speculative Decoding for LLM Serving* (2026, Journal of Systems Architecture) | Resource-aware trong bối cảnh serving/load variation — đối tượng khác (server) nhưng phương pháp luận điều khiển đáng tham khảo | 🟡 |
| 30 | Ryabinin et al. (2024), *SpecExec: Massively Parallel Speculative Decoding* (arXiv:2406.02532) | Speculative decoding với offloading trên consumer GPU, 4–18× speedup — kỹ thuật offload đáng cân nhắc khi cần chạy model >3B trên 6GB | 🟡 |

## 2.6. Tính đúng (exactness) và non-determinism dưới quantization — **mảng đề cương chưa đề cập, cần bổ sung ngay**

| # | Công trình | Đóng góp cốt lõi | Mức độ |
|---|---|---|---|
| 31 | Gond, Kamath, Ramjee, Panwar (Microsoft Research, 2026), *LLM-42: Enabling Determinism in LLM Inference with Verified Speculation* (arXiv:2601.17768) | Dùng chính ý tưởng speculative (decode-verify-rollback) để **đảm bảo tính xác định** dưới dynamic batching — góc nhìn đảo ngược rất thú vị, đáng trích dẫn trực tiếp trong phần Method của bạn | 🔴🔴 |
| 32 | *Component-Aware Self-Speculative Decoding in Hybrid Language Models* (2026, arXiv:2605.01106) | Báo cáo match rate 90–96% (không phải 100%) do non-determinism ở bf16, quy về ngưỡng chênh lệch logit top-1/top-2 dưới ~1e-2 | 🔴🔴 |
| 33 | *Batch Speculative Decoding Done Right* (2025/2026, arXiv:2510.22876) | Phương pháp luận validate exactness bằng exact-match + partial-match, quy đúng ~5% lệch còn lại về non-determinism và tie-breaking trong argmax — nên copy nguyên phương pháp luận này | 🔴🔴 |
| 34 | *Alignment Collapse Under KV Cache Quantization: Diagnosis and Mitigation* (2026, arXiv:2606.09864) | KV-cache quantization làm trôi trạng thái nội bộ, ảnh hưởng trực tiếp tới acceptance rate — liên quan nếu sau này bạn quantize cả KV cache để tiết kiệm thêm VRAM | 🟡 |

## 2.7. Verification song song/dạng cây và hệ thống serving (bối cảnh mở rộng, không bắt buộc cho bản đầu)

| # | Công trình | Đóng góp cốt lõi | Mức độ |
|---|---|---|---|
| 35 | Cai et al. (2024), ICML, *Medusa* | Nhiều decode head thay cho draft model độc lập | 🟢 |
| 36 | Li et al. (2024/2025), *EAGLE / EAGLE-2 / EAGLE-3* | Feature-level uncertainty, dynamic draft tree — chuẩn công nghiệp hiện nay (vLLM, TensorRT-LLM đều hỗ trợ) | 🟢 |
| 37 | Miao et al. (2024), ASPLOS, *SpecInfer* | Tree-based speculative inference + verification, nền tảng cho batch serving | 🟢 |
| 38 | *TETRIS: Optimal Draft Token Selection for Batch Speculative Decoding* (2025, arXiv:2502.15197) | Tối ưu lựa chọn token nháp khi serving theo batch | 🟢 |
| 39 | Spector & Re (2023), *Staged Speculative Decoding* (arXiv:2308.04623) | Speculative decoding nhiều tầng | 🟢 |

## 2.8. Phương pháp luận đo lường (bắt buộc cho phần Experiments/Implementation của paper)

| # | Công trình | Đóng góp cốt lõi | Mức độ |
|---|---|---|---|
| 40 | Chung et al. (Univ. Michigan, 2025), NeurIPS, *The ML.ENERGY Benchmark* (arXiv:2505.06371) | Chuẩn đo năng lượng inference bằng công cụ Zeus, 4 nguyên tắc thiết kế benchmark năng lượng — nên dùng trực tiếp thay vì tự viết `EnergyTracker` từ đầu | 🔴🔴 |
| 41 | *Watt Counts: Energy-Aware Benchmark for Sustainable LLM Inference* (2026, arXiv:2604.09048) | Benchmark năng lượng trên nhiều loại GPU, có thể dùng làm điểm tham chiếu cross-hardware | 🟡 |
| 42 | Zheng et al. (2023), *Judging LLM-as-a-judge with MT-Bench and Chatbot Arena* | Nguồn bộ prompt MT-Bench — dùng làm workload đánh giá chất lượng output khi generation length thay đổi theo cấu hình | 🟢 |
| 43 | Kwon et al. (2023), SOSP, *PagedAttention/vLLM* | Nền tảng hệ thống serving hiện đại — nên dùng làm baseline production-grade khi so sánh throughput thực tế | 🟢 |

---

# Phần 3 — Khoảng trống nghiên cứu thực sự còn lại (sau khi đối chiếu toàn bộ Phần 2)

Sau khi rà soát ~43 công trình ở trên, có thể kết luận: **"self-speculative decoding không tốn thêm VRAM" không còn là câu hỏi mở, và ngay cả "hardware-aware layer selection" (KnapSpec) và "adaptive K theo entropy/confidence" (SVIP, AdaptiveSD, SpecBound) cũng đã có lời giải chất lượng cao.** Ba khoảng trống còn khả thi, xếp theo độ khả thi giảm dần cho một đề tài quy mô luận văn/đồ án:

**A. Multi-objective Pareto control trên đúng một thiết bị laptop 6GB, không phải edge-cloud.** Toàn bộ nhóm hardware-aware hiện có (GELATO, SLED, Delay-Adaptive, Nightjar) đều nhắm vào bối cảnh **device–edge/serving nhiều người dùng**, có kênh truyền, có server. Chưa có công trình nào tối ưu đồng thời VRAM cứng + power budget + thermal throttling **trên một GPU laptop đơn lẻ, offline, không server**. Đây là khoảng trống thật, phù hợp với đúng những gì `hardware_controller.py` của bạn định làm — nhưng phải sửa xong Bug #1, #2 và implement KnapSpec làm baseline thật trước khi khẳng định.

**B. Kết hợp exactness-under-quantization với self-speculative layer-skip.** Nhóm LLM-42/Component-Aware SSD mới xuất hiện năm 2026, xử lý non-determinism nói chung; chưa ai xử lý nó **cụ thể trong bối cảnh self-speculative layer-skip có draft và verify chạy qua số layer khác nhau** — đây là câu hỏi hẹp nhưng mới, và trực tiếp là nguyên nhân Bug #3 của bạn. Biến bug thành nghiên cứu: đo và mô hình hóa mối quan hệ giữa (số layer bị skip, bit quantization, chênh lệch shape draft/verify) và tỷ lệ divergence.

**C. Systems study thuần túy (không claim SOTA).** Nếu (A) và (B) không ra được kết quả đủ mạnh, một bài "systems/empirical study" đo đạc kỹ, trung thực, có ablation đầy đủ trên GPU 6GB — bám sát đúng tinh thần mục 18.3 của đề cương ("nếu không tìm được lợi thế rõ ràng, paper trung thực vẫn có giá trị") — vẫn có thể phù hợp cho venue dạng workshop hoặc journal hệ thống tầm trung.

---

# Phần 4 — Kế hoạch kỹ thuật chi tiết

## 4.1. Giai đoạn 0 — Sửa nền tảng (2–3 tuần, không có ngoại lệ)

1. Viết `tests/test_exactness.py`: so token-by-token `self_speculative_generate` vs `vanilla_generate` trên ≥20 prompt cố định, temperature=0, greedy. Không chạy benchmark tốc độ nào khác cho tới khi test này minh bạch về tỷ lệ match (dùng đúng phương pháp luận exact-match + partial-match của tài liệu #33).
2. Implement `kv_cache.py` thật: draft cache persist qua các chu kỳ (chỉ tính phần token mới); verify dùng cache tăng dần thay vì recompute toàn bộ. Đây là thay đổi kỹ thuật đơn lẻ có khả năng đảo ngược hoàn toàn bảng ablation hiện tại.
3. Sửa `LayerManager` để reindex `layer_idx` khi hoán đổi layer list.
4. Thay hằng số hard-code trong `hardware_controller.py` bằng số đo thực từ `profiling/latency.py` cho từng candidate config.
5. Chạy lại toàn bộ `experiments/00→07`. Nếu vẫn có `speedup < 1.0`, đó là finding khoa học thật, không phải bug — lúc đó mới bắt đầu Giai đoạn 1.

## 4.2. Giai đoạn 1 — Baseline cạnh tranh thật (4–6 tuần)

- Implement **KnapSpec** (tài liệu #11) làm baseline thật trên đúng RTX 4050 — không dùng số trong paper gốc của họ vì khác hardware.
- Implement **SpecBound** (tài liệu #12) và **Dynamic Speculation Lookahead** (tài liệu #19) làm baseline cho phần adaptive K.
- Implement **Draft & Verify** gốc (Bayesian layer search, có code công khai) để đối chiếu công bằng với hướng CKA.
- Áp dụng phương pháp luận đo năng lượng của **ML.ENERGY/Zeus** (tài liệu #40) thay vì tự viết `EnergyTracker` — vừa tiết kiệm công vừa tăng độ tin cậy khi reviewer đối chiếu.
- Mở rộng model sang Llama-3.2-3B-Instruct như kế hoạch ban đầu; benchmark trên Spec-Bench hoặc MT-Bench + HumanEval + GSM8K để có nhiều loại workload.

## 4.3. Giai đoạn 2 — Tìm câu chuyện paper (4–6 tuần)

Theo đúng tiêu chí mục 18.3 của đề cương: HECC chỉ là contribution nếu Pareto frontier (throughput–energy–VRAM) của nó **strictly dominate** KnapSpec/SpecBound dưới ràng buộc VRAM cứng 6GB, ở ít nhất một vùng vận hành mà baseline không làm được (ví dụ: VRAM headroom < 500MB, hoặc power bị giới hạn để mô phỏng chạy pin). Nếu không tìm được vùng đó, pivot sang hướng B hoặc C ở Phần 3.

## 4.4. Giai đoạn 3 (song song) — Demo sản phẩm thực tế

- Đóng gói `zassd` thành drop-in cho `model.generate()` của HuggingFace `transformers`, tương tự cách LayerSkip công bố qua tham số `assistant_early_exit`.
- Cân nhắc implement như một **custom proposer method trong vLLM** (vLLM đã hỗ trợ sẵn kiến trúc proposer–verifier có thể mở rộng — xem tài liệu mục 2.8/#43) — đây là con đường ngắn nhất để có một demo đạt chất lượng production, dùng được cho việc trình bày trước hội đồng/nhà tuyển dụng.
- Web demo nhỏ (FastAPI + streaming) chạy trên máy 6GB, hiển thị real-time tok/s, VRAM, energy/token, chế độ HECC đang chọn (MAX-THROUGHPUT/MIN-ENERGY/SAFE-MEMORY) — hiện thực hóa trực quan ý tưởng Pareto mode switching.
- Container hóa (Docker, pin CUDA/driver) để tái lập đúng số liệu — hỗ trợ trực tiếp yêu cầu reproducibility khi nộp paper.

---

# Phần 5 — Chiến lược venue cho A*/Q1

| Loại venue | Ví dụ cụ thể | Phù hợp khi nào |
|---|---|---|
| Hội nghị hệ thống ML (A*) | MLSys, NeurIPS (Efficient ML track), ICML | Nếu hướng A (Pareto control trên 6GB) cho kết quả mạnh, rõ ràng hơn baseline |
| Hội nghị NLP (A*) | ACL/EMNLP (main hoặc Findings), đúng venue mà phần lớn tài liệu tham khảo ở trên đã công bố | Nếu đóng góp nghiêng về thuật toán layer-selection/adaptive-K hơn là hệ thống |
| Hội nghị kiến trúc máy tính (A*) | ASPLOS, ISCA, MICRO | Nếu mở rộng sang đo đạc/mô hình hóa phần cứng sâu hơn (thermal, power draw chi tiết theo kernel) |
| Tạp chí Q1 | IEEE TPDS, ACM TACO, ACM TOCS, Journal of Systems Architecture (nơi Nightjar công bố) | Khi cần một venue có review dài hơi hơn cho phần systems study (hướng C) |

Lưu ý quan trọng: với một đề tài quy mô cá nhân/luận văn trên một GPU laptop, **workshop tại các hội nghị lớn** (ví dụ ES-FoMo tại ICML/NeurIPS, Efficient NLP workshop tại ACL) là bước đệm thực tế và hợp lý trước khi nhắm thẳng main track A*.

---

# Phần 6 — Danh sách hành động ưu tiên (tổng hợp)

1. [Tuần 1] Viết và chạy `test_exactness.py` — biết chính xác quy mô của Bug #3.
2. [Tuần 1–3] Implement KV cache reuse thật cho cả draft và verify — kỳ vọng đảo ngược bảng ablation.
3. [Tuần 2–3] Sửa `LayerManager.layer_idx` và hardware_controller's hard-coded constants.
4. [Tuần 3–4] Đọc kỹ và implement KnapSpec + SpecBound làm baseline thật.
5. [Tuần 4–6] Áp dụng phương pháp luận ML.ENERGY cho phần đo năng lượng.
6. [Tuần 5–8] Mở rộng model/workload, chạy lại toàn bộ ablation, tìm vùng Pareto-dominance.
7. [Song song] Bắt đầu bản demo drop-in cho `transformers.generate()`.
8. [Sau khi có kết quả ổn định] Chốt câu chuyện paper theo 1 trong 3 hướng ở Phần 3, viết theo khung mục 18.2 của đề cương.

---

# Phụ lục — Bảng tổng hợp tài liệu tham khảo (liên kết đầy đủ)

*(Đánh số trùng với Phần 2 ở trên để tiện tra cứu)*

1. Stern, Shazeer, Uszkoreit (2018). *Blockwise Parallel Decoding for Deep Autoregressive Models*. NeurIPS. https://arxiv.org/abs/1811.03115
2. Leviathan, Kalman, Matias (2023). *Fast Inference from Transformers via Speculative Decoding*. ICML. https://proceedings.mlr.press/v202/leviathan23a.html
3. Chen et al. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling*. arXiv:2302.01318
4. Xia et al. (2024). *Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding*. Findings of ACL 2024. https://arxiv.org/abs/2401.07851
5. Ryu & Kim (2024). *A Closer Look at Efficient Inference Methods: A Survey of Speculative Decoding*. arXiv:2411.13157
6. Yan, Agarwal, Venkataraman (2025). *Decoding Speculative Decoding*. NAACL 2025.
7. Zhang, Wang, Li, Shou, Chen, Chen, Mehrotra (2024). *Draft & Verify: Lossless LLM Acceleration via Self-Speculative Decoding*. ACL 2024. https://arxiv.org/abs/2309.08168
8. Elhoushi et al. (2024). *LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding*. ACL 2024. https://arxiv.org/abs/2404.16710 — code: github.com/facebookresearch/LayerSkip
9. Liu, Tang, Liu, Ni, Tang, Han, Wang (2024). *Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting*. NeurIPS 2024. https://arxiv.org/abs/2404.18911 — code: github.com/Equationliu/Kangaroo
10. Chen, Shan, Wang, Wang, Liu, Luo, Wang, Alinejad-Rokny, Yang (2025). *CLaSp: In-Context Layer Skip for Self-Speculative Decoding*. ACL 2025. https://arxiv.org/abs/2505.24196
11. Cha, Kim, Han, Yang, Han (2026). *KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem*. ICML 2026. https://arxiv.org/abs/2602.20217
12. Wen & Feng (2026). *SpecBound: Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration*. Findings of ACL 2026. https://arxiv.org/abs/2604.12247
13. *Component-Aware Self-Speculative Decoding in Hybrid Language Models* (2026). arXiv:2605.01106
14. Huang & Wen (2026). *Quasar: Quantized Self-Speculative Acceleration for Rapid Inference via Memory-Efficient Verification*. arXiv:2603.01399
15. Miao et al. (2024). *QSpec: Speculative Decoding with Complementary Quantization Schemes*. arXiv:2410.11305
16. Kornblith, Norouzi, Lee, Hinton (2019). *Similarity of Neural Network Representations Revisited*. ICML 2019.
17. Men et al. (2024). *ShortGPT: Layers in LLMs Are More Redundant Than You Expect*. arXiv:2403.03853
18. *Sliding-Window Merging for Compacting Patch-Redundant Layers in LLMs* (2025). arXiv:2502.19159
19. Mamou, Pereg, Korat, Berchansky, Timor, Wasserblat, Schwartz (2024). *Dynamic Speculation Lookahead Accelerates Speculative Decoding of LLMs*. arXiv:2405.04304
20. Liu, Li, Lv, Liu, Zhu, Hu, Sun (2025). *PEARL: Parallel Speculative Decoding with Adaptive Draft Length*. arXiv:2408.11850
21. *AdaptiveSD: A Stability-Aware, Runtime-Adaptive Speculative Decoding Framework for CPU-Constrained LLM Inference* (2026). arXiv:2607.03876
22. *CoVSpec: Efficient Device-Edge Co-Inference for Vision-Language Models via Speculative Decoding* (2026). arXiv:2605.02218
23. Hou et al. (2025). *BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms*. ICML 2025.
24. Liu, Park, Shen (2025). *A Drop-In Solution for On-the-Fly Adaptation of Speculative Decoding in LLMs*. ACL 2025.
25. *GELATO: Generative Entropy- and Lyapunov-based Adaptive Token Offloading for Device-Edge Speculative LLM Inference* (2026). arXiv:2605.10124
26. *SLED: A Speculative LLM Decoding Framework for Efficient Edge Serving* (2025). arXiv:2506.09397
27. *Delay-Adaptive Speculation Control for Low-Latency Edge-Cloud LLM Inference* (2026). arXiv:2606.20591
28. Ning et al. (2025). *CAS-Spec: Cascade Adaptive Self-Speculative Decoding*. NeurIPS 2025.
29. *Nightjar: Dynamic Adaptive Speculative Decoding for LLM Serving*. Journal of Systems Architecture, Vol. 178 (2026). https://doi.org/10.1016/j.sysarc.2026.103889
30. Ryabinin et al. (2024). *SpecExec: Massively Parallel Speculative Decoding*. arXiv:2406.02532
31. Gond, Kamath, Ramjee, Panwar (Microsoft Research, 2026). *LLM-42: Enabling Determinism in LLM Inference with Verified Speculation*. arXiv:2601.17768
32. (= tài liệu 13 ở trên, phần non-determinism)
33. *Batch Speculative Decoding Done Right* (2025/2026). arXiv:2510.22876
34. *Alignment Collapse Under KV Cache Quantization: Diagnosis and Mitigation* (2026). arXiv:2606.09864
35. Cai et al. (2024). *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*. ICML 2024.
36. Li, Wei, Zhang, Zhang (2024/2025). *EAGLE / EAGLE-2 / EAGLE-3*. ICML 2024 / EMNLP 2024 / arXiv:2503.01840
37. Miao et al. (2024). *SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference and Verification*. ASPLOS 2024.
38. *TETRIS: Optimal Draft Token Selection for Batch Speculative Decoding* (2025). arXiv:2502.15197
39. Spector & Re (2023). *Accelerating LLM Inference with Staged Speculative Decoding*. arXiv:2308.04623
40. Chung, Liu, Ma, Wu, Kweon, Xia, Wu, Chowdhury (2025). *The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization*. NeurIPS 2025. https://arxiv.org/abs/2505.06371 — leaderboard: ml.energy/leaderboard
41. *Watt Counts: Energy-Aware Benchmark for Sustainable LLM Inference* (2026). arXiv:2604.09048
42. Zheng et al. (2023). *Judging LLM-as-a-judge with MT-Bench and Chatbot Arena*. NeurIPS 2023.
43. Kwon et al. (2023). *Efficient Memory Management for LLM Serving with PagedAttention*. SOSP 2023 (nền tảng vLLM).

**Ghi chú về độ tin cậy trích dẫn:** Các công trình đánh dấu năm 2026 (bao gồm KnapSpec, SpecBound, một số bài trong nhóm exactness/determinism) đã được xác minh chéo trực tiếp qua tra cứu thời điểm biên soạn báo cáo này (tháng 9/2026) — đều là arXiv ID/venue thật, không phải suy diễn. Một số công trình được liệt kê dựa trên tài liệu tham khảo gốc trong đề cương của dự án (mục DEL, phần SVIP dạng "Draft Model Knows When to Stop") chưa được tác giả báo cáo này tự tra cứu độc lập lần hai trong phiên làm việc này; nên xác minh lại DOI/link trước khi trích dẫn chính thức trong bản thảo cuối.
