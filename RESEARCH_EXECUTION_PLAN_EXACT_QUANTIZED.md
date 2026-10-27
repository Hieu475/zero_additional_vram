# Kế hoạch triển khai: exact quantized speculative decoding

**Cập nhật:** 26-10-2026. **Trạng thái:** kế hoạch thực thi, chưa có thuật toán exact được chứng minh cho BNB/HQQ. Đọc cùng [đề cương](RESEARCH_EXACT_QUANTIZED_SPECULATIVE_DECODING.md), [kết quả thăm dò](experiments/exact_quantized_feasibility/FINDINGS.md), [protocol](experiments/exact_quantized_feasibility/PROTOCOL.md) và [nghĩa vụ chứng minh](experiments/exact_quantized_feasibility/PROOF_OBLIGATIONS.md).

**Tiến độ B1:** đã có singleton reference, prototype Qwen2 fixed-row block, hai chiến lược attention (rowwise và padded), pilot 0.5B/3B, paired K=2/4/8 qua hai repeat rounds trên sáu prompt 3B, draft hợp đồng operator và manifest runtime `b1-fixed-row-v2`. Branch audit đã phủ reject tại mọi vị trí draft cho K=2/4/8, pending-token EOS, một accepted EOS draft candidate và budget 256; long-context audit thêm reject mọi vị trí cho K=1/2/4/8 trên ba prompt và hai sessions. Fixed-row probes M tối đa 3/5/9 hiện đã lặp trên hai prompt dài development. Per-prefix active-KV audit mới khớp singleton tại đủ 256 prefix trên `cnndm_002` cho K=2/4/8. Hai prompt cycle bổ sung khớp B1 12/12, BNB 0/12; K=2 dẫn trên một prompt, còn prompt kia K=2/4/8 gần hòa. Kết quả cập nhật nằm trong `FINDINGS.md`. Prototype khớp singleton B1 ở các lượt đã đo, nhưng chưa có proof toàn miền. Các prompt đều là development và một số đã xuất hiện trong thí nghiệm workspace; cần mở rộng và audit thêm trước khi freeze policy.

## 1. Quyết định chiến lược

**Ưu tiên B1:** xây một đường target 4-bit có phép tính từng vị trí và cập nhật KV bất biến khi đổi `M=1..K+1`; dùng chính đường này cho greedy tham chiếu mới và speculative verification. Sau khi chứng minh tương đương trong miền hỗ trợ, đo tốc độ so với cả greedy của kernel mới và greedy BNB gốc. Đây là tuyên bố exact khả thi nhất. **B2** (trùng token greedy BNB gốc) là mục tiêu phụ, chỉ theo đuổi khi có phép tái tạo số học BNB gốc hoặc cận sai số đã chứng minh. Không đổi ngầm tham chiếu B1 thành B2.

**Không dùng margin guard hiện tại làm lõi:** pilot BNB 30 prompt/64 token cho 29/30 exact và tỷ số latency 0.197; replay chiếm trung bình 77.8% thời gian. Pilot 3 prompt/256 token cho 3/3 exact nhưng tỷ số 0.061, replay 94.1%. Các phép trừ chi phí replay trong `FINDINGS.md` chỉ là phản thực tế về chi phí. Kết quả này không loại trừ một guard khác, nhưng không đủ căn cứ tối ưu threshold hiện tại.

**MVP hiện tại:** Qwen2.5-0.5B và 3B NF4 trên RTX 4050, batch 1, `K_max=8`, physical rows=9. Fixed-row prototype đã qua probe M=1..9 trên một prefix/model và pilot đa chu kỳ; `src/zassd/certification/greedy.py` vẫn chỉ là helper có điều kiện, không có bound backend hợp lệ.

## 2. Hiện trạng và các lỗ hổng cần xử lý

| Thành phần | Đã có | Còn thiếu |
|---|---|---|
| Tham chiếu và cohort | B1 contract `ca155f56…b2ca27`; validator và fail-closed accept/reject đã chạy; singleton tokens+KV khớp giữa 2 sessions trên 3 development prompts; GSM8K 300–349 vẫn kín | Mở rộng cohort/provenance và chạy vanilla consistency ở context/dải prompt rộng hơn |
| Sai khác số học | Cohort mở rộng 30 prompt × 7 checkpoint × M=3/5/9 chạy 2 process độc lập: mỗi session 630/630; logit drift 626/630, KV 623/630, candidate argmax 3539/3570; toàn bộ candidate IDs và metrics lặp chính xác ở 630/630. Phi-3.5/GSM8K/M9 outlier được truy đến layer-0 `qkv_proj` (input bằng bitwise, output lệch 0.03125), sau đó sai số row 5 khuếch đại tới 732.75 ở layer 29, tập trung ở MLP `down_proj`. Llama-3.2-1B và Phi-3.5-Mini pinned checkpoints có adapter B1: M=1…9 bitwise probes trên 3 task và pilot 64-token khớp mọi token/KV prefix (192/192 mỗi attention strategy, padded và rowwise). Gate nhận 3 Qwen2 checkpoint, Llama-3.2-1B, Phi-3.5-Mini và OLMo-2-0425-1B; bật thêm Llama-3.2-3B sau probes và pilot. Đây là bằng chứng thực nghiệm hữu hạn, không proof/cache guarantee tổng quát; cohort cũ không held-out. Phân tích: `experiments/exact_quantized_feasibility/crossmodel_shape_repeat_v1/analysis.json`, `experiments/exact_quantized_feasibility/phi35_gsm8k059_m9_error_growth_diagnosis_v1.json`, `experiments/exact_quantized_feasibility/crossmodel_b1_architecture_inventory_v8.json`, `experiments/exact_quantized_feasibility/FINDINGS.md` | B1 pilots đã phủ ba task × 64 token cho Qwen, Llama-1B/3B, Phi-3.5 và OLMo-2. Llama-3B K sweep BF16 tốt nhất K=2/rows3 nhưng chỉ 0.918× BNB; FP16 activations đạt 0.881× và match baseline 6/6, vẫn chưa nhanh hơn. Profiler quy 76% kernel time cho BNB 4-bit GEMM; attention dưới 1%. Ưu tiên prototype linear path giảm chi phí fixed-row mà vẫn giữ exactness, rồi benchmark lại. Sau đó mở cohort/context, audit terminal/EOS/cache rollback, và thử GPU/runtime/backend khác |
| Oracle | Cycle oracle K=1/2/4/8 trên prompt context 397 + 256; paired K=2/4/8 hai session trên prompt contexts 398 và 419; contract tách rows=3/5/9; cache fork/crop/checkpoint/restore đã đo | Mở rộng prompt/context; đo selector/synchronization/replay thực; oracle vẫn hindsight, không phải policy |
| Runtime | Có `b1_exact`, `b1_fixed_row_generate`, `b1_fixed_row_block_generate`, CLI `b1_block_experimental`, cùng rowwise/padded attention; đã đo K=2/4/8 trên sáu prompt qua hai repeat rounds | Mở rộng và provenance-audit development cohort, lặp paired measurements, sau đó tối ưu padded attention mask và linear wrappers |
| Cache | Static cache crop thay logical length; draft fork chia sẻ buffer; active KV hash khớp singleton trên đủ 256 output prefix của một prompt cho K=2/4/8 | Kiểm tra mọi nhánh/terminal và nhiều prompt; xác nhận write không làm lộ suffix bị reject trên toàn miền hỗ trợ |
| Tài liệu | HQQ factorial report đã đối chiếu raw: 0/16 sequence mismatch và 0/16 selected-row argmax divergence; BNB attribution được tách riêng | Hoàn tất bước audit tài liệu; các artifact đã ghi execution commit, model revision, backend và cohort |

`src/zassd/decoding/verification.py` đã giới hạn docstring: token acceptance không tự bảo đảm toàn quỹ đạo trong finite precision; cần logits và KV phù hợp với reference.

## 3. Hợp đồng exact của MVP

Chốt một manifest có hash cho: commit mã, checkpoint revision và hash quantization state, tokenizer/chat template, batch=1, backend/kernel build, PyTorch/CUDA/driver/GPU, attention, deterministic settings, prefill, static cache capacity **và logical length**, position IDs/mask, tie/NaN/EOS/budget rule, `K_max` và context domain. Cùng token input phải tạo cùng state đầu. `reference_contract_id` là hash canonical của manifest; decoder từ chối certified mode nếu runtime không khớp.

Tách rõ ba claim:

1. **B1-token:** speculative sinh token ID giống greedy tuần tự *của kernel mới* cho mọi input trong miền contract.
2. **B1-state:** sau mỗi token được commit, trạng thái liên quan đến forward kế tiếp giống bitwise hoặc nằm trong cận đã chứng minh.
3. **B2-token:** giống greedy BNB gốc; chỉ công bố khi chứng minh/kiểm chứng điều kiện bổ sung, không suy từ B1.

Một test `0/N` mismatch chỉ là kiểm tra implementation. Khi đổi GPU, version hoặc kernel schedule, contract cũ không tự chuyển sang môi trường mới.

## 4. Công việc theo cổng quyết định

### G0 — Khóa dữ liệu và phép đo

**Việc làm**

1. **Hoàn tất audit HQQ:** so sánh 16 raw JSONL rows với token IDs và summary; xác nhận 0/16 sequence mismatch, 0/16 selected-row argmax divergence, cohort IDs khớp. Report và đề cương đã tách 15/16, 1/16 attribution sang factorial BNB khác protocol; provenance HQQ giữ execution commit, model revision và backend.
2. **Đã tạo contract tham chiếu B1** `experiments/exact_quantized_feasibility/b1_reference_contract_v3.json` và `scripts/build_b1_reference_contract.py`; contract hash `ca155f567ba080347f765335e8999b0256abafe87c88c0263ac38344b2b2ca27`. Contract khóa Qwen2.5-3B NF4, padded rows=3 (K≤2), static KV 2048 và runtime/code hashes; manifest nguồn đã xác minh.
3. **Đã nối và chạy validator fail-closed:** `--reference-contract` so sánh runtime/model/tokenizer/weights/operator/cache/source identity trước khi ghi manifest/decode; contract v3 chấp nhận đúng runtime và từ chối `rowwise` khi contract chốt `padded`, không ghi manifest/rows.
4. Đã lặp singleton/reference state trên ba prompt development trong hai process session: 3/3 chuỗi và active-KV trace khớp; block khớp singleton 6/6 lượt. Mở rộng thêm cohort trước khi đóng G0. Giữ diagnostic mismatch riêng; giữ nguyên GSM8K 300–349 cho held-out. Không xem kết quả held-out khi thiết kế kernel/policy. Freeze thêm task/họ model độc lập **trước** final evaluation.
5. Đo paired latency và VRAM trên cùng cấu hình cho vanilla, PLD batch, PLD sequential, guard; ít nhất hai process session development, xoay thứ tự phương pháp. Lưu per-token first divergence, per-cycle timing, cache logical lengths, hashes.

**Đầu ra:** contract B1, validator fail-closed, independent repeat audit và cycle oracle sơ bộ đã có; còn `cohorts.json`, baseline suite rộng hơn và consistency audit trên nhiều prompt/context. **Qua cổng khi:** manifest tái tạo được, vanilla repeat trùng token/state theo contract, mọi baseline có provenance và metric cùng định nghĩa. Nếu vanilla không ổn định, cố định kernel/schedule hoặc thu hẹp contract trước.

### G1 — Xác định trần lợi ích và điểm sai đầu tiên

**Việc làm**

1. Thu trace development `K=1,2,4,8` và context ngắn/dài, output 64/256. Ở mỗi chu kỳ lưu prefix ID, draft tokens, target logits/argmax, trạng thái KV trước/sau, vị trí reject/bonus/EOS và timing draft/verify/commit.
2. Tạo oracle offline trên **cùng một quỹ đạo tham chiếu**: mô phỏng chọn `K`, verify, replay/checkpoint theo chu kỳ; chỉ dùng tương lai để tính trần, không làm policy. Cost table lấy từ các thao tác đo trên GPU ở cùng context/K, cộng copy, allocation, synchronization và cache restore. Kiểm tra tổng cost dự báo với trace thực, báo khoảng sai số.
3. Dùng cache clone độc lập và cùng token prefix để tìm operator đầu tiên có input bằng nhau nhưng output khác giữa M=1 và block; chạy qua q/k/v/o projection, RMSNorm, RoPE, attention, MLP, LM head, cache write. Không gọi logit norm khác 0 là argmax flip.
4. Lặp nhiều chu kỳ kể cả token vẫn trùng để định lượng KV drift trước first token divergence. Cố định mask, position và logical length trong mọi cặp so sánh.

**Đầu ra:** `cycle_oracle.jsonl`, cost calibration, bản đồ first divergent operator/layer/context/M. **Qua cổng khi:** phân rã thời gian giải thích được latency quan sát, và có một lựa chọn kernel/contract cụ thể để thử. Nếu oracle exact tốt nhất không vượt greedy kernel mới trên workload mục tiêu sau khi tính đủ chi phí, dừng tối ưu policy; chuyển sang đóng góp đo lường hoặc workload khác.

### G2 — Prototype kernel bất biến và chứng minh cục bộ

**Việc làm**

1. Viết bảng operator contract: input/layout/dtype, dequant NF4, reduction order, rounding/FMA, dispatch, output, cache write, miền shape. Bắt đầu với 4-bit linear `q_proj` vì audit đã chứng minh đây là nguồn sai khác sớm; sau đó phủ toàn bộ linear và các reduction khác. Không dùng một vòng gọi singleton-linear theo row làm giải pháp cuối: probe hiện tại cho thấy logits/KV vẫn khác.
2. Triển khai kernel/reference path nhỏ có cùng reduction schedule cho `M=1..5`, cùng logical token prefix. Có thể dùng fallback mỗi row để tạo mốc đúng nhưng phải đo chi phí; đường tăng tốc cần xử lý nhiều row mà vẫn giữ schedule và causal dependence.
3. Chứng minh bằng contract operator rằng một hàng block và single-token forward trên cùng prefix cho cùng output, rồi quy nạp qua layer và KV update. Kiểm tra attention khi KV cũ và candidate KV cùng tham gia; kiểm tra cache capacity, mask, RoPE và cả bonus row.
4. Property/stress tests: các `M`, vị trí reject, prefix/context boundary, capacity khác nhau, EOS, near-tie, NaN/Inf policy, nhiều chu kỳ; đối chiếu bytes trạng thái liên quan sau mỗi commit. Test dùng được trên model nhỏ trước GPU rộng hơn.

**Đầu ra:** source kernel, operator contract và proof note theo từng phép toán, trace bitwise/state equality, benchmark microkernel. **Qua cổng khi:** không còn operator thiếu contract trong đường B1; các stress test không có phản ví dụ; thời gian verify `K` cộng chi phí quản lý nhỏ hơn hoặc có cơ hội nhỏ hơn `K` lần greedy forward trên workload PLD hit. Nếu kernel đúng nhưng quá chậm, phân tích bottleneck rồi tối ưu schedule; không gọi exact speedup khi chưa đo request latency.

### G3 — Runtime certified fail-closed

**Việc làm**

1. Tạo `ReferenceContract`, `VerifierCapability`, `CanonicalCheckpoint`, `TemporaryBranch` và trace certificate trong `src/zassd/certification/`. Backend khai báo contract hỗ trợ; thiếu contract thì từ chối certified mode hoặc chạy greedy tham chiếu rõ ràng.
2. Tách `verify_block` khỏi `commit_prefix`: verify chỉ ghi vào nhánh tạm. Canonical state không bị sửa do draft, verify hoặc reject. Sau quyết định, commit đúng prefix và state tương ứng, hoặc restore/replay từ checkpoint được chứng nhận.
3. Dùng state machine có bất biến: emitted prefix đúng; pending token đúng; KV/positions/mask hợp lệ. Trước khi emit pending token, chứng nhận token ấy. Guard hiện tại có ca `cnndm_006` chọn pending token từ cache lệch rồi replay quá muộn; test hồi quy chính xác ca này.
4. Định nghĩa rõ hành động cho accepted draft, first reject, bonus, EOS và output budget. Replay từ **canonical checkpoint** rồi tái chọn pending token; không replay một token trên KV đã lệch. Ghi reason, contract ID, prefix length, certificate/bitwise evidence, replay count và cost.
5. Với hướng B1 có zero state error, certificate là sự tương đương của kernel contract, không phải threshold margin. Giữ helper margin cho nhánh A nghiên cứu riêng; không cho số đo calibration đóng vai trò global bound.

**Đầu ra:** `certified` mode, tests state-machine/cache, trace per-cycle, fallback khi contract không áp dụng. **Qua cổng khi:** trên mọi nhánh stress, token/state sau mỗi bước tương đương greedy B1; fail-closed đúng khi backend/version/shape ngoài contract; peak VRAM trong ngân sách thiết bị đã khai báo.

### G4 — Hiệu năng, policy và B2 tùy điều kiện

**Việc làm**

1. Profile `draft`, `verify`, `certificate`, checkpoint/copy, commit, replay, synchronization, prefill và end-to-end. Tối ưu trước các cost lớn thực đo; thử `K` và PLD candidate source trên development.
2. Nếu B1 batched verifier đúng và nhanh, policy chọn `K` bằng cost model đo thật; so sánh fixed K tối ưu. Đo riêng workload PLD hit/miss. Không dùng tỷ lệ accept trên held-out để chỉnh policy.
3. Chỉ thử B2 sau khi có kế hoạch tái tạo lịch số học BNB gốc hoặc bound state/logit bao trùm thực thi BNB. Nếu không có, báo B1 trung thực cùng tốc độ của greedy mới và greedy gốc.
4. Nếu nhánh A có bound đủ chặt, thêm interval/bound propagation và cost-aware fallback trên nhánh tạm; chứng minh quy nạp qua KV. Nếu bound nở khiến replay thường xuyên, dừng nhánh A.

**Đầu ra:** Pareto latency–VRAM–certificate rate, policy đã freeze. **Qua cổng khi:** trên development, exact implementation có lợi ích end-to-end lặp lại được ở ít nhất một vùng workload định trước, và overhead trọng số draft bằng 0. Nếu chỉ đúng mà chậm, vẫn giữ kết quả như một prototype correctness; không tuyên bố speedup.

### G5 — Đánh giá ngoài mẫu và công bố

**Việc làm**

1. Freeze code, contract, policy, baseline, prompt hashes và script phân tích **trước** khi chạy GSM8K 300–349. Thêm tập ngoài mẫu khác task và model. Chạy nhiều process session, xoay thứ tự, cùng output budget >=256, một nhánh dài hơn để xem drift.
2. Mỗi backend có greedy tham chiếu riêng. Mở rộng mục tiêu ≥3 họ/kích thước model, ≥3 backend/kernel và ≥2 GPU chỉ khi có contract cho từng cấu hình; nếu không, giới hạn claim và báo phần chưa hỗ trợ. RTX 4050 là GPU hiện có; GPU thứ hai cần môi trường ngoài workspace.
3. Báo full-sequence exact count, first divergence, latency paired và CI theo prompt/session, p50/p95, TPS decode/end-to-end, peak VRAM, copy/checkpoint/replay, certificate/fallback rate; giữ OOM/failure trong mẫu. `0/N` mismatch được báo kèm giới hạn thống kê, không thay định lý.
4. Rà lại prior art trước khi chốt novelty, nhất là [LLM-42](https://arxiv.org/abs/2601.17768) và [batch-invariant kernels](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/); tách đóng góp kernel 4-bit/state contract khỏi verify–rollback vốn đã có.

**Đầu ra:** artifact tái lập, raw rows, scripts CI/figure, paper với theorem giả định minh bạch. **Qua cổng khi:** mọi claim trong abstract có một proof obligation hoặc một bảng đánh giá cùng protocol hỗ trợ trực tiếp.

## 5. Thứ tự sửa mã và các test quan trọng

| Thứ tự | Tệp/nhóm | Thay đổi kiểm tra được |
|---:|---|---|
| 1 | `scripts/run_exact_quantized_feasibility.py`, manifest/cohort scripts | Contract hash, paired trace, session/rotation, không overwrite; giữ held-out kín |
| 2 | `scripts/audit_operator_shape.py`, `scripts/trace_multicycle_numerical_divergence.py` | Cache clone độc lập, first differing operator, drift nhiều chu kỳ |
| 3 | `src/zassd/backends/` | B1 kernel/reference capability và shape contract; tách BNB gốc |
| 4 | `src/zassd/cache/kv_cache.py` | Checkpoint/temporary branch/restore; test tránh aliasing và corruption |
| 5 | `src/zassd/decoding/speculative.py`, `verification.py` | Commit sau chứng nhận, pending/bonus/EOS/reject/fallback; sửa claim docstring |
| 6 | `tests/test_speculative_exactness.py`, `test_kv_cache_invariants.py`, `test_certification_bounds.py` | Invariant mỗi bước, `cnndm_006`, near ties, capacity, mask, nhiều chu kỳ |
| 7 | `experiments/exact_quantized_feasibility/` thư mục run mới | Lưu raw, manifest, analysis, proof evidence; không sửa số pilot cũ |

## 6. Tiêu chuẩn dừng và cách báo kết quả

- **Dừng B1 speedup:** sau tối ưu hợp lý, latency request của exact B1 không vượt greedy B1 ở vùng workload đã chốt, hoặc peak VRAM vượt ngân sách. Công bố tradeoff correctness–performance, không chọn riêng prompt thắng để làm claim tổng quát.
- **Dừng A:** không có bound finite-precision hợp lệ, hoặc certificate rate thấp/chi phí tính bound lớn hơn lợi ích. Threshold từ data chỉ được gọi là heuristic.
- **Dừng B2:** không thể tái tạo hoặc bao cận BNB gốc trên toàn state. B1 vẫn là một kết quả riêng.
- **Phản ví dụ:** bất kỳ sai token/state nào trong miền contract là lỗi contract hoặc implementation; thu nhỏ ca lỗi, sửa hoặc thu hẹp miền, rồi chạy lại toàn bộ validation trước khi dùng held-out.

**Việc cần bắt đầu tiếp:** (1) hoàn thiện reference contract và validator G0 trên development; (2) xây trace/cycle oracle G1; (3) prototype shape-invariant 4-bit linear + attention/state contract G2. HQQ audit đã chốt. Không chạy GSM8K 300–349 ở các bước này.

## 7. Lịch dự kiến và thước đo ưu tiên

Đây là ước lượng cho một người có quyền dùng RTX 4050; thời gian viết kernel phụ thuộc mức hỗ trợ source/compiler và có thể kéo dài. G5 cần thêm GPU độc lập.

| Mốc | Ước lượng | Phụ thuộc | Điều kiện bàn giao |
|---|---:|---|---|
| G0 | 2–4 ngày | Raw traces và môi trường hiện có | Manifest/cohort/baseline nhất quán |
| G1 | 4–7 ngày | G0 | Oracle chu kỳ và first divergent operator map |
| G2 | 2–4 tuần | G1 | Contract phủ hết đường forward + prototype đúng trên miền nhỏ |
| G3 | 1–2 tuần | G2 | Certified runtime và stress suite qua các nhánh |
| G4 | 1–2 tuần | G3 | Policy freeze và số liệu end-to-end development |
| G5 | 1–2 tuần sau khi có GPU thứ hai | G4 | Held-out multi-session và artifact/paper |

**Metric chọn phương án:** ưu tiên paired end-to-end latency theo prompt, báo cả median, geometric mean ratio và CI theo session; tiếp theo là decode TPS, p95, peak VRAM và certificate/fallback rate. Không dùng mean của tỷ số latency một mình để chốt thắng vì prompt ngắn/dài khác nhau. Để đáng đầu tư G2, oracle G1 nên cho thấy dư địa rõ hơn nhiễu đo và overhead kernel dự kiến; mục tiêu nội bộ là **≥1.15×** so với greedy B1 trên vùng PLD-hit đã định nghĩa trước, nhưng đây là cổng đầu tư chứ không phải kết quả hay điều kiện của định lý. Mục tiêu bài hoàn chỉnh là exact B1 có speedup dương với CI không cắt 1 trên vùng workload định trước, không vượt ngân sách VRAM, đồng thời báo trung thực tỷ số so với greedy BNB gốc.
