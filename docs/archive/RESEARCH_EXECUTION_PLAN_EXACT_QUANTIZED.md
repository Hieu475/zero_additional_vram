# Kế hoạch triển khai: exact quantized speculative decoding

**Cập nhật:** 28-10-2026. **Trạng thái:** kế hoạch thực thi, chưa có thuật toán exact được chứng minh cho BNB/HQQ. Đọc cùng [đề cương](RESEARCH_EXACT_QUANTIZED_SPECULATIVE_DECODING.md), [kết quả thăm dò](experiments/exact_quantized_feasibility/FINDINGS.md), [protocol](experiments/exact_quantized_feasibility/PROTOCOL.md) và [nghĩa vụ chứng minh](experiments/exact_quantized_feasibility/PROOF_OBLIGATIONS.md).

**Tiến độ B1:** đã có singleton reference, prototype Qwen2 fixed-row block, hai chiến lược attention (rowwise và padded), pilot 0.5B/3B, paired K=2/4/8 qua hai repeat rounds trên sáu prompt 3B, draft hợp đồng operator và manifest runtime `b1-fixed-row-v2`. Branch audit đã phủ reject tại mọi vị trí draft cho K=2/4/8, pending-token EOS, một accepted EOS draft candidate và budget 256; long-context audit thêm reject mọi vị trí cho K=1/2/4/8 trên ba prompt và hai sessions. Fixed-row probes M tối đa 3/5/9 hiện đã lặp trên hai prompt dài development. Per-prefix active-KV audit mới khớp singleton tại đủ 256 prefix trên `cnndm_002` cho K=2/4/8. Hai prompt cycle bổ sung khớp B1 12/12, BNB 0/12; K=2 dẫn trên một prompt, còn prompt kia K=2/4/8 gần hòa. Kết quả cập nhật nằm trong `FINDINGS.md`. Prototype khớp singleton B1 ở các lượt đã đo, nhưng chưa có proof toàn miền. Các prompt đều là development và một số đã xuất hiện trong thí nghiệm workspace; cần mở rộng và audit thêm trước khi freeze policy.

**B1 parallel-attention cập nhật:** fixed-M RMSNorm/LM head cùng `triton_batched` attention và `fixed_rows` linear đạt exact token/KV trên ba prompt development; chín paired runs cho 1.280× B1 singleton, 0.957× BNB. Schedule linear N=16/one-warp riêng đạt 1.199× B1 và 1.018× BNB qua chín cặp; một process session độc lập trên `cnndm_002` tiếp tục exact B1 nhưng chỉ 0.884× BNB. Hai lần chạy `cnndm_002` nhận 30.2% draft tokens, thấp hơn 63.1% ở `cnndm_001`; đây là đầu mối để kiểm tra proposer, chưa phải kết luận nhân quả. Chưa có BNB speedup ổn định hay B2 state equality. GSM8K 300–349 vẫn sealed.

**Proposer sweep cnndm_002:** runner nhận `--ngram-size`; ba paired rounds mỗi n=2/n=4 giữ B1 exact nhưng không tăng acceptance so với n=3 (n=2: 54/195; n=3/n=4: 57/189). BNB relative latency vẫn 0.878–0.879× cho n=2/4. Giữ n=3; ưu tiên profile target verification thay vì đổi proposer. Xem `FINDINGS.md` và các artifact session 27/28.

**Profile target verification:** trên trace 32 token `cnndm_002`, `_packed_two_rows_kernel` chiếm 450–461 ms self CUDA (6,300–7,812 calls); M=3 LM head 88–108 ms, Triton attention khoảng 20 ms. Đây là profiler diagnostics có event-clear warning, không phải benchmark latency. Ưu tiên giảm NF4 linear cost/launch overhead; xác nhận mọi thay đổi bằng paired end-to-end runs và exact B1 state checks.

**Profile lịch ứng viên R=2:** trace 32 token có 6,300 packed NF4 calls/371.6 ms self CUDA ở block, so với 7,812/461.2 ms ở singleton; projection M=2 là 86.6/107.6 ms. Block dùng 25 forwards, 33 target rows so với 31 singleton forwards/rows. Vẫn là profiler diagnostics có event-clear warning; NF4 linear vẫn chiếm phần chính.

**K=1/R=2 N=16/one-warp — initial 128-token screen:** hai session mỗi prompt exact B1 trên ba prompt development; paired geomean 1.229× vs B1 nhưng 1.014× BNB. BNB-relative result ở `cnndm_002` yếu hơn hai prompt còn lại, dẫn tới các repeat long-budget bên dưới.

**Long-budget K=1/R=2:** ở 256 token, 2 sessions × 3 prompt exact B1 (1,536/1,536 KV prefixes), geomean 1.291× B1 và 1.067× BNB qua 18 cặp; từng prompt/session đều trên parity BNB. 512 token trên `cnndm_002` thêm 2 sessions exact 512/512 prefix, 1.201× B1 / 1.009× BNB. Candidate này hiện dẫn về hiệu năng development, nhưng BNB token IDs vẫn phân kỳ; đây chỉ là B1 exactness. Profile R=2 xác nhận NF4 linear là chi phí chính. Composite contract đã được tạo: `0295c5c9…`; natural trace audit thấy reject-at-0, accept, no-candidate và hai kiểu budget finish, nhưng chưa có EOS. Kế tiếp audit EOS, tích hợp fail-closed runtime guard, rồi mới mở rộng paired benchmark. Không dùng held-out.

**Runtime audit:** adapter Qwen hiện đã fail-closed với input/mask/cache/row shape không hỗ trợ. Capability registry vẫn gate theo architecture/attention strategy và chưa xác minh model revision/state hash hoặc contract ID. EOS fixtures hiện tại chỉ bao phủ legacy singleton path; chưa có fixture R=2 cho block decoder. Kế tiếp cần nối composite contract vào selection path và audit EOS dưới cùng schedule trước khi tuyên bố certified mode.

**G2 mới:** ngoài kernel row-independent ban đầu, đã thêm strategy opt-in `triton_nf4_group_scale` với scale load một lần mỗi nhóm 64 weight. Tile q_proj N=8/K=128 nhanh hơn kernel ban đầu khoảng 44% tại M=5 và giữ bitwise invariance theo M. Hai prompt development khớp logits/KV M=1…5; hai pilot K4/R5 khớp singleton ở 32/32 prefix state. Nsight q_proj giảm 347.5→197.5 μs, register/thread 70→40, occupancy 55→95%. Thêm tile riêng N=8/K=256 (`triton_nf4_group_scale_k256`): operator probes M=1…5 trên cả hai prompt development và hai pilot đều exact theo contract riêng; từng pilot khớp state ở 32/32 prefix. BNB token IDs cũng trùng trong hai prompt mỗi tile, nhưng đây chỉ là quan sát hữu hạn. K256 synthetic audit đạt shape invariance 75/75 hàng qua 15 tổ hợp; một non-nested case khác BNB dequantize+F.linear tối đa 0.0078125. Ba paired sessions mỗi prompt đều khớp token B1/BNB và state 32/32. Block latency geometric mean là 1.300× B1 singleton (range 1.288–1.307×) và 1.815× BNB (1.762–1.845×); paired interval chỉ mang tính chẩn đoán do sáu lượt trên hai prompt cố định. Nsight q_proj K256 là 177.38 μs, với 62% excessive global sectors, 95% occupancy, 40 register/thread và không spill. K-major layout prototype shape-invariant theo singleton riêng nhưng chậm hơn 1.35–1.50× và lệch output K256 hiện tại đến 4.88e-4; đã loại khỏi hướng tích hợp. Sweep q_proj cho K256 nhanh hơn K128 khoảng 9.5% ở M=5, nhưng full-model block chậm hơn singleton/BNB. G2 chưa qua cổng hiệu năng/chứng minh, chưa thay đường mặc định `fixed_rows`.

**Cập nhật packed-byte:** thêm strategy experimental `triton_nf4_group_scale_packed_k256`, contract ID `cf0d7b2094c6eb4a794e07e551c9fd5f44e536d5092a9dc99ce4ae64b9673fdc`. Hai prompt operator M=1…5 và sáu pilot paired (ba sessions/prompt) khớp token B1/BNB, logits/KV operator, và 32/32 committed prefixes theo reference mới. q_proj nhanh hơn grouped K256 khoảng 30% ở M=1/3/5; NCU excessive sectors giảm 62%→15%, duration 177.38→124.61 μs. Full-model K4/R5 block/singleton latency là 1.197×, block/BNB 1.497×; cải thiện rõ nhưng chưa đạt speedup. K=2/R=3 có contract riêng (`4a190f8ae92791462997c64f06fa2857e9abf5e6f96d2ed7fdeac433c212088f`), operator M=1…3 exact trên hai prompt và 6/6 paired pilots khớp token/KV; latency block/singleton 1.090×, block/BNB 1.352×, vẫn chậm hơn mọi baseline. Runner có tùy chọn `--ngram-size`; màn hình thử n=2/4 chưa cho thấy acceptance cao hơn n=3 và số repeat chưa đủ để chọn policy. K-major layout đã thử và loại vì chậm hơn. Tiếp theo: ưu tiên profile target verification ở K1/R2, sau đó tối ưu block; không mở held-out.

**Cập nhật K=1/R=2:** thêm contract riêng `2dd35565577e8c0e2bc4f6b1dd6c4ef1e257dba15bcd488a8cfea0ecabb7ca67`. M=1…2 operator probes exact trên cả hai prompt; sáu phiên paired đều khớp token B1/BNB và KV 32/32, contract repeat audit đạt. Block/singleton latency geometric mean 1.018× (range 0.944–1.065×; paired-log CI 0.974–1.064×), nên chưa chứng minh được nhanh/chậm hơn singleton. Block/BNB là 1.247× (CI 1.176–1.324×), vẫn chậm hơn khoảng 25%. Đây là điểm gần hòa vốn nhất; ưu tiên profile K1 verify path và giảm chi phí block trước khi mở rộng K. Không dùng held-out.

**Sàng lọc proposer K=1:** n-gram 2 và 4 đều giữ token/KV exact trên hai prompt nhưng không tăng acceptance so với n-gram 3 (4/9 và 2/6); số repeat mỗi cấu hình chỉ là một, nên giữ n=3 và chưa chọn policy mới.

**Profile K=1/R=2:** trace block/singleton trên `cnndm_001` ghi nhận 27/31 target forwards và 36/31 hàng được xử lý; packed kernel có 6.804/7.812 launch nhưng tổng thời gian CUDA kernel tăng 13% ở block trong trace. Kết quả khoanh vùng chi phí ở các hàng quantized-linear bổ sung. Tiếp theo thử một kernel multi-row packed có reduction từng hàng giữ nguyên, trước khi lặp paired end-to-end; profile chỉ là chẩn đoán, không phải benchmark latency.

**Prototype reuse hai hàng:** strategy opt-in `triton_nf4_group_scale_packed_rows2_k256` giữ kernel packed gốc ở M=1 và reuse weight decode ở M=2. q_proj M=2 giảm latency 34%, bitwise-equal; toàn-model probes trên ba prompt exact ở M=1/2. Chín paired K1/R2 pilot đều exact token/KV; trên ba prompt, block/singleton latency 0.894× (paired-log CI 0.838–0.953×), block/BNB 1.093×. Đây là speedup B1 lặp lại được trong mẫu hiện tại, nhưng 1.119× vẫn dưới mục tiêu 1.15× và BNB vẫn nhanh hơn trung bình. Tiếp theo mở rộng operator boundary probes và tối ưu tile; chưa tích hợp mặc định, không dùng held-out.

**Tile N=16 và giới hạn end-to-end:** strategy riêng `triton_nf4_group_scale_packed_rows2_n16_k256` giữ shape-invariance M=1/2 theo singleton mới trên ba prompt; chín paired runs đều exact token/KV. Block/singleton là 0.882× (speedup 1.134×), block/BNB 1.082×. N16 giảm q_proj M=2 khoảng 49% so với packed N8, nhưng full-model latency chỉ khác N8 khoảng 1%; không đạt 1.15× và chưa nhanh hơn BNB. N16 giữ dạng experimental để làm baseline tối ưu kernel; bước tới đo/giảm chi phí toàn mạng ngoài q_proj, giữ mọi reference contract tách biệt.

**Stress N=16:** 15 synthetic K/N/scale/bias cases đều exact theo từng-row singleton ở M=2, lặp hàng exact, và khớp bitwise với BNB dequant + FP16 F.linear trong các case đó. Miền được kiểm tra dùng K chia hết quant block 64 và chỉ là stress hữu hạn; không chứng minh mọi kích thước hoặc metadata.

**NCU N=16:** M=2, K256 basic profile trên RTX 4050: q_proj 27.07 μs, 64 registers/thread, occupancy lý thuyết/đạt được 66.7%/48.1%; down_proj 131.04 μs, 72 registers/thread, 58.3%/46.1%. Register allocation giới hạn occupancy lý thuyết; cả compute và DRAM đều đáng kể. Counter local-memory ghi nhận 0 sector cho hai launch, không thấy spill traffic ở các shape này. Đây là một launch có profiler, không phải benchmark latency; bước tối ưu tiếp theo là giảm register live range mà giữ nguyên reduction order, rồi đo lại end-to-end. [Artifact](experiments/exact_quantized_feasibility/ncu_nf4_rows2_n16_v1/summary.json).

**Sàng lọc K=128:** N=16/K128 giảm registers/thread (q_proj 64→48, down_proj 72→56) nhưng achieved occupancy chỉ tăng khoảng 2–3 điểm. Tại M=2, event latency K128/K256 là 1.230× ở q_proj và 1.356× ở down_proj. Operator M=1/2 khớp singleton theo lịch riêng, nhưng K128 không được tích hợp vì chậm hơn và chưa có whole-model contract. [Artifact](experiments/exact_quantized_feasibility/triton_nf4_packed_rows2_n16_k128_exploration_v1/summary.json).

**Lịch N=16/1 warp:** opt-in contract riêng `49c5f6a606a9ef8c32ec8d39b9d6890971834762ac4bbee58ec0b8d60475afbd`. 9/9 pilot × 32 token khớp B1/BNB và 288/288 KV prefix. Geomean speedup là 1.142× so với B1 singleton (CI paired-log chẩn đoán 1.077–1.211×), vẫn dưới 1.15×; BNB ratio 0.916×. So với 4-warp cùng slot, block latency 0.923×. NCU cho thấy register/thread cao và achieved occupancy ~12%, dù M=2 operator nhanh hơn. Seed K=2/R=3 trên cnndm_000 không cải thiện: acceptance 11/18 so với K1 8/12, speedup 1.194× B1 và 0.940× BNB; chưa mở rộng K2. [Pilot summary](experiments/exact_quantized_feasibility/triton_nf4_packed_rows2_n16_warp_sweep_v1/pilot_summary.json), [K1/K2 comparison](experiments/exact_quantized_feasibility/triton_nf4_packed_rows2_n16_warp_sweep_v1/k1_vs_k2_cnndm000_seed.json).

**Operator breakdown & N tile:** trên profile trace cnndm_001, warp1 làm custom packed-kernel self time giảm 4.3%, verify giảm 6.7%; đây là profiler diagnostic. Sweep N=8/16/32 cho warp1 cho thấy N8 không nhất quán giữa q_proj/down_proj, N32 chậm hơn rõ rệt; giữ N16. [Profile](experiments/exact_quantized_feasibility/triton_nf4_packed_rows2_n16_warp_sweep_v1/profile_comparison_cnndm001.json), [tile sweep](experiments/exact_quantized_feasibility/triton_nf4_packed_rows2_n16_warp_sweep_v1/output_tile_summary.json).

**Pipeline stages:** `num_stages=1/2/3` không thay đổi đáng kể M=2 latency; stage1 giữ nguyên 168/166 registers/thread so với stage3. Không tích hợp thêm launch knob; cần chuyển sang dataflow/kernel redesign để giảm live state mà giữ accumulation contract. [Summary](experiments/exact_quantized_feasibility/triton_nf4_packed_rows2_n16_warp_sweep_v1/pipeline_stage_summary.json).

**Pairwise nibble dataflow:** bỏ interleave vector nhưng không có lợi ích latency ổn định. Warp sweep 1/2/4 làm down_proj chậm hơn 4.1%–31.2%; q_proj có một lần M=2/w1 nhanh hơn 2.8% nhưng không lặp lại nhất quán. NCU split variant dùng 246 registers/thread, cao hơn baseline 168/166. Không tích hợp, không chạy whole-model. Cần redesign có giảm register + latency trước khi mở contract mới. [Summary](experiments/exact_quantized_feasibility/triton_nf4_packed_rows2_pairwise_dataflow_v1/summary.json), [warp sweep](experiments/exact_quantized_feasibility/triton_nf4_packed_rows2_pairwise_dataflow_v1/operator_benchmark_warp_sweep.json).

**K-tile cho warp1:** so sánh K=64/128/256 trên q_proj/down_proj. Ở M=2, K128 gần ngang K256 trên q_proj (1.010×) nhưng chậm hơn 5.2% trên down_proj; K64 chậm hơn 0.6%/13.6%. Giữ K256, không mở contract mới. [Summary](experiments/exact_quantized_feasibility/triton_nf4_packed_rows2_n16_warp1_k256_k_tile_sweep_v1/summary.json).

**Split-K thử nghiệm cho down_proj:** split8 giảm khoảng 21% latency operator M=2 trên down_proj và đạt singleton equality ở cả 36 layer. Pilot 3 prompt × 3 session khớp B1/BNB tokens và 288 KV prefix; paired same-process A/B xác nhận split8 chậm hơn warp1-only 3.8% end-to-end (prompt-level diagnostic CI 1.018–1.058). Giữ như prototype correctness opt-in, loại khỏi performance schedule hiện tại. [Summary](experiments/exact_quantized_feasibility/triton_nf4_packed_rows2_splitk_v1/summary.json), [paired A/B](experiments/exact_quantized_feasibility/triton_nf4_packed_rows2_splitk_v1/paired_warp1_vs_split8.json).

Workspace reuse did not improve split-K down_proj latency, and two/four warps made it slower than one warp by 5.2%/15.6%. Stop tuning split-K; the next prototype must change the reduction/data mapping materially. [Workspace screen](experiments/exact_quantized_feasibility/triton_nf4_packed_rows2_splitk_v1/operator_workspace_benchmark.json), [warp sweep](experiments/exact_quantized_feasibility/triton_nf4_packed_rows2_splitk_v1/operator_warp_sweep.json).

**BNB fixed-row K2/R3 budget check:** 3 prompts × 3 sessions exact B1 at 64 and 256 tokens, all 2,880 audited KV prefixes matched. At 64 tokens it matched original BNB tokens 9/9 but averaged 0.908× BNB latency. At 256 tokens it matched BNB tokens 0/9; B1 speedup was 1.397×, while BNB-relative geomean was 1.024× with prompt-level CI 0.951–1.102. Treat this as a long-generation B1 candidate only; B2 identity and BNB speedup remain unestablished. [Summary](experiments/exact_quantized_feasibility/qwen3b_bnb_fixedrows_k2_r3_cnndm3_256_v1/summary.json).

**Initial BNB numerical drift localized to decode step 1:** forced replay on the three saved CNN/DailyMail BNB token paths found prompt-prefill logits/KV bitwise equal, then logit max-abs difference 0.25–0.28125 and KV mismatch after consuming the first generated token. Greedy argmax first changed at generated-token positions 91/93/99. Per-module tracing later localized the first padded-schedule mismatch to layer-0 SDPA. [Trace](experiments/exact_quantized_feasibility/qwen3b_bnb_fixedrows_k2_r3_cnndm3_256_v1/bnb_logit_kv_replay_v1.json), [replay utility](scripts/audit_b1_bnb_replay_drift.py).

**BNB-compatible row schedule prototype:** operator tracing found the first padded-schedule mismatch at layer-0 SDPA (M=1 vs M=3/GQA/mask layout). `bnb_rowwise` now preserves BNB's one-query GQA SDPA call and uses M=1 RMSNorm/LM-head rows. An initial singleton replay then found `layers.33.post_attention_layernorm` drift after 62 tokens; rowwise RMSNorm removed it. At K2/R3 and 256 tokens, 3 prompts × 2 process sessions matched B1 and BNB tokens (6/6), all 1,536 B1 block/singleton KV prefixes, and a separate forced BNB replay found no logit/KV drift through 256 tokens on all three prompts. Geomean speedup: 1.227× vs B1 (prompt CI 1.161–1.305), 0.944× vs original BNB (0.886–1.010). Treat as finite B2 evidence; BNB speedup is not established. [Summary](experiments/exact_quantized_feasibility/qwen3b_bnb_rowwise_k2_r3_cnndm3_256_multisession_summary_v1.json), [contract](experiments/exact_quantized_feasibility/qwen3b_bnb_rowwise_k2_r3_cnndm3_256_v1/reference_contract.json).

**Immediate next step:** keep `bnb_rowwise` behind its experimental capability gate. The first profile shows BNB 4-bit GEMM and one-row vocabulary projection are the largest CUDA costs (250.6 ms and 90.9 ms in one instrumented 24-token trace); SDPA is not the leading cost. Across the 256-token runs, draft acceptance was 51.5%. Reduce quantized GEMM/LM-head overhead only where M=1 outputs and KV remain bitwise equal to original BNB. The two-session latency estimate is 0.944× versus BNB on average; do not expand model/task coverage until a paired development run shows a stable BNB-relative gain or a clear cost ceiling.

A one-session K1/R2 screen also passed BNB/B1 token identity 3/3, 768 B1 KV prefixes, and forced BNB state replay through 256 tokens. Its latency ratios were 1.211× vs B1 and 0.948× vs BNB, so there is no evidence yet that reducing physical rows improves BNB-relative speed. [Screen](experiments/exact_quantized_feasibility/qwen3b_bnb_rowwise_k1_r2_cnndm3_256_v1/summary_screen_v1.json).

K4/R5 is a no-go for B2 under the current wrapper: M=5 fixed-row singleton first diverged from BNB at layer-0 q_proj on decode step 1; the 256-token block matched BNB 0/3 and was 0.818× BNB-relative, while remaining B1-exact. This reinforces the immediate kernel task: match BNB's M=1 quantized-linear arithmetic before raising physical_rows above 3. [Screen](experiments/exact_quantized_feasibility/qwen3b_bnb_rowwise_k4_r5_cnndm3_256_v1/summary_screen_v1.json), [operator trace](experiments/exact_quantized_feasibility/qwen3b_bnb_rowwise_k4_r5_cnndm3_256_v1/first_decode_operator_trace_v1.json).

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
2. Đã đối chiếu prototype với BNB dequantizer trên synthetic nested-NF4 microcases, kiểm tra logits/KV M=1…5 trên hai development prompt, audit PTX/SASS cho kernel ban đầu và stress 18 shape/scale/bias variants. Scale-group K128/K256 giảm đáng kể q_proj traffic/register pressure và cải thiện full-model latency so với kernel đầu, nhưng chưa vượt singleton/BNB. Bước kế tiếp là profile/optimize coalescing và store layout, mở rộng stress state/capacity, rồi paired multi-session pilot cho tile đang được chọn. Không bật mặc định hay tuyên bố speedup đến khi đạt exactness trong miền khóa và end-to-end latency ratio vượt 1.
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


## 8. Kết quả audit shape trực tiếp tiếp theo

Một phép đo trực tiếp trên activation decode đầu tiên của development prompt cnndm_000 đã gọi các BNB Linear4bit với cùng activation lặp thành M=1/2/3/5 hàng. Các q/k/v/o projection và MLP projection được lấy mẫu ở layer 0, 33, 35. M=2/3 khớp bitwise M=1 ở các projection lấy mẫu; tại M=5, q_proj và các MLP projection được lấy mẫu có sai khác, o_proj layer 0 cũng sai khác, còn k_proj/v_proj layer 0 vẫn khớp. LM head sai khác từ M=2 trở lên (max abs 0.0625). Đây là bằng chứng chẩn đoán trên một activation lặp, không phải xác nhận cho block với hidden states khác nhau. Artifact: [shape screen](experiments/exact_quantized_feasibility/qwen3b_bnb_linear_shape_invariance_cnndm000_v1.json); runner: [benchmark script](scripts/benchmark_bnb_linear_shape_invariance.py).

Việc kế tiếp trong G1 là chạy audit trên một số development prompt và vị trí decode, đồng thời ghi activation thật của từng hàng trong verifier để so sánh từng output với singleton. Chỉ sau đó chọn phạm vi kernel: ưu tiên giải quyết q_proj và các MLP projection gây mismatch ở M=5; giữ LM head chạy theo từng hàng nếu cần bảo toàn BNB. Chưa có căn cứ mở rộng B2 vượt physical_rows=3. Không dùng cohort GSM8K 300–349.


Phép audit trực tiếp thứ hai trên cnndm_001 tái hiện đúng mẫu của cnndm_000 ở cả 14 module: M=2/3 chỉ lm_head lệch; M=5 cùng 11 projection q/o/MLP lệch, còn k_proj/v_proj layer 0 khớp. Đây vẫn là hai prompt với activation từng hàng được lặp lại trong microbenchmark, chưa thay thế phép đối chiếu block activation thật. [Artifact cnndm_001](experiments/exact_quantized_feasibility/qwen3b_bnb_linear_shape_invariance_cnndm001_v1.json).


Audit correction và kết quả block activation thật: hook trước đây bỏ qua các input truyền bằng keyword/dict, nên chưa ghi được attention wrapper. Sau khi sửa, với K=2 / M=3 trên cnndm_000 và cnndm_001, hidden state vào layer-0 attention và output q/k/v đều khớp bitwise giữa block và singleton. Output của self_attn lại lệch (max abs lần lượt 0.00390625 và 0.0078125), trước khi o_proj nhận đầu vào lệch 0.001953125 ở cả hai prompt. Vì vậy sai khác đầu tiên được quan sát nằm bên trong đường attention sau q/k/v projection; audit chưa tách riêng RoPE, causal mask và SDPA. Xem [cnndm_000](experiments/exact_quantized_feasibility/qwen3b_bnb_actual_block_operator_audit_cnndm000_k2_v3.json), [cnndm_001](experiments/exact_quantized_feasibility/qwen3b_bnb_actual_block_operator_audit_cnndm001_k2_v2.json), và thay đổi hook trong [audit_operator_shape.py](scripts/audit_operator_shape.py).

Ưu tiên G1/G2 kế tiếp là hook post-RoPE q/k, context trước o_proj, mask/position và KV writes trên cùng hai prefix. Nếu mismatch còn trong context, prototype attention shape-invariant trước; nếu q/k đã lệch, xử lý RoPE/position trước. Giữ nguyên scope Qwen2.5-3B NF4 và không dùng GSM8K 300–349.


Kết quả tách sâu attention: ở layer 0, q/k sau RoPE, SDPA query, K/V prefix (sau khi chuẩn hóa cách biểu diễn GQA) đều khớp bitwise cho cả ba hàng trên hai prompt. Mask block trên phần prefix hợp lệ tương đương không mask của singleton. Tuy vậy context output SDPA gốc lệch ở cả ba hàng: cnndm_000 lần lượt max abs 0.0009765625 / 0.001953125 / 0.000244140625; cnndm_001 là 0.000244140625 / 0.001953125 / 0.001953125. Buộc PyTorch SDPBackend.MATH trên cnndm_000 giảm sai khác còn 0.000244140625 / 0 / 9.5367431640625e-7, nhưng vẫn không exact cho mọi hàng; mở rộng GQA thủ công không cải thiện thêm. Vì vậy chỉ đổi sang math backend chưa giải quyết được contract. Artifact: [SDPA screen cnndm_000](experiments/exact_quantized_feasibility/qwen3b_bnb_sdpa_backend_screen_cnndm000_k2_v2.json) và hai [trace nội bộ](experiments/exact_quantized_feasibility/qwen3b_bnb_sdpa_internal_audit_cnndm000_k2_v3.json), [cnndm_001](experiments/exact_quantized_feasibility/qwen3b_bnb_sdpa_internal_audit_cnndm001_k2_v1.json).

Bước G2 tiếp: prototype fixed per-query reduction hoặc kernel attention shape-invariant trên tensor layer 0 đã capture; yêu cầu bitwise equality từng row và causal prefix/KV đúng rồi mới profile cost. Giữ K=2/R=3; không mở held-out.


Prototype fallback SDPA theo từng query: block compacts GQA heads, cắt KV đúng causal prefix và gọi SDPA với shape/flags singleton; LM head cũng được gọi từng row. Với K=2/M=3 trên cnndm_000 và cnndm_001, **398/398 hooked module outputs khớp bitwise** với ba forward singleton, gồm output từng decoder layer và LM-head logits. Đây mới là kiểm tra hữu hạn trên prefix cố định; chưa so sánh trực tiếp canonical KV tensors, chưa chạy closed-loop dài, và chưa có timing. [Artifact cnndm_000](experiments/exact_quantized_feasibility/qwen3b_bnb_rowwise_sdpa_override_cnndm000_k2_v2.json), [cnndm_001](experiments/exact_quantized_feasibility/qwen3b_bnb_rowwise_sdpa_override_cnndm001_k2_v1.json); implementation chẩn đoán: [audit_operator_shape.py](scripts/audit_operator_shape.py).

Bước kế tiếp: đối chiếu canonical KV sau block/singleton và đo overhead của fallback; sau đó prototype fixed-reduction SDPA chạy các query rows song song nhưng giữ lịch tích lũy singleton. Chưa dùng fallback này làm claim hiệu năng.


KV đã được so trực tiếp: native BNB block K=2/M=3 lệch KV đầu tiên ở layer 1 (cnndm_000: key max abs 0.03125, value 0.00390625; cnndm_001: key 0.0625, value 0.00390625). Per-query SDPA + rowwise LM head khớp key/value cả 36 layer và 398/398 module outputs với singleton trên hai prefix này. Đây là bằng chứng hữu hạn, chưa phải chuỗi closed-loop dài.

Benchmark compute-only 15 vòng paired, cache restore ngoài timer, cho median singleton / native block / exact fallback: cnndm_000 là 70.41 / 26.77 / 35.31 ms; cnndm_001 là 67.65 / 25.84 / 33.13 ms. Exact fallback nhanh 1.99× / 2.04× so với ba singleton, nhưng chậm hơn native block 1.32× / 1.28×. Đây chưa phải end-to-end latency; fallback vẫn gọi attention từng row. [Artifact timing 000](experiments/exact_quantized_feasibility/qwen3b_rowwise_sdpa_timing_cnndm000_k2_v1.json), [timing 001](experiments/exact_quantized_feasibility/qwen3b_rowwise_sdpa_timing_cnndm001_k2_v1.json).

G2 đã có mục tiêu thực nghiệm rõ: giữ lịch số học SDPA singleton nhưng chạy độc lập query rows song song bằng kernel riêng; sau đó đo toàn chu kỳ cùng cache/commit. Fallback hiện tại chứng minh mục tiêu đúng đắn trên hai prefix; timing chỉ cho thấy dư địa tính toán, chưa chứng minh speedup end-to-end.


## 9. Triton fixed-row attention: first Qwen decode integration

A first Triton kernel prototype now gives one program per query/head with a fixed online-softmax key-tile schedule (BLOCK_K=32). On two K=2/M=3 transitions, fixed-row block and singleton matched 398/398 hooked module outputs, greedy next-token IDs, and canonical KV tensors for all 36 layers. This is a decode-transition result from a shared original-BNB prefill cache, not a long closed-loop proof. Clean artifacts: [cnndm_000](experiments/exact_quantized_feasibility/qwen3b_triton_fixed_row_attention_cnndm000_k2_v7.json), [cnndm_001](experiments/exact_quantized_feasibility/qwen3b_triton_fixed_row_attention_cnndm001_k2_v1.json). Early artifacts v1–v5 used an incorrect non-contiguous output layout and are superseded.

B1/B2 split remains essential: custom fixed-row singleton KV differs from original BNB starting at layer 1 on both prefixes, despite matching the three next-token IDs in this short trace. The Triton schedule is a candidate B1 reference and has not earned B2 equivalence.

In 15 paired compute-only rounds, fixed Triton singleton/block medians were 70.72/34.32 ms and 69.18/33.67 ms (2.06×/2.05× within B1). Original BNB singleton/block were 65.60/26.88 ms and 63.89/26.20 ms; fixed block cost 1.28× native BNB block. Timings exclude prefill, draft, cache restore and commit, so they do not demonstrate end-to-end gain. [Timing 000](experiments/exact_quantized_feasibility/qwen3b_triton_fixed_row_attention_timing_cnndm000_k2_v1.json), [timing 001](experiments/exact_quantized_feasibility/qwen3b_triton_fixed_row_attention_timing_cnndm001_k2_v1.json).

The first closed-loop gate is complete on `cnndm_000`: the K=2/R=3 block and fixed-row singleton produced identical token IDs and KV hashes for all 64 generated prefixes. Three separate, uninstrumented end-to-end timing rounds gave 1.179× geometric-mean speedup over B1 singleton, while the block remained 0.864× original-BNB speed (slower). The state-hash pass was excluded from timing; its initial instrumented wall time is invalid as a performance measure. This is one prompt and three rounds, not a confidence interval. See [runner](scripts/audit_triton_b1_closed_loop.py), [session 1 artifact](experiments/exact_quantized_feasibility/triton_b1_closed_loop_cnndm000_k2_r3_s1_v1.json), and [findings](experiments/exact_quantized_feasibility/FINDINGS.md).

The first closed-loop integration used `bnb_rowwise` and therefore called Triton attention serially by row. It served as an initial correctness baseline; the later `triton_batched` work below is the parallel-row path. The custom singleton's KV remains different from native BNB, so B2 state exactness is still not established.

The second development prompt, `cnndm_001`, matched B1 singleton tokens and all 128 KV prefixes at a 128-token budget in two independent process sessions. Serial-row end-to-end timing screens were 1.159× and 1.119× vs B1 singleton, and 0.843× and 0.807× vs original BNB. Original-BNB tokens first diverged from the Triton schedule at generated-token index 28. Artifacts: [session 2](experiments/exact_quantized_feasibility/triton_b1_closed_loop_cnndm001_k2_r3_128_s2_v1.json), [session 3](experiments/exact_quantized_feasibility/triton_b1_closed_loop_cnndm001_k2_r3_128_s3_v1.json).

A first genuinely batched Triton query-row path (`triton_batched`) passed B1 state checks, but its rowwise RMSNorm and vocabulary projection made it much slower. Profiling found the vocabulary projection dominated, so both are now run at fixed physical M=3 for singleton and block. The updated schedule matched all B1 tokens/KV prefixes at 64 tokens on `cnndm_000`, 128 tokens on `cnndm_001` and `cnndm_002`, and 256 tokens on `cnndm_001`. Three paired timings per prompt/budget gave descriptive combined geometric means of **1.280× vs B1 singleton** (diagnostic paired-log-t 95% interval 1.219–1.344) and **0.957× vs original BNB** (0.908–1.008). These intervals summarize three prompts only. Original BNB tokens matched at 64 tokens on `cnndm_000`, then diverged at generated index 28 on `cnndm_001` and 99 on `cnndm_002`. The 256-token single timing was 0.965× BNB. No BNB KV identity claim is supported.

The pre-optimization profile recorded 416 rowwise vocabulary projections and 1.476 s of CUDA time. After fixing M=3, the block profile had 10 vocabulary projections at 34 ms total; NF4 GEMMs became the largest remaining cost. A packed Triton linear screen retained B1 identity but regressed to 0.980× B1 and 0.834× BNB, so that integration was rejected for this schedule. Next gate: profile/tune the NF4 GEMMs or block acceptance without changing the now-exact B1 contract, then repeat across more development prompts and sessions. Keep BNB token and state comparisons separate. GSM8K 300–349 remains unopened.

The state auditor now also compares `TargetKVCache.get_seq_length()` and active lengths for all layers at shared commit frontiers. One 64-token `cnndm_000` run passed token IDs, every KV prefix hash, and all shared logical-length frontiers. See [full-state frontier audit](experiments/exact_quantized_feasibility/triton_b1_parallel_attention_fullstate_cnndm000_k2_r3_64_s20_v1.json). Extend this metadata check to longer outputs and terminal/rejection branches before treating the implementation as a closed runtime proof.
