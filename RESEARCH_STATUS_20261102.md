# Trạng thái nghiên cứu — 02/11/2026

Tài liệu này tổng hợp toàn bộ trạng thái kỹ thuật hiện tại, kết quả đã có, và các bước tiếp theo. Đọc cùng `FINDINGS.md`, `PROOF_OBLIGATIONS.md` và `PROTOCOL.md` trong `experiments/exact_quantized_feasibility/`.

---

## 1. Bức tranh tổng thể

Dự án nghiên cứu một câu hỏi: **có thể chạy speculative decoding trên mô hình NF4 4-bit và thu được chính xác cùng chuỗi token với greedy decoding tuần tự của backend gốc không?**

Hiện có **hai hướng tiếp cận** với trạng thái khác nhau:

| Hướng | Kernel | Reference | B2 match | Speedup vs BNB |
|---|---|---|---|---|
| **B1** | `triton_nf4_fixed_three_rows_group_scale_packed` | Fixed-row singleton (B1) — khác BNB | 6–8% runs | 1.228–1.300x latency |
| **B2 activationvec** | `_fused_nf4_m3_kernel` + `bnb_rowwise` attn | BNB M=1 singleton (B2) | **100% runs (23/23)** | Phụ thuộc acceptance |

**B1** là hướng cũ, có speedup cao nhưng reference khác BNB gốc. **B2 activationvec** là hướng mới hơn, đạt exact B2 nhưng speedup chỉ dương khi acceptance ≥ ~60%.

---

## 2. Kết quả B1 — Fixed-row schedule

### 2.1. Cơ chế

Mọi linear/RMSNorm/attention trong decode đều được pad đến `physical_rows` cố định. Singleton reference và block verification dùng cùng shape → cùng kernel dispatch → cùng arithmetic.

- `physical_rows = 3`, `K = 2` (2 draft + 1 bonus)
- Kernel: `triton_nf4_fixed_three_rows_group_scale_packed`
- Attention: rowwise hoặc padded

### 2.2. Kết quả chính

| Cohort | Runs | B1 match | Speedup vs B1 singleton | Speedup latency vs BNB | BNB token match |
|---|---:|---:|---:|---:|---:|
| 12 prompts, 256 tokens | 24 | **24/24** | 1.516x | 1.228x | 6/24 |
| 8 prompts, 512 tokens | 16 | **16/16** | 1.612x | 1.300x | 2/16 |

Intervals: prompt-cluster bootstrap trên development prompts.

### 2.3. Giới hạn B1

- Reference B1 ≠ BNB gốc: thay đổi `physical_rows` tạo trajectory khác nhau
- Speedup vs BNB latency không phản ánh exact B2 output — chỉ là so sánh latency
- BNB token match thấp (6/24, 2/16) vì B1 ≠ B2

---

## 3. Kết quả B2 — Activationvec approach (**kết quả mới nhất**)

### 3.1. Cơ chế

Kernel `_fused_nf4_m3_kernel` xử lý M=3 activation rows nhưng duy trì **cùng arithmetic per-row với BNB M=1 singleton**:
- Vectorized activation loading để giảm memory traffic
- Mỗi row compute độc lập (không cross-row reduction)
- Attention: `bnb_rowwise` — mỗi query row được SDPA riêng với causal prefix của nó

Kết quả: `block[row_i] = BNB_singleton(x[row_i], W)` bitwise với mọi `i`.

### 3.2. Evidence B2 identity

**One-cycle B2 identity check** (`activationvec_singleton_identity_summary_v1.json`):
- 2 prompts (cnndm_002, cnndm_003)
- Bitwise equal: logits SHA256 khớp, KV prefix SHA256 khớp
- `logits_max_abs_diff: 0.0` — bitwise exact

**Closed-loop B2 check** (`activationvec_closed_loop_summary_v1.json`):
- 23 runs trên 13 distinct prompt/budget configurations
- `all_runs_match_singleton: True` — toàn bộ token IDs khớp BNB greedy
- `all_runs_match_state_prefixes: True` — KV prefix hashes khớp

### 3.3. Speedup phụ thuộc acceptance

| Prompt | Acceptance | Speedup vs BNB | Nhận xét |
|---|---:|---:|---|
| HumanEval/112 (128 tok) | 65.2% | 1.242–1.284x | ✓ |
| HumanEval/116 (64 tok) | 87.5% | 1.265–1.275x | ✓ |
| HumanEval/119 (64 tok) | 66.7% | 1.162–1.206x | ✓ |
| HumanEval/112 (64 tok) | 60.0% | 1.143–1.146x | ✓ |
| HumanEval/120 (64 tok) | 65.5% | 0.953–0.990x | ✗ near break-even |
| HumanEval/113 (64 tok) | 36.8% | 0.858–0.884x | ✗ |
| HumanEval/117 (64 tok) | 0.0% | 0.779x | ✗ |
| Prompts /121–/124 | 24–39% | 0.825–0.891x | ✗ |

Geomean trên 13 configs (không có fallback): **0.954x** vs BNB — dưới BNB trung bình vì nhiều low-acceptance prompts.

### 3.4. Fallback policy

Khi không có draft proposal (empty-proposal cycle), activationvec kernel vẫn chạy M=3 với padding overhead. Native BNB M=1 nhanh hơn trong trường hợp này.

**Fallback ablation** (`fallback_fraction_ablation_256_summary_v1.json`):
- 100% fallback (BNB M=1 trên mọi empty-proposal cycle): +6.0% vs activationvec (geomean 3 prompts)
- 50% fallback: +2.8% vs activationvec
- Cả hai vẫn giữ B1 token match: `all_runs_exact_to_b1_singleton: True`
- B2 identity: Native BNB M=1 fallback IS B2 reference → fallback không phá B2

**Vấn đề chưa trả lời**: closed-loop với 100% fallback policy có giữ B2 identity end-to-end không? (Về mặt lý thuyết: có — BNB M=1 fallback là chính B2 reference; activationvec block cũng B2 exact → kết hợp vẫn B2 exact.)

### 3.5. Giới hạn B2 hiện tại

- Chỉ 2 prompts kiểm tra bitwise logits+KV (1 cycle, all-accepted path)
- Closed-loop 23 runs: kiểm tra token ID, không kiểm tra bitwise KV từng bước
- Chỉ HumanEval prompts (code generation); không có GSM8K, CNN/DM
- Một GPU (RTX 4050 Laptop), một model, một revision

---

## 4. Operator-level evidence

### 4.1. Linear4bit audit

Từ `bnb_simt_full_operator_audit_summary.json`:
- **252/252 Linear4bit modules** khớp BNB M=3 rowwise và BNB M=1 singleton
- 9,289,728/9,289,728 output elements bitwise equal
- 3 development prompts: HumanEval/112, /114, CNN/DM/005
- Đây là finite evidence trên staged-rounding path, không phải proof toàn domain

### 4.2. Attention — bnb_rowwise

Mỗi query row nhận SDPA call riêng với `enable_gqa=True`, causal prefix chính xác, không mask → giống hệt BNB singleton SDPA path.

### 4.3. LM head

Đối với `bnb_rowwise` strategy: mỗi row được call qua BNB riêng → giữ đúng BNB M=1 arithmetic cho vocabulary projection.

### 4.4. Gaps còn lại

- Không có mathematical proof cho bất kỳ operator nào (chỉ có empirical audit)
- Triton `tl.sum` reduction order phụ thuộc compiler: kết quả bitwise có thể thay đổi theo Triton version
- Proof obligations đầy đủ trong `PROOF_OBLIGATIONS.md`

---

## 5. Architecture của hệ thống hiện tại

```
Speculative generation loop (B2 activationvec path):

  Prefill → BNB standard path → cache state S_0

  While not done:
    1. PLD n-gram lookup → K draft tokens (hoặc empty)
    
    2a. If proposals exist:
          Block verify (M = K+1 = 3 rows):
          - Linear4bit: _fused_nf4_m3_kernel (activationvec)
          - Attention:  bnb_rowwise per-query-row SDPA
          - LM head:    BNB M=1 per row
          → Compare argmax với draft candidates
          → Accept longest matching prefix
          → Rollback speculative KV suffix
    
    2b. If no proposals (empty cycle):
          Fallback: BNB M=1 singleton forward
          → Always matches B2 reference
    
    3. Commit accepted prefix → S_{t+accepted}
```

KV rollback: `cache.crop(accepted_length)` loại bỏ speculative suffix, giữ nguyên accepted prefix.

---

## 6. Điều gì chưa được giải quyết

### 6.1. Chứng minh toán học (PROOF_OBLIGATIONS.md)

Không có operator nào được chứng minh formal:
- Lemma cho NF4 linear: chưa có (chỉ empirical)
- Lemma cho attention: chưa có
- Lemma cho RoPE, RMSNorm, residual: chưa có
- Định lý toàn chuỗi: conditional + unproven hypotheses

### 6.2. Thiếu multi-cycle KV bitwise check

Closed-loop 23 runs kiểm tra token IDs và KV prefix hashes. Nhưng chưa có test kiểm tra **từng intermediate KV state** qua nhiều chu kỳ accept/reject liên tục.

### 6.3. Reject path chưa kiểm tra kỹ

Bitwise identity check (`activationvec_singleton_identity`) chỉ test all-accepted path. Khi reject xảy ra tại vị trí `i < K`, KV rollback phải đảm bảo state sau commit == state BNB singleton. Đây là một proof obligation chưa được cover bởi test hiện tại.

### 6.4. Speedup chưa đủ để paper

Geomean 0.954x trên development prompts là dưới BNB. Cần:
- Held-out evaluation trên prompts có acceptance rate cao hơn
- Fallback policy để cải thiện low-acceptance cases
- Mở rộng sang tasks có nhiều text repetition (acceptance cao hơn)

### 6.5. Một GPU, một model

Kết quả chỉ có trên RTX 4050 Laptop (Ada, SM89), Qwen2.5-3B NF4, BNB 0.50.0. Tính tổng quát chưa được kiểm tra.

---

## 7. Hướng tiếp theo (ưu tiên)

### Ưu tiên 1: Kiểm tra reject path B2 identity

Viết test: khi reject xảy ra tại position `i`, logits tại position `i+1` của activationvec path (sau rollback) phải bằng BNB M=1 singleton tiếp theo. Đây là gap quan trọng nhất hiện tại.

```python
# Cần test:
# 1. Force reject tại i=0: verify block[0] != draft[0]
#    → commit 0 tokens, next BNB call == standalone BNB call
# 2. Force reject tại i=1: verify block[0] == draft[0], block[1] != draft[1]
#    → commit 1 token, next BNB call == standalone BNB call
# 3. Partial accept + reject + next cycle
```

### Ưu tiên 2: Closed-loop với 100% fallback policy

Mở rộng `benchmark_bnb_nf4_activationvec_closed_loop.py` để hỗ trợ `fallback_on_empty=True`. Verify B2 identity vẫn giữ. Measure speedup trên toàn bộ dev prompts.

### Ưu tiên 3: Mở rộng closed-loop sang thêm prompts

- GSM8K: tính toán số học → acceptance thường cao hơn code generation
- CNN/DM: summarization → nhiều text repetition từ nguồn
- Dùng development set (không phải held-out 300-349)

### Ưu tiên 4: Benchmark B2 activationvec vs B1

Đo trực tiếp speedup của B2 activationvec (với fallback) vs BNB greedy trên cùng cohort và cùng protocol với kết quả B1 hiện tại.

### Ưu tiên 5: Freeze và dùng held-out cohort

Sau khi algorithm, kernel, và fallback policy được khóa, mở GSM8K held-out (indices 300-349) để đo kết quả cuối cùng.

---

## 8. Tóm tắt "what's new" kể từ cleanup 02/11/2026

| Thay đổi | Nội dung |
|---|---|
| README cập nhật | B2 status từ "Not achieved" → "Token-trajectory achieved (23/23 runs)"; thêm B2 results table |
| Xác nhận LLM-42 (SOSP 2026) | Paper có thật, Microsoft Research; KHÔNG đề cập NF4/bitsandbytes → differentiation rõ ràng |
| bao_cao/references.bib | Thêm `gond2026llm42` entry |
| bao_cao/related_works.tex | Thêm đoạn mô tả LLM-42 + dòng trong bảng so sánh |
| PEER_REVIEW_SESSION_20261102.md | File ghi chép phiên review |
| RESEARCH_STATUS_20261102.md | File này |

---

## 9. Numbers to remember

| Số liệu | Giá trị | Nguồn |
|---|---|---|
| B1 B1-match | 24/24, 16/16 | README main table |
| B1 speedup vs B1 singleton | 1.516x, 1.612x | README main table |
| B1 latency speedup vs BNB | 1.228x, 1.300x | README main table (latency only) |
| B1 BNB token match | 6/24, 2/16 | README main table |
| B2 closed-loop match | **23/23** | activationvec_closed_loop_summary |
| B2 bitwise logits+KV | 2/2 (1 cycle each) | activationvec_singleton_identity_summary |
| B2 best speedup vs BNB | 1.242–1.284x (HE/112, 128 tok) | closed_loop_summary |
| B2 geomean speedup (no fallback) | 0.954x | computed from summary |
| Fallback (100%) boost | +6.0% vs activationvec | fallback_fraction_ablation_256_summary |
| Linear4bit audit | 252/252 modules, 9.3M elements exact | bnb_simt_full_operator_audit_summary |
| Peak VRAM (B2) | 2144 MB (HE/112, 128 tok) | closed_loop_summary |

---

## 10. Kết quả thực hiện Ưu tiên 1–3 (cập nhật cuối ngày 02/11/2026)

Scripts mới: `scripts/audit_activationvec_reject_path_b2.py`, `scripts/benchmark_bnb_nf4_activationvec_fallback_closed_loop.py`. Output trong `experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/` (`activationvec_reject_path_*`, `activationvec_fallback_closed_loop_*`).

**Ưu tiên 1 — reject path B2 (cnndm_003, 1 prompt): 5/5 scenario đạt.** Gồm reject@0, reject@1, all-accept, reject@0 rồi partial-accept+reject@1, all-accept rồi reject@0. Mỗi scenario: pending token == BNB, KV prefix SHA256 sau `crop` == BNB singleton, cache length đúng, logits BNB M=1 kế tiếp bitwise bằng tham chiếu. Chỉ 1 prompt, draft sai ép bằng `(t+1) % vocab`.

**Ưu tiên 2 + 3 — closed loop có fallback (token-ID khớp BNB gốc trong mọi round, 3 round/prompt):**

| Cohort | Prompts | Tokens | Geomean không fallback | Geomean có fallback |
|---|---:|---:|---:|---:|
| CNN/DM (002, 003) | 2 | 64 | 0.800x | 1.018x |
| GSM8K dev (000–005) | 6 | 128 | 0.783x | 1.045x |
| HumanEval dev (0–5) | 6 | 128 | 0.838x | 1.100x |

**Mục 1 (đã làm): vào schedule một lần mỗi lượt generate.** Đo được vào/thoát `block_ctx` tốn ~4.9 ms mỗi block cycle (schedule 1.7 ms + `triton_m3_overrides` 3.3 ms). `SwitchableSchedule` trong script benchmark vào context một lần và chỉ hoán đổi `module.forward` giữa plain BNB (prefill, cycle fallback) và patched (block). Kết quả chạy lại (v2, cùng prompt, token vẫn khớp BNB mọi round):

| Cohort | Không fallback v1 → v2 | Có fallback v1 → v2 |
|---|---:|---:|
| CNN/DM (2 prompts, 64 tok) | 0.800x → 0.910x | 1.018x → 1.062x |
| GSM8K dev (6, 128 tok) | 0.783x → 0.920x | 1.045x → 1.092x |
| HumanEval dev (6, 128 tok) | 0.838x → 0.974x | 1.100x → 1.152x |

Thời gian v2 đã gồm chi phí vào context một lần. Bảng v1 bên dưới giữ để đối chiếu.

**Kiểm KV từng bước qua `SwitchableSchedule`** (`scripts/audit_activationvec_fallback_kv_closed_loop.py`, output `activationvec_fallback_kv_closed_loop_{cnndm,gsm8k,humaneval}_v1.json`): với 14 prompt dev, cả chế độ có và không có fallback (28 lượt), token bằng BNB greedy gốc và KV prefix SHA256 bằng BNB greedy tại mọi prefix không-terminal (83–119 prefix mỗi lượt), gồm mọi cycle reject/rollback do PLD tạo ra. Còn thiếu: so logits từng bước (chỉ KV và token), và cohort vẫn nhỏ, chỉ tập dev.

**Ưu tiên 4 (đã làm): B2 có fallback vs B1** (`scripts/benchmark_b2_fallback_vs_b1.py`, output `b2_fallback_vs_b1_{gsm8k,humaneval}_v1.json`; 6 prompt dev mỗi cohort, 128 token, 3 round xoay thứ tự, B1 = Triton fixed-three-rows + rowwise attention):

| Cohort | Geomean B2 vs BNB | Geomean B1 vs BNB | B2 khớp BNB | B1 khớp BNB | B1 block khớp B1 singleton |
|---|---:|---:|---:|---:|---:|
| GSM8K dev | 1.113x | 1.083x | 6/6 | 4/6 | 6/6 |
| HumanEval dev | 1.155x | 1.286x | 6/6 | 1/6 | 6/6 |

B2 nhanh hơn hoặc ngang B1 trên hầu hết prompt và khớp BNB gốc 12/12, trong khi B1 chỉ khớp 5/12. Ngoại lệ HumanEval/2: B1 đạt 2.21x vì quỹ đạo token của B1 khác BNB (80 draft được chấp nhận so với 24 của B2), nên con số đó không so sánh được; geomean HumanEval của B1 bị kéo lên bởi prompt này. Kết luận sơ bộ: trên cohort dev nhỏ, B2 có fallback đạt tốc độ cùng cỡ B1 mà vẫn exact với BNB gốc. Chưa có khoảng tin cậy; chỉ 12 prompt.

**Ưu tiên 3 (đã làm cho GSM8K/HumanEval dev): 30 prompt mỗi cohort** (`b2_fallback_vs_b1_{gsm8k30,humaneval30}_v1.json`, tóm tắt `b2_fallback_vs_b1_30prompt_bootstrap_v1.json` bằng `scripts/summarize_b2_fallback_vs_b1.py`; bootstrap theo prompt, 10000 lần, 128 token, 3 round):

| Cohort | B2 vs BNB (CI95) | B1 vs BNB (CI95) | B2/B1 (CI95) | B2 khớp BNB | B1 khớp BNB |
|---|---|---|---|---:|---:|
| GSM8K dev (30) | 1.121x [1.101, 1.141] | 1.103x [1.073, 1.135] | 1.016 [1.000, 1.033] | 30/30 | 19/30 |
| HumanEval dev (30) | 1.194x [1.154, 1.239] | 1.231x [1.155, 1.322] | 0.970 [0.912, 1.013] | 30/30 | 17/30 |

B2 có fallback không chậm hơn BNB trên prompt nào (0/60 dưới 1.0x; B1: 6/60) và khớp BNB gốc 60/60, B1 khớp 36/60. Về tốc độ B2 và B1 không phân biệt được ở HumanEval (CI của B2/B1 chứa 1) và B2 nhỉnh hơn rất nhẹ ở GSM8K. Các CI chỉ phản ánh biến thiên giữa prompt dev đã chọn, không phải tổng thể workload; CNN/DM chưa mở rộng. Số 6 prompt ở trên là tập con của 30 prompt này.

**Ưu tiên 4 mục tăng acceptance (đã quét `ngram_size`, k=2 cố định):** `scripts/sweep_b2_fallback_ngram.py`, 20 prompt dev đầu mỗi cohort, 128 token, 2 round (`ngram_sweep_{gsm8k20,humaneval20}_v1.json`). `ngram_size` chỉ là cận trên của PLD (thử từ `ngram_size` xuống 2). `k` bị cố định ở 2 vì M=3 là hình dạng duy nhất kernel và audit B2 đã phủ.

| ngram_size | GSM8K dev vs BNB | HumanEval dev vs BNB | khớp BNB | prompt dưới BNB |
|---:|---:|---:|---:|---:|
| 2 | 1.123x | 1.176x | 40/40 | 0 |
| 3 | 1.127x | 1.190x | 40/40 | 0 |
| 4 | 1.128x | 1.195x | 40/40 | 0 |
| 5 | 1.130x | 1.201x | 40/40 | 0 |

Kết luận: `ngram_size` gần như không ảnh hưởng (chênh dưới 2% trên toàn dải, cùng cỡ nhiễu đo, chưa có CI); xu hướng tăng nhẹ theo ngram_size không đủ để đổi mặc định 3, nên không chạy xác nhận trên prompt 30–49. Muốn tăng tốc đáng kể phải thay đổi cơ chế (nguồn draft tốt hơn, hoặc kernel nhanh hơn), không phải chỉnh PLD.

**Dọn dẹp (đã làm, chưa commit):** thêm `tests/test_switchable_schedule.py` (test CPU cho `SwitchableSchedule`, qua) và `experiments/exact_quantized_feasibility/activationvec_b2_v1/experiment_manifest.json` ghim SHA256 của 16 file nguồn và 18 file bằng chứng. 6 file `activationvec_fallback_closed_loop_*` (v1, v2) ghi trong manifest là *stale source*: script benchmark đã đổi sau khi chúng chạy. Manifest chưa có validator riêng; `source_sha256` trong các JSON bằng chứng còn lại khớp script hiện tại.

Fallback đưa các prompt acceptance thấp từ ~0.75x lên ~1.0x và giữ B2. Giới hạn: thời gian đã gồm chi phí vào lại schedule mỗi block cycle (chưa tối ưu); identity chỉ kiểm token ID, không kiểm KV từng bước; cohort nhỏ, development-only; held-out 300–349 chưa dùng. Ưu tiên 4 (so với B1 cùng protocol) và 5 (held-out) chưa làm.

---

## 11. Held-out GSM8K 300–349 — cấu hình đăng ký trước khi chạy (03/11/2026)

Code khóa ở commit `b004262`. Cohort `data/benchmarks/gsm8k_exactness_holdout.jsonl`, SHA256 `07e43c96...30bf0` khớp `frozen_gsm8k_300_349.json`; chưa từng chạy trước đó. Chạy đúng một lần bằng `scripts/benchmark_b2_fallback_vs_b1.py`: 50 prompt, 128 token, 3 round xoay thứ tự, B2 = `ngram_size=3`, `k=2`, `physical_rows=3`, `bnb_rowwise`, fallback bật; B1 = Triton fixed-three-rows + rowwise attention; tóm tắt bằng `scripts/summarize_b2_fallback_vs_b1.py` (bootstrap theo prompt, 10000 lần, seed 0). Chỉ tiêu chính: số prompt B2 khớp BNB gốc (token ID), geomean tốc độ B2 so với BNB kèm CI. Không đổi cấu hình hay chạy lại sau khi thấy kết quả; kết quả báo cáo nguyên trạng kể cả khi xấu.

**Kết quả (một lần chạy, nguyên trạng; `heldout_gsm8k_300_349_b2_fallback_vs_b1_v1.json`, `heldout_gsm8k_300_349_bootstrap_v1.json`).** Lần chạy đầu tiên lỗi ngay từ đầu do cách gọi lệnh của tôi (zsh không tách danh sách ID, `KeyError` trước khi xử lý prompt nào, không có số liệu); chạy lại đúng cấu hình đăng ký.

| Chỉ tiêu | B2 có fallback | B1 |
|---|---|---|
| Khớp BNB gốc (token ID, 3 round) | **50/50** | 24/50 |
| Tốc độ vs BNB, geomean (CI95) | **1.117x [1.100, 1.135]** | 1.108x [1.087, 1.130] |
| B2/B1 (CI95) | 1.008 [0.995, 1.022] | — |
| Prompt chậm hơn BNB | 2/50 (thấp nhất 0.975x, cao nhất 1.274x) | 2/50 |
| B1 block khớp B1 singleton | — | 50/50 |

Kết luận: trên cohort held-out, B2 giữ exact với BNB gốc ở mọi prompt, tốc độ tăng khoảng 12% và không phân biệt được với B1 (CI của B2/B1 chứa 1). Hai prompt B2 chậm hơn BNB nhẹ (tệ nhất 0.975x). Số liệu khớp xu hướng của tập dev (1.121x trên 30 prompt GSM8K dev). Giới hạn: một GPU, một model, K=2, 128 token, chỉ GSM8K; bằng chứng bitwise logits/KV từng bước chỉ có trên tập dev, trên held-out chỉ kiểm token ID; CI chỉ phản ánh biến thiên giữa các prompt; cohort không loại trừ được việc dữ liệu công khai đã lọt vào dữ liệu huấn luyện mô hình.

---

*Ngày cập nhật: 02/11/2026. Tất cả kết quả trên development prompts, một GPU RTX 4050 Laptop 6 GB, Qwen2.5-3B-Instruct NF4, BNB 0.50.0, Triton 3.6.0, PyTorch 2.11.0.*
