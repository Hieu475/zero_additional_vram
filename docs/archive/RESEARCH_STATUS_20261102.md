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

## 12. Profile từng pha của vòng B2 (03/11/2026, tập dev)

`scripts/profile_b2_fallback_phases.py`, 8 prompt GSM8K dev (000–007), 128 token, một lượt đo mỗi prompt sau 1 lượt khởi động; output `phase_profile_gsm8k8_v1.json`. Bản đo có token khớp BNB và khớp `generate()` gốc ở cả 8 prompt.

| Pha | Tỷ lệ thời gian B2 |
|---|---:|
| forward fallback (BNB M=1) | 61.0% |
| forward block (M=3 activationvec) | 36.4% |
| prefill | 1.8% |
| chuyển schedule (plain↔patched) | 0.5% |
| vào/thoát context | 0.2% |
| tra n-gram | 0.1% |
| commit/rollback | 0.01% |

| Chỉ số | Giá trị |
|---|---:|
| Chi phí 1 forward fallback | 22.5 ms |
| Chi phí 1 forward block M=3 | 37.1 ms (1.65x fallback) |
| BNB decode mỗi token | 24.0 ms |
| Token sinh ra mỗi block cycle | 1.86 |
| Tỷ lệ cycle phải fallback | 73% |

Nhận định: (1) Overhead Python/điều phối gần như hết (dưới 1% tổng), mục 1 trước đó đã lấy hết phần này. (2) Nút thắt là kernel: một forward block M=3 tốn 1.65 lần một forward M=1, trong khi decode bị giới hạn bởi băng thông bộ nhớ nên về lý thuyết M=3 gần bằng M=1. Block chỉ có lợi khi sinh trên 1.65 token/cycle, tức acceptance trên khoảng 33% (ứng với 2 draft/cycle); dưới ngưỡng đó mỗi block cycle chậm hơn fallback. (3) Ở mức 1.86 token/cycle, mỗi token qua block tốn 37.1/1.86 = 20 ms so với 22.5 ms, tiết kiệm chỉ ~11% mỗi token, và chỉ 27% cycle là block, khớp với tốc độ ~1.12x. Nếu forward M=3 chỉ đắt bằng M=1, tốc độ cùng acceptance sẽ cao hơn rõ rệt: hướng đầu tư có lợi nhất là tối ưu kernel M=3 (giữ nguyên số học từng hàng), không phải chỉnh PLD. Giới hạn: 8 prompt GSM8K dev, một lượt đo mỗi prompt.

**Phân rã kernel GPU của một forward block** (`scripts/profile_b2_block_kernels.py`, `block_kernel_profile_gsm8k_000_v1.json`, 1 prompt): GPU bận 14.7 ms với BNB M=1 so với 26.2 ms với block M=3. Phần chênh ~11.5 ms gồm: LM head chạy 3 lần riêng từng hàng (3 kernel gemv bf16, ~3.3 ms mỗi lần, ~10 ms tổng, +6.6 ms so với M=1), các linear NF4 (Triton M=3 11.1 ms, so với ~8.9 ms của BNB M=1), và các phép rowwise (copy/cat, RMSNorm, attention từng hàng) +~2.7 ms. Phần còn lại giữa thời gian GPU và wall time là GPU rảnh do khởi chạy kernel.

**LM head lười** (`scripts/benchmark_b2_lazy_lm_head.py`, `lazy_lm_head_gsm8k10_v1.json`): chạy decoder (`model.model`) một lần cho cả khối rồi chỉ chiếu LM head từng hàng cho đến hàng bị reject đầu tiên; mỗi hàng vẫn đúng một lời gọi BNB M=1 như trước nên số học không đổi, KV không đổi. Trên 10 prompt GSM8K dev (128 token, 3 round xoay thứ tự): token khớp BNB 10/10, KV prefix SHA256 khớp BNB tại mọi prefix không-terminal (86–119 prefix mỗi prompt) 10/10, số hàng LM head thực tính 43/75 hàng block ở prompt đầu (khoảng 57%). Tốc độ so với BNB: B2 hiện tại 1.113x, B2 lười 1.161x; lười nhanh hơn B2 1.043x (geomean, mỗi prompt từ 1.03x đến 1.06x). Giới hạn: 10 prompt GSM8K dev, chưa có CI, chưa chạy HumanEval/CNN-DM, chưa chạy reject-path audit riêng; held-out GSM8K 300–349 đã dùng nên không thể dùng lại để xác nhận phiên bản này, cần cohort mới nếu muốn tuyên bố số liệu cuối cùng.

**LM head lười, mở rộng 30 prompt mỗi cohort** (`lazy_lm_head_{gsm8k30,humaneval30}_v1.json`, bootstrap theo prompt 10000 lần seed 0 trong `lazy_lm_head_30prompt_bootstrap_v1.json`; 128 token, 3 round, tập dev):

| Cohort | B2 vs BNB (CI95) | B2 lười vs BNB (CI95) | Lười/B2 (CI95) | Token khớp BNB | KV prefix khớp BNB |
|---|---|---|---|---:|---:|
| GSM8K dev (30) | 1.116x [1.095, 1.139] | 1.166x [1.145, 1.188] | 1.045 [1.037, 1.051] | 30/30 | 30/30 |
| HumanEval dev (30) | 1.212x [1.170, 1.260] | 1.249x [1.208, 1.296] | 1.031 [1.025, 1.037] | 30/30 | 30/30 |

Không có prompt nào chậm hơn BNB (60/60 trên 1.0x). Lười chậm hơn B2 thường ở 1/60 prompt (HumanEval), nhanh hơn ở 59/60. Reject-path audit riêng cho phiên bản lười (`scripts/audit_activationvec_reject_path_lazy_b2.py`, `lazy_reject_path_{cnndm_003,gsm8k_000,humaneval_0}_v1.json`): 5/5 scenario đạt ở cả 3 prompt. Wrapper dùng lại script audit cũ nên trường `source_sha256` trong JSON là của script audit gốc, không phải wrapper. Giới hạn: chỉ tập dev; CI chỉ phản ánh biến thiên giữa prompt; held-out GSM8K 300–349 đã dùng nên chưa có xác nhận held-out cho phiên bản lười.

## 13. Forward bị giới hạn bởi CPU khởi chạy kernel (03/11/2026, tập dev, 1 prompt)

`scripts/profile_b2_launch_bound.py` (`launch_bound_gsm8k_000_v1.json`): đo thời gian CPU đẩy một forward vào hàng đợi GPU (enqueue) so với tổng wall.

| Forward | Enqueue CPU | Wall | Tỷ lệ CPU/wall |
|---|---:|---:|---:|
| BNB M=1 | 18.4 ms | 22.0 ms | 84% |
| Block M=3, LM head 3 hàng | 25.6 ms | 35.7 ms | 72% |
| Block M=3, LM head 1 hàng | 25.7 ms | 29.2 ms | 88% |

Micro-benchmark một lớp (`down_proj`, 300 lần): BNB M=1 enqueue 28.5 µs, wall 72 µs; override Triton M=3 enqueue 25.9 µs, wall 89.8 µs; riêng lời gọi kernel Triton chỉ ~10 µs CPU. Nghĩa là: (1) lời gọi Triton không phải điểm nghẽn CPU; chi phí CPU nằm ở Python/HF/wrapper quanh các lớp và số lượng phép toán nhỏ (block M=3 phát khoảng 1.8 lần số kernel của M=1); (2) kernel Triton M=3 chậm hơn BNB M=1 khoảng 25% khi chạy riêng lẻ (89.8 vs 72 µs), tương ứng ~2 ms trên cả 252 lớp, nhưng vì forward bị giới hạn bởi CPU nên tối ưu kernel chỉ thu lại phần nhỏ; (3) tối ưu bớt kernel GPU không làm forward nhanh hơn nếu CPU vẫn là nút thắt. Hướng có thể thu thêm thật sự là giảm overhead CPU (ví dụ CUDA graph hoặc gộp phép rowwise), nhưng cả hai đều phức tạp: HF static cache ở đây dùng độ dài prefix thay đổi theo bước nên khó chụp graph, và gộp phép rowwise phải chứng minh bitwise bằng với từng hàng riêng.

**Lưu ý quan trọng cho việc diễn giải tốc độ:** baseline "BNB gốc" ở mọi bảng trên là HF eager, tức cũng bị giới hạn bởi CPU. Nếu baseline được tăng tốc bằng CUDA graph hay torch.compile, thời gian mỗi token của baseline sẽ giảm mạnh còn B2 thì không hưởng lợi theo cách đó, nên mức tăng ~1.12–1.25x chỉ đúng so với BNB eager và có thể thu hẹp hoặc đảo chiều so với baseline tối ưu hơn. Cần nêu rõ khi báo cáo, và nên đo thêm baseline mạnh hơn nếu muốn tuyên bố tốc độ.

## 14. Baseline mạnh hơn và mở rộng bằng chứng exactness (03/11/2026, tập dev)

**Hướng 1 — baseline đã biên dịch** (`scripts/benchmark_compiled_bnb_baseline.py`, `compiled_bnb_baseline_gsm8k8_v1.json`; 8 prompt GSM8K dev, 128 token, 3 round). HF `generate` tự bỏ qua compile cho mô hình lượng tử hóa ("unable to meet the criteria for compilation"), nên tôi tự dựng vòng decode với `StaticCache` của HF và `torch.compile(mode="reduce-overhead")`; warm-up/biên dịch ~34 s. Kết quả: BNB biên dịch nhanh hơn BNB eager **1.43x** (geomean), nhưng token **chỉ giống eager ở 6/8 prompt** (lệch ở vị trí 101 và 1 ở hai prompt còn lại), tức không phải tham chiếu bitwise của BNB gốc. So sánh gián tiếp: B2 lười đạt ~1.12–1.25x so với BNB eager trên tập dev, trong khi BNB biên dịch đạt ~1.43x; vì vậy **B2 chậm hơn BNB biên dịch** (ước lượng thô, ghép từ hai bảng khác prompt/lần chạy, chưa đo cạnh nhau trong cùng lượt chạy). Kết luận hợp lý: B2 cho tốc độ ~1.1–1.25x *với bảo đảm exact so với BNB eager*; nếu chấp nhận baseline không exact đã biên dịch thì baseline đó nhanh hơn. Chưa thử kết hợp compile với giải mã suy đoán.

**Hướng 2a — mô hình khác** (`scripts/run_with_model.py` mở rộng danh sách cho phép chỉ trong tiến trình thí nghiệm; mặc định fail-closed trong `src` giữ nguyên). Dùng lại nguyên audit KV closed-loop (có/không fallback, 64 token, 4 prompt GSM8K dev) và reject-path lười (5 scenario, `gsm8k_000`):

| Mô hình (revision) | Hidden/MLP | KV closed-loop (8 lượt) | Reject-path (5 scenario) |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct (7ae55760…) | 896 / 4864 | 8/8 token+KV khớp BNB | 5/5 |
| Qwen2.5-1.5B-Instruct (989aa798…) | 1536 / 8960 | 8/8 token+KV khớp BNB | 5/5 |

File: `model05b_*`, `model15b_*`. Giới hạn: 4 prompt, 64 token, chỉ họ Qwen2, cùng GPU/toolchain; chưa đo tốc độ trên hai mô hình này.

**Hướng 2b — nhánh EOS** (`scripts/audit_b2_eos_termination.py`, `eos_termination_synth_chat_v1.json`, mô hình 3B): trước đó 0/90 lượt dev dừng bằng EOS. Thêm 4 prompt tổng hợp có `<|im_end|>` lặp và 8 prompt chat câu trả lời ngắn: cả 12 đều dừng bằng EOS (1–72 token), bản thường và bản lười đều khớp BNB gốc 12/12 (BNB tham chiếu dừng ở EOS); 3 prompt có chu kỳ mà draft chứa EOS. Giới hạn: số prompt nhỏ, thế hệ ngắn; không kiểm KV từng bước ở nhánh này.

Thông tin còn thiếu: kiểm KV từng bước khi dừng bằng EOS, độ dài ≥256 token, tốc độ trên mô hình khác, so sánh cạnh nhau với baseline biên dịch trong cùng lượt chạy.

**Cập nhật cho mục 14 — đo cạnh nhau cùng lượt chạy** (`scripts/benchmark_b2_vs_compiled_bnb.py`, `b2_lazy_vs_compiled_bnb_gsm8k12_v1.json`; 12 prompt GSM8K dev, 128 token, 3 round xoay thứ tự xen kẽ theo từng prompt, trong cùng một tiến trình): so với BNB eager, B2 lười nhanh hơn **1.187x** và BNB biên dịch nhanh hơn **1.447x**; B2 lười chỉ bằng **0.820x** tốc độ BNB biên dịch (chậm hơn ở 12/12 prompt, từ 0.76x đến 0.88x). B2 lười khớp BNB eager 12/12; BNB biên dịch chỉ khớp eager 9/12. Thay thế ước lượng gián tiếp ở trên. Chưa có CI; chưa thử compile kết hợp giải mã suy đoán.

**Kiểm KV từng bước ở nhánh EOS và sinh dài** (`scripts/audit_b2_eos_kv_long.py`, `eos_kv_long_v1.json`; mô hình 3B, ngân sách 384 token, tham chiếu BNB gốc có dừng ở EOS và băm KV từng bước): 18 prompt gồm 6 câu hỏi GSM8K dạng chat, 8 prompt chat ngắn và 4 prompt tổng hợp có EOS trong lịch sử. 17/18 dừng bằng EOS, 3 prompt sinh ≥256 token (tối đa 384); token khớp BNB gốc và KV prefix SHA256 khớp tại mọi prefix không-terminal ở cả bản thường lẫn bản lười, **18/18 và 18/18** (tới 254 prefix mỗi prompt). Giới hạn: không kiểm KV ở trạng thái cuối sau EOS (chỉ các prefix không-terminal), cohort nhỏ, chỉ tập dev, một GPU.

## 15. Held-out thứ hai GSM8K 400–449 cho phiên bản lười — cấu hình đăng ký trước (03/11/2026)

Cohort `data/benchmarks/gsm8k_exactness_holdout2.jsonl` (GSM8K `main` test, chỉ số 400–449, dataset revision `740312ad…`), manifest `experiments/exact_quantized_feasibility/frozen_gsm8k_400_449.json`: 0 trùng ID/prompt với mọi benchmark cục bộ, 0 ID xuất hiện trong thư mục `experiments`, `data`, `scripts`, `src`. Chọn theo chỉ số công khai trước khi chạy bất kỳ bộ giải mã nào trên cohort này. Mã khóa ở commit chứa mục này. Chạy đúng một lần mỗi lệnh, không đổi cấu hình sau khi thấy kết quả:
1. `scripts/benchmark_b2_lazy_lm_head.py`: B2 lười (`ngram_size=3`, `k=2`, `physical_rows=3`, `bnb_rowwise`, fallback bật, LM head lười), 128 token, 3 round; chỉ tiêu chính: số prompt khớp BNB eager (token ID) và khớp KV prefix từng bước, geomean tốc độ vs BNB eager kèm CI bootstrap theo prompt (10000 lần, seed 0).
2. `scripts/benchmark_b2_vs_compiled_bnb.py`: cùng cohort, đo cạnh nhau B2 lười, BNB eager, BNB biên dịch; chỉ tiêu: tỷ lệ tốc độ B2 lười/biên dịch.
Kết quả báo cáo nguyên trạng.

**Kết quả (một lần chạy mỗi lệnh, nguyên trạng; `heldout2_gsm8k_400_449_lazy_v1.json`, `heldout2_gsm8k_400_449_vs_compiled_v1.json`, bootstrap trong `heldout2_gsm8k_400_449_bootstrap_v1.json`, 10000 lần, seed 0):**

| Chỉ tiêu (50 prompt, 128 token, 3 round) | Giá trị |
|---|---|
| B2 lười khớp BNB eager (token ID) | **50/50** |
| B2 lười khớp KV prefix từng bước (SHA256, mọi prefix không-terminal) | **50/50** |
| B2 lười vs BNB eager (CI95) | **1.154x [1.136, 1.172]**; không prompt nào dưới 1.0x |
| B2 eager-LM-head vs BNB eager (CI95) | 1.097x [1.080, 1.113] |
| Lười/B2 (CI95) | 1.052 [1.044, 1.060] |
| BNB biên dịch vs eager (CI95) | 1.285x [1.272, 1.297]; chỉ khớp eager 25/50 prompt |
| B2 lười vs BNB biên dịch (cùng lượt chạy, CI95) | **0.907x [0.893, 0.920]**; chậm hơn ở 49/50 prompt |
| B2 lười khớp BNB eager trong lượt đo cạnh nhau | 50/50 |

Kết luận: trên cohort held-out mới, B2 lười giữ exact so với BNB eager ở mọi prompt (token và KV từng bước), nhanh hơn eager khoảng 15%, và vẫn chậm hơn baseline biên dịch không exact (khoảng 9%). Lưu ý: BNB biên dịch chỉ nhanh hơn eager 1.29x ở đây so với 1.45x trên tập dev, và giữ token trùng eager chỉ 25/50 prompt, nên tỷ lệ B2/biên dịch phụ thuộc cohort (0.82x dev, 0.91x held-out); chưa tìm nguyên nhân sự chênh này. Giới hạn: một GPU, một mô hình, K=2, 128 token, chỉ GSM8K; CI chỉ phản ánh biến thiên giữa prompt; cohort không loại trừ được việc dữ liệu công khai đã lọt vào dữ liệu huấn luyện. Cohort này đã dùng, không dùng lại để chọn cấu hình.

## 16. Tốc độ trên mô hình khác, lập luận toán tử, dọn dẹp (03/11/2026)

**Tốc độ + exactness trên 0.5B và 1.5B** (`scripts/run_with_model.py` + `benchmark_b2_lazy_lm_head.py`, 10 prompt GSM8K dev, 128 token, 3 round; `model05b_lazy_speed_gsm8k10_v1.json`, `model15b_lazy_speed_gsm8k10_v1.json`; chưa có CI):

| Mô hình | Token khớp BNB | KV prefix khớp BNB | B2 vs BNB eager | B2 lười vs BNB eager | Lười/B2 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 10/10 | 10/10 | 1.207x | 1.243x | 1.030 |
| Qwen2.5-1.5B-Instruct | 10/10 | 10/10 | 1.125x | 1.163x | 1.034 |

(3B cùng họ: 1.166x lười trên 30 prompt GSM8K dev.) Giới hạn: 10 prompt dev, cùng họ Qwen2, cùng GPU và toolchain.

**Lập luận toán tử** (`experiments/exact_quantized_feasibility/B2_OPERATOR_ARGUMENT.md`): phân loại từng toán tử thành [S] theo cấu trúc mã, [E] thực nghiệm, [O] còn mở. Còn mở chính: kernel Triton bằng đúng thứ tự rút gọn của kernel bitsandbytes M=1 (chỉ có bằng chứng thực nghiệm), dispatch PyTorch không phụ thuộc số hàng, ổn định qua phiên bản toolchain/GPU, kiểm ở mức PTX.

**Dọn dẹp:** README cập nhật; `scripts/validate_activationvec_b2_manifest.py` kiểm hash nguồn/bằng chứng và phiên bản runtime; manifest `activationvec_b2_v2` thay thế v1 (v1 giữ nguyên).

## 17. Ngữ cảnh dài (03/11/2026, tập dev, mô hình 3B)

Trước đó mọi bằng chứng dùng prompt tối đa 605 token (đo bằng tokenizer). `scripts/make_long_context_prompts.py` dựng `data/benchmarks/long_context_dev.jsonl`: 12 prompt ghép nhiều bài CNN/DM dev (`cnndm_ext150`) tới 512, 1024, 2048, 3072 token (3 biến thể mỗi độ dài), kèm chỉ dẫn tóm tắt. Chạy `benchmark_b2_lazy_lm_head.py` (qua `scripts/run_long_context_lazy_benchmark.py`, thay hàm đo BNB eager bằng bản dung lượng cache lớn hơn vì `_common.run_vanilla_timed` mặc định chỉ 2048 token; lần chạy đầu `_v1` hỏng ở 2048/3072 vì lý do này, còn 512/1024 của `_v1` hợp lệ), 128 token sinh, 2 round; kết quả `long_context_{0512,1024,2048,3072}_lazy_v2.json`.

| Độ dài prompt | Prompt | Token khớp BNB | KV prefix khớp BNB (mọi bước) | B2 vs eager | B2 lười vs eager |
|---:|---:|---:|---:|---:|---:|
| 512 | 3 | 3/3 | 3/3 | 1.116x | 1.159x |
| 1024 | 3 | 3/3 | 3/3 | 1.047x | 1.121x |
| 2048 | 3 | 3/3 | 3/3 | 0.994x | 1.041x |
| 3072 | 3 | 3/3 | 3/3 | 1.000x | 1.055x |

(82–120 prefix KV được so sánh mỗi prompt; KV băm gồm toàn bộ prefix lời nhắc.) Kết luận: **exactness giữ nguyên tới ~3.2k vị trí** (không thấy lệch token hay KV khi độ dài KV tăng), nhưng **lợi thế tốc độ co lại khi ngữ cảnh dài**: B2 thường chỉ còn ngang BNB eager ở 2048–3072 và bản lười còn 1.04–1.06x; hai prompt 1024 và 2048 của B2 thường dưới 1.0x (0.97x, 0.96x). **Nguyên nhân đã đo (hiệu chỉnh giả thuyết ban đầu về attention):** chi phí một forward gần như không đổi theo độ dài (`block_kernel_profile_long_*`, `launch_bound_long_*`, `phase_profile_long_v1.json`): wall BNB M=1 23.2 ms ở 512 so với 23.5 ms ở 3072, block M=3 với 1 hàng LM head 30.9 so với 30.1 ms; thời gian GPU tăng nhẹ (block 28.6→32.2 ms, nhóm attention 0.9→2.1 ms) nhưng bị che bởi giới hạn CPU. Hai yếu tố thực sự gây co lại: (1) acceptance trên prompt tóm tắt dài thấp (0.24–0.29 ở 3072 so với 0.32–0.64 ở 512; token mỗi block cycle 1.5 so với 1.6–2.3, dưới ngưỡng hoà vốn ~1.65), (2) prefill 1.1 s ở 3072 (so với 0.19 s ở 512) là chi phí chung cộng vào cả hai phía và làm loãng tỷ lệ tổng (~25% tổng thời gian). Tốc độ decode riêng của BNB eager gần như không đổi (24.6→25.0 ms/token). Vì vậy bằng chứng hiện tại không cho thấy độ dài ngữ cảnh tự nó làm B2 chậm đi; đây là hiệu ứng của tác vụ (acceptance) và của prefill. Chưa tách tuyệt đối vì chưa có prompt dài mà acceptance cao; tối ưu `.contiguous()` của attention vì thế ít giá trị hơn dự kiến. Giới hạn: 12 prompt, 3 mỗi độ dài, chưa CI, chỉ tập dev, prompt ghép nhân tạo từ một nguồn (tóm tắt), 128 token sinh, chưa thử trên 3072 (GPU 6 GB) hay mô hình khác.

## 18. Chính sách chọn draft (03/11/2026, tập dev, B2 lười, k=2)

`scripts/benchmark_b2_draft_policies.py` (`draft_policies_{gsm8k20,cnndm20,humaneval20}_v1.json`; 20 prompt dev đầu mỗi cohort, 128 token, 2 round, tất cả chính sách khớp BNB eager 20/20 ở mọi cohort). Chính sách chỉ đổi token nào được đề xuất; mọi đề xuất vẫn được kiểm bằng cùng đường exact. `base` giống hệt PLD hiện tại (0 khác biệt trên 600 trường hợp ngẫu nhiên). Tốc độ vs BNB eager (geomean):

| Chính sách | GSM8K | CNN/DM | HumanEval | Block cycle trung bình (GSM8K) |
|---|---:|---:|---:|---:|
| `vote` (bỏ phiếu giữa các lần khớp, ngram≤3) | 1.176x | 1.179x | 1.239x | 29.9 |
| `base` (hiện tại) | 1.159x | 1.175x | 1.230x | 29.9 |
| `rep2` (n-gram 2 token cần ≥2 lần khớp) | 1.142x | 1.166x | 1.232x | 23.5 |
| `vote_s3` (bỏ phiếu, n-gram ≥3, ngram≤4) | 1.144x | 1.175x | 1.225x | 18.2 |
| `s3` (n-gram ≥3) | 1.138x | 1.172x | 1.213x | 18.2 |
| `s4` (n-gram ≥3, ngram≤4) | 1.135x | 1.171x | 1.223x | 18.1 |
| `s5` (n-gram ≥4, ngram≤5) | 1.104x | 1.146x | 1.209x | 11.5 |

Kết quả: **lọc bỏ đề xuất yếu không giúp** (ít block gần một nửa nhưng không nhanh hơn, `s5` chậm hơn); giả thuyết ban đầu bị bác bỏ. `vote` đứng đầu ở cả ba cohort nhưng chỉ hơn `base` 0.4–1.5% (cùng cỡ nhiễu đo). **Xác nhận trên prompt dev chưa dùng để chọn** (`draft_policies_confirm_*_v1.json`, tóm tắt `draft_policies_confirm_bootstrap_v1.json`; chỉ `base` và `vote`, 3 round, tỷ số ghép cặp theo prompt, bootstrap 10000 lần seed 0; tất cả khớp BNB eager):

| Cohort | Prompt | `base` | `vote` | `vote`/`base` (CI95) |
|---|---:|---:|---:|---|
| GSM8K 20–49 | 30 | 1.174x | 1.176x | 1.001 [0.995, 1.007] |
| HumanEval 20–49 | 30 | 1.311x | 1.315x | 1.003 [0.996, 1.011] |
| CNN/DM 20–39 | 20 | 1.155x | 1.159x | 1.003 [0.996, 1.009] |
| Gộp | 80 | — | — | **1.002 [0.998, 1.007]** |

Kết luận: mức nhỉnh 0.4–1.5% của `vote` trên 20 prompt đầu **không tái lập** (mọi CI chứa 1); **giữ `base` (PLD hiện tại)**. Chỉnh cách chọn n-gram (lọc, bỏ phiếu, ngưỡng lặp) không cải thiện được tốc độ; acceptance cần nguồn draft khác chứ không phải chính sách chọn n-gram. Tốc độ tuyệt đối khác nhau giữa các tập prompt (HumanEval 20–49 đạt 1.31x so với 1.23x ở 0–19), nên so sánh tuyệt đối giữa cohort không có ý nghĩa nếu không cùng prompt. Giới hạn: chỉ tập dev, k=2, một GPU, chưa thử nguồn draft ngoài PLD (ví dụ mô hình nhỏ, bảng n-gram từ văn bản huấn luyện), cũng chưa thử gộp nhiều ứng viên vào một block.

## 19. Đóng bốn việc còn lại (03/11/2026)

**(a) Thứ tự rút gọn của kernel so với bitsandbytes** (`scripts/emulate_nf4_gemv_orders.py`, `nf4_gemv_order_emulation_v1.json`): đọc `csrc/kernels.cu` tag 0.50.0 (`kgemm_4bit_inference_naive<half,128,16>`), mã nguồn cộng tích lũy **tuần tự** (`local_C += p_k`), còn kernel Triton cộng **theo cặp** (`acc + (p0+p1)`); về lý thuyết hai thứ tự này khác nhau. Mô phỏng cả hai thứ tự trong fp32 trên 28,672,000 đầu ra ngẫu nhiên đuôi nặng (lớp 0, 9, 18, 27, 35, đủ bảy phép chiếu): kernel BNB thật và kernel Triton **đều khớp mô phỏng cộng theo cặp 28,672,000/28,672,000 và đều khác mô phỏng tuần tự ở 398 đầu ra**. Phép thử đủ nhạy để phân biệt hai thứ tự và chọn đúng thứ tự theo cặp cho cả hai. Cận trên 95% của tỷ lệ lệch mỗi đầu ra là ~1.0e-7 (quy tắc ba). Chưa giải thích được vì sao bản nhị phân hành xử theo cặp (biến đổi của trình biên dịch, cờ build, hay khác với mã nguồn đã đọc) nên **mọi lần nâng cấp bitsandbytes phải chạy lại script này**. Đã cập nhật `B2_OPERATOR_ARGUMENT.md`. Chưa kiểm ở mức PTX/SASS.

**(b) Giảm chi phí CPU** (kết quả âm): cProfile một forward block cho thấy RMSNorm từng hàng chiếm ~30% thời gian CPU và điều phối Triton ~26 µs/lần; thử gọi thẳng kernel đã biên dịch (cùng nhị phân; bằng nhau từng bit trên 5,160,960 phần tử) nhưng enqueue 252 phép chiếu gần như không đổi (7.22 ms chuẩn so với 7.59 ms trực tiếp), nên bỏ script. RMSNorm và attention từng hàng là các chuỗi phép PyTorch nhỏ; giảm chúng đòi hỏi đổi tập kernel (mất cấu trúc "từng hàng giống tham chiếu"). Không có cải thiện thực.

**(c) Block lớn hơn (M=4, M=5)** (`scripts/simulate_block_size.py`, `block_size_simulation_v1.json`, chỉ mô phỏng CPU): chạy PLD hiện tại trên chuỗi tham chiếu BNB đã ghi để đếm đúng token được chấp nhận, và dùng mô hình chi phí `F − h + s·(hàng−1) + h·(hàng LM head thực tính)` với F=24.0 ms, h=3.3 ms, s hiệu chỉnh để K=2 khớp tốc độ B2 lười đã đo (s = 2.46 ms/hàng ở GSM8K dev 30 prompt, 2.03 ms/hàng ở HumanEval dev 30 prompt). Dự đoán so với eager: GSM8K K=1/2/3/4 = 1.131/1.166/1.164/1.154x; HumanEval = 1.178/1.249/1.265/1.281x; so với K=2, mức thay đổi dự đoán là −1% đến +0% ở GSM8K và tối đa +2.6% ở HumanEval (K=4) với s hiệu chỉnh; nếu s giảm một nửa thì +1.1% (GSM8K) và +4.1% (HumanEval). Mức lợi nhỏ và không chắc chắn, nên **không đáng viết kernel M=5**. Giới hạn: mô hình tuyến tính, hiệu chỉnh chỉ từ K=2, bỏ qua chi phí kernel tăng theo số hàng, chưa kiểm thực nghiệm với K>2 (kernel chỉ có M=3).

**(d) Tái lập:** `reproduce.sh` thêm kiểm tra manifest và chế độ `--b2` (reject-path, KV từng bước, benchmark B2 lười); chế độ `--quick` chạy: 78 test qua. Manifest v3 thay thế v2 (v2 không còn hợp lệ vì `B2_OPERATOR_ARGUMENT.md` được sửa sau khi ghim).

## 20. VRAM: B2 không dùng thêm bộ nhớ (03/11/2026, tập dev, mô hình 3B)

`scripts/measure_vram.py` (`vram_{bnb,b2_lazy}_{short,long3072}_v1.json`): mỗi phương pháp chạy trong một tiến trình riêng, tải mô hình, chạy một lượt khởi động, đặt lại số đo đỉnh rồi đo lượt thứ hai (128 token sinh). Cả hai phía dùng cùng dung lượng cache tĩnh (độ dài prompt + 136 token). Token của hai phương pháp giống hệt (SHA256 trùng) ở cả hai prompt.

| Prompt | Phương pháp | Đỉnh cấp phát (MiB) | Đỉnh dự trữ (MiB) | GPU dùng cuối (MiB) |
|---|---|---:|---:|---:|
| `gsm8k_000` (75 token) | BNB eager | 2007.90 | 2042 | 2226.1 |
| | B2 lười | 2007.97 | 2042 | 2226.1 |
| `long_3072_a` (3071 token) | BNB eager | 2994.61 | 3356 | 3540.1 |
| | B2 lười | 2994.68 | 3356 | 3540.1 |

Chênh lệch đỉnh cấp phát chỉ **0.07 MiB** (khoảng 0.003%); đỉnh dự trữ và bộ nhớ GPU thực dùng giống hệt nhau; không có mô hình draft nào. Mô hình sau khi tải chiếm 1969 MiB ở cả hai. Mục tiêu "không thêm VRAM" được xác nhận ở mức cấp phát tensor và tổng bộ nhớ thiết bị. Giới hạn: một mô hình (3B), hai prompt, batch 1, bộ nhớ đỉnh phụ thuộc cache tĩnh được cấp phát theo độ dài; chưa đo trên 0.5B/1.5B.

## 21. Nguồn gốc của sai lệch, hiệu chỉnh vai trò kernel Triton, thăm dò CUDA graph (03/11/2026)

**(a) Toán tử nào làm vỡ tính exact** (`scripts/audit_op_shape_sensitivity.py`, `op_shape_sensitivity_gsm8k20_v1.json`; 20 prompt GSM8K dev, 16 khối ba token teacher-forced mỗi prompt, 960 hàng; đường chuẩn không vá; mỗi toán tử được gọi lại trên cùng đầu vào từng hàng một và so từng bit): phép chiếu NF4 (7 loại) 0 lệch trong 990,904,320 phần tử; RMSNorm lớp 87/141,557,760; RMSNorm cuối 0/1,966,080; LM head 281,534/145,858,560 (0.19%); attention SDPA 13,196,548/70,778,880 (18.6%, 11502/11520 lượt gọi). Cuối-cuối: 960/960 hàng logits không bằng từng bit; 4/960 (0.42%) đổi argmax (margin ≤ 0.125, một hàng hòa); median max|Δlogit| 0.219, lớn nhất 1.36. Giải thích có thể: bitsandbytes 0.50.0 có kernel `gemm_4bit` tùy chọn cho M nhỏ (đọc mã cài đặt), nên kết quả phụ thuộc phiên bản. **Hiệu chỉnh:** mô tả trước đó coi kernel Triton M=3 là điều kiện của tính exact; điều đó không đúng. B2 lười không dùng kernel Triton (`scripts/run_b2_without_triton_linears.py`, `b2_lazy_no_triton_gsm8k10_v1.json`, 10 prompt GSM8K dev) vẫn khớp BNB 10/10 (token và KV) và đạt 1.117x so với 1.161x khi có kernel; kernel chỉ là tối ưu tốc độ khoảng 4%. Đã sửa README, `B2_OPERATOR_ARGUMENT.md`, báo cáo kỹ thuật và chương luận văn.

**(b) Dư địa của một đường exact rẻ hơn** (đo bằng `profile_b2_launch_bound.measure`, 1 prompt): forward chuẩn M=3 không exact tốn 25.5 ms (M=1: 21.9 ms), đường B2 hiện tại 29.0 ms (LM head lười một hàng) hoặc 36.4 ms (đủ hàng). Với mô phỏng đã hiệu chỉnh, một đường exact có chi phí gần khối chuẩn (eager) cho K=2: 1.228x (GSM8K), 1.320x (HumanEval) so với 1.166x/1.249x hiện tại, tức chỉ +5–6%.

**(c) Thăm dò CUDA graph** (`scripts/probe_cuda_graph_decode.py`, `cuda_graph_probe_m{1,3}_v1.json`): chụp thô (không torch.compile) một bước decode của mô hình bitsandbytes eager với `StaticCache`: M=1 chụp được, logits bằng từng bit ở 64/64 bước, 22.5 ms (eager) → 17.3 ms (graph, 1.30x); khối chuẩn M=3 (không exact) 27.4 → 23.9 ms. Ước lượng (mô phỏng, chưa cài đặt) cho vòng B2 chạy phần thân decoder trong graph với LM head lười ngoài graph: 1.36–1.48x so với eager (baseline biên dịch không exact: 1.29–1.45x), nhưng chỉ 1.04–1.14x so với baseline graph exact; chi phí decoder M=3 trong graph 21.7–24.0 ms là ước lượng từ số đo kernel. Hướng A thực tế vì vậy là cải thiện khiêm tốn, không phải bước nhảy; hướng C (đo lường, phân loại nguyên nhân trên nhiều backend) là phần có độ mới cao hơn.

---

*Ngày cập nhật: 02/11/2026. Tất cả kết quả trên development prompts, một GPU RTX 4050 Laptop 6 GB, Qwen2.5-3B-Instruct NF4, BNB 0.50.0, Triton 3.6.0, PyTorch 2.11.0.*

## 22. Hướng C và A: kiểm toán theo toán tử, bản sửa từng hàng, kiểm chứng độc lập (03–05/11/2026)

Chi tiết, bảng và giới hạn nằm ở `RESEARCH_DIRECTIONS_A_C.md` (Mục 1.8–1.14, 2.5b–2.5c), `LITERATURE_NOTES_DIRECTION_C.md` và `native_bnb_profiles/C_SHAPE_AUDIT_SUMMARY.md` (sinh bởi `scripts/shape_audit.py summarize`). Tóm tắt:

- **Công cụ:** `scripts/shape_audit.py` (`ops`, `trajectory`, `plan`, `summarize`). Với một mô hình và backend, `ops` đo toán tử nào cho kết quả khác từng bit khi chạy khối M hàng so với từng hàng; `plan` liệt kê lớp toán tử cần chạy từng hàng; `trajectory --rowwise` kiểm tra PLD với danh sách đó so với greedy M=1 tuần tự.
- **Kết quả chính (bitsandbytes 0.50.0, RTX 4050):** danh sách `plan` đưa số chuỗi lệch quỹ đạo từ 22–82% (PLD ngây thơ, tùy cấu hình) về 0 trong mọi lượt chạy của mã đã sửa, trên GSM8K, HumanEval và CNN/DailyMail (kể cả 60 prompt `cnndm_ext150` chưa dùng cho thiết kế), với Qwen2.5 0.5B/1.5B/3B, OLMo-2-1B, Phi-3.5 (bnb) và Qwen2.5-3B (torchao INT4). Danh sách khác nhau theo mô hình và backend, đúng như kiểm toán dự đoán. Khoảng Wilson 95% của mỗi ô 0/50–0/60 là khoảng [0, 6–7]%.
- **Phụ thuộc phiên bản và kernel:** phép chiếu NF4 bất biến theo hình dạng khi M ≤ 3 trên bitsandbytes 0.50.0 nhưng vỡ từ M=4; trên 0.49.2 vỡ ngay từ M=3. torchao INT4 (tinygemm) không vỡ đến M=4. Thay đổi này tương ứng với PR #1949 của bitsandbytes.
- **Đảo ngược/thu hồi đã xảy ra:** (i) HQQ+GemLite không lặp lại được giữa các lần chạy (tham chiếu M=1 lệch 5/50 chuỗi; đầu vào giống nhau cho đầu ra khác ở 64% phần tử), nên mọi số liệu HQQ về độ nhạy theo hình dạng bị thu hồi. (ii) Bản sửa SDPA từng hàng ban đầu sai với mô hình GQA ở ngữ cảnh dài (dùng K/V đã nhân bản thay vì K/V gốc với `enable_gqa=True`); đã sửa và kiểm chứng lại.
- **Hướng A (CUDA graph):** vòng giải mã exact trong graph đúng từng bit so với `StaticCache` M=1 (30/30 prompt); CUDA graph cho baseline exact nhanh 1.40x so với eager, giải mã suy đoán K=2 chỉ thêm 1.009x; điều kiện dừng A4 (dưới 1.05x) đã kích hoạt, không chạy A5. Tham chiếu của hướng A (`StaticCache` M=1) khác tham chiếu của B1/B2.
- **Chưa làm / cần người dùng:** Llama 3.2 (cần token HF), GPU khác và phiên bản bitsandbytes 0.50.1/0.50.2 (C7), GPTQ/AWQ (cần biên dịch kernel), mô hình lớn hơn 3B, đọc nguyên văn phần Evaluation và Related Works của LLM-42.
- **Giới hạn chung:** một GPU, K ≤ 3, greedy, batch 1, mô hình ≤ 3B, mỗi ô chạy một lần; "0 ca lệch" là quan sát trên mẫu, không phải chứng minh exact.
