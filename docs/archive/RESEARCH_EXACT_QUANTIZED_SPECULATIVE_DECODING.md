# Đề cương nghiên cứu: Speculative decoding chính xác có chứng minh dưới kernel lượng tử hóa phụ thuộc shape

**Trạng thái:** kế hoạch nghiên cứu, chưa phải kết quả đã chứng minh.
**Ngày soạn:** 23-10-2026.
**Nền tảng:** mã và dữ liệu của dự án ZASSD tại repository này.

**Cập nhật triển khai 26-10-2026:** [Kết quả thăm dò và giới hạn hiện tại](experiments/exact_quantized_feasibility/FINDINGS.md), [định lý có điều kiện và nghĩa vụ chứng minh còn thiếu](experiments/exact_quantized_feasibility/PROOF_OBLIGATIONS.md), [quy trình đánh giá và cohort đã đóng băng](experiments/exact_quantized_feasibility/PROTOCOL.md). Đây chưa phải thuật toán exact có chứng minh trên các backend 4-bit hiện tại.

**B1 Qwen2 prototype:** [Hợp đồng operator fixed-row và miền hỗ trợ hiện tại](experiments/exact_quantized_feasibility/QWEN2_FIXED_ROW_OPERATOR_CONTRACT.md). Hợp đồng là bản thực thi/kiểm tra, chưa phải chứng minh cho các kernel BNB và attention trên toàn miền.

**Kế hoạch thực thi 26-10-2026:** [Các cổng quyết định, thay đổi mã, tiêu chí nghiệm thu và thứ tự đánh giá](RESEARCH_EXECUTION_PLAN_EXACT_QUANTIZED.md).

## 1. Mục tiêu và tuyên bố nghiên cứu

Mục tiêu là thiết kế speculative decoding cho mô hình lượng tử hóa 4-bit sao cho chuỗi token sinh ra **trùng từng token** với greedy decoding tuần tự của một hệ tham chiếu được định nghĩa chính xác, kể cả khi đường xác minh nhiều token và đường tham chiếu một token dùng kernel, kích thước ma trận và thứ tự cộng khác nhau. Thuật toán phải xử lý cả sai lệch trạng thái KV tích lũy sau nhiều chu kỳ, không chỉ so sánh logits tại một vị trí.

Kết quả mong muốn gồm hai phần tách biệt:

1. **Định lý bảo đảm:** dưới các giả định được kiểm tra hoặc thực thi, mọi token được commit bằng đường nhanh đều giống token greedy của hệ tham chiếu; sau mỗi lần commit, trạng thái dùng cho tương lai vẫn thỏa bất biến đã nêu.
2. **Hệ thống thực tế:** đạt tốc độ cao hơn greedy tham chiếu hoặc ít nhất vượt rõ các cách exact hiện có trên workload có cơ hội speculation, trong khi không thêm trọng số draft model và tuân thủ giới hạn VRAM.

Không dùng “100% exact match trên tập thử” thay cho định lý. Kết quả thực nghiệm kiểm tra implementation và phạm vi áp dụng của định lý; nó không chứng minh định lý cho mọi đầu vào.

### 1.1. Hệ tham chiếu cần cố định

Một tuyên bố exact chỉ có nghĩa khi chỉ rõ tham chiếu. Gọi `R` là quá trình greedy tuần tự với:

- Cùng checkpoint **đã lượng tử hóa**, quy tắc dequantization, tokenizer, chat template, attention mask, EOS và quy tắc phá hòa argmax.
- Backend, phiên bản thư viện, loại GPU, cấu hình kernel và precision tích lũy đã cố định.
- Cách prefill và cập nhật KV tuần tự đã cố định; cache capacity chỉ là vùng cấp phát, logical length phải giống nhau.
- Cùng token đầu vào và giới hạn sinh token. Nếu kết quả phụ thuộc lịch tải máy, cần cố định lịch hoặc thiết kế phép tính bất biến với lịch đó.

Tuyên bố “trùng greedy BNB NF4” khác với “trùng greedy GemLite” và khác với “trùng mô hình FP16/FP32”. Không gộp các hệ tham chiếu này. Exact ở đây trước hết là **token-trajectory exactness**; bitwise equality của mọi tensor là tiêu chuẩn mạnh hơn và có thể không cần nếu có chứng nhận sai số hợp lệ.

## 2. Vì sao bài toán tồn tại trong dự án

Speculative decoding dự thảo `K` token rồi xác minh `K+1` vị trí với target. Về số học thực, việc xác minh theo khối có thể cho cùng phân phối/token như chạy target tuần tự. Trên GPU hữu hạn chính xác, kích thước đầu vào thay đổi làm kernel chọn lối tính khác. Đặc biệt với 4-bit, dequantization, GEMV/GEMM, attention và phép cộng tích lũy có thể tạo logits và KV khác nhau. Một token vẫn trùng ở chu kỳ hiện tại không bảo đảm KV trùng; sai lệch KV có thể ảnh hưởng những token sau.

Dữ liệu hiện có cho thấy vấn đề đáng nghiên cứu:

| Artifact | Quan sát | Cách sử dụng |
|---|---|---|
| [`README.md`](README.md) | Benchmark N=150: PLD 1.098×, router 1.090× vanilla; cả hai khớp toàn chuỗi 83.3% dưới NF4 | Chỉ là kết quả của protocol này, không suy rộng cho backend khác |
| [`experiments/backend_validation/paired_benchmark/heldout128_cohort_report.md`](experiments/backend_validation/paired_benchmark/heldout128_cohort_report.md) | BNB + PLD: 1.133×, exact 21/60; HQQ + PLD: 1.189×, exact 43/60, mỗi phương pháp so với vanilla của chính backend | Động lực cho bài toán tốc độ và exactness đồng thời |
| [`experiments/numerical_causality/precision_validation_final/report.md`](experiments/numerical_causality/precision_validation_final/report.md) | Trên cohort chẩn đoán 16 prompt, NF4 batch có mismatch; sequential verification đạt exact 16/16 trong phép thử này | Baseline chẩn đoán, không phải chứng minh lossless phổ quát |
| [`experiments/diagnostics/nf4_margin_guard_study_n10.json`](experiments/diagnostics/nf4_margin_guard_study_n10.json) | Guard replay chọn theo margin đạt exact trên tập held-out nhỏ nhưng còn khoảng 0.35× vanilla | Mốc chi phí cần vượt; cần kiểm tra lại đúng protocol khi tái chạy |

Các số trên đến từ nhiều protocol, độ dài và backend khác nhau. Không đặt chúng thành một bảng xếp hạng trực tiếp.

**Lưu ý kiểm toán dữ liệu HQQ:** [`experiments/backend_validation/numerical_audit/hqq_factorial_diagnostic16_report.md`](experiments/backend_validation/numerical_audit/hqq_factorial_diagnostic16_report.md) hiện ghi 0/16 mismatch và 0/16 selected-row argmax divergence, khớp [`hqq_factorial_diagnostic16_summary.json`](experiments/backend_validation/numerical_audit/hqq_factorial_diagnostic16_summary.json). Các số 15/16 ca KV-history-sensitive và 1/16 execution-sensitive thuộc phép can thiệp BNB riêng, khác backend/protocol; không dùng chúng để quy nguyên nhân cho HQQ. Đã đối chiếu trực tiếp 16 raw JSONL rows với token IDs và summary: 0/16 sequence mismatch, 0/16 selected-row argmax divergence, cohort IDs khớp. Report ghi rõ các số 15/16 và 1/16 thuộc factorial BNB riêng; execution commit, model revision và backend HQQ được lưu cùng report.

## 3. Bài toán toán học

Ký hiệu `x` là prompt, `R` là quá trình greedy tham chiếu và `F` là quá trình speculative. Sau `t` token đã commit:

- `y^R_{1:t}` và `y^F_{1:t}` là hai chuỗi token.
- `C^R_t` và `C^F_t` là toàn bộ trạng thái cần cho bước tiếp theo: KV từng layer, logical lengths, vị trí, mask và mọi trạng thái phụ có ảnh hưởng đến forward.
- `z^R_t` và `z^F_t` là logits cho token kế tiếp trên cùng prefix token.

Mục tiêu correctness là, với mọi prompt và mọi độ dài hợp lệ trong phạm vi hợp đồng hệ thống:

\[
y^F_{1:t}=y^R_{1:t}\quad \text{với mọi }t\le T.
\]

Mục tiêu hiệu năng:

\[
\min_\pi \mathbb E[T_{\mathrm{request}}(\pi)]
\quad\text{s.t.}\quad
M_{\mathrm{peak}}(\pi)\le B,
\quad y^F_{1:T}=y^R_{1:T}.
\]

`π` chọn độ dài draft, chế độ verify, hành động chứng nhận, checkpoint và replay. Nếu chuyển sang bảo đảm xác suất `Pr(mismatch) ≤ δ`, phải ghi đó là một bài toán khác và đưa ra định nghĩa phân phối đầu vào, độ tin cậy và cách tính sai số thống kê. Không gọi kết quả kiểu đó là exact tuyệt đối.

### 3.1. Ranh giới lật nhãn chỉ giải quyết một bước

Nếu có **cận thực sự** `b_t` sao cho `||z^F_t-z^R_t||_∞ ≤ b_t`, gọi `Δ^F_t` là khoảng cách top-1/top-2 của logits đường nhanh, thì:

\[
\Delta^F_t>2b_t
\quad\Longrightarrow\quad
\arg\max z^F_t=\arg\max z^R_t.
\]

Điều này chứng nhận token tại `t`, chưa chứng nhận những bước sau. Ngưỡng lấy từ lỗi lớn nhất trên tập calibration **không** phải cận toàn cục. Cũng không được tính `b_t` bằng cách chạy đầy đủ tham chiếu ở mọi bước rồi trình bày như một cơ chế tăng tốc.

### 3.2. Bất biến cần duy trì qua nhiều chu kỳ

Một định lý toàn chuỗi cần bất biến quy nạp dạng:

1. Prefix token đã commit giống tham chiếu.
2. Trạng thái nhanh đúng bằng trạng thái tham chiếu, **hoặc** có một tập bao/hạn mức lỗi có hiệu lực chứa trạng thái tham chiếu.
3. Khi commit thêm token, phép cập nhật trạng thái cho bước sau giữ nguyên điều kiện 2.

Ví dụ với một chuẩn đã chọn, đặt `e_t = ||C^F_t-C^R_t||`. Cần cận hợp lệ:

\[
e_{t+1}\le A_t e_t+\eta_t,
\qquad
||z^F_t-z^R_t||_\infty\le B_t e_t+\xi_t.
\]

`A_t`, `B_t`, `η_t`, `ξ_t` phải tính được với chi phí hợp lý và bao trùm rounding, dequantization, attention, mask và kernel dispatch thực tế. Trên thực tế một cận vô hướng có thể quá lỏng. Có thể cần biên theo layer, head, token hoặc interval trực tiếp cho những đại lượng ảnh hưởng top-2. Nếu cận nở quá nhanh, phải tái đồng bộ từ checkpoint chuẩn.

**Điểm quan trọng:** replay đúng một token nhưng dùng KV đã lệch không tương đương vanilla tham chiếu. Replay phải bắt đầu từ trạng thái được chứng nhận hoặc dùng một phép chứng nhận bao trùm sai lệch KV hiện có.

## 4. Hai hướng thuật toán

### Hướng A: verify chọn lọc có chứng nhận trạng thái

Đây là hướng có tiềm năng đóng góp thuật toán và lý thuyết lớn nhất.

**Đầu vào:** checkpoint tham chiếu, prefix token đã xác nhận, draft candidates, cận trạng thái hiện hành.
**Đầu ra:** một số token được commit kèm chứng chỉ, hoặc trạng thái sau replay tham chiếu.

Quy trình dự kiến:

1. Draft bằng PLD trước, vì PLD đang là baseline zero-weight mạnh của repo. Layer-skip chỉ thêm khi đã giải quyết correctness và chứng minh có lợi về thời gian.
2. Chạy target verification theo khối trên nhánh tạm. Giữ nhánh canonical/checkpoint được định nghĩa rõ; không ghi đè trạng thái cần cho replay.
3. Với từng vị trí có thể chấp nhận, tính biên sai số logits từ biên sai số trạng thái và sai số kernel; kiểm tra margin cùng quy tắc tie-breaking.
4. Commit một prefix liên tiếp mà mọi token đều có chứng nhận. Cập nhật cận KV tương ứng. Không commit token sau vị trí chưa được chứng nhận.
5. Tại vị trí không chứng nhận được, replay từ checkpoint chuẩn gần nhất. Cập nhật KV chuẩn cho đoạn replay và đặt lại biên ở mức đã chứng minh; sau đó tiếp tục.
6. Chốt EOS, giới hạn output và rollback cache bằng cùng hợp đồng trạng thái.

**Rủi ro kỹ thuật:** cận roundoff của toàn transformer quá lỏng; tính cận đắt ngang forward; checkpoint tốn VRAM; replay dài khiến throughput dưới vanilla. Cần phép thử oracle trước khi đầu tư tối ưu hóa.

**Không đủ để gọi là hướng A:** dùng threshold margin do tuning trên dev, thấy 100% match trên held-out, rồi gọi đó là chứng minh. Cách này chỉ tạo guard thực nghiệm.

### Hướng B: đường thực thi 4-bit bất biến theo shape trong phạm vi hữu hạn

Thiết kế một đường verification mà forward của một vị trí cho kết quả và cập nhật KV tương đương đường tham chiếu khi số token của block `M` thuộc `1..K_max+1`. Có thể cố định lịch reduction, toán tử dequantization và quy tắc update KV, hoặc đưa cả hai đường về cùng một kernel contract có chứng minh.

Tính chất cần có không chỉ là `argmax` trùng ở một block thử. Với mọi prefix hợp lệ trong phạm vi, cần chứng minh forward của token ở vị trí tương ứng tạo trạng thái tương thích để bước kế tiếp cũng trùng. Cần rà soát từng toán tử: quantized GEMV/GEMM, RMSNorm, RoPE, attention, softmax, causal mask, LM head, cache write và positional indexing. Bất biến theo `M` không tự động bảo đảm bất biến khi độ dài prefix hoặc scheduling attention thay đổi.

Có hai mục tiêu khác nhau:

- **Mục tiêu B1:** speculative và vanilla cùng dùng đường kernel mới và trùng nhau. Dễ chứng minh hơn; báo cáo tốc độ so với vanilla nguyên bản lẫn vanilla kernel mới.
- **Mục tiêu B2:** speculative dùng kernel mới nhưng trùng greedy của backend nguyên bản. Cần tái tạo đúng phép tính của backend nguyên bản hoặc có chứng nhận sai số đủ mạnh. Đây là yêu cầu khó hơn.

Hướng B phù hợp làm phép thử khả thi đầu tiên vì hợp đồng toán tử và kiểm thử đối chiếu rõ hơn. Hướng A có thể bổ sung để chỉ dùng đường bất biến hoặc replay ở những nơi cần thiết.

### 4.1. Thuật toán lai có thể thành đóng góp cuối

Một ứng viên là **certified fast path + exact fallback**:

```
state = canonical_prefill(prompt)
while output_not_finished:
    candidates = propose_from_prefix(state.tokens, K)
    fast = verify_block_on_temporary_state(state, candidates)
    for position in order:
        certificate = bound_next_token_and_state(fast, state, position)
        if certificate.valid:
            commit_token_and_bounded_state(position)
        else:
            restore_last_canonical_checkpoint()
            replay_reference_until_frontier()
            break
```

Pseudocode này là mục tiêu thiết kế, **chưa là thuật toán có chứng minh**. Điều kiện `certificate.valid`, cơ chế commit KV và điểm checkpoint là những phần nghiên cứu chính.

## 5. Định lý và bằng chứng cần tạo

### 5.1. Định lý token một bước

Chứng minh từ cận `∞` và margin, xử lý cả top-1 tie và NaN/Inf. Quy tắc argmax của backend tham chiếu phải nêu rõ. Nếu tie, dùng fallback trừ khi chứng nhận được đúng quy tắc phá hòa.

### 5.2. Định lý quy nạp toàn chuỗi

Giả sử bất biến trạng thái đúng tại đầu chu kỳ. Chứng minh mỗi hành động:

- **Commit có chứng nhận:** token mới giống tham chiếu; cập nhật cận hoặc đồng nhất KV vẫn đúng.
- **Reject/fallback:** khôi phục trạng thái chuẩn không chứa hậu quả từ nhánh tạm.
- **Replay:** chạy lại từ checkpoint đã chứng nhận, cho đúng trạng thái tham chiếu tại frontier mới.
- **Bonus token/EOS:** xử lý như một bước đầy đủ, không dùng riêng kết luận ở `K` draft tokens để suy ra bonus token.

Từ đó suy ra chuỗi token trùng tham chiếu theo quy nạp. Các giả định về kernel, bộ nhớ, dữ liệu đầu vào và precision phải được liệt kê, không giấu trong phụ lục thực nghiệm.

### 5.3. Chứng nhận cận số học

Có ba lựa chọn cần đánh giá, theo mức độ khó:

1. **Bất biến toán tử:** cố định lịch phép tính để logits và KV của hai đường tương đương bitwise trong miền được hỗ trợ.
2. **Cận roundoff có điều kiện:** dùng quy tắc làm tròn, phạm vi trị số và interval propagation; chứng minh bound cho từng operator rồi ghép với trạng thái.
3. **Cận thực nghiệm có bảo đảm xác suất:** đánh giá theo distribution; gọi đúng là probabilistic guarantee, không đổi tên thành exact.

Nếu chỉ mô tả “cùng công thức toán học” mà không kiểm soát thứ tự phép toán trên GPU, chưa có chứng minh finite-precision exactness.

## 6. Kế hoạch thực hiện theo cổng quyết định

### Giai đoạn 0: khóa protocol và kiểm toán dữ liệu

- Chọn một tham chiếu đầu tiên: Qwen2.5-3B, NF4 BNB, RTX 4050, greedy, một cache backend và một lịch prefill cụ thể.
- Ghi manifest: commit mã, model revision, quantizer, kernel/backend, CUDA/PyTorch, GPU, driver, seed, prompt hash, logical KV length, tham số inference.
- Đã xử lý mâu thuẫn HQQ 16 prompt ở mục 2 bằng đối chiếu raw traces và summary; giữ phân biệt rõ kết quả HQQ với factorial BNB.
- Chia lại dữ liệu thành `development`, `diagnostic` và `held-out` trước khi tune. Không dùng cohort được chọn vì mismatch làm held-out chính.
- Chuẩn hóa metric: exact toàn chuỗi theo token ID, first divergence, KV/logit drift, TPS decode và toàn request, peak VRAM, số lần replay, thời gian replay.

**Đầu ra:** một protocol Markdown/JSON bất biến, script kiểm tra hash và baseline tái lập được.

### Giai đoạn 1: phép thử oracle về dư địa hiệu năng

- Trên trace offline, cho oracle biết trước vị trí đầu tiên cần replay. Tính throughput tốt nhất khi chỉ replay tại các vị trí ấy, bao gồm chi phí checkpoint và khôi phục ước lượng từ đo thực.
- Quét `K`, độ dài prefix, độ dài output và loại workload. Báo cáo phần thời gian cho draft, verify, certificate, replay, cache management.
- So với vanilla và guard 0.35× theo cùng môi trường/protocol mới.

**Quyết định:** nếu oracle vẫn không vượt vanilla hoặc không có khoảng trống thực tế giữa oracle và policy đơn giản, giảm phạm vi hoặc chuyển trọng tâm sang kernel invariance. Đây là tiêu chí nghiên cứu, không phải con số kết quả đã đạt.

### Giai đoạn 2: xác định nguồn sai lệch tại cùng prefix

- Đối chiếu single-token và block `M=2..K_max+1` trên **cùng token prefix**, cùng logical positions và cache clone độc lập.
- Ghi lỗi đầu tiên ở mỗi toán tử/layer, phân biệt sai lệch do kernel shape, attention, prefix KV, mask và cache length.
- Dùng ma trận can thiệp KV × verification đã có để kiểm chứng nhân quả; chỉ phân loại nếu argmax thực sự khác, không suy từ norm logits khác 0.
- Kiểm tra nhiều chu kỳ liên tiếp trên các prompt không được chọn theo mismatch để đo drift khi token vẫn trùng.

**Đầu ra:** bản đồ lỗi theo operator, layer, độ dài context và `M`; dữ liệu này chọn giữa hướng A và B.

### Giai đoạn 3: nguyên mẫu chứng minh nhỏ

- Bắt đầu với model nhỏ và một kernel 4-bit có thể kiểm soát source, `K_max` nhỏ.
- Với hướng B: viết contract reduction và cache update; kiểm tra bitwise/state equivalence trên các `M`, context length và chu kỳ accept/reject.
- Với hướng A: xây dựng cận state/logit bảo thủ; mỗi certificate lưu margin, bound, phiên bản checkpoint và operator assumptions; fallback khi bound không áp dụng.
- Có test phá bất biến: thay đổi candidate length, chèn reject sớm, thay cache capacity, thay attention mask và đưa logits gần hòa.

**Quyết định:** chỉ mở rộng khi có ít nhất một đường đưa ra định lý với giả định có thể thực thi và overhead được đo đầy đủ.

### Giai đoạn 4: tích hợp vào runtime

Các điểm mở rộng trong repository:

| Thành phần | Tệp hiện tại | Việc cần làm |
|---|---|---|
| Sinh và xác minh token | [`src/zassd/decoding/speculative.py`](src/zassd/decoding/speculative.py), [`verification.py`](src/zassd/decoding/verification.py) | Thêm mode `certified`, tách commit khỏi verify tạm, xử lý bonus/EOS, ghi lý do fallback |
| Cache và checkpoint | [`src/zassd/cache/kv_cache.py`](src/zassd/cache/kv_cache.py) | API clone/checkpoint/restore với logical length rõ ràng; giữ canonical state và nhánh tạm |
| Backend | [`src/zassd/backends/base.py`](src/zassd/backends/base.py) | Mô tả khả năng kernel, shape contract, precision và reference ID; từ chối mode certified nếu backend không hỗ trợ |
| Đối chiếu nhân quả | [`scripts/run_kv_factorial_intervention.py`](scripts/run_kv_factorial_intervention.py) | Mở rộng trace qua nhiều chu kỳ và lưu first divergent operator/token |
| Guard hiện có | [`scripts/run_nf4_margin_guard_study.py`](scripts/run_nf4_margin_guard_study.py) | Tái chạy baseline cùng cohort và metric; tách threshold thực nghiệm khỏi certificate |
| Kiểm thử | [`tests/test_speculative_exactness.py`](tests/test_speculative_exactness.py), [`tests/test_kv_cache_invariants.py`](tests/test_kv_cache_invariants.py) | Kiểm tra bất biến, replay, reject, EOS, close ties và cache corruption |

Đặt mã mới dưới `src/zassd/certification/` cho certificate, trạng thái bound và checkpoint policy; đặt runner thực nghiệm dưới `scripts/`, kết quả mới dưới một thư mục versioned trong `experiments/`. Không sửa artifact cũ để làm như kết quả mới là kết quả lịch sử.

### Giai đoạn 5: đánh giá ngoài mẫu

Ma trận mục tiêu của bài hoàn chỉnh:

- Ít nhất **3 mô hình** từ nhiều họ hoặc kích thước; không chỉ ba checkpoint cùng một kiến trúc.
- Ít nhất **3 đường kernel/backend**: BNB NF4, HQQ/GemLite hoặc Marlin, llama.cpp nếu hỗ trợ so sánh protocol hợp lệ. Mỗi backend có tham chiếu riêng.
- Ít nhất **2 GPU** với đặc tính khác nhau; RTX 4050 là một thiết bị, cần thêm một thiết bị độc lập.
- Sinh **ít nhất 256 token** ở đánh giá chính và có nhánh dài hơn để thử tích lũy drift. Báo cáo theo context ngắn/dài, hit/miss PLD, task, model và backend.
- Held-out thực sự: freeze IDs, prompt hashes, model/backend setup và policy trước khi xem kết quả. Chạy nhiều process session, thứ tự phương pháp xoay vòng, warmup và paired timing.
- Tính confidence interval theo prompt/session. Với tuyên bố exact, báo cáo `0/N` mismatch cùng giới hạn thống kê; không đổi `0/N` thành chứng minh toán học.

**Baselines:** greedy vanilla theo từng backend; PLD K tối ưu trên development; speculative BNB/HQQ hiện có; guard margin 0.35× được tái chạy; replay mọi bước/sequential verification; replay định kỳ; kernel bất biến toàn phần nếu khả thi; các baseline liên quan từ MarginGate/LLM-42 nếu mã và protocol cho phép so sánh công bằng. Có thể báo cáo khả năng tương thích thay vì số tốc độ nếu không thể chạy cùng setting 4-bit.

**Metrics bắt buộc:** exact token sequence, first divergence distribution, token certificate rate, false certificate count, replay rate/length, TPS, end-to-end latency, p50/p95, memory peak, model-weight overhead, per-stage latency, energy nếu đo đáng tin cậy. Nếu có fallback hoàn toàn đúng nhưng chạy chậm, trình bày rõ.

## 7. Tiêu chí thành công và tiêu chí dừng

| Mức | Kết quả cần có | Diễn giải |
|---|---|---|
| Khả thi lý thuyết | Định lý toàn chuỗi bao gồm KV và mọi nhánh commit/replay | Điều kiện tối thiểu cho nhan đề “exact có chứng minh” |
| Khả thi hệ thống | Implementation đáp ứng giả định định lý; không phát hiện phản ví dụ trên stress tests | Kiểm chứng implementation, không thay thế định lý |
| Hiệu năng | Nhanh hơn guard exact hiện có rõ rệt; mục tiêu mạnh là nhanh hơn vanilla và baseline PLD trong vùng workload được xác định trước | Phải tính đủ certificate, checkpoint, replay |
| Tổng quát | Cùng ý tưởng áp dụng trên nhiều model/backend/GPU, với giới hạn được ghi rõ | Cần cho bài hệ thống hàng đầu |

**Dừng hoặc đổi hướng** nếu: cận sai số quá lỏng khiến phần lớn token phải replay; oracle đã không có dư địa tăng tốc; kernel bất biến làm vanilla hoặc verification chậm vượt lợi ích; hoặc không thể nêu một hợp đồng tham chiếu ổn định trên backend mục tiêu. Kết quả âm vẫn có thể thành một nghiên cứu đo lường nếu cơ chế và bằng chứng mới đủ mạnh, nhưng sẽ cần câu chuyện bài báo khác.

## 8. Phân biệt đóng góp với công trình liên quan

| Công trình | Điều đã có | Phần cần chứng minh là mới trong đề tài này |
|---|---|---|
| [MarginGate](https://arxiv.org/abs/2605.30218) | Verify chọn lọc theo margin, sửa KV trên BF16 và báo cáo determinism thực nghiệm | Cận có chứng minh qua nhiều chu kỳ và 4-bit speculative execution; so sánh chi phí trực tiếp |
| [LLM-42](https://arxiv.org/abs/2601.17768) | Verify–rollback dưới lịch thực thi nhất quán | Chứng nhận dựa trên trạng thái lượng tử hóa, hoặc kernel contract cho verify block; không tuyên bố replay là mới |
| [Thinking Machines: batch invariance](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | Kernel bất biến trước batching | Phạm vi 4-bit, shape của speculative verification, KV và tốc độ trên thiết bị VRAM hạn chế |
| [Orthrus losslessness study](https://arxiv.org/abs/2609.15504) | Đo sai khác quỹ đạo BF16/FP32 trong một kiến trúc speculative | Cơ chế bảo đảm exact và kiểm soát chi phí dưới kernel lượng tử hóa |
| [Greedy Decoding Is Not Precision-Invariant](https://arxiv.org/abs/2609.26621) | Phân tích lật token theo precision và selective FP32 LM head, không bảo đảm phổ quát | Exact so với greedy của **cùng backend 4-bit** và bất biến trạng thái trong speculative decoding |

Phần định vị này phải rà soát lại trước khi nộp bài vì literature thay đổi nhanh. Nếu công trình khác đã chứng minh đúng cùng định lý hoặc làm cùng kernel, cần sửa claim và chọn đóng góp khác.

## 9. Gói artifact và cấu trúc bài báo dự kiến

Artifact cần có: lệnh chạy tham chiếu, test bất biến, trace first divergence, manifest môi trường, cohort freeze, dữ liệu thô theo prompt và session, script tính CI, script tạo figure/table, phép kiểm tra hash và hướng dẫn tái lập trên ít nhất một cấu hình công khai. Lưu cấu hình thay cho những mô tả mơ hồ như “NF4 mặc định”.

Cấu trúc bài báo có thể là: (1) vấn đề finite-precision và ví dụ phản chứng; (2) hợp đồng tham chiếu; (3) định lý one-step và state invariant; (4) thuật toán; (5) triển khai 4-bit; (6) đánh giá exactness, latency, memory; (7) ablation/giới hạn. Nếu kết quả nghiêng về kernel, nhấn mạnh contribution hệ thống. Nếu kết quả nghiêng về chứng nhận state, nhấn mạnh theorem và cost-aware policy.

## 10. Danh sách việc cần làm theo thứ tự

- [ ] Khóa định nghĩa exact và tham chiếu cho cấu hình đầu tiên; viết protocol chạy lại được.
- [x] Sửa mâu thuẫn báo cáo HQQ, đối chiếu 16 raw JSONL rows với token IDs/summary và ghi rõ provenance riêng của HQQ với BNB.
- [ ] Freeze tập development/diagnostic/held-out mới, không chọn held-out theo mismatch.
- [ ] Tái chạy vanilla, PLD, guard margin và sequential verification cùng protocol; lưu per-prompt token IDs, timing, VRAM.
- [ ] Xây oracle replay offline và tính trần speedup thực tế.
- [ ] Xây trace operator/layer và KV drift nhiều chu kỳ trên cùng prefix.
- [ ] Quyết định nhánh A, B hoặc lai dựa trên trần hiệu năng và nguồn drift.
- [ ] Viết contract toán tử hoặc cận số học có thể kiểm tra; chứng minh one-step margin certificate.
- [ ] Chứng minh bất biến trạng thái toàn chuỗi, gồm bonus token, EOS, rollback và replay.
- [ ] Tạo prototype một model, một backend, một GPU; đo overhead từng thành phần.
- [ ] Chạy stress tests phản ví dụ và sửa mọi trường hợp certificate sai.
- [ ] Tích hợp mode certified vào runtime với manifest và trace đầy đủ.
- [ ] Mở rộng sang ≥3 model, ≥3 backend/kernel, ≥2 GPU, output ≥256 token.
- [ ] Đánh giá held-out nhiều session, tính khoảng tin cậy và kiểm tra từng claim của bài.
- [ ] Viết bản thảo và artifact; rà soát related work lần cuối trước khi nộp.

## 11. Kết luận triển khai

Bước đầu có giá trị thông tin lớn nhất là **khóa tham chiếu + oracle replay + bản đồ drift nhiều chu kỳ**. Ba việc này trả lời liệu có đủ lợi ích để chứng minh và triển khai một cơ chế exact hay không. Nếu có, hướng B là phép thử kỹ thuật rõ ràng để tạo nền tảng exact; hướng A là phần có thể đem lại thuật toán mới và giảm chi phí. Thành công cuối cùng đòi hỏi đồng thời một bất biến trạng thái được chứng minh, một implementation tuân thủ bất biến đó và benchmark ngoài mẫu cho thấy chi phí chứng minh không nuốt hết lợi ích speculation.
