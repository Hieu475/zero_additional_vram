# Hai hướng nghiên cứu tiếp theo: A (graph + exact) và C (kiểm toán độ bất biến theo hình dạng)

Ngày: 03/11/2026. Tài liệu này trình bày hai hướng đã chọn để nâng độ mới của công trình, dựa trên các kết quả đã đo trong `RESEARCH_STATUS_20261102.md`. Mọi con số lấy từ các file bằng chứng trong `experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/`. Phần ước lượng được ghi rõ là ước lượng.

---

## 0. Điểm xuất phát và một hiệu chỉnh quan trọng

**Đã có (B2, Qwen2.5-3B NF4, RTX 4050 6 GB, batch 1, greedy, k=2):**
- Token và KV khớp greedy bitsandbytes gốc ở mọi prompt đã kiểm (held-out 300–349: 50/50, held-out 400–449: 50/50 kèm KV từng bước; dev, EOS, ngữ cảnh tới 3071 token, Qwen2.5-0.5B/1.5B).
- Không thêm VRAM đáng kể (+0.07 MiB đỉnh cấp phát).
- Nhanh hơn eager 1.15x (held-out 400–449, CI [1.136, 1.172]), nhưng **chậm hơn** baseline `torch.compile` không exact (0.91x).

**Hiệu chỉnh (03/11/2026):** kernel Triton M=3 **không cần cho tính exact**. Phép đo toán tử (`op_shape_sensitivity_gsm8k20_v1.json`) cho thấy trong ngăn xếp bitsandbytes 0.50.0:

| Toán tử | Phần tử lệch từng bit khi gọi 3 hàng so với từng hàng |
|---|---:|
| Phép chiếu NF4 (7 loại, 252 module) | 0 / 990.904.320 |
| RMSNorm trong lớp | 87 / 141.557.760 |
| LM head | 281.534 / 145.858.560 (0.19%) |
| Attention SDPA | 13.196.548 / 70.778.880 (18.6%) |

Cuối-cuối: 960/960 hàng logits không bằng từng bit, 4/960 (0.42%) đổi argmax. B2 không dùng kernel Triton vẫn exact (10/10 prompt, token và KV) nhưng chậm hơn (1.117x so với 1.161x). Kết quả này chỉ cho một phiên bản bitsandbytes, một GPU, M=3.

Hệ quả cho cả hai hướng: nguồn gốc của sai lệch là **attention và LM head** (hiếm hơn: RMSNorm), không phải phép chiếu 4-bit.

---

## 1. Hướng C: kiểm toán độ bất biến theo hình dạng của suy luận lượng tử hóa

### 1.1 Câu hỏi và đóng góp dự kiến
- **Câu hỏi:** khi một mô hình lượng tử hóa chạy một khối K+1 token thay vì từng token, *toán tử nào* làm kết quả đổi (từng bit), mức đổi bao nhiêu, và hệ quả lên token sinh ra là gì? Điều này phụ thuộc thế nào vào backend, phiên bản thư viện, kích thước khối, độ dài ngữ cảnh, mô hình và GPU?
- **Đóng góp dự kiến:**
  1. Một phương pháp kiểm thử vi sai ở mức toán tử (cùng đầu vào thật, hình dạng khác nhau) và công cụ chạy được.
  2. Một bảng phân loại nguyên nhân (toán tử nào vỡ, ở điều kiện nào) trên nhiều ngăn xếp.
  3. Số liệu về hệ quả cuối-cuối (tỷ lệ đổi argmax, tỷ lệ lệch quỹ đạo trong vòng giải mã suy đoán).
  4. Hướng dẫn: toán tử nào cần xử lý riêng để giải mã suy đoán giữ exact.

### 1.2 Đã làm
- `scripts/audit_op_shape_sensitivity.py`: ghi đầu vào/đầu ra thật của mọi phép chiếu 4-bit, RMSNorm, LM head và mọi lời gọi SDPA khi chạy khối ba token trên trạng thái tuần tự chính xác, gọi lại từng toán tử từng hàng và so từng bit; kèm so logits cuối-cuối (bằng từng bit, đổi argmax, margin).
- Kết quả đầu tiên: bảng ở Mục 0 (20 prompt GSM8K dev, 960 hàng).
- Phát hiện phụ: bản nhị phân bitsandbytes cộng tích lũy theo cặp, khác mã nguồn đọc theo nghĩa đen (`nf4_gemv_order_emulation_v1.json`, 28.672.000 đầu ra).

### 1.3 Thiết kế mở rộng (ma trận thí nghiệm)

| Trục | Giá trị cần phủ | Ghi chú |
|---|---|---|
| Kích thước khối M | 2, 3, 4, 5, 8, 16 | Hiện mới M=3. bitsandbytes có ngưỡng riêng theo M nên đây là trục quan trọng nhất |
| Độ dài ngữ cảnh | 128, 512, 1024, 2048, 3072 | Dispatch SDPA có thể đổi theo độ dài KV; hiện chưa đo riêng |
| Backend lượng tử hóa | bitsandbytes NF4 (có/không lượng tử hóa kép), FP4; HQQ (repo đã có dấu vết); GPTQ/AWQ nếu cài được | Mỗi backend cần adapter nạp mô hình |
| Mô hình | Qwen2.5 0.5B/1.5B/3B; Llama 3.2 1B/3B (adapter có sẵn trong repo) | Cùng họ trước, khác họ sau |
| Dtype/attention | fp16 so với bf16; SDPA flash so với memory-efficient so với math | Chỉ rõ backend SDPA đã chọn |
| Phiên bản | bitsandbytes, PyTorch, Triton | Mỗi lần nâng cấp phải chạy lại |
| GPU | RTX 4050 (có sẵn); các GPU khác do bạn chạy | Kiến trúc khác có thể chọn kernel khác |

**Chỉ tiêu mỗi ô:** tỷ lệ phần tử lệch từng bit theo toán tử; độ lệch tối đa; tỷ lệ hàng đổi argmax; độ rộng margin top-1/top-2 của các hàng đổi; và (cho một tập con) tỷ lệ chuỗi lệch quỹ đạo khi chạy vòng giải mã suy đoán *không* xử lý riêng, so với vòng B2.

### 1.4 Việc cụ thể, theo thứ tự
1. **C2 (rẻ):** quét M và độ dài ngữ cảnh trên bitsandbytes NF4 + Qwen2.5-3B, tái dùng script hiện có (thêm tham số `--block-size`).
2. **C3:** Qwen2.5-0.5B/1.5B và Llama 3.2 (qua `scripts/run_with_model.py` hoặc adapter).
3. **C4:** thêm một backend lượng tử hóa thứ hai (HQQ là ứng viên rõ nhất vì repo đã có hạ tầng).
4. **C5:** gộp thành một công cụ dòng lệnh duy nhất, có đầu ra JSON chuẩn và bảng tóm tắt.
5. **C6:** đo hệ quả cuối-cuối: tỷ lệ lệch quỹ đạo của PLD không xử lý riêng, theo từng cấu hình.
6. **C7:** chạy các GPU và phiên bản khác (bạn thực hiện), thêm vào ma trận.

### 1.5 Tiêu chí thành công và dừng
- **Thành công:** có bảng nguyên nhân nhất quán trên ít nhất 2 backend và 2 họ mô hình, và công cụ chạy được từ một lệnh.
- **Dừng hoặc đổi hướng nếu:** (i) mọi cấu hình ngoài bitsandbytes cho cùng kết luận trivial (ví dụ chỉ attention lệch ở mọi nơi), thì giá trị của bảng phân loại thấp; (ii) khảo sát tài liệu cho thấy phát hiện đã được công bố đầy đủ.

### 1.6 Rủi ro
- **Tính mới chưa kiểm chứng:** tôi chưa khảo sát tài liệu có hệ thống. Cần tìm: suy luận batch-invariant/xác định (vLLM, SGLang và các bài kỹ thuật liên quan), tính exact của giải mã suy đoán dưới độ chính xác thấp, và ghi nhận về thứ tự rút gọn của kernel bitsandbytes. LLM-42 (SOSP 2026) là công trình gần nhất đã xác nhận.
- **Phụ thuộc phiên bản:** kết quả có thể chỉ đúng với bitsandbytes 0.50.0. Giảm nhẹ bằng cách đo nhiều phiên bản.
- **Phạm vi hẹp nếu thiếu backend/GPU:** C chỉ có giá trị lớn khi có độ rộng.

### 1.7 Nơi phù hợp
MLSys, các track dữ liệu và công cụ, TMLR, hoặc workshop về suy luận hiệu quả. Cần kiểm tra hạn nộp và yêu cầu hiện hành trên trang chính thức.

---

### 1.8 Kết quả đã đo (cập nhật 03/11/2026: C2–C6)

Công cụ: `scripts/shape_audit.py` (`ops`, `trajectory`, `summarize`). Bảng đầy đủ do `summarize` sinh ra: `native_bnb_profiles/C_SHAPE_AUDIT_SUMMARY.md`. Mọi số liệu: một GPU (RTX 4050), bitsandbytes 0.50.0, 20 prompt GSM8K dev (10 prompt cho trục ngữ cảnh), greedy.

| Phát hiện | Bằng chứng |
|---|---|
| Phép chiếu NF4 bất biến theo hình dạng khi M ≤ 3, vỡ từ M=4 (khoảng 13–15% phần tử lệch) | Qwen2.5 0.5B/1.5B/3B, OLMo-2-1B, Phi-3.5; M ∈ {2,3,4,5,8,16} trên 3B |
| Attention (SDPA) lệch khoảng 17–19% ở mọi M và mọi độ dài ngữ cảnh 512–3072; riêng Qwen 0.5B chỉ 3.6% | `c2_*`, `c3_*`, `c4_*` |
| LM head lệch 0–0.19% (0.5B: 0) | như trên |
| **HQQ+GemLite: số liệu lệch theo hình dạng KHÔNG hợp lệ** (xem Mục 1.9): bản thân GemLite không lặp lại được giữa hai lần chạy cùng đầu vào, nên "49% phần tử lệch" chủ yếu là nhiễu | `c4_qwen3b_hqq_*`, `run_to_run_noise_v1.json` |
| Hàng đổi argmax khoảng 0.1–1% ở bitsandbytes | các bảng trên |
| PLD ngây thơ K=2 làm lệch quỹ đạo (50 prompt, 128 token, khoảng Wilson 95%): Qwen 0.5B 66% [52, 78]; Qwen 1.5B 50% [37, 63]; Phi-3.5 46% [33, 60]; Qwen 3B 36% [24, 50]; OLMo-2-1B 34% [22, 48]; Qwen 0.5B cao hơn Qwen 3B và OLMo-2 (khoảng không giao nhau). HQQ+GemLite 6% [2, 16] nhưng KHÔNG so sánh được với bitsandbytes vì nó không lặp lại được (Mục 1.9). K=3 (chỉ 20 prompt) không khác K=2 đáng kể | `c6_*`, `c6x_*`, bảng gộp trong `C_SHAPE_AUDIT_SUMMARY.md` |
| Qwen 0.5B có 6.0% hàng với margin nhỏ hơn sai số logit trung vị (bnb khác: 1.5–4.6%), cao nhất, khớp với tỷ lệ lệch quỹ đạo cao nhất. (Ý giải thích riêng cho HQQ ở lượt trước đã bị thu hồi, xem Mục 1.9.) | các JSON `end_to_end` |
| Phép thử SDPA thuần: tỷ lệ lệch của hàng i không phụ thuộc M (M = 2, 3, 4, 6), nên M=2 trùng M=3 vì khác biệt nằm giữa đường q_len=1 và q_len≥2 (chưa đọc kernel để xác nhận) | script tạm, không lưu trong repo |

**Hệ quả:** (i) kết luận "kernel Triton M=3 không cần cho exact" (Mục 0) chỉ đúng cho K ≤ 2 trên bitsandbytes; từ K=3 phép chiếu cũng phải xử lý riêng. (ii) Với HQQ+GemLite chưa kết luận được: backend này không lặp lại được nên cần ép xác định hoặc dùng backend khác (Mục 1.9). (iii) Cách xử lý attention từng hàng dùng được cho mọi độ dài khảo sát.

**Giới hạn và điểm chưa giải thích:**
- Mẫu: ops 20 prompt; C6 K=2 50 prompt (toàn bộ benchmark GSM8K hiện có), K=3 chỉ 20 prompt; mỗi cấu hình chạy một lần.
- Chưa có Llama 3.2 (model gated, cần token HF); chưa có GPTQ/AWQ; chưa có GPU hay phiên bản thư viện khác (C7).
- Ngưỡng M=4 là suy luận từ số đo, chưa đối chiếu mã nguồn bitsandbytes.
- Hai điểm từng chưa giải thích (M=2 trùng M=3, Qwen 0.5B lệch nhiều) đã có giải thích dựa trên số liệu; điểm về HQQ bị thu hồi (Mục 1.9); chưa kiểm chứng bằng cách đọc kernel.
- Các hàng ngữ cảnh dài ghép prompt GSM8K làm đệm, không phải văn bản tự nhiên.
- Khảo sát tài liệu nhanh đã làm: xem `LITERATURE_NOTES_DIRECTION_C.md`. Ý tưởng độ nhạy theo hình dạng không mới (batch-invariant, LLM-42); phần 4-bit và liên hệ tới giải mã suy đoán chưa xác nhận là mới. Chưa đọc toàn văn bài nào.

---

### 1.9 Kiểm chứng bản sửa theo kiểm toán, và nhiễu của HQQ+GemLite (04/11/2026)

Lệnh `shape_audit.py plan` đọc kết quả kiểm toán và liệt kê lớp toán tử có bất kỳ phần tử lệch nào; `--rowwise` trong `audit_pld_trajectory_divergence.py` chạy các lớp đó từng hàng khi xác minh khối. 50 prompt GSM8K, 128 token, so với greedy M=1 tuần tự; khoảng Wilson 95%. Bằng chứng: `fix_*.json`, bảng gộp trong `C_SHAPE_AUDIT_SUMMARY.md`.

| Cấu hình (bitsandbytes NF4) | K | Từng hàng | Lệch quỹ đạo |
|---|---:|---|---:|
| Qwen 3B | 2 | không (PLD ngây thơ) | 18/50 |
| Qwen 3B | 2 | RMSNorm + SDPA + LM head (danh sách `plan`) | **0/50** [0, 7]% |
| Qwen 1.5B, 0.5B, OLMo-2-1B, Phi-3.5 | 2 | cùng danh sách | **0/50** mỗi mô hình |
| Qwen 3B | 2 | chỉ SDPA, hoặc SDPA + LM head (bỏ RMSNorm) | 3/50 |
| Qwen 3B | 3 | RMSNorm + SDPA + LM head (thiếu phép chiếu) | 18/50 |
| Qwen 3B | 3 | tất cả, gồm phép chiếu | **0/50** |

**Kết luận được hỗ trợ (bitsandbytes 0.50.0, sm_89):** kiểm toán theo toán tử dự đoán đúng tập toán tử cần xử lý riêng: danh sách do công cụ sinh ra đưa lệch quỹ đạo về 0/50 trên 5 mô hình/họ; bỏ bất kỳ toán tử nào trong danh sách (kể cả RMSNorm, chỉ lệch hiếm) làm lệch trở lại; K=3 cần thêm phép chiếu, đúng như kiểm toán. "0/50" nghĩa là không thấy lệch trong mẫu, không phải chứng minh exact. Danh sách này được chọn trên chính 50 prompt đã dùng để kiểm toán nên chưa phải kiểm chứng độc lập.

**Phát hiện về HQQ+GemLite (đảo ngược lượt trước):**
- Tham chiếu M=1 của HQQ+GemLite chạy hai lần trên cùng prompt cho kết quả khác nhau ở 5/50 chuỗi (bitsandbytes: 0/50).
- Gọi cùng một phép chiếu GemLite 5 lần với cùng đầu vào cho kết quả khác nhau ở 64% phần tử (M=1) và 37% (M=3) (đầu vào ngẫu nhiên fp16, 36 trong 252 module; một lần chạy). Nguyên nhân chưa xác định (chưa kiểm tra cộng nguyên tử/split-K hay autotune).
- Hệ quả: các số "phép chiếu HQQ lệch 49% phần tử", "HQQ ít đổi argmax" và "HQQ lệch quỹ đạo 6%" ở các bảng trước **không** đo độ nhạy theo hình dạng; chúng trộn với nhiễu giữa các lần chạy. Ngay cả khi mọi toán tử chạy từng hàng, HQQ vẫn lệch 3/50 và danh sách thiếu phép chiếu 7/50 — không diễn giải được vì tham chiếu của nó không ổn định. Muốn có số liệu HQQ hợp lệ phải ép GemLite chạy xác định (hoặc dùng backend khác như GPTQ/AWQ có kernel xác định).
- Lượt trước tôi đã đưa ra một giải thích về HQQ (sai số logit nhỏ hơn nên ít đổi argmax) trước khi kiểm tra nhiễu; giải thích đó bị thu hồi.

### 1.10 Kiểm chứng độc lập trên HumanEval (04/11/2026)

Danh sách toán tử cần xử lý riêng của **từng mô hình** được sinh ra bởi `shape_audit.py plan` từ kiểm toán GSM8K của chính mô hình đó (không chỉnh tay), rồi áp dụng nguyên trạng lên 50 prompt HumanEval (`humaneval_eval.jsonl`, chưa dùng để chọn danh sách), K=2, 128 token, bitsandbytes NF4. Script `run_heldout_humaneval.sh`; bằng chứng `heldout_humaneval_*.json`.

| Mô hình | Danh sách do `plan` sinh ra | PLD ngây thơ lệch | Với danh sách lệch |
|---|---|---:|---:|
| Qwen2.5-3B | lm_head, rmsnorm, sdpa | 19/50 (38%) | **0/50** |
| Qwen2.5-1.5B | lm_head, sdpa | 13/50 (26%) | **0/50** |
| Qwen2.5-0.5B | sdpa | 26/50 (52%) | **0/50** |
| OLMo-2-1B | lm_head, rmsnorm, sdpa | 17/50 (34%) | **0/50** |
| Phi-3.5-mini | lm_head, rmsnorm, sdpa | 31/50 (62%) | **0/50** |

Mỗi ô 0/50 có khoảng Wilson 95% là [0, 7]%. Các danh sách *khác nhau theo mô hình* (0.5B chỉ cần `sdpa`) vẫn đủ, nghĩa là kiểm toán không chỉ đoán đúng "danh sách chung" mà còn đoán đúng tập tối thiểu theo từng mô hình. Giới hạn: vẫn một GPU, bitsandbytes 0.50.0, K=2, mỗi ô chạy một lần; HumanEval là mã nguồn, có nhiều lặp lại nên thuận lợi cho PLD, chưa thử văn bản tự nhiên.

### 1.11 Backend 4-bit thứ hai hợp lệ: torchao INT4 chỉ trọng số (04/11/2026)

Thay cho HQQ+GemLite (không xác định, Mục 1.9). torchao 0.18.0 `Int4WeightOnlyConfig(group_size=128)`, định dạng `TILE_PACKED_TO_4D` (kernel tinygemm của PyTorch), kích hoạt bf16, Qwen2.5-3B-Instruct, cài trong venv riêng; nạp lên CPU rồi lượng tử hóa từng tầng trên GPU (bản 3B bf16 không vừa 6 GB). Script `run_torchao_chain.sh`; bằng chứng `c8_torchao_*.json`. Điều kiện tiên quyết đã kiểm tra: tham chiếu M=1 chạy hai lần cho kết quả giống hệt (0/20 chuỗi khác nhau).

| Kết quả (20 prompt cho kiểm toán, 50 prompt cho quỹ đạo) | torchao INT4 | bitsandbytes NF4 (so sánh) |
|---|---|---|
| Phép chiếu lệch theo hình dạng, M=3 | 0 phần tử | 0 phần tử |
| Phép chiếu lệch theo hình dạng, M=4 | **0 phần tử** | khoảng 13% phần tử |
| RMSNorm lớp (M=3) | lệch hiếm, max 0.0156 | lệch hiếm |
| LM head | 0.16% phần tử | 0.19% |
| SDPA | 17.4% phần tử | 18.6% |
| Hàng đổi argmax | 9/960 (M=3), 10/960 (M=4) | 4/960 |
| `plan` | lm_head, rmsnorm, sdpa (cả M=3 và M=4) | cùng (M=3); thêm linear (M=4) |
| PLD ngây thơ K=2 / K=3 lệch quỹ đạo | 19/50 / 20/50 | 18/50 / (20 prompt: 6/20) |
| PLD với danh sách `plan`, K=2 / K=3 | **0/50 / 0/50** | 0/50 (K=2); K=3 cần thêm linear |

**Kết luận:** kiểm toán và bản sửa hoạt động trên một backend 4-bit thứ hai với kernel khác hẳn, và *dự đoán đúng sự khác biệt*: phép chiếu torchao tinygemm không vỡ ở M=4 nên danh sách không cần `linear`, trong khi bitsandbytes cần. Ngưỡng M=4 là đặc thù của bitsandbytes 0.50.0 chứ không phải của mọi backend 4-bit. SDPA/LM head/RMSNorm lệch ở cả hai.

**Giới hạn:** một mô hình (Qwen2.5-3B), một GPU, một lần chạy mỗi ô; torchao chỉ kiểm đến M=4 (chưa thử M lớn hơn); kích hoạt bf16 khác fp16 của bitsandbytes nên so sánh tỷ lệ phần tử lệch giữa hai backend chỉ là gần đúng; danh sách được chọn trên cùng 50 prompt GSM8K dùng để đo (chưa chạy HumanEval cho torchao). GPTQ/AWQ chưa thử (cần biên dịch kernel).

### 1.12 CNN/DailyMail (văn bản tự nhiên, ngữ cảnh dài) và một lỗi của bản sửa từng hàng (04/11/2026)

50 prompt `cnndm_eval.jsonl` (164–605 token, trung vị 400), K=2, 128 token; danh sách toán tử là danh sách `plan` từ kiểm toán GSM8K, không chỉnh. Script `run_heldout_generic.sh` (lượt đầu) và `run_gqa_fix_rerun.sh` (sau khi sửa); bằng chứng `heldout_cnndm_*`.

**Lượt đầu cho kết quả âm với mô hình dùng GQA:** Qwen 3B 36/50, 1.5B 33/50, 0.5B 10/50 vẫn lệch dù xử lý riêng đúng danh sách; OLMo-2 và Phi-3.5 (không GQA) 0/50. **Nguyên nhân là lỗi của chính bản sửa từng hàng:** bước M=1 thật gọi SDPA với `enable_gqa=True` và K/V chỉ có số đầu K/V gốc; khối M=3 nhận K/V đã nhân bản (`repeat_kv`), và hàm SDPA từng hàng của tôi gọi với K/V nhân bản, tức một đường kernel khác. Ở ngữ cảnh ngắn (GSM8K, HumanEval) hai đường cho cùng kết quả (tình cờ), ở ngữ cảnh dài thì khác. Sửa: rút gọn K/V về số đầu gốc trước khi gọi từng hàng.

| Mô hình (CNN/DM, K=2) | PLD ngây thơ lệch | Danh sách `plan`, bản sửa lỗi |
|---|---:|---:|
| Qwen2.5-3B (bnb) | 33/50 | **0/50** |
| Qwen2.5-1.5B (bnb) | 36/50 | **0/50** |
| Qwen2.5-0.5B (bnb) | 38/50 | **0/50** |
| OLMo-2-1B (bnb) | 27/50 | **0/50** (không GQA, không bị ảnh hưởng) |
| Phi-3.5-mini (bnb) | 41/50 | **0/50** (không GQA) |
| Qwen2.5-3B (torchao INT4) | 38/50 | **0/50** |

Khoảng Wilson 95% của mỗi ô 0/50 là [0, 7]%. Các kết quả cũ trước bản sửa vẫn nằm trong bảng tổng hợp, gắn nhãn `PRE-GQA-FIX`, để minh bạch.

**Ảnh hưởng tới các kết quả trước:** (i) Tỷ lệ lệch của *kiểm toán* gần như không đổi sau khi sửa tham chiếu từng hàng (GSM8K: SDPA 18.64%, 18.30%, 3.59% giữ nguyên đến 4 chữ số; ngữ cảnh 512: 19.37% giữ nguyên; 2048: 18.59% → 18.44%), nên các bảng kiểm toán không cần thu hồi. (ii) Các kết quả 0/50 trên GSM8K và HumanEval vẫn đúng, vì chúng đo tính exact so với tham chiếu M=1 thật. (iii) Bài học: một bản sửa đúng ở ngữ cảnh ngắn có thể sai ở ngữ cảnh dài; kiểm chứng exact phải chạy cả hai.

**Giới hạn:** danh sách được chọn từ GSM8K rồi thử trên HumanEval và CNN/DM; bản sửa GQA được viết *sau khi* thấy lỗi trên CNN/DM, nên CNN/DM giờ không còn là kiểm chứng "chưa thấy" thuần túy cho riêng bản sửa này (danh sách `plan` thì chưa bị chỉnh); chưa có bộ dữ liệu thứ ba để kiểm chứng bản sửa GQA độc lập (có thể dùng `cnndm_ext150.jsonl`, 150 prompt chưa chạm tới); mô hình tối đa 3B, một GPU, bitsandbytes 0.50.0, mỗi ô một lần.

### 1.13 Kiểm chứng độc lập của bản sửa GQA trên `cnndm_ext150` (04/11/2026)

60 prompt đầu của `cnndm_ext150.jsonl` (0/150 trùng id và nội dung với `cnndm_eval`; chưa dùng cho bất kỳ quyết định thiết kế nào), K=2, 128 token, mã đã sửa GQA, danh sách `plan` giữ nguyên từ GSM8K. Script `run_cnndm_ext_validation.sh`; bằng chứng `ext150_cnndm_*.json`.

| Cấu hình | Từng hàng | Lệch quỹ đạo |
|---|---|---:|
| Qwen2.5-3B, bnb | không (PLD ngây thơ) | 47/60 (78%) |
| Qwen2.5-3B, bnb | lm_head + rmsnorm + sdpa | **0/60** |
| Qwen2.5-1.5B, bnb | lm_head + sdpa | **0/60** |
| Qwen2.5-0.5B, bnb | sdpa | **0/60** |
| Qwen2.5-3B, torchao INT4 | lm_head + rmsnorm + sdpa | **0/60** |
| Qwen2.5-3B, torchao INT4 | không (PLD ngây thơ) | 41/60 (68%) |
| Qwen2.5-1.5B, bnb | không (PLD ngây thơ) | 39/60 (65%) |
| Qwen2.5-0.5B, bnb | không (PLD ngây thơ) | 41/60 (68%) |
| OLMo-2-1B, bnb | không (PLD ngây thơ) | 29/60 (48%) |
| OLMo-2-1B, bnb | lm_head + rmsnorm + sdpa | **0/60** |
| Phi-3.5-mini, bnb | không (PLD ngây thơ) | 55/60 (92%) |
| Phi-3.5-mini, bnb | lm_head + rmsnorm + sdpa | **0/60** |

Khoảng Wilson 95% của mỗi ô 0/60 là [0, 6]%. Tổng hợp qua ba tập dữ liệu (GSM8K, HumanEval, CNN/DM) và hai backend xác định, danh sách do `shape_audit.py plan` sinh ra không có ca lệch nào trong mọi lượt chạy ở bản sửa đúng. Giới hạn: PLD ngây thơ làm mốc chỉ chạy cho Qwen 3B và hai mô hình không GQA (OLMo-2, Phi-3.5), chưa chạy cho Qwen 1.5B/0.5B và torchao trên ext150; mô hình tối đa 3B, một GPU, bitsandbytes 0.50.0, K=2, mỗi ô một lần; "0 ca lệch" là quan sát trên mẫu, không phải chứng minh.

### 1.14 bitsandbytes 0.49.2 (trước PR #1949) so với 0.50.0 (05/11/2026)

Cùng mô hình (Qwen2.5-3B NF4, GSM8K 20 prompt cho kiểm toán, 50 cho quỹ đạo), cùng GPU (RTX 4050), 0.49.2 cài trong venv riêng. Script `run_bnb0492_chain.sh`; bằng chứng `c10_bnb0492_*.json`.

| Phần tử phép chiếu lệch so với chạy từng hàng | M=3 | M=4 | M=8 |
|---|---:|---:|---:|
| bitsandbytes 0.50.0 | **0%** | 13.0% | 13.0% |
| bitsandbytes 0.49.2 | **34.2%** | 34.2% | 34.2% |

Theo từng phép chiếu, ở 0.50.0 M≥4: `k_proj` và `v_proj` vẫn bằng 0, các phép khác 9–19%; ở 0.49.2 mọi phép chiếu đều lệch 26–45% ngay từ M=3. SDPA (18.6%), LM head (0.19%) và RMSNorm (hiếm) giống nhau ở hai phiên bản.

| Quỹ đạo PLD (50 prompt GSM8K) | K=2 | K=3 |
|---|---:|---:|
| 0.49.2, ngây thơ | 11/50 | 12/50 |
| 0.49.2, từng hàng theo `plan` (linear + lm_head + rmsnorm + sdpa) | **0/50** | **0/50** |

**Kết luận:** (i) Việc phép chiếu NF4 không cần xử lý riêng ở K=2 (Mục 0) là hệ quả của kernel GEMM 4-bit mới có từ 0.50.0, không phải tính chất chung của bitsandbytes; với bản trước PR đó, phép chiếu phải nằm trong danh sách ngay từ M=3. (ii) Công cụ `plan` đoán đúng sự khác biệt giữa hai phiên bản mà không cần biết gì về kernel, và danh sách tương ứng đưa quỹ đạo về 0/50 ở cả hai. (iii) Điều này phù hợp với việc dự án ban đầu cần kernel Triton cho phép chiếu, nhưng tôi **chưa kiểm tra** phiên bản bitsandbytes lúc phát triển kernel đó nên không khẳng định đó là nguyên nhân.

**Giới hạn:** một mô hình, một GPU, mỗi ô một lần; 0.49.2 chạy cùng PyTorch 2.11/CUDA 12.8 với bản 0.50.0 (venv kế thừa gói hệ thống); chưa thử các bản 0.50.1/0.50.2 (đổi heuristic cho GB10) và các GPU khác; cờ "PLD ngây thơ 11/50" ở 0.49.2 thấp hơn 18/50 của 0.50.0 nhưng khác biệt này nằm trong khoảng nhiễu của 50 prompt (hai khoảng Wilson chồng lên nhau), không nên diễn giải.

### 1.15 Ngữ cảnh dài (511–3072 token) và nhân bản độc lập (05/11/2026)

Script `run_long_and_replicate.sh`; bằng chứng `longctx_*.json`, `ext150b_*.json`. K=2, 128 token, danh sách `plan` giữ nguyên từ GSM8K.

| Thử nghiệm | Cấu hình | PLD ngây thơ lệch | Với danh sách `plan` |
|---|---|---:|---:|
| `long_context_dev` (12 prompt, 511–3072 token) | Qwen 3B (bnb) | 11/12 | **0/12** |
| `long_context_dev` | OLMo-2-1B (bnb) | 4/12 | **0/12** |
| `cnndm_ext150` prompt 60–119 (60 prompt mới) | Qwen 3B (bnb) | 49/60 | **0/60** |
| `cnndm_ext150` prompt 60–119 | Qwen 3B (torchao INT4) | chưa chạy | **0/60** |

Lần nhân bản thứ hai trên 60 prompt khác cho kết quả khớp lần đầu (ngây thơ 47/60 và 49/60; danh sách 0/60 cả hai), nên kết quả không phụ thuộc vào 60 prompt cụ thể. Giới hạn: ngữ cảnh dài chỉ 12 prompt (khoảng Wilson 95% của 0/12 là [0, 24]%), các prompt dài là bài báo CNN/DM ghép nên chưa phải văn bản dài tự nhiên; Qwen 1.5B/0.5B chưa chạy ở ngữ cảnh dài; mô hình tối đa 3B, một GPU, bitsandbytes 0.50.0.

## 2. Hướng A: vòng giải mã exact chạy trong CUDA graph

### 2.1 Động cơ
Forward decode bị giới hạn bởi CPU khởi chạy kernel (84% thời gian một bước M=1 là CPU enqueue; `launch_bound_gsm8k_000_v1.json`). Đây là lý do baseline `torch.compile` nhanh hơn B2, và các phép từng hàng của B2 làm tăng số kernel. CUDA graph loại bỏ chi phí này mà **giữ nguyên các kernel**, nên về nguyên tắc không đổi số học.

### 2.2 Những gì đã thăm dò
Dùng `scripts/probe_cuda_graph_decode.py` (chụp thô, không `torch.compile`, mô hình bitsandbytes eager với `StaticCache`):

| Đường | Eager | Graph | Logits bằng từng bit |
|---|---:|---:|---:|
| Bước M=1 | 22.5 ms | 17.3 ms (1.30x) | 64/64 bước |
| Khối chuẩn M=3 (không exact) | 27.4 ms | 23.9 ms | 64/64 bước |

Kết luận khả thi: chụp được, đúng từng bit so với cùng đường eager.

### 2.3 Ước lượng tốc độ (mô phỏng, chưa cài đặt)
Dùng `scripts/simulate_block_size.py` (PLD trên chuỗi tham chiếu đã ghi, mô hình chi phí hiệu chỉnh khớp số đo K=2). Giả định: LM head lười chạy ngoài graph; chi phí decoder 3 hàng trong graph 21.7–24.0 ms (ước từ số đo kernel).

| Cohort dev (30 prompt), K=2 | So với baseline graph exact | So với BNB eager |
|---|---:|---:|
| GSM8K | 1.04–1.08x | 1.36–1.40x |
| HumanEval | 1.10–1.14x | 1.43–1.48x |

Tham chiếu: baseline biên dịch không exact đạt 1.29–1.45x so với eager. Nghĩa là A có thể **ngang hoặc nhỉnh hơn** baseline biên dịch mà vẫn exact, nhưng chỉ hơn baseline graph exact 4–14%. Đây là dự đoán, không phải số đo.

### 2.4 Thiết kế đề xuất (không cần kernel mới)
- **Tham chiếu:** định nghĩa lại là bước decode M=1 chạy với `StaticCache` (cùng kernel, chạy eager hoặc trong graph; hai cách này bằng từng bit theo thăm dò). Lưu ý: tham chiếu này **khác** greedy dùng cache động hiện tại, nên mọi tuyên bố exact phải ghi rõ tham chiếu nào.
- **Phần thân decoder:** giữ các phép từng hàng của B2 (attention SDPA từng hàng, RMSNorm từng hàng) nhưng với độ dài KV tĩnh và mặt nạ, đúng như đường M=1 tham chiếu dùng; chụp thành một graph cho M=1 (nhánh quay về bước gốc) và một graph cho M=3 (khối). Vòng từng hàng nằm trong graph nên không còn tốn CPU.
- **LM head:** chạy ngoài graph theo từng hàng (mỗi hàng là một graph nhỏ hoặc một lời gọi) để giữ tính lười.
- **Logic chấp nhận/quay lui:** ở CPU, ngoài graph; quay lui là đổi con trỏ vị trí và độ dài logic (không sao chép KV).
- **Kernel phép chiếu:** dùng bitsandbytes chuẩn (đã khớp từng hàng) hoặc kernel Triton tùy chọn nếu đo thấy nhanh hơn trong graph.

### 2.5 Mốc và điều kiện dừng

| Mốc | Nội dung | Điều kiện đi tiếp |
|---|---|---|
| A1 | Đường từng hàng với KV tĩnh, chạy **eager**, bằng từng bit với tham chiếu `StaticCache` M=1 (token và KV) trên ít nhất 30 prompt dev | Khớp 100% |
| A2 | Chụp graph M=3 phần thân decoder; đúng từng bit với bản eager A1 | Khớp 100% và chi phí decoder 3 hàng trong graph ≤ 24 ms |
| A3 | Vòng giải mã đầy đủ (fallback + khối + LM head lười) trong graph; token và KV khớp tham chiếu | Khớp 100% trên dev |
| A4 | Đo tốc độ cạnh nhau so với: BNB eager, baseline graph exact, `torch.compile` | **Dừng nếu** tốc độ so với baseline graph exact < 1.05x |
| A5 | Held-out mới (cohort chưa dùng) đúng một lần, cấu hình đăng ký trước | Chỉ sau khi khóa mã |

### 2.5b Kết quả A1–A2 (03/11/2026, tham chiếu StaticCache M=1 đã được chọn)

Script: `scripts/a1_rowwise_static_block.py`, `scripts/a2_graph_block.py`. Bằng chứng: `native_bnb_profiles/a1_*`, `a2_*`. Qwen2.5-3B bnb NF4, 30 prompt GSM8K dev, 48 bước, teacher-forcing, một GPU.

| Mốc | Kết quả |
|---|---|
| A1 (eager, từng hàng, KV tĩnh) | Đạt: logits từng hàng đã chốt và KV bằng từng bit với tham chiếu M=1, 30/30 prompt, hai mẫu từ chối (mỗi 3 và mỗi 2 chu kỳ) |
| A2 (chụp graph thân M=3) | Đúng từng bit trên 30/30 prompt (cache dài 768 và 320). Thân M=3: 27.2 ms (cache 768) và 23.6 ms (cache 320) trong graph, so với khoảng 31 ms eager. Mốc ≤ 24 ms chỉ đạt khi cache ngắn |

**Phát hiện kỹ thuật:** `StaticLayer.update` của HF ghi tại `cumulative_length` và bỏ qua `cache_position` khi ghi, nên quay lui phải đặt lại tensor `cumulative_length` tại chỗ (`fill_`). Quên bước này làm sai ngay sau chu kỳ từ chối đầu tiên (đã gặp và sửa).

**Giới hạn:** chi phí thân trong graph phụ thuộc độ dài cache cấp phát (attention từng hàng đọc toàn bộ cache tĩnh); cần phân nhóm theo độ dài nếu muốn chạy ngữ cảnh dài. Mới là teacher-forcing, chưa có vòng giải mã đầy đủ (A3), chưa đo tốc độ đầu-cuối (A4). Thời gian đo bằng đồng hồ tường có đồng bộ, một lần chạy.

### 2.5c Kết quả A3–A4 (03/11/2026): điều kiện dừng đã kích hoạt

Script: `scripts/a3_graph_pld_loop.py`; bằng chứng: `native_bnb_profiles/a3_graph_pld_k2_gsm8k30_v1.json`. Vòng đầy đủ (PLD K=2, n-gram 3, 128 token mới, 30 prompt GSM8K dev, cache 448): graph M=1 gồm LM head cho fallback, graph M=3 thân từng hàng, LM head lười ngoài graph, khối M=3 chỉ dùng khi tra n-gram ra đủ 2 bản nháp.

| Chỉ tiêu | Kết quả |
|---|---|
| A3: token và KV đã chốt bằng từng bit với tham chiếu M=1 StaticCache | Đạt 30/30 |
| Tốc độ eager M=1 (tham chiếu) | 41.6 token/s |
| Baseline graph exact (M=1 trong graph) | 58.1 token/s (1.40x eager) |
| Vòng A3 | 58.6 token/s (1.41x eager) |
| **A3 so với baseline graph exact** | **1.009x** |

**Kết luận A4:** tốc độ so với baseline graph exact thấp hơn 1.05x, nên theo điều kiện đã đăng ký (Mục 2.5) hướng A dừng ở đây, không chạy A5 (held-out). Nguyên nhân khả dĩ (chưa kiểm chứng riêng): thân M=3 trong graph tốn khoảng 24–27 ms so với khoảng 17 ms của bước M=1, nên mỗi khối cần chấp nhận nhiều bản nháp mới có lợi; ở đây chỉ khoảng 25 khối M=3 trên 128 token cho mỗi prompt, và nhiều bước rơi vào fallback M=1. Phần lớn lợi ích tốc độ (1.40x) đến từ CUDA graph, không phải từ giải mã suy đoán.

**Thử cứu A (graph M=2 cho trường hợp chỉ có 1 bản nháp):** kết quả 1.009x, không đổi (`a3_graph_pld_k2_m2_gsm8k30_v1.json`), vẫn đúng từng bit 30/30. Lý do: trường hợp 1 bản nháp gần như không xảy ra (khoảng 3 trên 3000 bước); 2044 trên khoảng 3000 bước rơi về M=1 vì tra n-gram không có kết quả, và mỗi khối M=3 chấp nhận trung bình 0.86 bản nháp. K=3 hay hơn cần projection xử lý riêng (C2) nên không thử.

**Ý nghĩa:** CUDA graph cho phép baseline exact nhanh 1.40x so với eager mà không đổi số học; đó là kết quả có giá trị riêng, nhưng giải mã suy đoán exact K=2 không cộng thêm đáng kể trên nền đó trong thiết lập này. Giới hạn: một GPU, một mô hình (Qwen2.5-3B bnb), một lần chạy mỗi prompt, thời gian đo bằng đồng hồ tường, chưa thử K khác hay các nguồn bản nháp khác.

### 2.6 Rủi ro
- **Tham chiếu mới:** bài toán đổi từ "exact so với bitsandbytes eager" sang "exact so với bước M=1 tĩnh". Cần lập luận rõ và ghi nhận tham chiếu cũ vẫn có giá trị với kết quả B2 hiện tại.
- **Cache tĩnh với độ dài KV cố định:** attention tính trên toàn độ dài với mặt nạ, khác số học so với cache động; mọi ngữ cảnh phải xử lý nhất quán giữa M=1 và M=3.
- **Lợi ích nhỏ so với baseline graph exact (4–14%):** nếu ước lượng đúng thì kết quả khiêm tốn.
- **Không phải ý tưởng thuật toán mới tuyệt đối:** là kỹ thuật hệ thống, giá trị nằm ở việc chứng minh nó giữ được tính exact.
- **Giới hạn bộ nhớ:** graph giữ bộ nhớ tĩnh; cần đo VRAM lại.

### 2.7 Nơi phù hợp
Một đóng góp hệ thống nếu A4 cho kết quả tích cực và ngang baseline biên dịch; phù hợp MLSys nếu kết hợp với C. Riêng A nhiều khả năng chưa đủ cho hội nghị hàng đầu.

---

## 3. Phối hợp và thứ tự khuyến nghị

1. **C2 trước:** rẻ nhất, tận dụng được ngay, và kết quả (đặc biệt là trục M và độ dài) định hướng cho A. Nếu attention lệch rất khác nhau theo độ dài hoặc M, điều đó thay đổi thiết kế A.
2. **A1–A2 song song:** hai mốc này rẻ và quyết định khả thi. Chạy xong A2 mới quyết định có đi tiếp A3–A5 hay không.
3. **C3–C4** trong lúc chờ A.
4. **C5–C6**, rồi tổng hợp.

**Định hướng khi viết bài (nếu cả hai thành công):** "Giải mã suy đoán exact trên mô hình lượng tử hóa: nguồn gốc sai lệch theo toán tử (C) và một vòng giải mã exact chạy trong CUDA graph (A)". C là phần mới và có độ rộng; A là minh chứng ứng dụng giải quyết điểm yếu về tốc độ.

---

## 4. Việc cần bạn

- Chạy các GPU và phiên bản Triton/bitsandbytes khác khi có điều kiện (C7); các script hiện có chạy được bằng `reproduce.sh --b2` cho phần B2 và có thể bổ sung chế độ cho phần C.
- Quyết định có chuyển tham chiếu sang `StaticCache` cho hướng A hay giữ cache động (nếu giữ cache động thì A phải giải thêm bài toán attention độ dài thay đổi trong graph, khó hơn đáng kể).
- Chọn cohort held-out mới cho A5 (cohort 300–349 và 400–449 đã dùng).

## 5. Bằng chứng và script liên quan

| Nội dung | Tệp |
|---|---|
| Công cụ thống nhất và bảng tổng hợp (C5) | `scripts/shape_audit.py`, `C_SHAPE_AUDIT_SUMMARY.md` |
| Quét M/ngữ cảnh, mô hình khác, HQQ, quỹ đạo PLD (C2–C6) | `scripts/run_c2_sweep.sh`, `run_c3_models.sh`, `run_c4_families.sh`, `run_c6.sh`, `audit_pld_trajectory_divergence.py` |
| Phân loại toán tử theo hình dạng (C1) | `scripts/audit_op_shape_sensitivity.py`, `op_shape_sensitivity_gsm8k20_v1.json` |
| B2 không dùng kernel Triton | `scripts/run_b2_without_triton_linears.py`, `b2_lazy_no_triton_gsm8k10_v1.json` |
| Thứ tự rút gọn kernel bitsandbytes | `scripts/emulate_nf4_gemv_orders.py`, `nf4_gemv_order_emulation_v1.json` |
| Thăm dò CUDA graph (A) | `scripts/probe_cuda_graph_decode.py`, `cuda_graph_probe_m{1,3}_v1.json` |
| Mô phỏng chi phí/tốc độ | `scripts/simulate_block_size.py`, `block_size_simulation_v1.json` |
| Chi phí CPU của forward | `scripts/profile_b2_launch_bound.py`, `launch_bound_gsm8k_000_v1.json` |
| Lập luận theo toán tử (đã hiệu chỉnh) | `experiments/exact_quantized_feasibility/B2_OPERATOR_ARGUMENT.md` |
| Ghi chép trạng thái | `RESEARCH_STATUS_20261102.md`, mục 21 |

**Giới hạn chung của mọi con số trên:** một GPU (RTX 4050 Laptop 6 GB), bitsandbytes 0.50.0, Triton 3.6.0, PyTorch 2.11.0, chủ yếu Qwen2.5-3B, batch 1, greedy, tập dev trừ khi ghi khác.
