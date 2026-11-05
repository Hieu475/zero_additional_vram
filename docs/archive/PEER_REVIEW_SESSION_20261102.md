# Ghi chú phiên đánh giá peer review — 02/11/2026

Phiên này đánh giá một bản nhận xét bên ngoài (peer review) về dự án, sau đó thực hiện các cập nhật cụ thể dựa trên kết quả kiểm tra.

---

## 1. Những gì đã làm được trong phiên này

### 1.1. Đánh giá độ chính xác của bản review

Bản review ngoài đưa ra một số nhận định về trạng thái nghiên cứu. Đã kiểm tra từng nhận định quan trọng nhất đối chiếu với code, FINDINGS.md và tài liệu thực tế:

| Nhận định trong review | Kiểm tra thực tế | Kết quả |
|---|---|---|
| B1/B2 distinction là vấn đề cốt lõi | README.md đã phân biệt rõ B1 vs B2 | Chính xác, nhưng review không phát hiện thêm gì mới |
| "Phase 1: audit 252 quantized linear modules chưa được làm" | FINDINGS.md ghi rõ đã audit 252 modules, 9,289,728/9,289,728 elements exact | **Review SAI** — Phase 1 đã hoàn thành |
| LLM-42 (Microsoft Research, SOSP 2026) là công trình cạnh tranh | Tìm kiếm độc lập, fetch abstract từ arXiv:2601.17768 và ACM DOI:10.1145/3830418.3843901 | **Xác nhận có thật** |
| LLM-42 chưa được biết đến trong project | Kiểm tra RESEARCH_EXACT_QUANTIZED_SPECULATIVE_DECODING.md dòng 256 và 276 | **Review SAI** — đã được reference trong research plan từ trước |
| LLM-42 chưa có trong báo cáo/bib | Kiểm tra references.bib và related_works.tex | **Đúng** — đây là gap thực sự cần bổ sung |

### 1.2. Phân tích LLM-42 để xác định khoảng cách thực sự

Fetched và đọc full abstract + nội dung của LLM-42:

**LLM-42 làm gì:**
- Decode–verify–rollback (DVR) để đảm bảo determinism trong LLM serving
- Mô hình: Llama-3.1-8B-Instruct (FP16)
- Hardware: 4× H100 PCIe 80GB
- Vấn đề giải quyết: non-determinism do dynamic batching + floating-point non-associativity
- Overhead chỉ phát sinh với traffic yêu cầu deterministic (selective)
- Implementation trên SGLang v0.5.3

**LLM-42 KHÔNG làm:**
- Không đề cập NF4, bitsandbytes, hay bất kỳ 4-bit quantized backend nào
- Không giải quyết kernel-level arithmetic equivalence trong quantized inference
- Không nghiên cứu edge/single-GPU scenario với memory constraints

**Kết luận về differentiation:** Khoảng cách là rõ ràng và thực chất. LLM-42 giải quyết system-level non-determinism ở serving scale (FP16, multi-GPU). Dự án này giải quyết arithmetic-level exactness dưới 4-bit quantization (NF4, single GPU, bitsandbytes). Đây là hai bài toán khác nhau về nguồn gốc vấn đề.

---

## 2. Thay đổi đã thực hiện

### 2.1. `bao_cao/references.bib`
Thêm entry `gond2026llm42`:
```bibtex
@inproceedings{gond2026llm42,
  author = {Gond, Raja and Kamath, Aditya K and Ramjee, Ramachandran and Panwar, Ashish},
  title = {{LLM-42}: Enabling Determinism in {LLM} Inference with Verified Speculation},
  booktitle = {Proceedings of the ACM Symposium on Operating Systems Principles},
  year = {2026},
  doi = {10.1145/3830418.3843901},
  url = {https://arxiv.org/abs/2601.17768}
}
```

### 2.2. `bao_cao/contents/related_works.tex`

**Thêm đoạn văn vào mục "Systems efficiency và memory management":**
Mô tả LLM-42 và phân biệt rõ bài toán của đồ án so với LLM-42 (FP16 vs NF4, serving multi-GPU vs single GPU, system-level vs arithmetic-level).

**Thêm dòng vào bảng so sánh (Tab. 2.1):**
```
LLM-42 | Decode–verify–rollback với fixed-shape schedule; không thêm trọng số draft. | Công trình gần nhất về verify–rollback; khác đồ án ở mô hình FP16, hệ đa GPU, và không xử lý lượng tử hóa 4-bit.
```

---

## 3. Những gì review ngoài đánh giá đúng

- B1/B2 distinction cần giữ sắc nét khi viết paper — đây là ranh giới quan trọng nhất
- Provenance/reproducibility (Phase 0) là việc nên làm trước khi viết paper: manifest cũ bị broken SHA là rủi ro thật
- Metrics phải bao gồm KV-prefix equality, first divergence position — không chỉ TPS
- MLSys là venue phù hợp nhất với hướng nghiên cứu hiện tại
- Không nên mở rộng tile/warp sweep nếu B2 chưa được giải quyết

## 4. Những gì review ngoài đánh giá sai hoặc thiếu

- **Phase 1 đã xong:** Review đề xuất "audit 252 linear modules" như việc phải làm, nhưng FINDINGS.md ghi nhận đã hoàn thành (9,289,728/9,289,728 elements exact cho BNB M=3 và BNB singleton M=1 trên 3 development prompts)
- **LLM-42 đã được biết:** Research plan đã reference LLM-42 tại dòng 256 và 276 của RESEARCH_EXACT_QUANTIZED_SPECULATIVE_DECODING.md
- **Review đọc docs là chính, không đọc code:** Nhiều scripts trong `scripts/` (audit_bnb_nf4_gate_reduction_orders.py, audit_nf4_singleton_k2_r3_closed_loop.py, ...) đã làm đúng việc reviewer đề xuất, nhưng reviewer không phát hiện ra
- **"252 modules" là cách frame sai:** Vấn đề là số kernel dispatch paths, không phải số modules

## 5. Việc chưa làm và cần làm tiếp

| Việc | Lý do chưa làm | Ưu tiên |
|---|---|---|
| Verify B2 equality (BNB NF4 singleton) trên kernel hiện tại | Cần chạy GPU | Cao nhất |
| Đóng băng artifact + tạo reproducible benchmark | Cần cố định code + data | Cao |
| Mở rộng held-out cohort (GSM8K indices 300–349) | Chưa dùng theo đúng protocol | Trung bình |
| Thêm GPU thứ hai (Ampere/Hopper) | Cần access hardware | Thấp hơn |
| Cập nhật RESEARCH_STATUS.md (còn timestamp 13/10/2026) | Chưa review | Thấp |

---

## 6. Đánh giá tổng thể về bản review ngoài

Review có giá trị như **external sanity check**. Nó xác nhận đúng các ranh giới quan trọng (B1 vs B2, LLM-42 là paper gần nhất), và roadmap theo phase là hợp lý. Nhưng nó không thể thay thế việc đọc code thực tế — một số recommendation đã được thực hiện từ trước. Giá trị thực của review là ở việc nhắc lại rằng **B2 vẫn là nút thắt số một** và provenance cần được chuẩn hóa trước khi viết paper.

---

*Tạo bởi phiên Claude Code 02/11/2026. LLM-42 citation đã được thêm vào `bao_cao/references.bib` và `bao_cao/contents/related_works.tex`.*
