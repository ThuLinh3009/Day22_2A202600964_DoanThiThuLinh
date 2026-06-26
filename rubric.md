# Bảng Chấm Điểm — Day 22: LangSmith + Prompt Versioning

**Tổng điểm: 100 điểm**
**Bài nộp:** GitHub repository public chứa mã nguồn + thư mục `evidence/` + URL LangSmith project

---

## Tổng quan điểm

| Nhiệm vụ                              | Điểm tối đa | Bằng chứng chính                         |
|---------------------------------------|-------------|------------------------------------------|
| Nhiệm vụ 1 — RAG Pipeline với LangSmith | 25đ         | Ít nhất 50 traces trên LangSmith         |
| Nhiệm vụ 2 — Prompt Hub & A/B Routing   | 25đ         | 2 phiên bản trên Hub + 50 traces nữa     |
| Nhiệm vụ 3 — RAGAS Evaluation           | 25đ         | Báo cáo JSON, faithfulness ≥ 0.8         |
| Nhiệm vụ 4 — Guardrails Validators      | 25đ         | PII bị che, JSON được sửa tự động        |

---

## Nhiệm vụ 1 — RAG Pipeline với LangSmith (25 điểm)

### Tiêu chí chấm điểm

| # | Tiêu chí                                                                                     | Điểm |
|---|----------------------------------------------------------------------------------------------|------|
| 1.1 | Knowledge base được chia chunks và index với FAISS đúng cách                              | 5đ   |
| 1.2 | RAG chain xây dựng với LangChain (retriever → prompt → LLM → parser)                     | 5đ   |
| 1.3 | Decorator `@traceable` được áp dụng; ít nhất 50 traces hiển thị trong LangSmith UI        | 10đ  |
| 1.4 | Các LangSmith traces chứa câu hỏi đầu vào, context được truy xuất và câu trả lời của LLM | 5đ   |

### Trừ điểm

| Lỗi                                                                       | Mức phạt |
|---------------------------------------------------------------------------|----------|
| Ít hơn 50 traces trong LangSmith                                          | −5đ      |
| Traces không chứa context được truy xuất (cấu trúc chain sai)            | −3đ      |
| API key bị hard-code và commit lên mã nguồn                               | −10đ     |
| Không bật `LANGCHAIN_TRACING_V2=true` dẫn đến không có traces nào        | −10đ     |

---

## Nhiệm vụ 2 — Prompt Hub & A/B Routing (25 điểm)

### Tiêu chí chấm điểm

| # | Tiêu chí                                                                                           | Điểm |
|---|----------------------------------------------------------------------------------------------------|------|
| 2.1 | Hai system prompt có ngữ nghĩa khác nhau được soạn thảo                                         | 5đ   |
| 2.2 | Cả 2 prompt được push lên LangSmith Prompt Hub và hiển thị trong giao diện                      | 8đ   |
| 2.3 | Prompt được pull từ Hub khi chạy (không chỉ dùng local, không bypass Hub)                       | 4đ   |
| 2.4 | A/B routing là tất định — cùng một `request_id` luôn được định tuyến đến cùng một phiên bản     | 5đ   |
| 2.5 | Cả 2 phiên bản đều nhận được câu hỏi; log console hiển thị nhãn phiên bản cho từng câu truy vấn | 3đ   |

### Trừ điểm

| Lỗi                                                              | Mức phạt |
|------------------------------------------------------------------|----------|
| Chỉ có 1 phiên bản prompt trên Hub                              | −8đ      |
| Routing là ngẫu nhiên (không tất định)                          | −5đ      |
| Prompt không được pull từ Hub mà dùng thẳng biến local          | −4đ      |
| Log console không hiển thị nhãn phiên bản (v1/v2)              | −3đ      |

---

## Nhiệm vụ 3 — RAGAS Evaluation (25 điểm)

### Tiêu chí chấm điểm

| # | Tiêu chí                                                                                                            | Điểm |
|---|---------------------------------------------------------------------------------------------------------------------|------|
| 3.1 | Tất cả 50 cặp QA được chạy qua **cả 2** phiên bản prompt                                                         | 5đ   |
| 3.2 | `EvaluationDataset` được xây dựng với đúng các trường của `SingleTurnSample`                                      | 5đ   |
| 3.3 | Cả 4 chỉ số được tính: `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`                  | 8đ   |
| 3.4 | Faithfulness ≥ 0.8 cho ít nhất một phiên bản prompt                                                               | 5đ   |
| 3.5 | Tệp `data/ragas_report.json` được lưu với điểm của cả V1 và V2                                                    | 2đ   |

### Trừ điểm

| Lỗi                                                                   | Mức phạt                          |
|-----------------------------------------------------------------------|-----------------------------------|
| Ít hơn 50 cặp QA được đánh giá                                       | −1đ cho mỗi 5 cặp bị thiếu       |
| Chỉ đánh giá 1 phiên bản prompt                                      | −5đ                               |
| Thiếu bất kỳ chỉ số RAGAS nào trong số 4 chỉ số                     | −2đ cho mỗi chỉ số bị thiếu      |
| Faithfulness < 0.8 ở cả 2 phiên bản                                  | −5đ                               |
| Không lưu tệp báo cáo `ragas_report.json`                            | −2đ                               |

### Điểm thưởng

| Tiêu chí thưởng                                                     | Điểm thưởng |
|---------------------------------------------------------------------|-------------|
| Faithfulness ≥ 0.9 ở **cả 2** phiên bản prompt                     | +3đ         |
| Bình luận phân tích giải thích vì sao V1 hoặc V2 có điểm cao hơn  | +2đ         |

---

## Nhiệm vụ 4 — Guardrails AI Validators (25 điểm)

### PII Detector (13 điểm)

| # | Tiêu chí                                                                                                  | Điểm |
|---|-----------------------------------------------------------------------------------------------------------|------|
| 4.1 | Validator tùy chỉnh được tạo bằng decorator `@register_validator`                                       | 3đ   |
| 4.2 | Phát hiện được ít nhất 3 loại PII (địa chỉ email, số điện thoại, SSN, hoặc số thẻ tín dụng)           | 5đ   |
| 4.3 | `on_fail=OnFailAction.FIX` được dùng đúng; đầu ra bị chặn được thay thế bằng chuỗi an toàn            | 3đ   |
| 4.4 | Demo trên ít nhất 5 test case (bao gồm đầu vào sạch và nhiều loại PII khác nhau)                       | 2đ   |

### JSON Formatter (12 điểm)

| # | Tiêu chí                                                                                                         | Điểm |
|---|------------------------------------------------------------------------------------------------------------------|------|
| 4.5 | Validator tùy chỉnh được tạo để kiểm tra khả năng parse JSON                                                   | 3đ   |
| 4.6 | Tự sửa lỗi được triển khai (ít nhất 2 trong số: gỡ markdown fences, sửa nháy đơn, xóa dấu phẩy thừa)         | 5đ   |
| 4.7 | JSON lỗi dự phòng được trả về khi không thể sửa được                                                           | 2đ   |
| 4.8 | Demo trên ít nhất 4 test case (JSON hợp lệ, có fences, bị lỗi nháy đơn, hoàn toàn sai định dạng)              | 2đ   |

### Trừ điểm

| Lỗi                                                                                          | Mức phạt |
|----------------------------------------------------------------------------------------------|----------|
| Dùng validator có sẵn từ Hub thay vì tự implement                                           | −5đ      |
| Tham số `on_fail` được truyền vào `Guard.use()` thay vì constructor của validator           | −3đ      |
| Phát hiện PII không dùng regex (chỉ dùng so khớp chuỗi đơn giản)                           | −3đ      |

---

## Điểm thưởng: Chất lượng bằng chứng & Nộp bài (tối đa +5 điểm)

| Tiêu chí                                                                             | Điểm thưởng |
|--------------------------------------------------------------------------------------|-------------|
| Đầy đủ 7 tệp bằng chứng bắt buộc, rõ ràng và có nhãn đúng                          | +3đ         |
| URL LangSmith project được nộp và có thể truy cập công khai                         | +1đ         |
| `evidence/README.md` chứa phân tích ngắn gọn về kết quả V1 so với V2               | +1đ         |

---

## Điểm thưởng: Chất lượng mã nguồn (tối đa +5 điểm)

| Tiêu chí                                                            | Điểm thưởng |
|---------------------------------------------------------------------|-------------|
| Mã nguồn sạch, có cấu trúc rõ ràng và có docstring                 | +2đ         |
| Tất cả các bước chạy được qua `run_all.py` mà không cần chỉnh sửa  | +2đ         |
| Có xử lý lỗi và fallback hợp lý                                    | +1đ         |

---

## Danh sách kiểm tra trước khi nộp bài

**Tệp mã nguồn — tất cả phải chạy được, không có lỗi:**
- [ ] `src/01_langsmith_rag_pipeline.py`
- [ ] `src/02_prompt_hub_ab_routing.py`
- [ ] `src/03_ragas_evaluation.py`
- [ ] `src/04_guardrails_validator.py`
- [ ] `data/ragas_report.json` — tồn tại và chứa điểm của cả V1 lẫn V2

**Thư mục bằng chứng — tất cả đều bắt buộc:**
- [ ] `evidence/01_langsmith_traces.png` — Giao diện LangSmith với ít nhất 50 traces hiển thị
- [ ] `evidence/02_prompt_hub.png` — Giao diện Prompt Hub hiển thị 2 phiên bản prompt được đặt tên
- [ ] `evidence/02_ab_routing_log.txt` — Log console của A/B routing (50 câu truy vấn, có nhãn v1/v2)
- [ ] `evidence/03_ragas_scores.png` — Output terminal hiển thị bảng so sánh V1 vs V2
- [ ] `evidence/03_ragas_report.json` — Bản sao của `data/ragas_report.json`
- [ ] `evidence/04_pii_demo_log.txt` — Output console của các test case PII
- [ ] `evidence/04_json_demo_log.txt` — Output console của các test case sửa JSON

**Nộp bài:**
- [ ] URL GitHub repository public đã được nộp
- [ ] URL LangSmith project đã được nộp (hiển thị tổng cộng ≥ 100 traces)
- [ ] Không có tệp `.env` nào được commit; không có API key nào trong mã nguồn

**Lưu ý quan trọng: Nếu phát hiện API key trong mã nguồn đã commit, bài bị trừ tự động 10 điểm.**
