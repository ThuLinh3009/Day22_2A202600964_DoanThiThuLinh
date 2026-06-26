# Evidence — Day 22 LangSmith + Prompt Versioning Lab

## File Index

| File | Task | Mô tả |
|------|------|--------|
| `01_langsmith_traces.jpg` | Task 1 | LangSmith UI với 121 traces (20 real + 79 SQLiteCache) |
| `02_prompt_hub.jpg` | Task 2 | Prompt Hub hiển thị V1 và V2 |
| `02_ab_routing_log.txt` | Task 2 | Log 50 câu truy vấn A/B routing có nhãn v1/v2 |
| `02_ab_routing_log.json` | Task 2 | Full JSON log với question/answer/version/cached |
| `03_ragas_scores.png` | Task 3 | Terminal output RAGAS evaluation |
| `03_ragas_report.json` | Task 3 | Báo cáo JSON V1 vs V2 |
| `04_pii_demo_log.txt` | Task 4 | Output 6 test cases PII detection |
| `04_json_demo_log.txt` | Task 4 | Output 5 test cases JSON repair |

---

## Phân tích V1 vs V2 — Prompt Hub A/B Testing

### Cấu hình Prompt

**V1 — Ngắn gọn, thân thiện:**
- Trả lời 2-4 câu, tập trung vào thông tin cốt lõi
- Phong cách hội thoại, dễ tiếp cận
- Phù hợp: câu hỏi nhanh, người dùng phổ thông

**V2 — Chuyên nghiệp, có cấu trúc:**
- Trình bày 3-5 câu theo 3 bước rõ ràng
- Tóm tắt → Phân tích → Giới hạn thông tin
- Phù hợp: câu hỏi phức tạp, người dùng chuyên môn

### Kết quả A/B Routing

- Tổng: 50 queries — V1: 19 queries | V2: 31 queries
- Routing tất định bằng MD5 hash: `hash(request_id) % 2`
- Cùng `request_id` → luôn ra cùng version

### Nhận xét

V2 có cấu trúc rõ ràng hơn và thường cung cấp câu trả lời đầy đủ hơn cho
câu hỏi kỹ thuật (RAG, LLM, embeddings). V1 phù hợp hơn cho người dùng
muốn câu trả lời nhanh và ngắn gọn. Trong môi trường sản xuất, V2 nên được
ưu tiên cho người dùng kỹ thuật, V1 cho chatbot hỗ trợ khách hàng phổ thông.

### Ghi chú RAGAS

RAGAS evaluation chạy đủ 50 QA pairs × 2 versions = 100 pairs qua BM25 retriever.
Scoring bị timeout do Gemini Free tier giới hạn concurrency (100 RPM / 1K RPD).
RAG pipeline hoạt động đúng — answers được generate từ context retrieved.
