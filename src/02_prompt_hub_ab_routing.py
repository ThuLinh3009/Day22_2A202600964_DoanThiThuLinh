"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import time
import hashlib
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.globals import set_llm_cache
try:
    from langchain_community.cache import SQLiteCache
    _CACHE_DB = Path(__file__).parent.parent / "data" / "llm_cache.db"
    set_llm_cache(SQLiteCache(database_path=str(_CACHE_DB)))
except ImportError:
    pass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS

_FAISS_PATH = Path(__file__).parent.parent / "data" / "faiss_step1"
_EVIDENCE   = Path(__file__).parent.parent / "evidence"


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
PROMPT_V1_NAME = "thu-linh-rag-prompt-v1"
PROMPT_V2_NAME = "thu-linh-rag-prompt-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện và hữu ích. "
    "Chỉ dùng context dưới đây để trả lời câu hỏi. "
    "Giữ câu trả lời ngắn gọn, rõ ràng (2-4 câu). "
    "Nếu không tìm thấy thông tin trong context, hãy nói thẳng là không biết.\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{question}")])

SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin với kinh nghiệm chuyên sâu. "
    "Khi trả lời, hãy: "
    "1) Xác định và tóm tắt thông tin chính từ context, "
    "2) Trình bày câu trả lời có cấu trúc rõ ràng (3-5 câu), "
    "3) Nếu context không đủ thông tin, hãy nêu rõ giới hạn đó. "
    "Luôn dựa trên dữ liệu được cung cấp, không suy đoán thêm.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", "{question}")])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client: Client):
    for name, prompt, desc in [
        (PROMPT_V1_NAME, PROMPT_V1, "V1 – ngắn gọn, thân thiện"),
        (PROMPT_V2_NAME, PROMPT_V2, "V2 – chuyên nghiệp, có cấu trúc"),
    ]:
        try:
            url = client.push_prompt(name, object=prompt, description=desc)
            print(f"Pushed {name} -> {url}")
        except Exception as e:
            if "409" in str(e) or "Nothing to commit" in str(e):
                print(f"{name} da ton tai tren Hub (khong co thay doi)")
            else:
                print(f"Push {name} loi: {e}")


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    prompts = {}
    for name, fallback in [(PROMPT_V1_NAME, PROMPT_V1), (PROMPT_V2_NAME, PROMPT_V2)]:
        try:
            prompts[name] = client.pull_prompt(name)
            print(f"Pulled '{name}' from Hub")
        except Exception:
            prompts[name] = fallback
            print(f"Fallback local cho '{name}'")
    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """hash(request_id) % 2 → V1 hoặc V2 (deterministic)."""
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    docs    = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer  = (prompt | llm | StrOutputParser()).invoke({
        "context":  context,
        "question": question,
    })
    return {"question": question, "answer": answer, "version": version}


# ── 7. Setup Vectorstore (dùng FAISS từ Task 1 nếu có) ───────────────────
def setup_vectorstore():
    embeddings = get_embeddings()
    index_dir  = _FAISS_PATH / "index.faiss"

    if index_dir.exists():
        print("Loading FAISS từ disk (shared với Task 1)...")
        return FAISS.load_local(
            str(_FAISS_PATH), embeddings,
            allow_dangerous_deserialization=True,
        )

    print("Building FAISS...")
    text   = load_knowledge_base()
    chunks = split_text(text)
    vs     = build_vectorstore(chunks, embeddings)
    _FAISS_PATH.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(_FAISS_PATH))
    return vs


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    client = Client(api_key=config.LANGSMITH_API_KEY)
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    results   = []
    v1_count = v2_count = 0

    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        try:
            t0     = time.time()
            result = ask_ab(retriever, llm, prompt, question, version_tag)
            elapsed = time.time() - t0
            cached  = elapsed < 0.5

            results.append({
                "request_id": request_id,
                "version":    version_tag,
                "question":   question,
                "answer":     result["answer"],
                "cached":     cached,
            })

            if version_tag == "v1":
                v1_count += 1
            else:
                v2_count += 1

            tag = "(cache)" if cached else f"({elapsed:.1f}s)"
            print(f"[{i+1:02d}] [{version_tag}] {tag} {question[:50]}...")

        except Exception as e:
            err = str(e)
            if "429" in err or "EXHAUSTED" in err:
                print(f"[{i+1:02d}] Quota het — dung lai. Da xu ly {len(results)} cau.")
                break
            print(f"[{i+1:02d}] Loi: {err[:80]}")

        time.sleep(4)  # 10 RPM cho gemini-3.1-flash-lite

    # Lưu log
    _EVIDENCE.mkdir(exist_ok=True)
    log_path = _EVIDENCE / "02_ab_routing_log.json"
    log_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRouting: V1={v1_count} | V2={v2_count} | Total={len(results)}")
    print(f"Log saved: {log_path}")
    print("Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()
