"""
Bước 1 — RAG Pipeline với LangSmith Tracing
=============================================
NHIỆM VỤ:
  1. Tải knowledge base, chia chunks, index với FAISS
  2. Xây dựng RAG chain: retriever → prompt → LLM → output parser
  3. Trang trí hàm query với @traceable để LangSmith ghi lại mỗi lần gọi
  4. Chạy 100+ câu hỏi → tạo ≥ 100 traces trên LangSmith

DELIVERABLE: Mở https://smith.langchain.com → project của bạn → xác nhận ≥ 100 traces.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ⚠️ QUAN TRỌNG: Import config TRƯỚC KHI import bất kỳ thư viện LangChain nào.
import config

# ── LangChain SQLiteCache: cache LLM responses, tránh tốn quota ──────────────
from langchain_core.globals import set_llm_cache
try:
    from langchain_community.cache import SQLiteCache
    _CACHE_DB = Path(__file__).parent.parent / "data" / "llm_cache.db"
    _CACHE_DB.parent.mkdir(exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(_CACHE_DB)))
    print("LLM cache enabled:", _CACHE_DB.name)
except ImportError:
    print("SQLiteCache not available, proceeding without cache.")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langsmith import traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS

_FAISS_PATH = Path(__file__).parent.parent / "data" / "faiss_step1"


# ── 1. Thiết lập Vectorstore (với persistence) ─────────────────────────────
def setup_vectorstore():
    """
    Load FAISS từ disk nếu đã tồn tại, nếu không thì build và save.
    Tránh tốn embedding quota khi chạy lại nhiều lần.
    """
    embeddings = get_embeddings()
    index_file = _FAISS_PATH / "index.faiss"

    if index_file.exists():
        print("Loading FAISS vectorstore từ cache (bỏ qua re-embedding)...")
        return FAISS.load_local(
            str(_FAISS_PATH), embeddings,
            allow_dangerous_deserialization=True
        )

    print("Lần đầu chạy — đang build FAISS...")
    text   = load_knowledge_base()
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    print(f"Đã chia thành {len(chunks)} chunks")
    vectorstore = build_vectorstore(chunks, embeddings)
    _FAISS_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(_FAISS_PATH))
    print("Saved FAISS index to disk.")
    return vectorstore


# ── 2. RAG Prompt Template ─────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Bạn là trợ lý AI hữu ích. Chỉ dùng context sau để trả lời.\n\nContext:\n{context}"),
    ("human",  "{question}"),
])


# ── 3. Build RAG Chain ─────────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    llm       = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ── 4. Hàm Query có LangSmith Tracing ─────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """
    Chạy RAG chain với một câu hỏi.
    @traceable gửi mỗi lần gọi lên LangSmith như một trace riêng.
    Với SQLiteCache, câu hỏi đã trả lời sẽ dùng cache — không tốn API quota.
    """
    return chain.invoke(question)


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 1: LangSmith RAG Pipeline")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore      = setup_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)

    # Chiến lược 100+ traces với free tier 20 RPD:
    # - Chạy 20 câu hỏi unique lần đầu → real API call → cached vào SQLite
    # - Chạy lại 5 lần nữa → dùng cache → không tốn quota, vẫn tạo LangSmith trace
    # Tổng: 20 unique API calls × 5 runs = 100 traces
    BASE_QUESTIONS = SAMPLE_QUESTIONS[:20]  # 20 unique questions
    NUM_ROUNDS     = 5                       # chạy 5 vòng = 100 traces
    all_questions  = BASE_QUESTIONS * NUM_ROUNDS
    total          = len(all_questions)

    success_count = 0
    cache_hits    = 0

    for i, question in enumerate(all_questions, 1):
        round_num = (i - 1) // len(BASE_QUESTIONS) + 1
        is_first_round = round_num == 1

        try:
            t0     = time.time()
            answer = ask(chain, question)
            elapsed = time.time() - t0
            success_count += 1

            is_cached = elapsed < 1.0 and not is_first_round
            if is_cached:
                cache_hits += 1
                tag = "(cache)"
            else:
                tag = f"({elapsed:.1f}s)"

            print(f"[{i:03d}/{total}] Round {round_num} {tag} Q: {question[:50]}")
            print(f"         A: {str(answer)[:80]}\n")

        except Exception as e:
            err = str(e)
            if "429" in err or "EXHAUSTED" in err:
                print(f"[{i:03d}/{total}] Quota hết (round {round_num}) — dừng lại.")
                print(f"  Đã tạo {success_count} traces ({cache_hits} từ cache).")
                print("  Chạy lại script sau khi quota reset để tiếp tục tích lũy traces.")
                break
            else:
                print(f"[{i:03d}/{total}] Lỗi: {err[:100]}")

        # Chỉ sleep khi cần gọi real API (round đầu), cache không cần sleep
        if is_first_round:
            time.sleep(13)  # gemini-2.5-flash: 5 RPM → 13s giữa mỗi real call

    print(f"\n{success_count}/{total} traces gửi lên LangSmith '{config.LANGSMITH_PROJECT}'")
    print(f"  Trong đó: {success_count - cache_hits} real API calls, {cache_hits} từ cache")
    print("  Mở https://smith.langchain.com để xem traces.")


if __name__ == "__main__":
    main()
