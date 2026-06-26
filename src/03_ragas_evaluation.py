"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness >= 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

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

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from ragas import evaluate, EvaluationDataset, SingleTurnSample, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, context_recall, context_precision

from utils.llm_factory import get_llm
from utils.data_loader import load_knowledge_base, split_text
from qa_pairs import QA_PAIRS

_RETRIEVAL_CACHE = Path(__file__).parent.parent / "data" / "retrieval_cache.json"

def _load_ret_cache() -> dict:
    if _RETRIEVAL_CACHE.exists():
        return json.loads(_RETRIEVAL_CACHE.read_text(encoding="utf-8"))
    return {}

def _save_ret_cache(cache: dict):
    _RETRIEVAL_CACHE.parent.mkdir(exist_ok=True)
    _RETRIEVAL_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

_ret_cache = _load_ret_cache()
print(f"Retrieval cache: {len(_ret_cache)} questions cached")


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện và hữu ích. "
    "Chỉ dùng context dưới đây để trả lời câu hỏi. "
    "Giữ câu trả lời ngắn gọn, rõ ràng (2-4 câu). "
    "Nếu không tìm thấy thông tin trong context, hãy nói thẳng là không biết.\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin với kinh nghiệm chuyên sâu. "
    "Khi trả lời, hãy: "
    "1) Xác định và tóm tắt thông tin chính từ context, "
    "2) Trình bày câu trả lời có cấu trúc rõ ràng (3-5 câu), "
    "3) Nếu context không đủ thông tin, hãy nêu rõ giới hạn đó. "
    "Luôn dựa trên dữ liệu được cung cấp, không suy đoán thêm.\n\n"
    "Context:\n{context}"
)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup BM25 Retriever (keyword-based, không cần embedding API) ─────
def setup_bm25_retriever():
    """BM25 dùng TF-IDF/keyword search — zero embedding API calls."""
    print("Building BM25 retriever từ knowledge base (no embedding needed)...")
    text   = load_knowledge_base()
    chunks = split_text(text)
    retriever = BM25Retriever.from_texts(chunks, k=3)
    print(f"BM25 ready với {len(chunks)} chunks.")
    return retriever


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.
    Dùng retrieval_cache để tránh gọi lại retriever nhiều lần.
    """
    global _ret_cache

    if question in _ret_cache:
        contexts = _ret_cache[question]
    else:
        docs     = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        _ret_cache[question] = contexts
        _save_ret_cache(_ret_cache)

    ctx_str = "\n\n".join(contexts)
    answer  = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(retriever, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    llm    = get_llm()
    prompt = PROMPTS[prompt_version]

    results = []
    print(f"\nDang chay 50 cau hoi voi prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset."""
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá với 3 LLM-based metrics (bỏ answer_relevancy vì cần embedding quota).
    faithfulness, context_recall, context_precision đều dùng LLM → không cần embedding API.
    """
    print(f"\nDang danh gia RAGAS cho prompt {version} ... (~5-10 phut)")

    dataset  = build_ragas_dataset(rag_results)
    llm_eval = LangchainLLMWrapper(get_llm(temperature=0))

    result = evaluate(
        dataset,
        metrics=[faithfulness, context_recall, context_precision],
        llm=llm_eval,
        run_config=RunConfig(max_workers=1, max_retries=5, timeout=120),
    )

    scores = {}
    for key in ["faithfulness", "context_recall", "context_precision"]:
        raw = result[key]
        scores[key] = float(np.mean([v for v in raw if v is not None]))

    print(f"\nKet qua RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ***" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    retriever = setup_bm25_retriever()

    v1_results = collect_rag_outputs(retriever, "v1")
    v2_results = collect_rag_outputs(retriever, "v2")

    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "<- V1" if s1 > s2 else "<- V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\nDat muc tieu: faithfulness = {best_faith:.4f} >= 0.8")
    else:
        print(f"\nChua dat muc tieu ({best_faith:.4f} < 0.8).")
        print("   Goi y: giam chunk_size, tang k, hoac dieu chinh prompt.")

    # Lưu báo cáo (answer_relevancy bị bỏ qua do embedding quota)
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
        "note": "answer_relevancy skipped (embedding quota exhausted); using BM25 retrieval",
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDa luu bao cao vao {report_path}")


if __name__ == "__main__":
    main()
