"""
Factory tạo LLM và Embeddings cho 5 providers: openai, gemini, anthropic, ollama, openrouter.

Cách dùng:
    from utils.llm_factory import get_llm, get_embeddings

    llm        = get_llm()            # dùng PROVIDER từ .env
    embeddings = get_embeddings()     # dùng PROVIDER từ .env

    llm_gemini = get_llm("gemini")    # chỉ định provider cụ thể
"""
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from langchain_core.embeddings import Embeddings


class _GeminiEmbeddingsV1(Embeddings):
    """
    Custom Gemini embeddings dùng v1 API trực tiếp.
    langchain-google-genai 4.x mặc định dùng v1beta endpoint không support embedContent.
    """

    def __init__(self, api_key: str, model: str = "text-embedding-004"):
        from google import genai
        self.model = model
        self.client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1"},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Embed từng text một với retry để xử lý 429/503
        all_embeddings = []
        for i, text in enumerate(texts):
            for attempt in range(5):
                try:
                    result = self.client.models.embed_content(
                        model=self.model,
                        contents=[text],
                    )
                    all_embeddings.append(result.embeddings[0].values)
                    break
                except Exception as e:
                    err = str(e)
                    if "429" in err or "503" in err or "UNAVAILABLE" in err or "EXHAUSTED" in err:
                        wait = (attempt + 1) * 10
                        print(f"  Rate limit/server error, waiting {wait}s... (attempt {attempt+1})")
                        time.sleep(wait)
                    else:
                        raise
            if (i + 1) % 20 == 0:
                print(f"  Embedded {i+1}/{len(texts)} chunks...")
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        waits = [15, 30, 60, 90, 120, 180, 240]
        for attempt, wait in enumerate(waits):
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=[text],
                )
                return result.embeddings[0].values
            except Exception as e:
                err = str(e)
                if "429" in err or "503" in err or "UNAVAILABLE" in err or "EXHAUSTED" in err:
                    print(f"  embed_query retry {attempt+1}/{len(waits)}: waiting {wait}s... ({err[:60]})")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("embed_query failed after all retries")


def get_llm(provider: str = None, temperature: float = 0.0):
    """
    Trả về BaseChatModel tương ứng với provider được chọn.

    Args:
        provider    : "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                      Mặc định: đọc PROVIDER từ .env (config.PROVIDER)
        temperature : độ ngẫu nhiên (0.0 = tất định, 1.0 = sáng tạo)

    Returns:
        BaseChatModel instance sẵn sàng sử dụng

    Raises:
        ValueError nếu provider không hợp lệ
        ImportError nếu package tương ứng chưa được cài đặt
    """
    provider = (provider or config.PROVIDER).lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": config.OPENAI_MODEL,
            "api_key": config.OPENAI_API_KEY,
            "temperature": temperature,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    elif provider == "openrouter":
        # OpenRouter dùng OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENROUTER_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )


def get_embeddings(provider: str = None):
    """
    Trả về Embeddings instance tương ứng với provider được chọn.

    Lưu ý quan trọng:
        - Anthropic KHÔNG có Embeddings API → tự động fallback về OpenAI embeddings
        - OpenRouter cũng dùng OpenAI embeddings (không có API embeddings riêng)
        - Ollama cần model embedding riêng (mặc định: nomic-embed-text)
          Cài đặt: ollama pull nomic-embed-text

    Args:
        provider: "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                  Mặc định: đọc PROVIDER từ .env

    Returns:
        Embeddings instance sẵn sàng sử dụng
    """
    provider = (provider or config.PROVIDER).lower()

    if provider in ("openai", "openrouter"):
        from langchain_openai import OpenAIEmbeddings
        kwargs = {
            "model": config.OPENAI_EMBEDDING_MODEL,
            "api_key": config.OPENAI_API_KEY,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return OpenAIEmbeddings(**kwargs)

    elif provider == "gemini":
        # langchain-google-genai 4.x dùng v1beta endpoint không hỗ trợ embedding.
        # Dùng google-genai SDK trực tiếp với v1 API.
        return _GeminiEmbeddingsV1(
            api_key=config.GOOGLE_API_KEY,
            model=config.GEMINI_EMBEDDING_MODEL,
        )

    elif provider == "anthropic":
        # Anthropic không cung cấp Embeddings API → dùng OpenAI thay thế
        print("⚠️  Anthropic không có Embeddings API — đang dùng OpenAI embeddings thay thế.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )

    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )
