from pathlib import Path

import torch
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
    YoutubeLoader,
    DirectoryLoader,
)
from llama_cpp import Llama


class SettingsConfig:
    '''Settings'''
    SUBTITLES_LANGUAGES = ['ru', 'en']
    LOADER_CLASSES = {
        '.csv': CSVLoader,
        '.doc': UnstructuredWordDocumentLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.html': UnstructuredHTMLLoader,
        '.md': UnstructuredMarkdownLoader,
        '.pdf': PDFMinerLoader,
        '.ppt': UnstructuredPowerPointLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.txt': TextLoader,
        'web': WebBaseLoader,
        'directory': DirectoryLoader,
        'youtube': YoutubeLoader,
    }


class PromptConfig:
    '''Prompts'''
    # системынй промт по умолчанию
    DEFAULT_SYSTEM_PROMPT: str | None = None
    # шаблон промта при условии контекста
    CONTEXT_TEMPLATE: str = '''Ответь на вопрос при условии контекста.

Контекст:
{context}

Вопрос:
{user_message}

Ответ:'''


class ModelConfig:
    '''Configuration of paths, models and generation parameters'''
    LLM_MODELS_PATH: Path = Path('models')
    EMBED_MODELS_PATH: Path = Path('embed_models')
    START_LLM_MODEL_REPO: str = 'bartowski/google_gemma-3-1b-it-GGUF'
    START_LLM_MODEL_FILE: str = 'google_gemma-3-1b-it-Q8_0.gguf'
    START_EMBED_MODEL_REPO: str = 'sergeyzh/rubert-tiny-turbo'
    LLM_MODELS: dict[str, Llama] = {}
    EMBED_MODELS: dict[str, Embeddings] = {}
    LLM_MODELS_PATH.mkdir(exist_ok=True)
    EMBED_MODELS_PATH.mkdir(exist_ok=True)
    LLAMA_MODEL_KWARGS = dict(
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=False,
        local_dir=LLM_MODELS_PATH,
    )
    GENERATE_KWARGS = dict(
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        repeat_penalty=1.0,
    )
    SHOW_THINKING = False
    EMBED_MODEL_USE_CUDA_IF_AVAILABLE = True


class ReposConfig:
    '''Links to repositories with ggu models'''
    LLM_MODEL_REPOS: list[str] = [
        # https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF
        'bartowski/google_gemma-3-1b-it-GGUF',
        # https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF
        'bartowski/google_gemma-3-4b-it-GGUF',
        # https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF
        'bartowski/Qwen_Qwen3-1.7B-GGUF',
        # https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF
        'bartowski/Qwen_Qwen3-4B-GGUF',
    ]
    EMBED_MODEL_REPOS: list[str] = [
        # https://huggingface.co/sergeyzh/rubert-tiny-turbo  # 117 MB
        'sergeyzh/rubert-tiny-turbo',
        # https://huggingface.co/intfloat/multilingual-e5-large  # 2.24 GB
        'intfloat/multilingual-e5-large',
        # https://huggingface.co/intfloat/multilingual-e5-base  # 1.11 GB
        'intfloat/multilingual-e5-base',
        # https://huggingface.co/intfloat/multilingual-e5-small  # 471 MB
        'intfloat/multilingual-e5-small',
        # https://huggingface.co/intfloat/multilingual-e5-large-instruct  # 1.12 GB
        'intfloat/multilingual-e5-large-instruct',
        # https://huggingface.co/sentence-transformers/all-mpnet-base-v2  # 438 MB
        'sentence-transformers/all-mpnet-base-v2',
        # https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2  # 1.11 GB
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        # https://huggingface.co/collections/ai-forever/ruelectra-family-6650d3b874664a42ddfec4d1  # 356 MB
        'ai-forever/ruElectra-medium',
        # https://huggingface.co/ai-forever/sbert_large_nlu_ru  # 1.71 GB
        'ai-forever/sbert_large_nlu_ru',  # 138 MB
        # https://huggingface.co/collections/deepvk/user2-6802650d7210f222ec60e05f
        'deepvk/USER2-small',
        # https://huggingface.co/BAAI/bge-m3#specs  # 1.16 GB
        'BAAI/bge-m3-retromae',
    ]