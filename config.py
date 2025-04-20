from pathlib import Path

import torch
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


# langchain classes for extracting text from various sources
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

# languages ​​for youtube subtitles
SUBTITLES_LANGUAGES = ['ru', 'en']

# prom template subject to context
CONTEXT_TEMPLATE = '''Ответь на вопрос при условии контекста.

Контекст:
{context}

Вопрос:
{user_message}

Ответ:'''

# paths to LLM and embeddings models 
LLM_MODELS_PATH = Path('models')
EMBED_MODELS_PATH = Path('embed_models')
LLM_MODELS_PATH.mkdir(exist_ok=True)
EMBED_MODELS_PATH.mkdir(exist_ok=True)

# dictionary for text generation config
GENERATE_KWARGS = dict(
    temperature=0.2,
    top_p=0.95,
    top_k=40,
    repeat_penalty=1.0,
    )

# llama-cpp-python model params
LLAMA_MODEL_KWARGS = dict(
    n_gpu_layers=-1,
    verbose=True,
    n_ctx=4096,  # context size, 2048, 4096, ...
)

# models devices
# EMBED_MODEL_DEVICE = 'cpu'
EMBED_MODEL_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LLM_MODEL_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if LLM_MODEL_DEVICE == 'cpu':
    LLAMA_MODEL_KWARGS['n_gpu_layers'] = 0

# available when running the LLM application models in GGUF format
LLM_MODEL_REPOS = [
    # https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF
    'bartowski/google_gemma-3-1b-it-GGUF',
    # https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF
    'bartowski/Qwen2.5-1.5B-Instruct-GGUF',
    # https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF
    'bartowski/Qwen2.5-3B-Instruct-GGUF',
    # https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF
    'bartowski/google_gemma-3-4b-it-GGUF',
    # https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF
    'https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF',
    # https://huggingface.co/bartowski/gemma-2-2b-it-GGUF
    'bartowski/gemma-2-2b-it-GGUF',
    # https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF
    'bartowski/Qwen2.5-1.5B-Instruct-GGUF',
]

# GGUF filename to LLM_MODEL_REPOS[0]
# START_LLM_MODEL_FILE = 'Qwen2.5-1.5B-Instruct-Q8_0.gguf'
START_LLM_MODEL_FILE = 'google_gemma-3-1b-it-Q8_0.gguf'

# Embedding models available at application startup
EMBED_MODEL_REPOS = [
    # https://huggingface.co/Alibaba-NLP/gte-multilingual-base  # 611 MB
    'Alibaba-NLP/gte-multilingual-base',
    # https://huggingface.co/intfloat/multilingual-e5-small  # 471 MB
    'intfloat/multilingual-e5-small',
    # https://huggingface.co/sergeyzh/rubert-tiny-turbo  # 117 MB
    'sergeyzh/rubert-tiny-turbo',
    # https://huggingface.co/sergeyzh/BERTA  # 513 MB
    'sergeyzh/BERTA',
    # https://huggingface.co/cointegrated/rubert-tiny2  # 118 MB
    'cointegrated/rubert-tiny2',
    # https://huggingface.co/cointegrated/LaBSE-en-ru  # 516 MB
    'cointegrated/LaBSE-en-ru',
    # https://huggingface.co/sergeyzh/LaBSE-ru-turbo  # 513 MB
    'sergeyzh/LaBSE-ru-turbo',
    # https://huggingface.co/intfloat/multilingual-e5-large  # 2.24 GB
    'intfloat/multilingual-e5-large',
    # https://huggingface.co/intfloat/multilingual-e5-base  # 1.11 GB
    'intfloat/multilingual-e5-base',
    # https://huggingface.co/intfloat/multilingual-e5-large-instruct  # 1.12 GB
    'intfloat/multilingual-e5-large-instruct',
    # https://huggingface.co/sentence-transformers/all-mpnet-base-v2  # 438 MB
    'sentence-transformers/all-mpnet-base-v2',
    # https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2  # 1.11 GB
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    # https://huggingface.co/ai-forever?search_models=ruElectra  # 356 MB
    'ai-forever/ruElectra-medium',
    # https://huggingface.co/ai-forever/sbert_large_nlu_ru  # 1.71 GB
    'ai-forever/sbert_large_nlu_ru',
]
