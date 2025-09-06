import os
import sys
from pathlib import Path
from typing import Any, ClassVar

import gradio as gr
import torch
from llama_cpp import Llama
from chromadb import EmbeddingFunction

from dotenv import load_dotenv
load_dotenv()


class ModelStorage:
    '''Global model storage'''
    LLM_MODEL: ClassVar[dict[str, Llama]] = {}
    EMBED_MODEL: ClassVar[dict[str, EmbeddingFunction]] = {}


class UiBlocksConfig:
    '''Gradio settings for gr.Blocks()'''
    CSS: str | None = '''
    .gradio-container {
        width: 70% !important;
        margin: 0 auto !important;
    }
    '''
    if hasattr(sys, 'getandroidapilevel') or 'ANDROID_ROOT' in os.environ:
        CSS = None
    UI_BLOCKS_KWARGS: dict[str, Any] = dict(
        theme=None,
        css=CSS,
        analytics_enabled=False,
    )


class InferenceConfig:
    '''Model inference settings'''
    def __init__(self):
        self.encode_kwargs: dict[str, Any] = dict(
            batch_size=300,
            normalize_embeddings=None,
        )
        self.sampling_kwargs: dict[str, Any] = dict(
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.0,
        )
        self.do_sample: bool = False 
        self.rag_mode: bool = False
        self.history_len: int = 0
        self.show_thinking: bool = False


class TextLoadConfig:
    '''Settings for loading texts from documents'''
    def __init__(self):
        self.partition_kwargs: dict[str, str | int | bool | None] = dict(
            chunking_strategy='basic',
            max_characters=800,
            new_after_n_chars=500,
            overlap=0,
            clean=True,
            bullets=True,
            extra_whitespace=True,
            dashes=False,
            trailing_punctuation=True,
            lowercase=False,
        )
        self.SUPPORTED_FILE_EXTS: str = '.csv .tsv .docx .md .org .pdf .pptx .xlsx'
        self.subtitle_lang: str = 'ru'
        self.SUBTITLE_LANGS: list[str] = ['ru', 'en']
        self.max_lines_text_view: int = 200


class DbConfig:
    '''Vector database parameters (Chroma)'''
    def __init__(self):
        self.create_collection_kwargs: dict[str, Any] = dict(
            configuration=dict(
                hnsw=dict(
                    space='cosine',  # l2, ip, cosine, default l2
                    ef_construction=200,
                )
            )
        )
        self.query_kwargs: dict[str, Any] = dict(
            n_results=2,
            max_distance_treshold=0.5,
        )


class PromptConfig:
    '''Prompts'''
    def __init__(self):
        self.system_prompt: str | None = None
        self.user_msg_with_context: str = ''
        self.context_template: str = '''Ответь на вопрос при условии контекста.

Контекст:
{context}

Вопрос:
{user_message}

Ответ:'''


class ModelConfig:
    '''Configuration of paths, models and generation parameters'''
    def __init__(self):
        self.LLM_MODELS_PATH: Path = Path('models')
        self.EMBED_MODELS_PATH: Path = Path('embed_models')
        self.LLM_MODELS_PATH.mkdir(exist_ok=True)
        self.EMBED_MODELS_PATH.mkdir(exist_ok=True)
        self.llm_model_repo: str = 'bartowski/google_gemma-3-1b-it-GGUF'
        self.llm_model_file: str = 'google_gemma-3-1b-it-Q8_0.gguf'
        self.embed_model_repo: str = 'Alibaba-NLP/gte-multilingual-base'
        self.embed_model_kwargs: dict[str, Any] = dict(
            device='cuda:0',
            trust_remote_code=True,
            cache_folder=self.EMBED_MODELS_PATH,
            token=os.getenv('HF_TOKEN'),
            model_kwargs=dict(
                torch_dtype='auto',
            )
        )
        self.llm_model_kwargs: dict[str, Any] = dict(
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False,
            local_dir=self.LLM_MODELS_PATH,
        )


class ReposConfig:
    '''Links to repositories with ggu models'''
    def __init__(self):
        self.llm_model_repos: list[str] = [
            'bartowski/google_gemma-3-1b-it-GGUF',
            'bartowski/google_gemma-3-4b-it-GGUF',
            'bartowski/Qwen_Qwen3-1.7B-GGUF',
            'bartowski/Qwen_Qwen3-4B-GGUF',
        ]
        self.embed_model_repos: list[str] = [
            'Alibaba-NLP/gte-multilingual-base',
            'sergeyzh/rubert-tiny-turbo',
            'intfloat/multilingual-e5-large',
            'intfloat/multilingual-e5-base',
            'intfloat/multilingual-e5-small',
            'intfloat/multilingual-e5-large-instruct',
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            'ai-forever/ruElectra-medium',
            'ai-forever/sbert_large_nlu_ru',
            'deepvk/USER2-small',
            'BAAI/bge-m3-retromae',
        ]


class Config:
    '''General config'''
    def __init__(self):
        self.Inference: InferenceConfig = InferenceConfig()
        self.TextLoad: TextLoadConfig = TextLoadConfig()
        self.Prompt: PromptConfig = PromptConfig()
        self.Db: DbConfig = DbConfig()
        self.Model: ModelConfig = ModelConfig()
        self.Repos: ReposConfig = ReposConfig()

        self.generation_kwargs: dict[str, Any] = dict(
            do_sample=self.Inference.do_sample,
            temperature=self.Inference.sampling_kwargs['temperature'],
            top_p=self.Inference.sampling_kwargs['top_p'],
            top_k=self.Inference.sampling_kwargs['top_k'],
            repeat_penalty=self.Inference.sampling_kwargs['repeat_penalty'],
            history_len=self.Inference.history_len,
            system_prompt=self.Prompt.system_prompt,
            context_template=self.Prompt.context_template,
            show_thinking=self.Inference.show_thinking,
            n_results=self.Db.query_kwargs['n_results'],
            max_distance_treshold=self.Db.query_kwargs['max_distance_treshold'],
            user_msg_with_context=self.Prompt.user_msg_with_context,
            rag_mode=self.Inference.rag_mode,
        )
        self.load_text_kwargs: dict[str, Any] = dict(
            chunking_strategy=self.TextLoad.partition_kwargs['chunking_strategy'],
            max_characters=self.TextLoad.partition_kwargs['max_characters'],
            new_after_n_chars=self.TextLoad.partition_kwargs['new_after_n_chars'],
            overlap=self.TextLoad.partition_kwargs['overlap'],
            clean=self.TextLoad.partition_kwargs['clean'],
            bullets=self.TextLoad.partition_kwargs['bullets'],
            extra_whitespace=self.TextLoad.partition_kwargs['extra_whitespace'],
            dashes=self.TextLoad.partition_kwargs['dashes'],
            trailing_punctuation=self.TextLoad.partition_kwargs['trailing_punctuation'],
            lowercase=self.TextLoad.partition_kwargs['lowercase'],
            subtitle_lang=self.TextLoad.subtitle_lang,
        )
        self.load_model_kwargs: dict[str, Any] = dict(
            llm_model_repo=self.Model.llm_model_repo,
            llm_model_file=self.Model.llm_model_file,
            embed_model_repo=self.Model.embed_model_repo,
            n_gpu_layers=self.Model.llm_model_kwargs['n_gpu_layers'],
            n_ctx=self.Model.llm_model_kwargs['n_ctx'],
        )
        self.view_text_kwargs: dict[str, Any] = dict(
            max_lines_text_view=self.TextLoad.max_lines_text_view,
        )

    def get_sampling_kwargs(self) -> dict[str, Any]:
        return dict(
            temperature=self.generation_kwargs['temperature'],
            top_p=self.generation_kwargs['top_p'],
            top_k=self.generation_kwargs['top_k'],
            repeat_penalty=self.generation_kwargs['repeat_penalty'],
        )
    def get_rag_kwargs(self) -> dict[str, Any]:
        return dict(
            n_results=self.generation_kwargs['n_results'],
            max_distance_treshold=self.generation_kwargs['max_distance_treshold'],
            user_msg_with_context=self.generation_kwargs['user_msg_with_context'],
            context_template=self.generation_kwargs['context_template'],
        )
    def get_partition_kwargs(self) -> dict[str, Any]:
        return dict(
            chunking_strategy=self.load_text_kwargs['chunking_strategy'],
            max_characters=self.load_text_kwargs['max_characters'],
            new_after_n_chars=self.load_text_kwargs['new_after_n_chars'],
            overlap=self.load_text_kwargs['overlap'],
            clean=self.load_text_kwargs['clean'],
            bullets=self.load_text_kwargs['bullets'],
            extra_whitespace=self.load_text_kwargs['extra_whitespace'],
            dashes=self.load_text_kwargs['dashes'],
            trailing_punctuation=self.load_text_kwargs['trailing_punctuation'],
            lowercase=self.load_text_kwargs['lowercase'],
        )
    def get_clean_kwargs(self) -> dict[str, Any]:
        return dict(
            bullets=self.load_text_kwargs['bullets'],
            extra_whitespace=self.load_text_kwargs['extra_whitespace'],
            dashes=self.load_text_kwargs['dashes'],
            trailing_punctuation=self.load_text_kwargs['trailing_punctuation'],
            lowercase=self.load_text_kwargs['lowercase'],
        )
    def get_chunking_kwargs(self):
        return dict(
            max_characters=self.load_text_kwargs['max_characters'],
            new_after_n_chars=self.load_text_kwargs['new_after_n_chars'],
            overlap=self.load_text_kwargs['overlap'],
        )
    def get_embed_model_kwargs(self) -> dict[str, Any]:
        return self.Model.embed_model_kwargs

    def get_encode_kwargs(self) -> dict[str, Any]:
        return self.Inference.encode_kwargs

    def get_llm_model_kwargs(self) -> dict[str, Any]:
        return self.Model.llm_model_kwargs

    def get_query_kwargs(self) -> dict[str, Any]:
        return dict(
            n_results=self.generation_kwargs['n_results'],
            max_distance_treshold=self.generation_kwargs['max_distance_treshold'],
        )
