import os
import sys
from pathlib import Path
from typing import Any, ClassVar

from chromadb import EmbeddingFunction


class ModelStorage:
    '''Global model storage'''
    EMBED_MODEL: ClassVar[dict[str, EmbeddingFunction]] = {}


class UiGradioConfig:
    '''Gradio settings for gr.Blocks()'''
    css: str | None = '''
    .gradio-container {
        width: 70% !important;
        margin: 0 auto !important;
    }
    '''
    if hasattr(sys, 'getandroidapilevel') or 'ANDROID_ROOT' in os.environ:
        css = None
    theme: str | None = None
    fill_height: bool = False
    footer_links: list[str] = ['gradio', 'settings']
    delete_cache: tuple[int, int] | None = None

    @classmethod
    def get_demo_launch_kwargs(cls):
        return dict(
            css=cls.css,
            theme=cls.theme,
            footer_links=cls.footer_links,
        )

    @classmethod
    def get_demo_blocks_kwargs(cls):
        return dict(
            fill_height=cls.fill_height,
            delete_cache=cls.delete_cache,
        )


class InferenceConfig:
    '''Model inference settings'''
    def __init__(self):
        self.encode_kwargs: dict[str, Any] = dict(
            batch_size=300,
            normalize_embeddings=None,
        )
        self.reasoning_format = 'none'
        self.sampling_kwargs: dict[str, Any] = dict(
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1,
        )
        self.max_tokens = -1
        self.do_sample: bool = False 
        self.rag_mode: bool = False
        self.history_len: int = 0
        self.enable_thinking = False
        self.show_thinking: bool = False
        self.resize_size = 256


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
{user_message}'''


class ModelConfig:
    '''Configuration of paths, models and generation parameters'''
    def __init__(self):
        self.LLM_MODEL_DIR: Path = Path(os.getenv('LLAMA_CACHE', 'llm_models'))
        self.EMBED_MODEL_DIR: Path = Path(os.getenv('EMBED_MODEL_DIR', 'embed_models'))
        self.LLM_MODEL_DIR.mkdir(exist_ok=True)
        self.EMBED_MODEL_DIR.mkdir(exist_ok=True)

        self.llm_model_repo: str = None
        self.llm_model_file: str = None
        self.llm_model_mmproj: str = None

        self.embed_model_repo: str = os.getenv('EMBED_MODEL_REPO', 'Alibaba-NLP/gte-multilingual-base')
        self.embed_model_kwargs: dict[str, Any] = dict(
            cache_folder=self.EMBED_MODEL_DIR,
            trust_remote_code=True,
            token=os.getenv('HF_TOKEN'),
            model_kwargs=dict(
                dtype='auto',
                device_map='auto',
            )
        )
        self.llm_model_kwargs: dict[str, Any] = dict(
            n_gpu_layers=-1,
            n_ctx=4096,
        )


class ReposConfig:
    '''Links to repositories with ggu models'''
    def __init__(self):
        self.llm_model_repos: list[str] = [
            'bartowski/google_gemma-3-1b-it-GGUF',
            'bartowski/google_gemma-3-4b-it-GGUF',
            'bartowski/Qwen_Qwen3-1.7B-GGUF',
            'bartowski/Qwen_Qwen3-4B-GGUF',
            'bartowski/Qwen_Qwen3-0.6B-GGUF',
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
        self.CHATBOT_RAG_ENABLED = os.getenv('CHATBOT_RAG_ENABLED', '1').lower() in ('1', 'true')

        self.generation_kwargs: dict[str, Any] = dict(
            do_sample=self.Inference.do_sample,
            temperature=self.Inference.sampling_kwargs['temperature'],
            top_p=self.Inference.sampling_kwargs['top_p'],
            top_k=self.Inference.sampling_kwargs['top_k'],
            repeat_penalty=self.Inference.sampling_kwargs['repeat_penalty'],
            max_tokens=self.Inference.max_tokens,
            history_len=self.Inference.history_len,
            system_prompt=self.Prompt.system_prompt,
            context_template=self.Prompt.context_template,
            enable_thinking=self.Inference.enable_thinking,
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
            llm_model_mmproj=self.Model.llm_model_mmproj,
            embed_model_repo=self.Model.embed_model_repo,
            n_gpu_layers=self.Model.llm_model_kwargs['n_gpu_layers'],
            n_ctx=self.Model.llm_model_kwargs['n_ctx'],
        )
        self.view_text_kwargs: dict[str, Any] = dict(
            max_lines_text_view=self.TextLoad.max_lines_text_view,
        )

    def update_env(self, **kwargs) -> None:
        LLAMA_ARG_MMPROJ=self.load_model_kwargs['llm_model_mmproj']
        dict_to_updating = dict(
            LLAMA_ARG_CTX_SIZE=str(self.load_model_kwargs['n_ctx']),
            LLAMA_ARG_N_GPU_LAYERS=str(self.load_model_kwargs['n_gpu_layers']),
        )
        for k, v in kwargs.items():
            dict_to_updating[k] = str(v)
        os.environ.pop('LLAMA_ARG_MODEL_URL', None)
        os.environ.pop('LLAMA_ARG_MMPROJ_URL', None)
        os.environ.pop('LLAMA_ARG_HF_REPO', None)
        os.environ.pop('LLAMA_ARG_HF_FILE', None)
        os.environ.pop('LLAMA_ARG_MMPROJ', None)
        if LLAMA_ARG_MMPROJ:
            dict_to_updating['LLAMA_ARG_MMPROJ'] = str(self.Model.LLM_MODEL_DIR / LLAMA_ARG_MMPROJ)
        os.environ.update(dict_to_updating)

    def get_completions_kwargs(self) -> dict[str, Any]:
        return dict(
            temperature=self.generation_kwargs['temperature'],
            top_p=self.generation_kwargs['top_p'],
            max_tokens=self.generation_kwargs['max_tokens'],
            extra_body=dict(
                top_k=self.generation_kwargs['top_k'],
                repeat_penalty=self.generation_kwargs['repeat_penalty'],
                reasoning_format=self.Inference.reasoning_format,
                chat_template_kwargs=dict(
                    enable_thinking=self.generation_kwargs['enable_thinking'],
                ),
            ),
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
