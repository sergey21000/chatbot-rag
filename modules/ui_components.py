from typing import Any
import gradio as gr
from loguru import logger

from config import Config
from modules.db import ChromaDb
from modules.ui_fn import UiFnModel


CONF = Config()


class UiUpdateComponent:
    '''Functionality for updating gradio components'''
    @staticmethod
    def update_visibility(visible: bool, num_componets: int)  -> dict | list[dict]:
        if num_componets == 1:
            return gr.update(visible=visible)
        logger.debug(f'num conponents to update: {num_componets}, visible: {visible}')
        return [gr.update(visible=visible) for _ in range(num_componets)]


    @staticmethod
    def view_user_msg_with_context(config: Config) -> dict:
        text = config.generation_kwargs['user_msg_with_context']
        visible = config.generation_kwargs['rag_mode']
        return gr.update(value=text, visible=visible)


    @staticmethod
    def update_system_prompt() -> dict:
        support_system_role = UiFnModel.check_support_system_role()
        if support_system_role:
            value = ''
        else:
            value = 'System prompt is not supported by this model'
        return gr.update(value=value, interactive=support_system_role)


    @staticmethod
    def view_texts(texts: list[str], max_lines: int) -> dict:
        if not texts:
            return gr.update(value='No texts loaded')
        text = f'\n'.join(texts)
        num_lines = len(text.split('\n'))
        if max_lines < 0:
            max_lines = num_lines
        sep = f"\n{'=' * 20}\n"
        text = f'{sep}'.join(text.split('\n')[:max_lines])
        text = f'Всего строк текста: {num_lines}\n{text}'
        return gr.update(value=text) 


    @staticmethod
    def update_rag_mode_if_db_exists(request: gr.Request) -> dict:
        db = ChromaDb()
        exists = db.collection_exists(collection_name=request.session_hash)
        logger.debug(f'is collection exists: {exists}')
        if not exists:
            return gr.update(value=False, visible=False)
        return gr.update(value=True, visible=True)


    @staticmethod
    def update_kwargs(
        config_kwargs: dict,
        matching_kwargs: dict | list,
        args: tuple[Any],
    ) -> None:
        kwargs = dict(zip(matching_kwargs, args))
        config_kwargs.update(kwargs)


class UiBase:
    '''Base class for gradio UI components'''
    def get_matching_kwargs(self, kwargs_for_matching: dict):
        return {k: self.__dict__[k] for k in kwargs_for_matching if k in self.__dict__}

    def get_matching_args(self, kwargs_for_matching: dict):
        return list(self.get_matching_kwargs(kwargs_for_matching).values())


class UiChatbot(UiBase):
    '''Gradio UI components for Chatbot Tab'''
    def __init__(self):
        self.chatbot = gr.Chatbot(
            sanitize_html=False,
            render=False,
            # buttons=['copy', 'copy_all', 'share'],
            # height=480,
            # api_visibility='public',  # 'private'
        )
        self.user_msg = gr.MultimodalTextbox(
            interactive=True,
            file_count='single',
            placeholder='Enter a message or attach a file',
            show_label=False,
            sources=['upload'],  # ['upload', 'microphone']
            render=False,
        )
        self.user_msg_btn = gr.Button('Send', render=False)
        self.stop_btn = gr.Button('Stop', render=False)
        self.clear_btn = gr.Button('Clear', render=False)
        self.system_prompt = gr.Textbox(
            value=CONF.Prompt.system_prompt,
            label='Edit System prompt',
            interactive=False,
            render=False,
        )
        self.history_len = gr.Slider(
            minimum=0,
            maximum=10,
            value=CONF.generation_kwargs['history_len'],
            step=1,
            info='Number of previous user-bot message pairs to keep in chat history',
            label='history len',
            show_label=False,
            render=False,
        )
        self.do_sample = gr.Checkbox(
            value=CONF.generation_kwargs['do_sample'],
            label='do_sample',
            info='Activate random sampling',
            render=False,
        )
        self.temperature = gr.Slider(
            minimum=0.1,
            maximum=3,
            value=CONF.generation_kwargs['temperature'],
            step=0.1,
            label='temperature',
            visible=CONF.generation_kwargs['do_sample'],
            render=False,
        )
        self.top_p = gr.Slider(
            minimum=0.1,
            maximum=1,
            value=CONF.generation_kwargs['top_p'],
            step=0.01,
            label='top_p',
            visible=CONF.generation_kwargs['do_sample'],
            render=False,
        )
        self.top_k = gr.Slider(
            minimum=1,
            maximum=50,
            value=CONF.generation_kwargs['top_k'],
            step=1,
            label='top_k',
            visible=CONF.generation_kwargs['do_sample'],
            render=False,
        )
        self.repeat_penalty = gr.Slider(
            minimum=1,
            maximum=5,
            value=CONF.generation_kwargs['repeat_penalty'],
            step=0.1,
            label='repeat_penalty',
            visible=CONF.generation_kwargs['do_sample'],
            render=False,
        )
        self.rag_mode = gr.Checkbox(
            value=False,
            label='RAG Mode',
            scale=1,
            visible=False,
            render=False,
        )
        self.n_results = gr.Radio(
            choices=[1, 2, 3, 4, 5, 'max', 'all'],  # ToDo: replace with Slider
            value=CONF.generation_kwargs['n_results'],
            label='Number of relevant texts for vector search',
            visible=self.rag_mode.value,
            render=False,
        )
        self.max_distance_treshold = gr.Slider(
            minimum=0,
            maximum=2,
            value=CONF.generation_kwargs['max_distance_treshold'],
            step=0.01,
            label='max_distance_treshold',
            visible=self.rag_mode.value,
            render=False,
        )
        self.context_template = gr.Textbox(
            value=CONF.generation_kwargs['context_template'],
            label='Edit Context Template',
            lines=len(CONF.generation_kwargs['context_template'].split('\n')),
            visible=False,
            render=False,
        )
        self.user_msg_with_context = gr.Textbox(
            value='',
            label='User Message With Context',
            interactive=False,
            visible=False,
            render=False,
        )
        self.enable_thinking = gr.Checkbox(
            value=False,
            label='Enable thinking',
            show_label=False,
            visible=True,
            render=False,
        )
        self.show_thinking = gr.Checkbox(
            value=False,
            label='Show thinking',
            show_label=False,
            visible=True,
            render=False,
        )


class UiLoadTexts(UiBase):
    '''Gradio UI components for Load Texts Tab'''
    def __init__(self):
        # загрузка файлов
        self.upload_files = gr.File(
            file_count='multiple',
            label=f'Loading text files ({CONF.TextLoad.SUPPORTED_FILE_EXTS})',
            render=False,
            # height=110,
        )
        self.urls = gr.Textbox(
            label='Links to Web sites or YouTube',
            lines=4,
            render=False,
        )
        self.load_texts_btn = gr.Button('Upload texts and initialize database', render=False)
        self.load_texts_log = gr.Textbox(
            label='Status of loading and splitting texts',
            interactive=False,
            render=False,
        )
        self.chunking_strategy = gr.Radio(
            choices=['basic', 'by_title'],
            value='basic',
            label='Chunking strategy',
            scale=1,
            render=False,
        )
        self.max_characters = gr.Slider(
            minimum=50,
            maximum=2000,
            value=CONF.load_text_kwargs['max_characters'],
            step=50,
            label='Max characters',
            scale=1,
            render=False,
        )
        self.new_after_n_chars = gr.Slider(
            minimum=50,
            maximum=2000,
            value=CONF.load_text_kwargs['new_after_n_chars'],
            step=50,
            label='New after n chars',
            scale=1,
            render=False,
        )
        self.overlap = gr.Slider(
            minimum=0,
            maximum=200,
            value=CONF.load_text_kwargs['overlap'],
            step=10,
            label='Chunk overlap',
            scale=1,
            render=False,
        )
        self.clean = gr.Checkbox(
            value=CONF.load_text_kwargs['clean'],
            label='clean',
            info='Additionally clear text',
            scale=0,
            render=False,
        )
        self.bullets = gr.Checkbox(
            value=CONF.load_text_kwargs['bullets'],
            label='bullets',
            info='Remove bullets (in beginning of text)',
            visible=CONF.load_text_kwargs['clean'],
            render=False,
        )
        self.extra_whitespace = gr.Checkbox(
            value=CONF.load_text_kwargs['extra_whitespace'],
            label='extra_whitespace',
            info='Remove spaces',
            visible=CONF.load_text_kwargs['clean'],
            render=False,
        )
        self.dashes = gr.Checkbox(
            value=CONF.load_text_kwargs['dashes'],
            label='dashes',
            info='Remove dashes',
            visible=CONF.load_text_kwargs['clean'],
            render=False,
        )
        self.trailing_punctuation = gr.Checkbox(
            value=CONF.load_text_kwargs['trailing_punctuation'],
            label='trailing_punctuation',
            info='Remove trailing punctuation',
            visible=CONF.load_text_kwargs['clean'],
            render=False,
        )
        self.lowercase = gr.Checkbox(
            value=CONF.load_text_kwargs['lowercase'],
            label='lowercase',
            info='Сonvert to lowercase',
            visible=CONF.load_text_kwargs['clean'],
            render=False,
        )
        self.subtitle_lang = gr.Radio(
            choices=CONF.TextLoad.SUBTITLE_LANGS,
            value=CONF.TextLoad.subtitle_lang,
            label='YouTube subtitle language',
            render=False,
        )


class UiViewTexts(UiBase):
    '''Gradio UI components for View Texts Tab'''
    def __init__(self):
        self.view_texts_btn = gr.Button('Show downloaded text chunks', render=False)
        self.view_texts_textbox = gr.Textbox(
            value='',
            label='Uploaded chunks',
            placeholder='To view chunks, load documents in the Load texts tab',
            lines=10,
            render=False,
        )
        self.max_lines_text_view = gr.Slider(
            minimum=-1,
            maximum=499,
            value=CONF.view_text_kwargs['max_lines_text_view'],
            step=50,
            label='Max lines',
            info='Max number of lines of text to display',
            scale=0,
            show_label=False,
            render=False,
        )


class UiLoadModel(UiBase):
    '''Gradio UI components for Load Models Tab'''
    def __init__(self):
        self.new_llm_model_repo = gr.Textbox(
            value='',
            label='Add repository',
            placeholder='Link to repository of HF models in GGUF format',
            render=False,
        )
        self.new_llm_model_repo_btn = gr.Button('Add LLM repository', render=False)
        self.llm_model_repo = gr.Dropdown(
            choices=CONF.Repos.llm_model_repos,
            value=None,
            label='HF Model Repository',
            render=False,
        )
        self.llm_model_file = gr.Dropdown(
            choices=None,
            value=None,
            label='GGUF model file',
            allow_custom_value=True,
            render=False,
        )
        self.llm_model_mmproj = gr.Dropdown(
            choices=None,
            value=None,
            label='MMPROJ model file',
            allow_custom_value=True,
            render=False,
        )
        self.load_llm_model_btn = gr.Button('Loading and initializing LLM model', render=False)
        self.load_llm_model_log = gr.Textbox(
            value=None,
            label='LLM model loading status',
            interactive=False,
            lines=6,
            max_lines=6,
            render=False,
        )
        self.clear_llm_folder_btn = gr.Button('Clear folder', render=False)
        self.n_gpu_layers = gr.Number(
            value=CONF.load_model_kwargs['n_gpu_layers'],
            label='n_gpu_layers',
            info='Number of layers to offload to GPU',
            show_label=True,
            render=False,
        )
        self.n_ctx = gr.Number(
            value=CONF.load_model_kwargs['n_ctx'],
            label='n_ctx',
            info='Model context size, 0 = from model',
            show_label=True,
            render=False,
        )
        self.new_embed_model_repo = gr.Textbox(
            value='',
            label='Add repository',
            placeholder='Link to HF model repository',
            render=False,
        )
        self.new_embed_model_repo_btn = gr.Button('Add Embed repository', render=False)
        self.embed_model_repo = gr.Dropdown(
            choices=CONF.Repos.embed_model_repos,
            value=None,
            label='HF Embed model repository',
            render=False,
        )
        self.load_embed_model_btn = gr.Button('Loading and initializing Embed model', render=False)
        self.load_embed_model_log = gr.Textbox(
            value=None,
            label='Embed model loading status',
            interactive=False,
            lines=7,
            render=False,
        )
        self.clear_embed_folder_btn = gr.Button('Clear folder', render=False)
