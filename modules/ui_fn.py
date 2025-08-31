import gc
from pathlib import Path
from shutil import rmtree
from typing import Iterator
from tqdm import tqdm

import requests
import psutil
import torch
import gradio as gr

from llama_cpp import Llama
from huggingface_hub import list_repo_tree, repo_exists

from config import Config, ModelStorage
from modules.db import ChromaDb, SentenceTransformerEF
from modules.text_load import TextLoader


CHAT_HISTORY = list[gr.ChatMessage | dict[str, str | gr.Component]]
CONF = Config()


class UiFnService:
    '''Service UI funcs'''
    @staticmethod
    def get_memory_usage() -> str:
        memory_for_printing = ''
        memory_type = 'Disk'
        psutil_stats = psutil.disk_usage('.')
        memory_total = psutil_stats.total / 1024**3
        memory_usage = psutil_stats.used / 1024**3
        memory_for_printing += f'{memory_type} Menory Usage: {memory_usage:.2f} / {memory_total:.2f} GB\n'
        memory_type = 'CPU'
        psutil_stats = psutil.virtual_memory()
        memory_total = psutil_stats.total / 1024**3
        memory_usage =  memory_total - (psutil_stats.available / 1024**3)
        memory_for_printing += f'{memory_type} Menory Usage: {memory_usage:.2f} / {memory_total:.2f} GB\n'
        if torch.cuda.is_available():
            memory_type = 'GPU'
            memory_free, memory_total = torch.cuda.mem_get_info()
            memory_usage = memory_total - memory_free
            memory_for_printing += f'{memory_type} Menory Usage: {memory_usage / 1024**3:.2f} / {memory_total:.2f} GB\n'
        memory_for_printing = f'---------------\n{memory_for_printing}---------------'
        return memory_for_printing


    @staticmethod
    def clear_memory() -> None:
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def cleanup_storage(request: gr.Request) -> None:
        for k, v in ModelStorage.__dict__.items():
            if not k.startswith('__'):
                dct = ModelStorage.__dict__[k]
                if isinstance(dct, dict):
                    if request.session_hash in dct:
                        del dct[request.session_hash]
        db = ChromaDb()
        db.delete_collection(collection_name=request.session_hash)


class UiFnModel:
    '''Models UI funcs'''
    @staticmethod
    def download_file(file_url: str, file_path: str | Path) -> None:
        response = requests.get(file_url, stream=True)
        if response.status_code != 200:
            raise Exception(f'The file is not available for download at the link: {file_url}')
        total_size = int(response.headers.get('content-length', 0))
        progress_tqdm = tqdm(desc='Loading GGUF file', total=total_size, unit='iB', unit_scale=True)
        progress_gradio = gr.Progress()
        completed_size = 0
        with open(file_path, 'wb') as file:
            for data in response.iter_content(chunk_size=4096):
                size = file.write(data)
                progress_tqdm.update(size)
                completed_size += size
                desc = f'Loading GGUF file, {completed_size/1024**3:.3f}/{total_size/1024**3:.3f} GB'
                progress_gradio(completed_size/total_size, desc=desc)


    @classmethod
    def load_llm_model(cls, config: Config, request: gr.Request) -> str:
        load_log = ''
        model_repo = config.load_model_kwargs['llm_model_repo']
        model_file = config.load_model_kwargs['llm_model_file']
        if isinstance(model_repo, (list, type(None))) or isinstance(model_file, (list, type(None))):
            load_log += 'No model repo or file selected\n'
            return load_log
        if '(' in config.load_model_kwargs['llm_model_file']:
            config.load_model_kwargs['llm_model_file'] = config.load_model_kwargs['llm_model_file'].split('(')[0].rstrip()
        progress = gr.Progress()
        progress(0.3, desc='Step 1/2: Download the GGUF file')
        model_path = config.Model.LLM_MODELS_PATH / model_file
        if not model_path.is_file():
            try:
                gguf_url = f'https://huggingface.co/{model_repo}/resolve/main/{model_file}'
                cls.download_file(gguf_url, model_path)
                load_log += f'Model {model_file} loaded\n'
            except Exception as ex:
                model_path = ''
                load_log += f'Error downloading model: {ex}\n'
                return load_log
        if model_path.is_file():
            load_log += f'Model {model_file} already loaded, initialization\n'
            progress(0.7, desc='Step 2/2: Initialize the model')
            try:
                llm_model = Llama(model_path=str(model_path), **config.get_llm_model_kwargs())
                ModelStorage.LLM_MODEL[request.session_hash] = llm_model
                load_log += f'Model {model_repo}/{model_file} initialized, max context size is {llm_model.n_ctx()} tokens\n'
            except Exception as ex:
                load_log += f'Error initializing LLM model on path: {model_path}: {ex}\n'
        else:
            load_log += f'Model {model_path} not is file\n'
        return load_log


    @staticmethod
    def load_embed_model(config: Config, request: gr.Request) -> str:
        embedding_function = None
        load_log = ''
        model_repo = config.load_model_kwargs['embed_model_repo']
        if isinstance(model_repo, (list, type(None))):
            load_log = 'No model repo selected\n'
            return load_log
        progress = gr.Progress()
        progress(0.5, desc='Downloading/initializing model')
        try:
            embedding_function = SentenceTransformerEF(
                model_name_or_path=model_repo,
                model_kwargs=config.get_embed_model_kwargs(),
                encode_kwargs=config.get_encode_kwargs(),
            )
            if embedding_function is not None:
                ModelStorage.EMBED_MODEL[request.session_hash] = embedding_function
                load_log += f'Embeddings model {model_repo} initialized\n'
        except Exception as ex:
            load_log += f'Error initializing Embedding model: {ex}\n'
        return load_log


    @staticmethod
    def add_new_model_repo(new_model_repo: str, model_repos: list[str]) -> tuple[dict, str]:
        if not new_model_repo.strip():
            return gr.skip(), 'Specify the model repository'
        load_log = ''
        repo = new_model_repo.strip().split('/tree/main')[0]
        if repo:
            repo = repo.split('/')[-2:]
            if len(repo) == 2:
                repo = '/'.join(repo).split('?')[0]
                if repo_exists(repo) and repo not in model_repos:
                    model_repos.insert(0, repo)
                    load_log += f'Model repository {repo} successfully added\n'
                else:
                    load_log += 'Invalid HF repository name or model already in the list\n'
            else:
                load_log += 'Invalid link to HF repository\n'
        else:
            load_log += 'Empty line in HF repository field\n'
        update_dropdown = gr.update(choices=model_repos, value=None)
        return update_dropdown, load_log


    @staticmethod
    def get_gguf_model_names(model_repo: str) -> dict:
        if not model_repo:
            return gr.skip()
        repo_files = list(list_repo_tree(model_repo))
        repo_files = [file for file in repo_files if file.path.endswith('.gguf')]
        model_paths_names = [f'{file.path} ({file.size / 1000 ** 3:.2f}G)' for file in repo_files]
        model_paths_values = [file.path for file in repo_files]
        model_paths_dropdown = list(zip(model_paths_names, model_paths_values))
        return gr.update(
            choices=model_paths_dropdown,
            value=model_paths_values[0],
        )


    @staticmethod
    def clear_llm_folder(except_gguf_filename: str | None) -> None:
        if not except_gguf_filename:
            gr.Info(f'The name of the model file that does not need to be deleted is not selected.')
            return
        if '(' in except_gguf_filename:
            except_gguf_filename = except_gguf_filename.split('(')[0].rstrip()
        for path in CONF.Model.LLM_MODELS_PATH.iterdir():
            if path.name == except_gguf_filename:
                continue
            if path.is_file():
                path.unlink(missing_ok=True)
                gr.Info(f'All files removed from directory {CONF.Model.LLM_MODELS_PATH} except {except_gguf_filename}')


    @staticmethod
    def clear_embed_folder(except_model_repo: str) -> None:
        if except_model_repo is None:
            gr.Info(f'The name of the model that does not need to be deleted is not selected.')
            return
        model_folder_name = except_model_repo.replace('/', '_')
        for path in CONF.Model.EMBED_MODELS_PATH.iterdir():
            if path.name == model_folder_name:
                continue
            if path.is_dir():
                rmtree(path, ignore_errors=True)
                gr.Info(f'All directories have been removed from the {CONF.Model.EMBED_MODELS_PATH} directory except {model_folder_name}')


    @staticmethod
    def check_support_system_role(model: Llama) -> bool:
        support_system_role = 'System role not supported' not in model.metadata['tokenizer.chat_template']
        return support_system_role



class UiFnDb:
    '''Load documents and create database UI funcs'''
    @staticmethod
    def load_texts_and_create_db(
            upload_files: list[str] | None,
            urls: str,
            config: Config,
            request: gr.Request,
    ) -> tuple[list[str], str]:
        load_log = ''
        texts = []
        embedding_function = ModelStorage.EMBED_MODEL.get(request.session_hash)
        if embedding_function is None:
            load_log += 'Embeddings model not initialized, DB cannot be created'
            return texts, load_log
        texts, log = TextLoader.load_texts_from_files_and_urls(
            upload_files=upload_files,
            urls=urls,
            config=config,
        )
        load_log += log
        if not texts:
            return texts, load_log
        progress = gr.Progress()
        progress(0.7, desc='Step 2/2: Initialize database')
        try:
            db = ChromaDb()
            db.create_collection_and_add_texts(
                collection_name=request.session_hash,
                texts=texts,
                embedding_function=embedding_function,
                create_collection_kwargs=config.Db.create_collection_kwargs,
            )
            load_log += 'DB is initialized, RAG mode is activated and can be activated in the Chatbot tab'
        except Exception as ex:
            load_log += f'Error creating database: {ex}'
        return texts, load_log


class UiFnChat:
    '''Chatbot UI funcs'''
    @staticmethod
    def user_message_to_chatbot(user_message: str, chatbot: CHAT_HISTORY) -> tuple[str, CHAT_HISTORY]:
        if len(chatbot) > 0 and chatbot[-1]['role'] == 'user':
            chatbot = chatbot[:-1]
        chatbot.append(dict(role='user', content=user_message))
        return '', chatbot


    @staticmethod
    def update_user_msg_with_context(
            chatbot: CHAT_HISTORY,
            config: Config,
            request: gr.Request,
    ) -> None:
        config.generation_kwargs['user_msg_with_context'] = ''
        user_message = chatbot[-1]['content']
        if not config.generation_kwargs['rag_mode'] or not user_message.strip():
            return
        context_template = config.generation_kwargs['context_template']
        if '{user_message}' not in context_template and '{context}' not in context_template:
            gr.Info('Context template must include {user_message} and {context}')
            return
        embedding_function = ModelStorage.EMBED_MODEL.get(request.session_hash)
        if embedding_function is None:
            gr.Info('Embedding model not initialized')
            return user_msg_with_context
        db = ChromaDb()
        collection = db.get_collection(
            collection_name=request.session_hash,
            embedding_function=embedding_function,
        )
        if collection is None:
            gr.Info(f'Collection {request.session_hash} not exists')
            return
        texts_and_dists = db.search_similar_texts(
            collection=collection,
            query_text=user_message,
            query_kwargs=config.get_query_kwargs(),
        )
        if len(texts_and_dists.get('documents', [])) > 0:
            retriever_context = '\n\n'.join(texts_and_dists['documents'])
            user_msg_with_context = context_template.format(
                user_message=user_message,
                context=retriever_context,
            )
            config.generation_kwargs['user_msg_with_context'] = user_msg_with_context


    @staticmethod
    def _stream_llm_response_to_chatbot(
            llm_model: Llama,
            messages: CHAT_HISTORY,
            chatbot: CHAT_HISTORY,
            sampling_kwargs: dict[str, int | float],
            show_thinking: bool,
    ) -> Iterator[CHAT_HISTORY]:
        stream_response = llm_model.create_chat_completion(
            messages=messages,
            stream=True,
            **sampling_kwargs,
        )
        is_think = False
        for chunk in stream_response:
            token = chunk['choices'][0]['delta'].get('content')
            if token is None:
                continue
            if show_thinking:
                if token == '<think>':
                    gr.Info('Thinking...')
                chatbot[-1]['content'] += token
                yield chatbot
                continue
            if token == '<think>':
                is_think = True
                gr.Info('Thinking...')
                chatbot[-1]['content'] = 'Thinking...'
            elif token == '</think>':
                is_think = False
                chatbot[-1]['content'] = ''
                continue
            if not is_think:
                chatbot[-1]['content'] += token
            yield chatbot


    @staticmethod
    def _prepare_messages(
            system_prompt: str,
            support_system_role: bool,
            history_len: int,
            user_message: str,
            chatbot: CHAT_HISTORY,
    ):
        messages = []
        if support_system_role and system_prompt:
            messages.append(dict(role='system', content=system_prompt))
        if history_len != 0:
            messages.extend(chatbot[:-1][-(history_len*2):])
        messages.append(dict(role='user', content=user_message))
        return messages


    @classmethod
    def yield_chatbot_with_llm_response(
            cls,
            chatbot: CHAT_HISTORY,
            config: Config,
            request: gr.Request,
    ) -> Iterator[CHAT_HISTORY]:
        user_message = chatbot[-1]['content']
        if not user_message.strip():
            yield chatbot[:-1]
            return
        llm_model = ModelStorage.LLM_MODEL.get(request.session_hash)
        if llm_model is None:
            gr.Info('LLM model not initialized')
            yield chatbot[:-1]
            return
        sampling_kwargs = config.get_sampling_kwargs()
        if not config.generation_kwargs['do_sample']:
            sampling_kwargs['top_p'] = 0.0
            sampling_kwargs['top_k'] = 1
            sampling_kwargs['repeat_penalty'] = 1.0
        if config.generation_kwargs['rag_mode']:
            if config.generation_kwargs['user_msg_with_context']:
                user_message = config.generation_kwargs['user_msg_with_context']
            else:
                gr.Info((
                    'No documents relevant to the query were found, generation in RAG mode is not possible.\n'
                    'Or Context template is specified incorrectly.\n'
                    'Try reducing `max_distance_treshold` or disable RAG mode for normal generation'
                    ))
                yield chatbot[:-1]
                return
        support_system_role = UiFnModel.check_support_system_role(llm_model)
        messages = cls._prepare_messages(
            system_prompt=config.generation_kwargs['system_prompt'],
            support_system_role=support_system_role,
            history_len=config.generation_kwargs['history_len'],
            user_message=user_message,
            chatbot=chatbot,
        )
        chatbot.append(dict(role='assistant', content=''))
        try:
            yield from cls._stream_llm_response_to_chatbot(
                llm_model=llm_model,
                messages=messages,
                chatbot=chatbot,
                sampling_kwargs=sampling_kwargs,
                show_thinking=config.Inference.show_thinking,
            )
        except Exception as ex:
            gr.Info(f'Error generating LLM response: {ex}')
            yield chatbot[:-2]
            return