import gc
import csv
from pathlib import Path
from dataclasses import dataclass, field
from shutil import rmtree
from textwrap import dedent
from typing import Any, Iterable, Iterator
from tqdm import tqdm

import psutil
import requests
from requests.exceptions import MissingSchema

import torch
import gradio as gr
from llama_cpp import Llama

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from huggingface_hub import (
    hf_hub_download,
    list_repo_tree,
    list_repo_files,
    repo_info,
    repo_exists,
    snapshot_download,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from config import ModelConfig, SettingsConfig, PromptConfig


CHAT_HISTORY = list[gr.ChatMessage | dict[str, str | gr.Component]]

class UiFnService:
    '''Service UI funcs'''
    @staticmethod
    def get_memory_usage() -> str:
        print_memory = ''
        memory_type = 'Disk'
        psutil_stats = psutil.disk_usage('.')
        memory_total = psutil_stats.total / 1024**3
        memory_usage = psutil_stats.used / 1024**3
        print_memory += f'{memory_type} Menory Usage: {memory_usage:.2f} / {memory_total:.2f} GB\n'
        memory_type = 'CPU'
        psutil_stats = psutil.virtual_memory()
        memory_total = psutil_stats.total / 1024**3
        memory_usage =  memory_total - (psutil_stats.available / 1024**3)
        print_memory += f'{memory_type} Menory Usage: {memory_usage:.2f} / {memory_total:.2f} GB\n'
        if torch.cuda.is_available():
            memory_type = 'GPU'
            memory_free, memory_total = torch.cuda.mem_get_info()
            memory_usage = memory_total - memory_free
            print_memory += f'{memory_type} Menory Usage: {memory_usage / 1024**3:.2f} / {memory_total:.2f} GB\n'
        print_memory = f'---------------\n{print_memory}---------------'
        return print_memory


    @staticmethod
    def clear_memory() -> None:
        gc.collect()
        torch.cuda.empty_cache()


    @staticmethod
    def check_subtitles_available(yt_video_link: str, target_lang: str) -> tuple[bool, str]:
        load_log = ''
        available = True
        video_id = yt_video_link.split('watch?v=')[-1].split('&')[0].split('live/')[-1].split('&')[0]
        if len(video_id) != 11:
            load_log += f'Invalid video url ({yt_video_link}): Video ID ({video_id}) must be 11 characters long\n'
            available = False
            return available, load_log
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            try:
                transcript = transcript_list.find_transcript([target_lang])
                if transcript.is_generated:
                    load_log += f'Automatic subtitles will be loaded, manual subtitles are not available for video {yt_video_link}\n'
                else:
                    load_log += f'Manual subtitles will be downloaded for the video {yt_video_link}\n'
            except NoTranscriptFound:
                load_log += f'Subtitle language {target_lang} is not available for video {yt_video_link}\n'
                available = False
        except TranscriptsDisabled:
            load_log += f'Invalid video url ({yt_video_link}) or current server IP is blocked for YouTube\n'
            available = False
        return available, load_log


    @staticmethod
    def view_documents(documents: list[Document]) -> str:
        sep = '=' * 20
        return f'\n{sep}\n\n'.join([doc.page_content for doc in documents])


class UiFnModel:
    '''Models UI funcs'''
    @staticmethod
    def cleanup_models(request: gr.Request) -> None:
        if request.session_hash in ModelConfig.LLM_MODELS:
            del ModelConfig.LLM_MODELS[request.session_hash]
        if request.session_hash in ModelConfig.EMBED_MODELS:
            del ModelConfig.EMBED_MODELS[request.session_hash]


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
    def load_llm_model(cls, model_repo: str | None, model_file: str | None, request: gr.Request) -> tuple[bool, str]:
        llm_model = None
        load_log = ''
        support_system_role = False
        if isinstance(model_repo, (list, type(None))) or isinstance(model_file, (list, type(None))):
            load_log += 'No model repo or file selected\n'
            return support_system_role, load_log
        if '(' in model_file:
            model_file = model_file.split('(')[0].rstrip()
        progress = gr.Progress()
        progress(0.3, desc='Step 1/2: Download the GGUF file')
        model_path = ModelConfig.LLM_MODELS_PATH / model_file
        if not model_path.is_file():
            try:
                gguf_url = f'https://huggingface.co/{model_repo}/resolve/main/{model_file}'
                cls.download_file(gguf_url, model_path)
                load_log += f'Model {model_file} loaded\n'
            except Exception as ex:
                model_path = ''
                load_log += f'Error downloading model, error code:\n{ex}\n'
                return support_system_role, load_log
        if model_path.is_file():
            load_log += f'Model {model_file} already loaded, reinitializing\n'
            progress(0.7, desc='Step 2/2: Initialize the model')
            try:
                llm_model = Llama(model_path=str(model_path), **ModelConfig.LLAMA_MODEL_KWARGS)
                ModelConfig.LLM_MODELS[request.session_hash] = llm_model
                support_system_role = 'System role not supported' not in llm_model.metadata['tokenizer.chat_template']
                load_log += f'Model {model_file} initialized, max context size is {llm_model.n_ctx()} tokens\n'
            except Exception as ex:
                load_log += f'Error initializing LLM model on path: {model_path}, error code:\n{ex}\n'
                return support_system_role, load_log
        else:
            load_log += f'Model {model_path} not is file\n'
        return support_system_role, load_log


    @staticmethod
    def load_embed_model(model_repo: str | None, request: gr.Request) -> str:
        embed_model = None
        load_log = ''
        if isinstance(model_repo, (list, type(None))):
            load_log = 'No model repo selected\n'
            return load_log
        progress = gr.Progress()
        folder_name = model_repo.replace('/', '_')
        folder_path = ModelConfig.EMBED_MODELS_PATH / folder_name
        if not Path(folder_path).is_dir():
            progress(0.5, desc='Step 1/2: Download model repository')
            snapshot_download(
                repo_id=model_repo,
                local_dir=folder_path,
                ignore_patterns='*.h5',
            )
            load_log += f'Model {model_repo} loaded\n'
        else:
            load_log += f'Reinitializing model {model_repo} \n'
        progress(0.7, desc='Step 2/2: Initialize the model')
        try:
            device = 'cuda' if torch.cuda.is_available() and ModelConfig.EMBED_MODEL_USE_CUDA_IF_AVAILABLE else 'cpu'
            embed_model = HuggingFaceEmbeddings(
                model_name=str(folder_path),
                model_kwargs={'device': device},
                # encode_kwargs={'normalize_embeddings': True},
                )
            load_log += f'Embeddings model {model_repo} initialized\n'
            ModelConfig.EMBED_MODELS[request.session_hash] = embed_model
        except Exception as ex:
            load_log += f'Error initializing Embedding model on path: {folder_path}, error code:\n{ex}\n'
        return load_log


    @staticmethod
    def add_new_model_repo(new_model_repo: str, model_repos: list[str]) -> tuple[gr.Dropdown, str]:
        if not new_model_repo.strip():
            return gr.skip(), 'Specify the model repository'
        load_log = ''
        repo = new_model_repo.strip()
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
        model_repo_dropdown = gr.Dropdown(choices=model_repos, value=model_repos[0])
        return model_repo_dropdown, load_log


    @staticmethod
    def get_gguf_model_names(model_repo: str) -> gr.Dropdown:
        if not model_repo:
            return gr.skip()
        repo_files = list(list_repo_tree(model_repo))
        repo_files = [file for file in repo_files if file.path.endswith('.gguf')]
        model_paths_names = [f'{file.path} ({file.size / 1000 ** 3:.2f}G)' for file in repo_files]
        model_paths_values = [file.path for file in repo_files]
        model_paths_dropdown = list(zip(model_paths_names, model_paths_values))
        model_paths_component = gr.Dropdown(
            choices=model_paths_dropdown,
            value=model_paths_values[0],
            label='GGUF model file',
            )
        return model_paths_component


    @staticmethod
    def clear_llm_folder(gguf_filename: str | None) -> None:
        if not gguf_filename:
            gr.Info(f'The name of the model file that does not need to be deleted is not selected.')
            return
        if '(' in gguf_filename:
            gguf_filename = gguf_filename.split('(')[0].rstrip()
        for path in ModelConfig.LLM_MODELS_PATH.iterdir():
            if path.name == gguf_filename:
                continue
            if path.is_file():
                path.unlink(missing_ok=True)
        gr.Info(f'All files removed from directory {ModelConfig.LLM_MODELS_PATH} except {gguf_filename}')


    @staticmethod
    def clear_embed_folder(model_repo: str) -> None:
        if model_repo is None:
            gr.Info(f'The name of the model that does not need to be deleted is not selected.')
            return
        model_folder_name = model_repo.replace('/', '_')
        for path in ModelConfig.EMBED_MODELS_PATH.iterdir():
            if path.name == model_folder_name:
                continue
            if path.is_dir():
                rmtree(path, ignore_errors=True)
        gr.Info(f'All directories have been removed from the {ModelConfig.EMBED_MODELS_PATH} directory except {model_folder_name}')


class UiFnDb:
    '''Rag UI funcs'''
    @staticmethod
    def load_documents_from_files(upload_files: list[str]) -> tuple[list[Document], str]:
        load_log = ''
        documents = []
        for upload_file in upload_files:
            file_extension = f".{upload_file.split('.')[-1]}"
            if file_extension in SettingsConfig.LOADER_CLASSES:
                loader_class = SettingsConfig.LOADER_CLASSES[file_extension]
                loader_kwargs = {}
                if file_extension == '.csv':
                    with open(upload_file) as csvfile:
                        delimiter = csv.Sniffer().sniff(csvfile.read(4096)).delimiter
                    loader_kwargs = {'csv_args': {'delimiter': delimiter}}
                try:
                    loaded_documents = loader_class(upload_file, **loader_kwargs).load()
                    documents.extend(loaded_documents)
                except Exception as ex:
                    load_log += f'Error uploading file {upload_file}\n'
                    load_log += f'Error code: {ex}\n'
                    continue
            else:
                load_log += f'Unsupported file format {upload_file}\n'
                continue
        return documents, load_log


    @staticmethod
    def load_documents_from_links(
            web_links: str,
            subtitles_lang: str,
    ) -> tuple[list[Document], str]:
        load_log = ''
        documents = []
        web_links = [web_link.strip() for web_link in web_links.split() if web_link.strip()]
        for web_link in web_links:
            loader_class_kwargs = {}
            if 'youtube.com' in web_link:
                available, log = UiFnService.check_subtitles_available(web_link, subtitles_lang)
                load_log += log
                if not available:
                    continue
                loader_class = SettingsConfig.LOADER_CLASSES['youtube'].from_youtube_url
                loader_class_kwargs = {'language': subtitles_lang}
            else:
                loader_class = SettingsConfig.LOADER_CLASSES['web']
            try:
                if requests.get(web_link).status_code != 200:
                    load_log += f'Link not available for Python requests: {web_link}\n'
                    continue
                load_documents = loader_class(web_link, **loader_class_kwargs).load()
                if len(load_documents) == 0:
                    load_log += f'No text chunks were found at the link: {web_link}\n'
                    continue
                documents.extend(load_documents)
            except MissingSchema:
                load_log += f'Invalid link: {web_link}\n'
                continue
            except Exception as ex:
                load_log += f'Error loading data by web loader at link: {web_link}\n'
                load_log += f'Error code: {ex}\n'
                continue
        return documents, load_log


    @classmethod
    def load_documents_and_create_db(
            cls,
            upload_files: list[str] | None,
            web_links: str,
            subtitles_lang: str,
            chunk_size: int,
            chunk_overlap: int,
            request: gr.Request,
    ) -> tuple[list[Document], VectorStore | None, str]:
        load_log = ''
        all_documents = []
        db = None
        progress = gr.Progress()
        embed_model = ModelConfig.EMBED_MODELS.get(request.session_hash)
        if embed_model is None:
            load_log += 'Embeddings model not initialized, DB cannot be created'
            return all_documents, db, load_log
        if upload_files is None and not web_links:
            load_log = 'No files or links selected'
            return all_documents, db, load_log
        if upload_files is not None:
            progress(0.3, desc='Step 1/2: Upload documents from files')
            docs, log = cls.load_documents_from_files(upload_files)
            all_documents.extend(docs)
            load_log += log
        if web_links:
            progress(0.3 if upload_files is None else 0.5, desc='Step 1/2: Upload documents via links')
            docs, log = cls.load_documents_from_links(web_links, subtitles_lang)
            all_documents.extend(docs)
            load_log += log
        if len(all_documents) == 0:
            load_log += 'Download was interrupted because no documents were extracted\n'
            load_log += 'RAG mode cannot be activated'
            return all_documents, db, load_log
        load_log += f'Documents loaded: {len(all_documents)}\n'
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # distance_strategy=DistanceStrategy.COSINE,
        )
        documents = text_splitter.split_documents(all_documents)
        documents = cls.clear_documents(documents)
        load_log += f'Documents are divided, number of text chunks: {len(documents)}\n'
        progress(0.7, desc='Step 2/2: Initialize DB')
        db = FAISS.from_documents(documents=documents, embedding=embed_model)
        load_log += 'DB is initialized, RAG mode is activated and can be activated in the Chatbot tab'
        return documents, db, load_log


    @staticmethod
    def clear_documents(documents: list[Document]) -> list[Document]:
        def clear_text(text: str) -> str:
            lines = text.split('\n')
            lines = [line for line in lines if len(line.strip()) > 2]
            text = '\n'.join(lines).strip()
            return text
        output_documents = []
        for document in documents:
            text = clear_text(document.page_content)
            if len(text) > 10:
                document.page_content = text
                output_documents.append(document)
        return output_documents


class UiFnChat:
    '''Chatbot UI funcs'''
    @staticmethod
    def user_message_to_chatbot(user_message: str, chatbot: CHAT_HISTORY) -> tuple[str, CHAT_HISTORY]:
        if len(chatbot) > 0 and chatbot[-1]['role'] == 'user':
            chatbot = chatbot[:-1]
        chatbot.append(dict(role='user', content=user_message))
        return '', chatbot


    @staticmethod
    def update_user_message_with_context(
            chatbot: CHAT_HISTORY,
            rag_mode: bool,
            db: VectorStore,
            k: int | str,
            score_threshold: float,
    ) -> str:
        user_message = chatbot[-1]['content']
        user_message_with_context = ''
        if '{user_message}' not in PromptConfig.CONTEXT_TEMPLATE and '{context}' not in PromptConfig.CONTEXT_TEMPLATE:
            gr.Info('Context template must include {user_message} and {context}')
            return user_message_with_context
        if db is not None and rag_mode and user_message.strip():
            if k == 'all':
                k = len(db.docstore._dict)
            docs_and_distances = db.similarity_search_with_relevance_scores(
                user_message,
                k=k,
                score_threshold=score_threshold,
                )
            if len(docs_and_distances) > 0:
                retriever_context = '\n\n'.join([doc[0].page_content for doc in docs_and_distances])
                user_message_with_context = PromptConfig.CONTEXT_TEMPLATE.format(
                    user_message=user_message,
                    context=retriever_context,
                    )
        return user_message_with_context


    @staticmethod
    def _stream_llm_response_to_chatbot(
            llm_model: Llama,
            messages: CHAT_HISTORY,
            chatbot: CHAT_HISTORY,
            gen_kwargs: dict[str, int | float],
            show_thinking: bool,
    ) -> Iterator[CHAT_HISTORY]:
        stream_response = llm_model.create_chat_completion(
            messages=messages,
            stream=True,
            **gen_kwargs,
            )
        is_think = False
        for chunk in stream_response:
            token = chunk['choices'][0]['delta'].get('content')
            if token is None:
                continue
            if show_thinking:
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
    ) -> CHAT_HISTORY:
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
            user_message_with_context: str,
            rag_mode: bool,
            system_prompt: str,
            support_system_role: bool,
            history_len: int,
            do_sample: bool,
            request: gr.Request,
            *generate_args,
    ) -> Iterator[CHAT_HISTORY]:
        llm_model = ModelConfig.LLM_MODELS.get(request.session_hash)
        if llm_model is None:
            gr.Info('Model not initialized')
            yield chatbot[:-1]
            return
        gen_kwargs = dict(zip(ModelConfig.GENERATE_KWARGS.keys(), generate_args))
        gen_kwargs['top_k'] = int(gen_kwargs['top_k'])
        if not do_sample:
            gen_kwargs['top_p'] = 0.0
            gen_kwargs['top_k'] = 1
            gen_kwargs['repeat_penalty'] = 1.0
        user_message = chatbot[-1]['content']
        if not user_message.strip():
            yield chatbot[:-1]
            return
        if rag_mode:
            if user_message_with_context:
                user_message = user_message_with_context
            else:
                gr.Info((
                    'No documents relevant to the query were found, generation in RAG mode is not possible.\n'
                    'Or Context template is specified incorrectly.\n'
                    'Try reducing searh_score_threshold or disable RAG mode for normal generation'
                    ))
                yield chatbot[:-1]
                return
        messages = cls._prepare_messages(
            system_prompt=system_prompt,
            support_system_role=support_system_role,
            history_len=history_len,
            user_message=user_message,
            chatbot=chatbot,
        )
        chatbot.append(dict(role='assistant', content=''))
        try:
            yield from cls._stream_llm_response_to_chatbot(
                llm_model=llm_model,
                messages=messages,
                chatbot=chatbot,
                gen_kwargs=gen_kwargs,
                show_thinking=ModelConfig.SHOW_THINKING,
            )
        except Exception as ex:
            gr.Info(f'Error generating LLM response, error code:: {ex}')
            yield chatbot[:-2]

            return
