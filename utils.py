import csv
from pathlib import Path
from shutil import rmtree
from typing import List, Tuple, Dict, Union, Optional, Any, Iterable
from tqdm import tqdm

import psutil
import requests
from requests.exceptions import MissingSchema

import torch
import gradio as gr

from llama_cpp import Llama
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from huggingface_hub import hf_hub_download, list_repo_tree, list_repo_files, repo_info, repo_exists, snapshot_download

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# imports for annotations
from langchain.docstore.document import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from config import (
    LLM_MODELS_PATH,
    EMBED_MODELS_PATH,
    GENERATE_KWARGS,
    LOADER_CLASSES,
    CONTEXT_TEMPLATE,
)


# type annotations
CHAT_HISTORY = List[Tuple[Optional[str], Optional[str]]]
LLM_MODEL_DICT = Dict[str, Llama]
EMBED_MODEL_DICT = Dict[str, Embeddings]


# ===================== ADDITIONAL FUNCS =======================

# getting the amount of free memory on disk, CPU and GPU
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


# clearing the list of documents
def clear_documents(documents: Iterable[Document]) -> Iterable[Document]:
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


# ===================== INTERFACE FUNCS =============================


# ------------- LLM AND EMBEDDING MODELS LOADING ------------------------

# функция для загрузки файла по URL ссылке и отображением прогресс баров tqdm и gradio
def download_file(file_url: str, file_path: Union[str, Path]) -> None:
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


# loading and initializing the GGUF model
def load_llm_model(model_repo: str, model_file: str) -> Tuple[LLM_MODEL_DICT, str, str]:
    llm_model = None
    load_log = ''
    support_system_role = False

    if isinstance(model_file, list):
        load_log += 'No model selected\n'
        return llm_model, load_log
    if '(' in model_file:
        model_file = model_file.split('(')[0].rstrip()

    progress = gr.Progress()
    progress(0.3, desc='Step 1/2: Download the GGUF file')
    model_path = LLM_MODELS_PATH / model_file

    if model_path.is_file():
        load_log += f'Model {model_file} already loaded, reinitializing\n'
    else:
        try:
            gguf_url = f'https://huggingface.co/{model_repo}/resolve/main/{model_file}'
            download_file(gguf_url, model_path)
            load_log += f'Model {model_file} loaded\n'
        except Exception as ex:
            model_path = ''
            load_log += f'Error loading model, error code:\n{ex}\n'

    if model_path:
        progress(0.7, desc='Step 2/2: Initialize the model')
        try:
            llm_model = Llama(model_path=str(model_path), n_gpu_layers=-1, verbose=False)
            support_system_role = 'System role not supported' not in llm_model.metadata['tokenizer.chat_template']
            load_log += f'Model {model_file} initialized, max context size is {llm_model.n_ctx()} tokens\n'
        except Exception as ex:
            load_log += f'Error initializing model, error code:\n{ex}\n'

    llm_model = {'model': llm_model}
    return llm_model, support_system_role, load_log


# loading and initializing the embedding model
def load_embed_model(model_repo: str) -> Tuple[Dict[str, HuggingFaceEmbeddings], str]:
    embed_model = None
    load_log = ''

    if isinstance(model_repo, list):
        load_log = 'No model selected'
        return embed_model, load_log

    progress = gr.Progress()
    folder_name = model_repo.replace('/', '_')
    folder_path = EMBED_MODELS_PATH / folder_name
    if Path(folder_path).is_dir():
        load_log += f'Reinitializing model {model_repo} \n'
    else:
        progress(0.5, desc='Step 1/2: Download model repository')
        snapshot_download(
            repo_id=model_repo,
            local_dir=folder_path,
            ignore_patterns='*.h5',
        )
        load_log += f'Model {model_repo} loaded\n'

    progress(0.7, desc='Шаг 2/2: Инициализация модели')
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    embed_model = HuggingFaceEmbeddings(
        model_name=str(folder_path), 
        model_kwargs=model_kwargs,
        # encode_kwargs={'normalize_embeddings': True},
        )
    load_log += f'Embeddings model {model_repo} initialized\n'
    load_log += f'Please upload documents and initialize database again\n'
    embed_model = {'embed_model': embed_model}
    return embed_model, load_log


# adding a new HF repository new_model_repo to the current list of model_repos
def add_new_model_repo(new_model_repo: str, model_repos: List[str]) -> Tuple[gr.Dropdown, str]:
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


# get list of GGUF models from HF repository
def get_gguf_model_names(model_repo: str) -> gr.Dropdown:
    repo_files = list(list_repo_tree(model_repo))
    repo_files = [file for file in repo_files if file.path.endswith('.gguf')]
    model_paths = [f'{file.path} ({file.size / 1000 ** 3:.2f}G)' for file in repo_files]
    model_paths_dropdown = gr.Dropdown(
        choices=model_paths,
        value=model_paths[0],
        label='GGUF model file',
        )
    return model_paths_dropdown


# delete model files and folders to clear space except for the current model gguf_filename
def clear_llm_folder(gguf_filename: str) -> None:
    if gguf_filename is None:
        gr.Info(f'The name of the model file that does not need to be deleted is not selected.')
        return
    if '(' in gguf_filename:
        gguf_filename = gguf_filename.split('(')[0].rstrip()
    for path in LLM_MODELS_PATH.iterdir():
        if path.name == gguf_filename:
            continue
        if path.is_file():
            path.unlink(missing_ok=True)
    gr.Info(f'All files removed from directory {LLM_MODELS_PATH} except {gguf_filename}')


# delete model folders to clear space except for the current model model_folder_name
def clear_embed_folder(model_repo: str) -> None:
    if model_repo is None:
        gr.Info(f'The name of the model that does not need to be deleted is not selected.')
        return
    model_folder_name = model_repo.replace('/', '_')
    for path in EMBED_MODELS_PATH.iterdir():
        if path.name == model_folder_name:
            continue
        if path.is_dir():
            rmtree(path, ignore_errors=True)
    gr.Info(f'All directories have been removed from the {EMBED_MODELS_PATH} directory except {model_folder_name}')


# ------------------------ YOUTUBE ------------------------

# function to check availability of subtitles, if manual or automatic are available - returns True and logs
# if subtitles are not available - returns False and logs
def check_subtitles_available(yt_video_link: str, target_lang: str) -> Tuple[bool, str]:
    video_id = yt_video_link.split('watch?v=')[-1].split('&')[0]
    load_log = ''
    available = True
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript([target_lang])
            if transcript.is_generated:
                load_log += f'Automatic subtitles will be loaded, manual ones are not available for video {yt_video_link}\n'
            else:
                load_log += f'Manual subtitles will be downloaded for the video {yt_video_link}\n'
        except NoTranscriptFound:
            load_log += f'Subtitle language {target_lang} is not available for video {yt_video_link}\n'
            available = False
    except TranscriptsDisabled:
        load_log += f'Invalid video url ({yt_video_link}) or current server IP is blocked for YouTube'
        available = False
    return available, load_log


# ------------- UPLOADING DOCUMENTS FOR RAG ------------------------

# extract documents (in langchain Documents format) from downloaded files
def load_documents_from_files(upload_files: List[str]) -> Tuple[List[Document], str]:
    load_log = ''
    documents = []
    for upload_file in upload_files:
        file_extension = f".{upload_file.split('.')[-1]}"
        if file_extension in LOADER_CLASSES:
            loader_class = LOADER_CLASSES[file_extension]
            loader_kwargs = {}
            if file_extension == '.csv':
                with open(upload_file) as csvfile:
                    delimiter = csv.Sniffer().sniff(csvfile.read(4096)).delimiter
                loader_kwargs = {'csv_args': {'delimiter': delimiter}}
            try:
                load_documents = loader_class(upload_file, **loader_kwargs).load()
                documents.extend(load_documents)
            except Exception as ex:
                load_log += f'Error uploading file {upload_file}\n'
                load_log += f'Error code: {ex}\n'
                continue
        else:
            load_log += f'Unsupported file format {upload_file}\n'
            continue
    return documents, load_log


# extracting documents (in langchain Documents format) from WEB links
def load_documents_from_links(
        web_links: str,
        subtitles_lang: str,
        ) -> Tuple[List[Document], str]:

    load_log = ''
    documents = []
    loader_class_kwargs = {}
    web_links = [web_link.strip() for web_link in web_links.split('\n') if web_link.strip()]
    for web_link in web_links:
        if 'youtube.com' in web_link:
            available, log = check_subtitles_available(web_link, subtitles_lang)
            load_log += log
            if not available:
                continue
            loader_class = LOADER_CLASSES['youtube'].from_youtube_url
            loader_class_kwargs = {'language': subtitles_lang}
        else:
            loader_class = LOADER_CLASSES['web']
            
        try:
            if requests.get(web_link).status_code != 200:
                load_log += f'Ссылка недоступна для Python requests: {web_link}\n'
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


# uploading files and generating documents and databases
def load_documents_and_create_db(
        upload_files: Optional[List[str]],
        web_links: str,
        subtitles_lang: str,
        chunk_size: int,
        chunk_overlap: int,
        embed_model_dict: EMBED_MODEL_DICT,
        ) -> Tuple[List[Document], Optional[VectorStore], str]:

    load_log = ''
    all_documents = []
    db = None
    progress = gr.Progress()

    embed_model = embed_model_dict.get('embed_model')
    if embed_model is None:
        load_log += 'Embeddings model not initialized, DB cannot be created'
        return all_documents, db, load_log

    if upload_files is None and not web_links:
        load_log = 'No files or links selected'
        return all_documents, db, load_log

    if upload_files is not None:
        progress(0.3, desc='Step 1/2: Upload documents from files')
        docs, log = load_documents_from_files(upload_files)
        all_documents.extend(docs)
        load_log += log

    if web_links:
        progress(0.3 if upload_files is None else 0.5, desc='Step 1/2: Upload documents via links')
        docs, log = load_documents_from_links(web_links, subtitles_lang)
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
    )
    documents = text_splitter.split_documents(all_documents)
    documents = clear_documents(documents)
    load_log += f'Documents are divided, number of text chunks: {len(documents)}\n'

    progress(0.7, desc='Step 2/2: Initialize DB')
    db = FAISS.from_documents(documents=documents, embedding=embed_model)
    load_log += 'DB is initialized, RAG mode is activated and can be activated in the Chatbot tab'
    return documents, db, load_log


# ------------------ ФУНКЦИИ ЧАТ БОТА ------------------------

# adding a user message to the chat bot window
def user_message_to_chatbot(user_message: str, chatbot: CHAT_HISTORY) -> Tuple[str, CHAT_HISTORY]:
    chatbot.append([user_message, None])
    return '', chatbot


# formatting prompt with adding context if DB is available and RAG mode is enabled
def update_user_message_with_context(
        chatbot: CHAT_HISTORY,
        rag_mode: bool,
        db: VectorStore,
        k: Union[int, str],
        score_threshold: float,
        ) -> Tuple[str, CHAT_HISTORY]:

    user_message = chatbot[-1][0]
    user_message_with_context = ''
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
            user_message_with_context = CONTEXT_TEMPLATE.format(
                user_message=user_message,
                context=retriever_context,
                )
    return user_message_with_context


# model response generation
def get_llm_response(
        chatbot: CHAT_HISTORY,
        llm_model_dict: LLM_MODEL_DICT,
        user_message_with_context: str,
        rag_mode: bool,
        system_prompt: str,
        support_system_role: bool,
        history_len: int,
        do_sample: bool,
        *generate_args,
        ) -> CHAT_HISTORY:

    user_message = chatbot[-1][0]
    if not user_message.strip():
        yield chatbot[:-1]
        return None

    if rag_mode:
        if user_message_with_context:
            user_message = user_message_with_context
        else:
            gr.Info((
                f'No documents relevant to the query were found, generation in RAG mode is not possible.\n'
                f'Try reducing searh_score_threshold or disable RAG mode for normal generation'
                ))
            yield chatbot[:-1]
            return None

    llm_model = llm_model_dict.get('model')
    gen_kwargs = dict(zip(GENERATE_KWARGS.keys(), generate_args))
    gen_kwargs['top_k'] = int(gen_kwargs['top_k'])
    if not do_sample:
        gen_kwargs['top_p'] = 0.0
        gen_kwargs['top_k'] = 1
        gen_kwargs['repeat_penalty'] = 1.0

    messages = []
    if support_system_role and system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})

    if history_len != 0:
        for user_msg, bot_msg in chatbot[:-1][-history_len:]:
            messages.append({'role': 'user', 'content': user_msg})
            messages.append({'role': 'assistant', 'content': bot_msg})

    messages.append({'role': 'user', 'content': user_message})
    stream_response = llm_model.create_chat_completion(
        messages=messages,
        stream=True,
        **gen_kwargs,
        )
    try:
        chatbot[-1][1] = ''
        for chunk in stream_response:
            token = chunk['choices'][0]['delta'].get('content')
            if token is not None:
                chatbot[-1][1] += token
                yield chatbot
    except Exception as ex:
        gr.Info(f'Error generating response, error code: {ex}')
        yield chatbot
