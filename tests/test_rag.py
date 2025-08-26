# pytest tests/test_rag.py -vs

import sys
from pathlib import Path
from copy import deepcopy
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import gradio as gr
import torch
from llama_cpp import Llama
from langchain_core.embeddings import Embeddings

from config import ModelConfig
from utils import (
    UiFnService,
    UiFnModel,
    UiFnDb,
    UiFnChat,
)

CHAT_HISTORY = list[gr.ChatMessage | dict[str, str | gr.Component]]
ModelConfig.START_LLM_MODEL_REPO = 'bartowski/google_gemma-3-1b-it-GGUF'
ModelConfig.START_LLM_MODEL_FILE = 'google_gemma-3-1b-it-Q8_0.gguf'
ModelConfig.START_EMBED_MODEL_REPO = 'sergeyzh/rubert-tiny-turbo'


class FakeRequest:
    session_hash = '123'


@pytest.fixture
def llm_model():
    load_log: str = UiFnModel.load_llm_model(
        model_repo=ModelConfig.START_LLM_MODEL_REPO,
        model_file=ModelConfig.START_LLM_MODEL_FILE,
        request=FakeRequest,
    )
    model: Llama = ModelConfig.LLM_MODELS.get(FakeRequest.session_hash)
    print(f'LLM Loading logs: {load_log}')
    assert isinstance(model, Llama), 'LLM model failed to load'
    return model


@pytest.fixture
def embed_model():
    load_log: str = UiFnModel.load_embed_model(
        model_repo=ModelConfig.START_EMBED_MODEL_REPO,
        request=FakeRequest,
    )
    model: Embeddings = ModelConfig.EMBED_MODELS.get(FakeRequest.session_hash)
    print(f'Embeddings model loading logs: {load_log}')
    assert isinstance(model, Embeddings), 'Embeddings model failed to load'
    return model


@pytest.fixture
def test_db(embed_model):
    upload_files: list[str] = ['tests/test_files/Pasport-huter-MK-7000-7800.pdf']
    # web_links: str = 'https://www.youtube.com/watch?v=CFVABT8wtl4 https://www.youtube.com/watch?v=EEGk7gHoKfY'
    web_links: str = ''
    documents, db, load_log = UiFnDb.load_documents_and_create_db(
        upload_files=upload_files,
        web_links=web_links,
        subtitles_lang='ru',
        chunk_size=500,
        chunk_overlap=20,
        request=FakeRequest,
    )
    print(f'DB Load logs: {load_log}')
    assert len(documents) > 0, 'Empty list of DB documents'
    assert db is not None, 'DB not created'
    return db


def test_rag_pipepline(llm_model, test_db):
    chatbot: CHAT_HISTORY = []
    user_message: str = 'Какое масло заливать в двигатель и сколько?'
    _, chatbot = UiFnChat.user_message_to_chatbot(
        user_message=user_message,
        chatbot=chatbot,
    )
    assert len(chatbot) > 0,'The message was not added to the chat'

    user_message_with_context: str = UiFnChat.update_user_message_with_context(
        chatbot=chatbot,
        rag_mode=True,
        db=test_db,
        k=2,
        score_threshold=0.1,
    )
    print(f'User message enriched with context: {user_message_with_context}')
    assert 'масло' in user_message_with_context.lower()
    assert len(user_message_with_context) > len(user_message), 'No context added'

    stream_chatbot: Iterator[CHAT_HISTORY] = UiFnChat.yield_chatbot_with_llm_response(
        chatbot,  # chatbot
        user_message_with_context,  # user_message_with_context
        True,  # rag_mode
        '',  # system_prompt
        True,  # support_system_role
        0,  # history_len
        False,  # do_sample
        FakeRequest,  # reques
        *list(ModelConfig.GENERATE_KWARGS.values()),  # generate_args
    )
    for result_chatbot in stream_chatbot:
        pass
    assistant_message: str = result_chatbot[-1].get('content')
    assert len(assistant_message) > 0, 'LLM did not respond'
    print(f'Chatbot response: {assistant_message}')