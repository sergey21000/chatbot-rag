import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from typing import Iterator

import pytest
import gradio as gr
import torch
from llama_cpp import Llama
from chromadb import EmbeddingFunction

from config import Config, ModelStorage
from modules.db import ChromaDb
from modules.ui_fn import (
    UiFnModel,
    UiFnDb,
    UiFnChat,
)


CHAT_HISTORY = list[gr.ChatMessage | dict[str, str | gr.Component]]
CONF = Config()
CONF.generation_kwargs['rag_mode'] = True
CONF.load_model_kwargs['llm_model_repo'] = 'bartowski/google_gemma-3-1b-it-GGUF'
CONF.load_model_kwargs['llm_model_file'] = 'google_gemma-3-1b-it-Q8_0.gguf'
CONF.load_model_kwargs['embed_model_repo'] = 'sergeyzh/rubert-tiny-turbo'


class FakeRequest:
    session_hash = '123'


@pytest.fixture  # (scope='function', autouse=True)
def llm_model():
    load_log: str = UiFnModel.load_llm_model(
        config=CONF,
        request=FakeRequest,
    )
    model: Llama = ModelStorage.LLM_MODEL.get(FakeRequest.session_hash)
    print(f'LLM Loading logs: {load_log}')
    assert isinstance(model, Llama), 'LLM model failed to load'
    return model


@pytest.fixture
def embed_model():
    load_log: str = UiFnModel.load_embed_model(
        config=CONF,
        request=FakeRequest,
    )
    embedding_function: EmbeddingFunction = ModelStorage.EMBED_MODEL.get(FakeRequest.session_hash)
    print(f'Embeddings model loading logs: {load_log}')
    assert isinstance(embedding_function, EmbeddingFunction), 'Embeddings model failed to load'
    return embedding_function


@pytest.fixture
def test_db(embed_model):
    # файлы и ссылки на видео для проверки
    upload_files: list[str] = ['tests/test_files/Pasport-huter-MK-7000-7800.pdf']
    web_links: str = 'https://www.youtube.com/watch?v=CFVABT8wtl4 https://www.youtube.com/watch?v=EEGk7gHoKfY'
    texts, load_log = UiFnDb.load_texts_and_create_db(
        upload_files=upload_files,
        urls=web_links,
        config=CONF,
        request=FakeRequest,
    )
    print(f'DB Load logs: {load_log}')
    assert len(texts) > 0, 'Empty list of DB documents'
    print(f'Number of texts downloaded: {len(texts)}')

    embedding_function: EmbeddingFunction = ModelStorage.EMBED_MODEL.get(FakeRequest.session_hash)
    db = ChromaDb()
    collection = db.get_collection(
        collection_name=FakeRequest.session_hash,
        embedding_function=embedding_function,
    )
    assert collection is not None, 'Collection not created'


def test_rag_pipepline(llm_model, test_db):
    chatbot: CHAT_HISTORY = []
    user_message: str = 'Какое масло заливать в двигатель и сколько?'
    _, chatbot = UiFnChat.user_message_to_chatbot(
        user_message=user_message,
        chatbot=chatbot,
    )
    assert len(chatbot) > 0,'The message was not added to the chat'

    UiFnChat.update_user_msg_with_context(
        chatbot=chatbot,
        config=CONF,
        request=FakeRequest,
    )
    user_msg_with_context = CONF.generation_kwargs['user_msg_with_context']
    assert len(user_msg_with_context) > 0, 'user_msg_with_context not created'
    print(f'User message enriched with context: {user_msg_with_context}')
    assert 'масло' in user_msg_with_context.lower()
    assert len(user_msg_with_context) > len(user_message), 'No context added'

    stream_chatbot: Iterator[CHAT_HISTORY] = UiFnChat.yield_chatbot_with_llm_response(
        chatbot=chatbot,
        config=CONF,
        request=FakeRequest,
    )
    for result_chatbot in stream_chatbot:
        pass
    assistant_message: str = result_chatbot[-1].get('content', '')
    assert len(assistant_message) > 0, 'LLM did not respond'
    print(f'Chatbot response: {assistant_message}')