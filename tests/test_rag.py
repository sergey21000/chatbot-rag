from typing import Iterator

import pytest
import gradio as gr
from chromadb import EmbeddingFunction

from config import Config


CHAT_HISTORY = list[gr.ChatMessage | dict[str, str | list | gr.Component]]


def test_db_creation(db):
    print(f"Embeddings model loading logs: {db['embed_load_log']}")
    assert isinstance(db['embed_model'], EmbeddingFunction), 'Embeddings model failed to load'
    assert len(db['texts']) > 0, 'Empty list of DB documents'
    assert db['collection'] is not None, 'Collection not created'
    print(f"Number of texts downloaded: {len(db['texts'])}")
    print(f"DB Load logs: {db['db_load_log']}")


@pytest.mark.order(1)
def test_load_llm_model(fake_request):
    from modules.ui_fn import UiFnModel

    CONF = Config()
    CONF.load_model_kwargs['llm_model_repo'] = 'bartowski/Qwen_Qwen3-0.6B-GGUF'
    CONF.load_model_kwargs['llm_model_file'] = 'Qwen_Qwen3-0.6B-Q4_K_M.gguf'
    load_log: str = UiFnModel.load_llm_model(
        config=CONF,
        request=fake_request,
    )
    print(f'LLM Loading logs: {load_log}')


@pytest.mark.order(2)
def test_llm_server():
    from modules.llm import llm_server, llm_client

    llm_server.start()
    assert llm_client.check_health(), 'llm_client.check_health() failed'
    llm_server.stop()


@pytest.mark.order(3)
def test_rag_pipeline(db, conf, fake_request):
    from modules.ui_fn import UiFnChat

    if db['collection'] is None:
        pytest.skip('DB not created, skipping RAG test')

    chatbot: CHAT_HISTORY = []
    user_message: dict[str, str | list[str]] = {
        'text': 'Какое масло заливать в двигатель и сколько?',
        'files': [],
    }
    _, chatbot = UiFnChat.user_message_to_chatbot(
        user_message=user_message,
        chatbot=chatbot,
    )
    assert len(chatbot) > 0,'The message was not added to the chat'

    UiFnChat.update_user_msg_with_context(
        chatbot=chatbot,
        config=conf,
        request=fake_request,
    )
    user_msg_with_context = conf.generation_kwargs['user_msg_with_context']
    assert len(user_msg_with_context) > 0, 'user_msg_with_context not created'
    print(f'User message enriched with context: {user_msg_with_context}')
    assert 'масло' in user_msg_with_context.lower()
    assert len(user_msg_with_context) > len(user_message['text']), 'No context added'

    stream_chatbot: Iterator[CHAT_HISTORY] = UiFnChat.yield_chatbot_with_llm_response(
        chatbot=chatbot,
        config=conf,
        request=fake_request,
    )
    for result_chatbot in stream_chatbot:
        pass
    assert chatbot, "yield_chatbot_with_llm_response() didn't respond"
    assistant_message: str = result_chatbot[-1].get('content', '')
    assert len(assistant_message) > 0, 'Empty assistant_message'
    print(f'Chatbot response: {assistant_message}')
