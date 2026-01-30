from types import SimpleNamespace

import pytest
import gradio as gr

from config import Config, ModelStorage


CHAT_HISTORY = list[gr.ChatMessage | dict[str, str | list | gr.Component]]


@pytest.fixture(scope='session', autouse=True)
def load_test_env() -> None:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path='tests/env_tests')


@pytest.fixture(scope='session')
def fake_request() -> SimpleNamespace:
    return SimpleNamespace(session_hash='123')


@pytest.fixture(scope='session')
def conf() -> Config:
    config = Config()
    config.generation_kwargs['rag_mode'] = True
    config.CHATBOT_RAG_ENABLED = True
    return config


@pytest.fixture(scope='session')
def db(conf: Config, fake_request: SimpleNamespace) -> dict:
    from modules.db import ChromaDb
    from modules.ui_fn import UiFnDb
    from modules.ui_fn import UiFnModel
    
    embed_load_log = UiFnModel.load_embed_model(
        config=conf,
        request=fake_request,
    )
    embed_model = ModelStorage.EMBED_MODEL.get(fake_request.session_hash)
    
    upload_files = ['tests/test_files/Pasport-huter-MK-7000-7800.pdf']
    web_links = ''
    texts, db_load_log = UiFnDb.load_texts_and_create_db(
        upload_files=upload_files,
        urls=web_links,
        config=conf,
        request=fake_request,
    )
    db = ChromaDb()
    collection = db.get_collection(
        collection_name=fake_request.session_hash,
        embedding_function=embed_model,
    )
    return dict(
        texts=texts,
        embed_model=embed_model,
        embed_load_log=embed_load_log,
        db_load_log=db_load_log,
        collection=collection,
        db=db,
    )
