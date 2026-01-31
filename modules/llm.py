import os

from loguru import logger
from llama_cpp_py import (
    LlamaReleaseManager,
    LlamaSyncServer,
    LlamaSyncClient,
)


tag = os.getenv('LLAMACPP_RELEASE_TAG', 'latest')
release_zip_url = os.getenv('LLAMACPP_RELEASE_ZIP_URL')
llamacpp_dir = os.getenv('LLAMACPP_DIR')
openai_base_url = os.getenv('OPENAI_BASE_URL')

if openai_base_url:
    llama_server = None
    llm_client = LlamaSyncClient(openai_base_url=openai_base_url)
else:
    if llamacpp_dir:
        llama_server = LlamaSyncServer(verbose=True, llama_dir=llamacpp_dir)
    elif release_zip_url:
        release_manager = LlamaReleaseManager(release_zip_url=release_zip_url)
        llama_server = LlamaSyncServer(verbose=True, release_manager=release_manager)
    else:
        release_manager = LlamaReleaseManager(tag=tag)
        llama_server = LlamaSyncServer(verbose=True, release_manager=release_manager)
    llm_client = LlamaSyncClient(openai_base_url=llama_server.server_url)

logger.debug(f'llm_client initialized, server_url: {llm_client.openai_base_url}')
if llama_server:
    logger.debug(f'llama_server initialized, llama_dir: {llama_server.llama_dir}')
