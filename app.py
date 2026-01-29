import pprint
from dotenv import load_dotenv
load_dotenv()

from modules.logging_config import setup_logging
setup_logging()

from loguru import logger

from modules.ui_create import demo
from modules.llm import llama_server, llm_client
from config import UiGradioConfig


if __name__ == '__main__':
    try:
        llama_server.start()
        logger.debug((
            'llama.cpp server started, props: '
            f'{pprint.pformat(llm_client.get_props())}'
        ))
        demo.launch(**UiGradioConfig.get_demo_launch_kwargs())
        demo.close()
        demo.launch()
    finally:
        llama_server.stop()
