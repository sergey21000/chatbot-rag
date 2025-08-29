from dotenv import load_dotenv
load_dotenv()

from modules.ui_create import interface


if __name__ == '__main__':
    interface.launch()