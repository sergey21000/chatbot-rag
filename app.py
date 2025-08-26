import gradio as gr
from langchain_core.vectorstores import VectorStore

from config import (
    SettingsConfig,
    PromptConfig,
    ModelConfig,
    ReposConfig,
)
from utils import (
    UiFnService,
    UiFnModel,
    UiFnDb,
    UiFnChat,
)


class UiComponent:
    '''Gradio UI components'''
    # theme = gr.themes.Monochrome()
    # theme = gr.themes.Base(primary_hue='green', secondary_hue='yellow', neutral_hue='zinc').set(
    #     loader_color='rgb(0, 255, 0)',
    #     slider_color='rgb(0, 200, 0)',
    #     body_text_color_dark='rgb(0, 200, 0)',
    #     button_secondary_background_fill_dark='green',
    # )
    theme = None
    css = '''
    .gradio-container {
        width: 70% !important;
        margin: 0 auto !important;
    }
    '''

    @staticmethod
    def rag_settings(rag_mode: bool, render: bool = True) -> tuple[gr.Component]:
        k = gr.Radio(
            choices=[1, 2, 3, 4, 5, 'all'],
            value=2,
            label='Number of relevant documents for search',
            visible=rag_mode,
            render=render,
            )
        score_threshold = gr.Slider(
            minimum=0,
            maximum=1,
            value=0.5,
            step=0.05,
            label='relevance_scores_threshold',
            visible=rag_mode,
            render=render,
            )
        return k, score_threshold

    @staticmethod
    def user_message_with_context(text: str, rag_mode: bool) -> gr.Component:
        num_lines = len(text.split('\n'))
        max_lines = 10
        num_lines = max_lines if num_lines > max_lines else num_lines
        return gr.Textbox(
            text,
            visible=rag_mode,
            interactive=False,
            label='User Message With Context',
            lines=num_lines,
            )

    @staticmethod
    def system_prompt(interactive: bool) -> gr.Textbox:
        value = '' if interactive else 'System prompt is not supported by this model'
        return gr.Textbox(value=value, label='System prompt', interactive=interactive)

    @staticmethod
    def generate_args(do_sample: bool) -> list[gr.Component]:
        # если do_sample включен (элемент gr.Checkbox() активен) то отображать слайдера с параметрами генерации
        KWARGS = ModelConfig.GENERATE_KWARGS
        generate_args = [
            gr.Slider(minimum=0.1, maximum=3, value=KWARGS['temperature'], step=0.1, label='temperature', visible=do_sample),
            gr.Slider(minimum=0.1, maximum=1, value=KWARGS['top_p'], step=0.01, label='top_p', visible=do_sample),
            gr.Slider(minimum=1, maximum=50, value=KWARGS['top_k'], step=1, label='top_k', visible=do_sample),
            gr.Slider(minimum=1, maximum=5, value=KWARGS['repeat_penalty'], step=0.1, label='repeat_penalty', visible=do_sample),
        ]
        return generate_args

    @staticmethod
    def rag_mode(db: VectorStore | None) -> gr.Checkbox:
        value = visible = db is not None
        return gr.Checkbox(value=value, label='RAG Mode', scale=1, visible=visible)



with gr.Blocks(theme=UiComponent.theme, css=UiComponent.css) as interface:
    documents = gr.State([])
    db = gr.State(None)
    user_message_with_context = gr.State('')
    support_system_role = gr.State(False)
    llm_model_repos = gr.State(ReposConfig.LLM_MODEL_REPOS)
    embed_model_repos = gr.State(ReposConfig.EMBED_MODEL_REPOS)

    with gr.Tab(label='Chatbot'):
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    type='messages',
                    show_copy_button=True,
                    height=480,
                )
                user_message = gr.Textbox(label='User')
                with gr.Row():
                    user_message_btn = gr.Button('Send')
                    stop_btn = gr.Button('Stop')
                    clear_btn = gr.Button('Clear')

            with gr.Column(scale=1, min_width=80):
                with gr.Group():
                    gr.Markdown('History size')
                    history_len = gr.Slider(
                        minimum=0,
                        maximum=5,
                        value=0,
                        step=1,
                        info='Number of previous user-bot message pairs to keep in chat history',
                        label='history len',
                        show_label=False,
                        )
                    with gr.Group():
                        gr.Markdown('Generation parameters')
                        do_sample = gr.Checkbox(
                            value=False,
                            label='do_sample',
                            info='Activate random sampling',
                            )
                        generate_args = UiComponent.generate_args(do_sample.value)
                        do_sample.change(
                            fn=UiComponent.generate_args,
                            inputs=do_sample,
                            outputs=generate_args,
                            show_progress=False,
                            show_api=False,
                            )
        rag_mode = UiComponent.rag_mode(db=db.value)
        k, score_threshold = UiComponent.rag_settings(rag_mode=rag_mode.value, render=False)
        rag_mode.change(
            fn=UiComponent.rag_settings,
            inputs=[rag_mode],
            outputs=[k, score_threshold],
            show_api=False,
            )
        with gr.Row():
            k.render()
            score_threshold.render()
        with gr.Accordion('Prompt', open=True):
            system_prompt = UiComponent.system_prompt(interactive=support_system_role.value)
            user_message_with_context = UiComponent.user_message_with_context(text='', rag_mode=rag_mode.value)

        generate_event = gr.on(
            triggers=[user_message.submit, user_message_btn.click],
            fn=UiFnChat.user_message_to_chatbot,
            inputs=[user_message, chatbot],
            outputs=[user_message, chatbot],
            queue=False,
        ).then(
            fn=UiFnChat.update_user_message_with_context,
            inputs=[chatbot, rag_mode, db, k, score_threshold],
            outputs=[user_message_with_context],
        ).then(
            fn=UiComponent.user_message_with_context,
            inputs=[user_message_with_context, rag_mode],
            outputs=[user_message_with_context],
            queue=False,
        ).then(
            fn=UiFnChat.yield_chatbot_with_llm_response,
            inputs=[chatbot, user_message_with_context, rag_mode, system_prompt,
                    support_system_role, history_len, do_sample, *generate_args],
            outputs=[chatbot],
        )
        # кнопка Стоп
        stop_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=generate_event,
            queue=False,
            show_api=False,
        )
        # кнопка Очистить чат
        clear_btn.click(
            fn=lambda: (None, ''),
            inputs=None,
            outputs=[chatbot, user_message_with_context],
            queue=False,
            show_api=False,
            )


    with gr.Tab(label='Load documents'):
        with gr.Row(variant='compact'):
            upload_files = gr.File(file_count='multiple', label='Loading text files')
            web_links = gr.Textbox(lines=6, label='Links to Web sites or YouTube')

        with gr.Row(variant='compact'):
            chunk_size = gr.Slider(50, 2000, value=500, step=50, label='Chunk size')
            chunk_overlap = gr.Slider(0, 200, value=20, step=10, label='Chunk overlap')
            subtitles_lang = gr.Radio(
                SettingsConfig.SUBTITLES_LANGUAGES,
                value=SettingsConfig.SUBTITLES_LANGUAGES[0],
                label='YouTube subtitle language',
                )
        load_documents_btn = gr.Button(value='Upload documents and initialize database')
        load_docs_log = gr.Textbox(label='Status of loading and splitting documents', interactive=False)
        load_documents_btn.click(
            fn=UiFnDb.load_documents_and_create_db,
            inputs=[upload_files, web_links, subtitles_lang, chunk_size, chunk_overlap],
            outputs=[documents, db, load_docs_log],
        ).success(
            fn=UiComponent.rag_mode,
            inputs=[db],
            outputs=[rag_mode],
            show_api=False,
        )


    with gr.Tab(label='View documents'):
        view_documents_btn = gr.Button(value='Show downloaded text chunks')
        view_documents_textbox = gr.Textbox(
            placeholder='To view chunks, load documents in the Load documents tab',
            label='Uploaded chunks',
            lines=1,
            )
        view_documents_btn.click(
            fn=UiFnService.view_documents,
            inputs=[documents],
            outputs=[view_documents_textbox],
        )


    with gr.Tab('Load LLM model'):
        new_llm_model_repo = gr.Textbox(
            value='',
            label='Add repository',
            placeholder='Link to repository of HF models in GGUF format',
            )
        new_llm_model_repo_btn = gr.Button('Add repository')
        curr_llm_model_repo = gr.Dropdown(
            choices=ReposConfig.LLM_MODEL_REPOS,
            value=None,
            label='HF Model Repository',
            )
        curr_llm_model_path = gr.Dropdown(
            choices=None,
            value=None,
            label='GGUF model file',
            allow_custom_value=True,
            )
        load_llm_model_btn = gr.Button('Loading and initializing model')
        load_llm_model_log = gr.Textbox(
            value=f'Model {ModelConfig.START_LLM_MODEL_REPO}/{ModelConfig.START_LLM_MODEL_FILE} loaded at application startup',
            label='Model loading status',
            interactive=False,
            lines=6,
            )
        with gr.Group():
            gr.Markdown('Free up disk space by deleting all models except the currently selected one')
            clear_llm_folder_btn = gr.Button('Clear folder')
        new_llm_model_repo_btn.click(
            fn=UiFnModel.add_new_model_repo,
            inputs=[new_llm_model_repo, llm_model_repos],
            outputs=[curr_llm_model_repo, load_llm_model_log],
            show_api=False,
        ).success(
            fn=lambda: '',
            inputs=None,
            outputs=[new_llm_model_repo],
            show_api=False,
        ).then(
            fn=lambda: (None, None),
            inputs=None,
            outputs=[curr_llm_model_repo, curr_llm_model_path],
            show_api=False,
        )
        curr_llm_model_repo.input(  # input, change
            fn=UiFnModel.get_gguf_model_names,
            inputs=[curr_llm_model_repo],
            outputs=[curr_llm_model_path],
        )
        load_llm_model_btn.click(
            fn=UiFnModel.load_llm_model,
            inputs=[curr_llm_model_repo, curr_llm_model_path],
            outputs=[support_system_role, load_llm_model_log],
            show_api=False,
        ).success(
            fn=lambda log: log + UiFnService.get_memory_usage(),
            inputs=[load_llm_model_log],
            outputs=[load_llm_model_log],
            show_api=False,
        ).then(
            fn=UiComponent.system_prompt,
            inputs=[support_system_role],
            outputs=[system_prompt],
            show_api=False,
        )
        clear_llm_folder_btn.click(
            fn=UiFnModel.clear_llm_folder,
            inputs=[curr_llm_model_path],
            outputs=None,
        ).success(
            fn=lambda model_path: f'Models other than {model_path} removed',
            inputs=[curr_llm_model_path],
            outputs=[load_llm_model_log],
            show_api=False,
        )


    with gr.Tab('Load embed model'):
        new_embed_model_repo = gr.Textbox(
            value='',
            label='Add repository',
            placeholder='Link to HF model repository',
            )
        new_embed_model_repo_btn = gr.Button('Add repository')
        curr_embed_model_repo = gr.Dropdown(
            choices=ReposConfig.EMBED_MODEL_REPOS,
            value=None,
            label='HF model repository',
            )
        load_embed_model_btn = gr.Button('Loading and initializing model')
        load_embed_model_log = gr.Textbox(
            value=f'Model {ModelConfig.START_EMBED_MODEL_REPO} loaded at application startup',
            label='Model loading status',
            interactive=False,
            lines=7,
            )
        with gr.Group():
            gr.Markdown('Free up disk space by deleting all models except the currently selected one')
            clear_embed_folder_btn = gr.Button('Clear folder')
        new_embed_model_repo_btn.click(
            fn=UiFnModel.add_new_model_repo,
            inputs=[new_embed_model_repo, embed_model_repos],
            outputs=[curr_embed_model_repo, load_embed_model_log],
            show_api=False,
        ).success(
            fn=lambda: '',
            inputs=None,
            outputs=new_embed_model_repo,
            show_api=False,
        )
        load_embed_model_btn.click(
            fn=UiFnModel.load_embed_model,
            inputs=[curr_embed_model_repo],
            outputs=[load_embed_model_log],
            show_api=False,
        ).success(
            fn=lambda log: log + UiFnService.get_memory_usage(),
            inputs=[load_embed_model_log],
            outputs=[load_embed_model_log],
            show_api=False,
        )
        clear_embed_folder_btn.click(
            fn=UiFnModel.clear_embed_folder,
            inputs=[curr_embed_model_repo],
            outputs=None,
        ).success(
            fn=lambda model_repo: f'Models other than {model_repo} removed',
            inputs=[curr_embed_model_repo],
            outputs=[load_embed_model_log],
            show_api=False,
        )

    start_llm_model_repo = gr.State(ModelConfig.START_LLM_MODEL_REPO)
    start_llm_model_file = gr.State(ModelConfig.START_LLM_MODEL_FILE)
    start_embed_model_repo = gr.State(ModelConfig.START_EMBED_MODEL_REPO)
    interface.load(
        fn=lambda: gr.Info('Loading LLM and Embeddings models, the status is displayed on the Load models tabs'),
        inputs=None,
        outputs=None,
        show_api=False,
    ).then(
        fn=UiFnModel.load_llm_model,
        inputs=[start_llm_model_repo, start_llm_model_file],
        outputs=[support_system_role, load_llm_model_log],
    ).then(
        fn=UiComponent.system_prompt,
        inputs=[support_system_role],
        outputs=[system_prompt],
        show_api=False,
    )
    interface.load(
        fn=UiFnModel.load_embed_model,
        inputs=[start_embed_model_repo],
        outputs=[load_embed_model_log],
    )
    interface.unload(UiFnModel.cleanup_models)


if __name__ == '__main__':
    interface.launch()
    # interface.launch(debug=True, show_error=True)  # debug=True, show_error=True, mcp_server=True
    # interface.launch(server_name='0.0.0.0', server_port=7860)