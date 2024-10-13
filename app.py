from typing import List, Optional

import gradio as gr
from langchain_core.vectorstores import VectorStore

from config import (
    LLM_MODEL_REPOS,
    EMBED_MODEL_REPOS,
    SUBTITLES_LANGUAGES,
    GENERATE_KWARGS,
)

from utils import (
    load_llm_model,
    load_embed_model,
    load_documents_and_create_db,
    user_message_to_chatbot,
    update_user_message_with_context,
    get_llm_response,
    get_gguf_model_names,
    add_new_model_repo,
    clear_llm_folder,
    clear_embed_folder,
    get_memory_usage,
)


# ============ INTERFACE COMPONENT INITIALIZATION FUNCS ============

def get_rag_settings(rag_mode: bool, render: bool = True):
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


def get_user_message_with_context(text: str, rag_mode: bool) -> gr.component:
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


def get_system_prompt_component(interactive: bool) -> gr.Textbox:
    value = '' if interactive else 'System prompt is not supported by this model'
    return gr.Textbox(value=value, label='System prompt', interactive=interactive)


def get_generate_args(do_sample: bool) -> List[gr.component]:
    generate_args = [
        gr.Slider(minimum=0.1, maximum=3, value=GENERATE_KWARGS['temperature'], step=0.1, label='temperature', visible=do_sample),
        gr.Slider(minimum=0, maximum=1, value=GENERATE_KWARGS['top_p'], step=0.01, label='top_p', visible=do_sample),
        gr.Slider(minimum=1, maximum=50, value=GENERATE_KWARGS['top_k'], step=1, label='top_k', visible=do_sample),
        gr.Slider(minimum=1, maximum=5, value=GENERATE_KWARGS['repeat_penalty'], step=0.1, label='repeat_penalty', visible=do_sample),
    ]
    return generate_args


def get_rag_mode_component(db: Optional[VectorStore]) -> gr.Checkbox:
    value = visible = db is not None
    return gr.Checkbox(value=value, label='RAG Mode', scale=1, visible=visible)


# ================ LOADING AND INITIALIZING MODELS ========================

start_llm_model, start_support_system_role, load_log = load_llm_model(LLM_MODEL_REPOS[0], 'gemma-2-2b-it-Q8_0.gguf')
start_embed_model, load_log = load_embed_model(EMBED_MODEL_REPOS[0])



# ================== APPLICATION WEB INTERFACE ============================

css = '''.gradio-container {width: 60% !important}'''

with gr.Blocks(css=css) as interface:

    # ==================== GRADIO STATES ===============================

    documents = gr.State([])
    db = gr.State(None)
    user_message_with_context = gr.State('')
    support_system_role = gr.State(start_support_system_role)
    llm_model_repos = gr.State(LLM_MODEL_REPOS)
    embed_model_repos = gr.State(EMBED_MODEL_REPOS)
    llm_model = gr.State(start_llm_model)
    embed_model = gr.State(start_embed_model)



    # ==================== BOT PAGE =================================

    with gr.Tab(label='Chatbot'):
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    type='messages',  # new in gradio 5+
                    show_copy_button=True,
                    bubble_full_width=False,
                    height=480,
                )
                user_message = gr.Textbox(label='User')

                with gr.Row():
                    user_message_btn = gr.Button('Send')
                    stop_btn = gr.Button('Stop')
                    clear_btn = gr.Button('Clear')

            # ------------- GENERATION PARAMETERS -------------------

            with gr.Column(scale=1, min_width=80):
                with gr.Group():
                    gr.Markdown('History size')
                    history_len = gr.Slider(
                        minimum=0,
                        maximum=5,
                        value=0,
                        step=1,
                        info='Number of previous messages taken into account in history',
                        label='history_len',
                        show_label=False,
                        )

                    with gr.Group():
                        gr.Markdown('Generation parameters')
                        do_sample = gr.Checkbox(
                            value=False,
                            label='do_sample',
                            info='Activate random sampling',
                            )
                        generate_args = get_generate_args(do_sample.value)
                        do_sample.change(
                            fn=get_generate_args,
                            inputs=do_sample,
                            outputs=generate_args,
                            show_progress=False,
                            )

        rag_mode = get_rag_mode_component(db=db.value)
        k, score_threshold = get_rag_settings(rag_mode=rag_mode.value, render=False)
        rag_mode.change(
            fn=get_rag_settings,
            inputs=[rag_mode],
            outputs=[k, score_threshold],
            )
        with gr.Row():
            k.render()
            score_threshold.render()

        # ---------------- SYSTEM PROMPT AND USER MESSAGE -----------

        with gr.Accordion('Prompt', open=True):
            system_prompt = get_system_prompt_component(interactive=support_system_role.value)
            user_message_with_context = get_user_message_with_context(text='', rag_mode=rag_mode.value)

        # ---------------- SEND, CLEAR AND STOP BUTTONS ------------

        generate_event = gr.on(
            triggers=[user_message.submit, user_message_btn.click],
            fn=user_message_to_chatbot,
            inputs=[user_message, chatbot],
            outputs=[user_message, chatbot],
            queue=False,
        ).then(
            fn=update_user_message_with_context,
            inputs=[chatbot, rag_mode, db, k, score_threshold],
            outputs=[user_message_with_context],
        ).then(
            fn=get_user_message_with_context,
            inputs=[user_message_with_context, rag_mode],
            outputs=[user_message_with_context],
        ).then(
            fn=get_llm_response,
            inputs=[chatbot, llm_model, user_message_with_context, rag_mode, system_prompt,
                    support_system_role, history_len, do_sample, *generate_args],
            outputs=[chatbot],
        )

        stop_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=generate_event,
            queue=False,
        )

        clear_btn.click(
            fn=lambda: (None, ''),
            inputs=None,
            outputs=[chatbot, user_message_with_context],
            queue=False,
            )



    # ================= FILE DOWNLOAD PAGE =========================

    with gr.Tab(label='Load documents'):
        with gr.Row(variant='compact'):
            upload_files = gr.File(file_count='multiple', label='Loading text files')
            web_links = gr.Textbox(lines=6, label='Links to Web sites or YouTube')

        with gr.Row(variant='compact'):
            chunk_size = gr.Slider(50, 2000, value=500, step=50, label='Chunk size')
            chunk_overlap = gr.Slider(0, 200, value=20, step=10, label='Chunk overlap')

            subtitles_lang = gr.Radio(
                SUBTITLES_LANGUAGES,
                value=SUBTITLES_LANGUAGES[0],
                label='YouTube subtitle language',
                )

        load_documents_btn = gr.Button(value='Upload documents and initialize database')
        load_docs_log = gr.Textbox(label='Status of loading and splitting documents', interactive=False)

        load_documents_btn.click(
            fn=load_documents_and_create_db,
            inputs=[upload_files, web_links, subtitles_lang, chunk_size, chunk_overlap, embed_model],
            outputs=[documents, db, load_docs_log],
        ).success(
            fn=get_rag_mode_component,
            inputs=[db],
            outputs=[rag_mode],
        )

        gr.HTML("""<h3 style='text-align: center'>
        <a href="https://github.com/sergey21000/chatbot-rag" target='_blank'>GitHub Repository</a></h3>
        """)



    # ================= VIEW PAGE FOR ALL DOCUMENTS =================

    with gr.Tab(label='View documents'):
        view_documents_btn = gr.Button(value='Show downloaded text chunks')
        view_documents_textbox = gr.Textbox(
            lines=1,
            placeholder='To view chunks, load documents in the Load documents tab',
            label='Uploaded chunks',
            )
        sep = '=' * 20
        view_documents_btn.click(
            lambda documents: f'\n{sep}\n\n'.join([doc.page_content for doc in documents]),
            inputs=[documents],
            outputs=[view_documents_textbox],
        )


    # ============== GGUF MODELS DOWNLOAD PAGE =====================

    with gr.Tab('Load LLM model'):
        new_llm_model_repo = gr.Textbox(
            value='',
            label='Add repository',
            placeholder='Link to repository of HF models in GGUF format',
            )
        new_llm_model_repo_btn = gr.Button('Add repository')
        curr_llm_model_repo = gr.Dropdown(
            choices=LLM_MODEL_REPOS,
            value=None,
            label='HF Model Repository',
            )
        curr_llm_model_path = gr.Dropdown(
            choices=[],
            value=None,
            label='GGUF model file',
            )
        load_llm_model_btn = gr.Button('Loading and initializing model')
        load_llm_model_log = gr.Textbox(
            value=f'Model {LLM_MODEL_REPOS[0]} loaded at application startup',
            label='Model loading status',
            lines=6,
            )

        with gr.Group():
            gr.Markdown('Free up disk space by deleting all models except the currently selected one')
            clear_llm_folder_btn = gr.Button('Clear folder')

        new_llm_model_repo_btn.click(
            fn=add_new_model_repo,
            inputs=[new_llm_model_repo, llm_model_repos],
            outputs=[curr_llm_model_repo, load_llm_model_log],
        ).success(
            fn=lambda: '',
            inputs=None,
            outputs=[new_llm_model_repo],
        )

        curr_llm_model_repo.change(
            fn=get_gguf_model_names,
            inputs=[curr_llm_model_repo],
            outputs=[curr_llm_model_path],
        )

        load_llm_model_btn.click(
            fn=load_llm_model,
            inputs=[curr_llm_model_repo, curr_llm_model_path],
            outputs=[llm_model, support_system_role, load_llm_model_log],
            queue=True,
        ).success(
            fn=lambda log: log + get_memory_usage(),
            inputs=[load_llm_model_log],
            outputs=[load_llm_model_log],
        ).then(
            fn=get_system_prompt_component,
            inputs=[support_system_role],
            outputs=[system_prompt],
        )

        clear_llm_folder_btn.click(
            fn=clear_llm_folder,
            inputs=[curr_llm_model_path],
            outputs=None,
        ).success(
            fn=lambda model_path: f'Models other than {model_path} removed',
            inputs=[curr_llm_model_path],
            outputs=None,
        )


    # ============== EMBEDDING MODELS DOWNLOAD PAGE =============

    with gr.Tab('Load embed model'):
        new_embed_model_repo = gr.Textbox(
            value='',
            label='Add repository',
            placeholder='Link to HF model repository',
            )
        new_embed_model_repo_btn = gr.Button('Add repository')
        curr_embed_model_repo = gr.Dropdown(
            choices=EMBED_MODEL_REPOS,
            value=None,
            label='HF model repository',
            )

        load_embed_model_btn = gr.Button('Loading and initializing model')
        load_embed_model_log = gr.Textbox(
            value=f'Model {EMBED_MODEL_REPOS[0]} loaded at application startup',
            label='Model loading status',
            lines=7,
            )
        with gr.Group():
            gr.Markdown('Free up disk space by deleting all models except the currently selected one')
            clear_embed_folder_btn = gr.Button('Clear folder')

        new_embed_model_repo_btn.click(
            fn=add_new_model_repo,
            inputs=[new_embed_model_repo, embed_model_repos],
            outputs=[curr_embed_model_repo, load_embed_model_log],
        ).success(
            fn=lambda: '',
            inputs=None,
            outputs=new_embed_model_repo,
        )

        load_embed_model_btn.click(
            fn=load_embed_model,
            inputs=[curr_embed_model_repo],
            outputs=[embed_model, load_embed_model_log],
        ).success(
            fn=lambda log: log + get_memory_usage(),
            inputs=[load_embed_model_log],
            outputs=[load_embed_model_log],
        )

        clear_embed_folder_btn.click(
            fn=clear_embed_folder,
            inputs=[curr_embed_model_repo],
            outputs=None,
        ).success(
            fn=lambda model_repo: f'Models other than {model_repo} removed',
            inputs=[curr_embed_model_repo],
            outputs=None,
        )


interface.launch(server_name='0.0.0.0', server_port=7860)  # debug=True
