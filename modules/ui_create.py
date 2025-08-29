import gradio as gr

from config import Config, UiBlocksConfig
from modules.ui_fn import (
    UiFnService,
    UiFnModel,
    UiFnDb,
    UiFnChat,
)
from modules.ui_components import (
    UiChatbot,
    UiLoadTexts,
    UiLoadModel,
    UiViewTexts,
    UiUpdateComponent,
)


CONF = Config()
interface = gr.Blocks(**UiBlocksConfig.UI_BLOCKS_KWARGS)


with interface:
    config = gr.State(Config())
    texts = gr.State([])

    with gr.Tab(label='Chatbot'):
        ui_chatbot = UiChatbot()
        with gr.Row():
            with gr.Column(scale=3):
                ui_chatbot.chatbot.render()
                ui_chatbot.user_msg.render()
                with gr.Row():
                    ui_chatbot.user_msg_btn.render()
                    ui_chatbot.stop_btn.render()
                    ui_chatbot.clear_btn.render()

            with gr.Column(scale=1, min_width=80):
                with gr.Group():
                    gr.Markdown('History size')
                    ui_chatbot.history_len.render()
                    with gr.Group():
                        gr.Markdown('Show LLM thinking')
                        ui_chatbot.show_thinking.render()
                    with gr.Group():
                        url = 'https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion'
                        gr.Markdown('Generation [parameters]({url})')
                        ui_chatbot.do_sample.render()
                        ui_chatbot.temperature.render()
                        ui_chatbot.top_p.render()
                        ui_chatbot.top_k.render()
                        ui_chatbot.repeat_penalty.render()

        with gr.Group():
            gr.Markdown('RAG parameters')
            ui_chatbot.rag_mode.render()
            with gr.Row():
                ui_chatbot.n_results.render()
                ui_chatbot.max_distance_treshold.render()

        ui_chatbot.n_results.input(
            fn=lambda n_results: UiUpdateComponent.update_visibility(
                visible=n_results != 'all',
                num_componets=1,
            ),
            inputs=[ui_chatbot.n_results],
            outputs=[ui_chatbot.max_distance_treshold],
            show_progress=False,
            show_api=False,
        )
        sampling_args = ui_chatbot.get_matching_args(CONF.Inference.sampling_kwargs)
        ui_chatbot.do_sample.input(
            fn=lambda visible: UiUpdateComponent.update_visibility(
                visible=visible,
                num_componets=len(sampling_args),
            ),
            inputs=[ui_chatbot.do_sample],
            outputs=sampling_args,
            show_progress=False,
            show_api=False,
        )
        rag_args = ui_chatbot.get_matching_args(CONF.get_rag_kwargs()) + [ui_chatbot.user_msg_with_context]
        ui_chatbot.rag_mode.change(
            fn=lambda visible: UiUpdateComponent.update_visibility(
                visible=visible,
                num_componets=len(rag_args),
            ),
            inputs=[ui_chatbot.rag_mode],
            outputs=rag_args,
            show_api=False,
            show_progress=False,
        )

        with gr.Accordion('Prompt', open=True):
            ui_chatbot.system_prompt.render()
            ui_chatbot.context_template.render()
            ui_chatbot.user_msg_with_context.render()

        generation_kwargs = ui_chatbot.get_matching_kwargs(CONF.generation_kwargs)
        generation_args = list(generation_kwargs.values())
        generate_event = gr.on(
            triggers=[ui_chatbot.user_msg.submit, ui_chatbot.user_msg_btn.click],
            fn=UiFnChat.user_message_to_chatbot,
            inputs=[ui_chatbot.user_msg, ui_chatbot.chatbot],
            outputs=[ui_chatbot.user_msg, ui_chatbot.chatbot],
        ).then(
            fn=lambda config, *args: UiUpdateComponent.update_kwargs(
                config_kwargs=config.generation_kwargs,
                matching_kwargs=generation_kwargs,
                args=args,
            ),
            inputs=[config, *generation_args],
            outputs=None,
        ).then(
            fn=UiFnChat.update_user_msg_with_context,
            inputs=[ui_chatbot.chatbot, config],
            outputs=None,
        ).then(
            fn=UiUpdateComponent.view_user_msg_with_context,
            inputs=[config],
            outputs=[ui_chatbot.user_msg_with_context],
        ).then(
            fn=UiFnChat.yield_chatbot_with_llm_response,
            inputs=[ui_chatbot.chatbot, config],
            outputs=[ui_chatbot.chatbot],
        )
        ui_chatbot.stop_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=generate_event,
            queue=False,
            show_api=False,
        )
        ui_chatbot.clear_btn.click(
            fn=lambda: (None, ''),
            inputs=None,
            outputs=[ui_chatbot.chatbot, ui_chatbot.user_msg_with_context],
            queue=False,
            show_api=False,
        )


    with gr.Tab(label='Load Texts'):
        ui_load_texts = UiLoadTexts()
        with gr.Group():
            gr.Markdown('Uploading files and links')
            with gr.Row(variant='compact'):
                # загрузка файлов и ссылок
                with gr.Column(variant='compact'):
                    ui_load_texts.upload_files.render()
                with gr.Column(variant='compact'):
                    ui_load_texts.urls.render()
                    ui_load_texts.subtitle_lang.render()
        with gr.Group():
            url = 'https://docs.unstructured.io/open-source/core-functionality/partitioning#partition'
            gr.Markdown(f'Text loading [parameters]({url})')
            with gr.Row(variant='compact'):
                # параметры нарезки текста на фрагменты
                ui_load_texts.chunking_strategy.render()
                ui_load_texts.max_characters.render()
                ui_load_texts.new_after_n_chars.render()
                ui_load_texts.overlap.render()
        with gr.Group():
            url = 'https://docs.unstructured.io/open-source/core-functionality/cleaning#clean'
            gr.Markdown(f'Text cleaning [parameters]({url})')
            with gr.Row(variant='compact'):
                # параметры очистки текста
                ui_load_texts.clean.render()
                ui_load_texts.bullets.render()
                ui_load_texts.extra_whitespace.render()
                ui_load_texts.dashes.render()
                ui_load_texts.trailing_punctuation.render()
                ui_load_texts.lowercase.render()

        clean_args = ui_load_texts.get_matching_args(CONF.get_clean_kwargs())
        ui_load_texts.clean.change(
            fn=lambda visible: UiUpdateComponent.update_visibility(
                visible=visible,
                num_componets=len(clean_args),
            ),
            inputs=[ui_load_texts.clean],
            outputs=clean_args,
            show_progress=False,
            show_api=False,
        )
        ui_load_texts.load_texts_btn.render()
        ui_load_texts.load_texts_log.render()
        load_text_kwargs = ui_load_texts.get_matching_kwargs(CONF.load_text_kwargs)
        load_text_args = list(load_text_kwargs.values())

        ui_load_texts.load_texts_btn.click(
            fn=lambda config, *args: UiUpdateComponent.update_kwargs(
                config_kwargs=config.load_text_kwargs,
                matching_kwargs=load_text_kwargs,
                args=args,
            ),
            inputs=[config, *load_text_args],
            outputs=None,
        ).then(
            fn=UiFnDb.load_texts_and_create_db,
            inputs=[ui_load_texts.upload_files, ui_load_texts.urls, config],
            outputs=[texts, ui_load_texts.load_texts_log],
        ).success(
            fn=UiUpdateComponent.update_rag_mode_if_db_exists,
            inputs=None,
            outputs=ui_chatbot.rag_mode,
            show_api=False,
        ).success(
            fn=lambda visible: UiUpdateComponent.update_visibility(
                visible=visible,
                num_componets=len(rag_args),
            ),
            inputs=[ui_chatbot.rag_mode],
            outputs=rag_args,
            show_api=False,
            show_progress=False,
        )


    with gr.Tab(label='View Texts'):
        ui_view_texts = UiViewTexts()
        ui_view_texts.max_lines_text_view.render()
        ui_view_texts.view_texts_btn.render()
        ui_view_texts.view_texts_textbox.render()
        view_text_kwargs = ui_view_texts.get_matching_kwargs(CONF.view_text_kwargs)
        view_text_args = list(view_text_kwargs.values())
        ui_view_texts.view_texts_btn.click(
            fn=lambda config, *args: UiUpdateComponent.update_kwargs(
                config_kwargs=config.view_text_kwargs,
                matching_kwargs=view_text_kwargs,
                args=args,
            ),
            inputs=[config, *view_text_args],
            outputs=None,
        ).then(
            fn=UiUpdateComponent.view_texts,
            inputs=[texts, ui_view_texts.max_lines_text_view],
            outputs=[ui_view_texts.view_texts_textbox],
        )


    with gr.Tab('Load models'):
        ui_load_model = UiLoadModel()
        ui_load_model.new_llm_model_repo.render()
        ui_load_model.new_llm_model_repo_btn.render()
        ui_load_model.llm_model_repo.render()
        ui_load_model.llm_model_file.render()
        with gr.Group():
            url = 'https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__'
            gr.Markdown('Model initialization [parameters]({url})')
            with gr.Row(variant='compact'):
                ui_load_model.n_gpu_layers.render()
                ui_load_model.n_ctx.render()
            ui_load_model.load_llm_model_btn.render()
        ui_load_model.load_llm_model_log.render()
        with gr.Group():
            gr.Markdown('Free up disk space by deleting all models except the currently selected one')
            ui_load_model.clear_llm_folder_btn.render()
        ui_load_model.new_llm_model_repo_btn.click(
            fn=lambda model_repo: UiFnModel.add_new_model_repo(
                new_model_repo=model_repo,
                model_repos=config.value.Repos.llm_model_repos,
            ),
            inputs=[ui_load_model.new_llm_model_repo],
            outputs=[ui_load_model.llm_model_repo, ui_load_model.load_llm_model_log],
            show_api=False,
        ).success(
            fn=lambda: gr.update(value=''),
            inputs=None,
            outputs=[ui_load_model.new_llm_model_repo],
            show_api=False,
        )
        ui_load_model.llm_model_repo.input(
            fn=UiFnModel.get_gguf_model_names,
            inputs=[ui_load_model.llm_model_repo],
            outputs=[ui_load_model.llm_model_file],
        )
        load_model_kwargs = ui_load_model.get_matching_kwargs(CONF.load_model_kwargs)
        load_model_args = list(load_model_kwargs.values())
        ui_load_model.load_llm_model_btn.click(
            fn=lambda config, *args: UiUpdateComponent.update_kwargs(
                config_kwargs=config.load_model_kwargs,
                matching_kwargs=load_model_kwargs,
                args=args,
            ),
            inputs=[config, *load_model_args],
            outputs=None,
        ).then(
            fn=UiFnModel.load_llm_model,
            inputs=[config],
            outputs=[ui_load_model.load_llm_model_log],
            show_api=False,
        ).then(
            fn=lambda log: log + UiFnService.get_memory_usage(),
            inputs=[ui_load_model.load_llm_model_log],
            outputs=[ui_load_model.load_llm_model_log],
            show_api=False,
        ).success(
            fn=UiUpdateComponent.update_system_prompt,
            inputs=[config],
            outputs=[ui_chatbot.system_prompt],
            show_api=False,
        )
        ui_load_model.clear_llm_folder_btn.click(
            fn=UiFnModel.clear_llm_folder,
            inputs=[ui_load_model.llm_model_file],
            outputs=None,
        )


    with gr.Tab('Load embed model'):
        ui_load_model.new_embed_model_repo.render()
        ui_load_model.new_embed_model_repo_btn.render()
        ui_load_model.embed_model_repo.render()
        ui_load_model.load_embed_model_btn.render()
        ui_load_model.load_embed_model_log.render()
        with gr.Group():
            gr.Markdown('Free up disk space by deleting all models except the currently selected one')
            ui_load_model.clear_embed_folder_btn.render()
        ui_load_model.new_embed_model_repo_btn.click(
            fn=lambda model_repo: UiFnModel.add_new_model_repo(
                new_model_repo=model_repo,
                model_repos=config.value.Repos.llm_model_repos,
            ),
            inputs=[ui_load_model.new_embed_model_repo],
            outputs=[ui_load_model.embed_model_repo, ui_load_model.load_embed_model_log],
            show_api=False,
        ).success(
            fn=lambda: gr.update(value=''),
            inputs=None,
            outputs=[ui_load_model.new_embed_model_repo],
            show_api=False,
        )
        ui_load_model.load_embed_model_btn.click(
            fn=lambda config, *args: UiUpdateComponent.update_kwargs(
                config_kwargs=config.load_model_kwargs,
                matching_kwargs=load_model_kwargs,
                args=args,
            ),
            inputs=[config, *load_model_args],
            outputs=None,
        ).then(
            fn=UiFnModel.load_embed_model,
            inputs=[config],
            outputs=[ui_load_model.load_embed_model_log],
            show_api=False,
        ).success(
            fn=lambda log: log + UiFnService.get_memory_usage(),
            inputs=[ui_load_model.load_embed_model_log],
            outputs=[ui_load_model.load_embed_model_log],
            show_api=False,
        )
        ui_load_model.clear_embed_folder_btn.click(
            fn=UiFnModel.clear_embed_folder,
            inputs=[ui_load_model.embed_model_repo],
            outputs=None,
        )

    interface.load(
        fn=lambda: gr.Info('Loading LLM and Embeddings models, the status is displayed on the Load models tabs'),
        inputs=None,
        outputs=None,
        show_api=False,
    ).then(
        fn=UiFnModel.load_llm_model,
        inputs=[config],
        outputs=[ui_load_model.load_llm_model_log],
    ).success(
        fn=UiUpdateComponent.update_system_prompt,
        inputs=[config],
        outputs=[ui_chatbot.system_prompt],
        show_api=False,
    )
    interface.load(
        fn=UiFnModel.load_embed_model,
        inputs=[config],
        outputs=[ui_load_model.load_embed_model_log],
    )
    interface.unload(UiFnService.cleanup_storage)
