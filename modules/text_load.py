import gradio as gr
from loguru import logger
from requests.exceptions import MissingSchema

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from youtube_transcript_api.formatters import TextFormatter

from unstructured.partition.auto import partition
from unstructured.partition.text import partition_text
from unstructured.chunking.basic import chunk_elements
from unstructured.cleaners.core import clean
from unstructured.documents.elements import Element

from config import Config


class YouTubeSubLoader:
    '''YouTube subtitle downloader'''
    @staticmethod
    def get_subtitle_text_from_yt_video(
        yt_video_url: str,
        subtitle_lang: str,
    ) -> tuple[str | None, str]:
        """
        Returns text subtitles (manual or automatic) as a string.
        If there are no subtitles, returns None.
        """
        load_log = ''
        subtitle_text = None
        video_id = yt_video_url.split('watch?v=')[-1].split('&')[0].split('live/')[-1].split('&')[0]
        if len(video_id) != 11:
            load_log += f'Invalid video url ({yt_video_url}): Video ID ({video_id}) must be 11 characters long\n'
            return subtitle_text, load_log
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.list(video_id)
            try:
                transcript = transcript_list.find_transcript([subtitle_lang])
                if transcript.is_generated:
                    load_log += f'Automatic subtitles will be loaded, manual subtitles are not available for video {yt_video_url}\n'
                else:
                    load_log += f'Manual subtitles will be downloaded for the video {yt_video_url}\n'
                fetched_transcript = transcript.fetch()
                formatter = TextFormatter()
                subtitle_text = formatter.format_transcript(fetched_transcript)
            except NoTranscriptFound:
                load_log += f'Subtitle language [{subtitle_lang}] is not available for video {yt_video_url}\n'
        except TranscriptsDisabled:
            load_log += f'There are no subtitles for the video {yt_video_url}'
        except Exception as ex:
            load_log += f'Error loading subtitles for video ({yt_video_url}): {ex}\n'
        return subtitle_text, load_log


class TextLoader:
    '''Text downloader from files and links'''
    @staticmethod
    def elements_to_texts(elements: list[Element]) -> list[str]:
        texts = []
        for el in elements:
            texts.append(str(el) + '\n')
        return texts


    @classmethod
    def load_texts(cls, file_or_url: str, config: Config, is_url: bool = False) -> list[str]:
        partition_kwargs_key = 'filename' if not is_url else 'url'
        partition_kwargs = {partition_kwargs_key: file_or_url, **config.get_partition_kwargs()}
        elements = partition(**partition_kwargs)
        texts = cls.elements_to_texts(elements)
        return texts


    @classmethod
    def chunking_text(cls, text: str, config: Config):
        KWARGS = config.get_partition_kwargs()
        if KWARGS['clean']:
            text = clean(text=text, **config.get_clean_kwargs())
        elements = partition_text(text=text)
        chunks = chunk_elements(elements=elements, **config.get_chunking_kwargs())
        texts = cls.elements_to_texts(chunks)
        return texts


    @classmethod
    def load_texts_from_files(
        cls,
        upload_files: list[str],
        config: Config,
    ) -> tuple[list[str], str]:
        load_log = ''
        texts = []
        for upload_file in upload_files:
            file_extension = f".{upload_file.split('.')[-1]}"
            if file_extension in config.TextLoad.SUPPORTED_FILE_EXTS:
                try:
                    loaded_texts = cls.load_texts(file_or_url=upload_file, config=config)
                    texts.extend(loaded_texts)
                except Exception as ex:
                    msg = f'Error loading text from file {upload_file}: {ex}\n'
                    load_log += msg
                    logger.error(msg)
                    continue
            else:
                load_log += f'Unsupported file format {upload_file}\n'
                continue
        return texts, load_log


    @classmethod
    def load_texts_from_urls(
        cls,
        urls: str,
        config: Config,
    ) -> tuple[list[str], str]:
        load_log = ''
        texts = []
        urls = [url.strip() for url in urls.split() if url.strip()]
        for url in urls:
            if 'youtube.com' in url:
                loaded_text, log = YouTubeSubLoader.get_subtitle_text_from_yt_video(
                    yt_video_url=url,
                    subtitle_lang=config.load_text_kwargs['subtitle_lang'],
                )
                load_log += log
                if loaded_text:
                    loaded_texts = cls.chunking_text(text=loaded_text, config=config)
                    texts.extend(loaded_texts)
                continue
            try:
                loaded_texts = cls.load_texts(
                    file_or_url=url,
                    config=config,
                    is_url=True,
                )
                if len(loaded_texts) == 0:
                    load_log += f'No text chunks were found at the url: {url}\n'
                    continue
                texts.extend(loaded_texts)
            except MissingSchema:
                load_log += f'Invalid url: {url}\n'
            except Exception as ex:
                load_log += f'Error loading data by web loader at url: {url}: {ex}\n'
        return texts, load_log


    @classmethod
    def load_texts_from_files_and_urls(
        cls,
        upload_files: list[str] | None,
        urls: str | None,
        config: Config,
    ) -> tuple[list[str], str]:
        load_log = ''
        all_texts = []
        progress = gr.Progress()
        if not upload_files and not urls:
            load_log += 'No files or links selected'
            return all_texts, load_log
        if upload_files:
            progress(0.3, desc='Step 1/2: Upload texts from files')
            texts, log = cls.load_texts_from_files(upload_files=upload_files, config=config)
            all_texts.extend(texts)
            load_log += log
        if urls:
            progress(0.3 if upload_files is None else 0.5, desc='Step 1/2: Upload texts via links')
            texts, log = cls.load_texts_from_urls(urls=urls, config=config)
            all_texts.extend(texts)
            load_log += log
        if len(all_texts) == 0:
            load_log += 'Download was interrupted because no texts were extracted\n'
            load_log += 'RAG mode cannot be activated'
            return all_texts, load_log
        load_log += f'Number of loaded text chunks: {len(all_texts)}\n'

        return all_texts, load_log

