
## Gradio llama-cpp-python langchain Chatbot RAG


### Описание

Чат-бот с веб-интерфейсом на `Gradio` и механизмом RAG с использованием библиотек `llama-cpp-python` и `langchain`

<a href="https://hub.docker.com/r/sergey21000/chatbot-rag/screenshots">Скриншоты</a> интерфейса приложения

<div align="center">

![App interface](./screenshots/chatbot_rag_query2.png)
</div>


---
### Ссылки

<div align="center">
<a href="https://colab.research.google.com/github/sergey21000/hatbot-rag/blob/main/RAG_Chatbot_Gradio_llamacpp.ipynb"><img src="https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20" alt="Open in Colab"></a>
<a href="https://huggingface.co/spaces/sergey21000/chatbot-rag"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow" alt="Hugging Face Spaces"></a>
<a href="https://hub.docker.com/r/sergey21000/hatbot-rag"><img src="https://img.shields.io/badge/Docker-Hub-blue?logo=docker" alt="Docker Hub "></a>
</div>

В Google Colab ноутбуке находится код приложения с комментариями


### Функционал

#### Функционал LLM

- Обычная генерация ответа с использованием моделей в формате GGUF
- Настройка параметров генерации (`temperature`, `top_k`, `top_p`, `repetition_penalty`)
- Возможность указать системный промт (если модель его не поддерживает это будет отображено)
- Выбор количества учитываемых сообщений в истории при подаче запроса пользователя в модель (по умолчанию 0 - не учитывать предыдущую историю переписки)
- Выбор и загрузка LLM моделей в формате GGUF из репозиториев HuggingFace с индикацией прогресса загрузки и отображением размера файлов
- Отображение используемой ОЗУ и/или видео памяти и используемого места на диске с возможностью очистки от неиспользуемых  Embedding и LLM моделей

#### Функционал RAG

- Генерация ответа с использованием механизма RAG, переключение между режимом обычной генерации и RAG (для активации режима RAG необходимо загрузить документы на вкладке `Load documents`, тогда на странице `Chatbot` появится переключатель режимов)
- Выбор и загрузка Embedding моделей из репозиториев HuggingFace
- Просмотр загруженных текстов из документов
- Отображение полного запроса пользователя, обогащенного контекстом в режиме RAG
- Поддержка следующих форматов файлов для RAG - `csv doc docx html md pdf ppt pptx txt`, поддержка передачи ссылок на YouTube для загрузки субтитров к видео, выбор языка субтитров
- Настройки длины фрагментов текста и длины перекрытия между ними (в символах) (параметры `chunk_size` и `chunk_overlap`)
- Настройка количества релевантных фрагментов, которые будут обогощать запрос пользователя (параметр `k`)
- При установке `k` = `all` в контекст будут подаваться абсолютно все фрагменты, это может быть актуально при передачи небольших документов 
- Настройка порога для поиска похожих фрагментов текста на запрос пользователя (параметр `relevance_scores_threshold`, от 0 до 1, возвращать тексты которые похожи на запрос более чем на это число, чем больше тем меньше текстов будет находить)

Установить свой шаблон промта при условии контекста можно в переменной `CONTEXT_TEMPLATE` в модуле `config.py`  
При деплое на удаленных серверах их IP часто оказываются в черных списках YouTube, поэтому загрузка субтитров для всех видео с YouTube будет показывать статус `Invalid video url or current server IP is blocked for YouTube`. Подробнее в [обсуждении youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api/issues/303) (там же способы обхода, например через прокси)


### Стек

- [python](https://www.python.org/) >=3.8
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) для инференса моделей в формате GGUF
- [gradio](https://github.com/gradio-app/gradio) для написания веб-интерфейса
- [langchain](https://github.com/langchain-ai/langchain) для загрузки текста из файлов, разделения текстов на фрагменты и векторного хранилища (FAISS)
- [LLM Модель](https://huggingface.co/bartowski/gemma-2-2b-it-GGUF) `gemma-2-2b-it-Q8_0.gguf` в качестве языковой модели по умолчанию
- [Embedding Модель](https://huggingface.co/sergeyzh/rubert-tiny-turbo) `sergeyzh/rubert-tiny-turbo` в качестве Embedding модели по умолчанию


---
## **Установка и запуск через Python**

**1) Клонирование репозитория**  

```
git clone https://github.com/sergey21000/chatbot-rag.git
cd gradio-llamacpp-chatbot
```

**2) Установка зависимостей**  

С поддержкой CPU
```
pip install -r requirements-cpu.txt
```

С поддержкой CUDA 12.5
```
pip install -r requirements-cuda.txt
```
Инструкции по установке [llama-cpp-python](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation-configuration) и [torch](https://pytorch.org/get-started/locally/#start-locally) для других версий и систем

**3) Запуск сервера Gradio**  
```
python3 app.py
```
После запуска сервера перейти в браузере по адресу http://localhost:7860/  
При первом запуске произойдет загрузка модели LLM `gemma-2-2b-it-Q8_0.gguf` (2.7GB) в папку `./models`, а так же загрузка Embedding модели `sergeyzh/rubert-tiny-turbo` (117MB) в папку `./embed_models`, поэтому приложение доступно не сразу


---
## **Установка и запуск через Docker**

Для запуска приложения с поддержкой GPU CUDA необходима установка [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation).


### Запуск контейнера из готового образа Docker HUB

*С поддержкой CPU*
```
docker run -it -p 7860:7860 \
	-v ./embed_models:/app/embed_models \
	-v ./models:/app/models \
	--name chatbot-rag \
	sergey21000/chatbot-rag:cpu
```

*С поддержкой CUDA*
```
docker run -it --gpus all -p 7860:7860 \
	-v ./embed_models:/app/embed_models \
	-v ./models:/app/models \
	--name chatbot-rag \
	sergey21000/chatbot-rag:cuda
```


### Сборка своего образа

**1) Клонирование репозитория**  
```bash
git clone https://github.com/sergey21000/chatbot-rag.git
cd gradio-llamacpp-chatbot
```

**2) Сборка образа и запуск контейнера**

*С поддержкой CPU*

Сборка образа
```
docker build -t chatbot-rag:cpu -f Dockerfile-cpu .
```

Запуск контейнера
```
docker run -it -p 7860:7860 \
	-v ./embed_models:/app/embed_models \
	-v ./models:/app/models \
	--name chatbot-rag \
	chatbot-rag:cpu
```

*С поддержкой CUDA*

Сборка образа
```
docker build -t chatbot-rag:cuda -f Dockerfile-cuda .
```

Запуск контейнера
```
docker run -it --gpus all -p 7860:7860 \
	-v ./embed_models:/app/embed_models \
	-v ./models:/app/models \
	--name chatbot-rag \
	chatbot-rag:cuda
```
После запуска сервера перейти в браузере по адресу http://localhost:7860/  
При первом запуске произойдет загрузка модели LLM `gemma-2-2b-it-Q8_0.gguf` (2.7GB) в папку `./models`, а так же загрузка Embedding модели `sergeyzh/rubert-tiny-turbo` (117MB) в папку `./embed_models`, поэтому приложение доступно не сразу

---

Работоспособность приложения проверялась на Ubuntu 22.04 (python 3.10) и Windows 10 (python 3.12)  
Приложение создавалось для тестирования моделей с использованием RAG как любительский проект  
Оно написано для демонстрационных и образовательных целей и не предназначалось / не тестировалось для промышленного использования


## Лицензия

Этот проект лицензирован на условиях лицензии [Apache-2.0 license](./LICENSE).

