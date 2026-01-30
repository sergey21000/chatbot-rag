FROM python:3.12 AS builder

COPY requirements/requirements-base.txt requirements/requirements-cpu.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-cpu.txt


FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
	libmagic-dev \
	poppler-utils \
	libgl1 libglib2.0-0 \
	# tesseract-ocr \
	# tesseract-ocr-rus \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
	
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

WORKDIR /app
COPY modules modules
COPY app.py config.py .

ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860
CMD ["python3", "app.py"]