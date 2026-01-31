FROM ghcr.io/sergey21000/chatbot-rag:main-cpu

RUN useradd -m -u 1000 user \
    && chown -R user:user /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
	RUN_IN_DOCHER=0

WORKDIR /app
COPY --chown=user config.py .
COPY --chown=user env.example .env

CMD ["python3", "app.py"]