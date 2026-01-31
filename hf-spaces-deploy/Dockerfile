FROM ghcr.io/sergey21000/chatbot-rag:main-hf

RUN useradd -m -u 1000 user \
    && chown -R user:user /opt/llama.cpp \
	&& chown -R user:user /app

USER user
ENV RUN_IN_DOCKER=0
#ENV HOME=/home/user \
#    PATH=/home/user/.local/bin:$PATH \
#    RUN_IN_DOCKER=0

WORKDIR /app

COPY --chown=user hf-spaces-deploy/config.py  .
COPY --chown=user hf-spaces-deploy/env.example .env

CMD ["python3", "app.py"]
