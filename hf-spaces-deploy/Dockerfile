# ---------- stage 1 ----------
FROM ghcr.io/ggml-org/llama.cpp:server AS llama

# ---------- stage 2 ----------
FROM ghcr.io/sergey21000/chatbot-rag:main-cpu

RUN useradd -m -u 1000 user \
    && chown -R user:user /app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
	RUN_IN_DOCHER=0

WORKDIR /app

COPY --from=llama /app /opt/llama.cpp
COPY --chown=user config.py .
COPY --chown=user env.example .env

ENV LD_LIBRARY_PATH=/opt/llama.cpp:$LD_LIBRARY_PATH
ENV LLAMACPP_DIR=/opt/llama.cpp
ENV PATH=/opt/llama.cpp:$PATH
ENV RUN_IN_DOCKER=0

CMD ["python3", "app.py"]