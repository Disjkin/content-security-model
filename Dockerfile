FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
WORKDIR /app

COPY --from=builder /install /usr/local

# 复制应用代码
COPY app/ ./app/
COPY sensitive_words/ ./sensitive_words/
COPY training/ ./training/
COPY config.yaml run.py ./

# 创建数据目录
RUN mkdir -p data logs models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "run.py"]
