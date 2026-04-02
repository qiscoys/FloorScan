# 使用官方 Python 瘦身镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统级依赖：Tesseract OCR 和 OpenCV 所需的底层库
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-chi-sim \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装 Python 包
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 暴露端口（Render 会自动注入 $PORT 环境变量）
EXPOSE 10000

# --preload  让所有 worker 共享一份已加载的内存，避免 OOM
# --timeout 300  给 OCR+OpenCV 足够的处理时间
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --timeout 300 --preload app:app"]