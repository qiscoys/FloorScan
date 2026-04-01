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

# 暴露端口
EXPOSE 5000

# 使用 gunicorn 启动 Flask 应用
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--timeout", "120"]