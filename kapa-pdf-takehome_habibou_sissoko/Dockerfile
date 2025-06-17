FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
        && rm -rf /var/lib/apt/lists/*

COPY . /workspace

WORKDIR /workspace

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download YOLO model weights
RUN wget -O /workspace/doclayout_yolo_docstructbench_imgsz1024.pt \
    https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/resolve/main/doclayout_yolo_docstructbench_imgsz1024.pt

# Copy source, ui, and sample data
COPY src ./src
COPY app ./app
COPY data ./data

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
