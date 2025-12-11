# ✅ CUDA 11.8 + cuDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ✅ CUDA libs dans le PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

WORKDIR /app

# ✅ Installer Python 3.10 et dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-distutils python3.10-venv \
    software-properties-common \
    swig \
    build-essential \
    libmupdf-dev \
    wget \
    tesseract-ocr \
    libglib2.0-0 \
    libgl1 \
    libglu1-mesa \
    libsm6 \
    libxrender1 \
    libxext6 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ✅ Définir python3.10 comme python3 par défaut
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# ✅ Installer pip pour Python 3.10
RUN wget https://bootstrap.pypa.io/get-pip.py \
 && python3 get-pip.py \
 && rm get-pip.py \
 && python3 -m pip install --upgrade pip

# ✅ Installer PaddlePaddle GPU compatible CUDA 11.8 / Python 3.10
RUN python3 -m pip install --no-cache-dir paddlepaddle-gpu==2.5.2 -f \
    https://www.paddlepaddle.org.cn/whl/mkl/stable.html

# ✅ Installer PaddleOCR (sans redéclencher les deps)
RUN python3 -m pip install --no-cache-dir --ignore-installed blinker \
 && python3 -m pip install --no-cache-dir paddleocr==2.7 --no-deps

# ✅ OpenCV headless (version stable)
RUN python3 -m pip install --no-cache-dir opencv-python-headless==4.6.0.66

# ✅ Installer numpy + scipy compatibles (avant le reste pour éviter les conflits)
RUN python3 -m pip install --no-cache-dir numpy==1.26.4 scipy==1.11.4

# ✅ Dépendances de l'application
COPY requirements.txt /app/

RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir --ignore-installed -r requirements.txt

# ✅ Pré-télécharger les modèles PaddleOCR (CPU à la build)
RUN mkdir -p /models/paddleocr && \
    python3 - << 'EOF'
from paddleocr import PaddleOCR

print("⬇ Downloading PaddleOCR models...")
PaddleOCR(
    lang="fr",
    use_gpu=False,
    use_angle_cls=True,
    det_model_dir="/models/paddleocr/det",
    rec_model_dir="/models/paddleocr/rec"
)
EOF

RUN chmod -R 777 /models

# ✅ Symlinks cuDNN / cuBLAS (au cas où)
RUN ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn.so || true && \
    ln -sf /usr/local/cuda/lib64/libcublas.so.11 /usr/local/cuda/lib64/libcublas.so || true

# ✅ Copier le code applicatif
COPY . /app/

# ✅ Commande de lancement (FastAPI + Gunicorn + UvicornWorker)
CMD ["gunicorn", "app:app", "-k", "uvicorn.workers.UvicornWorker", "--workers", "1", "--threads", "4", "--timeout", "300", "--bind", "0.0.0.0:8000"]
