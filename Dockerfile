# Dockerfile
FROM python:3.11-slim

# Installer les dépendances système minimales
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    poppler-utils \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Créer un répertoire de travail
WORKDIR /app

# Copier d'abord les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances Python (optimisé pour le cache Docker)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le reste du code
COPY api.py main.py run.py ./

# Créer le répertoire static
RUN mkdir -p static

# Copier l'interface utilisateur
COPY index.html static/

# Exposer le port
EXPOSE 8000

# Variable d'environnement pour optimiser PaddleOCR
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Commande de démarrage
CMD ["python", "run.py"]