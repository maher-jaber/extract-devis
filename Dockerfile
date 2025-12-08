# Dockerfile
FROM python:3.11-slim

# Dépendances système minimales
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Copier uniquement les fichiers essentiels
COPY requirements.txt .
COPY api.py .
COPY main.py .

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Exposer le port
EXPOSE 8000

# Commande de démarrage
CMD ["python", "api.py"]