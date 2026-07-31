FROM python:3.11-slim

WORKDIR /app

# Dépendances système utiles (compilation de certains packages, healthcheck, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python en premier (meilleur cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/", \
    "--server.port=8501", \
    "--server.address=0.0.0.0"]