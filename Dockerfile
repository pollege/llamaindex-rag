# Immagine base: Python gia installato su Linux.
# Usiamo 3.12 per stare su una versione stabile e ben supportata dalle librerie.
FROM python:3.12-slim

# Evita che Python scriva file .pyc e forza log immediati nel terminale Docker.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cartella di lavoro dentro al container.
# Tutti i comandi successivi partono da qui.
WORKDIR /app

# Copiamo prima requirements.txt per sfruttare la cache Docker:
# se cambi solo il codice ma non le dipendenze, Docker non reinstalla tutto.
COPY requirements.txt .

# Installa le librerie Python necessarie alla webapp.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia il codice dell'app dentro al container.
# .dockerignore decide cosa NON entra nell'immagine.
COPY . .

# Documenta la porta usata dall'app.
# Non pubblica la porta da solo: la pubblicazione si fa con docker compose o -p.
EXPOSE 8000

# Comando avviato quando parte il container.
# 0.0.0.0 e fondamentale in Docker: rende il server raggiungibile dall'host.
CMD ["uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8000"]
