# RAG locale con LlamaIndex

Metti i tuoi file `.txt`, `.pdf` o `.docx` nella cartella `documenti`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Poi apri `.env` e inserisci la tua `OPENAI_API_KEY`.

## Uso

Ricrea l'indice:

```bash
python3 rag_cli.py indicizza
```

Fai una domanda singola:

```bash
python3 rag_cli.py chiedi "Di cosa parlano i documenti?"
```

Avvia la chat da terminale:

```bash
python3 rag_cli.py chat
```

## Webapp locale

Avvia la webapp:

```bash
uvicorn web_app:app --reload
```

Poi apri:

```text
http://127.0.0.1:8000
```

La webapp indicizza in modo incrementale i nuovi upload. Per i documenti gia
indicizzati prima dalla CLI, usa una volta il pulsante "Sincronizza tutto" per
creare il manifest usato dalla cancellazione file per file.

Se usi la CLI e aggiungi/modifichi manualmente file dentro `documenti/`, ricrea
l'indice con:

```bash
python3 rag_cli.py indicizza
```

Se invece carichi o elimini file dalla webapp, non serve reindicizzare tutto:
la webapp aggiorna solo il documento coinvolto.

## Docker

Docker serve a far girare la webapp dentro un container, cioe un ambiente Linux
isolato con Python e tutte le dipendenze gia installate. In questo modo la
stessa app puo partire anche su Windows senza ricreare manualmente virtualenv e
installazioni Python.

File Docker principali:

- `Dockerfile`: descrive come costruire l'immagine della webapp.
- `docker-compose.yml`: descrive come avviare il container, porta, `.env` e volumi.
- `.dockerignore`: evita di copiare nell'immagine file locali inutili o sensibili.

### Avvio con Docker

Assicurati di avere un file `.env` con:

```env
OPENAI_API_KEY=sk-...
```

Poi avvia:

```bash
docker compose up --build
```

Apri:

```text
http://127.0.0.1:8000
```

Per fermare:

```bash
docker compose down
```

### Perche usiamo i volumi

Nel `docker-compose.yml` ci sono questi volumi:

```yaml
volumes:
  - ./documenti:/app/documenti
  - ./storage:/app/storage
```

Significa:

- `documenti/` resta una cartella del tuo computer, non solo del container;
- `storage/` resta persistente anche se spegni o ricrei il container;
- upload, delete e indicizzazione continuano a funzionare senza perdere dati.

### Prova su Windows

Sul PC Windows:

1. Installa Docker Desktop.
2. Copia questa cartella progetto sul PC Windows.
3. Crea il file `.env` con la tua `OPENAI_API_KEY`.
4. Apri PowerShell nella cartella del progetto.
5. Esegui:

```powershell
docker compose up --build
```

Poi apri nel browser:

```text
http://localhost:8000
```
