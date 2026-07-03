"""
Webapp locale RAG con LlamaIndex + OpenAI.

Questa app aggiunge una interfaccia web al progetto CLI:
- carichi file .txt, .pdf e .docx dal browser;
- ogni file viene indicizzato appena caricato;
- puoi eliminare un file e rimuovere solo i suoi chunk dall'indice;
- fai domande ai documenti con risposta in streaming.

Perche esiste manifest.json?
LlamaIndex permette di cancellare documenti dall'indice tramite "ref_doc_id".
Pero un singolo PDF puo generare piu Document interni, ad esempio una pagina
per volta. Per fare delete incrementale in modo preciso salviamo quindi una
mappa:

    nome_file -> lista dei ref_doc_id creati per quel file

Questa mappa vive in "storage/manifest.json" ed e il pezzo che rende possibile
eliminare o aggiornare un solo documento senza reindicizzare tutto.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from openai import OpenAI as OpenAIClient
from pydantic import BaseModel


PROJECT_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_DIR / "documenti"
STORAGE_DIR = PROJECT_DIR / "storage"
STATIC_DIR = PROJECT_DIR / "static"
MANIFEST_PATH = STORAGE_DIR / "manifest.json"
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}
LLM_MODEL = "gpt-5.4-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


app = FastAPI(title="RAG locale LlamaIndex")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    """Payload JSON ricevuto dall'endpoint /api/ask."""

    question: str


def configure_openai() -> None:
    """Configura LlamaIndex per usare OpenAI.

    E la stessa idea dello script CLI:
    - carichiamo OPENAI_API_KEY da .env;
    - impostiamo il modello LLM per generare risposte;
    - impostiamo il modello embedding per creare e confrontare vettori.
    """

    load_dotenv(PROJECT_DIR / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Manca OPENAI_API_KEY. Crea un file .env e inserisci la tua API key."
        )

    Settings.llm = LlamaIndexOpenAI(model=LLM_MODEL)
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 120


def ensure_dirs() -> None:
    """Crea le cartelle necessarie se non esistono."""

    DOCUMENTS_DIR.mkdir(exist_ok=True)
    STORAGE_DIR.mkdir(exist_ok=True)


def load_manifest() -> dict[str, Any]:
    """Legge manifest.json.

    Se il file non esiste, restituiamo una struttura vuota.
    Questo succede al primo avvio della webapp oppure se l'indice era stato
    creato in passato dalla CLI, che non aveva bisogno del manifest.
    """

    if not MANIFEST_PATH.exists():
        return {"files": {}}

    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_manifest(manifest: dict[str, Any]) -> None:
    """Salva manifest.json in modo leggibile."""

    STORAGE_DIR.mkdir(exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def safe_filename(filename: str) -> str:
    """Riduce il nome file alla sola parte finale.

    Serve a evitare path come "../../qualcosa".
    Per una webapp locale e gia abbastanza, per una app pubblica si farebbero
    controlli piu severi.
    """

    name = Path(filename).name.strip()

    if not name:
        raise HTTPException(status_code=400, detail="Nome file non valido.")

    if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Formato non supportato. Usa .txt, .pdf oppure .docx.",
        )

    return name


def load_index() -> VectorStoreIndex:
    """Carica l'indice persistente o ne crea uno vuoto.

    VectorStoreIndex([]) crea un indice senza documenti.
    Lo usiamo quando la webapp parte per la prima volta e non esiste ancora
    nulla in storage/.
    """

    if (STORAGE_DIR / "index_store.json").exists():
        storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
        return load_index_from_storage(storage_context)

    index = VectorStoreIndex([])
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    return index


def persist_index(index: VectorStoreIndex) -> None:
    """Scrive su disco lo stato aggiornato dell'indice."""

    index.storage_context.persist(persist_dir=str(STORAGE_DIR))


def read_file_as_documents(file_path: Path, file_id: str) -> list:
    """Legge un singolo file e assegna doc_id stabili ai Document LlamaIndex.

    SimpleDirectoryReader puo restituire piu Document per un solo file.
    Per esempio un PDF puo essere letto pagina per pagina.

    Noi assegniamo id del tipo:

        nome.pdf::0
        nome.pdf::1
        nome.pdf::2

    Questi id vengono salvati nel manifest e poi usati da delete_ref_doc.
    """

    documents = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()

    for i, document in enumerate(documents):
        document.id_ = f"{file_id}::{i}"
        document.metadata = {
            **(document.metadata or {}),
            "uploaded_file": file_id,
            "file_name": file_id,
        }

    return documents


def delete_file_from_index(index: VectorStoreIndex, filename: str) -> bool:
    """Rimuove dall'indice solo i Document collegati a un file.

    Ritorna True se il file era presente nel manifest, False altrimenti.
    Se il file non e nel manifest non sappiamo con certezza quali ref_doc_id
    cancellare, quindi evitiamo cancellazioni approssimative.
    """

    manifest = load_manifest()
    record = manifest["files"].get(filename)

    if not record:
        return False

    for ref_doc_id in record.get("ref_doc_ids", []):
        index.delete_ref_doc(ref_doc_id, delete_from_docstore=True)

    manifest["files"].pop(filename, None)
    save_manifest(manifest)
    persist_index(index)
    return True


def index_single_file(file_path: Path) -> dict[str, Any]:
    """Indicizza solo il file indicato.

    Se il file era gia indicizzato dalla webapp:
    - prima eliminiamo i vecchi ref_doc_id;
    - poi inseriamo solo il nuovo contenuto.

    Questo evita duplicati quando carichi una nuova versione dello stesso file.
    """

    filename = file_path.name
    index = load_index()
    delete_file_from_index(index, filename)

    documents = read_file_as_documents(file_path, filename)

    for document in documents:
        index.insert(document)

    manifest = load_manifest()
    manifest["files"][filename] = {
        "ref_doc_ids": [document.id_ for document in documents],
        "size": file_path.stat().st_size,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_manifest(manifest)
    persist_index(index)

    return manifest["files"][filename]


def rebuild_all_documents() -> dict[str, Any]:
    """Ricostruisce l'intero indice partendo dai file presenti in documenti/.

    Questa funzione non e il flusso normale.
    Serve soprattutto quando passi dalla CLI alla webapp e vuoi creare il
    manifest.json iniziale. Dopo questa operazione, upload e delete possono
    lavorare in modo incrementale file per file.
    """

    if (STORAGE_DIR / "index_store.json").exists():
        # Creiamo un indice nuovo invece di tentare cancellazioni parziali:
        # qui l'obiettivo e riallineare tutto in modo certo.
        index = VectorStoreIndex([])
    else:
        index = load_index()

    manifest = {"files": {}}
    files = [
        path
        for path in sorted(DOCUMENTS_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    for file_path in files:
        documents = read_file_as_documents(file_path, file_path.name)

        for document in documents:
            index.insert(document)

        manifest["files"][file_path.name] = {
            "ref_doc_ids": [document.id_ for document in documents],
            "size": file_path.stat().st_size,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    save_manifest(manifest)
    persist_index(index)
    return manifest


@app.on_event("startup")
def startup() -> None:
    """Inizializzazione dell'app.

    FastAPI chiama questa funzione all'avvio del server.
    Qui configuriamo OpenAI e ci assicuriamo che le cartelle esistano.
    """

    ensure_dirs()
    configure_openai()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Restituisce la pagina HTML principale."""

    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/documents")
def list_documents() -> JSONResponse:
    """Elenca i file presenti nella cartella documenti/.

    indexed=True significa che il file e registrato nel manifest della webapp,
    quindi puo essere rimosso dal vector store in modo incrementale.
    """

    ensure_dirs()
    manifest = load_manifest()
    files = []

    for file_path in sorted(DOCUMENTS_DIR.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        record = manifest["files"].get(file_path.name)
        files.append(
            {
                "name": file_path.name,
                "size": file_path.stat().st_size,
                "indexed": record is not None,
                "updated_at": record.get("updated_at") if record else None,
            }
        )

    return JSONResponse({"files": files})


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """Riceve un file dal browser, lo salva e indicizza solo quel file."""

    filename = safe_filename(file.filename or "")
    destination = DOCUMENTS_DIR / filename

    content = await file.read()
    destination.write_bytes(content)

    record = index_single_file(destination)
    return JSONResponse({"ok": True, "file": filename, "record": record})


@app.delete("/api/documents/{filename}")
def delete_document(filename: str) -> JSONResponse:
    """Elimina un file e i suoi vettori dall'indice, se presenti nel manifest."""

    filename = safe_filename(filename)
    index = load_index()
    removed_from_index = delete_file_from_index(index, filename)

    file_path = DOCUMENTS_DIR / filename
    if file_path.exists():
        file_path.unlink()

    return JSONResponse(
        {
            "ok": True,
            "file": filename,
            "removed_from_index": removed_from_index,
        }
    )


@app.post("/api/rebuild")
def rebuild_index() -> JSONResponse:
    """Ricostruisce indice e manifest da tutti i file presenti."""

    manifest = rebuild_all_documents()
    return JSONResponse({"ok": True, "manifest": manifest})


@app.post("/api/ask")
def ask(request: AskRequest) -> StreamingResponse:
    """Risponde in streaming a una domanda.

    Qui usiamo LlamaIndex per la parte RAG, cioe per recuperare i chunk piu
    pertinenti dai documenti indicizzati. Poi usiamo direttamente il client
    OpenAI in streaming.

    Motivo: il query engine streaming di LlamaIndex puo arrivare al browser in
    blocchi poco visibili. Con il client OpenAI diretto controlliamo il flusso
    token per token e la UI puo aggiornarsi mentre la risposta viene generata.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Domanda vuota.")

    manifest = load_manifest()
    if not manifest["files"]:
        raise HTTPException(
            status_code=400,
            detail="Carica e indicizza almeno un documento prima di fare domande.",
        )

    def token_stream():
        index = load_index()
        retriever = index.as_retriever(similarity_top_k=4)
        retrieved_nodes = retriever.retrieve(question)

        context_parts = []
        for i, node_with_score in enumerate(retrieved_nodes, start=1):
            node = node_with_score.node
            metadata = node.metadata or {}
            file_name = metadata.get("file_name", "sorgente sconosciuta")
            context_parts.append(
                f"[Fonte {i}: {file_name}]\n{node.get_content().strip()}"
            )

        context = "\n\n---\n\n".join(context_parts)
        client = OpenAIClient()

        stream = client.chat.completions.create(
            model=LLM_MODEL,
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sei un assistente RAG. Rispondi in italiano basandoti "
                        "solo sul contesto fornito. Se il contesto non contiene "
                        "l'informazione richiesta, dillo chiaramente."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Contesto recuperato dai documenti:\n\n{context}\n\n"
                        f"Domanda dell'utente:\n{question}"
                    ),
                },
            ],
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    return StreamingResponse(
        token_stream(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
