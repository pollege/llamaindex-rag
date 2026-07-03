"""
CLI RAG locale con LlamaIndex + OpenAI.

Obiettivo del file:
- leggere documenti inseriti nella cartella "documenti";
- creare un indice vettoriale locale con LlamaIndex;
- fare domande da terminale;
- usare OpenAI sia per gli embedding sia per generare le risposte.

Nota sulla API key:
- NON conviene scriverla hardcoded nel codice, anche se il progetto gira solo sul tuo PC.
- Questo script la legge da una variabile d'ambiente chiamata OPENAI_API_KEY.
- Il modo piu comodo e creare un file ".env" nella stessa cartella del progetto.
  Esempio:

      OPENAI_API_KEY=sk-...

  Il file ".env" viene escluso da Git tramite ".gitignore".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI


# Path principali del progetto.
# Path(__file__) indica questo file Python; parent indica la sua cartella.
# In questo modo lo script funziona anche se lo lanci da una cartella diversa.
PROJECT_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_DIR / "documenti"
STORAGE_DIR = PROJECT_DIR / "storage"
MANIFEST_PATH = STORAGE_DIR / "manifest.json"
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}
LLM_MODEL = "gpt-5.4-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


def configure_openai() -> None:
    """Carica la configurazione OpenAI usata da LlamaIndex.

    LlamaIndex usa l'oggetto globale "Settings" per sapere quali modelli usare.
    Qui configuriamo:
    - llm: il modello che scrive la risposta finale in linguaggio naturale;
    - embed_model: il modello che trasforma i testi in vettori numerici.

    I vettori sono fondamentali nel RAG: quando fai una domanda, anche la domanda
    viene trasformata in un vettore; LlamaIndex cerca poi i pezzi di documento
    con vettori piu simili e li passa al modello come contesto.
    """

    # load_dotenv cerca un file ".env" e aggiunge le variabili al processo Python.
    # Se OPENAI_API_KEY e gia presente nel terminale, non viene sovrascritta.
    load_dotenv(PROJECT_DIR / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Manca OPENAI_API_KEY. Crea un file .env partendo da .env.example "
            "oppure esporta la variabile nel terminale.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Modello LLM usato per generare la risposta finale.
    Settings.llm = LlamaIndexOpenAI(model=LLM_MODEL)

    # text-embedding-3-small e adatto per un primo sistema RAG locale.
    # Se un giorno vuoi piu qualita, puoi provare "text-embedding-3-large".
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

    # Dimensione massima dei chunk: LlamaIndex spezza i documenti in blocchi.
    # Chunk troppo grandi recuperano contesto rumoroso; chunk troppo piccoli
    # rischiano di perdere il significato. 1024 e un buon punto di partenza.
    Settings.chunk_size = 1024

    # Sovrapposizione tra chunk consecutivi.
    # Serve a non perdere frasi importanti che stanno a cavallo tra due blocchi.
    Settings.chunk_overlap = 120


def read_file_as_documents(file_path: Path) -> list:
    """Legge un singolo file e assegna id stabili ai documenti LlamaIndex.

    Questa funzione e importante per mantenere compatibile la CLI con la
    webapp. La webapp cancella un documento dal vector store usando i suoi
    "ref_doc_id"; quindi anche la CLI deve creare id prevedibili e salvarli
    nel manifest.

    Per un file chiamato esempio.pdf gli id diventano:
    - esempio.pdf::0
    - esempio.pdf::1
    - esempio.pdf::2

    Se il lettore produce un solo Document, ci sara solo esempio.pdf::0.
    """

    documents = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()

    for i, document in enumerate(documents):
        document.id_ = f"{file_path.name}::{i}"
        document.metadata = {
            **(document.metadata or {}),
            "uploaded_file": file_path.name,
            "file_name": file_path.name,
        }

    return documents


def load_documents() -> tuple[list, dict[str, Any]]:
    """Legge tutti i documenti supportati dalla cartella "documenti".

    SimpleDirectoryReader e il lettore base di LlamaIndex per cartelle locali.
    Con i pacchetti indicati in requirements.txt puo leggere file comuni come:
    - .txt
    - .pdf
    - .docx

    recursive=True permette di creare sottocartelle dentro "documenti".
    """

    DOCUMENTS_DIR.mkdir(exist_ok=True)

    documents = []
    manifest: dict[str, Any] = {"files": {}}

    files = [
        path
        for path in sorted(DOCUMENTS_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    for file_path in files:
        file_documents = read_file_as_documents(file_path)
        documents.extend(file_documents)
        manifest["files"][file_path.name] = {
            "ref_doc_ids": [document.id_ for document in file_documents],
            "size": file_path.stat().st_size,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    if not documents:
        print(
            f"Nessun documento trovato in: {DOCUMENTS_DIR}\n"
            "Inserisci almeno un file .txt, .pdf o .docx e rilancia il comando.",
            file=sys.stderr,
        )
        sys.exit(1)

    return documents, manifest


def build_index() -> VectorStoreIndex:
    """Crea un nuovo indice dai documenti e lo salva su disco.

    VectorStoreIndex e l'indice vettoriale standard di LlamaIndex:
    - prende i documenti;
    - li spezza in chunk;
    - calcola gli embedding OpenAI;
    - salva tutto in uno storage locale.

    Lo storage locale evita di ricalcolare gli embedding a ogni domanda.
    Quando aggiungi/modifichi documenti, rilancia:

        python3 rag_cli.py indicizza
    """

    documents, manifest = load_documents()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))

    with MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(f"Indice creato con {len(documents)} documenti letti.")
    print(f"Indice salvato in: {STORAGE_DIR}")
    print(f"Manifest webapp salvato in: {MANIFEST_PATH}")
    return index


def load_or_build_index(rebuild: bool = False) -> VectorStoreIndex:
    """Carica l'indice esistente oppure lo crea se non esiste.

    Parametro rebuild:
    - False: usa l'indice gia salvato se presente;
    - True: ricostruisce l'indice dai file in "documenti".
    """

    if rebuild or not STORAGE_DIR.exists():
        return build_index()

    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    return load_index_from_storage(storage_context)


def create_query_engine(index: VectorStoreIndex) -> Any:
    """Trasforma l'indice in un motore di domanda/risposta.

    similarity_top_k=4 significa:
    - per ogni domanda recupera i 4 chunk piu pertinenti;
    - passa quei chunk al modello OpenAI;
    - il modello genera una risposta basata sul contesto recuperato.

    Se le risposte sono troppo povere puoi aumentare a 6 o 8.
    Se diventano confuse o troppo lunghe puoi scendere a 2 o 3.
    """

    return index.as_query_engine(similarity_top_k=4)


def ask_once(question: str, rebuild: bool = False, show_sources: bool = True) -> None:
    """Esegue una singola domanda passata da riga di comando."""

    index = load_or_build_index(rebuild=rebuild)
    query_engine = create_query_engine(index)
    response = query_engine.query(question)

    print("\nRisposta:\n")
    print(response)

    if show_sources:
        print_sources(response)


def chat_loop(rebuild: bool = False, show_sources: bool = True) -> None:
    """Avvia una piccola chat interattiva nel terminale.

    Questa non e una chat con memoria lunga: ogni domanda viene risolta cercando
    nei documenti indicizzati. Per un primo RAG e spesso la cosa piu chiara,
    perche puoi vedere bene il ciclo:
    domanda -> recupero documenti -> risposta.
    """

    index = load_or_build_index(rebuild=rebuild)
    query_engine = create_query_engine(index)

    print("RAG CLI pronta. Scrivi una domanda oppure 'exit' per uscire.\n")

    while True:
        question = input("Domanda> ").strip()

        if question.lower() in {"exit", "quit", "q", "esci"}:
            print("Ciao!")
            break

        if not question:
            continue

        response = query_engine.query(question)
        print("\nRisposta:\n")
        print(response)

        if show_sources:
            print_sources(response)

        print()


def print_sources(response) -> None:
    """Mostra da quali file/chunk arriva il contesto usato per la risposta.

    In un sistema RAG e utile non fidarsi solo della risposta finale.
    Le source_nodes sono i pezzi di documento recuperati dal motore di ricerca
    semantica. Stampiamo:
    - nome/path del file;
    - punteggio di similarita;
    - breve estratto del testo.
    """

    source_nodes = getattr(response, "source_nodes", None) or []

    if not source_nodes:
        return

    print("\nFonti recuperate:\n")

    for i, source_node in enumerate(source_nodes, start=1):
        node = source_node.node
        metadata = node.metadata or {}

        # SimpleDirectoryReader salva spesso il path originale in metadata.
        file_name = (
            metadata.get("file_name")
            or metadata.get("file_path")
            or metadata.get("filename")
            or "sorgente sconosciuta"
        )

        score = source_node.score
        score_text = f"{score:.3f}" if score is not None else "n/d"

        # get_content restituisce il testo del chunk.
        # Lo accorciamo per mantenere leggibile il terminale.
        snippet = node.get_content().replace("\n", " ").strip()
        snippet = snippet[:300] + ("..." if len(snippet) > 300 else "")

        print(f"[{i}] {file_name} | score: {score_text}")
        print(f"    {snippet}\n")


def parse_args() -> argparse.Namespace:
    """Definisce i comandi disponibili da terminale.

    Esempi:

        python3 rag_cli.py indicizza
        python3 rag_cli.py chiedi "Qual e il tema principale dei documenti?"
        python3 rag_cli.py chat
        python3 rag_cli.py chat --reindex
        python3 rag_cli.py chiedi "..." --no-sources
    """

    parser = argparse.ArgumentParser(
        description="CLI RAG locale con LlamaIndex, documenti locali e OpenAI."
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "indicizza",
        help="Legge i documenti nella cartella documenti e ricrea l'indice locale.",
    )

    ask_parser = subparsers.add_parser(
        "chiedi",
        help="Fa una singola domanda ai documenti indicizzati.",
    )
    ask_parser.add_argument("question", help="La domanda da fare ai documenti.")
    ask_parser.add_argument(
        "--reindex",
        action="store_true",
        help="Ricrea l'indice prima di rispondere.",
    )
    ask_parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Non mostrare le fonti recuperate.",
    )

    chat_parser = subparsers.add_parser(
        "chat",
        help="Avvia una sessione interattiva nel terminale.",
    )
    chat_parser.add_argument(
        "--reindex",
        action="store_true",
        help="Ricrea l'indice prima di iniziare la chat.",
    )
    chat_parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Non mostrare le fonti recuperate.",
    )

    return parser.parse_args()


def main() -> None:
    """Punto di ingresso dello script."""

    args = parse_args()
    configure_openai()

    if args.command == "indicizza":
        build_index()
    elif args.command == "chiedi":
        ask_once(
            question=args.question,
            rebuild=args.reindex,
            show_sources=not args.no_sources,
        )
    elif args.command == "chat":
        chat_loop(rebuild=args.reindex, show_sources=not args.no_sources)
    else:
        print(
            "Scegli un comando. Esempio:\n"
            "  python3 rag_cli.py indicizza\n"
            "  python3 rag_cli.py chiedi \"Di cosa parlano i documenti?\"\n"
            "  python3 rag_cli.py chat"
        )


if __name__ == "__main__":
    main()
