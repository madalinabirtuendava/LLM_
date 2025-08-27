# build_index.py
import os
import re
from typing import List, Tuple

from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

BOOKS_FILE = "book_summaries.md"
PERSIST_DIR = "chroma_db"           # unde se salvează vectorii pe disc (persistă)
COLLECTION_NAME = "books"           # numele colecției din Chroma


def parse_book_summaries(path: str) -> List[Tuple[str, str]]:
    """
    Extrage (title, summary) din fișierul book_summaries.md.
    Format așteptat:
      ## Title: TITLU
      rezumat pe 3–5 rânduri...
    (blocurile pot fi separate de linii goale)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nu găsesc '{path}'. Asigură-te că există lângă acest script.")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r"(?m)^##\s*Title:\s*(.+?)\s*\n(.*?)(?=^##\s*Title:|\Z)"
    matches = re.findall(pattern, content, flags=re.DOTALL | re.MULTILINE)

    books: List[Tuple[str, str]] = []
    for title, body in matches:
        clean_title = title.strip()
        clean_summary = re.sub(r"\n\s*\n+", "\n\n", body.strip())
        if clean_title and clean_summary:
            books.append((clean_title, clean_summary))

    if len(books) < 10:
        raise ValueError(f"Am găsit doar {len(books)} cărți; ai nevoie de cel puțin 10 în '{path}'.")
    return books


def main():
    # 0) Încarcă .env (dacă există) și citește cheia din variabila de mediu
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY nu este setată. Seteaz-o în mediul sistemului sau în .env."
        )

    # 1) Parse summaries
    books = parse_book_summaries(BOOKS_FILE)

    # 2) Creează Chroma persistent + embedding function (OpenAI)
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small",
    )
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    # 3) Refă colecția (șterge dacă există deja)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    # 4) Adaugă documentele
    ids, documents, metadatas = [], [], []
    for i, (title, summary) in enumerate(books):
        ids.append(f"book-{i+1}")
        documents.append(f"Title: {title}\n\n{summary}")
        metadatas.append({"title": title})

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"[OK] Indexed {len(ids)} books în colecția '{COLLECTION_NAME}' (persistă în '{PERSIST_DIR}').")

    # 5) Test scurt de căutare
    test_query = "prieteni și magie, aventură"
    results = collection.query(query_texts=[test_query], n_results=3)
    print("\n[Test] Query:", test_query)
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f"- {meta.get('title')}  (distance={dist:.4f})")


if __name__ == "__main__":
    main()
