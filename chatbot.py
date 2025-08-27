import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

# Config
OPENAI_KEY_FILE = "OpenAI.txt"
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "books"

# Citește cheia din fișier
def read_openai_key(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token.startswith("sk-"):
                return token
    raise RuntimeError("Nu am găsit cheia în fișier!")

# Inițializează Chroma + OpenAI
def init():
    api_key = read_openai_key(OPENAI_KEY_FILE)
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)
    openai_client = OpenAI(api_key=api_key)
    return collection, openai_client

# Funcția chatbotului
def chatbot():
    collection, openai_client = init()
    print("Chatbot-ul este gata! Scrie 'exit' ca să ieși.\n")

    while True:
        query = input("Tu: ")
        if query.lower() in ["exit", "quit"]:
            print("Chatbot: La revedere! 👋")
            break

        # 1) Căutăm în Chroma
        results = collection.query(query_texts=[query], n_results=2)
        docs = results["documents"][0]
        context = "\n\n".join(docs)

        # 2) Trimitem către GPT cu context
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ești un asistent care recomandă cărți pe baza descrierilor."},
                {"role": "user", "content": f"Întrebarea mea: {query}\n\nContext (rezumate cărți):\n{context}\n\nTe rog recomandă o carte potrivită."}
            ]
        )

        answer = response.choices[0].message.content
        print("Chatbot:", answer, "\n")

if __name__ == "__main__":
    chatbot()
