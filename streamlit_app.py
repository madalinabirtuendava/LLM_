# streamlit_app.py
import os
import json
import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

OPENAI_KEY_FILE = "OpenAI.txt"
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "books"

# ====== 1) REZUMATE COMPLETE (tool local) ======
book_summaries_dict = {
    "1984": (
        "Roman distopic despre un stat totalitar ce controlează viața publică și privată "
        "prin supraveghere, manipulare și rescrierea adevărului. Winston Smith încearcă să "
        "gândească liber și să iubească într-o lume unde independența este o crimă. "
        "Cartea explorează teme precum libertatea, controlul social și natura adevărului."
    ),
    "The Hobbit": (
        "Bilbo Baggins, un hobbit comod, pornește într-o aventură alături de treisprezece pitici "
        "și de vrăjitorul Gandalf pentru a recupera comoara furată de dragonul Smaug. "
        "Pe drum, Bilbo își descoperă curajul, ingeniozitatea și loialitatea. "
        "Povestea este despre prietenie, maturizare și bucuria călătoriei."
    ),
    "Harry Potter and the Philosopher’s Stone": (
        "Harry descoperă că este vrăjitor și începe școala Hogwarts, unde își face prieteni "
        "apropiați și înfruntă primele provocări magice. Misterul Piatrei Filozofale îl aduce "
        "față în față cu amenințarea lui Voldemort. "
        "Teme: prietenie, curaj, alegeri morale."
    ),
    "The Lord of the Rings: The Fellowship of the Ring": (
        "Frodo primește Inelul Unic și pornește, alături de o Frăție, într-o misiune de a-l distruge "
        "în focurile Muntelui Osândei. Călătoria este plină de pericole, tentații și sacrificii. "
        "Teme: loialitate, speranță, lupta împotriva răului."
    ),
    "Pride and Prejudice": (
        "Elizabeth Bennet sfidează așteptările sociale ale vremii și îl cunoaște pe domnul Darcy, "
        "cu care trece de la prejudecăți la înțelegere și respect. "
        "Roman despre iubire, statut social și autocunoaștere."
    ),
    "To Kill a Mockingbird": (
        "În sudul american marcat de rasism, Scout Finch învață despre empatie și dreptate "
        "în timp ce tatăl ei, avocatul Atticus, apără un bărbat de culoare acuzat pe nedrept. "
        "O meditație despre moralitate, inocență și curaj civil."
    ),
    "Dune": (
        "Pe planeta deșertică Arrakis, Paul Atreides este aruncat în jocuri politice și profeții "
        "ce îi modelează destinul. Controlul asupra mirodeniei determină echilibrul puterii. "
        "Teme: ecologie, leadership, credință, destin."
    ),
    "The Catcher in the Rye": (
        "Holden Caulfield rătăcește prin New York după ce pleacă de la școală, căutând sens și "
        "autenticitate într-o lume pe care o percepe ipocrită. "
        "Teme: alienare, identitate, protejarea inocenței."
    ),
    "The Great Gatsby": (
        "Jay Gatsby urmărește un vis imposibil: iubirea lui Daisy, pe fundalul strălucitor al anilor ’20. "
        "Sub aparența opulenței se află iluzii, trădare și deznădejde. "
        "Teme: visul american, clase sociale, decepție."
    ),
    "Brave New World": (
        "O societate aparent perfectă își menține stabilitatea prin condiționare și hedonism controlat. "
        "Individualitatea și profunzimea emoțională sunt sacrificate pentru confort. "
        "Teme: tehnologie, conformism, libertate."
    ),
}

def get_summary_by_title(title: str) -> str:
    # căutare robustă: exactă, apoi case-insensitive / contains
    if title in book_summaries_dict:
        return book_summaries_dict[title]
    t = title.strip().lower()
    for k in book_summaries_dict.keys():
        if k.lower() == t:
            return book_summaries_dict[k]
    for k in book_summaries_dict.keys():
        if t in k.lower() or k.lower() in t:
            return book_summaries_dict[k]
    return "Nu am găsit rezumatul pentru acest titlu."

# ====== 2) Utilitare OpenAI/Chroma ======
def read_openai_key(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nu găsesc fișierul cu cheia: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token.startswith("sk-"):
                return token
    raise RuntimeError("Nu am găsit o cheie care începe cu 'sk-' în OpenAI.txt")

@st.cache_resource(show_spinner=False)
def init_clients():
    api_key = st.secrets["OPENAI_API_KEY"]  # cheia vine din .streamlit/secrets.toml

    embedding_fn = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = chroma_client.get_collection(
        COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    openai_client = OpenAI(api_key=api_key)
    return collection, openai_client

# ====== 3) RAG + Function Calling ======
def recommend_and_fetch_summary(user_query: str, collection, openai_client):
    # RAG — top3 documente ca context
    results = collection.query(query_texts=[user_query], n_results=3)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    titles = [m.get("title") for m in metas]
    context = "\n\n---\n\n".join(docs) if docs else "Niciun context găsit."

    # 1) Cerem modelului să aleagă UN TITLU din context și să apeleze tool-ul cu acel titlu
    system_msg = (
        "Ești un asistent care recomandă exact O SINGURĂ carte din context. "
        "După ce alegi titlul, apelează funcția get_summary_by_title cu acel titlu. "
        "Vei furniza răspunsul final abia după ce primești rezultatul tool-ului."
    )
    user_msg = (
        f"Întrebarea utilizatorului: {user_query}\n\n"
        f"Context (cărți candidate):\n{context}\n\n"
        "Alege un singur titlu EXACT din context și apelează funcția cu argumentul 'title'."
    )

    tools = [{
        "type": "function",
        "function": {
            "name": "get_summary_by_title",
            "description": "Returnează rezumatul complet pentru un titlu exact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Titlul exact al cărții recomandate"}
                },
                "required": ["title"]
            },
        },
    }]

    # Prima chemare: modelul decide titlul și (ideal) produce un tool_call
    first = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        tools=tools,
        # IMPORTANT: obligă modelul să apeleze exact funcția noastră
        tool_choice={"type": "function", "function": {"name": "get_summary_by_title"}},
        temperature=0.2,
    )

    msg = first.choices[0].message

    # Dacă a cerut tool-ul, îl executăm local și trimitem rezultatul înapoi modelului
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls or []},
    ]

    if msg.tool_calls:
        for call in msg.tool_calls:
            if call.function.name == "get_summary_by_title":
                try:
                    args = json.loads(call.function.arguments or "{}")
                    title_arg = args.get("title", "")
                except Exception:
                    title_arg = ""
                summary = get_summary_by_title(title_arg)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": "get_summary_by_title",
                    "content": summary
                })

        # A doua chemare: compune răspunsul final folosind rezultatul tool-ului
        final = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages + [
                {"role": "system", "content":
                    "Formatează răspunsul final exact așa:\n"
                    "Recomandare: <Titlul exact>\n"
                    "Motivare: <2–4 fraze clare>\n"
                    "Rezumat complet: <textul întors de tool>"
                 }
            ],
            temperature=0.3,
        )
        return final.choices[0].message.content.strip(), titles

    # Fallback: dacă nu a apelat tool-ul, construim manual
    fallback_answer = "Nu am putut apela tool-ul pentru rezumat. Încearcă din nou."
    return fallback_answer, titles

# ====== 4) UI Streamlit ======
def main():
    st.set_page_config(page_title="Smart Librarian (RAG + Tool)", page_icon="📚")
    st.title("📚 Smart Librarian — RAG + Tool")
    st.caption("Scrie o cerință (ex: „Vreau o carte despre prietenie și magie”).")

    collection, openai_client = init_clients()

    query = st.text_input("Mesajul tău")
    if st.button("Recomandă + Rezumat detaliat"):
        if not query.strip():
            st.warning("Te rog introdu o întrebare / descriere.")
            st.stop()

        with st.spinner("Caut în bibliotecă, aleg o carte și generez rezumatul..."):
            answer, titles = recommend_and_fetch_summary(query, collection, openai_client)

        st.subheader("Răspuns")
        st.write(answer)

        st.markdown("---")
        st.caption("Cărți candidate (top 3 din căutare): " + ", ".join([t for t in titles if t]))

if __name__ == "__main__":
    main()
