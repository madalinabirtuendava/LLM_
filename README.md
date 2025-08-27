# Smart Librarian — RAG + Tool (Streamlit)

## 1) Descriere
Acest proiect construiește un chatbot care recomandă **o singură carte** în funcție de interesul utilizatorului, folosind:
- **RAG** cu **ChromaDB** (vector store persistent pe disc)
- **OpenAI Embeddings** (`text-embedding-3-small`)
- **OpenAI Chat** pentru răspuns conversațional
- **Tool local** `get_summary_by_title(title)` care oferă **rezumatul complet** al cărții recomandate
- **UI** în **Streamlit**

## 2) Structura proiectului
```text
.
├─ book_summaries.md       # 10+ cărți: "## Title: <titlu>" + rezumat (3–5 rânduri)
├─ build_index.py          # indexare în ChromaDB (embeddings OpenAI) – folosește variabila de mediu OPENAI_API_KEY
├─ streamlit_app.py        # aplicația Streamlit (RAG + GPT + tool) – citește cheia din .streamlit/secrets.toml
├─ requirements.txt        # dependențe
├─ .streamlit/
│  └─ secrets.toml         # OPENAI_API_KEY = "sk-..." (local, necomitat)
└─ chroma_db/              # se creează la rularea indexării (persistență Chroma)
```

> **Notă:** nu comita secretele sau artefactele locale. Vezi secțiunea **.gitignore**.

## 3) Cerințe
- Python **3.9+**
- Cont **OpenAI** și **cheie API**

## 4) Instalare dependențe
Recomandat într-un mediu virtual:
```bash
pip install -r requirements.txt
# sau
pip install chromadb openai python-dotenv streamlit
```

## 5) Pregătire date
Creează `book_summaries.md` cu **cel puțin 10** titluri; fiecare bloc:
```text
## Title: Numele cărții
Rezumat pe 3–5 rânduri...
```

## 6) Configurare cheie API (fără a o pune în cod)

### 6.1 Streamlit (UI) — `st.secrets`
Creează fișierul `.streamlit/secrets.toml` cu:
```toml
OPENAI_API_KEY = "sk-INTRODU-CHEIA-TA-AICI"
```

### 6.2 Scriptul de indexare — variabilă de mediu
Setează variabila **OPENAI_API_KEY** (doar o dată):
- **Windows PowerShell**
  ```powershell
  setx OPENAI_API_KEY "sk-INTRODU-CHEIA-TA-AICI"
  ```
  Apoi **redeschide terminalul**.
- **macOS/Linux (bash/zsh)**
  ```bash
  export OPENAI_API_KEY="sk-INTRODU-CHEIA-TA-AICI"
  ```
  (Opțional: pune în `.env` – `python-dotenv` o va citi.)

## 7) Construirea indexului (RAG)
Rulează din folderul proiectului:
```bash
python build_index.py
```
Dacă totul e OK, vei vedea ceva de forma:
```text
[OK] Indexed 10 books în colecția 'books' (persistă în 'chroma_db').
[Test] Query: ...
- <titlu 1> (distance=...)
- <titlu 2> (distance=...)
...
```

## 8) Rulare UI (Streamlit)
```bash
streamlit run streamlit_app.py
```
Deschide `http://localhost:8501`, introdu o întrebare și apasă **„Recomandă + Rezumat detaliat”**.

**Cum oprești serverul:** `Ctrl + C` în terminal (sau butonul roșu **Stop** în PyCharm).

## 9) Cum funcționează
1. **RAG:** întrebarea este codificată cu embeddings → căutare semantică în Chroma (top‑k documente).
2. **GPT:** modelul alege **un singur** titlu dintre candidați.
3. **Tool:** se apelează `get_summary_by_title(title)` pentru **rezumatul complet**.
4. **UI:** afișează:
   - `Recomandare: <titlul exact>`
   - `Motivare: <2–4 fraze>`
   - `Rezumat complet: <textul întors de tool>`
   - *(jos)* „Cărți candidate” = top 3 rezultate din RAG.

## 10) Exemple de întrebări
- „Vreau o carte despre libertate și control social.”
- „Ce îmi recomanzi dacă iubesc poveștile fantastice?”
- „Vreau o carte despre prietenie și magie.”
- „Ce carte vorbește despre dragoste și prejudecăți?”

## 11) Depanare rapidă
- **`OPENAI_API_KEY nu este setată`** → verifică variabila de mediu sau `.env` înainte de a rula `build_index.py`.
- **`Unbalanced quotes în secrets.toml`** → folosește ghilimele drepte `"` și nu pune alte caractere după.
- **`No module named 'chromadb'`** → rulează `pip install -r requirements.txt`.
- **Index cu <10 cărți** → verifică formatul titlurilor (regex: blocuri care încep cu `## Title:`) și separarea clară a rezumatelor.
- **Recomandări irelevante** → ajustează formularea întrebării sau verifică rezumatele.

## 12) .gitignore recomandat
```gitignore
# secrete / config local
.streamlit/
.env

# artefacte locale
chroma_db/
__pycache__/
*.pyc
.DS_Store
```
