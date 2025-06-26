# Breva Voice‑of‑Customer (VOC) Processing Pipeline

> End‑to‑end workflow that ingests raw VOC survey data, builds an **embedded ChromaDB**, migrates vectors to **Pinecone**, produces **LLM map–reduce summaries**, and serves them through a **Streamlit chatbot**.

---

## 📑 Script‑to‑Script Flow

```text
CSV ─▶ 1️⃣ VOC_chroma_db_upload.py ─┐
                                     │  (vector + metadata)
                                     ▼
                                ChromaDB (voc_responses)
                                     │
                        ┌────────────┴────────────┐
                        │                         │
                        ▼                         ▼
2️⃣ VOC_chroma_to_pinecone.py        3️⃣ VOC_map_reduce.py
   (vectors ➜ Pinecone)                (batch + meta summaries ➜ ChromaDB)
                        │                         │
                        ▼                         │
                 Pinecone (voc‑index)             │
                        │                         │
                        └────────────┬────────────┘
                                     ▼
                               4️⃣ app.py (Streamlit)
```

- **ChromaDB** is the source‑of‑truth for *raw* embeddings **and** the generated `*_summary` documents.
- **Pinecone** only stores the raw response vectors and the meta‑summaries required by the chatbot.

---

## 🛠️ Prerequisites

| Requirement                       | Version / Notes                                                                               |
| --------------------------------- | --------------------------------------------------------------------------------------------- |
| Python                            | ≥ 3.10                                                                                        |
| `pip install -r requirements.txt` | ChromaDB, Pinecone, Anthropic, Streamlit, etc.                                                |
| Accounts / keys                   | Pinecone (**PINECONE\_API\_KEY**), Anthropic (**ANTHROPIC\_API\_KEY**), optional Azure OpenAI |
| GPU (optional)                    | Speeds up SentenceTransformer embedding                                                       |

> **Tip:** store secrets in a `.env` file or your CI secrets store; *never* commit them.

---

## 🚀 Quick‑Start

### 0. Set environment variables

```bash
export PINECONE_API_KEY="<your‑pinecone‑key>"
export ANTHROPIC_API_KEY="<your‑anthropic‑key>"
# Optional if you swap in an Azure model for embeddings
export AZURE_OPENAI_API_KEY="<azure‑key>"
export AZURE_OPENAI_ENDPOINT="<endpoint>"
```

### 1. Build the local ChromaDB

```bash
python VOC_chroma_db_upload.py \
  --csv data/merged_grant_applications_q2_2025.csv \
  --persist_dir chroma_database_update_2025_q2 \
  --no-clustering           # or --do-clustering --chunk_tokens 150
```

*If the script is run without CLI flags, edit the paths in **`main()`**.*

### 2. Migrate vectors to Pinecone

```bash
python VOC_chroma_to_pinecone.py \
  --chroma-dir   chroma_database_update_2025_q2 \
  --collection-name voc_responses \
  --index-name   voc-index-2025-q2 \
  --batch-size   100
```

### 3. Generate map–reduce summaries

```bash
python VOC_map_reduce.py \
  --batch-size 100           # optional if you expose argparse
```

The job loops until *all* `question_type`s have a corresponding `*_summary` document in ChromaDB.

### 4. Launch the chatbot UI

```bash
streamlit run app.py
```

The app searches Pinecone for the matching `question_type_summary` and crafts a Claude prompt.

---

## 🔧 Hard‑coded Values & Where to Change Them

| Script                           | Variable / Arg                              | Current Default                                                                 | What It Does                    |
| -------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------- |
| **VOC\_chroma\_db\_upload.py**   | `persist_directory` (class init & `main()`) | `/Users/sveerisetti/Desktop/Breva_VOC_Chat-main/chroma_database_update_2025_q2` | Folder where ChromaDB is stored |
|                                  | `csv_path` (`main()`)                       | `/Users/.../merged_grant_applications_q2_2025.csv`                              | Input dataset                   |
| **VOC\_chroma\_to\_pinecone.py** | `--pinecone-api-key` default                | `pcsk_…`                                                                        | Your Pinecone secret key        |
|                                  | `--index-name`                              | `voc-index-2025-q2`                                                             | Pinecone index name             |
| **VOC\_map\_reduce.py**          | `anthropic_api_key` (class init & `main()`) | `sk-ant-…`                                                                      | Anthropic/Claude key            |
|                                  | `batch_size`                                | `60` (init), `100` (main)                                                       | Responses per Claude call       |
| **app.py**                       | `pinecone_api_key` secret fallback          | `pcsk_…`                                                                        | Used by `VOCDatabaseQuerier`    |
|                                  | `index_name`                                | `voc-index-2025-q2`                                                             | Must match step 2               |

> **Recommendation:** refactor these into CLI flags or environment variables. `argparse` is already in place for script #2.

---

## 🧪 Local Development Tips

1. **Embed locally first.** Re‑running step 1 is fast; keep the CSV small while testing.
2. **Use Pinecone in “starter” environment** to avoid accidental charges while iterating.
3. **Set **`` if you want thematic grouping; it takes longer but reduces noise.
4. **Rotate API keys** before committing code or sharing the repo.
5. **Streamlit Hot‑Reload**: run `streamlit run app.py --server.runOnSave true` to auto‑refresh when you edit the UI.

---

## 🏗️ Roadmap / TODO

-

Feel free to open issues or submit PRs! 🎉

