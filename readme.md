# Breva Voice‑of‑Customer (VOC) Pipeline

A **4‑script toolchain** that turns raw survey CSVs into a fully searchable, summarized knowledge base and exposes it through a Streamlit chatbot.

---

## 1  Architecture at a Glance

### 1.1 Recommended "One‑Shot" Flow (raw *and* summaries sent to Pinecone)

```text
CSV ─▶ ① VOC_chroma_db_upload.py  ─┐
                                   │ vectors + metadata
                                   ▼
                              ChromaDB (voc_responses)
                                   │
                                   ▼
                            ② VOC_map_reduce.py
                              (adds *_summary docs)
                                   │
                                   ▼
                            ③ VOC_chroma_to_pinecone.py
                              (migrate ALL docs)
                                   │
                                   ▼
                            Pinecone index (voc-index‑*)
                                   │
                                   ▼
                          ④ app.py  (Streamlit chatbot)
```

**Why this order?** A *single* migration (Step ③) pushes both raw responses **and** meta‑summaries to Pinecone, keeping the search index in sync.

### 1.2 Legacy Flow (two migrations)

```text
CSV → ① Upload → ChromaDB
           │
           ▼
③ Migrate raw vectors → Pinecone  (first pass)
           │
           ▼
② Map‑reduce summaries → ChromaDB
           │
           ▼
③ Migrate again → Pinecone        (push summaries)
```

If you already ran the first migration, **rerun Step ③** after Step ② finishes so the `*_summary` docs make it to Pinecone.

---

## 2  Script Cheat‑Sheet

| # | Script                      | What It Does                                                                                                     | Key Inputs                                                    | Key Outputs                                        |
| - | --------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------- |
| ① | `VOC_chroma_db_upload.py`   | Reads the CSV, cleans & chunks text, embeds with **Sentence‑Transformer**, stores in **ChromaDB**                | `--csv`, `--persist_dir`, `--do-clustering`, `--chunk_tokens` | `voc_responses` collection with vectors & metadata |
| ② | `VOC_map_reduce.py`         | Batches responses per `question_type`, calls **Claude Haiku • Sonnet**, stores `*_summary` docs back to ChromaDB | `--batch-size`, `ANTHROPIC_API_KEY`                           | New docs tagged `question_type: <type>_summary`    |
| ③ | `VOC_chroma_to_pinecone.py` | Pulls *everything* from ChromaDB and upserts to **Pinecone** (V2 API)                                            | `--chroma-dir`, `--index-name`, `PINECONE_API_KEY`            | Pinecone vectors with full metadata (incl. `text`) |
| ④ | `app.py`                    | Streamlit UI → maps user query → finds matching summary in Pinecone → crafts Claude prompt → chat                | `pinecone_api_key`, `anthropic_api_key`, `index_name`         | Web app at `http://localhost:8501`                 |

---

## 3  Prerequisites

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # chromadb, pinecone-client, anthropic, streamlit, etc.
```

Set secrets (bash or `.env`):

```bash
export PINECONE_API_KEY="…"
export ANTHROPIC_API_KEY="…"
# (optional) AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
```

A GPU is nice but optional.

---

## 4  Step‑by‑Step (Recommended Flow)

1. **Embed & store raw data**

   ```bash
   python VOC_chroma_db_upload.py \
     --csv data/merged_grant_applications_q2_2025.csv \
     --persist_dir chroma_db_q2_2025 \
     --no-clustering
   ```

2. **Generate map‑reduce summaries**

   ```bash
   python VOC_map_reduce.py --batch-size 100 \
     --persist_dir chroma_db_q2_2025
   ```

   The script loops until every `question_type` has a meta‑summary.

3. **Migrate ALL docs to Pinecone**

   ```bash
   python VOC_chroma_to_pinecone.py \
     --chroma-dir chroma_db_q2_2025 \
     --index-name voc-index-2025-q2 \
     --batch-size 100
   ```

4. **Launch the chatbot**

   ```bash
   streamlit run app.py
   ```

   Ask something like *"What challenges do founders have with cash‑flow forecasting?"* — the app fetches the cached summary and responds via Claude.

---

## 5  Where to Tweak Hard‑coded Paths / Keys

| Location            | Variable             | Default                                   | Suggestion                       |
| ------------------- | -------------------- | ----------------------------------------- | -------------------------------- |
| **upload.py**       | `persist_directory`  | `/Users/…/chroma_database_update_2025_q2` | Pass via `--persist_dir`         |
|                     | CSV path in `main()` | absolute path                             | Promote to `argparse` flag       |
| **map\_reduce.py**  | `anthropic_api_key`  | hard‑coded                                | Use `os.environ` + `argparse`    |
| **to\_pinecone.py** | `--pinecone-api-key` | hard‑coded sample                         | Remove default, require flag/env |
| **app.py**          | fallback secrets     | hard‑coded sample                         | Store in **Streamlit Secrets**   |

---

## 6  Tips & Tricks

- **Chunking** (`--chunk_tokens`) helps with very long answers; leave 0 to store full response.
- To **dedupe** bad responses, adjust `VOCDatabaseCreator.preprocess_text()`.
- **Pinecone “starter” pod** keeps costs low (<5M vectors free).
- **Streamlit hot‑reload**: `streamlit run app.py --server.runOnSave true`.
- Rotate keys before pushing to Git.

---

## 7  Roadmap

-

Contributions welcome 🎉

