# MSSQL Table Replicator for Hybrid Search

Streamlit application for copying a SQL Server source table into a target table with:

- SQL Server 2025 `VECTOR(...)` embedding storage
- Traditional Chinese full-text indexing
- OpenAI-compatible embedding APIs
- Batch-oriented replication instead of loading large tables into memory

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## SQL Server Requirements

- SQL Server 2025
- Full-Text Search installed
- Target database user can create tables, indexes, full-text catalogs, and full-text indexes
- Traditional Chinese full-text language installed; the app validates LCID `1028`
- Vector index creation requires SQL Server 2025 preview vector index support to be enabled

## Notes

The app does not persist database passwords or embedding API keys. It keeps them in the active Streamlit session only.

For large tables, use a stable numeric or comparable key column for batching and resumability.

The target load behavior has two modes:

- `Create only if empty`: the safe default. The app stops if the target table already contains rows.
- `Truncate target and rebuild`: deletes all target rows before loading. The UI requires explicit confirmation.

Rows with empty embedding text are still copied to the target table, but the app skips the embedding API call and stores `NULL` for embedding metadata and vector values.
