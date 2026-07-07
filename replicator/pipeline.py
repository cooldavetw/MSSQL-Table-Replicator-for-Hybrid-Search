from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from .config import EmbeddingConfig, SourceTableConfig, TargetTableConfig
from .embeddings import EmbeddingClient
from .sqlserver import (
    ColumnInfo,
    create_fulltext_index,
    create_target_table,
    create_vector_index,
    drop_target_table,
    identifiers_equal,
    get_columns,
    insert_target_rows,
    read_source_batch,
    target_table_has_rows,
    truncate_target_table,
    validate_traditional_chinese_fulltext,
    Connection,
)


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class ReplicationProgress:
    status: str
    rows_processed: int
    detail: str


def build_embedding_text(row: pd.Series, columns: list[str]) -> str:
    parts: list[str] = []
    for column in columns:
        value = row.get(column)
        if pd.isna(value):
            continue
        parts.append(str(value).strip())
    return "\n".join(part for part in parts if part)


def prepare_rows(
    df: pd.DataFrame,
    source_columns: list[ColumnInfo],
    source: SourceTableConfig,
    target: TargetTableConfig,
    embedding_config: EmbeddingConfig,
    embedding_texts: list[str],
    vectors: list[list[float] | None],
) -> list[dict[str, object]]:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows: list[dict[str, object]] = []
    copied_column_names = [c.name for c in source_columns]
    for (_, row), embedding_text, vector in zip(df.iterrows(), embedding_texts, vectors):
        output = {column: row.get(column) for column in copied_column_names}
        output[target.embedding_text_column] = embedding_text or None
        output[target.vector_column] = vector
        output["embedding_model"] = embedding_config.model_name if vector is not None else None
        output["embedding_created_at"] = now if vector is not None else None
        rows.append(output)
    return rows


def embed_non_empty_texts(
    embedder: EmbeddingClient,
    embedding_texts: list[str],
) -> tuple[list[list[float] | None], int]:
    vectors: list[list[float] | None] = [None] * len(embedding_texts)
    non_empty_positions = [index for index, text_value in enumerate(embedding_texts) if text_value.strip()]
    if not non_empty_positions:
        return vectors, 0

    non_empty_texts = [embedding_texts[index] for index in non_empty_positions]
    embedded_vectors = embedder.embed_texts(non_empty_texts)
    for index, vector in zip(non_empty_positions, embedded_vectors):
        vectors[index] = vector
    return vectors, len(non_empty_positions)


def run_replication(
    engine: Connection,
    source: SourceTableConfig,
    target: TargetTableConfig,
    embedding_config: EmbeddingConfig,
) -> Iterator[ReplicationProgress]:
    if not validate_traditional_chinese_fulltext(engine):
        raise RuntimeError("Traditional Chinese full-text language LCID 1028 is not installed on this SQL Server.")

    source_columns = get_columns(engine, source.schema_name, source.table_name)
    if not source_columns:
        raise RuntimeError("Source table was not found or has no columns.")

    source_column_names = [c.name for c in source_columns]
    missing = [
        c
        for c in [source.key_column, *source.embedding_columns]
        if not any(identifiers_equal(c, source_column) for source_column in source_column_names)
    ]
    if missing:
        raise RuntimeError(f"Missing source columns: {', '.join(missing)}")

    if target.load_mode == "drop_recreate":
        yield ReplicationProgress("setup", 0, "Dropping target table")
        drop_target_table(engine, target)

    yield ReplicationProgress("setup", 0, "Creating target table")
    create_target_table(engine, source_columns, source, target, embedding_config.dimensions)

    if target.load_mode == "create_only_if_empty":
        if target_table_has_rows(engine, target):
            raise RuntimeError(
                "Target table already contains rows. Choose truncate and rebuild, or use an empty target table."
            )
    elif target.load_mode == "truncate":
        yield ReplicationProgress("setup", 0, "Truncating target table")
        truncate_target_table(engine, target)
    elif target.load_mode == "drop_recreate":
        pass
    else:
        raise RuntimeError(f"Unsupported target load mode: {target.load_mode}")

    embedder = EmbeddingClient(embedding_config)
    rows_processed = 0
    last_key: object | None = None

    while True:
        df = read_source_batch(engine, source, last_key)
        if df.empty:
            break

        texts = [build_embedding_text(row, source.embedding_columns) for _, row in df.iterrows()]
        non_empty_count = sum(1 for text in texts if text.strip())
        empty_count = len(texts) - non_empty_count
        yield ReplicationProgress(
            "embedding",
            rows_processed,
            f"Embedding {non_empty_count} rows; {empty_count} rows have empty embedding text",
        )
        vectors, embedded_count = embed_non_empty_texts(embedder, texts)

        rows = prepare_rows(df, source_columns, source, target, embedding_config, texts, vectors)
        inserted = insert_target_rows(engine, rows, source_columns, source, target)
        rows_processed += inserted
        last_key = df[source.key_column].iloc[-1]
        yield ReplicationProgress(
            "insert",
            rows_processed,
            f"Inserted {inserted} rows; embedded {embedded_count}, skipped {empty_count} empty",
        )

    yield ReplicationProgress("indexing", rows_processed, "Creating Traditional Chinese full-text index")
    create_fulltext_index(engine, source_columns, source, target)

    if target.create_vector_index:
        yield ReplicationProgress("indexing", rows_processed, "Creating SQL Server vector index")
        create_vector_index(engine, target)

    yield ReplicationProgress("done", rows_processed, "Replication complete")
