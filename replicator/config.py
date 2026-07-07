from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FULLTEXT_TRADITIONAL_CHINESE_LCID = 1028
LoadMode = Literal["create_only_if_empty", "truncate"]


@dataclass(frozen=True)
class SqlServerConfig:
    server: str
    database: str
    username: str
    password: str
    port: int = 1433


@dataclass(frozen=True)
class SourceTableConfig:
    schema_name: str
    table_name: str
    key_column: str
    embedding_columns: list[str]
    batch_size: int = 100


@dataclass(frozen=True)
class EmbeddingConfig:
    base_url: str
    model_name: str
    api_key: str
    dimensions: int


@dataclass(frozen=True)
class TargetTableConfig:
    schema_name: str
    table_name: str
    load_mode: LoadMode = "create_only_if_empty"
    vector_column: str = "embedding"
    embedding_text_column: str = "embedding_text"
    fulltext_catalog: str = "ft_catalog_hybrid_search"
    create_vector_index: bool = False
    vector_index_name: str = "ix_vector_embedding"
    vector_metric: Literal["cosine", "euclidean", "dot"] = "cosine"
    fulltext_language_lcid: int = FULLTEXT_TRADITIONAL_CHINESE_LCID
