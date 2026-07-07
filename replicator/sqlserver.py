from __future__ import annotations

import ctypes.util
import json
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import (
    FULLTEXT_TRADITIONAL_CHINESE_LCID,
    SourceTableConfig,
    SqlServerConfig,
    TargetTableConfig,
)


TEXT_TYPES = {"char", "varchar", "text", "nchar", "nvarchar", "ntext"}
UNSUPPORTED_COPY_TYPES = {"timestamp", "rowversion"}


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    data_type: str
    max_length: int | None
    numeric_precision: int | None
    numeric_scale: int | None
    is_nullable: bool
    ordinal_position: int

    @property
    def is_text(self) -> bool:
        return self.data_type.lower() in TEXT_TYPES


def quote_identifier(name: str) -> str:
    if not name or "\x00" in name:
        raise ValueError("Identifier cannot be empty or contain null bytes.")
    return f"[{name.replace(']', ']]')}]"


def sql_string_literal(value: str) -> str:
    return "N'" + value.replace("'", "''") + "'"


def qualified_name(schema_name: str, table_name: str) -> str:
    return f"{quote_identifier(schema_name)}.{quote_identifier(table_name)}"


def assert_odbc_runtime_available() -> None:
    if ctypes.util.find_library("odbc") is None:
        raise RuntimeError(
            "UnixODBC runtime is not installed. On Ubuntu/WSL, install unixodbc and "
            "Microsoft ODBC Driver 18 for SQL Server, then restart Streamlit."
        )


def make_engine(config: SqlServerConfig) -> Engine:
    assert_odbc_runtime_available()
    raw = (
        f"DRIVER={{{config.driver}}};"
        f"SERVER={config.server},{config.port};"
        f"DATABASE={config.database};"
        f"UID={config.username};"
        f"PWD={config.password};"
        f"Encrypt={'yes' if config.encrypt else 'no'};"
        f"TrustServerCertificate={'yes' if config.trust_server_certificate else 'no'};"
    )
    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={quote_plus(raw)}",
        fast_executemany=True,
        pool_pre_ping=True,
    )


def test_connection(engine: Engine) -> str:
    with engine.connect() as conn:
        row = conn.execute(text("SELECT @@VERSION AS version")).one()
        return str(row.version)


def list_tables(engine: Engine) -> list[tuple[str, str]]:
    sql = text(
        """
        SELECT TABLE_SCHEMA, TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_SCHEMA, TABLE_NAME
        """
    )
    with engine.connect() as conn:
        return [(r.TABLE_SCHEMA, r.TABLE_NAME) for r in conn.execute(sql)]


def get_columns(engine: Engine, schema_name: str, table_name: str) -> list[ColumnInfo]:
    sql = text(
        """
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            CHARACTER_MAXIMUM_LENGTH,
            NUMERIC_PRECISION,
            NUMERIC_SCALE,
            IS_NULLABLE,
            ORDINAL_POSITION
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :schema_name
          AND TABLE_NAME = :table_name
        ORDER BY ORDINAL_POSITION
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"schema_name": schema_name, "table_name": table_name})
        return [
            ColumnInfo(
                name=r.COLUMN_NAME,
                data_type=r.DATA_TYPE,
                max_length=r.CHARACTER_MAXIMUM_LENGTH,
                numeric_precision=r.NUMERIC_PRECISION,
                numeric_scale=r.NUMERIC_SCALE,
                is_nullable=r.IS_NULLABLE == "YES",
                ordinal_position=r.ORDINAL_POSITION,
            )
            for r in rows
        ]


def validate_traditional_chinese_fulltext(engine: Engine) -> bool:
    sql = text("SELECT 1 FROM sys.fulltext_languages WHERE lcid = :lcid")
    with engine.connect() as conn:
        return conn.execute(sql, {"lcid": FULLTEXT_TRADITIONAL_CHINESE_LCID}).first() is not None


def column_definition(column: ColumnInfo) -> str:
    data_type = column.data_type.lower()
    nullable = "NULL" if column.is_nullable else "NOT NULL"
    name = quote_identifier(column.name)

    if data_type in {"varchar", "char", "nvarchar", "nchar", "binary", "varbinary"}:
        if column.max_length == -1:
            size = "MAX"
        elif data_type.startswith("n") and column.max_length is not None:
            size = str(column.max_length // 2)
        else:
            size = str(column.max_length)
        return f"{name} {data_type.upper()}({size}) {nullable}"

    if data_type in {"decimal", "numeric"}:
        precision = column.numeric_precision or 18
        scale = column.numeric_scale or 0
        return f"{name} {data_type.upper()}({precision},{scale}) {nullable}"

    if data_type in {"datetime2", "datetimeoffset", "time"} and column.numeric_scale is not None:
        return f"{name} {data_type.upper()}({column.numeric_scale}) {nullable}"

    return f"{name} {data_type.upper()} {nullable}"


def build_create_target_table_sql(
    source_columns: Iterable[ColumnInfo],
    source: SourceTableConfig,
    target: TargetTableConfig,
    embedding_dimensions: int,
) -> str:
    columns = [
        column_definition(c)
        for c in source_columns
        if c.data_type.lower() not in UNSUPPORTED_COPY_TYPES
    ]
    columns.extend(
        [
            f"{quote_identifier(target.embedding_text_column)} NVARCHAR(MAX) NULL",
            f"{quote_identifier(target.vector_column)} VECTOR({embedding_dimensions}) NULL",
            "[embedding_model] NVARCHAR(256) NULL",
            "[embedding_created_at] DATETIME2(3) NULL",
        ]
    )
    key = quote_identifier(source.key_column)
    target_object = f"{target.schema_name}.{target.table_name}"
    column_sql = ",\n        ".join(columns)
    return f"""
IF OBJECT_ID({sql_string_literal(target_object)}, N'U') IS NULL
BEGIN
    CREATE TABLE {qualified_name(target.schema_name, target.table_name)}
    (
        {column_sql},
        CONSTRAINT {quote_identifier("PK_" + target.table_name)} PRIMARY KEY ({key})
    );
END
"""


def create_target_table(
    engine: Engine,
    source_columns: list[ColumnInfo],
    source: SourceTableConfig,
    target: TargetTableConfig,
    embedding_dimensions: int,
) -> None:
    sql = build_create_target_table_sql(source_columns, source, target, embedding_dimensions)
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)


def target_table_has_rows(engine: Engine, target: TargetTableConfig) -> bool:
    sql = text(f"SELECT TOP (1) 1 FROM {qualified_name(target.schema_name, target.table_name)}")
    with engine.connect() as conn:
        return conn.execute(sql).first() is not None


def truncate_target_table(engine: Engine, target: TargetTableConfig) -> None:
    sql = f"TRUNCATE TABLE {qualified_name(target.schema_name, target.table_name)}"
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)


def create_fulltext_index(
    engine: Engine,
    source_columns: list[ColumnInfo],
    source: SourceTableConfig,
    target: TargetTableConfig,
) -> None:
    text_columns = [c for c in source_columns if c.name in source.embedding_columns and c.is_text]
    if not text_columns:
        raise ValueError("At least one selected embedding column must be a text column for full-text indexing.")

    table = qualified_name(target.schema_name, target.table_name)
    catalog = quote_identifier(target.fulltext_catalog)
    ft_columns = ",\n        ".join(
        f"{quote_identifier(c.name)} LANGUAGE {target.fulltext_language_lcid}" for c in text_columns
    )
    pk_name = quote_identifier("PK_" + target.table_name)
    sql = f"""
IF NOT EXISTS (SELECT 1 FROM sys.fulltext_catalogs WHERE name = {sql_string_literal(target.fulltext_catalog)})
    CREATE FULLTEXT CATALOG {catalog} AS DEFAULT;

IF NOT EXISTS (
    SELECT 1
    FROM sys.fulltext_indexes i
    JOIN sys.objects o ON i.object_id = o.object_id
    JOIN sys.schemas s ON o.schema_id = s.schema_id
    WHERE s.name = {sql_string_literal(target.schema_name)}
      AND o.name = {sql_string_literal(target.table_name)}
)
BEGIN
    CREATE FULLTEXT INDEX ON {table}
    (
        {ft_columns}
    )
    KEY INDEX {pk_name}
    ON {catalog}
    WITH CHANGE_TRACKING AUTO;
END
"""
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)


def create_vector_index(engine: Engine, target: TargetTableConfig) -> None:
    sql = f"""
IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = {sql_string_literal(target.vector_index_name)}
      AND object_id = OBJECT_ID({sql_string_literal(target.schema_name + "." + target.table_name)})
)
BEGIN
    CREATE VECTOR INDEX {quote_identifier(target.vector_index_name)}
    ON {qualified_name(target.schema_name, target.table_name)}({quote_identifier(target.vector_column)})
    WITH (METRIC = '{target.vector_metric}', TYPE = 'diskann');
END
"""
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)


def read_source_batch(
    engine: Engine,
    source: SourceTableConfig,
    last_key: object | None,
) -> pd.DataFrame:
    table = qualified_name(source.schema_name, source.table_name)
    key = quote_identifier(source.key_column)
    predicates: list[str] = []
    params: dict[str, object] = {}
    if last_key is not None:
        predicates.append(f"{key} > :last_key")
        params["last_key"] = last_key
    where_sql = f"WHERE {' AND '.join(predicates)}" if predicates else ""
    sql = text(f"SELECT TOP ({source.batch_size}) * FROM {table} {where_sql} ORDER BY {key}")
    return pd.read_sql_query(sql, engine, params=params)


def insert_target_rows(
    engine: Engine,
    rows: list[dict[str, object]],
    source_columns: list[ColumnInfo],
    source: SourceTableConfig,
    target: TargetTableConfig,
) -> int:
    if not rows:
        return 0
    copied_columns = [c.name for c in source_columns if c.data_type.lower() not in UNSUPPORTED_COPY_TYPES]
    extra_columns = [
        target.embedding_text_column,
        target.vector_column,
        "embedding_model",
        "embedding_created_at",
    ]
    all_columns = copied_columns + extra_columns
    insert_columns = ", ".join(quote_identifier(c) for c in all_columns)
    values = ", ".join(f":{c}" for c in all_columns)
    sql = text(f"INSERT INTO {qualified_name(target.schema_name, target.table_name)} ({insert_columns}) VALUES ({values})")
    clean_rows = []
    for row in rows:
        clean_row = {col: row.get(col) for col in all_columns}
        for key, value in clean_row.items():
            if not isinstance(value, list) and pd.isna(value):
                clean_row[key] = None
        vector = clean_row.get(target.vector_column)
        if isinstance(vector, list):
            clean_row[target.vector_column] = json.dumps(vector, ensure_ascii=False)
        clean_rows.append(clean_row)
    with engine.begin() as conn:
        conn.execute(sql, clean_rows)
    return len(clean_rows)
