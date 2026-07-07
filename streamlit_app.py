from __future__ import annotations

import streamlit as st

from replicator.config import (
    FULLTEXT_TRADITIONAL_CHINESE_LCID,
    EmbeddingConfig,
    SourceTableConfig,
    SqlServerConfig,
    TargetTableConfig,
)
from replicator.pipeline import run_replication
from replicator.sqlserver import (
    get_columns,
    list_tables,
    make_engine,
    read_source_batch,
    test_connection,
    validate_traditional_chinese_fulltext,
)


st.set_page_config(page_title="MSSQL Hybrid Search Replicator", layout="wide")

st.title("MSSQL Hybrid Search Replicator")


@st.cache_resource(show_spinner=False)
def get_engine(sql_config: SqlServerConfig):
    return make_engine(sql_config)


def get_engine_from_state():
    sql_config = st.session_state.get("sql_config")
    if not sql_config:
        return None
    return get_engine(sql_config)


with st.sidebar:
    st.header("SQL Server")
    server = st.text_input("Server", value="192.192.3.170")
    port = st.number_input("Port", min_value=1, max_value=65535, value=1433)
    database = st.text_input("Database", value="CustomerDB")
    username = st.text_input("Username", value="sa")
    password = st.text_input("Password", type="password")

    if st.button("Connect", type="primary", use_container_width=True):
        try:
            config = SqlServerConfig(
                server=server,
                port=int(port),
                database=database,
                username=username,
                password=password,
            )
            engine = get_engine(config)
            version = test_connection(engine)
            has_zh_tw = validate_traditional_chinese_fulltext(engine)
            st.session_state["sql_config"] = config
            st.session_state["server_version"] = version
            st.session_state["has_zh_tw_fulltext"] = has_zh_tw
            st.success("Connected")
        except Exception as exc:
            st.error(f"Connection failed: {exc}")

engine = get_engine_from_state()

if engine is None:
    st.info("Configure SQL Server in the sidebar and connect to continue.")
    st.stop()

with st.expander("Connection Status", expanded=True):
    st.code(st.session_state.get("server_version", "Connected"))
    if st.session_state.get("has_zh_tw_fulltext"):
        st.success(f"Traditional Chinese full-text language LCID {FULLTEXT_TRADITIONAL_CHINESE_LCID} is available.")
    else:
        st.error(f"Traditional Chinese full-text language LCID {FULLTEXT_TRADITIONAL_CHINESE_LCID} is not available.")

try:
    tables = list_tables(engine)
except Exception as exc:
    st.error(f"Could not load table list: {exc}")
    st.stop()

if not tables:
    st.warning("No base tables were found in this database.")
    st.stop()

left, right = st.columns(2)

with left:
    st.subheader("Source Table")
    table_labels = [f"{schema}.{table}" for schema, table in tables]
    selected_label = st.selectbox("Table", table_labels)
    selected_schema, selected_table = selected_label.split(".", 1)
    columns = get_columns(engine, selected_schema, selected_table)
    column_names = [c.name for c in columns]
    text_column_names = [c.name for c in columns if c.is_text]

    key_column = st.selectbox("Stable unique key column", column_names)
    embedding_columns = st.multiselect(
        "Columns used to calculate embeddings",
        column_names,
        default=text_column_names[: min(2, len(text_column_names))],
    )
    batch_size = st.number_input("Batch size", min_value=1, max_value=1000, value=100, step=50)

with right:
    st.subheader("Target Table")
    target_schema = st.text_input("Target schema", value=selected_schema)
    target_table = st.text_input("Target table", value=f"{selected_table}_hybrid")
    load_mode_label = st.radio(
        "Load behavior",
        ["Create only if empty", "Truncate target and rebuild", "Drop and recreate target table"],
        horizontal=True,
    )
    vector_column = st.text_input("Vector column", value="embedding")
    embedding_text_column = st.text_input("Embedding text column", value="embedding_text")
    fulltext_catalog = st.text_input("Full-text catalog", value="ft_catalog_hybrid_search")
    st.text_input("Full-text language", value="Traditional Chinese (LCID 1028)", disabled=True)
    create_vector_index = st.checkbox("Create vector index after load", value=False)
    vector_metric = st.selectbox("Vector metric", ["cosine", "euclidean", "dot"])

st.subheader("Embedding API")
api_left, api_right = st.columns(2)
with api_left:
    base_url = st.text_input("OpenAI-compatible base URL", value="http://llm-proxy:4000/v1")
    model_name = st.text_input("Embedding model", value="embedding")
with api_right:
    api_key = st.text_input("API key", value="abcd", type="password")
    dimensions = st.number_input("Embedding dimensions", min_value=1, max_value=1998, value=1024)

preview_col, run_col = st.columns([1, 1])

with preview_col:
    if st.button("Preview Source Batch", use_container_width=True):
        try:
            source_config = SourceTableConfig(
                schema_name=selected_schema,
                table_name=selected_table,
                key_column=key_column,
                embedding_columns=embedding_columns,
                batch_size=int(batch_size),
            )
            preview = read_source_batch(engine, source_config, None)
            st.dataframe(preview, use_container_width=True)
        except Exception as exc:
            st.error(f"Preview failed: {exc}")

with run_col:
    truncate_selected = load_mode_label == "Truncate target and rebuild"
    drop_recreate_selected = load_mode_label == "Drop and recreate target table"
    destructive_confirmed = True
    if truncate_selected:
        destructive_confirmed = st.checkbox("I understand this will delete all rows in the target table.")
    if drop_recreate_selected:
        destructive_confirmed = st.checkbox("I understand this will drop the target table and all target indexes.")

    run_disabled = (
        not embedding_columns
        or not api_key
        or not st.session_state.get("has_zh_tw_fulltext")
        or ((truncate_selected or drop_recreate_selected) and not destructive_confirmed)
    )
    if st.button("Create Target and Replicate", type="primary", disabled=run_disabled, use_container_width=True):
        source_config = SourceTableConfig(
            schema_name=selected_schema,
            table_name=selected_table,
            key_column=key_column,
            embedding_columns=embedding_columns,
            batch_size=int(batch_size),
        )
        target_config = TargetTableConfig(
            schema_name=target_schema,
            table_name=target_table,
            load_mode=(
                "drop_recreate"
                if drop_recreate_selected
                else "truncate"
                if truncate_selected
                else "create_only_if_empty"
            ),
            vector_column=vector_column,
            embedding_text_column=embedding_text_column,
            fulltext_catalog=fulltext_catalog,
            create_vector_index=create_vector_index,
            vector_metric=vector_metric,
        )
        embedding_config = EmbeddingConfig(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            dimensions=int(dimensions),
        )

        progress_bar = st.progress(0)
        log = st.empty()
        try:
            last_count = 0
            for progress in run_replication(engine, source_config, target_config, embedding_config):
                last_count = progress.rows_processed
                progress_bar.progress(100 if progress.status == "done" else min(95, max(5, last_count % 100)))
                log.info(f"{progress.status}: {progress.detail} ({progress.rows_processed} rows)")
            st.success(f"Replication finished. Rows processed: {last_count}")
        except Exception as exc:
            st.error(f"Replication failed: {exc}")

with st.expander("Generated Traditional Chinese Full-Text Behavior"):
    st.markdown(
        """
The generated full-text index uses `LANGUAGE 1028` for selected text columns, targeting Traditional Chinese tokenization.
The application validates `sys.fulltext_languages` before enabling replication.
"""
    )
