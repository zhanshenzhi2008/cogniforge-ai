"""
PostgreSQL pgvector store implementation.
"""
import json
import logging
import os
import re
from typing import Dict, List, Optional, Any

from .base import BaseVectorStore, VectorEntry, SearchResult

logger = logging.getLogger(__name__)


class PGVectorStore(BaseVectorStore):
    """PostgreSQL with pgvector extension for vector storage."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.host = host or os.getenv("PGSQL_HOST", "localhost")
        self.port = port or int(os.getenv("PGSQL_PORT", "5432"))
        self.database = database or os.getenv("PGSQL_DB", "cogniforge")
        self.user = user or os.getenv("PGSQL_USERNAME", "postgres")
        self.password = password or os.getenv("PGSQL_PASSWORD", "")

        self._conn = None
        self._cursor = None

    def connect(self) -> None:
        """Connect to PostgreSQL.

        用关键字参数连接，避免密码含 @/# 等字符时 URI 被拆错
        （曾出现 host 被解析成「…#@pgsql」导致 RAG 起不来）。
        """
        try:
            import psycopg2
            from psycopg2.extras import execute_values
            self.psycopg2 = psycopg2
            self.execute_values = execute_values

            self._conn = self.psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.database,
                user=self.user,
                password=self.password,
            )
            self._cursor = self._conn.cursor()
            logger.info(f"Connected to PostgreSQL at {self.host}:{self.port}")

            self._cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._conn.commit()

        except ImportError:
            raise ImportError(
                "psycopg2 is required. Install with: pip install psycopg2-binary"
            )
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        if self._cursor:
            self._cursor.close()
        if self._conn:
            self._conn.close()
        logger.info("Disconnected from PostgreSQL")

    def _rollback_quiet(self) -> None:
        """出错后清掉 aborted 事务，避免共享连接被永久毒化。"""
        if not self._conn:
            return
        try:
            self._conn.rollback()
        except Exception as e:
            logger.warning("Rollback failed: %s", e)

    def create_collection(self, collection_name: str, dimension: int, **kwargs) -> None:
        """Create a collection (table) for vectors."""
        table_name = self._sanitize_table_name(collection_name)
        self._rollback_quiet()

        existing_dim = self.get_dimension(collection_name)
        if existing_dim is not None and existing_dim != dimension:
            logger.warning(
                "Recreating %s: embedding dim %s -> %s",
                table_name,
                existing_dim,
                dimension,
            )
            self.delete_collection(collection_name)

        try:
            self._cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id VARCHAR(64) PRIMARY KEY,
                    text_content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{{}}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding vector({dimension})
                )
                """
            )
        except Exception as e:
            self._rollback_quiet()
            logger.warning("Could not create table with embedding column: %s", e)
            self._cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id VARCHAR(64) PRIMARY KEY,
                    text_content TEXT NOT NULL,
                    embedding_id INTEGER,
                    metadata JSONB DEFAULT '{{}}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            try:
                self._cursor.execute(
                    f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS embedding vector({dimension})"
                )
            except Exception as alter_err:
                self._rollback_quiet()
                logger.warning("Could not add vector column: %s", alter_err)
                raise

        try:
            self._cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {table_name}_hnsw_idx
                ON {table_name}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                """
            )
        except Exception as e:
            self._rollback_quiet()
            logger.warning(f"Could not create HNSW index: {e}")

        self._conn.commit()

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection (table)."""
        table_name = self._sanitize_table_name(collection_name)
        self._rollback_quiet()
        self._cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
        self._conn.commit()

    def insert(self, collection_name: str, entry: VectorEntry) -> str:
        """Insert a single vector entry."""
        results = self.insert_batch(collection_name, [entry])
        return results[0] if results else entry.id

    def insert_batch(self, collection_name: str, entries: List[VectorEntry]) -> List[str]:
        """Insert multiple vector entries."""
        if not entries:
            return []

        table_name = self._sanitize_table_name(collection_name)
        ids = []

        try:
            for entry in entries:
                meta = entry.metadata if isinstance(entry.metadata, dict) else {}
                self._cursor.execute(
                    f"""
                    INSERT INTO {table_name} (id, text_content, metadata, embedding)
                    VALUES (%s, %s, %s::jsonb, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        text_content = EXCLUDED.text_content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,
                    (entry.id, entry.text, json.dumps(meta, ensure_ascii=False), entry.vector),
                )
                ids.append(entry.id)

            self._conn.commit()
            return ids
        except Exception:
            self._rollback_quiet()
            raise

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filter_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        table_name = self._sanitize_table_name(collection_name)

        query = f"""
            SELECT id, text_content, metadata,
                   1 - (embedding <=> %s::vector) as similarity
            FROM {table_name}
        """

        params: List[Any] = [query_vector]

        if filter_kwargs:
            filter_conditions = []
            for key, value in filter_kwargs.items():
                filter_conditions.append("metadata->>%s = %s")
                params.extend([key, str(value)])
            if filter_conditions:
                query += " WHERE " + " AND ".join(filter_conditions)

        query += f"""
            ORDER BY embedding <=> %s::vector
            LIMIT {int(top_k)}
        """
        params.append(query_vector)

        try:
            self._cursor.execute(query, params)
            rows = self._cursor.fetchall()
        except Exception:
            self._rollback_quiet()
            raise

        results = []
        for row in rows:
            results.append(
                SearchResult(
                    id=row[0],
                    text=row[1],
                    metadata=row[2] or {},
                    score=float(row[3]) if row[3] is not None else 0.0,
                )
            )

        return results

    def delete(self, collection_name: str, entry_id: str) -> None:
        """Delete an entry by ID."""
        table_name = self._sanitize_table_name(collection_name)
        try:
            self._cursor.execute(f"DELETE FROM {table_name} WHERE id = %s", (entry_id,))
            self._conn.commit()
        except Exception:
            self._rollback_quiet()
            raise

    def get_dimension(self, collection_name: str) -> Optional[int]:
        """Get the dimension of vectors in a collection."""
        table_name = self._sanitize_table_name(collection_name)
        try:
            self._cursor.execute(
                "SELECT atttypmod FROM pg_attribute WHERE attname = 'embedding' "
                "AND attrelid = %s::regclass",
                (table_name,),
            )
            result = self._cursor.fetchone()
            if result and result[0] > 0:
                return result[0] - 4
        except Exception as e:
            self._rollback_quiet()
            logger.warning(f"Could not get dimension: {e}")
        return None

    def _sanitize_table_name(self, name: str) -> str:
        """Sanitize table name to prevent SQL injection."""
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        return f"kb_{sanitized[:50]}"
