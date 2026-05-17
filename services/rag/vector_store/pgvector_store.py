"""
PostgreSQL pgvector store implementation.
"""
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
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = port or int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = database or os.getenv("POSTGRES_DB", "cogniforge")
        self.user = user or os.getenv("POSTGRES_USER", "postgres")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "postgres")

        self._conn = None
        self._cursor = None

    def _get_connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )

    def connect(self) -> None:
        """Connect to PostgreSQL."""
        try:
            import psycopg2
            from psycopg2.extras import execute_values
            self.psycopg2 = psycopg2
            self.execute_values = execute_values

            self._conn = self.psycopg2.connect(self._get_connection_string())
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

    def create_collection(self, collection_name: str, dimension: int, **kwargs) -> None:
        """Create a collection (table) for vectors."""
        table_name = self._sanitize_table_name(collection_name)

        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id VARCHAR(64) PRIMARY KEY,
            text_content TEXT NOT NULL,
            metadata JSONB DEFAULT '{{}}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        try:
            self._cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS embedding vector({dimension})")
        except Exception as e:
            logger.warning(f"Could not add vector column: {e}")
            sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id VARCHAR(64) PRIMARY KEY,
                text_content TEXT NOT NULL,
                embedding_id INTEGER,
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            self._cursor.execute(sql)

        try:
            self._cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {table_name}_hnsw_idx
                ON {table_name}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)
        except Exception as e:
            logger.warning(f"Could not create HNSW index: {e}")

        self._conn.commit()

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection (table)."""
        table_name = self._sanitize_table_name(collection_name)
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

        for entry in entries:
            self._cursor.execute(
                f"""
                INSERT INTO {table_name} (id, text_content, metadata, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    text_content = EXCLUDED.text_content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                (entry.id, entry.text, entry.metadata, entry.vector)
            )
            ids.append(entry.id)

        self._conn.commit()
        return ids

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

        params = [query_vector]

        if filter_kwargs:
            filter_conditions = []
            for key, value in filter_kwargs.items():
                filter_conditions.append(f"metadata->>%s = %s")
                params.extend([key, str(value)])
            if filter_conditions:
                query += " WHERE " + " AND ".join(filter_conditions)

        query += f"""
            ORDER BY embedding <=> %s::vector
            LIMIT {top_k}
        """
        params.append(query_vector)

        self._cursor.execute(query, params)

        results = []
        for row in self._cursor.fetchall():
            results.append(SearchResult(
                id=row[0],
                text=row[1],
                score=float(row[2]) if row[2] is not None else 0.0,
                metadata=row[3] or {}
            ))

        return results

    def delete(self, collection_name: str, entry_id: str) -> None:
        """Delete an entry by ID."""
        table_name = self._sanitize_table_name(collection_name)
        self._cursor.execute(f"DELETE FROM {table_name} WHERE id = %s", (entry_id,))
        self._conn.commit()

    def get_dimension(self, collection_name: str) -> Optional[int]:
        """Get the dimension of vectors in a collection."""
        table_name = self._sanitize_table_name(collection_name)
        try:
            self._cursor.execute(
                f"SELECT atttypmod FROM pg_attribute WHERE attname = 'embedding' "
                f"AND attrelid = '{table_name}'::regclass"
            )
            result = self._cursor.fetchone()
            if result and result[0] > 0:
                return result[0] - 4
        except Exception as e:
            logger.warning(f"Could not get dimension: {e}")
        return None

    def _sanitize_table_name(self, name: str) -> str:
        """Sanitize table name to prevent SQL injection."""
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        return f"kb_{sanitized[:50]}"
