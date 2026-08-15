"""
Load the active AI provider from PostgreSQL ai_providers (same table as Go).

API keys are AES-256-GCM blobs produced by cogniforge/internal/crypto.
Requires ENCRYPTION_KEY to match the Go backend.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ActiveProvider:
    id: str
    name: str
    provider: str
    base_url: str
    api_key: str
    default_model: str
    extra_headers: dict[str, str] = field(default_factory=dict)

    @property
    def openai_compatible(self) -> bool:
        return self.provider != "anthropic"


def derive_key(key_str: str) -> bytes:
    """Match Go crypto.deriveKey: UTF-8, pad/truncate to 32 bytes."""
    if len(key_str) < 16:
        raise ValueError("encryption key must be at least 16 characters")
    raw = key_str.encode("utf-8")
    if len(raw) > 32:
        raw = raw[:32]
    if len(raw) < 32:
        raw = raw + b"\x00" * (32 - len(raw))
    return raw


def decrypt_api_key(ciphertext_b64: str, key_str: str) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = derive_key(key_str)
    blob = base64.b64decode(ciphertext_b64)
    nonce_size = 12
    if len(blob) < nonce_size:
        raise ValueError("ciphertext too short")
    nonce, data = blob[:nonce_size], blob[nonce_size:]
    plaintext = AESGCM(key).decrypt(nonce, data, None)
    return plaintext.decode("utf-8")


def _normalize_headers(raw: Any) -> dict[str, str]:
    if not raw:
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if v is not None}


def load_active_provider() -> Optional[ActiveProvider]:
    """Same priority as Go GetActive: enabled default, else first enabled by priority."""
    enc_key = os.getenv("ENCRYPTION_KEY", "")
    if not enc_key:
        logger.warning("ENCRYPTION_KEY is empty; cannot read ai_providers")
        return None

    host = os.getenv("PGSQL_HOST", "localhost")
    port = int(os.getenv("PGSQL_PORT", "5432"))
    dbname = os.getenv("PGSQL_DB", "cogniforge")
    user = os.getenv("PGSQL_USERNAME", "postgres")
    password = os.getenv("PGSQL_PASSWORD", "")

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        logger.error("psycopg2 is required to load ai_providers")
        return None

    sql = """
        SELECT id, name, provider, base_url, api_key, default_model, extra_headers
        FROM ai_providers
        WHERE deleted_at IS NULL AND is_enabled = TRUE
        ORDER BY is_default DESC, priority ASC, created_at ASC
        LIMIT 1
    """
    conn = None
    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=dbname, user=user, password=password
        )
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            row = cur.fetchone()
    except Exception as e:
        logger.error("failed to query ai_providers: %s", e)
        return None
    finally:
        if conn is not None:
            conn.close()

    if not row:
        logger.warning("no enabled row in ai_providers")
        return None

    try:
        api_key = decrypt_api_key(row["api_key"] or "", enc_key)
    except Exception as e:
        logger.error("failed to decrypt ai_providers.api_key id=%s: %s", row.get("id"), e)
        return None

    active = ActiveProvider(
        id=str(row["id"]),
        name=str(row.get("name") or row["id"]),
        provider=str(row.get("provider") or "openai"),
        base_url=str(row.get("base_url") or "").rstrip("/"),
        api_key=api_key,
        default_model=str(row.get("default_model") or ""),
        extra_headers=_normalize_headers(row.get("extra_headers")),
    )
    logger.info(
        "loaded ai_providers id=%s provider=%s model=%s",
        active.id,
        active.provider,
        active.default_model,
    )
    return active


def build_providers(active: Optional[ActiveProvider] = None) -> dict:
    """Build the in-process provider map from the Models-page row."""
    from llm.anthropic_provider import AnthropicProvider
    from llm.openai_provider import OpenAIProvider

    if active is None:
        active = load_active_provider()
    if active is None:
        return {}

    if active.provider == "anthropic":
        impl = AnthropicProvider(api_key=active.api_key)
    else:
        impl = OpenAIProvider(
            api_key=active.api_key,
            base_url=active.base_url or None,
            default_model=active.default_model or None,
            extra_headers=active.extra_headers or None,
            name=active.provider,
        )

    providers = {
        active.provider: impl,
        "default": impl,
    }
    # Existing /api/llm clients default to provider=openai
    providers.setdefault("openai", impl)
    return providers
