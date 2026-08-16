"""
Shared model-config cache (same Redis keys as Go internal/modelcache).

Java Caffeine  →  Python cachetools.TTLCache
Go 进程内 TTL  →  本模块 _Local
一致性：每次读先对 Redis cogniforge:modelcfg:rev，对不上就丢本地。
不明文缓存 API Key；encrypted_key 字段忽略。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from cachetools import TTLCache

logger = logging.getLogger(__name__)

KEY_REV = "cogniforge:modelcfg:rev"
KEY_SNAPSHOT = "cogniforge:modelcfg:snapshot"
LOCAL_TTL_SEC = 30


@dataclass
class ModelItem:
    id: str
    name: str


@dataclass
class Snapshot:
    rev: int = 0
    id: str = ""
    name: str = ""
    provider: str = ""
    base_url: str = ""
    default_model: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)
    models: list[ModelItem] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Snapshot":
        models = []
        for m in raw.get("models") or []:
            mid = str(m.get("id") or "")
            if mid:
                models.append(ModelItem(id=mid, name=str(m.get("name") or mid)))
        headers = raw.get("extra_headers") or {}
        return cls(
            rev=int(raw.get("rev") or 0),
            id=str(raw.get("id") or ""),
            name=str(raw.get("name") or ""),
            provider=str(raw.get("provider") or ""),
            base_url=str(raw.get("base_url") or ""),
            default_model=str(raw.get("default_model") or ""),
            extra_headers={str(k): str(v) for k, v in headers.items()},
            models=models,
        )


class _Local:
    """cachetools.TTLCache ≈ Java Caffeine（size + TTL）。rev 对不上视为未命中。"""

    def __init__(self, ttl: float = LOCAL_TTL_SEC):
        self._lock = threading.Lock()
        self._cache: TTLCache = TTLCache(maxsize=8, ttl=ttl)

    def get_if_rev(self, rev: int) -> Optional[Snapshot]:
        with self._lock:
            return self._cache.get(rev)

    def get_if_fresh(self) -> Optional[Snapshot]:
        with self._lock:
            return self._cache.get("_fresh")

    def set(self, snap: Snapshot) -> None:
        with self._lock:
            self._cache[snap.rev] = snap
            self._cache["_fresh"] = snap

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class ModelConfigCache:
    def __init__(self, redis_client=None, ttl: float = LOCAL_TTL_SEC):
        self._local = _Local(ttl=ttl)
        self._redis = redis_client

    def get(self) -> Optional[Snapshot]:
        if self._redis is None:
            return self._local.get_if_fresh()
        try:
            raw_rev = self._redis.get(KEY_REV)
            rev = int(raw_rev) if raw_rev is not None else 0
        except Exception as e:
            logger.warning("model cache redis rev failed: %s", e)
            return self._local.get_if_fresh()

        hit = self._local.get_if_rev(rev)
        if hit is not None:
            return hit

        try:
            raw = self._redis.get(KEY_SNAPSHOT)
        except Exception as e:
            logger.warning("model cache redis load failed: %s", e)
            return self._local.get_if_fresh()

        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            snap = Snapshot.from_dict(json.loads(raw))
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("model cache snapshot decode failed: %s", e)
            return None
        if snap.id and (rev == 0 or snap.rev == rev):
            self._local.set(snap)
            return snap
        return None


_cache: Optional[ModelConfigCache] = None


def _make_redis():
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD") or os.getenv("REDIS_PWD") or None
    db = int(os.getenv("REDIS_DB", "0"))
    try:
        import redis
    except ImportError:
        logger.warning("redis package missing; model config local-only")
        return None
    try:
        client = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            socket_connect_timeout=2,
            socket_timeout=0.5,
            decode_responses=True,
        )
        client.ping()
        logger.info("model config redis connected %s:%s", host, port)
        return client
    except Exception as e:
        logger.warning("model config redis unavailable: %s", e)
        return None


def init_cache() -> ModelConfigCache:
    global _cache
    _cache = ModelConfigCache(redis_client=_make_redis())
    return _cache


def get_cache() -> Optional[ModelConfigCache]:
    return _cache


def get_snapshot() -> Optional[Snapshot]:
    if _cache is None:
        return None
    return _cache.get()
