from llm.model_config import KEY_REV, KEY_SNAPSHOT, ModelConfigCache, Snapshot


class FakeRedis(dict):
    def get(self, key):
        return super().get(key)


def test_rev_mismatch_drops_local():
    r = FakeRedis()
    r[KEY_REV] = "1"
    r[KEY_SNAPSHOT] = '{"rev":1,"id":"deepseek","default_model":"deepseek-chat","models":[{"id":"deepseek-chat","name":"deepseek-chat"}]}'
    cache = ModelConfigCache(redis_client=r, ttl=60)
    snap = cache.get()
    assert snap is not None
    assert snap.default_model == "deepseek-chat"

    r[KEY_REV] = "2"
    del r[KEY_SNAPSHOT]
    assert cache.get() is None


def test_snapshot_from_dict_ignores_encrypted_key():
    snap = Snapshot.from_dict(
        {
            "rev": 3,
            "id": "x",
            "encrypted_key": "should-be-ignored-by-python",
            "default_model": "gpt-4o",
            "models": [{"id": "gpt-4o", "name": "gpt-4o"}],
        }
    )
    assert snap.default_model == "gpt-4o"
    assert not hasattr(snap, "encrypted_key") or "should-be-ignored-by-python" not in str(snap)
