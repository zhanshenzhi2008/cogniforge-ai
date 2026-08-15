import base64

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from llm.active_config import decrypt_api_key, derive_key


def test_derive_key_pads_to_32():
    key = derive_key("sixteen-char-key")
    assert len(key) == 32
    assert key.startswith(b"sixteen-char-key")
    assert key.endswith(b"\x00" * (32 - len("sixteen-char-key")))


def test_decrypt_matches_go_aes_gcm_layout():
    secret = "test-encryption-key-32bytes-ok!!"
    key = derive_key(secret)
    nonce = b"\x01" * 12
    plaintext = b"sk-test-deepseek-key"
    blob = nonce + AESGCM(key).encrypt(nonce, plaintext, None)
    encoded = base64.b64encode(blob).decode("ascii")
    assert decrypt_api_key(encoded, secret) == plaintext.decode("utf-8")
