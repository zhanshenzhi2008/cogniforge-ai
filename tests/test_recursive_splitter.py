from services.rag.splitters.recursive_splitter import RecursiveCharacterSplitter


def test_short_text_does_not_explode_with_overlap():
    """613 chars / chunk 512 / overlap 50 used to stall at start=end-overlap."""
    text = "甲" * 563 + "。" + "乙" * 49
    splitter = RecursiveCharacterSplitter(chunk_size=512, overlap=50)
    chunks = splitter.split(text)
    assert 1 <= len(chunks) <= 5
    assert all(c.content for c in chunks)


def test_empty_text():
    splitter = RecursiveCharacterSplitter(chunk_size=512, overlap=50)
    assert splitter.split("") == []
