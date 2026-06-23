"""Offline tests for the RAG chunking logic (#B1).

These exercise the pure string->chunks functions only — no network (no docs
clone) and no OpenAI (no embeddings). The test skips if the optional ``rag``
extra (tiktoken) isn't installed, mirroring how #0's smoke tests run keyless.
"""

import pytest

pytest.importorskip("tiktoken")

from agent.rag import ingest  # noqa: E402

SAMPLE = """\
# FastAPI

Intro paragraph.

## Tutorial

Tutorial intro.

### Request Body

Use a Pydantic model to declare a request body.

## Deployment

Ship it.
"""


def test_sections_track_heading_breadcrumb():
    sections = ingest._iter_sections(SAMPLE)
    crumbs = [c for c, _ in sections]
    assert "FastAPI" in crumbs
    assert "FastAPI > Tutorial" in crumbs
    # A deeper heading nests under its parents...
    assert "FastAPI > Tutorial > Request Body" in crumbs
    # ...and a sibling H2 drops back out of the H3.
    assert "FastAPI > Deployment" in crumbs


def test_chunk_markdown_carries_provenance():
    chunks = ingest.chunk_markdown(SAMPLE, source="index.md")
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert c.source == "index.md"
        assert c.id.startswith("index.md::")
        # The breadcrumb is prepended into the embedded text and kept as metadata.
        if c.section:
            assert c.text.startswith(c.section)
    # IDs are unique (stable per-file index -> upsert-friendly).
    assert len({c.id for c in chunks}) == len(chunks)
    # The Request Body chunk should mention Pydantic.
    assert any("Pydantic" in c.text for c in chunks)


def test_long_section_splits_with_overlap():
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    long_text = " ".join(f"word{i}" for i in range(3000))  # well over MAX_TOKENS

    pieces = ingest._split_tokens(long_text, enc, ingest.MAX_TOKENS, ingest.OVERLAP)
    assert len(pieces) > 1
    # Each window respects the ceiling.
    for p in pieces:
        assert len(enc.encode(p)) <= ingest.MAX_TOKENS
    # Consecutive windows overlap: the tail of one reappears at the head of the next.
    tail = enc.encode(pieces[0])[-ingest.OVERLAP:]
    head = enc.encode(pieces[1])[: ingest.OVERLAP]
    assert tail == head


def test_short_text_stays_single_chunk():
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    assert ingest._split_tokens("just a few words", enc, ingest.MAX_TOKENS, ingest.OVERLAP) == [
        "just a few words"
    ]
