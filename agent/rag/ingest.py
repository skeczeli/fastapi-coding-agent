"""RAG ingestion pipeline (#B1).

Fetches FastAPI's official docs (English markdown from ``fastapi/fastapi``),
splits each file into heading-aware chunks, embeds them with OpenAI, and stores
them in a persisted Chroma collection. Entry point: ``python -m agent.rag.ingest``.

Chunking strategy (documented for #E3):
- Split each ``.md`` by markdown headings, tracking the heading breadcrumb
  (e.g. ``Tutorial > Request Body``) so every chunk keeps its section context.
- Sections longer than ``MAX_TOKENS`` are sub-split into token windows with a
  small overlap, so no chunk blows past the embedding model's sweet spot and
  adjacent windows don't lose context at the seam.
- The breadcrumb is prepended to each chunk's text (helps retrieval) and also
  stored as metadata (``section``) for source attribution in #B2.

Run examples:
    python -m agent.rag.ingest --dry-run      # chunk + stats, no API cost
    python -m agent.rag.ingest --rebuild      # wipe + repopulate the store
    python -m agent.rag.ingest --limit 20     # only first 20 files (quick test)
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from agent.rag import store

# --- Source ---------------------------------------------------------------

# tiangolo/fastapi redirects here; English docs live under docs/en/docs.
DOCS_REPO = "https://github.com/fastapi/fastapi.git"
DOCS_SUBPATH = "docs/en/docs"
DEFAULT_DOCS_DIR = "docs/fastapi"  # gitignored; the clone lands here

# --- Chunking knobs -------------------------------------------------------

MAX_TOKENS = 800  # target ceiling per chunk (ticket suggests ~500–1000)
OVERLAP = 100  # token overlap between windows of an over-long section

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
# FastAPI headings carry explicit anchor ids, e.g. "Request Body { #request-body }".
# Strip them so breadcrumbs (embedded text + ``section`` metadata) read cleanly.
_ANCHOR = re.compile(r"\s*\{\s*#[^}]*\}")


@dataclass
class Chunk:
    """One embeddable unit of a doc, with provenance for attribution (#B2)."""

    id: str
    text: str  # breadcrumb + body, the string we embed and store
    source: str  # file path relative to the docs root
    section: str  # heading breadcrumb, e.g. "Tutorial > Request Body"


# --- Fetch ----------------------------------------------------------------


def fetch_docs(target: str = DEFAULT_DOCS_DIR, *, refresh: bool = False) -> Path:
    """Sparse-clone the FastAPI English docs and return the markdown root.

    Only ``DOCS_SUBPATH`` is checked out (blobless + sparse) so we don't pull the
    whole framework. If the clone already exists and ``refresh`` is False, the
    network step is skipped.
    """
    target_path = Path(target)
    md_root = target_path / DOCS_SUBPATH

    if md_root.exists() and not refresh:
        return md_root

    if not target_path.exists():
        subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "--filter=blob:none", "--sparse",
                DOCS_REPO, str(target_path),
            ],
            check=True,
        )
        # --no-cone: only DOCS_SUBPATH lands in the working tree. Cone mode would
        # also materialize the repo's root files (LICENSE, pyproject.toml, ...),
        # which we never ingest.
        subprocess.run(
            ["git", "-C", str(target_path), "sparse-checkout", "set", "--no-cone", DOCS_SUBPATH],
            check=True,
        )
    elif refresh:
        subprocess.run(["git", "-C", str(target_path), "pull", "--ff-only"], check=True)

    if not md_root.exists():
        raise FileNotFoundError(f"expected docs at {md_root} after fetch")
    return md_root


# --- Chunking (pure: string -> chunks, no network/API) --------------------


def _iter_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (breadcrumb, section_text) pairs by heading.

    A section runs from one heading up to the next heading of any level, and
    includes the heading line itself. The breadcrumb is the chain of enclosing
    headings (parents whose level is lower than the current one).
    """
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []  # (level, title) of open headings
    buf: list[str] = []
    breadcrumb = ""

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((breadcrumb, body))
        buf.clear()

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()  # close the previous section before opening a new one
            level = len(m.group(1))
            title = _ANCHOR.sub("", m.group(2)).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            breadcrumb = " > ".join(t for _, t in stack)
        buf.append(line)
    flush()
    return sections


def _split_tokens(text: str, enc, max_tokens: int, overlap: int) -> list[str]:
    """Split text into token windows of ``max_tokens`` with ``overlap`` between them."""
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return [text]
    out: list[str] = []
    start = 0
    step = max_tokens - overlap
    while start < len(toks):
        out.append(enc.decode(toks[start : start + max_tokens]))
        if start + max_tokens >= len(toks):
            break
        start += step
    return out


def chunk_markdown(text: str, source: str, enc=None) -> list[Chunk]:
    """Turn one markdown file into heading-aware, size-bounded chunks."""
    if enc is None:
        import tiktoken  # cl100k_base matches text-embedding-3-* tokenization

        enc = tiktoken.get_encoding("cl100k_base")

    chunks: list[Chunk] = []
    for breadcrumb, body in _iter_sections(text):
        for piece in _split_tokens(body, enc, MAX_TOKENS, OVERLAP):
            # Prepend the breadcrumb so the embedded text carries section context.
            full = f"{breadcrumb}\n\n{piece}" if breadcrumb else piece
            cid = f"{source}::{len(chunks)}"
            chunks.append(
                Chunk(id=cid, text=full, source=source, section=breadcrumb)
            )
    return chunks


def build_chunks(md_root: Path, *, limit: int | None = None) -> list[Chunk]:
    """Walk every ``.md`` under ``md_root`` and produce all chunks."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    files = sorted(md_root.rglob("*.md"))
    if limit is not None:
        files = files[:limit]

    chunks: list[Chunk] = []
    for path in files:
        source = str(path.relative_to(md_root))
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), source, enc))
    return chunks


# --- Orchestration --------------------------------------------------------


def ingest(
    md_root: Path | None = None,
    *,
    dry_run: bool = False,
    rebuild: bool = False,
    limit: int | None = None,
    refresh: bool = False,
) -> dict:
    """Run the full pipeline; returns a small stats dict.

    With ``dry_run`` it stops after chunking (no embeddings, no store) so you can
    inspect the chunking offline without spending on the API.
    """
    if md_root is None:
        md_root = fetch_docs(refresh=refresh)

    chunks = build_chunks(md_root, limit=limit)
    n_files = len({c.source for c in chunks})
    stats = {"files": n_files, "chunks": len(chunks)}

    if dry_run:
        print(f"[dry-run] {n_files} files -> {len(chunks)} chunks (nothing embedded)")
        return stats

    embeddings = store.embed_texts([c.text for c in chunks])
    collection = store.get_collection(reset=rebuild)
    collection.upsert(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[{"source": c.source, "section": c.section} for c in chunks],
    )
    print(f"[ingest] embedded {len(chunks)} chunks from {n_files} files into '{store.COLLECTION_NAME}'")
    return stats


def main() -> None:
    # Load .env so OPENAI_API_KEY is available when running as a CLI entrypoint.
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Ingest FastAPI docs into the RAG store.")
    parser.add_argument("--docs-dir", default=None, help="markdown root (skips fetch)")
    parser.add_argument("--dry-run", action="store_true", help="chunk + stats only, no API/store")
    parser.add_argument("--rebuild", action="store_true", help="wipe the collection first")
    parser.add_argument("--limit", type=int, default=None, help="cap number of files (quick test)")
    parser.add_argument("--refresh", action="store_true", help="re-pull the docs clone")
    args = parser.parse_args()

    ingest(
        md_root=Path(args.docs_dir) if args.docs_dir else None,
        dry_run=args.dry_run,
        rebuild=args.rebuild,
        limit=args.limit,
        refresh=args.refresh,
    )


if __name__ == "__main__":
    main()
