from __future__ import annotations


def split_text(text: str, *, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """Simple production-safe splitter without external LangChain dependency."""
    clean = ' '.join(text.split())
    if not clean:
        return []
    if chunk_size <= overlap:
        raise ValueError('chunk_size must be greater than overlap')

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))

        # Prefer ending on a sentence/space near the boundary.
        if end < len(clean):
            boundary = max(clean.rfind('. ', start, end), clean.rfind(' ', start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1

        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(clean):
            break
        start = max(0, end - overlap)

    return chunks
