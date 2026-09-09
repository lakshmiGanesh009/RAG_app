"""Split a parsed document into chunks small enough to embed.

Splits on document structure (headings, tables) rather than blind character
counts, and keeps the filename, page and heading on each chunk so answers can
cite their source.

To be implemented in Phase 3.
"""
