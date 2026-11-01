# Task: a Markdown table parser

Create `mdtable.py` exposing `parse_tables(markdown: str) -> list[Table]` and a pytest suite in
`test_mdtable.py`.

## Requirements

- `Table` is a dataclass with `headers: list[str]`, `alignments: list[str]` (each `"left"`, `"right"`,
  `"center"` or `"none"`, from the `:---`, `---:`, `:---:` delimiter row) and `rows: list[list[str]]`.
- Find every GitHub-flavoured pipe table in the document, in order, ignoring tables inside fenced
  code blocks (``` or ~~~).
- Cells are stripped of surrounding whitespace. Leading and trailing pipes are optional. An escaped
  pipe `\|` is part of the cell text, not a separator.
- Rows with fewer cells than the header are padded with `""`; extra cells are dropped.
- The delimiter row must have the same number of columns as the header, otherwise it is not a table.
- Add `Table.to_dicts()` returning one `{header: cell}` dict per row, and `Table.column(name)`
  returning that column's cells (`KeyError` for an unknown header).
- Standard library only.

## Edge cases the tests must cover

- no tables, two tables separated by text, a table at end of file without a trailing newline,
  CRLF line endings, empty cells, duplicate header names, unicode content.
