"""Generate a deterministic, incomplete corpus for readiness-audit demos."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .ingest import (
    DEFAULT_CORPUS_DIR,
    HEADING_RE,
    PROJECT_ROOT,
    SLUG_IN_TITLE_RE,
    ingest,
)


DEFAULT_DEGRADED_ROOT = PROJECT_ROOT / "data" / "degraded"
DEFAULT_DEGRADED_CORPUS_DIR = DEFAULT_DEGRADED_ROOT / "corpus"
DEFAULT_DEGRADED_INDEX_PATH = DEFAULT_DEGRADED_ROOT / "index" / "chunks.jsonl"
DEFAULT_DEGRADED_REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "degraded"

# This fixture intentionally combines a missing policy file with missing sections.
# Keep these omissions stable so audit regressions are reproducible.
OMITTED_FILES = ("refund_policy.md",)
OMITTED_SECTIONS: dict[str, tuple[str, ...]] = {
    "enterprise_plan_faq.md": (
        "enterprise_pricing_quote",
        "enterprise_support_response_time",
    ),
}


@dataclass(frozen=True)
class DegradedFixture:
    source_dir: str
    corpus_dir: str
    index_path: str
    indexed_chunks: int
    omitted_files: list[str]
    omitted_sections: dict[str, list[str]]


def _heading_slug(line: str) -> str | None:
    heading = HEADING_RE.match(line)
    if not heading:
        return None
    slug = SLUG_IN_TITLE_RE.search(heading.group(2))
    return slug.group(1).lower() if slug else None


def _without_sections(markdown: str, omitted: set[str]) -> str:
    """Return Markdown with selected heading sections removed."""

    output: list[str] = []
    skipping = False
    skipped_level = 0
    for line in markdown.splitlines(keepends=True):
        heading = HEADING_RE.match(line.rstrip("\r\n"))
        if heading:
            level = len(heading.group(1))
            slug = _heading_slug(line.rstrip("\r\n"))
            if slug in omitted:
                skipping = True
                skipped_level = level
                continue
            if skipping and level <= skipped_level:
                skipping = False
        if not skipping:
            output.append(line)
    return "".join(output).rstrip() + "\n"


def generate_degraded_fixture(
    *,
    source_dir: Path = DEFAULT_CORPUS_DIR,
    corpus_dir: Path = DEFAULT_DEGRADED_CORPUS_DIR,
    index_path: Path = DEFAULT_DEGRADED_INDEX_PATH,
) -> DegradedFixture:
    """Generate the incomplete corpus and index without modifying the source."""

    source = source_dir.resolve()
    output = corpus_dir.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("degraded corpus output must not be the primary corpus")
    if not source.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {source_dir}")

    corpus_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in corpus_dir.glob("*.md"):
        stale_path.unlink()

    omitted_files = set(OMITTED_FILES)
    for source_path in sorted(source_dir.glob("*.md")):
        if source_path.name in omitted_files:
            continue
        markdown = source_path.read_text(encoding="utf-8")
        omitted = set(OMITTED_SECTIONS.get(source_path.name, ()))
        if omitted:
            markdown = _without_sections(markdown, omitted)
        (corpus_dir / source_path.name).write_text(markdown, encoding="utf-8")

    count = ingest(corpus_dir, index_path)
    fixture = DegradedFixture(
        source_dir=str(source_dir),
        corpus_dir=str(corpus_dir),
        index_path=str(index_path),
        indexed_chunks=count,
        omitted_files=sorted(OMITTED_FILES),
        omitted_sections={
            name: sorted(sections) for name, sections in OMITTED_SECTIONS.items()
        },
    )
    manifest_path = corpus_dir / "fixture_manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(fixture), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_DEGRADED_CORPUS_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_DEGRADED_INDEX_PATH)
    args = parser.parse_args()

    fixture = generate_degraded_fixture(
        source_dir=args.source,
        corpus_dir=args.corpus_dir,
        index_path=args.index,
    )
    print(f"Generated degraded corpus: {fixture.corpus_dir}")
    print(f"Omitted files: {', '.join(fixture.omitted_files)}")
    for name, sections in fixture.omitted_sections.items():
        print(f"Omitted sections from {name}: {', '.join(sections)}")
    print(f"Indexed {fixture.indexed_chunks} degraded corpus chunks")
    print(f"Wrote: {fixture.index_path}")


if __name__ == "__main__":
    main()
