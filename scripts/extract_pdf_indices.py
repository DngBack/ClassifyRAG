#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import fitz  # PyMuPDF


def parse_pages(spec: str) -> list[int]:
    """
    Parse page spec like:
    - "0,2,5"
    - "0-4,8,10-12"
    Returns sorted unique 0-based page indices.
    """
    if not spec.strip():
        raise ValueError("Empty --pages spec.")
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_s, b_s = part.split("-", 1)
            a, b = int(a_s), int(b_s)
            if a < 0 or b < a:
                raise ValueError(f"Invalid range: {part}")
            out.update(range(a, b + 1))
        else:
            p = int(part)
            if p < 0:
                raise ValueError(f"Negative page index: {p}")
            out.add(p)
    if not out:
        raise ValueError("No pages parsed from --pages.")
    return sorted(out)


def default_output_name(input_pdf: Path, pages: list[int]) -> str:
    if not pages:
        return f"{input_pdf.stem}_pages.pdf"
    if len(pages) <= 12:
        chunk = "_".join(str(p) for p in pages)
        return f"{input_pdf.stem}_idx_{chunk}.pdf"
    return f"{input_pdf.stem}_idx_{pages[0]}_{pages[-1]}_{len(pages)}pages.pdf"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Extract selected page indices from a PDF into a new PDF."
    )
    p.add_argument("--input", type=Path, required=True, help="Input PDF path.")
    p.add_argument(
        "--pages",
        type=str,
        required=True,
        help='Page indices (0-based), e.g. "0,2,5,10-15".',
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path. If omitted, auto-name in same folder.",
    )
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    pages = parse_pages(args.pages)
    src = fitz.open(args.input)
    total = src.page_count
    bad = [i for i in pages if i >= total]
    if bad:
        raise SystemExit(f"Page indices out of range (total={total}): {bad}")

    out_path = (
        args.output
        if args.output is not None
        else args.input.with_name(default_output_name(args.input, pages))
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out = fitz.open()
    for i in pages:
        out.insert_pdf(src, from_page=i, to_page=i)
    out.save(out_path)
    out.close()
    src.close()

    print(f"Wrote {len(pages)} pages to {out_path}")
    print(f"Indices: {pages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
