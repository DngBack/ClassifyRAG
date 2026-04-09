#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import fitz  # PyMuPDF


def main() -> int:
    p = argparse.ArgumentParser(description="Export a page range from a PDF to a new PDF.")
    p.add_argument("--input", type=Path, required=True, help="Input PDF path.")
    p.add_argument("--output", type=Path, required=True, help="Output PDF path.")
    p.add_argument("--start", type=int, default=0, help="Start page index (0-based, inclusive).")
    p.add_argument("--end", type=int, required=True, help="End page index (0-based, inclusive).")
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input PDF not found: {args.input}")
    if args.start < 0 or args.end < args.start:
        raise SystemExit("Invalid page range. Require: 0 <= start <= end.")

    src = fitz.open(args.input)
    total = src.page_count
    if args.end >= total:
        raise SystemExit(f"End page out of range: {args.end} (total pages: {total})")

    out = fitz.open()
    out.insert_pdf(src, from_page=args.start, to_page=args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output)
    out.close()
    src.close()

    print(f"Wrote pages {args.start}-{args.end} ({args.end - args.start + 1} pages) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
