from __future__ import annotations

import argparse
import sys
from pathlib import Path

from classifyrag.blank_page import build_blank_index_from_dir
from classifyrag.colsmol_scorer import DEFAULT_MODEL_ID


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build ColSmol embeddings for blank-page reference images.")
    p.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("data/blank_data"),
        help="Directory with blank reference images (png/jpg/…).",
    )
    p.add_argument("--output", type=Path, default=Path("data/index/blank_prototypes.pt"), help="Output .pt index.")
    p.add_argument("--manifest-json", type=Path, default=Path("data/index/blank_manifest.json"))
    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", type=str, default=None, help="cuda:0 | cpu | mps (default: auto)")
    args = p.parse_args(argv)

    if not args.samples_dir.is_dir():
        print(f"Not a directory: {args.samples_dir}", file=sys.stderr)
        return 1

    try:
        build_blank_index_from_dir(
            args.samples_dir,
            args.output,
            manifest_json=args.manifest_json,
            model_id=args.model_id,
            device=args.device,
            batch_size=args.batch_size,
        )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    print(f"Wrote {args.output} and {args.manifest_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
