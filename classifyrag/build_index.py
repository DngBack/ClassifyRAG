from __future__ import annotations

import argparse
import sys
from pathlib import Path

from classifyrag.colsmol_scorer import DEFAULT_MODEL_ID, embed_images, load_model, save_index, save_manifest_json
from classifyrag.labels import label_from_filename
from classifyrag.pdf_pages import iter_pdf_pages


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build ColSmol prototype index from labeled sample PDFs.")
    p.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("data/Sample_document"),
        help="Directory containing sample PDFs (filename prefix = label).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/index/prototypes.pt"),
        help="Output torch index path.",
    )
    p.add_argument("--manifest-json", type=Path, default=Path("data/index/manifest.json"), help="Human-readable manifest.")
    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--dpi", type=float, default=144.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", type=str, default=None, help="cuda:0 | cpu | mps (default: auto)")
    p.add_argument(
        "--ocr",
        action="store_true",
        help="When the PDF has no text layer, run Tesseract OCR (requires system tesseract).",
    )
    p.add_argument("--ocr-dpi", type=int, default=150)
    p.add_argument("--ocr-lang", type=str, default="vie+eng")
    args = p.parse_args(argv)

    pdfs = sorted(args.samples_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {args.samples_dir}", file=sys.stderr)
        return 1

    manifest: list[dict] = []
    images: list = []
    for pdf_path in pdfs:
        lab = label_from_filename(pdf_path.name)
        if lab is None:
            print(f"Skip (unknown prefix): {pdf_path.name}", file=sys.stderr)
            continue
        for page in iter_pdf_pages(
            pdf_path,
            dpi=args.dpi,
            ocr=args.ocr,
            ocr_dpi=args.ocr_dpi,
            ocr_language=args.ocr_lang,
        ):
            manifest.append(
                {
                    "label": lab,
                    "source_file": str(pdf_path.resolve()),
                    "page_index": page.page_index,
                }
            )
            images.append(page.image)

    if not images:
        print("No pages indexed.", file=sys.stderr)
        return 1

    print(f"Indexing {len(images)} prototype pages from {args.samples_dir} ...")
    model, processor, device = load_model(model_id=args.model_id, device=args.device)
    embs = embed_images(model, processor, images, device=device, batch_size=args.batch_size)

    save_index(args.output, args.model_id, manifest, embs)
    save_manifest_json(args.manifest_json, manifest)
    print(f"Wrote {args.output} and {args.manifest_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
