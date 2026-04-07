"""Evaluate page-level accuracy on data/Sample_document using the built index (single model load)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from classifyrag.characteristic_text import apply_characteristic_text
from classifyrag.colsmol_scorer import classify_page, load_index, load_model
from classifyrag.labels import label_from_filename
from classifyrag.pdf_pages import iter_pdf_pages


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Smoke-eval: classify all sample PDFs with one model load.")
    p.add_argument("--samples-dir", type=Path, default=Path("data/Sample_document"))
    p.add_argument("--index", type=Path, default=Path("data/index/prototypes.pt"))
    p.add_argument("--w-img", type=float, default=0.7)
    p.add_argument(
        "--mode",
        choices=("image", "fused"),
        default="image",
        help="Match classify_pdf: image or fused (text+VLM path not used in batch eval).",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ocr", action="store_true")
    p.add_argument("--ocr-dpi", type=int, default=150)
    p.add_argument("--ocr-lang", type=str, default="vie+eng")
    p.add_argument(
        "--characteristic-text",
        action="store_true",
        help="Match classify_pdf / index: use titles and field labels only for the text branch.",
    )
    args = p.parse_args(argv)

    pdfs = sorted(args.samples_dir.glob("*.pdf"))
    idx = load_index(args.index)
    model, processor, device = load_model(model_id=idx.model_id, device=args.device)
    proto_labels = [m["label"] for m in idx.manifest]

    total_pages = 0
    correct = 0
    for pdf in pdfs:
        exp = label_from_filename(pdf.name)
        if exp is None:
            continue
        for page in iter_pdf_pages(
            pdf,
            ocr=args.ocr,
            ocr_dpi=args.ocr_dpi,
            ocr_language=args.ocr_lang,
        ):
            qtext = apply_characteristic_text(page.text, args.characteristic_text)
            pred, _, _, _, _, _, _ = classify_page(
                processor=processor,
                device=device,
                query_image=page.image,
                query_text=qtext,
                proto_embs=idx.image_embs,
                proto_labels=proto_labels,
                model=model,
                w_img=args.w_img,
                proto_text_embs=idx.text_embs,
                pred_from=args.mode,
            )
            total_pages += 1
            if pred == exp:
                correct += 1
            else:
                print(f"Mismatch: {pdf.name} page {page.page_index} expected {exp} got {pred}", file=sys.stderr)

    print(f"Page accuracy: {correct}/{total_pages}")
    return 0 if correct == total_pages else 1


if __name__ == "__main__":
    raise SystemExit(main())
