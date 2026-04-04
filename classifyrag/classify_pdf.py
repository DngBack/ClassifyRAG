from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from classifyrag.colsmol_scorer import classify_page, load_index, load_model
from classifyrag.llm_keywords import DEFAULT_VLM_MODEL, keywords_from_image_vlm
from classifyrag.pdf_pages import iter_pdf_pages


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Classify each page of a PDF using a ColSmol prototype index.")
    p.add_argument("--pdf", type=Path, required=True, help="Input PDF path.")
    p.add_argument("--index", type=Path, default=Path("data/index/prototypes.pt"), help="Index from build_index.")
    p.add_argument("--output", type=Path, required=True, help="Output .csv or .json path.")
    p.add_argument("--format", choices=("csv", "json"), default="csv")
    p.add_argument("--w-img", type=float, default=0.7, help="Weight for image branch (text gets 1-w).")
    p.add_argument("--dpi", type=float, default=144.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--ocr",
        action="store_true",
        help="If text layer is empty (typical for scanned PDFs), run Tesseract OCR via PyMuPDF (requires system tesseract).",
    )
    p.add_argument("--ocr-dpi", type=int, default=150, help="OCR render DPI (default 150).")
    p.add_argument(
        "--ocr-lang",
        type=str,
        default="vie+eng",
        help='Tesseract language pack(s), e.g. "vie+eng" (default).',
    )
    p.add_argument(
        "--vlm-keywords",
        action="store_true",
        help="When page has no usable text (typical for scans), run a vision LLM on the page image to produce keywords for ColSmol's text branch (default model: Qwen2-VL-2B).",
    )
    p.add_argument("--vlm-model", type=str, default=DEFAULT_VLM_MODEL, help="Hugging Face model id for VLM keywords.")
    p.add_argument("--vlm-device", type=str, default=None, help="Device for VLM (default: same as CUDA if available else cpu).")
    p.add_argument("--vlm-max-tokens", type=int, default=256, help="Max new tokens for VLM generation.")
    p.add_argument(
        "--vlm-always",
        action="store_true",
        help="Always replace page text with VLM keywords (slow; ignores existing OCR/text).",
    )
    args = p.parse_args(argv)

    if not args.pdf.is_file():
        print(f"Not a file: {args.pdf}", file=sys.stderr)
        return 1
    if not args.index.is_file():
        print(f"Index not found: {args.index}", file=sys.stderr)
        return 1

    idx = load_index(args.index)
    model, processor, device = load_model(model_id=idx.model_id, device=args.device)
    proto_labels = [m["label"] for m in idx.manifest]

    rows: list[dict] = []
    for page in iter_pdf_pages(
        args.pdf,
        dpi=args.dpi,
        ocr=args.ocr,
        ocr_dpi=args.ocr_dpi,
        ocr_language=args.ocr_lang,
    ):
        query_text = page.text
        query_text_kind = "page_text" if page.text.strip() else "empty"
        vlm_snippet = ""

        if args.vlm_keywords:
            need_vlm = args.vlm_always or not page.text.strip()
            if need_vlm:
                try:
                    kw = keywords_from_image_vlm(
                        page.image,
                        model_id=args.vlm_model,
                        device=args.vlm_device,
                        max_new_tokens=args.vlm_max_tokens,
                    )
                except Exception as e:
                    print(f"VLM keywords failed page {page.page_index}: {e}", file=sys.stderr)
                    kw = ""
                if kw.strip():
                    query_text = kw
                    query_text_kind = "vlm_keywords"
                    vlm_snippet = kw[:500]
                elif not query_text.strip():
                    query_text_kind = "empty"

        pred, fused, sim_img, sim_txt = classify_page(
            processor=processor,
            device=device,
            query_image=page.image,
            query_text=query_text,
            proto_embs=idx.image_embs,
            proto_labels=proto_labels,
            model=model,
            w_img=args.w_img,
            batch_size=args.batch_size,
        )
        row = {
            "page_index": page.page_index,
            "predicted_label": pred,
            "text_source": page.text_source,
            "query_text_kind": query_text_kind,
            "text_chars": len(page.text),
            "text_empty": not bool(page.text.strip()),
            "vlm_keywords_snippet": vlm_snippet,
        }
        for k, v in fused.items():
            row[f"fused_{k}"] = round(v, 6)
        for k, v in sim_img.items():
            row[f"img_{k}"] = round(v, 4)
        for k, v in sim_txt.items():
            row[f"txt_{k}"] = round(v, 4)
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        if not rows:
            args.output.write_text("", encoding="utf-8")
        else:
            fieldnames = list(rows[0].keys())
            with args.output.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
