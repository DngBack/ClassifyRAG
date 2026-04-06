from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from classifyrag.blank_page import blank_scores, load_blank_index
from classifyrag.colsmol_scorer import embed_images, load_model
from classifyrag.pdf_pages import iter_pdf_pages


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Score PDF pages against blank reference index; optional compare to no-text ground truth."
    )
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--index", type=Path, default=Path("data/index/blank_prototypes.pt"))
    p.add_argument("--output", type=Path, default=None, help="Write CSV or JSON (from --format). Default: print table only.")
    p.add_argument("--format", choices=("csv", "json"), default="csv")
    p.add_argument("--max-pages", type=int, default=50, help="Only first N pages (default 50).")
    p.add_argument("--threshold", type=float, default=0.95, help="Blank if cosine01 >= this (default 0.95).")
    p.add_argument("--dpi", type=float, default=144.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--gt-ocr",
        action="store_true",
        help="Ground truth: page is non-blank if Tesseract OCR finds any text (use for scanned PDFs; slower). "
        "Default GT is only the PDF text layer (fine for born-digital PDFs).",
    )
    p.add_argument("--ocr-dpi", type=int, default=150)
    p.add_argument("--ocr-lang", type=str, default="vie+eng")
    args = p.parse_args(argv)

    if not args.pdf.is_file():
        print(f"Not a file: {args.pdf}", file=sys.stderr)
        return 1
    if not args.index.is_file():
        print(f"Index not found: {args.index}", file=sys.stderr)
        return 1

    idx = load_blank_index(args.index)
    model, processor, device = load_model(model_id=idx.model_id, device=args.device)

    rows: list[dict] = []
    for page in iter_pdf_pages(
        args.pdf,
        dpi=args.dpi,
        ocr=args.gt_ocr,
        ocr_dpi=args.ocr_dpi,
        ocr_language=args.ocr_lang,
    ):
        if page.page_index >= args.max_pages:
            break
        q_emb = embed_images(model, processor, [page.image], device=device, batch_size=args.batch_size)[0]
        cos01, maxsim = blank_scores(processor, device, q_emb, idx.image_embs)
        # Blank if no text: with --gt-ocr, ``page.text`` may include OCR when the layer was empty.
        gt_blank_no_text = not bool(page.text.strip())
        pred_blank = cos01 >= args.threshold
        rows.append(
            {
                "page_index": page.page_index,
                "cosine01_vs_blank": round(cos01, 6),
                "raw_maxsim_vs_blank": round(maxsim, 4),
                "pred_blank": pred_blank,
                "gt_blank_no_text": gt_blank_no_text,
                "text_chars": len(page.text),
                "text_source": page.text_source,
            }
        )

    # Summary: pred_blank vs gt_blank_no_text (with --gt-ocr, gt reflects OCR; otherwise PDF text layer only)
    tp = sum(1 for r in rows if r["pred_blank"] and r["gt_blank_no_text"])
    fp = sum(1 for r in rows if r["pred_blank"] and not r["gt_blank_no_text"])
    fn = sum(1 for r in rows if not r["pred_blank"] and r["gt_blank_no_text"])
    tn = sum(1 for r in rows if not r["pred_blank"] and not r["gt_blank_no_text"])
    gt_desc = "OCR + embedded text empty" if args.gt_ocr else "embedded PDF text empty only (not OCR)"
    print(
        f"Pages: {len(rows)} | threshold cosine01>={args.threshold}\n"
        f"Ground truth blank: {gt_desc}. "
        f"Confusion vs pred_blank: TP={tp} FP={fp} FN={fn} TN={tn}"
    )

    # Compact table to stdout
    for r in rows:
        print(
            f"page {r['page_index']:3d}  cosine01={r['cosine01_vs_blank']:.4f}  "
            f"maxsim={r['raw_maxsim_vs_blank']:.2f}  pred_blank={r['pred_blank']!s:5}  "
            f"gt_blank={r['gt_blank_no_text']!s:5}  chars={r['text_chars']}"
        )

    if args.output is not None:
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
        print(f"Wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
