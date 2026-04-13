from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from classifyrag.colsmol_scorer import load_index, load_model
from classifyrag.llm_keywords import DEFAULT_VLM_MODEL
from classifyrag.postprocess_split import group_by_position
from classifyrag.web_runner import ClassifyRunConfig, iter_classify_rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Classify each page of a PDF using a ColQwen3.5 prototype index.")
    p.add_argument("--pdf", type=Path, required=True, help="Input PDF path.")
    p.add_argument("--index", type=Path, default=Path("data/index/prototypes.pt"), help="Index from build_index.")
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Detailed output (.csv or .json): scores per label, text metadata, for debugging.",
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional second file: CSV with only page_index and predicted_label for quick review (no ground truth).",
    )
    p.add_argument("--format", choices=("csv", "json"), default="csv")
    p.add_argument("--w-img", type=float, default=0.7, help="Weight for image branch (text gets 1-w).")
    p.add_argument(
        "--mode",
        choices=("image", "text", "fused", "compare"),
        default="image",
        help="image=MaxSim ảnh only; text=VLM keywords → nhánh text vs prototype text; fused=ảnh+text; "
        "compare=ba cột pred cùng lúc (VLM text | image | fused).",
    )
    p.add_argument("--dpi", type=float, default=144.0)
    p.add_argument(
        "--max-pages",
        type=int,
        default=None,
        metavar="N",
        help="Only classify the first N pages (0-based: pages 0..N-1). Default: all pages.",
    )
    p.add_argument("--batch-size", type=int, default=5)
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
        help="When page has no usable text, run VLM: outputs the fixed token ``blank`` (same as blank index), matching blank prototypes. "
        "When the page has PDF/OCR text, VLM uses the structural keyword prompt. "
        "Use --vlm-always to run VLM on every page (non-empty pages still get the structural prompt).",
    )
    p.add_argument("--vlm-model", type=str, default=DEFAULT_VLM_MODEL, help="Hugging Face model id for VLM keywords.")
    p.add_argument("--vlm-device", type=str, default=None, help="Device for VLM (default: same as CUDA if available else cpu).")
    p.add_argument("--vlm-max-tokens", type=int, default=256, help="Max new tokens for VLM generation.")
    p.add_argument(
        "--vlm-keyword-count",
        type=int,
        default=10,
        help="Max distinctive VLM keywords when --vlm-keywords. 0 = legacy long list.",
    )
    p.add_argument(
        "--vlm-always",
        action="store_true",
        help="Always replace page text with VLM keywords (slow; ignores existing OCR/text).",
    )
    p.add_argument(
        "--characteristic-text",
        action="store_true",
        help="Use only titles and field labels for the text branch (strip values after ':'). "
        "Match the index built with the same flag.",
    )
    p.add_argument(
        "--label-score-agg",
        choices=("max", "topk_mean"),
        default="topk_mean",
        help="How to aggregate prototype scores per label. "
        "topk_mean is more robust when labels have different prototype counts.",
    )
    p.add_argument(
        "--label-score-topk",
        type=int,
        default=3,
        help="k for --label-score-agg topk_mean (default 3). Also: intrinsic image/text use top-k prototype mean per label.",
    )
    p.add_argument(
        "--score-style",
        choices=("colpali", "intrinsic"),
        default="colpali",
        help="colpali=late-interaction on text + branch min-max fusion (default). "
        "intrinsic=image: top-k MaxSim mean / num_query_patches → [0,1]; "
        "text: mean-pool embeddings, cosine→[0,1], top-k mean; fuse without branch min-max.",
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

    cfg = ClassifyRunConfig(
        mode=args.mode,
        w_img=args.w_img,
        dpi=args.dpi,
        max_pages=args.max_pages,
        batch_size=args.batch_size,
        ocr=args.ocr,
        ocr_dpi=args.ocr_dpi,
        ocr_lang=args.ocr_lang,
        vlm_keywords=args.vlm_keywords,
        vlm_model=args.vlm_model,
        vlm_device=args.vlm_device,
        vlm_max_tokens=args.vlm_max_tokens,
        vlm_keyword_count=args.vlm_keyword_count,
        vlm_always=args.vlm_always,
        characteristic_text=args.characteristic_text,
        label_score_agg=args.label_score_agg,
        label_score_topk=args.label_score_topk,
        score_style=args.score_style,
    )
    rows = iter_classify_rows(args.pdf, idx, model, processor, device, cfg, proto_labels)

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

    summary_fields: tuple[str, ...] = ("page_index", "predicted_label")
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        if args.mode == "compare":
            summary_fields = (
                "page_index",
                "predicted_label_vlm_text",
                "predicted_label_image",
                "predicted_label_fused",
                "top1_score_vlm_text",
                "top1_score_image",
                "top1_score_fused",
                "margin_norm_fused",
            )
        else:
            summary_fields = (
                "page_index",
                "predicted_label",
                "predicted_label_2",
                "predicted_label_base",
                "top1_score",
                "top2_score",
                "margin_top1_top2",
                "margin_norm",
                "z_gap_top1_top2",
            )
        with args.summary.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in summary_fields})

    # Document-level grouping based on label_2 (start/mid/end)
    has_label_2 = any(r.get("predicted_label_2") for r in rows)
    if has_label_2:
        docs = group_by_position(rows)
        docs_path = args.output.with_name(args.output.stem + "_docs.json")
        docs_path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {len(docs)} document groups to {docs_path}")

    print(f"Wrote {len(rows)} rows to {args.output}")
    if args.summary is not None:
        print(f"Wrote summary {summary_fields} to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
