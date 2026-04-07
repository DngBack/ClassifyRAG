from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from classifyrag.characteristic_text import apply_characteristic_text
from classifyrag.colsmol_scorer import classify_page, classify_triple, load_index, load_model
from classifyrag.llm_keywords import DEFAULT_VLM_MODEL, keywords_from_image_vlm
from classifyrag.pdf_pages import iter_pdf_pages


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
        help="When page has no usable text (typical for scans), run a vision LLM on the page image to produce keywords for the retriever text branch (see --vlm-model). "
        "For --mode fused, use with --vlm-always if index used VLM on every page.",
    )
    p.add_argument("--vlm-model", type=str, default=DEFAULT_VLM_MODEL, help="Hugging Face model id for VLM keywords.")
    p.add_argument("--vlm-device", type=str, default=None, help="Device for VLM (default: same as CUDA if available else cpu).")
    p.add_argument("--vlm-max-tokens", type=int, default=256, help="Max new tokens for VLM generation.")
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
        if args.max_pages is not None and page.page_index >= args.max_pages:
            break
        query_text = page.text
        query_text_kind = "page_text" if page.text.strip() else "empty"
        vlm_snippet = ""
        vlm_raw = ""

        need_vlm_for_fused = args.vlm_keywords and (args.vlm_always or not page.text.strip())
        need_vlm_for_compare = args.mode == "compare"
        need_vlm_for_text_mode = args.mode == "text"

        if need_vlm_for_fused or need_vlm_for_compare or need_vlm_for_text_mode:
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
            vlm_raw = kw.strip()
            if need_vlm_for_fused and vlm_raw:
                query_text = kw
                query_text_kind = "vlm_keywords"
                vlm_snippet = kw[:500]
            elif need_vlm_for_fused and not query_text.strip():
                query_text_kind = "empty"
            if need_vlm_for_compare or need_vlm_for_text_mode:
                vlm_snippet = (kw or "")[:500]

        query_text_fused = apply_characteristic_text(query_text, args.characteristic_text)
        if args.characteristic_text and query_text_fused.strip():
            if query_text_kind == "page_text":
                query_text_kind = "characteristic_text"
            elif query_text_kind == "vlm_keywords":
                query_text_kind = "vlm_keywords_characteristic"

        query_text_vlm_only = apply_characteristic_text(vlm_raw, args.characteristic_text)

        if args.mode == "compare":
            triple = classify_triple(
                processor=processor,
                device=device,
                query_image=page.image,
                query_text_vlm=query_text_vlm_only,
                query_text_fused=query_text_fused,
                proto_embs=idx.image_embs,
                proto_labels=proto_labels,
                model=model,
                w_img=args.w_img,
                batch_size=args.batch_size,
                proto_text_embs=idx.text_embs,
            )
            pred_v = triple["pred_vlm_text"]
            pred_i = triple["pred_image"]
            pred_f = triple["pred_fused"]
            fused = triple["fused_scores"]
            sim_img = triple["sim_img"]
            sim_txt_vlm = triple["sim_txt_vlm"]
            sim_txt_fused = triple["sim_txt_fused"]
            row = {
                "page_index": page.page_index,
                "mode": "compare",
                "predicted_label": pred_i,
                "predicted_label_vlm_text": pred_v,
                "predicted_label_image": pred_i,
                "predicted_label_fused": pred_f,
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
            for k, v in sim_txt_vlm.items():
                row[f"txt_vlm_{k}"] = round(v, 4)
            for k, v in sim_txt_fused.items():
                row[f"txt_fused_{k}"] = round(v, 4)
            rows.append(row)
            continue

        pred_from = "image" if args.mode == "image" else ("text" if args.mode == "text" else "fused")
        query_for_single = query_text_vlm_only if args.mode == "text" else query_text_fused

        pred, fused, sim_img, sim_txt, pred_img, pred_fused, pred_txt = classify_page(
            processor=processor,
            device=device,
            query_image=page.image,
            query_text=query_for_single,
            proto_embs=idx.image_embs,
            proto_labels=proto_labels,
            model=model,
            w_img=args.w_img,
            batch_size=args.batch_size,
            proto_text_embs=idx.text_embs,
            pred_from=pred_from,
        )
        row = {
            "page_index": page.page_index,
            "mode": args.mode,
            "predicted_label": pred,
            "predicted_label_image": pred_img,
            "predicted_label_fused": pred_fused,
            "predicted_label_text": pred_txt,
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

    summary_fields: tuple[str, ...] = ("page_index", "predicted_label")
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        if args.mode == "compare":
            summary_fields = (
                "page_index",
                "predicted_label_vlm_text",
                "predicted_label_image",
                "predicted_label_fused",
            )
        else:
            summary_fields = ("page_index", "predicted_label")
        with args.summary.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in summary_fields})

    print(f"Wrote {len(rows)} rows to {args.output}")
    if args.summary is not None:
        print(f"Wrote summary {summary_fields} to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
