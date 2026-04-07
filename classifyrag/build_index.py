from __future__ import annotations

import argparse
import sys
from pathlib import Path

from classifyrag.colsmol_scorer import (
    DEFAULT_MODEL_ID,
    DEFAULT_TEXT_CHARS,
    embed_images,
    embed_query_texts,
    load_model,
    save_index,
    save_manifest_json,
    truncate_text,
)
from classifyrag.characteristic_text import apply_characteristic_text
from classifyrag.labels import label_from_filename
from classifyrag.llm_keywords import DEFAULT_VLM_MODEL, keywords_from_image_vlm
from classifyrag.pdf_pages import iter_pdf_pages


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build ColQwen3.5 prototype index from labeled sample PDFs.")
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
    p.add_argument("--text-batch-size", type=int, default=8, help="Batch size for embedding prototype texts.")
    p.add_argument("--device", type=str, default=None, help="cuda:0 | cpu | mps (default: auto)")
    p.add_argument(
        "--ocr",
        action="store_true",
        help="When the PDF has no text layer, run Tesseract OCR (requires system tesseract).",
    )
    p.add_argument("--ocr-dpi", type=int, default=150)
    p.add_argument("--ocr-lang", type=str, default="vie+eng")
    p.add_argument(
        "--vlm-keywords",
        action="store_true",
        help="Use a vision LLM on each page image to fill empty text / optionally replace text (see --vlm-always).",
    )
    p.add_argument("--vlm-model", type=str, default=DEFAULT_VLM_MODEL, help="Hugging Face model id for VLM keywords.")
    p.add_argument("--vlm-device", type=str, default=None, help="Device for VLM (default: CUDA if available else cpu).")
    p.add_argument("--vlm-max-tokens", type=int, default=256, help="Max new tokens for VLM generation.")
    p.add_argument(
        "--vlm-always",
        action="store_true",
        help="Always replace page text with VLM keywords when --vlm-keywords is set (slow).",
    )
    p.add_argument(
        "--characteristic-text",
        action="store_true",
        help="For the text embedding branch, keep only titles and field labels (strip values after ':'). "
        "Use the same flag when classifying. Recommended for form-like PDFs.",
    )
    args = p.parse_args(argv)

    pdfs = sorted(args.samples_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {args.samples_dir}", file=sys.stderr)
        return 1

    manifest: list[dict] = []
    images: list = []
    texts_raw: list[str] = []

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
            t = page.text.strip()
            text_src = page.text_source if t else "empty"
            if args.vlm_keywords:
                need_vlm = args.vlm_always or not t
                if need_vlm:
                    try:
                        kw = keywords_from_image_vlm(
                            page.image,
                            model_id=args.vlm_model,
                            device=args.vlm_device,
                            max_new_tokens=args.vlm_max_tokens,
                        )
                    except Exception as e:
                        print(f"VLM keywords failed {pdf_path.name} page {page.page_index}: {e}", file=sys.stderr)
                        kw = ""
                    if kw.strip():
                        t = kw.strip()
                        text_src = "vlm"

            t = apply_characteristic_text(t, args.characteristic_text)
            t_embed = truncate_text(t, DEFAULT_TEXT_CHARS)
            texts_raw.append(t_embed)
            manifest.append(
                {
                    "label": lab,
                    "source_file": str(pdf_path.resolve()),
                    "page_index": page.page_index,
                    "prototype_text": t_embed,
                    "prototype_text_source": text_src,
                }
            )
            images.append(page.image)

    if not images:
        print("No pages indexed.", file=sys.stderr)
        return 1

    print(f"Indexing {len(images)} prototype pages from {args.samples_dir} ...")
    model, processor, device = load_model(model_id=args.model_id, device=args.device)
    embs = embed_images(model, processor, images, device=device, batch_size=args.batch_size)
    text_embs = embed_query_texts(
        model,
        processor,
        texts_raw,
        device=device,
        batch_size=args.text_batch_size,
    )

    save_index(args.output, args.model_id, manifest, embs, text_embs=text_embs)
    save_manifest_json(args.manifest_json, manifest)
    print(f"Wrote {args.output} and {args.manifest_json} (with text prototypes: {sum(1 for e in text_embs if e is not None)}/{len(text_embs)} non-empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
