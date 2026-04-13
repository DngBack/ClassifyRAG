from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

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
from classifyrag.labels import BLANK_LABEL, label_from_filename
from classifyrag.llm_keywords import DEFAULT_VLM_MODEL, keywords_blank_page_vlm, keywords_from_image_vlm
from classifyrag.pdf_pages import iter_pdf_pages, page_count

# Blank prototypes: raster files in ``blank_dir`` (one prototype per file).
_BLANK_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})


def _list_blank_images(blank_dir: Path) -> list[Path]:
    if not blank_dir.is_dir():
        return []
    return sorted(p for p in blank_dir.iterdir() if p.is_file() and p.suffix.lower() in _BLANK_IMAGE_SUFFIXES)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build ColQwen3.5 prototype index from labeled sample PDFs.")
    p.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("data/Sample_document"),
        help="Directory containing sample PDFs (filename prefix = label).",
    )
    p.add_argument(
        "--blank-dir",
        type=Path,
        default=Path("data/blank_data"),
        help="Optional directory of blank prototypes: *.pdf (per page) and/or images "
        "(%s) — one prototype per image file. label=%r, label_2=null, text 'blank'. "
        "If missing or has no supported files, blank raster/PDF prototypes are skipped."
        % (", ".join(sorted(_BLANK_IMAGE_SUFFIXES)), BLANK_LABEL),
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
    p.add_argument("--batch-size", type=int, default=5)
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
        "--vlm-keyword-count",
        type=int,
        default=10,
        help="Max distinctive keywords from VLM (comma-separated, de-duped). Use 0 for legacy long list (10-25).",
    )
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
    blank_pdfs = sorted(args.blank_dir.glob("*.pdf")) if args.blank_dir.is_dir() else []
    blank_images = _list_blank_images(args.blank_dir)
    if not pdfs and not blank_pdfs and not blank_images:
        print(
            f"No PDFs in {args.samples_dir} and no blank PDFs/images in {args.blank_dir}",
            file=sys.stderr,
        )
        return 1

    manifest: list[dict] = []
    images: list = []
    texts_raw: list[str] = []

    for pdf_path in pdfs:
        lab = label_from_filename(pdf_path.name)
        if lab is None:
            print(f"Skip (unknown prefix): {pdf_path.name}", file=sys.stderr)
            continue
        n_pages = page_count(pdf_path)
        for page in iter_pdf_pages(
            pdf_path,
            dpi=args.dpi,
            ocr=args.ocr,
            ocr_dpi=args.ocr_dpi,
            ocr_language=args.ocr_lang,
        ):
            if n_pages == 1:
                label_2 = "start"
            elif page.page_index == 0:
                label_2 = "start"
            elif page.page_index == n_pages - 1:
                label_2 = "end"
            else:
                label_2 = "mid"

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
                            max_keywords=args.vlm_keyword_count if args.vlm_keyword_count > 0 else None,
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
                    "label_2": label_2,
                    "source_file": str(pdf_path.resolve()),
                    "page_index": page.page_index,
                    "prototype_text": t_embed,
                    "prototype_text_source": text_src,
                }
            )
            images.append(page.image)

    for pdf_path in blank_pdfs:
        for page in iter_pdf_pages(
            pdf_path,
            dpi=args.dpi,
            ocr=args.ocr,
            ocr_dpi=args.ocr_dpi,
            ocr_language=args.ocr_lang,
        ):
            t_plain = page.text.strip()
            text_src = "blank_fixed"
            if args.vlm_keywords and (args.vlm_always or not t_plain):
                try:
                    keywords_blank_page_vlm(
                        page.image,
                        model_id=args.vlm_model,
                        device=args.vlm_device,
                        max_new_tokens=min(64, args.vlm_max_tokens),
                    )
                except Exception as e:
                    print(f"VLM blank-page failed {pdf_path.name} page {page.page_index}: {e}", file=sys.stderr)
                text_src = "vlm_blank"
            t_embed = truncate_text("blank", DEFAULT_TEXT_CHARS)
            texts_raw.append(t_embed)
            manifest.append(
                {
                    "label": BLANK_LABEL,
                    "label_2": None,
                    "source_file": str(pdf_path.resolve()),
                    "page_index": page.page_index,
                    "prototype_text": t_embed,
                    "prototype_text_source": text_src,
                }
            )
            images.append(page.image)

    for img_path in blank_images:
        try:
            page_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skip blank image (cannot open): {img_path.name}: {e}", file=sys.stderr)
            continue
        t_plain = ""
        text_src = "blank_fixed"
        if args.vlm_keywords and (args.vlm_always or not t_plain):
            try:
                keywords_blank_page_vlm(
                    page_image,
                    model_id=args.vlm_model,
                    device=args.vlm_device,
                    max_new_tokens=min(64, args.vlm_max_tokens),
                )
            except Exception as e:
                print(f"VLM blank-page failed {img_path.name}: {e}", file=sys.stderr)
            text_src = "vlm_blank"
        t_embed = truncate_text("blank", DEFAULT_TEXT_CHARS)
        texts_raw.append(t_embed)
        manifest.append(
            {
                "label": BLANK_LABEL,
                "label_2": None,
                "source_file": str(img_path.resolve()),
                "page_index": 0,
                "prototype_text": t_embed,
                "prototype_text_source": text_src,
            }
        )
        images.append(page_image)

    if not images:
        print("No pages indexed.", file=sys.stderr)
        return 1

    print(
        f"Indexing {len(images)} prototype pages "
        f"({len(pdfs)} PDFs from {args.samples_dir}, "
        f"{len(blank_pdfs)} blank PDFs + {len(blank_images)} blank images from {args.blank_dir}) ..."
    )
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
