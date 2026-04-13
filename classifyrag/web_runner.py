"""Batch classify / blank scoring logic shared by CLI and Streamlit."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from classifyrag.characteristic_text import apply_characteristic_text
from classifyrag.colsmol_scorer import (
    PrototypeIndex,
    classify_page,
    classify_page_with_position,
    classify_triple,
    embed_images,
    predict_position_colpali,
    score_diagnostics,
    softmax_scores,
)
from classifyrag.labels import position_key_for_manifest
from classifyrag.llm_keywords import DEFAULT_VLM_MODEL, keywords_blank_page_vlm, keywords_from_image_vlm
from classifyrag.pdf_pages import iter_pdf_pages

from classifyrag.blank_page import BlankPageIndex, blank_scores

logger = logging.getLogger(__name__)


def _has_usable_scores(scores: dict[str, float]) -> bool:
    return any(math.isfinite(v) for v in scores.values()) if scores else False


def _row_text_source(page_text_source: str, query_text_kind: str) -> str:
    """CSV ``text_source``: PDF layer is native/ocr/empty; VLM query string → ``vlm``."""
    if query_text_kind in (
        "vlm_keywords",
        "vlm_keywords_characteristic",
        "vlm_blank",
        "vlm_blank_characteristic",
    ):
        return "vlm"
    return page_text_source


def _pred_probs_conf(scores: dict[str, float]) -> tuple[str, dict[str, float], float]:
    """Argmax on raw scores + softmax probs for logging (no ``other`` threshold/margin)."""
    probs = softmax_scores(scores)
    diag = score_diagnostics(scores)
    label = str(diag["top1_label"])
    if not label and scores:
        finite = {k: v for k, v in scores.items() if math.isfinite(v)}
        if finite:
            label = max(finite, key=finite.get)
    conf = float(diag["top1_score"])
    return label, probs, conf


@dataclass
class ClassifyRunConfig:
    mode: str
    w_img: float = 0.7
    dpi: float = 144.0
    max_pages: Optional[int] = None
    batch_size: int = 5
    ocr: bool = False
    ocr_dpi: int = 150
    ocr_lang: str = "vie+eng"
    vlm_keywords: bool = False
    vlm_model: str = DEFAULT_VLM_MODEL
    vlm_device: Optional[str] = None
    vlm_max_tokens: int = 256
    vlm_keyword_count: int = 10
    vlm_always: bool = False
    characteristic_text: bool = False
    label_score_agg: Literal["max", "topk_mean"] = "topk_mean"
    label_score_topk: int = 3
    #: colpali = legacy MaxSim + branch min-max; intrinsic = image norm01 + pooled cosine text + fuse without branch min-max
    score_style: Literal["colpali", "intrinsic"] = "colpali"


def iter_classify_rows(
    pdf_path: Path,
    idx: PrototypeIndex,
    model: Any,
    processor: Any,
    device: str,
    cfg: ClassifyRunConfig,
    proto_labels: list[str],
) -> list[dict]:
    rows: list[dict] = []
    for page in iter_pdf_pages(
        pdf_path,
        dpi=cfg.dpi,
        ocr=cfg.ocr,
        ocr_dpi=cfg.ocr_dpi,
        ocr_language=cfg.ocr_lang,
    ):
        if cfg.max_pages is not None and page.page_index >= cfg.max_pages:
            break
        query_text = page.text
        query_text_kind = "page_text" if page.text.strip() else "empty"
        vlm_snippet = ""
        vlm_raw = ""

        need_vlm_for_fused = cfg.vlm_keywords and (cfg.vlm_always or not page.text.strip())
        need_vlm_for_compare = cfg.mode == "compare"
        need_vlm_for_text_mode = cfg.mode == "text"
        page_has_no_text = not page.text.strip()

        if need_vlm_for_fused or need_vlm_for_compare or need_vlm_for_text_mode:
            kw = ""
            if page_has_no_text:
                try:
                    kw = keywords_blank_page_vlm(
                        page.image,
                        model_id=cfg.vlm_model,
                        device=cfg.vlm_device,
                        max_new_tokens=min(64, cfg.vlm_max_tokens),
                    )
                except Exception as e:
                    logger.warning("VLM blank page failed page %s: %s", page.page_index, e)
                    kw = "blank"
                vlm_raw = kw.strip() or "blank"
                if need_vlm_for_fused and vlm_raw:
                    query_text = vlm_raw
                    query_text_kind = "vlm_blank"
                    vlm_snippet = vlm_raw[:500]
                elif need_vlm_for_fused:
                    query_text_kind = "empty"
                if need_vlm_for_compare or need_vlm_for_text_mode:
                    vlm_snippet = vlm_raw[:500]
            else:
                try:
                    kw = keywords_from_image_vlm(
                        page.image,
                        model_id=cfg.vlm_model,
                        device=cfg.vlm_device,
                        max_new_tokens=cfg.vlm_max_tokens,
                        max_keywords=cfg.vlm_keyword_count if cfg.vlm_keyword_count > 0 else None,
                    )
                except Exception as e:
                    logger.warning("VLM keywords failed page %s: %s", page.page_index, e)
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

        query_text_fused = apply_characteristic_text(query_text, cfg.characteristic_text)
        char_flag = cfg.characteristic_text
        if char_flag and query_text_fused.strip():
            if query_text_kind == "page_text":
                query_text_kind = "characteristic_text"
            elif query_text_kind == "vlm_keywords":
                query_text_kind = "vlm_keywords_characteristic"
            elif query_text_kind == "vlm_blank":
                query_text_kind = "vlm_blank_characteristic"

        query_text_vlm_only = apply_characteristic_text(vlm_raw, char_flag)

        text_source = _row_text_source(page.text_source, query_text_kind)

        if cfg.mode == "compare":
            triple = classify_triple(
                processor=processor,
                device=device,
                query_image=page.image,
                query_text_vlm=query_text_vlm_only,
                query_text_fused=query_text_fused,
                proto_embs=idx.image_embs,
                proto_labels=proto_labels,
                model=model,
                w_img=cfg.w_img,
                batch_size=cfg.batch_size,
                proto_text_embs=idx.text_embs,
                label_score_agg=cfg.label_score_agg,
                label_score_topk=cfg.label_score_topk,
                score_style=cfg.score_style,
            )
            fused = triple["fused_scores"]
            sim_img = triple["sim_img"]
            sim_txt_vlm = triple["sim_txt_vlm"]
            sim_txt_fused = triple["sim_txt_fused"]
            pred_v, probs_v, conf_v = _pred_probs_conf(sim_txt_vlm if sim_txt_vlm else sim_img)
            pred_i, probs_i, conf_i = _pred_probs_conf(sim_img)
            pred_f, probs_f, conf_f = _pred_probs_conf(fused)
            diag_f = score_diagnostics(fused)
            margin_f = float(diag_f["margin_norm"])
            row = {
                "page_index": page.page_index,
                "mode": "compare",
                "predicted_label": pred_i,
                "predicted_label_vlm_text": pred_v,
                "predicted_label_image": pred_i,
                "predicted_label_fused": pred_f,
                "text_source": text_source,
                "query_text_kind": query_text_kind,
                "text_chars": len(page.text),
                "text_empty": not bool(page.text.strip()),
                "vlm_keywords_snippet": vlm_snippet,
                "top1_score_vlm_text": round(conf_v, 6),
                "top1_score_image": round(conf_i, 6),
                "top1_score_fused": round(conf_f, 6),
                "margin_norm_fused": round(margin_f, 6),
                "score_style": cfg.score_style,
            }
            for k, v in fused.items():
                row[f"fused_{k}"] = round(v, 6)
            for k, v in sim_img.items():
                row[f"img_{k}"] = round(v, 4)
            for k, v in sim_txt_vlm.items():
                row[f"txt_vlm_{k}"] = round(v, 4)
            for k, v in sim_txt_fused.items():
                row[f"txt_fused_{k}"] = round(v, 4)
            for k, v in probs_v.items():
                row[f"prob_vlm_{k}"] = round(v, 6)
            for k, v in probs_i.items():
                row[f"prob_img_{k}"] = round(v, 6)
            for k, v in probs_f.items():
                row[f"prob_fused_{k}"] = round(v, 6)
            rows.append(row)
            continue

        query_for_single = query_text_vlm_only if cfg.mode == "text" else query_text_fused

        # --- intrinsic v2: image top-1, text MaxSim top-5, label_2 prediction ---
        if cfg.score_style == "intrinsic":
            proto_labels_2 = [position_key_for_manifest(m) for m in idx.manifest]
            res = classify_page_with_position(
                processor=processor,
                device=device,
                query_image=page.image,
                query_text=query_for_single,
                proto_embs=idx.image_embs,
                proto_labels=proto_labels,
                proto_labels_2=proto_labels_2,
                model=model,
                w_img=cfg.w_img,
                batch_size=cfg.batch_size,
                proto_text_embs=idx.text_embs,
                topk_tokens=5,
            )
            fused = res["fused"]
            sim_img = res["sim_img"]
            sim_txt = res["sim_txt"]
            pred_img = res["pred_img"]
            pred_fused = res["pred_fused"]
            pred_txt = res["pred_txt"]
            pred_label_2 = res["pred_label_2"]
            fused_pos = res["fused_pos"]

            chosen_scores = fused if _has_usable_scores(sim_txt) else sim_img
            pred_final, probs, confidence = _pred_probs_conf(chosen_scores)
            diag = score_diagnostics(chosen_scores)

            row = {
                "page_index": page.page_index,
                "mode": cfg.mode,
                "predicted_label": pred_final,
                "predicted_label_2": pred_label_2,
                "predicted_label_image": pred_img,
                "predicted_label_fused": pred_fused,
                "predicted_label_text": pred_txt,
                "predicted_label_base": res["pred_label"],
                "text_source": text_source,
                "query_text_kind": query_text_kind,
                "text_chars": len(page.text),
                "text_empty": not bool(page.text.strip()),
                "vlm_keywords_snippet": vlm_snippet,
                "top1_score": round(confidence, 6),
                "top1_label_by_score": str(diag["top1_label"]),
                "top2_score": round(float(diag["top2_score"]), 6),
                "margin_top1_top2": round(float(diag["margin_top1_top2"]), 6),
                "margin_norm": round(float(diag["margin_norm"]), 6),
                "z_gap_top1_top2": round(float(diag["z_gap_top1_top2"]), 6),
                "label_score_agg": cfg.label_score_agg,
                "label_score_topk": cfg.label_score_topk,
                "score_style": cfg.score_style,
            }
            for k, v in fused.items():
                row[f"fused_{k}"] = round(v, 6)
            for k, v in sim_img.items():
                row[f"img_{k}"] = round(v, 4)
            for k, v in sim_txt.items():
                row[f"txt_{k}"] = round(v, 4)
            for k, v in fused_pos.items():
                row[f"pos_{k}"] = round(v, 6)
            for k, v in probs.items():
                row[f"prob_{k}"] = round(v, 6)
            rows.append(row)
            continue

        # --- legacy colpali path + label_2 (position) same fusion as document label ---
        pred_from = "image" if cfg.mode == "image" else ("text" if cfg.mode == "text" else "fused")

        pred, fused, sim_img, sim_txt, pred_img, pred_fused, pred_txt, q_img, q_txt = classify_page(
            processor=processor,
            device=device,
            query_image=page.image,
            query_text=query_for_single,
            proto_embs=idx.image_embs,
            proto_labels=proto_labels,
            model=model,
            w_img=cfg.w_img,
            batch_size=cfg.batch_size,
            proto_text_embs=idx.text_embs,
            pred_from=pred_from,
            label_score_agg=cfg.label_score_agg,
            label_score_topk=cfg.label_score_topk,
            score_style=cfg.score_style,
        )

        proto_labels_2 = [position_key_for_manifest(m) for m in idx.manifest]
        pos_res = predict_position_colpali(
            processor=processor,
            device=device,
            query_image_emb=q_img,
            query_text_emb=q_txt,
            proto_embs=idx.image_embs,
            proto_labels_2=proto_labels_2,
            model=model,
            w_img=cfg.w_img,
            proto_text_embs=idx.text_embs,
            pred_from=pred_from,
            label_score_agg=cfg.label_score_agg,
            label_score_topk=cfg.label_score_topk,
        )
        pred_label_2 = str(pos_res["pred_label_2"])

        if cfg.mode == "text":
            chosen_scores = sim_txt if _has_usable_scores(sim_txt) else sim_img
        elif cfg.mode == "fused":
            chosen_scores = fused if _has_usable_scores(sim_txt) else sim_img
        else:
            chosen_scores = sim_img
        pred_final, probs, confidence = _pred_probs_conf(chosen_scores)
        diag = score_diagnostics(chosen_scores)

        row = {
            "page_index": page.page_index,
            "mode": cfg.mode,
            "predicted_label": pred_final,
            "predicted_label_2": pred_label_2,
            "predicted_label_image": pred_img,
            "predicted_label_fused": pred_fused,
            "predicted_label_text": pred_txt,
            "predicted_label_base": pred,
            "text_source": text_source,
            "query_text_kind": query_text_kind,
            "text_chars": len(page.text),
            "text_empty": not bool(page.text.strip()),
            "vlm_keywords_snippet": vlm_snippet,
            "top1_score": round(confidence, 6),
            "top1_label_by_score": str(diag["top1_label"]),
            "top2_score": round(float(diag["top2_score"]), 6),
            "margin_top1_top2": round(float(diag["margin_top1_top2"]), 6),
            "margin_norm": round(float(diag["margin_norm"]), 6),
            "z_gap_top1_top2": round(float(diag["z_gap_top1_top2"]), 6),
            "label_score_agg": cfg.label_score_agg,
            "label_score_topk": cfg.label_score_topk,
            "score_style": cfg.score_style,
        }
        for k, v in fused.items():
            row[f"fused_{k}"] = round(v, 6)
        for k, v in sim_img.items():
            row[f"img_{k}"] = round(v, 4)
        for k, v in sim_txt.items():
            row[f"txt_{k}"] = round(v, 4)
        for k, v in pos_res["fused_pos"].items():
            row[f"pos_{k}"] = round(v, 6)
        for k, v in probs.items():
            row[f"prob_{k}"] = round(v, 6)
        rows.append(row)
    return rows


@dataclass
class BlankRunConfig:
    max_pages: int = 500
    threshold: float = 0.85
    dpi: float = 144.0
    batch_size: int = 5
    gt_ocr: bool = False
    ocr_dpi: int = 150
    ocr_lang: str = "vie+eng"


def iter_blank_rows(
    pdf_path: Path,
    idx: BlankPageIndex,
    model: Any,
    processor: Any,
    device: str,
    cfg: BlankRunConfig,
) -> list[dict]:
    rows: list[dict] = []
    for page in iter_pdf_pages(
        pdf_path,
        dpi=cfg.dpi,
        ocr=cfg.gt_ocr,
        ocr_dpi=cfg.ocr_dpi,
        ocr_language=cfg.ocr_lang,
    ):
        if page.page_index >= cfg.max_pages:
            break
        q_emb = embed_images(model, processor, [page.image], device=device, batch_size=cfg.batch_size)[0]
        cos01, maxsim = blank_scores(processor, device, q_emb, idx.image_embs)
        gt_blank_no_text = not bool(page.text.strip())
        pred_blank = cos01 >= cfg.threshold
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
    return rows
