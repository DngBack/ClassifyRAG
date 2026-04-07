from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import torch
from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor
from colpali_engine.utils.torch_utils import get_torch_device, unbind_padded_multivector_embeddings
from PIL import Image

from classifyrag.labels import ORDERED_LABELS

DEFAULT_MODEL_ID = "athrael-soju/colqwen3.5-4.5B-v3"
DEFAULT_TEXT_CHARS = 1024


@dataclass
class PrototypeIndex:
    model_id: str
    manifest: list[dict[str, Any]]
    image_embs: list[torch.Tensor]
    #: Parallel to ``image_embs`` / manifest; used for the text branch vs text prototypes (optional).
    text_embs: Optional[list[Optional[torch.Tensor]]] = None


def truncate_text(text: str, max_chars: int = DEFAULT_TEXT_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[ColQwen3_5, ColQwen3_5Processor, str]:
    dev = device or get_torch_device("auto")
    model = ColQwen3_5.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model = model.to(dev)
    model.eval()
    processor = ColQwen3_5Processor.from_pretrained(model_id)
    return model, processor, dev


def _clear_rope_cache(model: torch.nn.Module) -> None:
    """Qwen3.5 caches ``rope_deltas`` across forwards; clear before text/query embeds."""
    if hasattr(model, "rope_deltas"):
        model.rope_deltas = None


def _batch_to_device(batch: Any, device: str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


def _unbind_embeddings(emb: torch.Tensor) -> list[torch.Tensor]:
    emb = emb.float().cpu()
    return unbind_padded_multivector_embeddings(emb, padding_value=0.0, padding_side="left")


@torch.inference_mode()
def embed_images(
    model: ColQwen3_5,
    processor: ColQwen3_5Processor,
    images: list[Image.Image],
    device: str,
    batch_size: int = 4,
) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for i in range(0, len(images), batch_size):
        chunk = images[i : i + batch_size]
        batch = _batch_to_device(processor.process_images(chunk), device)
        emb = model(**batch)
        out.extend(_unbind_embeddings(emb))
    return out


@torch.inference_mode()
def embed_query_texts(
    model: ColQwen3_5,
    processor: ColQwen3_5Processor,
    texts: list[str],
    device: str,
    batch_size: int = 8,
) -> list[Optional[torch.Tensor]]:
    indices = [i for i, t in enumerate(texts) if t and t.strip()]
    out: list[Optional[torch.Tensor]] = [None] * len(texts)
    if not indices:
        return out
    to_embed = [texts[i] for i in indices]
    for start in range(0, len(to_embed), batch_size):
        batch_texts = to_embed[start : start + batch_size]
        _clear_rope_cache(model)
        batch = _batch_to_device(processor.process_queries(texts=batch_texts), device)
        emb = model(**batch)
        vecs = _unbind_embeddings(emb)
        for k, v in enumerate(vecs):
            out[indices[start + k]] = v
    return out


def _minmax_norm(scores: dict[str, float]) -> dict[str, float]:
    """Min–max to [0,1] using **finite** values only; non-finite scores → 0.0 (no NaN)."""
    if not scores:
        return {}
    finite_items = [(k, v) for k, v in scores.items() if math.isfinite(v)]
    if not finite_items:
        n = len(scores)
        return {k: 1.0 / n for k in scores}
    vals = [v for _, v in finite_items]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-8:
        n = len(scores)
        return {k: 1.0 / n for k in scores}
    out: dict[str, float] = {}
    for k, v in scores.items():
        if math.isfinite(v):
            out[k] = (v - lo) / (hi - lo)
        else:
            out[k] = 0.0
    return out


def _text_scores_usable(scores_txt: Optional[dict[str, float]]) -> bool:
    """False if missing or all non-finite (e.g. no text prototypes → all -inf). Skip fusion text branch."""
    if not scores_txt:
        return False
    return any(math.isfinite(v) for v in scores_txt.values())


def maxsim_vs_prototypes(
    processor: ColQwen3_5Processor,
    query_emb: torch.Tensor,
    proto_embs: list[torch.Tensor],
    device: str,
) -> float:
    if not proto_embs:
        return float("-inf")
    scores = processor.score([query_emb], proto_embs, device=device)
    return float(scores[0].max().item())


def scores_per_label(
    processor: ColQwen3_5Processor,
    query_emb: torch.Tensor,
    proto_embs: list[Optional[torch.Tensor]],
    proto_labels: list[str],
    labels: Iterable[str],
    device: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for lab in labels:
        ps = [e for e, lb in zip(proto_embs, proto_labels, strict=True) if lb == lab and e is not None]
        out[lab] = maxsim_vs_prototypes(processor, query_emb, ps, device=device)
    return out


def fuse_image_text_scores(
    scores_img: dict[str, float],
    scores_txt: Optional[dict[str, float]],
    w_img: float,
) -> dict[str, float]:
    """Min–max normalize each branch, then fuse. If scores_txt is None or unusable, return normalized image scores."""
    ni = _minmax_norm(scores_img)
    if scores_txt is None or not _text_scores_usable(scores_txt):
        return ni
    nt = _minmax_norm(scores_txt)
    w_img = max(0.0, min(1.0, w_img))
    return {k: w_img * ni[k] + (1.0 - w_img) * nt[k] for k in ni}


def _pred_from_txt(sim_txt: dict[str, float], fallback: str) -> str:
    if _text_scores_usable(sim_txt):
        return max(sim_txt, key=sim_txt.get)
    return fallback


@torch.inference_mode()
def classify_triple(
    processor: ColQwen3_5Processor,
    device: str,
    query_image: Image.Image,
    query_text_vlm: str,
    query_text_fused: str,
    proto_embs: list[torch.Tensor],
    proto_labels: list[str],
    model: ColQwen3_5,
    w_img: float = 0.7,
    text_max_chars: int = DEFAULT_TEXT_CHARS,
    batch_size: int = 4,
    proto_text_embs: Optional[list[Optional[torch.Tensor]]] = None,
) -> dict[str, Any]:
    """
    One image forward + two text forwards: VLM-only query vs prototype text (``sim_txt_vlm``),
    and production fused query (``sim_txt_fused``) for min–max fusion with image scores.

    Use this for side-by-side **vlm_text | image | fused** comparison on the same page.
    """
    q_img = embed_images(model, processor, [query_image], device=device, batch_size=batch_size)[0]
    sim_img = scores_per_label(
        processor, q_img, proto_embs, proto_labels, ORDERED_LABELS, device=device
    )
    pred_img = max(sim_img, key=sim_img.get)

    proto_for_txt = proto_text_embs if proto_text_embs is not None else proto_embs

    tv = truncate_text(query_text_vlm, text_max_chars)
    qv_list = embed_query_texts(model, processor, [tv], device=device, batch_size=1)
    qv = qv_list[0]
    sim_txt_vlm: dict[str, float] = {}
    if qv is not None:
        sim_txt_vlm = scores_per_label(
            processor, qv, proto_for_txt, proto_labels, ORDERED_LABELS, device=device
        )

    tf = truncate_text(query_text_fused, text_max_chars)
    qf_list = embed_query_texts(model, processor, [tf], device=device, batch_size=1)
    qf = qf_list[0]
    sim_txt_fused: dict[str, float] = {}
    if qf is not None:
        sim_txt_fused = scores_per_label(
            processor, qf, proto_for_txt, proto_labels, ORDERED_LABELS, device=device
        )

    pred_vlm_text = _pred_from_txt(sim_txt_vlm, pred_img)

    txt_for_fuse = sim_txt_fused if (qf is not None and _text_scores_usable(sim_txt_fused)) else None
    fused = fuse_image_text_scores(
        sim_img, txt_for_fuse, w_img=w_img if txt_for_fuse is not None else 1.0
    )
    pred_fused = max(fused, key=fused.get)

    return {
        "pred_vlm_text": pred_vlm_text,
        "pred_image": pred_img,
        "pred_fused": pred_fused,
        "sim_img": sim_img,
        "sim_txt_vlm": sim_txt_vlm,
        "sim_txt_fused": sim_txt_fused,
        "fused_scores": fused,
    }


def classify_page(
    processor: ColQwen3_5Processor,
    device: str,
    query_image: Image.Image,
    query_text: str,
    proto_embs: list[torch.Tensor],
    proto_labels: list[str],
    model: ColQwen3_5,
    w_img: float = 0.7,
    text_max_chars: int = DEFAULT_TEXT_CHARS,
    batch_size: int = 4,
    proto_text_embs: Optional[list[Optional[torch.Tensor]]] = None,
    pred_from: Literal["image", "text", "fused"] = "image",
) -> tuple[str, dict[str, float], dict[str, float], dict[str, float], str, str, str]:
    """
    Returns predicted label, fused scores, raw image scores, raw text scores (or empty),
    pred_img, pred_fused, pred_txt.

    ``pred_from``: ``image`` = MaxSim ảnh; ``text`` = MaxSim nhánh text (prototype text);
    ``fused`` = sau khi gộp ảnh+text. Chế độ ``image`` không embed query text (nhanh hơn).
    """
    q_img = embed_images(model, processor, [query_image], device=device, batch_size=batch_size)[0]
    sim_img = scores_per_label(
        processor, q_img, proto_embs, proto_labels, ORDERED_LABELS, device=device
    )
    pred_img = max(sim_img, key=sim_img.get)

    if pred_from == "image":
        fused = fuse_image_text_scores(sim_img, None, 1.0)
        pred_fused = max(fused, key=fused.get)
        return pred_img, fused, sim_img, {}, pred_img, pred_fused, pred_img

    t = truncate_text(query_text, text_max_chars)
    q_txt_list = embed_query_texts(model, processor, [t], device=device, batch_size=1)
    q_txt = q_txt_list[0]

    sim_txt: dict[str, float] = {}
    if q_txt is not None:
        proto_for_txt = proto_text_embs if proto_text_embs is not None else proto_embs
        sim_txt = scores_per_label(
            processor, q_txt, proto_for_txt, proto_labels, ORDERED_LABELS, device=device
        )

    txt_for_fuse = sim_txt if (q_txt is not None and _text_scores_usable(sim_txt)) else None
    fused = fuse_image_text_scores(sim_img, txt_for_fuse, w_img=w_img if txt_for_fuse is not None else 1.0)

    pred_fused = max(fused, key=fused.get)
    pred_txt = _pred_from_txt(sim_txt, pred_img)

    if pred_from == "text":
        pred = pred_txt
    else:
        pred = pred_fused

    return pred, fused, sim_img, sim_txt, pred_img, pred_fused, pred_txt


def save_index(
    path: str | Path,
    model_id: str,
    manifest: list[dict[str, Any]],
    image_embs: list[torch.Tensor],
    text_embs: Optional[list[Optional[torch.Tensor]]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model_id": model_id, "manifest": manifest, "image_embs": image_embs}
    if text_embs is not None:
        payload["text_embs"] = text_embs
    torch.save(payload, path)


def load_index(path: str | Path) -> PrototypeIndex:
    path = Path(path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return PrototypeIndex(
        model_id=blob["model_id"],
        manifest=blob["manifest"],
        image_embs=blob["image_embs"],
        text_embs=blob.get("text_embs"),
    )


def save_manifest_json(path: str | Path, manifest: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
