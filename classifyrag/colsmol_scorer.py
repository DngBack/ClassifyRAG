from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from colpali_engine.utils.torch_utils import get_torch_device, unbind_padded_multivector_embeddings
from PIL import Image

from classifyrag.labels import ORDERED_LABELS

DEFAULT_MODEL_ID = "vidore/colSmol-256M"
DEFAULT_TEXT_CHARS = 1024


@dataclass
class PrototypeIndex:
    model_id: str
    manifest: list[dict[str, Any]]
    image_embs: list[torch.Tensor]


def truncate_text(text: str, max_chars: int = DEFAULT_TEXT_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[ColIdefics3, ColIdefics3Processor, str]:
    dev = device or get_torch_device("auto")
    model = ColIdefics3.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(dev)
    model.eval()
    processor = ColIdefics3Processor.from_pretrained(model_id)
    return model, processor, dev


def _batch_to_device(batch: Any, device: str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


def _unbind_embeddings(emb: torch.Tensor) -> list[torch.Tensor]:
    emb = emb.float().cpu()
    return unbind_padded_multivector_embeddings(emb, padding_value=0.0, padding_side="left")


@torch.inference_mode()
def embed_images(
    model: ColIdefics3,
    processor: ColIdefics3Processor,
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
    model: ColIdefics3,
    processor: ColIdefics3Processor,
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
        batch = _batch_to_device(processor.process_queries(texts=batch_texts), device)
        emb = model(**batch)
        vecs = _unbind_embeddings(emb)
        for k, v in enumerate(vecs):
            out[indices[start + k]] = v
    return out


def _minmax_norm(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-8:
        n = len(scores)
        return {k: 1.0 / n for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def maxsim_vs_prototypes(
    processor: ColIdefics3Processor,
    query_emb: torch.Tensor,
    proto_embs: list[torch.Tensor],
    device: str,
) -> float:
    if not proto_embs:
        return float("-inf")
    scores = processor.score([query_emb], proto_embs, device=device)
    return float(scores[0].max().item())


def scores_per_label(
    processor: ColIdefics3Processor,
    query_emb: torch.Tensor,
    proto_embs: list[torch.Tensor],
    proto_labels: list[str],
    labels: Iterable[str],
    device: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for lab in labels:
        ps = [e for e, lb in zip(proto_embs, proto_labels, strict=True) if lb == lab]
        out[lab] = maxsim_vs_prototypes(processor, query_emb, ps, device=device)
    return out


def fuse_image_text_scores(
    scores_img: dict[str, float],
    scores_txt: Optional[dict[str, float]],
    w_img: float,
) -> dict[str, float]:
    """Min–max normalize each branch, then fuse. If scores_txt is None, return normalized image scores."""
    ni = _minmax_norm(scores_img)
    if scores_txt is None:
        return ni
    nt = _minmax_norm(scores_txt)
    w_img = max(0.0, min(1.0, w_img))
    return {k: w_img * ni[k] + (1.0 - w_img) * nt[k] for k in ni}


def classify_page(
    processor: ColIdefics3Processor,
    device: str,
    query_image: Image.Image,
    query_text: str,
    proto_embs: list[torch.Tensor],
    proto_labels: list[str],
    model: ColIdefics3,
    w_img: float = 0.7,
    text_max_chars: int = DEFAULT_TEXT_CHARS,
    batch_size: int = 4,
) -> tuple[str, dict[str, float], dict[str, float], dict[str, float]]:
    """
    Returns predicted label, fused scores, raw image scores, raw text scores (or empty).
    """
    q_img = embed_images(model, processor, [query_image], device=device, batch_size=batch_size)[0]
    t = truncate_text(query_text, text_max_chars)
    q_txt_list = embed_query_texts(model, processor, [t], device=device, batch_size=1)
    q_txt = q_txt_list[0]

    sim_img = scores_per_label(
        processor, q_img, proto_embs, proto_labels, ORDERED_LABELS, device=device
    )
    sim_txt: Optional[dict[str, float]] = None
    if q_txt is not None:
        sim_txt = scores_per_label(
            processor, q_txt, proto_embs, proto_labels, ORDERED_LABELS, device=device
        )

    fused = fuse_image_text_scores(sim_img, sim_txt, w_img=w_img if q_txt is not None else 1.0)
    pred = max(fused, key=fused.get)
    return pred, fused, sim_img, sim_txt or {}


def save_index(path: str | Path, model_id: str, manifest: list[dict[str, Any]], image_embs: list[torch.Tensor]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_id": model_id, "manifest": manifest, "image_embs": image_embs}, path)


def load_index(path: str | Path) -> PrototypeIndex:
    path = Path(path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return PrototypeIndex(
        model_id=blob["model_id"],
        manifest=blob["manifest"],
        image_embs=blob["image_embs"],
    )


def save_manifest_json(path: str | Path, manifest: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
