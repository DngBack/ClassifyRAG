from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence

import torch
from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor
from colpali_engine.utils.torch_utils import get_torch_device, unbind_padded_multivector_embeddings
from PIL import Image

from classifyrag.labels import ORDERED_LABELS, POSITION_LABELS

DEFAULT_MODEL_ID = "athrael-soju/colqwen3.5-4.5B-v3"
DEFAULT_TEXT_CHARS = 1024

# colpali = late-interaction MaxSim + branch min–max fusion (legacy).
# intrinsic = image: top-k prototype MaxSim mean / num_query_tokens → [0,1];
#            text: mean-pool multivector → unit vector, cosine → (cos+1)/2, top-k mean; fuse without branch min–max.
ScoreStyle = Literal["colpali", "intrinsic"]


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
    batch_size: int = 5,
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


def aggregate_label_score(
    processor: ColQwen3_5Processor,
    query_emb: torch.Tensor,
    proto_embs: list[torch.Tensor],
    device: str,
    agg: Literal["max", "topk_mean"] = "topk_mean",
    topk: int = 3,
) -> float:
    """
    Aggregate query-vs-prototypes score for one label.
    - max: classic MaxSim over prototypes (can favor labels with many prototypes)
    - topk_mean: mean of top-k prototype scores (more robust to label-size imbalance)
    """
    if not proto_embs:
        return float("-inf")
    scores = processor.score([query_emb], proto_embs, device=device)[0].float().cpu()
    if scores.numel() == 0:
        return float("-inf")
    if agg == "max":
        return float(scores.max().item())
    k = max(1, min(int(topk), int(scores.numel())))
    top_vals = torch.topk(scores, k=k).values
    return float(top_vals.mean().item())


def scores_per_label(
    processor: ColQwen3_5Processor,
    query_emb: torch.Tensor,
    proto_embs: list[Optional[torch.Tensor]],
    proto_labels: list[str],
    labels: Iterable[str],
    device: str,
    label_score_agg: Literal["max", "topk_mean"] = "topk_mean",
    label_score_topk: int = 3,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for lab in labels:
        ps = [e for e, lb in zip(proto_embs, proto_labels, strict=True) if lb == lab and e is not None]
        out[lab] = aggregate_label_score(
            processor,
            query_emb,
            ps,
            device=device,
            agg=label_score_agg,
            topk=label_score_topk,
        )
    return out


def _num_query_tokens(emb: torch.Tensor) -> int:
    if emb.dim() == 2:
        return int(emb.shape[0])
    return 1


def _pool_unit_vector(emb: torch.Tensor) -> torch.Tensor:
    """Mean-pool token/patch rows then L2-normalize to unit vector."""
    x = emb.float()
    if x.dim() == 2:
        v = x.mean(dim=0)
    else:
        v = x.flatten()
    n = float(v.norm(p=2).item())
    if n < 1e-8:
        return torch.zeros_like(v)
    return v / n


def _cosine_01_pooled(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of pooled unit vectors, mapped to [0, 1]."""
    cos = float(torch.dot(a.flatten(), b.flatten()).clamp(-1.0, 1.0).item())
    return max(0.0, min(1.0, (cos + 1.0) * 0.5))


def scores_per_label_image_intrinsic(
    processor: ColQwen3_5Processor,
    query_emb: torch.Tensor,
    proto_embs: list[Optional[torch.Tensor]],
    proto_labels: list[str],
    labels: Iterable[str],
    device: str,
    topk: int = 3,
) -> dict[str, float]:
    """
    Per label: mean of top-``topk`` **page-level** MaxSim scores (colpali ``processor.score``),
    divided by number of query image tokens, clamped to [0, 1].
    """
    nq = max(_num_query_tokens(query_emb), 1)
    out: dict[str, float] = {}
    tk = max(1, int(topk))
    for lab in labels:
        ps = [e for e, lb in zip(proto_embs, proto_labels, strict=True) if lb == lab and e is not None]
        if not ps:
            out[lab] = 0.0
            continue
        scores = processor.score([query_emb], ps, device=device)[0].float().cpu()
        if scores.numel() == 0:
            out[lab] = 0.0
            continue
        k = min(tk, int(scores.numel()))
        top_mean = float(torch.topk(scores, k=k).values.mean().item())
        raw = top_mean / float(nq)
        out[lab] = max(0.0, min(1.0, raw))
    return out


def scores_per_label_text_pooled_cosine(
    query_emb: Optional[torch.Tensor],
    proto_embs: Sequence[Optional[torch.Tensor]],
    proto_labels: list[str],
    labels: Iterable[str],
    topk: int = 3,
) -> dict[str, float]:
    """
    Mean-pool query + prototype text multivectors → unit vectors; cosine → [0,1]; per label: top-k mean over prototypes.
    """
    out: dict[str, float] = {}
    if query_emb is None:
        return {lab: 0.0 for lab in labels}
    tk = max(1, int(topk))
    qv = _pool_unit_vector(query_emb)
    for lab in labels:
        cos_vals: list[float] = []
        for e, lb in zip(proto_embs, proto_labels, strict=True):
            if lb != lab or e is None:
                continue
            cos_vals.append(_cosine_01_pooled(qv, _pool_unit_vector(e)))
        if not cos_vals:
            out[lab] = 0.0
        else:
            t = torch.tensor(cos_vals, dtype=torch.float32)
            k = min(tk, int(t.numel()))
            out[lab] = float(torch.topk(t, k=k).values.mean().item())
    return out


@torch.inference_mode()
def _maxsim_top_k_query_tokens(
    query_emb: torch.Tensor,
    proto_emb: torch.Tensor,
    k: int = 5,
) -> float:
    """MaxSim using only the top-k query tokens with highest per-token max similarity.

    Token vectors are L2-normalised so similarities are cosine in [-1, 1],
    then mapped to [0, 1] via (cos+1)/2.
    """
    q = torch.nn.functional.normalize(query_emb.float(), dim=-1)
    p = torch.nn.functional.normalize(proto_emb.float(), dim=-1)
    sim = q @ p.T  # (n_q, n_p)
    per_token_max = sim.max(dim=1).values  # (n_q,)
    top_k = min(k, per_token_max.shape[0])
    top_vals = per_token_max.topk(top_k).values
    cos_mean = float(top_vals.mean().item())
    return max(0.0, min(1.0, (cos_mean + 1.0) / 2.0))


def scores_per_label_text_maxsim_top_tokens(
    query_emb: Optional[torch.Tensor],
    proto_embs: Sequence[Optional[torch.Tensor]],
    proto_labels: list[str],
    labels: Iterable[str],
    topk_tokens: int = 5,
) -> dict[str, float]:
    """Per label: MaxSim with top-k query tokens, top-1 prototype score."""
    out: dict[str, float] = {}
    if query_emb is None:
        return {lab: 0.0 for lab in labels}
    for lab in labels:
        ps = [e for e, lb in zip(proto_embs, proto_labels, strict=True) if lb == lab and e is not None]
        if not ps:
            out[lab] = 0.0
            continue
        scores = [_maxsim_top_k_query_tokens(query_emb, p, k=topk_tokens) for p in ps]
        out[lab] = max(scores)  # top-1 prototype
    return out


def fuse_image_text_intrinsic(
    scores_img: dict[str, float],
    scores_txt: Optional[dict[str, float]],
    w_img: float,
) -> dict[str, float]:
    """Fuse branch scores in ~[0,1] without per-branch min–max."""
    w_img = max(0.0, min(1.0, w_img))
    keys = list(scores_img.keys())
    if not _text_scores_usable(scores_txt):
        return {k: max(0.0, min(1.0, float(scores_img[k]))) for k in keys}
    assert scores_txt is not None
    out: dict[str, float] = {}
    for k in keys:
        vi = float(scores_img[k])
        vi = max(0.0, min(1.0, vi)) if math.isfinite(vi) else 0.0
        vt_raw = scores_txt.get(k, 0.0)
        vt = float(vt_raw)
        vt = max(0.0, min(1.0, vt)) if math.isfinite(vt) else 0.0
        out[k] = max(0.0, min(1.0, w_img * vi + (1.0 - w_img) * vt))
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


def softmax_scores(scores: dict[str, float]) -> dict[str, float]:
    """Numerically stable softmax over finite scores; non-finite values get 0.0 probability."""
    if not scores:
        return {}
    finite = {k: v for k, v in scores.items() if math.isfinite(v)}
    if not finite:
        n = len(scores)
        return {k: 1.0 / n for k in scores}
    m = max(finite.values())
    exps = {k: math.exp(v - m) for k, v in finite.items()}
    z = sum(exps.values())
    out: dict[str, float] = {}
    for k in scores:
        if k in exps and z > 0:
            out[k] = exps[k] / z
        else:
            out[k] = 0.0
    return out


def predict_with_other(
    scores: dict[str, float],
    *,
    other_threshold: Optional[float] = None,
    other_label: str = "other",
) -> tuple[str, dict[str, float], float]:
    """
    Predict from scores using raw top-1 score threshold.
    If ``other_threshold`` is set and top-1 score is below it, return ``other_label``.
    Softmax probabilities are returned for logging/debug only.
    """
    if not scores:
        return other_label, {}, 0.0

    finite_scores = {k: v for k, v in scores.items() if math.isfinite(v)}
    if not finite_scores:
        probs = softmax_scores(scores)
        return other_label, probs, 0.0

    best_label = max(finite_scores, key=finite_scores.get)
    best_score = finite_scores[best_label]
    probs = softmax_scores(scores)
    if other_threshold is not None and best_score < other_threshold:
        return other_label, probs, best_score
    return best_label, probs, best_score


def score_diagnostics(scores: dict[str, float]) -> dict[str, float | str]:
    """Per-page diagnostics to make thresholding comparable across score scales."""
    finite = [(k, v) for k, v in scores.items() if math.isfinite(v)]
    if not finite:
        return {
            "top1_label": "",
            "top1_score": 0.0,
            "top2_score": 0.0,
            "margin_top1_top2": 0.0,
            "margin_norm": 0.0,
            "z_gap_top1_top2": 0.0,
        }
    finite.sort(key=lambda x: x[1], reverse=True)
    top1_label, top1 = finite[0]
    top2 = finite[1][1] if len(finite) > 1 else top1
    vals = [v for _, v in finite]
    lo, hi = min(vals), max(vals)
    rng = max(1e-8, hi - lo)
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = max(1e-8, math.sqrt(var))
    margin = top1 - top2
    return {
        "top1_label": top1_label,
        "top1_score": top1,
        "top2_score": top2,
        "margin_top1_top2": margin,
        "margin_norm": margin / rng,
        "z_gap_top1_top2": margin / std,
    }


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
    batch_size: int = 5,
    proto_text_embs: Optional[list[Optional[torch.Tensor]]] = None,
    label_score_agg: Literal["max", "topk_mean"] = "topk_mean",
    label_score_topk: int = 3,
    score_style: ScoreStyle = "colpali",
) -> dict[str, Any]:
    """
    One image forward + two text forwards: VLM-only query vs prototype text (``sim_txt_vlm``),
    and production fused query (``sim_txt_fused``) for min–max fusion with image scores.

    Use this for side-by-side **vlm_text | image | fused** comparison on the same page.
    """
    q_img = embed_images(model, processor, [query_image], device=device, batch_size=batch_size)[0]
    proto_for_txt = proto_text_embs if proto_text_embs is not None else proto_embs

    if score_style == "intrinsic":
        sim_img = scores_per_label_image_intrinsic(
            processor,
            q_img,
            proto_embs,
            proto_labels,
            ORDERED_LABELS,
            device=device,
            topk=label_score_topk,
        )
        pred_img = max(sim_img, key=sim_img.get)

        tv = truncate_text(query_text_vlm, text_max_chars)
        qv_list = embed_query_texts(model, processor, [tv], device=device, batch_size=1)
        qv = qv_list[0]
        sim_txt_vlm: dict[str, float] = {}
        if qv is not None:
            sim_txt_vlm = scores_per_label_text_pooled_cosine(
                qv, proto_for_txt, proto_labels, ORDERED_LABELS, topk=label_score_topk
            )

        tf = truncate_text(query_text_fused, text_max_chars)
        qf_list = embed_query_texts(model, processor, [tf], device=device, batch_size=1)
        qf = qf_list[0]
        sim_txt_fused: dict[str, float] = {}
        if qf is not None:
            sim_txt_fused = scores_per_label_text_pooled_cosine(
                qf, proto_for_txt, proto_labels, ORDERED_LABELS, topk=label_score_topk
            )

        pred_vlm_text = _pred_from_txt(sim_txt_vlm, pred_img)
        txt_for_fuse = (
            sim_txt_fused if (qf is not None and _text_scores_usable(sim_txt_fused)) else None
        )
        fused = fuse_image_text_intrinsic(
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

    sim_img = scores_per_label(
        processor,
        q_img,
        proto_embs,
        proto_labels,
        ORDERED_LABELS,
        device=device,
        label_score_agg=label_score_agg,
        label_score_topk=label_score_topk,
    )
    pred_img = max(sim_img, key=sim_img.get)

    tv = truncate_text(query_text_vlm, text_max_chars)
    qv_list = embed_query_texts(model, processor, [tv], device=device, batch_size=1)
    qv = qv_list[0]
    sim_txt_vlm = {}
    if qv is not None:
        sim_txt_vlm = scores_per_label(
            processor,
            qv,
            proto_for_txt,
            proto_labels,
            ORDERED_LABELS,
            device=device,
            label_score_agg=label_score_agg,
            label_score_topk=label_score_topk,
        )

    tf = truncate_text(query_text_fused, text_max_chars)
    qf_list = embed_query_texts(model, processor, [tf], device=device, batch_size=1)
    qf = qf_list[0]
    sim_txt_fused = {}
    if qf is not None:
        sim_txt_fused = scores_per_label(
            processor,
            qf,
            proto_for_txt,
            proto_labels,
            ORDERED_LABELS,
            device=device,
            label_score_agg=label_score_agg,
            label_score_topk=label_score_topk,
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
    batch_size: int = 5,
    proto_text_embs: Optional[list[Optional[torch.Tensor]]] = None,
    pred_from: Literal["image", "text", "fused"] = "image",
    label_score_agg: Literal["max", "topk_mean"] = "topk_mean",
    label_score_topk: int = 3,
    score_style: ScoreStyle = "colpali",
) -> tuple[
    str,
    dict[str, float],
    dict[str, float],
    dict[str, float],
    str,
    str,
    str,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    """
    Returns predicted label, fused scores, raw image scores, raw text scores (or empty),
    pred_img, pred_fused, pred_txt, query image embedding, optional query text embedding.

    ``pred_from``: ``image`` = MaxSim ảnh; ``text`` = MaxSim nhánh text (prototype text);
    ``fused`` = sau khi gộp ảnh+text. Chế độ ``image`` không embed query text (nhanh hơn).

    The final two tensors are for re-use (e.g. label_2 / position scoring) without a second image encode.
    """
    q_img = embed_images(model, processor, [query_image], device=device, batch_size=batch_size)[0]

    if score_style == "intrinsic":
        sim_img = scores_per_label_image_intrinsic(
            processor,
            q_img,
            proto_embs,
            proto_labels,
            ORDERED_LABELS,
            device=device,
            topk=label_score_topk,
        )
        pred_img = max(sim_img, key=sim_img.get)
        if pred_from == "image":
            fused = dict(sim_img)
            pred_fused = max(fused, key=fused.get)
            return pred_img, fused, sim_img, {}, pred_img, pred_fused, pred_img, q_img, None

        t = truncate_text(query_text, text_max_chars)
        q_txt_list = embed_query_texts(model, processor, [t], device=device, batch_size=1)
        q_txt = q_txt_list[0]

        sim_txt: dict[str, float] = {}
        if q_txt is not None:
            proto_for_txt = proto_text_embs if proto_text_embs is not None else proto_embs
            sim_txt = scores_per_label_text_pooled_cosine(
                q_txt, proto_for_txt, proto_labels, ORDERED_LABELS, topk=label_score_topk
            )

        txt_for_fuse = sim_txt if (q_txt is not None and _text_scores_usable(sim_txt)) else None
        fused = fuse_image_text_intrinsic(
            sim_img, txt_for_fuse, w_img=w_img if txt_for_fuse is not None else 1.0
        )
        pred_fused = max(fused, key=fused.get)
        pred_txt = _pred_from_txt(sim_txt, pred_img)
        if pred_from == "text":
            pred = pred_txt
        else:
            pred = pred_fused
        return pred, fused, sim_img, sim_txt, pred_img, pred_fused, pred_txt, q_img, q_txt

    sim_img = scores_per_label(
        processor,
        q_img,
        proto_embs,
        proto_labels,
        ORDERED_LABELS,
        device=device,
        label_score_agg=label_score_agg,
        label_score_topk=label_score_topk,
    )
    pred_img = max(sim_img, key=sim_img.get)

    if pred_from == "image":
        fused = fuse_image_text_scores(sim_img, None, 1.0)
        pred_fused = max(fused, key=fused.get)
        return pred_img, fused, sim_img, {}, pred_img, pred_fused, pred_img, q_img, None

    t = truncate_text(query_text, text_max_chars)
    q_txt_list = embed_query_texts(model, processor, [t], device=device, batch_size=1)
    q_txt = q_txt_list[0]

    sim_txt = {}
    if q_txt is not None:
        proto_for_txt = proto_text_embs if proto_text_embs is not None else proto_embs
        sim_txt = scores_per_label(
            processor,
            q_txt,
            proto_for_txt,
            proto_labels,
            ORDERED_LABELS,
            device=device,
            label_score_agg=label_score_agg,
            label_score_topk=label_score_topk,
        )

    txt_for_fuse = sim_txt if (q_txt is not None and _text_scores_usable(sim_txt)) else None
    fused = fuse_image_text_scores(sim_img, txt_for_fuse, w_img=w_img if txt_for_fuse is not None else 1.0)

    pred_fused = max(fused, key=fused.get)
    pred_txt = _pred_from_txt(sim_txt, pred_img)

    if pred_from == "text":
        pred = pred_txt
    else:
        pred = pred_fused

    return pred, fused, sim_img, sim_txt, pred_img, pred_fused, pred_txt, q_img, q_txt


@torch.inference_mode()
def predict_position_colpali(
    processor: ColQwen3_5Processor,
    device: str,
    query_image_emb: torch.Tensor,
    query_text_emb: Optional[torch.Tensor],
    proto_embs: Sequence[Optional[torch.Tensor]],
    proto_labels_2: list[str],
    model: ColQwen3_5,
    w_img: float = 0.7,
    proto_text_embs: Optional[Sequence[Optional[torch.Tensor]]] = None,
    pred_from: Literal["image", "text", "fused"] = "fused",
    label_score_agg: Literal["max", "topk_mean"] = "topk_mean",
    label_score_topk: int = 3,
) -> dict[str, Any]:
    """Position (label_2: start/mid/end/none) using the same colpali + branch min–max fusion as ``classify_page``."""
    sim_img_pos = scores_per_label(
        processor,
        query_image_emb,
        list(proto_embs),
        proto_labels_2,
        POSITION_LABELS,
        device=device,
        label_score_agg=label_score_agg,
        label_score_topk=label_score_topk,
    )

    if pred_from == "image" or query_text_emb is None:
        fused_pos = fuse_image_text_scores(sim_img_pos, None, 1.0)
        pred_label_2 = max(fused_pos, key=fused_pos.get)
        return {
            "pred_label_2": pred_label_2,
            "fused_pos": fused_pos,
            "sim_img_pos": sim_img_pos,
            "sim_txt_pos": {},
        }

    proto_for_txt = list(proto_text_embs) if proto_text_embs is not None else list(proto_embs)
    sim_txt_pos = scores_per_label(
        processor,
        query_text_emb,
        proto_for_txt,
        proto_labels_2,
        POSITION_LABELS,
        device=device,
        label_score_agg=label_score_agg,
        label_score_topk=label_score_topk,
    )
    txt_for_fuse = sim_txt_pos if _text_scores_usable(sim_txt_pos) else None
    fused_pos = fuse_image_text_scores(
        sim_img_pos,
        txt_for_fuse,
        w_img=w_img if txt_for_fuse is not None else 1.0,
    )
    pred_label_2 = max(fused_pos, key=fused_pos.get)
    return {
        "pred_label_2": pred_label_2,
        "fused_pos": fused_pos,
        "sim_img_pos": sim_img_pos,
        "sim_txt_pos": sim_txt_pos,
    }


@torch.inference_mode()
def classify_page_with_position(
    processor: ColQwen3_5Processor,
    device: str,
    query_image: Image.Image,
    query_text: str,
    proto_embs: list[torch.Tensor],
    proto_labels: list[str],
    proto_labels_2: list[str],
    model: ColQwen3_5,
    w_img: float = 0.7,
    text_max_chars: int = DEFAULT_TEXT_CHARS,
    batch_size: int = 5,
    proto_text_embs: Optional[list[Optional[torch.Tensor]]] = None,
    topk_tokens: int = 5,
) -> dict[str, Any]:
    """Classify page for both document type (label) and position (label_2).

    Image branch: intrinsic top-1 prototype per group.
    Text branch: MaxSim top-k tokens, top-1 prototype per group.
    Fusion: weighted sum without per-branch min-max.
    """
    q_img = embed_images(model, processor, [query_image], device=device, batch_size=batch_size)[0]

    sim_img = scores_per_label_image_intrinsic(
        processor, q_img, proto_embs, proto_labels, ORDERED_LABELS, device=device, topk=1,
    )
    sim_img_pos = scores_per_label_image_intrinsic(
        processor, q_img, proto_embs, proto_labels_2, POSITION_LABELS, device=device, topk=1,
    )
    pred_img = max(sim_img, key=sim_img.get)

    t = truncate_text(query_text, text_max_chars)
    q_txt_list = embed_query_texts(model, processor, [t], device=device, batch_size=1)
    q_txt = q_txt_list[0]

    proto_for_txt = proto_text_embs if proto_text_embs is not None else proto_embs

    sim_txt: dict[str, float] = {}
    sim_txt_pos: dict[str, float] = {}
    if q_txt is not None:
        sim_txt = scores_per_label_text_maxsim_top_tokens(
            q_txt, proto_for_txt, proto_labels, ORDERED_LABELS, topk_tokens=topk_tokens,
        )
        sim_txt_pos = scores_per_label_text_maxsim_top_tokens(
            q_txt, proto_for_txt, proto_labels_2, POSITION_LABELS, topk_tokens=topk_tokens,
        )

    txt_ok = _text_scores_usable(sim_txt)
    fused = fuse_image_text_intrinsic(sim_img, sim_txt if txt_ok else None, w_img)
    fused_pos = fuse_image_text_intrinsic(
        sim_img_pos, sim_txt_pos if _text_scores_usable(sim_txt_pos) else None, w_img,
    )

    pred_label = max(fused, key=fused.get)
    pred_label_2 = max(fused_pos, key=fused_pos.get)
    pred_txt = _pred_from_txt(sim_txt, pred_img)

    return {
        "pred_label": pred_label,
        "pred_label_2": pred_label_2,
        "pred_img": pred_img,
        "pred_fused": pred_label,
        "pred_txt": pred_txt,
        "fused": fused,
        "sim_img": sim_img,
        "sim_txt": sim_txt,
        "fused_pos": fused_pos,
        "sim_img_pos": sim_img_pos,
        "sim_txt_pos": sim_txt_pos,
    }


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
