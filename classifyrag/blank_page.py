from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor

from classifyrag.colsmol_scorer import DEFAULT_MODEL_ID, embed_images, load_model, maxsim_vs_prototypes

BLANK_INDEX_VERSION = 1


@dataclass
class BlankPageIndex:
    """ColQwen3.5 image embeddings for blank-page reference samples."""

    model_id: str
    manifest: list[dict[str, Any]]
    image_embs: list[torch.Tensor]


def _image_paths(samples_dir: Path) -> list[Path]:
    out: list[Path] = []
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG"):
        out.extend(sorted(samples_dir.glob(f"*{ext}")))
    return sorted(set(out))


def load_blank_sample_images(samples_dir: Path) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    paths = _image_paths(samples_dir)
    manifest: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        manifest.append({"source_file": str(p.resolve()), "kind": "blank_reference"})
        images.append(img)
    return manifest, images


def mean_pooled_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of L2-normalized mean pools (multi-vector embeddings)."""
    av = a.float().mean(dim=0)
    bv = b.float().mean(dim=0)
    av = av / (av.norm() + 1e-8)
    bv = bv / (bv.norm() + 1e-8)
    return float((av * bv).sum().item())


def cosine01(cos: float) -> float:
    """Map cosine from [-1, 1] to [0, 1]."""
    return (cos + 1.0) / 2.0


def max_mean_cosine_vs_prototypes(query_emb: torch.Tensor, proto_embs: list[torch.Tensor]) -> float:
    if not proto_embs:
        return float("-inf")
    return max(mean_pooled_cosine(query_emb, p) for p in proto_embs)


def blank_scores(
    processor: Any,
    device: str,
    query_emb: torch.Tensor,
    blank_embs: list[torch.Tensor],
) -> tuple[float, float]:
    """
    Returns (cosine_01, raw_maxsim).

    ``cosine_01`` is in [0, 1] (mean-pooled cosine mapped); tune threshold (default 0.85 in ``is_blank_page``).
    ``raw_maxsim`` is the late-interaction MaxSim score (unbounded; for diagnostics).
    """
    cos = max_mean_cosine_vs_prototypes(query_emb, blank_embs)
    cos01 = cosine01(cos)
    maxsim = maxsim_vs_prototypes(processor, query_emb, blank_embs, device=device)
    return cos01, maxsim


def is_blank_page(
    processor: Any,
    device: str,
    query_emb: torch.Tensor,
    blank_embs: list[torch.Tensor],
    *,
    threshold: float = 0.85,
    use_maxsim: bool = False,
    maxsim_threshold: Optional[float] = None,
) -> tuple[bool, float, float]:
    """
    Decide if a page matches blank references.

    By default uses mean-pooled cosine mapped to [0,1] (``threshold`` default 0.85).
    If ``use_maxsim`` is True, uses raw MaxSim instead (set ``maxsim_threshold``; ``threshold`` ignored).
    """
    cos01, maxsim = blank_scores(processor, device, query_emb, blank_embs)
    if use_maxsim:
        if maxsim_threshold is None:
            raise ValueError("use_maxsim=True requires maxsim_threshold")
        return maxsim >= maxsim_threshold, cos01, maxsim
    return cos01 >= threshold, cos01, maxsim


def save_blank_index(path: str | Path, model_id: str, manifest: list[dict[str, Any]], image_embs: list[torch.Tensor]) -> None:
    payload = {
        "blank_index_version": BLANK_INDEX_VERSION,
        "model_id": model_id,
        "manifest": manifest,
        "image_embs": image_embs,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_blank_index(path: str | Path) -> BlankPageIndex:
    path = Path(path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return BlankPageIndex(
        model_id=blob["model_id"],
        manifest=blob["manifest"],
        image_embs=blob["image_embs"],
    )


def save_blank_manifest_json(path: str | Path, manifest: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def score_blank_for_image(
    image: Image.Image,
    *,
    model: ColQwen3_5,
    processor: ColQwen3_5Processor,
    device: str,
    blank_index: BlankPageIndex,
    batch_size: int = 5,
) -> tuple[float, float]:
    """
    Embed one page image and score against a blank reference index.

    Returns (cosine01, raw_maxsim); use ``is_blank_page`` if you need a boolean threshold.
    """
    img = image.convert("RGB")
    q_emb = embed_images(model, processor, [img], device=device, batch_size=batch_size)[0]
    return blank_scores(processor, device, q_emb, blank_index.image_embs)


def build_blank_index_from_dir(
    samples_dir: Path,
    output_pt: Path,
    manifest_json: Optional[Path] = None,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[str] = None,
    batch_size: int = 5,
) -> BlankPageIndex:
    manifest, images = load_blank_sample_images(samples_dir)
    if not images:
        raise FileNotFoundError(f"No images (.png/.jpg/…) found in {samples_dir}")

    model, processor, dev = load_model(model_id=model_id, device=device)
    embs = embed_images(model, processor, images, device=dev, batch_size=batch_size)
    idx = BlankPageIndex(model_id=model_id, manifest=manifest, image_embs=embs)
    save_blank_index(output_pt, model_id, manifest, embs)
    if manifest_json is not None:
        save_blank_manifest_json(manifest_json, manifest)
    return idx
