from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SPLIT_RULES: dict[str, list[int]] = {
    # 1 page per slip
    "giay_gui_tien_tiet_kiem": [1],
    # usually 1 page, some templates are 2 pages
    "the_tiet_kiem_ban_sao": [1, 2],
    # keep flexible; default split each page as one instance
    "to_trinh_uu_dai_lai_suat": [1],
    # often 1 page; keep 2-page option configurable
    "giay_to_chung_minh_cu_tru": [1, 2],
}


@dataclass
class PagePred:
    page_index: int
    label: str


def load_split_rules(path: str | Path | None) -> dict[str, list[int]]:
    rules = {k: sorted(set(v)) for k, v in DEFAULT_SPLIT_RULES.items()}
    if path is None:
        return rules
    p = Path(path)
    if not p.is_file():
        return rules
    blob = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(blob, dict):
        return rules
    for k, v in blob.items():
        if isinstance(k, str) and isinstance(v, list):
            vals = [int(x) for x in v if isinstance(x, int) and x > 0]
            if vals:
                rules[k] = sorted(set(vals))
    return rules


def _partition_run(n: int, allowed_lengths: list[int]) -> list[int]:
    """Partition a run length into allowed chunk sizes via DP.

    Preference: exact coverage; tie-break by fewer chunks.
    Fallback to 1-page chunks if no exact partition is possible.
    """
    allowed = sorted(set(x for x in allowed_lengths if x > 0))
    if not allowed:
        return [1] * n
    if n == 0:
        return []

    # dp[i] = best partition for i pages, or None
    dp: list[list[int] | None] = [None] * (n + 1)
    dp[0] = []
    for i in range(1, n + 1):
        best: list[int] | None = None
        for a in allowed:
            if i - a >= 0 and dp[i - a] is not None:
                cand = dp[i - a] + [a]
                if best is None or len(cand) < len(best):
                    best = cand
        dp[i] = best
    if dp[n] is not None:
        return dp[n]  # exact
    return [1] * n


def split_predicted_pages(rows: list[dict[str, Any]], *, label_field: str = "predicted_label") -> list[dict[str, Any]]:
    pages: list[PagePred] = []
    for r in rows:
        pages.append(PagePred(page_index=int(r["page_index"]), label=str(r[label_field])))
    return split_page_preds(pages)


def split_page_preds(
    pages: list[PagePred],
    *,
    rules: dict[str, list[int]] | None = None,
) -> list[dict[str, Any]]:
    if rules is None:
        rules = DEFAULT_SPLIT_RULES
    if not pages:
        return []

    pages = sorted(pages, key=lambda x: x.page_index)
    docs: list[dict[str, Any]] = []

    i = 0
    doc_id = 1
    while i < len(pages):
        lab = pages[i].label
        j = i + 1
        while j < len(pages) and pages[j].label == lab and pages[j].page_index == pages[j - 1].page_index + 1:
            j += 1
        run = pages[i:j]
        run_len = len(run)
        chunk_sizes = _partition_run(run_len, rules.get(lab, [1]))

        start = 0
        for c in chunk_sizes:
            seg = run[start : start + c]
            docs.append(
                {
                    "doc_id": doc_id,
                    "label": lab,
                    "start_page": seg[0].page_index + 1,  # 1-based for readability
                    "end_page": seg[-1].page_index + 1,
                    "page_count": c,
                }
            )
            doc_id += 1
            start += c
        i = j
    return docs


def _pick_feature_columns(rows: list[dict[str, Any]], preferred_prefix: str = "fused_") -> list[str]:
    if not rows:
        return []
    keys = list(rows[0].keys())
    cols = [k for k in keys if k.startswith(preferred_prefix)]
    if cols:
        return cols
    # fallback to image scores
    cols = [k for k in keys if k.startswith("img_")]
    return cols


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return 0.0
    if not math.isfinite(x):
        return 0.0
    return x


def _cosine_from_rows(a: dict[str, Any], b: dict[str, Any], cols: list[str]) -> float:
    va = [_safe_float(a.get(c, 0.0)) for c in cols]
    vb = [_safe_float(b.get(c, 0.0)) for c in cols]
    dot = sum(x * y for x, y in zip(va, vb, strict=True))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    if na < 1e-12 or nb < 1e-12:
        return -1.0
    return dot / (na * nb)


def split_by_adjacent_cosine(
    rows: list[dict[str, Any]],
    *,
    label_field: str = "predicted_label",
    pair_label: str = "the_tiet_kiem_ban_sao",
    pair_labels: list[str] | None = None,
    cosine_threshold: float = 0.985,
    feature_prefix: str = "fused_",
    anti_merge_delta: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Split pages into document instances:
    - All labels except ``pair_label``: always 1-page docs.
    - ``pair_label``: check consecutive pages with same label and contiguous index.
      If cosine(feature(page_i), feature(page_{i+1})) >= threshold => merge into 2-page doc,
      else keep 1-page.
    """
    if not rows:
        return []
    cols = _pick_feature_columns(rows, preferred_prefix=feature_prefix)
    rs = sorted(rows, key=lambda r: int(r["page_index"]))
    docs: list[dict[str, Any]] = []
    doc_id = 1
    i = 0
    allowed_pair_labels = set(pair_labels or [pair_label])

    while i < len(rs):
        r = rs[i]
        lab = str(r[label_field])
        p = int(r["page_index"])

        if lab not in allowed_pair_labels:
            docs.append(
                {
                    "doc_id": doc_id,
                    "label": lab,
                    "start_page": p + 1,
                    "end_page": p + 1,
                    "page_count": 1,
                    "split_reason": "single_default",
                }
            )
            doc_id += 1
            i += 1
            continue

        can_pair = False
        cos = -1.0
        left_alt = None
        right_alt = None
        if i + 1 < len(rs):
            r2 = rs[i + 1]
            p2 = int(r2["page_index"])
            if str(r2[label_field]) == lab and p2 == p + 1:
                cos = _cosine_from_rows(r, r2, cols) if cols else -1.0
                if i - 1 >= 0:
                    rl = rs[i - 1]
                    pl = int(rl["page_index"])
                    if str(rl[label_field]) == lab and p == pl + 1:
                        left_alt = _cosine_from_rows(rl, r, cols) if cols else -1.0
                if i + 2 < len(rs):
                    rr = rs[i + 2]
                    pr = int(rr["page_index"])
                    if str(rr[label_field]) == lab and pr == p2 + 1:
                        right_alt = _cosine_from_rows(r2, rr, cols) if cols else -1.0
                alt_max = max(x for x in [left_alt, right_alt] if x is not None) if (left_alt is not None or right_alt is not None) else None
                pass_delta = True if alt_max is None else (cos - alt_max) >= anti_merge_delta
                can_pair = cos >= cosine_threshold and pass_delta
        if can_pair:
            docs.append(
                {
                    "doc_id": doc_id,
                    "label": lab,
                    "start_page": p + 1,
                    "end_page": int(rs[i + 1]["page_index"]) + 1,
                    "page_count": 2,
                    "adjacent_cosine": round(cos, 6),
                    "left_alt_cosine": round(left_alt, 6) if left_alt is not None and left_alt > -1 else None,
                    "right_alt_cosine": round(right_alt, 6) if right_alt is not None and right_alt > -1 else None,
                    "anti_merge_delta": anti_merge_delta,
                    "split_reason": "paired_by_cosine",
                }
            )
            doc_id += 1
            i += 2
        else:
            docs.append(
                {
                    "doc_id": doc_id,
                    "label": lab,
                    "start_page": p + 1,
                    "end_page": p + 1,
                    "page_count": 1,
                    "adjacent_cosine": round(cos, 6) if cos > -1 else None,
                    "left_alt_cosine": round(left_alt, 6) if left_alt is not None and left_alt > -1 else None,
                    "right_alt_cosine": round(right_alt, 6) if right_alt is not None and right_alt > -1 else None,
                    "anti_merge_delta": anti_merge_delta,
                    "split_reason": "single_by_cosine",
                }
            )
            doc_id += 1
            i += 1
    return docs
