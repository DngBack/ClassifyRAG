from __future__ import annotations

import os
import unicodedata
from typing import Optional

# Stable internal keys for the four document groups (filename prefix → label id)
_PREFIX_TO_LABEL: tuple[tuple[str, str], ...] = (
    ("Giấy gửi tiền tiết kiệm_", "giay_gui_tien_tiet_kiem"),
    ("Thẻ tiết kiệm_bản sao_", "the_tiet_kiem_ban_sao"),
    ("Tờ trình ưu đãi lãi suất_", "to_trinh_uu_dai_lai_suat"),
    ("Giấy tờ chứng minh tình trạng cư trú_", "giay_to_chung_minh_cu_tru"),
)

# Fixed display order for reports and CSV columns
ORDERED_LABELS: tuple[str, ...] = (
    "giay_gui_tien_tiet_kiem",
    "the_tiet_kiem_ban_sao",
    "to_trinh_uu_dai_lai_suat",
    "giay_to_chung_minh_cu_tru",
)

LABELS: tuple[str, ...] = ORDERED_LABELS


def normalize_filename(name: str) -> str:
    return unicodedata.normalize("NFC", name)


def label_from_filename(path_or_name: str) -> Optional[str]:
    """
    Map a sample PDF path or basename to one of LABELS using the Vietnamese prefix
    before the numeric suffix (e.g. ..._7.pdf).
    """
    base = normalize_filename(os.path.basename(path_or_name))
    if not base.lower().endswith(".pdf"):
        return None
    stem = base[:-4]
    for prefix, label in _PREFIX_TO_LABEL:
        if stem.startswith(prefix):
            return label
    return None
