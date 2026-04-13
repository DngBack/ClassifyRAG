from __future__ import annotations

import os
import unicodedata
from typing import Any, Optional

# Prototype / classifier label for blank or near-blank pages (indexed from ``data/blank_data``).
BLANK_LABEL = "blank"

# Stable internal keys for document groups (filename prefix → label id)
_PREFIX_TO_LABEL: tuple[tuple[str, str], ...] = (
    ("Giấy gửi tiền tiết kiệm_", "giay_gui_tien_tiet_kiem"),
    ("Thẻ tiết kiệm_bản sao_", "the_tiet_kiem_ban_sao"),
    ("Tờ trình ưu đãi lãi suất_", "to_trinh_uu_dai_lai_suat"),
    ("to_trinh_lai_suat_uu_dai_khach_hang_", "to_trinh_lai_suat_uu_dai_khach_hang"),
    ("Giấy tờ chứng minh tình trạng cư trú_", "giay_to_chung_minh_cu_tru"),
    ("so_quy_", "so_quy"),
    ("liet_ke_giao_dich_", "liet_ke_giao_dich"),
    ("giay_de_nghi_tiep_quy_", "giay_de_nghi_tiep_quy"),
    ("giay_de_nghi_phong_toa_", "giay_de_nghi_phong_toa"),
    ("giay_phong_toa_tam_khoa_tk_", "giay_phong_toa_tam_khoa_tk"),
    ("giay_go_tam_khoa_tai_khoan_", "giay_go_tam_khoa_tai_khoan"),
    # Must precede giay_rut_tien_ — same stem prefix.
    ("giay_rut_tien_tiet_kiem_", "giay_rut_tien_tiet_kiem"),
    ("giay_rut_tien_", "giay_rut_tien"),
    ("giay_nop_chuyen_tien_", "giay_nop_chuyen_tien"),
    ("lenh_chuyen_tien_", "lenh_chuyen_tien"),
    ("giay_de_nghi_hoan_quy_", "giay_de_nghi_hoan_quy"),
)

# Fixed display order for reports and CSV columns
ORDERED_LABELS: tuple[str, ...] = (
    "giay_gui_tien_tiet_kiem",
    "the_tiet_kiem_ban_sao",
    "to_trinh_uu_dai_lai_suat",
    "to_trinh_lai_suat_uu_dai_khach_hang",
    "giay_to_chung_minh_cu_tru",
    "so_quy",
    "liet_ke_giao_dich",
    "giay_de_nghi_tiep_quy",
    "giay_de_nghi_phong_toa",
    "giay_phong_toa_tam_khoa_tk",
    "giay_go_tam_khoa_tai_khoan",
    "giay_rut_tien",
    "giay_rut_tien_tiet_kiem",
    "giay_nop_chuyen_tien",
    "lenh_chuyen_tien",
    "giay_de_nghi_hoan_quy",
    BLANK_LABEL,
)

LABELS: tuple[str, ...] = ORDERED_LABELS

POSITION_LABELS: tuple[str, ...] = ("start", "mid", "end", "none")


def position_key_for_manifest(m: dict[str, Any]) -> str:
    """Map manifest row to a position bucket for scoring (label_2 / intrinsic path)."""
    if m.get("label") == BLANK_LABEL:
        return "none"
    v = m.get("label_2")
    if v is None:
        return "start"
    return str(v)


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
