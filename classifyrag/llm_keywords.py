"""
Vision-language keyword extraction for scanned PDF pages.

When OCR / text layer is empty, a vision LLM (default: Qwen3-VL-4B-Instruct) reads the
page image and outputs a comma-separated keyword string for the retriever text query path.
Text-only LLMs cannot replace this step for image-only pages.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ~4B, image+text; requires recent transformers (Qwen3VLForConditionalGeneration).
DEFAULT_VLM_MODEL = "Qwen/Qwen3-VL-4B-Instruct"

KEYWORD_PROMPT_VI_LEGACY = (
    "Trang giấy tờ tiếng Việt (ngân hàng, thẻ, tờ trình, chứng minh cư trú…). "
    "Chỉ trích TỪ KHÓA ĐẶC TRƯNG để nhận diện LOẠI tài liệu: tên/loại giấy, dòng tiêu đề chính in trên form, "
    "và tối đa vài cụm thuật ngữ/ô gắn với loại đó — đủ để phân biệt với loại giấy khác. "
    "KHÔNG liệt kê dài toàn bộ nhãn ô (STT, ngày, chữ ký, User, giao dịch thông thường…); không “đọc hết” biểu mẫu. "
    "TUYỆT ĐỐI không: họ tên, giá trị số (tiền, mã, TK), ngày tháng cụ thể, địa chỉ, CMND/CCCD, SĐT. "
    "In 8–15 cụm ngắn (mỗi cụm 1–4 từ), cách nhau dấu phẩy; không giải thích, không đánh số."
)

# Alias for callers that still import KEYWORD_PROMPT_VI
KEYWORD_PROMPT_VI = KEYWORD_PROMPT_VI_LEGACY


def keyword_prompt_vi_capped(max_keywords: int) -> str:
    """Prompt for a short, document-type keyword list (retriever text branch)."""
    n = max(1, min(int(max_keywords), 12))
    return (
        "Trang giấy tờ tiếng Việt (ngân hàng, form, tờ trình, chứng minh cư trú…). "
        f"Chỉ in tối đa {n} TỪ KHÓA ĐẶC TRƯNG (mỗi cụm 1–4 từ) để phân biệt LOẠI giấy: "
        "tên/loại giấy hoặc tiêu đề chính; thêm vài cụm chỉ đúng loại đó nếu cần. "
        "KHÔNG dump nhãn ô chung, không liệt kê cột giao dịch dài. "
        "TUYỆT ĐỐI không: họ tên, số/mã/giá trị, ngày cụ thể, địa chỉ, CMND/CCCD, SĐT. "
        "Nối bằng dấu phẩy; không giải thích, không đánh số."
    )


STRUCTURAL_FEATURE_PROMPT = (
    "Phân tích trang tài liệu tiếng Việt (ngân hàng, form, tờ trình…). "
    "Mô tả các ĐẶC ĐIỂM CẤU TRÚC giúp xác định LOẠI tài liệu và VỊ TRÍ trang (đầu/giữa/cuối):\n"
    "- Tiêu đề/tên loại tài liệu chính (nếu có — thường ở trang đầu)\n"
    "- Có phần chữ ký / con dấu / ô ký tên không (thường ở trang cuối)\n"
    "- Bảng: số cột, tiêu đề cột, loại dữ liệu bảng chứa\n"
    "- Bố cục: form điền thông tin, bảng liệt kê, văn bản liên tục, tóm tắt\n"
    "- Đặc điểm vị trí: header/logo công ty (trang đầu), phần tổng cộng/kết luận (trang cuối), "
    "bảng tiếp nối không tiêu đề (trang giữa)\n"
    "TUYỆT ĐỐI KHÔNG liệt kê: mã số, ngày tháng cụ thể, tên người, số tài khoản, giá trị tiền, "
    "mã giao dịch, số in — những thông tin không cố định và tài liệu nào cũng có.\n"
    "Tối đa 10 cụm mô tả ngắn, cách nhau dấu phẩy; không giải thích, không đánh số."
)


BLANK_PAGE_VLM_PROMPT = (
    "Trang này có phải trang giấy trắng hoặc gần như không có nội dung (không form, không chữ đọc được) không? "
    "Chỉ trả lời đúng MỘT từ tiếng Anh: blank nếu đúng là trang trắng / gần trắng; "
    "nếu có nội dung rõ ràng thì trả lời: content. "
    "Không dấu câu, không giải thích."
)


def structural_prompt_vi(max_keywords: int = 10) -> str:
    """Prompt for structural features (layout, position cues) — used for label + label_2."""
    n = max(1, min(int(max_keywords), 15))
    return (
        "Phân tích trang tài liệu tiếng Việt (ngân hàng, form, tờ trình…). "
        f"Mô tả tối đa {n} ĐẶC ĐIỂM CẤU TRÚC giúp xác định LOẠI tài liệu và VỊ TRÍ trang "
        "(trang đầu/giữa/cuối của tài liệu):\n"
        "- Tiêu đề/tên loại tài liệu (nếu có)\n"
        "- Có phần chữ ký / con dấu / ô ký tên không\n"
        "- Bảng: số cột, tiêu đề cột chính\n"
        "- Bố cục: form điền, bảng liệt kê, văn bản liên tục\n"
        "- Header/logo (trang đầu), tổng cộng/kết luận (trang cuối), bảng tiếp nối (trang giữa)\n"
        "KHÔNG liệt kê: mã số, ngày tháng, tên người, số tài khoản, giá trị tiền, mã giao dịch.\n"
        "Nối bằng dấu phẩy; không giải thích, không đánh số."
    )


def normalize_keyword_string(raw: str, max_keywords: int | None) -> str:
    """
    Split comma/semicolon-separated keywords, de-duplicate in order, cap count.
    If max_keywords is None or <= 0, return stripped raw only.
    """
    if not raw or not raw.strip():
        return ""
    s = raw.strip()
    if max_keywords is None or max_keywords <= 0:
        return s
    cap = int(max_keywords)
    out: list[str] = []
    for chunk in s.replace(";", ",").replace("\n", ",").split(","):
        part = chunk.strip().strip(".")
        if not part:
            continue
        key = part.casefold()
        if any(o.casefold() == key for o in out):
            continue
        out.append(part)
        if len(out) >= cap:
            break
    return ", ".join(out)

_vlm_cache: Optional[tuple[str, Any, Any, str]] = None


def _pick_device(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _vlm_model_class(model_id: str) -> type[Any]:
    from transformers import AutoConfig, Qwen2VLForConditionalGeneration

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    mt = getattr(cfg, "model_type", None)
    if mt == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    if mt in ("qwen2_vl", "qwen2vl"):
        return Qwen2VLForConditionalGeneration
    raise ValueError(
        f"Unsupported VLM model_type={mt!r} for {model_id!r}. "
        "Use a Qwen2-VL or Qwen3-VL checkpoint, or pass a compatible --vlm-model."
    )


def load_vlm(model_id: str, device: Optional[str] = None) -> tuple[Any, Any, str]:
    """Load and cache Qwen2-VL / Qwen3-VL (or compatible) model + processor."""
    global _vlm_cache
    dev = _pick_device(device)
    if _vlm_cache is not None and _vlm_cache[0] == model_id and _vlm_cache[3] == dev:
        return _vlm_cache[1], _vlm_cache[2], dev

    from transformers import AutoProcessor

    ModelCls = _vlm_model_class(model_id)
    logger.info("Loading VLM %s (%s) on %s ...", model_id, ModelCls.__name__, dev)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if dev.startswith("cuda") else torch.float32
    model = ModelCls.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    )
    model = model.to(dev)
    model.eval()
    _vlm_cache = (model_id, model, processor, dev)
    return model, processor, dev


@torch.inference_mode()
def keywords_from_image_vlm(
    image: Image.Image,
    *,
    model_id: str = DEFAULT_VLM_MODEL,
    device: Optional[str] = None,
    max_new_tokens: int = 256,
    prompt: Optional[str] = None,
    max_keywords: Optional[int] = 10,
    use_structural_prompt: bool = True,
) -> str:
    """
    Generate a keyword string from a page image using Qwen2/3-VL Instruct.
    ``max_keywords``: cap and light de-duplication; use ``None`` or ``0`` for legacy long lists.
    ``use_structural_prompt``: use the structural feature prompt (layout/position cues).
    Returns empty string on failure.
    """
    image = image.convert("RGB")
    model, processor, dev = load_vlm(model_id, device=device)
    if prompt is None:
        if use_structural_prompt:
            prompt = structural_prompt_vi(max_keywords or 10)
        elif max_keywords is not None and max_keywords > 0:
            prompt = keyword_prompt_vi_capped(max_keywords)
        else:
            prompt = KEYWORD_PROMPT_VI_LEGACY

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    in_len = inputs["input_ids"].shape[1]
    gen = processor.batch_decode(out[:, in_len:], skip_special_tokens=True)[0]
    return normalize_keyword_string(gen, max_keywords)


@torch.inference_mode()
def keywords_blank_page_vlm(
    image: Image.Image,
    *,
    model_id: str = DEFAULT_VLM_MODEL,
    device: Optional[str] = None,
    max_new_tokens: int = 32,
) -> str:
    """
    Run VLM on a blank-page prototype; prototype text for the index is always ``blank``.
    Raw generation is logged at debug level.
    """
    image = image.convert("RGB")
    model, processor, dev = load_vlm(model_id, device=device)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": BLANK_PAGE_VLM_PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    in_len = inputs["input_ids"].shape[1]
    gen = processor.batch_decode(out[:, in_len:], skip_special_tokens=True)[0].strip()
    logger.debug("blank_page_vlm raw: %s", gen)
    return "blank"


def clear_vlm_cache() -> None:
    global _vlm_cache
    _vlm_cache = None
