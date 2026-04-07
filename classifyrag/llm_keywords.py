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

KEYWORD_PROMPT_VI = (
    "Đây là một trang giấy tờ tiếng Việt (giấy ngân hàng, thẻ, tờ trình, giấy cư trú, v.v.). "
    "Chỉ liệt kê các manh mối cố định theo LOẠI tài liệu: tên/loại giấy hoặc tiêu đề in sẵn, "
    "tên các ô/trường trên biểu mẫu (ví dụ: số tiền gửi, loại tiền, định kỳ trả lãi, ngày mở, ghi nợ), "
    "và thuật ngữ đặc trưng của loại giấy đó. "
    "TUYỆT ĐỐI KHÔNG ghi: họ tên người, số tiền hoặc số cụ thể, số tài khoản, mã seri, ngày tháng năm, "
    "địa chỉ, CMND/CCCD, số điện thoại. "
    "Từ 10 đến 25 cụm ngắn, cách nhau bởi dấu phẩy. Chỉ in danh sách, không giải thích."
)

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
    prompt: str = KEYWORD_PROMPT_VI,
) -> str:
    """
    Generate a keyword string from a page image using Qwen2-VL or Qwen3-VL Instruct.
    Returns empty string on failure.
    """
    image = image.convert("RGB")
    model, processor, dev = load_vlm(model_id, device=device)

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
    return gen.strip()


def clear_vlm_cache() -> None:
    global _vlm_cache
    _vlm_cache = None
