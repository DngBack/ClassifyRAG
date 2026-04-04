"""
Vision-language keyword extraction for scanned PDF pages.

When OCR / text layer is empty, a small VLM (default: Qwen2-VL-2B-Instruct) reads the
page image and outputs a comma-separated keyword string for ColSmol's text query path.
Text-only LLMs cannot replace this step for image-only pages.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_VLM_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

KEYWORD_PROMPT_VI = (
    "Đây là một trang giấy tờ tiếng Việt (có thể là giấy ngân hàng, thẻ, tờ trình, hoặc giấy tờ cư trú). "
    "Hãy liệt kê từ 10 đến 20 từ khóa ngắn (danh từ hoặc cụm 2–4 từ) mô tả nội dung và loại giấy, "
    "cách nhau bởi dấu phẩy. Chỉ in danh sách từ khóa, không giải thích thêm."
)

_vlm_cache: Optional[tuple[str, Any, Any, str]] = None


def _pick_device(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def load_vlm(model_id: str, device: Optional[str] = None) -> tuple[Any, Any, str]:
    """Load and cache Qwen2-VL (or compatible) model + processor."""
    global _vlm_cache
    dev = _pick_device(device)
    if _vlm_cache is not None and _vlm_cache[0] == model_id and _vlm_cache[3] == dev:
        return _vlm_cache[1], _vlm_cache[2], dev

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    logger.info("Loading VLM %s on %s ...", model_id, dev)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if dev.startswith("cuda") else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(
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
    Generate a keyword string from a page image using Qwen2-VL Instruct.
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
