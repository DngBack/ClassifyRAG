"""
Streamlit UI cho ClassifyRAG: model chỉ nạp khi bạn chạy Phân loại / Trang trắng.

Chạy từ thư mục repo:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import csv
import io
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import streamlit as st

from classifyrag.llm_keywords import DEFAULT_VLM_MODEL
from classifyrag.postprocess_split import split_by_adjacent_cosine

ROOT = Path(__file__).resolve().parent
WEB_DEBUG_DIR = ROOT / "data" / "web_debug"

# Suppress noisy upstream deprecation warning from transformers dynamic modules.
warnings.filterwarnings(
    "ignore",
    message=r"Accessing `__path__` from `\.models\.vit\.image_processing_vit`.*",
)


def _init_state() -> None:
    if "timing_log" not in st.session_state:
        st.session_state.timing_log = []
    if "model_bundles" not in st.session_state:
        st.session_state.model_bundles = {}


def _log_timing(action: str, seconds: float, **extra: Any) -> None:
    _init_state()
    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "action": action,
        "seconds": round(seconds, 3),
    }
    for k, v in extra.items():
        row[k] = str(v)[:300]
    st.session_state.timing_log.insert(0, row)


def _bundle_key(model_id: str, device: Optional[str]) -> str:
    return f"{model_id}\t{device or 'auto'}"


def get_embedding_model(model_id: str, device: Optional[str]) -> tuple[Any, Any, str]:
    """Lazy-load ColQwen bundle; cache theo (model_id, device)."""
    _init_state()
    key = _bundle_key(model_id, device)
    if key not in st.session_state.model_bundles:
        from classifyrag.colsmol_scorer import load_model

        dev_s = device.strip() if device else ""
        with st.spinner(f"Đang nạp model (lần đầu): `{model_id}` …"):
            t0 = time.perf_counter()
            m, p, d = load_model(model_id=model_id, device=dev_s or None)
            _log_timing(
                "load_embedding_model",
                time.perf_counter() - t0,
                model_id=model_id,
                device=str(d),
            )
        st.session_state.model_bundles[key] = (m, p, d)
    return st.session_state.model_bundles[key]


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _rows_to_csv_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _write_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def page_classify() -> None:
    st.subheader("Phân loại trang PDF")
    up = st.file_uploader("PDF", type=["pdf"], key="cls_pdf")
    idx_path = st.text_input("Prototype index (.pt)", value="data/index/prototypes.pt", key="cls_idx")
    mode = st.selectbox(
        "Chế độ",
        options=["image", "text", "fused", "compare"],
        format_func=lambda m: {
            "image": "Chỉ ảnh (MaxSim)",
            "text": "Chỉ text (VLM → prototype text)",
            "fused": "Fused ảnh + text",
            "compare": "So sánh cả 3 (VLM-text | ảnh | fused)",
        }[m],
        key="cls_mode",
    )
    with st.expander("Rule split sau classify", expanded=False):
        st.caption("Mặc định tất cả label tách 1 trang. Chỉ một label được phép ghép 2 trang theo cosine của 2 trang liên tiếp.")
        pair_labels_text = st.text_input(
            "Label cho phép ghép 2 trang (phân tách bằng dấu phẩy)",
            value="the_tiet_kiem_ban_sao,to_trinh_uu_dai_lai_suat",
            key="cls_pair_labels",
        )
        pair_cosine = st.slider("Ngưỡng cosine để ghép cặp", 0.8, 0.9999, 0.985, 0.0005)
        anti_merge = st.checkbox("Bật anti-merge guard (tránh ghép nhầm trang đơn)", value=False)
        anti_merge_delta = st.slider(
            "Delta guard: cos(i,i+1) - max(cos(i-1,i), cos(i+1,i+2)) >= delta",
            0.0,
            0.02,
            0.001,
            0.0005,
            disabled=not anti_merge,
        )
        feature_prefix = st.selectbox(
            "Nguồn score để tính cosine",
            options=["fused_", "img_"],
            index=0,
            help="Dùng vector score theo label (fused_* hoặc img_).",
        )
    with st.expander("Tuỳ chọn nâng cao"):
        w_img = st.slider("Trọng số nhánh ảnh (fused)", 0.0, 1.0, 0.7, 0.05)
        dpi = st.number_input("DPI render", min_value=72.0, max_value=300.0, value=144.0)
        max_pages = st.number_input("Giới hạn số trang (0 = hết)", min_value=0, value=0)
        batch_size = st.number_input("Batch size", min_value=1, value=4)
        device_in = st.text_input("Device (để trống = auto)", value="", key="cls_dev")
        ocr = st.checkbox("OCR (Tesseract) khi PDF không có text layer", value=False)
        ocr_dpi = st.number_input("OCR DPI", min_value=72, value=150)
        ocr_lang = st.text_input("OCR lang", value="vie+eng")
        vlm_keywords = st.checkbox("--vlm-keywords (cho fused: bổ sung VLM khi trang trống text)", value=False)
        vlm_always = st.checkbox("--vlm-always", value=False)
        vlm_model = st.text_input("VLM model id", value=DEFAULT_VLM_MODEL)
        vlm_max_tokens = st.number_input("VLM max tokens", min_value=32, value=256)
        vlm_keyword_count = st.number_input(
            "VLM tối đa số keyword (0 = danh sách dài kiểu cũ)",
            min_value=0,
            max_value=12,
            value=5,
        )
        char_text = st.checkbox("Characteristic text (giống build_index)", value=False)
        score_style = st.selectbox(
            "Score style",
            options=("colpali", "intrinsic"),
            index=0,
            help="intrinsic: ảnh ~[0,1], text cosine pooled ~[0,1], fused không min-max theo nhánh.",
        )

    if st.button("Chạy phân loại", type="primary", key="cls_run"):
        if up is None:
            st.warning("Tải lên một file PDF.")
            return
        idx_file = _resolve_path(idx_path)
        if not idx_file.is_file():
            st.error(f"Không tìm thấy index: {idx_file}")
            return
        from classifyrag.colsmol_scorer import load_index
        from classifyrag.web_runner import ClassifyRunConfig, iter_classify_rows

        t_idx = time.perf_counter()
        idx = load_index(idx_file)
        _log_timing("load_prototype_index", time.perf_counter() - t_idx, path=str(idx_file))
        text_total = 0
        text_usable = 0
        if idx.text_embs is not None:
            text_total = len(idx.text_embs)
            text_usable = sum(1 for e in idx.text_embs if e is not None)
        if text_total == 0 or text_usable == 0:
            st.warning(
                "Index hiện không có text prototypes usable "
                f"({text_usable}/{text_total}). Chế độ `text`/`compare` sẽ fallback về ảnh "
                "và các cột txt_* có thể là -inf. Hãy build_index lại với OCR/VLM để có text embeddings."
            )

        dev = device_in.strip() or None
        model, processor, device = get_embedding_model(idx.model_id, dev)
        proto_labels = [m["label"] for m in idx.manifest]

        cfg = ClassifyRunConfig(
            mode=mode,
            w_img=w_img,
            dpi=dpi,
            max_pages=None if max_pages == 0 else int(max_pages),
            batch_size=int(batch_size),
            ocr=ocr,
            ocr_dpi=int(ocr_dpi),
            ocr_lang=ocr_lang,
            vlm_keywords=vlm_keywords,
            vlm_model=vlm_model,
            vlm_device=dev,
            vlm_max_tokens=int(vlm_max_tokens),
            vlm_keyword_count=int(vlm_keyword_count),
            vlm_always=vlm_always,
            characteristic_text=char_text,
            score_style=score_style,
        )

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(up.getvalue())
            tmp_path = Path(f.name)
        try:
            prog = st.progress(0.0, text="Đang phân loại…")
            t0 = time.perf_counter()
            rows = iter_classify_rows(tmp_path, idx, model, processor, device, cfg, proto_labels)
            _log_timing(
                "classify_pdf",
                time.perf_counter() - t0,
                pages=str(len(rows)),
                mode=mode,
            )
            prog.progress(1.0, text="Xong.")
        finally:
            tmp_path.unlink(missing_ok=True)

        st.success(f"{len(rows)} trang — xem tab **Thời gian chạy** để biết chi tiết.")
        # Always overwrite latest debug files for quick inspection.
        classify_debug_path = WEB_DEBUG_DIR / "classify_latest.csv"
        _write_rows_csv(classify_debug_path, rows)

        if mode == "compare":
            ui_rows = [
                {
                    "page": int(r["page_index"]) + 1,
                    "label_vlm_text": r.get("predicted_label_vlm_text"),
                    "label_image": r.get("predicted_label_image"),
                    "label_fused": r.get("predicted_label_fused"),
                }
                for r in rows
            ]
        else:
            ui_rows = [
                {
                    "page": int(r["page_index"]) + 1,
                    "label": r.get("predicted_label"),
                }
                for r in rows
            ]
        st.dataframe(ui_rows, use_container_width=True, hide_index=True)

        # Rule-based document grouping (no LLM)
        docs = split_by_adjacent_cosine(
            rows,
            label_field="predicted_label",
            pair_labels=[s.strip() for s in pair_labels_text.split(",") if s.strip()],
            cosine_threshold=float(pair_cosine),
            feature_prefix=feature_prefix,
            anti_merge_delta=float(anti_merge_delta if anti_merge else 0.0),
        )

        docs_path = WEB_DEBUG_DIR / "classify_docs_latest.csv"
        _write_rows_csv(docs_path, docs)
        st.markdown("**Tách bộ tài liệu (rule-based)**")
        st.dataframe(docs, use_container_width=True, hide_index=True)
        st.caption(f"Debug docs split (ghi đè): `{docs_path}`")
        st.download_button(
            "Tải CSV split tài liệu",
            _rows_to_csv_bytes(docs),
            file_name="classify_docs_split.csv",
            mime="text/csv",
        )
        st.download_button(
            "Tải CSV chi tiết",
            _rows_to_csv_bytes(rows),
            file_name="classify_result.csv",
            mime="text/csv",
        )
        st.caption(f"Debug file (ghi đè mỗi lần chạy): `{classify_debug_path}`")
        if mode == "compare":
            sum_cols = [
                "page_index",
                "predicted_label_vlm_text",
                "predicted_label_image",
                "predicted_label_fused",
            ]
            if rows and all(c in rows[0] for c in sum_cols):
                summary_rows = [{c: r[c] for c in sum_cols} for r in rows]
                compare_summary_path = WEB_DEBUG_DIR / "classify_compare_summary_latest.csv"
                _write_rows_csv(compare_summary_path, summary_rows)
                st.download_button(
                    "Tải CSV tóm tắt (compare)",
                    _rows_to_csv_bytes(summary_rows),
                    file_name="classify_summary.csv",
                    mime="text/csv",
                )
                st.caption(f"Debug compare summary (ghi đè): `{compare_summary_path}`")


def page_blank() -> None:
    st.subheader("Điểm trang trắng")
    up = st.file_uploader("PDF", type=["pdf"], key="blk_pdf")
    idx_path = st.text_input("Blank index (.pt)", value="data/index/blank_prototypes.pt", key="blk_idx")
    threshold = st.slider("Ngưỡng cosine01 (blank nếu ≥)", 0.5, 0.99, 0.85, 0.01)
    max_pages = st.number_input("Tối đa số trang", min_value=1, value=200)
    dpi = st.number_input("DPI", min_value=72.0, max_value=300.0, value=144.0)
    batch_size = st.number_input("Batch size", min_value=1, value=4, key="blk_bs")
    gt_ocr = st.checkbox("GT: dùng OCR để biết trang có chữ (chậm hơn)", value=False)
    ocr_dpi = st.number_input("OCR DPI (blank)", min_value=72, value=150, key="blk_ocrdpi")
    ocr_lang = st.text_input("OCR lang", value="vie+eng", key="blk_ocr_lang")
    device_in = st.text_input("Device (để trống = auto)", value="", key="blk_dev")

    if st.button("Chạy blank scoring", type="primary", key="blk_run"):
        if up is None:
            st.warning("Tải lên một file PDF.")
            return
        idx_file = _resolve_path(idx_path)
        if not idx_file.is_file():
            st.error(f"Không tìm thấy index: {idx_file}")
            return
        from classifyrag.blank_page import load_blank_index
        from classifyrag.web_runner import BlankRunConfig, iter_blank_rows

        t_idx = time.perf_counter()
        bidx = load_blank_index(idx_file)
        _log_timing("load_blank_index", time.perf_counter() - t_idx, path=str(idx_file))

        dev = device_in.strip() or None
        model, processor, device = get_embedding_model(bidx.model_id, dev)

        cfg = BlankRunConfig(
            max_pages=int(max_pages),
            threshold=float(threshold),
            dpi=float(dpi),
            batch_size=int(batch_size),
            gt_ocr=gt_ocr,
            ocr_dpi=int(ocr_dpi),
            ocr_lang=ocr_lang,
        )

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(up.getvalue())
            tmp_path = Path(f.name)
        try:
            t0 = time.perf_counter()
            rows = iter_blank_rows(tmp_path, bidx, model, processor, device, cfg)
            _log_timing(
                "blank_pdf",
                time.perf_counter() - t0,
                pages=str(len(rows)),
                threshold=str(threshold),
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        st.success(f"{len(rows)} trang — threshold={threshold}")
        blank_debug_path = WEB_DEBUG_DIR / "blank_latest.csv"
        _write_rows_csv(blank_debug_path, rows)
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.download_button(
            "Tải CSV",
            _rows_to_csv_bytes(rows),
            file_name="blank_scores.csv",
            mime="text/csv",
        )
        st.caption(f"Debug file (ghi đè mỗi lần chạy): `{blank_debug_path}`")


def page_timing() -> None:
    st.subheader("Nhật ký thời gian")
    st.caption("Mỗi lần nạp model hoặc chạy pipeline được ghi lại (session hiện tại).")
    _init_state()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Xoá log trong session"):
            st.session_state.timing_log = []
            st.rerun()
    with c2:
        if st.button("Huỷ cache model (unload GPU/RAM)"):
            st.session_state.model_bundles = {}
            _log_timing("unload_all_models", 0.0)
            st.rerun()
    if not st.session_state.timing_log:
        st.info("Chưa có sự kiện. Hãy chạy Phân loại hoặc Trang trắng.")
        return
    st.dataframe(st.session_state.timing_log, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="ClassifyRAG", layout="wide")
    _init_state()
    st.title("ClassifyRAG")
    st.caption(f"Thư mục dự án: `{ROOT}` — model chỉ nạp khi bạn bấm chạy.")

    tab_cls, tab_blk, tab_time = st.tabs(["Phân loại", "Trang trắng", "Thời gian chạy"])
    with tab_cls:
        page_classify()
    with tab_blk:
        page_blank()
    with tab_time:
        page_timing()


if __name__ == "__main__":
    main()
