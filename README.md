# ClassifyRAG

Phân loại trang PDF bằng **ColQwen3.5** (mặc định `athrael-soju/colqwen3.5-4.5B-v3`): so khớp ảnh trang và (tuỳ chọn) nhánh văn bản với **chỉ mẫu (prototype)** đã index. Repo cũng hỗ trợ **phát hiện trang trắng / không có nội dung chữ** dựa trên ảnh tham chiếu trong `data/blank_data`.

## Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Giao diện Streamlit

Model ColQwen **chỉ được nạp** khi bạn bấm chạy (Phân loại / Trang trắng); tab **Thời gian chạy** ghi lại thời gian nạp model và xử lý PDF trong session.

```bash
cd /path/to/ClassifyRAG
./scripts/run_streamlit.sh
```

Script sẽ tự chọn cổng trống trong dải `7860-7870`. Có thể ép cổng cụ thể:

```bash
./scripts/run_streamlit.sh 7865
```

Trong tab **Phân loại**, phần **Rule split sau classify** có bộ tách hậu xử lý (không dùng LLM):
- Mặc định tách 1 trang/doc cho mọi nhãn.
- Với các nhãn cho phép ghép 2 trang, hệ thống xét cosine của cặp trang liên tiếp trên vector score (`fused_*` hoặc `img_*`).
- Có tuỳ chọn **anti-merge guard**: chỉ ghép nếu `cos(i,i+1) - max(cos(i-1,i), cos(i+1,i+2)) >= delta` để giảm ghép nhầm các trang đơn có mẫu quá giống nhau.

- **GPU** khuyến nghị cho ColQwen3.5 (khoảng ~4.5B tham số).
- **OCR (Tesseract)** tùy chọn: cần khi PDF scan không có lớp text (`classify_pdf --ocr`, `eval_blank_pdf --gt-ocr`). Ví dụ Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-vie tesseract-ocr-eng`.

## Tính năng tổng quan

| Tính năng | Mô tả | Entry point |
|-----------|--------|-------------|
| **Index chỉ mẫu theo nhãn** | Embed ảnh + văn bản từ PDF mẫu (tiền tố tên file → nhãn) | `python -m classifyrag.build_index` |
| **Phân loại từng trang PDF** | Dự đoán nhóm tài liệu cho mỗi trang | `python -m classifyrag.classify_pdf` |
| **Đánh giá độ chính xác mẫu** | So khớp với nhãn kỳ vọng từ tên file | `python -m classifyrag.eval_samples` |
| **Index trang trắng** | Embed ảnh tham chiếu “trắng” (PNG/JPG trong thư mục) | `python -m classifyrag.build_blank_index` |
| **Phát hiện trang trắng** | So điểm ảnh trang với index trắng + ngưỡng | `python -m classifyrag.eval_blank_pdf` |

---

## 1. Phân loại tài liệu (ColQwen3.5 prototype)

### Dữ liệu mẫu

- Thư mục mặc định: `data/Sample_document/`.
- Tên file PDF phải bắt đầu bằng một trong các tiền tố đã định nghĩa (xem `classifyrag/labels.py` → `label_from_filename`). Mỗi trang trong file đó được gán cùng nhãn với file.

### Tạo index

```bash
python -m classifyrag.build_index \
  --samples-dir data/Sample_document \
  --output data/index/prototypes.pt \
  --manifest-json data/index/manifest.json
```

Các tùy chọn quan trọng:

- `--dpi` — DPI render trang (mặc định 144).
- `--ocr` — bật OCR khi không có text layer (cần Tesseract).
- `--vlm-keywords` — sinh từ khóa bằng vision LLM khi thiếu text (nặng, cần model HF). Kèm `--vlm-keyword-count` (mặc định 5) để chỉ giữ vài từ khóa đặc trưng; `0` = prompt dài hơn, khoảng 8–15 cụm đặc trưng (không dump cả form).
- `--characteristic-text` — chỉ giữ tiêu đề/nhãn trường cho nhánh text (phù hợp form); dùng **cùng cờ** khi classify.

### Phân loại một PDF

```bash
python -m classifyrag.classify_pdf \
  --pdf path/to/doc.pdf \
  --index data/index/prototypes.pt \
  --output out/pages.csv \
  --format csv
```

- `--max-pages N` — chỉ xử lý N trang đầu.
- `--w-img` — trọng số nhánh ảnh (còn lại cho text), mặc định 0.7.
- `--summary PATH` — thêm file CSV chỉ hai cột `page_index`, `predicted_label` (xem nhanh khi chưa có ground truth); `--output` vẫn là bản đầy đủ điểm số / debug.
- `--mode image` (mặc định) — chỉ MaxSim ảnh. `text` — VLM trích keyword rồi so với prototype **text** (giống `build_index --vlm-keywords`). `fused` — gộp ảnh + text (text lấy từ PDF/OCR + `--vlm-keywords` / `--vlm-always` như pipeline hiện tại). `compare` — một lần chạy, CSV có ba nhãn: VLM-text | image | fused; `--summary` ghi bốn cột (page + ba nhãn).
- `--ocr`, `--vlm-keywords`, `--characteristic-text` — giống logic lúc build index (nên **khớp** với cách đã build).
- `--other-threshold` — gán `other` khi top-1 score trên **nhánh đang dùng để chấm** thấp hơn ngưỡng. Với `fused` + text usable, score đã min-max trong trang nên ngưỡng này **thường ít kích hoạt**; nên kết hợp `--other-min-margin-norm`.
- `--other-min-margin-norm M` — gán `other` khi `margin_norm` (khoảng cách top1–top2 sau chuẩn hóa trong trang) **nhỏ hơn M** (trang mơ hồ, 4 lớp gần nhau). Thử 0.05–0.12 trên tập validation. CSV có `other_triggered_top1` / `other_triggered_margin_norm`.
- `--label-score-agg topk_mean --label-score-topk 3` (mặc định) — tính điểm label bằng trung bình top-k prototype, giảm lệch khi số prototype mỗi nhãn không đều.
- `--score-style intrinsic` — nhánh ảnh: trung bình top-k điểm MaxSim/trang prototype, chia cho số token ảnh query, clamp [0,1]; nhánh text: mean-pool embedding rồi cosine mapped [0,1], top-k mean/prototype; `fused_*` = cộng có trọng số **không** min-max từng nhánh. Mặc định `colpali` giữ hành vi cũ.

### Đánh giá nhanh trên thư mục mẫu

```bash
python -m classifyrag.eval_samples \
  --samples-dir data/Sample_document \
  --index data/index/prototypes.pt
```

In `Page accuracy: đúng/tổng` và in lỗi ra stderr nếu có trang sai.

---

## 2. Phát hiện trang trắng (blank / không có chữ trên trang)

Ý tưởng: index các **ảnh tham chiếu** “trắng” trong `data/blank_data/` (PNG, JPG, …), khi inference embed ảnh trang PDF và so với các prototype đó.

### Tạo index trang trắng

```bash
python -m classifyrag.build_blank_index \
  --samples-dir data/blank_data \
  --output data/index/blank_prototypes.pt \
  --manifest-json data/index/blank_manifest.json
```

### Chạy đánh giá / gán nhãn trên PDF

```bash
python -m classifyrag.eval_blank_pdf \
  --pdf path/to/doc.pdf \
  --index data/index/blank_prototypes.pt \
  --max-pages 50 \
  --threshold 0.85 \
  --output out/blank_scores.csv
```

Cột chính:

- `cosine01_vs_blank` — điểm trong **[0, 1]** (dùng cho ngưỡng; mặc định CLI `0.85`).
- `raw_maxsim_vs_blank` — điểm **MaxSim** gốc (tổng tương tác muộn, **không** giới hạn [0,1]; chỉ để debug).
- `pred_blank` — `cosine01_vs_blank >= threshold`.

**Cách tính `cosine01`:** trung bình pooling vector đa-patch, chuẩn hoá L2, cosine similarity với từng prototype, lấy **max** trên các ảnh tham chiếu, rồi ánh xạ \([-1,1] \rightarrow [0,1]\) bằng \((\cos + 1) / 2\). **Không** dùng cùng một ngưỡng trên `raw_maxsim` như trên `cosine01` vì thang đo khác nhau.

### Ground truth “có chữ / không chữ” khi đánh giá

- **Mặc định:** chỉ xét **text nhúng trong PDF**. Với file scan không có lớp text, mọi trang đều “không có text” → ma trận nhầm lẫn **không phản ánh** “trắng có nội dung hình/scan”.
- **`--gt-ocr`:** coi trang là **không trắng** nếu OCR (Tesseract) bắt được chữ. Phù hợp PDF scan; chậm hơn và cần cài Tesseract + `tessdata` (ví dụ `--ocr-lang vie+eng`).

### API Python (một trang ảnh)

```python
from classifyrag.blank_page import load_blank_index, load_model, score_blank_for_image

idx = load_blank_index("data/index/blank_prototypes.pt")
model, processor, device = load_model(model_id=idx.model_id)
cosine01, maxsim = score_blank_for_image(
    page_image, model=model, processor=processor, device=device, blank_index=idx
)
is_blank = cosine01 >= 0.85
```

Các hàm khác trong `classifyrag/blank_page.py`: `blank_scores`, `is_blank_page`, `build_blank_index_from_dir`.

### Gợi ý chỉnh ngưỡng

Nếu quá ít/ nhiều trang bị gán “blank”, hãy chỉnh `--threshold` (thường thử **0.80–0.92**) hoặc bổ sung ảnh mẫu trong `data/blank_data` sát với DPI/nền/scan thực tế.

---

## 3. Module Python chính

| Module | Vai trò |
|--------|---------|
| `classifyrag/colsmol_scorer.py` | Load model, embed ảnh/text, MaxSim, `classify_page` |
| `classifyrag/pdf_pages.py` | Render trang, đọc text / OCR |
| `classifyrag/labels.py` | Ánh xạ tiền tố tên file → nhãn |
| `classifyrag/blank_page.py` | Index & điểm trang trắng |

---

## 4. Output index

- `data/index/prototypes.pt` — phân loại nhóm tài liệu.
- `data/index/blank_prototypes.pt` — phát hiện trang trắng.

Mỗi file lưu `model_id` đã dùng lúc embed; `classify_pdf` / `eval_*` load đúng checkpoint đó. **Đổi mô hình embedding** (ví dụ nâng ColQwen3.5) cần **build lại** cả hai index để vector prototype khớp với model mới.

Có thể tái tạo bằng các lệnh `build_*` ở trên sau khi đổi dữ liệu mẫu.
