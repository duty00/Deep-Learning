# DL — Phân loại tổn thương da HAM10000 (CO3133)

Đề tài (đã chốt): **Metadata-Conditioned Feature Modulation and Prior-Aware Training for Long-Tailed Skin
Lesion Classification under Domain Shift**. ConvNeXt-T + FiLM theo metadata (tuổi, giới, vị trí) + logit-adjusted
loss trên HAM10000 (chia theo `lesion_id`), test domain shift trên PAD-UFES-20. Kế hoạch: [PLAN_BTL.md](PLAN_BTL.md).
Chưa có kết quả thí nghiệm chính thức.

## Cài đặt (Windows, PowerShell)

```powershell
cd D:\DL
python -m venv .venv                      # Python 3.11 x64 chính thức (không dùng Python MSYS2)
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[train]"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Cài torch CUDA **trước** `.[train]`; nếu làm ngược lại pip sẽ kéo bản torch CPU từ PyPI.

## Dữ liệu

Tải theo [scripts/download_data.md](scripts/download_data.md) vào `data/raw/ham10000/`, rồi xác nhận phép chia đã commit:

```powershell
.\.venv\Scripts\python.exe scripts\make_splits.py --verify
```

EDA: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) (hình lưu ở `reports/figures/`).

## Chạy thí nghiệm

```powershell
# Kiểm tra nhanh toàn bộ pipeline (2 epoch x 8 batch), kết quả không có ý nghĩa
.\.venv\Scripts\python.exe -m dl_project train --config configs/experiments/e07_film_la.json --debug
# Huấn luyện 1 seed (checkpoint chọn theo val macro-F1)
.\.venv\Scripts\python.exe -m dl_project train --config configs/experiments/e07_film_la.json --seed 42
# Đánh giá trên validation (test chỉ dùng cho cấu hình đã chốt)
.\.venv\Scripts\python.exe -m dl_project evaluate --config configs/experiments/e07_film_la.json --seed 42 `
    --checkpoint outputs/e07_film_la-s42/train/<run-id>/best.pt --split validation
# Gom mean ± std qua seed -> reports/tables/, experiments/results.csv
.\.venv\Scripts\python.exe scriptsggregate_results.py --split validation
```

- Ma trận thí nghiệm: [experiments/matrix.csv](experiments/matrix.csv); config sinh bằng `scripts/make_configs.py`.
- Chạy hàng loạt trên Kaggle: [notebooks/kaggle_runner.ipynb](notebooks/kaggle_runner.ipynb) (`--micro-batch-size 32`).
- Máy RTX 2050 4 GB: batch hiệu dụng 32 = micro-batch 8 × tích lũy 4, AMP. Xem [docs/HARDWARE.md](docs/HARDWARE.md).
- Mỗi lượt chạy ghi vào `outputs/<experiment>/<train|evaluate>/<UTC-id>/`: config, commit, môi trường, `train.log`,
  `history.csv`, `best.pt`, `metrics.json`, `predictions.csv`.

## Đọc theo thứ tự

1. [Tổng quan đề bài](docs/ASSIGNMENT.md).
2. [Phân công 3 người](docs/TEAM.md) — thay A/B/C bằng tên thành viên.
3. [Checklist và kế hoạch](docs/CHECKLIST.md).
4. [Đề xuất đề tài](docs/PROPOSAL.md) — cùng điền trước khi huấn luyện dài.
5. [Thiết kế kỹ thuật và cách thêm bài toán](docs/DEVELOPMENT.md).
6. [Quy trình GitHub](docs/GITHUB.md).

## Cấu trúc

```text
configs/                 Cấu hình baseline và phương pháp đề xuất
src/dl_project/
  data/                  Nạp dữ liệu, chia tập, tiền xử lý
  models/                Baseline và cải tiến liên quan đến mô hình
  tasks/                 Điểm kết nối bài toán với lệnh train/evaluate
  engine/                Vòng lặp huấn luyện/đánh giá dùng chung
  metrics/               Metric và tổng hợp kết quả
scripts/                 Lệnh gọi train/evaluate
tests/                   Kiểm tra cấu hình và nền tảng dùng chung
data/{raw,interim,processed,splits}/  Dữ liệu cục bộ; không đưa lên Git
checkpoints/             Trọng số cục bộ; không đưa lên Git
outputs/                 Kết quả chạy cục bộ; không đưa lên Git
experiments/             Nhật ký, ma trận và bảng kết quả đã kiểm tra
notebooks/               Khám phá dữ liệu, phân tích lỗi
reports/                 Hướng dẫn LNCS và hình/bảng được chọn cho báo cáo
docs/                    Hướng dẫn, phân công, checklist, đề xuất
.github/                 Mẫu issue/PR và kiểm tra tự động
```

## Trạng thái hiện tại

- Đã có: split theo lesion + manifest, Dataset/metadata encoder, model B0/B1/FiLM (ConvNeXt-T, EfficientNet-B0),
  4 loss (CE, weighted CE, focal, logit-adjusted), trainer (AMP, tích lũy gradient, early stopping theo val),
  metric (macro-F1, rare-recall, AUROC, ECE, bootstrap CI), 17 config thí nghiệm, notebook EDA và Kaggle, test tự động.
- Cần làm: chạy E1–E12 × 3 seed, đánh giá PAD-UFES-20 (chưa tải), Grad-CAM, đo latency, phân tích lỗi.
- Không có checkpoint hoặc số liệu giả trong repo.

## Nguyên tắc

Validation dùng để chọn cấu hình; test chỉ dùng đánh giá cuối. Lưu cấu hình, seed, môi trường,
phần cứng và log. Mỗi thành phần đề xuất cần thí nghiệm đối chứng/ablation phù hợp.
Không commit dữ liệu nhạy cảm, token, dữ liệu lớn hoặc checkpoint; cung cấp hướng dẫn truy cập đúng giấy phép.
Ghi nguồn code tái sử dụng. Chưa chọn giấy phép phát hành cho code nhóm: ghi quyết định ở `docs/ATTRIBUTION.md`.
