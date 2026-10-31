# DL — Deep Learning Group Project

Bộ khung làm việc cho nhóm 3 người, môn CO3133 — Deep Learning and Applications.
**Đề tài, dữ liệu và mô hình chưa được chốt.** Chưa có kết quả huấn luyện hoặc đóng góp nghiên cứu được xác nhận.

## Bắt đầu

Yêu cầu Python 3.10 trở lên và Git. Chạy trong PowerShell:

```powershell
cd D:\DL
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m dl_project check --config configs/baseline.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Lệnh `check` kiểm tra cấu hình và báo các quyết định còn thiếu; **không huấn luyện mô hình**.
Để cài thư viện cho huấn luyện sau khi nhóm chọn hướng:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[train]"
```

Chọn bản PyTorch phù hợp GPU/driver trước khi chạy thật. Dependencies hiện là danh sách khởi đầu,
chưa phải môi trường tái lập cuối cùng; lưu bản freeze cho từng thí nghiệm.

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

- Đã có: kiểm tra cấu hình, CLI, lưu metadata lượt chạy, mẫu nhiệm vụ và kiểm tra tự động.
- Cần làm: chọn đề tài, triển khai dữ liệu/mô hình/metric/vòng lặp và đăng ký nhiệm vụ.
- `train`/`evaluate` sẽ dừng với thông báo rõ ràng đến khi nhiệm vụ được triển khai.
- Không có dữ liệu, checkpoint hoặc số liệu giả trong bộ khung.

## Nguyên tắc

Validation dùng để chọn cấu hình; test chỉ dùng đánh giá cuối. Lưu cấu hình, seed, môi trường,
phần cứng và log. Mỗi thành phần đề xuất cần thí nghiệm đối chứng/ablation phù hợp.
Không commit dữ liệu nhạy cảm, token, dữ liệu lớn hoặc checkpoint; cung cấp hướng dẫn truy cập đúng giấy phép.
Ghi nguồn code tái sử dụng. Chưa chọn giấy phép phát hành cho code nhóm: ghi quyết định ở `docs/ATTRIBUTION.md`.
