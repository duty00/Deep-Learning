# Cách phát triển bộ khung

## 1. Chốt và điền cấu hình
Điền proposal và configs. Không coi các giá trị mẫu là hyperparameter đã được chọn.
Metric, loss, optimizer/schedule, pretrained weights, frozen layers, stopping rule và preprocessing cụ thể
phải được bổ sung theo đề tài; chưa có trong skeleton.

## 2. Triển khai các phần
B thêm data loader/split/preprocessing trong `data/` và baseline trong `models/`.
A thêm cải tiến trong `models/`; cả nhóm hoàn thiện engine và nối task.
C thêm metric/evaluation; mọi phương pháp dùng chung quy trình đánh giá.
Phần xử lý random seed của Python/NumPy/PyTorch, GPU, worker và determinism phải được triển khai theo runtime.
Lưu seed trong config không tự bảo đảm tái lập.

## 3. Đăng ký nhiệm vụ
Tạo class kế thừa `Task` trong `src/dl_project/tasks/<ten_bai_toan>.py`, triển khai
`train(config, project_root, run_dir)` và `evaluate(...)`, rồi đưa class vào `TASKS` trong `tasks/__init__.py`.
Truy cập các đường dẫn tương đối bằng `project_root / ...`.
Train chỉ dùng train/validation; checkpoint chọn bằng validation. Evaluate tải `model.checkpoint`.
Tự ghi log, checkpoint và metric thực tế; bộ khung chỉ lưu config/metadata/environment.

## 4. Chạy từ thư mục repo
```powershell
python -m dl_project check --config configs/baseline.json
python -m dl_project train --config configs/baseline.json
python -m dl_project evaluate --config configs/baseline.json
```
Hai lệnh sau cố ý báo chưa cấu hình cho đến khi task được triển khai và config điền đủ.
Trước evaluate, đặt `model.checkpoint` vào file đã huấn luyện và chọn validation/test theo mục đích.
Không dùng kết quả test để quay lại chọn cấu hình.

## 5. Lưu bằng chứng
CLI tạo `outputs/<experiment>/<mode>/<UTC-id>/` gồm cấu hình, commit và danh sách thư viện.
Thêm log phần cứng, dữ liệu/manifest, thay đổi chưa commit và kết quả bằng RUN_TEMPLATE.
Ghi chỉ số đã xác minh vào results.csv; không có số liệu mặc định.
CI chỉ kiểm tra nền tảng Python/config, chưa kiểm chứng mô hình hoặc giả thuyết nghiên cứu.
