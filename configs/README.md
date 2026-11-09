# Cấu hình

Đề tài đã chốt: phân loại tổn thương da HAM10000 (xem `PLAN_BTL.md`).
`baseline.json` = E2 (ConvNeXt-T, concat metadata, CE). `proposed.json` = E7 (FiLM stage 2–3, logit-adjusted loss).
Các hyperparameter là điểm khởi đầu theo kế hoạch, sẽ được tinh chỉnh trên validation. Config ablation E1–E12 tạo theo `PLAN_BTL.md` mục 5.10.
Điền task, dữ liệu, mô hình, metric; triển khai và đăng ký task trước khi chạy huấn luyện.
Mọi đường dẫn tương đối trong config được hiểu từ thư mục gốc repo, không phải từ thư mục configs.
`check` kiểm tra cấu trúc và giá trị nền tảng, không chứng minh tính đúng của phép chia dữ liệu hay thí nghiệm.
