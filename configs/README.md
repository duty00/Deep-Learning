# Cấu hình

Hai file là mẫu cấu trúc; batch size, learning rate và epoch chỉ là giá trị giữ chỗ, chưa được chọn theo dữ liệu.
`proposed.json` chưa bật cải tiến nào. Điền `components` sau khi chốt phương pháp, tạo thêm config ablation.
Điền task, dữ liệu, mô hình, metric; triển khai và đăng ký task trước khi chạy huấn luyện.
Mọi đường dẫn tương đối trong config được hiểu từ thư mục gốc repo, không phải từ thư mục configs.
`check` kiểm tra cấu trúc và giá trị nền tảng, không chứng minh tính đúng của phép chia dữ liệu hay thí nghiệm.
