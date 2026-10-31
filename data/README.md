# Dữ liệu — B phụ trách

- `raw/`: bản gốc, không sửa trực tiếp.
- `interim/`: dữ liệu trung gian.
- `processed/`: dữ liệu đã xử lý.
- `splits/`: manifest chia train/validation/test, seed và nhóm chia.

Các thư mục trên được loại khỏi Git. Không mặc định đẩy dữ liệu lên GitHub.
Ghi nguồn, phiên bản, điều kiện sử dụng, cách tải và checksum/định danh dữ liệu ở đây khi chốt đề tài.
Nếu cần chia sẻ manifest nhỏ, đã kiểm tra giấy phép/quyền riêng tư, đặt bản thích hợp trong `experiments/`.
Không chia các phiên bản cùng mẫu hoặc cùng nhóm liên quan sang nhiều tập.
