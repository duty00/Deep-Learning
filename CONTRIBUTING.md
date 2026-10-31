# Làm việc nhóm

- Nhận một issue có người phụ trách, đầu ra và tiêu chí hoàn thành rõ ràng.
- Tạo nhánh `feat/...`, `fix/...`, `docs/...` hoặc `experiment/...` từ `main`.
- Commit nhỏ, mô tả việc thay đổi; không đẩy trực tiếp vào `main` sau khi nhóm thiết lập review.
- Mở pull request, dẫn issue, ghi cách kiểm tra và nhờ ít nhất một thành viên khác review.
- Mỗi thí nghiệm ghi cấu hình, seed, commit chạy, môi trường, tập đánh giá và log.
- Không dùng test để chọn cấu hình; không đưa số liệu chưa kiểm chứng vào báo cáo.
- Không commit dataset, checkpoint, secrets; xem `.gitignore` và vẫn kiểm tra `git status`.
- Chạy `python -m unittest discover -s tests -v` trước khi gửi PR.

Chi tiết: [quy trình GitHub](docs/GITHUB.md).
