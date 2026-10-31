# Phân công 3 người

Thay tên trước khi bắt đầu: A = bạn/trưởng nhóm; B = thành viên 2; C = thành viên 3.

| Người | Trách nhiệm chính | Đầu ra |
|---|---|---|
| A | Điều phối; câu hỏi nghiên cứu; triển khai cải tiến | Proposal, proposed model/task, config ablation, phần phương pháp |
| B | Dữ liệu; chia tập; tiền xử lý; baseline | Thống kê, manifest, dataloader, baseline, checkpoint, phần dữ liệu/huấn luyện |
| C | Metric; đánh giá; ma trận thí nghiệm; tổng hợp LNCS | Evaluation, bảng kết quả, phân tích lỗi, tích hợp báo cáo |

Mỗi người đọc paper và viết phần mình làm. Cả ba cùng chạy các lượt thí nghiệm được chia trong matrix.csv.
C không gánh toàn bộ huấn luyện hoặc viết báo cáo. Mỗi issue có đúng một người chịu trách nhiệm chính.

Kiểm tra chéo: C kiểm tra chia tập của B; B kiểm tra triển khai/thí nghiệm của A;
A kiểm tra metric/bảng kết quả của C. Không coi review code là bằng chứng phương pháp hiệu quả.

Mỗi tuần hai lần cập nhật: đã xong gì (kèm link), đang vướng gì, việc kế tiếp và hạn.
