# Checklist dự kiến — điều chỉnh theo tiến độ thực tế

Hạn trong đề: 13/12/2026. Mục tiêu nội bộ sẵn sàng nộp 12/12. Các mốc dưới đây là kế hoạch nhóm đề xuất.

## 31/10–03/11: Chọn đề tài
- [ ] Cả nhóm: mỗi người một hướng + dataset + baseline + paper liên quan.
- [ ] B: xem dữ liệu thật, giấy phép, dung lượng, nhãn và chất lượng sơ bộ.
- [ ] C: tổng hợp GPU/VRAM/thời gian; xác định metric và chi phí dự kiến.
- [ ] A: chốt proposal một trang, giả thuyết và phạm vi; điền tên A/B/C.
- [ ] A: xác nhận số trang, giờ/nơi nộp và bảo vệ nếu có.

## 04/11–10/11: Dữ liệu và baseline
- [ ] B: thống kê, kiểm tra trùng/nhãn lỗi/mất cân bằng, tạo split manifest.
- [ ] C: kiểm tra chéo leakage và metric bằng ví dụ biết trước kết quả.
- [ ] B: dataloader, preprocessing fit trên train, baseline và checkpoint theo validation.
- [ ] A+B: kiểm tra một lượt nhỏ trước huấn luyện dài.
- [ ] B: lưu cấu hình, seed, môi trường, phần cứng và hướng dẫn chạy lại.

## 11/11–20/11: Cải tiến
- [ ] Cả nhóm: đọc nghiên cứu gần nhất, phân biệt phần tái sử dụng và đóng góp.
- [ ] A: mô tả/sơ đồ, code cải tiến và cấu hình bật/tắt thành phần.
- [ ] B: giữ dữ liệu/điều kiện đối chứng nhất quán hoặc giải thích khác biệt.
- [ ] A+B: chọn cấu hình trên validation; sửa lỗi/thu hẹp hướng nếu cần.
- [ ] C: chốt matrix.csv, câu hỏi, người chạy và ngân sách từng lượt.

## 21/11–29/11: Thí nghiệm
- [ ] Cả nhóm: chốt config/subset trước khi xem test; chia lượt chạy.
- [ ] B+C: đánh giá mô hình nền và phương pháp đối chứng phù hợp.
- [ ] A+C: đánh giá phương pháp đề xuất và ablation.
- [ ] B+C: kiểm tra nhóm khó/tổng quát hóa có liên quan.
- [ ] C: nhiều seed và độ biến thiên khi khả thi; giải thích phần không thực hiện được.
- [ ] C: đo tài nguyên đúng quy trình nếu có tuyên bố hiệu quả tính toán.
- [ ] Cả nhóm: lưu thất bại/thành công, đánh đổi; kiểm tra số liệu với log.

## 30/11–09/12: Hoàn thiện (viết báo cáo dần từ sớm)
- [ ] A: Introduction, Proposed method; B: Dataset/Training; C: Evaluation/Results.
- [ ] Cả nhóm: Related work, Limitations; A tổng hợp Abstract/Conclusion.
- [ ] C: tích hợp LNCS, hình/bảng, trích dẫn; không sửa lề/cỡ chữ.
- [ ] B: README dữ liệu/môi trường/chạy; A+C bổ sung task/evaluation.
- [ ] Cả nhóm: nguồn tài nguyên và khai báo AI; quyền checkpoint.
- [ ] Người khác tác giả: chạy lại ít nhất một kết quả chính theo README.

## 10/12–12/12: Kiểm tra và nộp
- [ ] Cả nhóm: đối chiếu báo cáo/log/checkpoint, kiểm tra tiếng Anh và references.
- [ ] Cả nhóm: đủ baseline, ablation, phân tích lỗi, giới hạn và hướng dẫn tái lập.
- [ ] A: kiểm tra quyền tải dữ liệu/trọng số, đóng gói đúng yêu cầu.
- [ ] Cả nhóm duyệt bản cuối; A nộp và lưu xác nhận.
