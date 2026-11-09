# Đề xuất nghiên cứu — đã chốt hướng (09/11/2026)

> Chi tiết triển khai: [PLAN_BTL.md](../PLAN_BTL.md) mục 5. Các mục ghi "Cần nhóm xác nhận" phải được kiểm tra trên dữ liệu thật trước khi đưa vào báo cáo.

- **Tên đề tài (làm việc):** Metadata-Conditioned Feature Modulation and Prior-Aware Training for Long-Tailed Skin Lesion Classification under Domain Shift.
- **Thành viên A/B/C:** TBD (A: phương pháp đề xuất · B: data + baseline · C: metric, đánh giá, domain shift).
- **Bài toán:** Đầu vào = ảnh dermoscopy + metadata (tuổi, giới, vị trí tổn thương). Đầu ra = 1 trong 7 lớp (akiec, bcc, bkl, df, mel, nv, vasc). Tình huống sử dụng: hỗ trợ sàng lọc, ưu tiên không bỏ sót lớp hiếm/nguy hiểm. Không phải công cụ chẩn đoán.
- **Dataset:**
  - HAM10000 (Harvard Dataverse / ISIC), CC BY-NC 4.0, 10.015 ảnh, 7 lớp, lệch mạnh về `nv`. Số mẫu từng lớp, số lesion, tỉ lệ thiếu tuổi: *cần nhóm xác nhận khi chạy EDA*.
  - PAD-UFES-20 (Mendeley Data), ảnh điện thoại: **chỉ dùng làm test ngoài**, sau khi map nhãn về lớp chung (MEL→mel, BCC→bcc, NEV→nv, SEK→bkl, ACK/SCC→akiec; đây là map xấp xỉ). Giấy phép và số mẫu: *cần nhóm xác nhận*.
- **Chất lượng dữ liệu & chống leakage:** Nhiều ảnh cùng `lesion_id` → chia train/val/test 70/15/15 bằng StratifiedGroupKFold theo `lesion_id`, stratify theo `dx`. Kiểm tra trùng ảnh bằng hash. Prior lớp, class weight, thống kê chuẩn hóa tuổi chỉ tính trên train. Test chỉ dùng đánh giá cuối.
- **Baseline:** ConvNeXt-Tiny pretrained ImageNet (timm, Apache-2.0) fine-tune bằng CE: (B0) chỉ ảnh, (B1) concat metadata (baseline chính). So sánh thêm weighted CE và focal loss. Backbone phụ: EfficientNet-B0.
- **Nghiên cứu liên quan & hạn chế:** Fine-tune bằng CE cho accuracy cao nhờ lớp `nv` nhưng recall lớp hiếm thấp. Metadata thường được ghép (concat) ở cuối mạng nên đóng góp ít. Hiệu năng sụt khi chuyển từ dermoscopy sang ảnh điện thoại. *Danh sách paper cụ thể: nhóm bổ sung ở tuần 1.*
- **Cải tiến dự kiến:** (1) FiLM: metadata sinh (γ, β) để điều biến feature ở stage 2–3 của backbone, khởi tạo identity. (2) Logit-adjusted loss với prior lớp tính từ train. (3) Tùy chọn: augmentation mô phỏng ảnh điện thoại. Khác biệt cần định vị so với các paper fusion metadata đã có: *nhóm xác minh tính mới khi đọc paper*.
- **Giả thuyết:** Kết hợp FiLM-metadata và logit-adjusted loss tăng macro-F1 và rare-recall so với B1, với cùng ngân sách huấn luyện, trên test HAM10000 (chia theo lesion). Mức sụt macro-F1 khi chuyển sang PAD-UFES-20 cũng nhỏ hơn.
- **Metric:** Chính = macro-F1. Phụ = balanced accuracy, recall/F1 từng lớp, rare-recall (trung bình recall của mel, bcc, akiec, df, vasc; định nghĩa trước khi xem test), AUROC mel-vs-rest, ECE, cùng params/latency/VRAM. Hạn chế: macro-F1 không phân biệt mức nguy hiểm của từng loại lỗi; test set có ít mẫu lớp hiếm nên phương sai cao; map nhãn PAD chỉ xấp xỉ.
- **So sánh & ablation:** Ma trận E1–E12 (PLAN_BTL mục 5.10), gồm loại fusion, loại loss, augmentation domain, số stage FiLM, backbone và kích thước tập train. Thí nghiệm chính chạy 3 seed (42, 43, 44), báo mean ± std.
- **Tài nguyên:** Kaggle/Colab T4 hoặc P100. Ước tính 20–40 phút/run, khoảng 35–40 run, tức khoảng 20h GPU.
- **Phạm vi tối thiểu / phương án thu hẹp:** Tối thiểu là E1, E2, E5, E6, E7 (3 seed) và đánh giá trên PAD-UFES-20. Nếu thiếu thời gian: bỏ E10–E12 và augmentation domain. Nếu phương pháp không cải thiện: báo cáo trung thực kèm ablation và phân tích lỗi.
- **Câu hỏi cho giảng viên:** Map nhãn PAD-UFES-20 (gộp SCC vào akiec) có chấp nhận được không? Mức đóng góp FiLM + logit adjustment đã đủ chưa, hay nên thêm thành phần khác?
