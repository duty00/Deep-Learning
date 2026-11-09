# Phân công 3 người (cập nhật 09/11/2026)

A = trưởng nhóm, **làm toàn bộ code và thí nghiệm**. B, C = **viết báo cáo LNCS**. Thay A/B/C bằng tên thật.

| Người | Trách nhiệm chính | Đầu ra |
|---|---|---|
| A | Code (data, model, loss, train, eval), chạy thí nghiệm E1–E12, xuất bảng/hình, README tái lập | Codebase, checkpoint, `experiments/results.csv`, `reports/tables/`, `reports/figures/` |
| B | Báo cáo phần **Introduction, Related work, Proposed method**; quản lý trích dẫn (`.bib`) | Các section tương ứng trên Overleaf |
| C | Báo cáo phần **Dataset, Experimental setup, Results & analysis, Limitations**; định dạng LNCS, ghép hình/bảng | Các section tương ứng, bản PDF cuối |
| Cả nhóm | Abstract, Conclusion, đọc soát số liệu/trích dẫn, khai báo AI | — |

## B và C làm được ngay (không cần chờ kết quả)

| Tuần | B | C |
|---|---|---|
| W1 (09–15/11) | Tạo project Overleaf bằng **template LNCS chính thức**, mời cả nhóm. Đọc 8–12 paper (long-tail loss, metadata fusion, FiLM, skin lesion ISIC, domain shift), lập file `.bib` + bảng tóm tắt: paper làm gì, hạn chế gì | Viết phần **Dataset**: nguồn, license, 7 lớp, mất cân bằng, ảnh trùng lesion, lý do chia theo `lesion_id`; dùng bảng phân phối split ở `data/splits/manifest.json` (mục `summary`) |
| W2 (16–22/11) | Viết **Related work** + nháp **Introduction** (động cơ lâm sàng, hạn chế CE/concat, câu hỏi nghiên cứu RQ1–RQ4) | Viết **Experimental setup** theo `PLAN_BTL.md` mục 5.5–5.9 (tiền xử lý, hyperparameter, metric và hạn chế của metric); vẽ khung các bảng kết quả (cột/hàng, chưa có số) |
| W3 (23–29/11) | Viết **Proposed method**: công thức FiLM, logit adjustment, **vẽ sơ đồ kiến trúc** (draw.io/TikZ); A kiểm tra đúng với code | Nháp **Limitations** (dữ liệu ít trung tâm, test có ít mẫu df/vasc, map nhãn PAD xấp xỉ, không phải công cụ chẩn đoán) |
| W4 (30/11–06/12) | Hoàn thiện Intro + danh sách đóng góp theo kết quả thật | Điền số vào bảng từ `reports/tables/`, viết **Results & analysis** (ablation, per-class, domain shift, ca thất bại) |
| W5 (07–13/12) | Abstract, Conclusion, soát trích dẫn | Ghép PDF, soát định dạng LNCS, khai báo AI; cả nhóm đọc lại lần cuối |

## Quy ước bàn giao A → B/C
- A chỉ bàn giao số đã kiểm tra: mỗi số trong báo cáo phải có dòng tương ứng trong `experiments/results.csv` (run_path + commit).
- Bảng: `reports/tables/*.csv` (+ bản LaTeX). Hình: `reports/figures/*.pdf|png`. B/C **không tự gõ lại số** bằng tay khi đã có file.
- B/C ghi câu hỏi kỹ thuật vào GitHub Issues gắn nhãn `report`; A trả lời trong 1 ngày.
- Tùy chọn: B/C có thể giúp chạy notebook Kaggle bằng tài khoản riêng (chỉ bấm chạy theo hướng dẫn của A, không cần code) để tăng quota GPU.

Cập nhật tiến độ 2 lần/tuần: đã xong gì (kèm link), đang vướng gì, việc kế tiếp và hạn.
