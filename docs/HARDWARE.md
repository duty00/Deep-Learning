# Phần cứng: RTX 2050 4 GB (local) và Kaggle

## Quy ước batch (áp dụng từ giai đoạn 3 — trainer)

- `training.batch_size` = **batch hiệu dụng = 32**, giống nhau ở mọi máy ⇒ kết quả local và Kaggle so sánh được.
- `training.micro_batch_size` = số ảnh mỗi lần forward/backward vừa VRAM. Số bước tích lũy
  `accum = batch_size / micro_batch_size` (config bắt buộc chia hết; đã kiểm tra trong `config.py`).
- Config mặc định `micro_batch_size = 8` (accum 4) cho RTX 2050. Trên Kaggle T4/P100 (16 GB) có thể đặt 32 (accum 1)
  mà không đổi thí nghiệm. Ghi giá trị thực tế vào run log.
- Trainer phải: chia loss cho `accum`; `GradScaler` (AMP fp16) + `optimizer.step()` + scheduler chỉ sau mỗi `accum`
  micro-batch; scheduler tính theo **số optimizer step**, không theo số micro-batch; bỏ nhóm micro-batch thiếu ở cuối
  epoch (hoặc chia lại đúng tỉ lệ) để mọi update có cùng batch 32.
- ConvNeXt-T dùng LayerNorm ⇒ tích lũy gradient tương đương batch 32 thật (trừ nhiễu stochastic depth/dropout).
  **EfficientNet-B0 dùng BatchNorm** ⇒ thống kê BN tính trên 8 ảnh, không tương đương batch 32. Vì vậy E10/E11
  (RQ4) nên chạy trên Kaggle với `micro_batch_size = 32`, hoặc ghi rõ khác biệt này trong báo cáo.
- Ước lượng VRAM/thời gian chưa đo; giai đoạn 3 sẽ ghi `torch.cuda.max_memory_allocated()` vào run dir.
  Nếu micro-batch 8 vẫn hết VRAM: giảm 4 (accum 8) hoặc bật gradient checkpointing; không giảm batch hiệu dụng.

## Chạy ở đâu

| Việc | Local RTX 2050 | Kaggle | Ghi chú |
|---|---|---|---|
| Tạo/verify split, unit test, EDA (`notebooks/01_eda.ipynb`) | ✅ | — | Không cần GPU |
| Sanity check: overfit 1 batch, `--debug` 2 batch, kiểm tra shape/loss | ✅ | — | Nhanh, sửa lỗi dễ |
| Một run đầy đủ E2 để đo thời gian/VRAM | ✅ (tùy chọn) | ✅ | Dùng số đo thật để chia ngân sách |
| Ma trận E1–E9, 3 seed (~27 run × tối đa 30 epoch) | ⚠️ quá lâu | ✅ **nên chạy** | Chia run giữa tài khoản Kaggle của 3 người |
| E10–E11 EfficientNet-B0 (BatchNorm) | ⚠️ | ✅ **nên chạy** | `micro_batch_size = 32` |
| E12 (25%/50% train) | ✅ | ✅ | Run ngắn hơn |
| Đánh giá test HAM + PAD-UFES-20, Grad-CAM | ✅ | ✅ | Chỉ inference, nhẹ |
| Đo latency/VRAM cho bảng hiệu năng | ✅ | ✅ | Ghi đúng tên GPU; không trộn số đo hai máy trong một bảng |

Kaggle: một phiên tối đa ~12 giờ, quota GPU theo tuần/tài khoản ⇒ lưu `best.pt`, `last.pt`, `history.csv` vào
`/kaggle/working` và tải về sau mỗi run. Thiết lập dữ liệu trên Kaggle: `scripts/download_data.md` mục 4.

## Môi trường Python local (cần làm trước giai đoạn 2)

Máy hiện chỉ có Python của MSYS2 (`C:\msys64\ucrt64\bin\python.exe`, nền tảng `mingw_x86_64_ucrt`). PyPI không có
wheel PyTorch CUDA cho nền tảng này. Cần Python 3.11 x64 chính thức (python.org hoặc
`winget install -e --id Python.Python.3.11`), rồi tạo `.venv` theo `README.md` và cài PyTorch bản CUDA phù hợp driver
theo hướng dẫn trên pytorch.org. DataLoader trên Windows dùng `spawn`: với `num_workers > 0`, code phải nằm sau
`if __name__ == "__main__"` (đã thêm vào `src/dl_project/__main__.py`).
