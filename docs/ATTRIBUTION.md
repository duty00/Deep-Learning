# Nguồn và giấy phép

| Tài nguyên | Nguồn/phiên bản | Giấy phép/điều kiện | Dùng ở đâu | Phần nhóm sửa |
|---|---|---|---|---|
| HAM10000 | Tschandl et al. 2018, Harvard Dataverse doi:10.7910/DVN/DBW86T (nhóm tải qua Kaggle `kmader/skin-cancer-mnist-ham10000`) | CC BY-NC 4.0, trích dẫn bài Scientific Data | Train/val/test | Không sửa; chỉ chia split (`data/splits/manifest.json`) |
| PAD-UFES-20 | Pacheco et al. 2020, Mendeley Data doi:10.17632/zr7vgbcyr2.1 | CC BY 4.0 (xác nhận khi tải) | Chỉ test ngoài | Map nhãn xấp xỉ (giai đoạn sau) |
| scikit-learn | `StratifiedGroupKFold` | BSD-3-Clause | `src/dl_project/data/splits.py` | Gọi thư viện, không sao chép code |

Model/code bên ngoài (timm, grad-cam...) chưa dùng; điền nguồn trước khi tái sử dụng.
Nhóm chưa chọn giấy phép cho mã nguồn tự viết; quyết định trước khi phát hành công khai.
Khai báo AI hỗ trợ ngôn ngữ trong báo cáo nếu có, phù hợp hướng dẫn môn.
