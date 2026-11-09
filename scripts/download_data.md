# Tải dữ liệu vào `data/raw/`

Ảnh và metadata **không** được commit (xem `.gitignore`). Chỉ `data/splits/manifest.json` (danh sách
`image_id → split`, không chứa ảnh) được commit để cả nhóm và Kaggle dùng **cùng một phép chia**.

> [!IMPORTANT]
> Cả nhóm phải dùng **cùng một file** `HAM10000_metadata.csv`. Manifest lưu SHA-256 của file này;
> `python scripts/make_splits.py --verify` sẽ báo lỗi nếu file khác. Nguồn chuẩn của nhóm: **Kaggle**
> (mục 1, cách A), vì cũng chính là dữ liệu dùng khi huấn luyện trên Kaggle.

## Cấu trúc thư mục cần có

```text
data/raw/
  ham10000/
    HAM10000_metadata.csv          # lesion_id, image_id, dx, dx_type, age, sex, localization
    HAM10000_images_part_1/        # ISIC_xxxxxxx.jpg
    HAM10000_images_part_2/        # tổng 2 thư mục = 10.015 ảnh
  pad_ufes_20/
    metadata.csv                   # patient_id, lesion_id, ..., diagnostic, ..., img_id, biopsed
    images/                        # PAT_xxx_yyy_zzz.png (gộp imgs_part_1..3)
```

Giữ nguyên tên thư mục gốc của HAM10000 (giống Kaggle) để cùng một code chạy được cả local và Kaggle.
Các file khác trong bản tải (vd. `hmnist_*.csv` của Kaggle) không dùng, có thể xóa.

## 1. HAM10000 (dữ liệu chính: train/val/test)

- Tác giả: Tschandl, Rosendahl, Kittler (2018), *Scientific Data* 5:180161, doi:10.1038/sdata.2018.161.
- Bản gốc: Harvard Dataverse, doi:10.7910/DVN/DBW86T. Giấy phép **CC BY-NC 4.0** (phi thương mại, phải trích dẫn).
- Kỳ vọng: 10.015 ảnh, 7 lớp `akiec, bcc, bkl, df, mel, nv, vasc`. Số liệu chi tiết phải lấy từ EDA, không chép từ đây.

**Cách A — Kaggle (khuyến nghị):** dataset `kmader/skin-cancer-mnist-ham10000` (~5–6 GB khi giải nén).

```powershell
cd D:\DL
.\.venv\Scripts\python.exe -m pip install kaggle
# Tạo API token tại kaggle.com > Settings > API, lưu vào %USERPROFILE%\.kaggle\kaggle.json (KHÔNG commit)
.\.venv\Scripts\kaggle.exe datasets download -d kmader/skin-cancer-mnist-ham10000 -p data\raw\ham10000 --unzip
```

Hoặc tải zip bằng trình duyệt từ trang dataset rồi giải nén vào `data\raw\ham10000\`. Nếu bản giải nén có thêm
thư mục trùng tên viết thường (`ham10000_images_part_1`), đó là bản sao; giữ bản viết hoa, xóa bản còn lại.

**Cách B — Harvard Dataverse:** tải `HAM10000_images_part_1.zip`, `HAM10000_images_part_2.zip` và
`HAM10000_metadata` (chọn *Original File Format* để được CSV; nếu nhận được `.tab` thì vẫn đọc được bằng
`--metadata`). Bản Dataverse có thể khác Kaggle theo từng byte ⇒ SHA-256 khác ⇒ chỉ dùng nếu **cả nhóm** cùng dùng
và tạo lại manifest một lần duy nhất.

## 2. PAD-UFES-20 (chỉ test ngoài — domain shift)

- Tác giả: Pacheco et al. (2020), *Data in Brief* 32:106221. Mendeley Data, doi:10.17632/zr7vgbcyr2.1.
- Giấy phép: **CC BY 4.0** (nhóm xác nhận lại trên trang Mendeley khi tải và ghi vào `docs/ATTRIBUTION.md`).
- Ảnh chụp bằng điện thoại (không phải dermoscopy), 6 chẩn đoán `ACK, BCC, MEL, NEV, SCC, SEK`. Số ảnh/bệnh nhân/
  lesion: xác nhận bằng EDA (tài liệu gốc ghi ~2.298 ảnh).
- **Không** dùng để train hay chọn hyperparameter. Map nhãn (xấp xỉ, giai đoạn sau): MEL→mel, BCC→bcc, NEV→nv,
  SEK→bkl, ACK/SCC→akiec.

Tải zip "Download All" từ trang Mendeley, giải nén; bên trong có `metadata.csv` và các zip ảnh
`imgs_part_1/2/3`. Gộp toàn bộ ảnh `.png` vào `data\raw\pad_ufes_20\images\`:

```powershell
cd D:\DL
New-Item -ItemType Directory -Force data\raw\pad_ufes_20\images | Out-Null
# Giả sử đã giải nén bản tải về vào D:\Downloads\pad_ufes_20 (sửa đường dẫn cho đúng)
Copy-Item D:\Downloads\pad_ufes_20\metadata.csv data\raw\pad_ufes_20\
Get-ChildItem D:\Downloads\pad_ufes_20 -Recurse -Filter imgs_part_*.zip |
  ForEach-Object { Expand-Archive $_.FullName -DestinationPath data\raw\pad_ufes_20\_tmp -Force }
Get-ChildItem data\raw\pad_ufes_20\_tmp -Recurse -Filter *.png | Move-Item -Destination data\raw\pad_ufes_20\images
Remove-Item data\raw\pad_ufes_20\_tmp -Recurse
```

## 3. Kiểm tra sau khi tải

```powershell
cd D:\DL
(Get-ChildItem data\raw\ham10000\HAM10000_images_part_*\*.jpg).Count     # kỳ vọng 10015
(Get-ChildItem data\raw\pad_ufes_20\images\*.png).Count                 # đối chiếu với metadata.csv
Get-FileHash data\raw\ham10000\HAM10000_metadata.csv -Algorithm SHA256   # so với manifest["source"]["sha256"]
```

Tạo split (chỉ **một lần**, người phụ trách B; sau đó commit manifest):

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[data]"
.\.venv\Scripts\python.exe scripts\make_splits.py            # ghi data/splits/manifest.json
.\.venv\Scripts\python.exe scripts\make_splits.py --verify   # các thành viên khác chạy lệnh này
git add data/splits/manifest.json
```

`make_splits.py` từ chối ghi đè manifest đã có (phải thêm `--force`). Không tạo lại split sau khi đã xem kết quả test.

## 4. Trên Kaggle

1. Notebook mới → *Add Data*: `kmader/skin-cancer-mnist-ham10000`. PAD-UFES-20: tải lên thành **dataset riêng tư**
   của nhóm từ bản Mendeley (CC BY 4.0 cho phép, ghi nguồn).
2. Bật GPU (T4 ×2 hoặc P100) và Internet (để clone repo/cài gói).
3. Trong notebook (thư mục `/kaggle/input` chỉ đọc, nên dùng symlink thay vì copy):

```bash
!git clone <URL_REPOSITORY_CUA_NHOM> /kaggle/working/DL
%cd /kaggle/working/DL
!pip install -q -e ".[data]"
!mkdir -p data/raw && ln -s /kaggle/input/skin-cancer-mnist-ham10000 data/raw/ham10000
!ln -s /kaggle/input/<ten-dataset-pad-ufes-20> data/raw/pad_ufes_20
!python scripts/make_splits.py --verify     # bắt buộc: xác nhận đúng metadata + manifest đã commit
```

Kiểm tra lại tên thư mục thật trong `/kaggle/input/...` (Kaggle có thể lồng thêm một cấp thư mục).
Xem `docs/HARDWARE.md` để biết bước nào chạy local, bước nào chạy Kaggle.
