# Yêu cầu sửa code và bổ sung — giao cho Codex

Tài liệu này liệt kê các việc cần làm trong repo `D:\DL` (GitHub `duty00/Deep-Learning`, branch `main`) sau lần
review code ngày 12/11/2026. Đọc **toàn bộ mục 0** trước khi sửa bất cứ thứ gì. Mỗi task có: lý do, việc cần làm,
file liên quan, và tiêu chí nghiệm thu. Làm theo thứ tự ưu tiên P0 → P1 → P2; mỗi task một commit riêng.

Bối cảnh: bài tập lớn CO3133 (Deep Learning), phân loại 7 lớp tổn thương da HAM10000, ConvNeXt-Tiny / EfficientNet-B0
(timm), metadata fusion (none / concat / FiLM), 4 loss (CE, weighted CE, focal, logit-adjusted). Kế hoạch đầy đủ:
`PLAN_BTL.md`. Đã chạy 15 thí nghiệm × 3 seed (45 run) trên Kaggle T4, chỉ chấm **validation**; kết quả ở
`reports/tables/results_validation.csv`, artefact ở `outputs/` (không có trên Git, ~8 GB, chỉ có ở máy nhóm).

---

## 0. Quy tắc bắt buộc (vi phạm = làm hỏng tính trung thực của bài)

1. **Không chạy evaluate trên tập test** (`--split test`) dưới bất kỳ hình thức nào, kể cả để thử. Test chỉ được
   chấm một lần bởi nhóm sau khi chốt cấu hình (task P1-6 chỉ *viết* script, không chạy).
2. **Không sửa** `data/splits/manifest.json`, không tạo lại split. `python scripts/make_splits.py --verify` phải
   tiếp tục báo OK.
3. **Không thay đổi hành vi huấn luyện của các config hiện có** (dữ liệu vào, augmentation, loss, optimizer,
   scheduler, số batch mỗi epoch). 45 run đã có phải vẫn tái lập được bằng code mới. Mọi hành vi mới phải là
   tuỳ chọn (mặc định tắt) hoặc chỉ áp dụng cho config mới.
4. **Không sửa/xoá** bất cứ thứ gì trong `outputs/` và không commit `outputs/`, dữ liệu, checkpoint.
5. Không bịa số liệu. Không chỉnh tay `reports/tables/*` — các bảng chỉ được sinh bằng script.
6. Mọi test hiện có (46 test: `python -m unittest discover -s tests -v`) phải pass; thêm test cho code mới.
   CI (`.github/workflows/ci.yml`) chỉ cài `.[data]` (không có torch) — test cần torch phải tự skip khi thiếu torch
   như các test hiện có.
7. Commit message ngắn gọn, **không thêm dòng `Co-Authored-By` hay bất kỳ dòng ghi công AI nào**.
8. Giữ phong cách code hiện có: import torch/timm lazily trong module không cần torch, docstring nêu lý do,
   không thêm dependency ngoài những gì đã có trong `pyproject.toml` (riêng `grad-cam` đã có trong `[train]`).

Kiểm tra cuối cùng sau mọi task:

```bash
python -m unittest discover -s tests -v
python scripts/make_splits.py --verify
python scripts/aggregate_results.py --split validation   # trên máy có outputs/: bảng phải giống hệt bản đã commit
git diff --stat reports/tables/results_validation.csv     # phải rỗng
```

---

## P0 — Sửa ngay (không làm thay đổi kết quả đã train)

### P0-1. Ghi cứng tên trọng số pretrained vào config

**Lý do.** `timm.create_model("convnext_tiny", pretrained=True)` tải tag mặc định `convnext_tiny.in12k_ft_in1k`
(pretrain ImageNet-12k rồi fine-tune 1k), còn `efficientnet_b0` tải `efficientnet_b0.ra_in1k` (ImageNet-1k). Hai
backbone **không** cùng nguồn pretrain; báo cáo phải ghi đúng, và một bản timm khác có thể đổi tag mặc định, làm
run mới không tái lập được.

**Việc cần làm.**
- `src/dl_project/models/backbones.py`: `StagedBackbone(name, pretrained, pretrained_tag=None, ...)`. Nếu có tag,
  gọi `timm.create_model(f"{name}.{pretrained_tag}", ...)`. Lưu `self.pretrained_tag` = tag thực sự được dùng
  (lấy từ `self.model.pretrained_cfg["tag"]` nếu không truyền).
- `src/dl_project/models/classifier.py` (`build_model`): đọc `config["model"].get("pretrained_tag")`.
- `scripts/make_configs.py` + `configs/baseline.json` + `configs/proposed.json`: thêm `"pretrained_tag"` —
  `"in12k_ft_in1k"` cho `convnext_tiny`, `"ra_in1k"` cho `efficientnet_b0`. Chạy lại `make_configs.py` để sinh
  lại `configs/experiments/*.json` (chỉ được thêm khoá này, `git diff` không được đổi giá trị nào khác).
- `src/dl_project/tasks/skin_lesion.py`: ghi tag thực tế vào dòng log `model: ...` và vào `train_summary.json`
  (`"pretrained_tag"`).
- Ghi vào `docs/ATTRIBUTION.md`: tên đầy đủ hai bộ trọng số và nguồn (HF Hub `timm/convnext_tiny.in12k_ft_in1k`,
  `timm/efficientnet_b0.ra_in1k`), giấy phép.

**Nghiệm thu.** Test mới: build model với `pretrained=False` + tag hợp lệ không lỗi; tag sai báo lỗi rõ ràng.
Config sinh lại chỉ khác ở khoá `pretrained_tag`. Kiểm tra với timm 1.0.30 (bản đã dùng trên Kaggle) rằng tag mặc
định đúng là hai tag trên — ghi kết quả kiểm tra vào commit message.

### P0-2. Chặn việc vô tình chấm tập test

**Lý do.** Mọi config có `"evaluation": {"split": "test"}`, nên `python -m dl_project evaluate ...` thiếu
`--split` sẽ chấm **test**. Đề cấm tuning trên test; một lần chấm nhầm cũng phải khai báo.

**Việc cần làm.**
- `src/dl_project/cli.py`: với lệnh `evaluate`, **bắt buộc** có `--split` (báo lỗi rõ nếu thiếu, không dùng giá
  trị trong config). `--split test` chỉ chạy khi có thêm cờ `--final-test`; thiếu cờ → lỗi với thông báo giải thích.
- Khi chấm test thành công, nối một dòng vào `experiments/test_log.csv` (tạo nếu chưa có):
  `utc_time, experiment, seed, checkpoint, git_commit, run_dir`. Nếu cùng checkpoint đã có dòng trong log thì từ
  chối chạy lại trừ khi có `--allow-repeat-test` (và in cảnh báo phải khai báo trong báo cáo).
- `configs/baseline.json`, `make_configs.py`: đổi mặc định `evaluation.split` thành `"validation"`.
- Cập nhật `notebooks/kaggle_runner.ipynb` (ô chạy) và README để truyền đúng cờ.

**Nghiệm thu.** Test: `evaluate` thiếu `--split` → exit code 2; `--split test` thiếu `--final-test` → exit code 2;
validation chạy như cũ. Không chạy test thật.

### P0-3. Truy vết commit của 6 run đầu

**Lý do.** 6 run E2/E7 (`outputs/e02_concat_ce-s4*`, `outputs/e07_film_la-s4*`) ghi `git_commit`
`0b6f28af1e2cf7ef63ca4a7d065b4a258ad5caae`. Commit này đã bị viết lại (chỉ đổi commit message, cây file giữ
nguyên) thành `40cb77ca4dd09cb654dd857dd368989e2c722b94` và không còn tồn tại trên GitHub. Tương tự
`114c1f4…` → `08818be…`.

**Việc cần làm.**
- Tạo `experiments/COMMIT_MAP.csv`: `recorded_commit,current_commit,note` với hai dòng trên
  (note: "message-only rewrite; identical tree").
- `scripts/aggregate_results.py`: khi ghi `experiments/results.csv`, thêm cột `git_commit_current` (tra theo
  COMMIT_MAP, mặc định bằng `git_commit`). Không đổi cột cũ.
- Một đoạn ngắn trong README mục "Tái lập" giải thích việc này.

**Nghiệm thu.** `results.csv` có cột mới; mọi giá trị `git_commit_current` là commit đang tồn tại
(`git cat-file -e <sha>` thành công).

### P0-4. Không tính trùng run khi tổng hợp

**Lý do.** `scripts/aggregate_results.py` lấy mọi `outputs/*/evaluate/*/metrics.json`. Nếu một seed được evaluate
hai lần (ví dụ chạy lại), seed đó bị tính hai lần vào mean ± std.

**Việc cần làm.**
- Nhóm theo `(experiment, seed, split)`; nếu có nhiều run, giữ run mới nhất (theo `created_utc` trong
  `metadata.json`) và in cảnh báo liệt kê các run bị bỏ.
- Chỉ nhận evaluate run có checkpoint nằm trong `outputs/<experiment>-s<seed>/train/` (bỏ và cảnh báo nếu không).
- Cột `n_seeds` phải là số seed khác nhau.
- `scripts/analyze_results.py` đã chọn run mới nhất theo tên thư mục — đổi sang cùng quy tắc (`created_utc`) cho
  nhất quán, và tách hàm chọn run dùng chung (ví dụ `src/dl_project/reporting.py`) để hai script không lệch nhau.

**Nghiệm thu.** Test với thư mục `outputs` giả (tempdir) có seed bị evaluate 2 lần → n_seeds đúng, chỉ tính 1 lần.
Trên dữ liệu thật: bảng validation không đổi.

### P0-5. Khoảng tin cậy bootstrap theo lesion trong `evaluate`

**Lý do.** `bootstrap_ci` (`src/dl_project/metrics/classification.py`) lấy mẫu lại theo **ảnh**, nhưng nhiều ảnh
cùng một lesion tương quan với nhau → CI hẹp hơn thực tế. `scripts/analyze_results.py` đã làm đúng (theo lesion).

**Việc cần làm.**
- `bootstrap_ci(..., groups=None)`: nếu có `groups` (mảng nhóm cho từng mẫu), lấy mẫu lại theo nhóm.
  Giữ hành vi cũ khi `groups=None`.
- `SkinLesionTask.evaluate`: truyền `lesion_id` của từng ảnh; ghi `macro_f1_ci95` (theo lesion) và thêm
  `"ci_resampling": "lesion"` vào `metrics.json`. Thêm CI cho `balanced_accuracy` và `rare_recall`.
- Script `scripts/recompute_ci.py`: tính lại CI theo lesion cho các run đã có từ `predictions.csv`, ghi ra file mới
  `metrics_ci_lesion.json` cạnh `metrics.json` (không sửa file cũ).
- Vector hoá cho nhanh (tham khảo `weighted_confusion` trong `analyze_results.py`); 2000 lần lặp phải chạy < 10 s.

**Nghiệm thu.** Test: dữ liệu giả có lesion lặp → CI theo lesion rộng hơn hoặc bằng CI theo ảnh; `groups=None`
cho kết quả y hệt bản cũ với cùng seed.

### P0-6. Chống lỗi batch 1 ảnh với BatchNorm

**Lý do.** Train loader không `drop_last`. Với EfficientNet-B0 (BatchNorm), một micro-batch cuối chỉ có 1 ảnh sẽ
gây lỗi "Expected more than 1 value per channel". Hiện chưa xảy ra (7011 % 32 = 3), nhưng có thể xảy ra với
`train_fraction` hoặc micro-batch khác.

**Việc cần làm.** Trong `Trainer.fit` (`src/dl_project/engine/trainer.py`): nếu micro-batch có đúng 1 mẫu và model
đang train thì bỏ qua nó (ghi log một lần). **Không** bật `drop_last` toàn cục (vi phạm quy tắc 0.3 vì làm đổi số
bước mỗi epoch của run cũ).

**Nghiệm thu.** Test: dataset 33 mẫu, micro-batch 32, model có BatchNorm → train không lỗi.

### P0-7. README, khoá phiên bản thư viện

**Việc cần làm.**
- Sửa lỗi gõ `scriptsggregate_results.py` → `scripts\aggregate_results.py`.
- Cập nhật mục "Trạng thái hiện tại": đã có 15 thí nghiệm × 3 seed trên validation; liệt kê việc còn lại thật.
  Bỏ câu "Chưa có kết quả thí nghiệm chính thức". Số config đúng với thực tế.
- Thêm mục "Tái lập bảng kết quả" theo đúng thứ tự: tải dữ liệu → verify split → train/evaluate (hoặc notebook
  Kaggle) → `aggregate_results.py` → `analyze_results.py`.
- Tạo `requirements-lock.txt` với phiên bản **đã dùng trên Kaggle** (lấy từ
  `outputs/*/train/*/environment.txt`): `torch==2.11.0+cu130`, `torchvision==0.26.0+cu130`, `timm==1.0.30`,
  `numpy==1.26.4`, `scikit-learn==1.6.1`, `pillow==13.0.0`, cộng `matplotlib`, `pandas`, `grad-cam` với bản
  đang cài ở `.venv`. Ghi chú: Python 3.13 (Kaggle) / 3.11 (máy local).
- Sửa `pyproject.toml` `description` cho đúng đề tài.

---

## P1 — Bổ sung thí nghiệm/phân tích còn thiếu (ảnh hưởng điểm "Evidence & generalization")

### P1-1. Đánh giá ngoài trên PAD-UFES-20 (RQ3)

**Lý do.** Đề tài hứa kiểm tra domain shift sang ảnh điện thoại; config đã khai báo
`dataset.external_test = pad_ufes_20` nhưng chưa có code nào đọc dữ liệu này.

**Việc cần làm.**
- `src/dl_project/data/pad_ufes20.py`: đọc `data/raw/pad_ufes_20/metadata.csv` (cột `patient_id, lesion_id,
  diagnostic, age, gender, region, img_id, biopsed, ...`) và ảnh `.png` trong `data/raw/pad_ufes_20/images/`
  (hướng dẫn tải ở `scripts/download_data.md`). Map nhãn theo `docs/PROPOSAL.md`: MEL→mel, BCC→bcc, NEV→nv,
  SEK→bkl, ACK→akiec, SCC→akiec (map xấp xỉ, ghi rõ trong docstring và báo cáo). Không có df, vasc.
- Map metadata sang encoder HAM10000 (`MetadataEncoder`, dùng thống kê đã fit trên train HAM lưu trong checkpoint):
  `gender` MALE/FEMALE→male/female, khác→unknown; `region` → bảng map tường minh sang `LOCALIZATIONS`
  (ví dụ FACE/NOSE/LIP/EAR→face hoặc ear, FOREARM/ARM→upper extremity, THIGH→lower extremity, …; giá trị không map
  được → unknown). Đặt bảng map thành hằng số có test.
- `SkinLesionTask.evaluate` hỗ trợ `--split external` (CLI cho phép giá trị này; không cần `--final-test` vì
  không dùng để chọn cấu hình, nhưng ghi vào `experiments/test_log.csv` với split `external`).
- Metric trên tập ngoài: macro-F1 / balanced accuracy / recall **chỉ trên 5 lớp có mặt** (akiec, bcc, bkl, mel, nv);
  báo cáo thêm tỉ lệ ảnh bị dự đoán thành df/vasc. AUROC melanoma. CI bootstrap theo `lesion_id` của PAD.
- `scripts/aggregate_results.py --split external` sinh `results_external.csv`.
- Cập nhật `notebooks/kaggle_runner.ipynb` để có thể evaluate external (dataset PAD được upload thành dataset
  Kaggle riêng tư, đường dẫn symlink như HAM).
- Thêm config E8 (`e08_film_la_domainaug`) và **thêm config mới `e14_concat_la_domainaug`** (concat + LA + domain
  aug) qua `make_configs.py`, vì cấu hình tốt nhất hiện tại là E5 (concat + LA), không phải E7.

**Nghiệm thu.** Test với `metadata.csv` giả vài dòng: map nhãn/metadata đúng; macro-F1 chỉ tính trên lớp có mặt.
Code chạy được với `--debug`. Chưa cần có số liệu thật (nhóm tự tải dữ liệu và chạy).

### P1-2. Quét τ cho logit-adjusted loss (validation)

**Lý do.** τ đang cố định 1.0. `PLAN_BTL.md` dự kiến chọn τ trên validation. Cấu hình được chọn hiện là E5
(concat + LA), nên quét cho E5 chứ không phải E7.

**Việc cần làm.** Trong `make_configs.py` thêm `t05_concat_la_tau05`, `t05_concat_la_tau15`,
`t05_concat_la_tau20` (concat, logit_adjusted, τ = 0.5 / 1.5 / 2.0), 3 seed mỗi cái, giữ nguyên mọi thứ khác so với
E5. Cập nhật `experiments/matrix.csv` (qua script). Chuẩn bị danh sách `EXPERIMENTS` cho notebook Kaggle (đợt 4)
trong `notebooks/README.md`. **Không** thay đổi τ của E5 trong code.

**Nghiệm thu.** Config sinh đúng, `--debug` chạy được.

### P1-3. Grad-CAM cho phân tích lỗi

**Việc cần làm.** `scripts/gradcam.py --checkpoint <best.pt> --split validation --n 16`: dùng `pytorch-grad-cam`
(gói `grad-cam`, đã có trong `[train]`), target layer = stage cuối của backbone (`model.backbone.stages[-1]`),
model nhận 3 input nên viết wrapper cố định metadata của từng ảnh. Chọn ảnh theo quy tắc **định nghĩa trước**:
các melanoma bị đoán đúng / bị đoán thành nv, theo thứ tự `image_id` (không chọn tay). Xuất lưới ảnh PNG vào
`reports/figures/gradcam_<experiment>_validation.png`. Chỉ dùng validation.

**Nghiệm thu.** Chạy được trên CPU với 1 checkpoint thật (`outputs/e05_concat_la-s42/...`) hoặc model random trong
test (skip nếu thiếu thư viện).

### P1-4. Đo hiệu năng (tham số, latency, bộ nhớ)

**Việc cần làm.** `scripts/measure_efficiency.py` dùng `src/dl_project/metrics/efficiency.py`: cho E1, E2, E5, E7,
E10, E13 — số tham số, latency (batch 1 và 32, AMP, warmup 20, 100 lần), peak memory, phần cứng, kích thước ảnh.
Ghi `reports/tables/efficiency.csv`. FiLM/LA không đổi chi phí suy luận đáng kể — bảng này để chứng minh điều đó.

### P1-5. Bảng và hình cho đủ 15 thí nghiệm

**Việc cần làm.** `scripts/analyze_results.py`:
- Thêm các cặp so sánh mới vào `PAIRS`: E13 − E10 (LA vs CE trên EfficientNet), E11 − E13 (FiLM vs concat trên
  EfficientNet), E9 − E7 (FiLM stage 3 vs stage 2+3), E12e − E12a và E12f − E12b (LA vs CE ở 25%/50% dữ liệu).
- Thêm nhãn cho mọi thí nghiệm vào `LABELS` (heatmap hiện chỉ có E1–E7).
- Hình mới `reports/figures/data_size_validation.png`: macro-F1 và rare recall theo % dữ liệu train (25/50/100)
  cho CE (E12a, E12b, E2) và LA (E12e, E12f, E5), có error bar ±std qua seed.
- Không đổi màu/phong cách hình hiện có.

**Nghiệm thu.** Chạy script trên `outputs/` thật không lỗi; các cặp cũ cho số y hệt bản đã commit
(`reports/tables/paired_bootstrap_validation.csv`).

### P1-6. Script chấm test cuối cùng (chỉ viết, KHÔNG chạy)

**Việc cần làm.** `scripts/final_test.py --experiments e01_image_only_ce e02_concat_ce e05_concat_la e07_film_la
e10_effb0_concat_ce e13_effb0_concat_la --seeds 42 43 44`: với mỗi (experiment, seed) tìm đúng một `best.pt`, gọi
`evaluate --split test --final-test`; từ chối nếu `experiments/test_log.csv` đã có checkpoint đó; cuối cùng chạy
`aggregate_results.py --split test` và `analyze_results.py --split test`. In ra danh sách sẽ chạy và yêu cầu gõ
`yes` để xác nhận. Danh sách thí nghiệm cuối cùng do nhóm chốt.

---

## P2 — Nên có nếu còn thời gian

- **P2-1. Temperature scaling.** LA và FiLM làm ECE tăng (0.086–0.096 so với 0.059). Thêm `scripts/calibrate.py`:
  fit nhiệt độ T trên logit validation (lưu logit ở `predictions.csv` hoặc tính lại từ checkpoint), báo ECE
  trước/sau trên validation; khi chấm test dùng T đã fit trên validation. Không thay đổi argmax.
- **P2-2. CI chạy được phần torch.** Thêm job CI thứ hai cài torch CPU + timm (không pretrained) để chạy
  `tests/test_models.py`, `tests/test_training.py`; đánh dấu `continue-on-error` nếu chậm.
- **P2-3. Kiểm tra đồng nhất.** Test: mọi `configs/experiments/*.json` khớp đúng với `make_configs.py`
  (sinh lại trong tempdir và so sánh), để tránh config bị sửa tay.
- **P2-4. Dọn repo.** `src/dl_group_project.egg-info/` không nên được commit (thêm vào `.gitignore`, `git rm
  --cached`). `docs/CHECKLIST.md`, `docs/TEAM.md`: cập nhật trạng thái thật.

---

## Thông tin tham khảo cho Codex

- Cấu trúc: `src/dl_project/{cli.py,config.py,tracking.py}`, `data/{splits.py,ham10000.py}`,
  `models/{backbones.py,film.py,classifier.py}`, `engine/{trainer.py,losses.py,seed.py}`,
  `metrics/{classification.py,efficiency.py}`, `tasks/skin_lesion.py`; scripts trong `scripts/`.
- Mỗi run: `outputs/<experiment>-s<seed>/<train|evaluate>/<UTC-id>/` với `config.json`, `metadata.json`
  (`git_commit`, `created_utc`), `environment.txt`, `train.log`, `history.csv`, `best.pt`, `last.pt`,
  `encoder.json`, `train_summary.json` | `metrics.json`, `predictions.csv`.
- Checkpoint: dict `{model, epoch, val_macro_f1, config, encoder, classes}`.
- Lớp: `akiec, bcc, bkl, df, mel, nv, vasc`; lớp hiếm (định nghĩa trước): `akiec, bcc, df, mel, vasc`.
- Cài đặt train đã dùng cho cả 45 run: batch 32 (micro 32, không tích luỹ), AdamW lr 1e-4 (head 1e-3),
  wd 0.05, cosine + warmup 1 epoch, tối đa 30 epoch, early stopping patience 5 theo val macro-F1, AMP fp16,
  grad clip 1.0, ảnh 224, Tesla T4.
- Kết quả hiện tại (validation, mean ± std, 3 seed): xem `reports/tables/results_validation.csv`,
  so sánh cặp ở `reports/tables/paired_bootstrap_validation.csv`, bản nháp phân tích ở `reports/RESULTS_DRAFT.md`.
