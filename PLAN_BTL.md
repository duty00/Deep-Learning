# PLAN BTL — CO3133 Deep Learning and Applications (HK252)

> File này tổng hợp **toàn bộ yêu cầu đề bài**, **kế hoạch thực hiện**, **các đề tài đề xuất** và **đặc tả kỹ thuật** để giao cho model code.
> Ngày lập: 09/11/2026 · Deadline nộp: **Chủ nhật 13/12/2026** → còn **~5 tuần**.
> Repo: `D:\DL` (bộ khung nhóm 3 người đã có sẵn: `src/dl_project/`, `configs/`, `experiments/`, `docs/`...).
>
> ✅ **ĐÃ CHỐT (09/11/2026): Đề tài 1 — phân loại tổn thương da mất cân bằng (HAM10000) với FiLM-metadata + logit-adjusted loss, kiểm tra domain shift trên PAD-UFES-20.** Đặc tả triển khai: mục 5. Tóm tắt đề xuất: `docs/PROPOSAL.md`. Config: `configs/baseline.json` (= E2), `configs/proposed.json` (= E7).

---

## 0. Hướng dẫn cho model code (đọc trước)

1. Đọc mục **1 (tóm tắt đề)** và mục **5 (đặc tả đề tài đã chốt)**. Chỉ code đề tài đã chốt, không tự đổi đề tài.
2. Tuân thủ bộ khung repo hiện có:
   - Task mới: tạo `src/dl_project/tasks/<ten_task>.py`, kế thừa `Task` (`src/dl_project/tasks/base.py`), cài `train(config, project_root, run_dir)` và `evaluate(config, project_root, run_dir)`, đăng ký vào `TASKS` trong `src/dl_project/tasks/__init__.py`.
   - Data → `src/dl_project/data/`, model → `src/dl_project/models/`, vòng lặp train/eval → `src/dl_project/engine/`, metric → `src/dl_project/metrics/`.
   - Cấu hình JSON trong `configs/` (đang có `baseline.json`, `proposed.json` với các trường `TBD` cần điền). Mỗi biến thể ablation = 1 file config.
   - Chạy: `python -m dl_project check|train|evaluate --config configs/<file>.json`.
   - CLI tự tạo `outputs/<experiment>/<mode>/<UTC-id>/` lưu config + commit + môi trường. Task phải tự ghi log, checkpoint, metric thật vào thư mục đó.
3. **Quy tắc bất biến** (vi phạm = mất điểm/không hợp lệ):
   - Test set chỉ dùng để đánh giá cuối. Chọn hyperparameter & checkpoint bằng **validation**.
   - Mọi tiền xử lý phụ thuộc dữ liệu (mean/std, class weights, prior lớp, tokenizer vocab mới...) **chỉ fit trên train**.
   - Chia tập **theo nhóm** (bệnh nhân / tổn thương / tài liệu / người dùng) khi mẫu có liên quan → không rò rỉ.
   - Cố định seed Python/NumPy/PyTorch/CUDA + `worker_init_fn`; ghi seed vào kết quả.
   - **Không** dùng ChatGPT/API đóng để dự đoán trên dữ liệu dự án. Chỉ dùng model mở trọng số.
   - **Không** bịa số liệu, không để số mặc định trong `experiments/results.csv`.
   - Ghi nguồn code tái sử dụng vào `docs/ATTRIBUTION.md`, đánh dấu phần reused/modified trong code.
4. Thứ tự code: data + EDA → baseline chạy được end-to-end trên subset nhỏ (sanity check: overfit 1 batch, kiểm tra shape/loss) → baseline đầy đủ → phương pháp đề xuất → ablation → phân tích.
5. Viết unit test nhỏ cho: split không rò rỉ nhóm, shape đầu ra model, metric tính đúng trên ví dụ tay.

---

## 1. Tóm tắt đề bài

**Mục tiêu:** Tự chọn một đề tài nghiên cứu **Computer Vision hoặc NLP (gồm LLM)** (vision–language cũng được). Xác định rõ input, output, use case, câu hỏi nghiên cứu.

**Đóng góp phải gắn với mô hình DL**, ví dụ:
- sửa kiến trúc;
- ensemble / cơ chế tổng hợp (aggregation);
- thay đổi hàm mục tiêu (loss);
- chiến lược thích nghi theo tác vụ (adaptation, PEFT...);
- nén / chưng cất (distillation, pruning, quantization).

❌ Chỉ bọc model có sẵn vào app là **không đủ**. Chỉ đổi dataset/giao diện **không** phải đóng góp.

**Phải chứng minh một điểm mạnh rõ ràng** bằng bằng chứng: tốt hơn tổng thể, hoặc tốt hơn ở ca khó cụ thể, hoặc tổng quát hóa tốt hơn, hoặc giảm tham số/bộ nhớ/latency mà giữ chất lượng. So với baseline phù hợp, báo cáo trade-off.

### 6 bước & checkpoint

| Bước | Nội dung | Checkpoint |
|---|---|---|
| 1. Dataset & bài toán | Nguồn, giấy phép, quy trình thu thập, số mẫu, nhãn, phân phối lớp, ví dụ. Kiểm tra trùng lặp, nhãn nhiễu/thiếu, mất cân bằng, bias. Đọc paper, tìm **hạn chế cụ thể**. Phát biểu **giả thuyết kiểm chứng được**. Tránh MNIST/CIFAR. | Problem statement, lý do chọn data, baseline, claim đo được |
| 2. Tiền xử lý & huấn luyện | Normalize, resize/tokenize, augmentation, xử lý mất cân bằng. Chia train/val/test **theo nhóm**. Fit tiền xử lý chỉ trên train. Ghi rõ pretrained weights, phần freeze, loss, optimizer, LR schedule, batch, epoch, early stopping, seed. **Giải thích lý do** mỗi lựa chọn. | Protocol tái lập + giải thích chống leakage |
| 3. Metric | Metric kiểm tra đúng claim. Phân loại: precision/recall/macro-F1 (accuracy có thể che lớp thiểu số). Sinh văn bản: protocol chuẩn + rubric người. Hiệu năng: params, latency, memory, throughput + phần cứng, input size, batch size. Cùng test set, cùng implementation metric. | Metric chính, metric phụ, hạn chế của chúng |
| 4. Thí nghiệm tổng quát hóa | So với phương pháp có sẵn đáng tin + base model chưa sửa. **Ablation**. Test ca khó/nhiễu/kích thước data/domain chưa thấy (định nghĩa subset **trước** khi xem kết quả). Nhiều seed, báo độ biến thiên. Show cả ca thất bại. | **Experiment matrix** nối mỗi so sánh với 1 câu hỏi nghiên cứu |
| 5. Code | Jupyter, NumPy, pandas, PyTorch, Matplotlib. Baseline trước. README: truy cập data, cài môi trường, lệnh/thứ tự notebook để tái lập. | Codebase chạy được, kết quả truy vết được |
| 6. Báo cáo | **Tiếng Anh, LaTeX, Springer LNCS** (không đổi lề/font). Cấu trúc: Abstract; Introduction & contributions; Related work; Proposed method; Experimental setup; Results & analysis; Limitations; Conclusion; References. Có sơ đồ kiến trúc/công thức, bảng so sánh, thí nghiệm thất bại. Được dùng AI dịch nhưng phải **khai báo**. | — |

### Nộp bài (13/12/2026)
1. **PDF báo cáo tiếng Anh** format LNCS.
2. **Codebase gốc**: implementation, hướng dẫn train/eval tái lập, **checkpoint model**, dependencies, configs, hướng dẫn truy cập data/checkpoint (tôn trọng license).

### Rubric (40 điểm = 40% môn)

| Hạng mục | Điểm | Tiêu chí |
|---|---|---|
| **Method** | 16 | Problem & motivation (4) · **Contribution (6)** · Technical soundness (6) |
| **Experiment** | 16 | Protocol & baselines (5) · **Evidence & generalization (7)** · Analysis (4) |
| **Presentation** | 8 | Report quality (5) · Code presentation (3) |

≥30/40 = chất lượng có thể nộp hội nghị top. **Một cải tiến nhỏ nhưng được kiểm chứng kỹ > tham vọng lớn bằng chứng yếu.** Kết quả âm báo cáo trung thực vẫn được điểm phân tích.

### Hỗ trợ
- Có thể nhờ GV kết nối chuyên gia lĩnh vực (Medical AI, toán ứng dụng...).
- GV chọn 1–2 nhóm tiềm năng nhất để tài trợ thêm GPU → nên có **kết quả sơ bộ sớm**.

---

## 2. Giả định về tài nguyên

- Nhóm 3 người (A/B/C theo `docs/TEAM.md`).
- GPU: Kaggle (T4×2 / P100, ~30h/tuần/tài khoản) hoặc Colab miễn phí; có thể có GPU cá nhân.
- ⇒ Ưu tiên dataset ≤ vài chục GB, backbone ≤ ~100M tham số (hoặc LLM ≤1.5B với QLoRA), mỗi lần train ≤ 2–3h để chạy được **3 seed × nhiều ablation**.

---

## 3. Đề tài đề xuất

Thang đánh giá: Khả thi (compute + thời gian 5 tuần) · Tiềm năng điểm Contribution/Evidence · Rủi ro.

### ⭐ Đề tài 1 (KHUYẾN NGHỊ) — Phân loại tổn thương da mất cân bằng, có kết hợp metadata, kiểm tra tổng quát hóa chéo domain

- **Bài toán:** Ảnh dermoscopy (+ metadata tuổi/giới/vị trí) → 1 trong 7 loại tổn thương. Use case: hỗ trợ sàng lọc ung thư da, đặc biệt **không bỏ sót lớp hiếm/nguy hiểm** (melanoma, bcc, akiec, df, vasc).
- **Data chính:** HAM10000 (10.015 ảnh, 7 lớp, lệch nặng: `nv` ~67%, `df`/`vasc` ~1–1,5%; license CC BY-NC 4.0). Có `lesion_id` → nhiều ảnh cùng 1 tổn thương ⇒ **phải chia theo `lesion_id`** (điểm cộng lớn cho phần chống leakage).
- **Data kiểm tra domain shift:** PAD-UFES-20 (ảnh chụp bằng điện thoại, không phải dermoscopy, ~2.3k ảnh) — chỉ dùng **đánh giá zero-shot** sau khi map nhãn về các lớp chung (MEL→mel, BCC→bcc, NEV→nv, SEK→bkl, ACK/SCC→akiec; ghi rõ map là xấp xỉ). Phương án thay thế: tập ISIC 2019 loại bỏ ảnh trùng HAM10000.
- **Hạn chế cần giải quyết:** CNN/ViT fine-tune với CE thường đạt accuracy cao nhờ lớp `nv` nhưng recall lớp hiếm thấp; metadata thường được nối (concat) đơn giản ở cuối, ít cải thiện.
- **Đóng góp đề xuất (model-related):**
  1. **Metadata-conditioned FiLM**: metadata sinh hệ số (γ, β) điều biến feature map ở các stage giữa của backbone (thay vì concat cuối).
  2. **Logit-adjusted loss** (Menon et al., 2021) dùng prior lớp ước lượng **từ train** — so với CE, weighted CE, focal loss.
  3. (Tùy chọn) Augmentation mô phỏng domain điện thoại (blur, color jitter mạnh, JPEG) để tăng robustness.
- **Giả thuyết:** *"FiLM-metadata + logit-adjusted loss tăng macro-F1 và recall trung bình 4 lớp hiếm so với baseline ConvNeXt-T/EfficientNet-B0 fine-tune CE + concat-metadata, cùng ngân sách huấn luyện, trên HAM10000 test (chia theo lesion), và giảm mức sụt hiệu năng khi chuyển sang PAD-UFES-20."*
- **Metric:** chính = **macro-F1**; phụ = balanced accuracy, recall từng lớp, AUROC mel-vs-rest, ECE (calibration); domain shift = macro-F1 trên các lớp chung của PAD-UFES-20.
- **Khả thi:** ⭐⭐⭐⭐⭐ (ảnh 224–384px, 10k ảnh, ~20–40 phút/run trên T4). **Rủi ro:** đề tài phổ biến → phải nhấn mạnh phần FiLM + phân tích domain shift để thể hiện originality.

### Đề tài 2 — Phát hiện ngôn từ thù ghét tiếng Việt cho lớp thiểu số (NLP)

- **Data:** ViHSD (UIT, ~33k bình luận mạng xã hội, 3 nhãn CLEAN/OFFENSIVE/HATE, rất lệch về CLEAN). Có thể test chéo trên ViHOS / UIT-ViCTSD (độc tính) để kiểm tra tổng quát hóa.
- **Baseline:** PhoBERT-base / ViSoBERT fine-tune CE.
- **Đóng góp:** (a) **Supervised contrastive loss có trọng số theo lớp** + logit adjustment; (b) **pooling attention đa tầng** (kết hợp CLS của nhiều layer bằng trọng số học được); (c) xử lý teencode/viết tắt bằng augmentation (thay thế từ lóng, bỏ dấu) chỉ trên train.
- **Giả thuyết:** tăng macro-F1 và F1 lớp HATE so với PhoBERT CE; robust hơn với văn bản bỏ dấu/teencode (tạo test set nhiễu **có kiểm soát** từ test gốc).
- **Lưu ý leakage:** loại bình luận trùng/gần trùng giữa các split (dedupe sau chuẩn hóa).
- **Khả thi:** ⭐⭐⭐⭐⭐ (~15–30 phút/run). **Rủi ro:** nhãn nhiễu; cần phân tích lỗi kỹ.

### Đề tài 3 — Chưng cất tri thức cho mô hình nhẹ phân loại bệnh lá cây ngoài thực địa (Efficiency)

- **Data:** PlantDoc (ảnh thực địa, khó) + PlantVillage (ảnh phòng lab) → train trên một domain, test trên domain thực địa.
- **Teacher:** ViT-B/16 hoặc ConvNeXt-B fine-tune. **Student:** MobileNetV3 / EfficientNet-B0.
- **Đóng góp:** distillation **kết hợp logit + attention/feature map** với trọng số theo độ khó mẫu (confidence của teacher), có augmentation domain.
- **Metric:** macro-F1 + params, FLOPs, latency (CPU & GPU, batch=1, ghi rõ phần cứng), memory.
- **Giả thuyết:** student giữ ≥95% macro-F1 của teacher với ≤10% tham số, latency CPU giảm ≥5×; vượt student train thẳng và KD Hinton chuẩn.
- **Khả thi:** ⭐⭐⭐⭐. **Rủi ro:** PlantDoc nhỏ và nhiễu nhãn.

### Đề tài 4 — Tổng quát hóa đa domain với CLIP adapter (Vision–Language)

- **Data:** Office-Home (15.5k ảnh, 65 lớp, 4 domain Art/Clipart/Product/Real) hoặc PACS. Protocol **leave-one-domain-out**.
- **Baseline:** CLIP ViT-B/32 zero-shot, linear probe, CoOp/CLIP-Adapter.
- **Đóng góp:** adapter nhẹ có **regularization giữ gần không gian zero-shot** (ví dụ KL tới dự đoán zero-shot) + text prompt ensemble theo domain → tránh overfit domain nguồn.
- **Metric:** accuracy trung bình trên domain chưa thấy, worst-domain accuracy, số tham số train.
- **Khả thi:** ⭐⭐⭐⭐ (CLIP frozen, chỉ train adapter). **Rủi ro:** cạnh tranh mạnh với nhiều paper → khó vượt SOTA, nên claim là "hiệu quả tham số + worst-domain".

### Đề tài 5 — LoRA phân bổ rank thích nghi cho QA đọc hiểu tiếng Việt (LLM)

- **Data:** UIT-ViQuAD 2.0 (~35k câu hỏi, có câu không trả lời được).
- **Model:** Qwen2.5-0.5B/1.5B-Instruct (mở trọng số) + QLoRA 4-bit trên T4.
- **Đóng góp:** phân bổ rank LoRA khác nhau theo layer dựa trên độ nhạy gradient (đơn giản hóa AdaLoRA), hoặc thêm head phát hiện "unanswerable".
- **Metric:** EM, F1 (theo protocol ViQuAD), riêng cho câu unanswerable; số tham số train, VRAM.
- **Khả thi:** ⭐⭐⭐ (train lâu, eval sinh văn bản chậm). **Rủi ro:** cao nhất về thời gian.

### Bảng so sánh nhanh

| # | Đề tài | Mảng | Khả thi | Contribution | Chống leakage dễ trình bày | Thời gian/run |
|---|---|---|---|---|---|---|
| 1 ⭐ | Da liễu mất cân bằng + FiLM metadata + domain shift | CV (medical) | Rất cao | Cao | Rất tốt (lesion_id) | 20–40' |
| 2 | Hate speech tiếng Việt | NLP | Rất cao | Trung bình–cao | Tốt (dedupe) | 15–30' |
| 3 | KD bệnh lá cây | CV (efficiency) | Cao | Trung bình–cao | Trung bình | 30–60' |
| 4 | CLIP adapter đa domain | V–L | Cao | Trung bình | Tốt (domain split) | 10–30' |
| 5 | LoRA ViQuAD | LLM | Trung bình | Cao | Tốt | 2–4h |

**Quyết định:** Đã chốt **Đề tài 1**. Các đề tài 2–5 chỉ giữ lại để tham khảo; model code **không** triển khai chúng.

---

## 4. Timeline 5 tuần (09/11 → 13/12/2026)

| Tuần | Thời gian | Việc chính | Người | Đầu ra |
|---|---|---|---|---|
| W1 | 09–15/11 | Chốt đề tài, tải data, EDA (phân phối lớp, trùng lặp `lesion_id`, metadata thiếu), đọc 6–10 paper, điền `docs/PROPOSAL.md`, tạo split theo nhóm + manifest | Cả nhóm | Problem statement, split manifest, notebook EDA |
| W2 | 16–22/11 | Data pipeline + baseline (CE, concat metadata) chạy end-to-end; sanity check; baseline 3 seed; module metric | B: data/baseline · C: metric/eval · A: bắt đầu FiLM | Kết quả baseline thật trong `results.csv` |
| W3 | 23–29/11 | Cài phương pháp đề xuất (FiLM + logit-adjusted loss + aug), tune trên **val**; chạy so sánh focal/weighted CE | A: method · B: tuning · C: eval domain shift | Kết quả sơ bộ → **gửi GV xin feedback/GPU** |
| W4 | 30/11–06/12 | Ablation đầy đủ × 3 seed, đánh giá PAD-UFES-20, phân tích lỗi, Grad-CAM, calibration; chốt test 1 lần | Cả nhóm | Bảng kết quả + hình cho báo cáo |
| W5 | 07–13/12 | Viết báo cáo LNCS (Overleaf), README tái lập, upload checkpoint, rà soát integrity, nộp | A: Method/Intro · B: Setup/Results · C: Analysis/README | PDF + codebase |

**Mốc dừng an toàn:** hết W2 phải có baseline thật; hết W3 nếu phương pháp không cải thiện → vẫn giữ, chuyển trọng tâm sang phân tích trung thực "vì sao không cải thiện" + ablation (vẫn được điểm).

---

## 5. Đặc tả kỹ thuật Đề tài 1 (giao cho model code)

### 5.1 Tiêu đề làm việc
*"Metadata-Conditioned Feature Modulation and Prior-Aware Training for Long-Tailed Skin Lesion Classification under Domain Shift"*

### 5.2 Câu hỏi nghiên cứu
- **RQ1:** FiLM-metadata có tốt hơn concat-metadata và không dùng metadata không? (macro-F1)
- **RQ2:** Logit-adjusted loss có tăng recall lớp hiếm tốt hơn weighted CE / focal loss không, và đánh đổi gì với lớp `nv`?
- **RQ3:** Phương pháp có giảm sụt hiệu năng khi chuyển sang ảnh điện thoại (PAD-UFES-20) không?
- **RQ4:** Kết quả có ổn định qua 3 seed và các backbone khác nhau không?

### 5.3 Dữ liệu
- `data/raw/ham10000/`: ảnh + `HAM10000_metadata.csv` (cột: `lesion_id, image_id, dx, dx_type, age, sex, localization`).
- `data/raw/pad_ufes_20/`: ảnh + `metadata.csv` (dùng cột chẩn đoán, tuổi, giới, vị trí; map vị trí sang tập `localization` của HAM khi được, còn lại → `unknown`).
- Script tải/hướng dẫn: `scripts/download_data.md` (không commit data; HAM10000 qua Harvard Dataverse/Kaggle, PAD-UFES-20 qua Mendeley Data).
- **EDA bắt buộc** (`notebooks/01_eda.ipynb`): phân phối lớp, số ảnh/lesion, tuổi thiếu, phân phối vị trí theo lớp, ví dụ ảnh mỗi lớp, kiểm tra ảnh trùng (hash).

### 5.4 Chia tập (chống leakage)
- `StratifiedGroupKFold` (sklearn) theo `group = lesion_id`, stratify theo `dx` → train 70% / val 15% / test 15%.
- Lưu `data/splits/manifest.json` (image_id → split) + checksum; unit test: không `lesion_id` nào xuất hiện ở >1 split.
- Mean/std chuẩn hóa: dùng ImageNet (do pretrained) — ghi rõ lý do. Prior lớp, class weight, thống kê tuổi (mean/std để chuẩn hóa) **chỉ tính trên train**.
- PAD-UFES-20: **chỉ dùng làm test ngoài**, không train/tune.

### 5.5 Tiền xử lý & augmentation
- Resize ngắn → 256, RandomResizedCrop 224 (train), CenterCrop 224 (val/test).
- Train aug: flip ngang/dọc, xoay 0–360°, ColorJitter nhẹ, RandomAffine.
- Aug "domain điện thoại" (component tùy chọn, có ablation): GaussianBlur, ColorJitter mạnh, JPEG compression, giảm độ phân giải rồi phóng lại.
- Metadata: tuổi → chuẩn hóa (mean/std train) + cờ missing; giới → one-hot (+unknown); vị trí → embedding (+unknown).

### 5.6 Mô hình
- **Backbone chính:** ConvNeXt-Tiny pretrained ImageNet (`timm`: `convnext_tiny`). Backbone phụ cho RQ4: EfficientNet-B0.
- **B0 – Image-only:** backbone → GAP → Linear(7).
- **B1 – Concat-metadata (baseline chính):** [img_feat ; MLP(meta)] → Linear(7).
- **Proposed – FiLM-metadata:** `meta_enc = MLP(meta) → h (dim 64)`. Sau stage 2 và stage 3 của ConvNeXt: `F' = F * (1 + γ(h)) + β(h)`, γ/β là Linear từ h sang số kênh, **khởi tạo 0** (bắt đầu = identity, ổn định khi fine-tune). Đầu phân loại: [img_feat ; h] → Linear(7).
- Config hóa: `components.film_stages: [2,3] | []`, `components.meta_fusion: "none"|"concat"|"film"`.

### 5.7 Loss
- `ce` · `weighted_ce` (trọng số = inverse frequency chuẩn hóa, từ train) · `focal` (γ=2) · `logit_adjusted`: `CE(logits + τ·log(prior))` khi train, dự đoán bằng logits gốc khi test; τ ∈ {0.5, 1.0, 1.5} chọn trên val.
- Config: `components.loss`, `components.tau`.

### 5.8 Huấn luyện
- AdamW, lr backbone 1e-4, lr head/FiLM 1e-3, weight decay 0.05; cosine schedule + warmup 1 epoch; batch 32; tối đa 30 epoch; early stopping patience 5 theo **val macro-F1**; AMP fp16.
- Seeds: 42, 43, 44. Ghi `torch.__version__`, CUDA, GPU name vào run dir.
- Lưu `best.pt` (theo val macro-F1) + `last.pt`, `history.csv` (loss/metric từng epoch).

### 5.9 Đánh giá
- `src/dl_project/metrics/`: macro-F1, balanced accuracy, precision/recall/F1 từng lớp, confusion matrix, AUROC mel-vs-rest, ECE (15 bin), "rare-recall" = trung bình recall của {mel, bcc, akiec, df, vasc} (**định nghĩa trước khi xem test**).
- Hiệu năng: số tham số, latency (batch 1 & 32, GPU ghi tên, warmup 20 lần, đo 100 lần), VRAM đỉnh.
- PAD-UFES-20: báo macro-F1 trên các lớp chung + mức sụt so với HAM test.
- Xuất `metrics.json` + `predictions.csv` (image_id, y_true, y_pred, probs) vào run dir; script `scripts/aggregate_results.py` gom mean ± std qua seed → `reports/tables/*.csv` (+ bản LaTeX).
- Phân tích: Grad-CAM vài ca đúng/sai mỗi lớp, ma trận nhầm lẫn, ví dụ thất bại trên PAD-UFES-20.

### 5.10 Ma trận thí nghiệm (điền vào `experiments/matrix.csv`)

| ID | RQ | Meta fusion | Loss | Aug domain | Backbone | Seeds |
|---|---|---|---|---|---|---|
| E1 | baseline | none | ce | no | ConvNeXt-T | 3 |
| E2 | baseline chính | concat | ce | no | ConvNeXt-T | 3 |
| E3 | RQ2 | concat | weighted_ce | no | ConvNeXt-T | 3 |
| E4 | RQ2 | concat | focal | no | ConvNeXt-T | 3 |
| E5 | RQ2 | concat | logit_adjusted | no | ConvNeXt-T | 3 |
| E6 | RQ1 | film | ce | no | ConvNeXt-T | 3 |
| E7 | **Proposed** | film | logit_adjusted | no | ConvNeXt-T | 3 |
| E8 | RQ3 | film | logit_adjusted | yes | ConvNeXt-T | 3 |
| E9 | ablation FiLM | film (chỉ stage 3) | logit_adjusted | no | ConvNeXt-T | 3 |
| E10 | RQ4 | concat | ce | no | EfficientNet-B0 | 3 |
| E11 | RQ4 | film | logit_adjusted | no | EfficientNet-B0 | 3 |
| E12 | data size | E2 vs E7 với 25%/50% train | — | — | ConvNeXt-T | 1–3 |

Tổng ~33–40 run × ~30' ≈ 20h GPU → vừa quota Kaggle 1–2 tuần.

### 5.11 Cấu trúc file cần tạo

```text
src/dl_project/
  data/ham10000.py          # Dataset, đọc metadata, encode meta, transforms
  data/pad_ufes.py          # Dataset test ngoài + map nhãn
  data/splits.py            # Tạo split theo lesion_id, lưu/đọc manifest
  models/backbones.py       # Tạo backbone timm, lấy feature theo stage
  models/film.py            # FiLM layer + MetaEncoder
  models/classifier.py      # B0/B1/FiLM theo config
  engine/losses.py          # ce, weighted_ce, focal, logit_adjusted
  engine/trainer.py         # train loop, AMP, early stopping, checkpoint
  engine/seed.py            # set_seed, worker_init_fn
  metrics/classification.py # macro-F1, bal-acc, rare-recall, AUROC, ECE
  metrics/efficiency.py     # params, latency, VRAM
  tasks/skin_lesion.py      # SkinLesionTask(Task) → đăng ký TASKS["skin_lesion"]
configs/e01_image_only.json … configs/e12_*.json
scripts/make_splits.py, scripts/aggregate_results.py, scripts/gradcam.py
notebooks/01_eda.ipynb, 02_error_analysis.ipynb
tests/test_splits.py, tests/test_models.py, tests/test_metrics.py
```

Thêm vào `pyproject.toml` extra `train`: `timm`, `scikit-learn`, `pillow`, `grad-cam` (ghi version cụ thể khi chốt môi trường; freeze `requirements-lock.txt`).

### 5.12 Ví dụ config (`configs/e07_proposed.json`)

```json
{
  "experiment_name": "e07_film_logitadj",
  "task": "skin_lesion",
  "seed": 42,
  "dataset": {"name": "ham10000", "root": "data/raw/ham10000", "split_manifest": "data/splits/manifest.json", "group_key": "lesion_id",
              "external_test": {"name": "pad_ufes_20", "root": "data/raw/pad_ufes_20"}},
  "model": {"name": "convnext_tiny", "pretrained": "imagenet", "checkpoint": null},
  "training": {"batch_size": 32, "epochs": 30, "learning_rate": 1e-4, "head_learning_rate": 1e-3, "weight_decay": 0.05,
               "scheduler": "cosine", "warmup_epochs": 1, "early_stopping_patience": 5, "amp": true, "selection_split": "validation"},
  "evaluation": {"split": "test", "primary_metric": "macro_f1"},
  "components": {"meta_fusion": "film", "film_stages": [2, 3], "loss": "logit_adjusted", "tau": 1.0, "domain_aug": false},
  "output_dir": "outputs"
}
```

---

## 6. Báo cáo LNCS — dàn ý (≈12–16 trang)

1. **Abstract** (150–250 từ): vấn đề, phương pháp, kết quả chính có số (mean ± std), giới hạn.
2. **Introduction & Contributions**: động cơ lâm sàng; hạn chế của CE/concat; 3 gạch đầu dòng đóng góp.
3. **Related Work**: phân loại tổn thương da (ISIC challenges), long-tail learning (class-balanced loss, focal, logit adjustment), fusion metadata (concat, FiLM, attention), domain shift dermoscopy ↔ smartphone.
4. **Proposed Method**: sơ đồ kiến trúc, công thức FiLM, công thức logit adjustment, lý do thiết kế (khởi tạo identity...).
5. **Experimental Setup**: dataset + EDA, split theo lesion, tiền xử lý, hyperparameter, phần cứng, metric + hạn chế metric.
6. **Results & Analysis**: bảng chính (E1–E8), ablation, domain shift, data size, per-class recall, calibration, Grad-CAM, ca thất bại, thí nghiệm không thành công.
7. **Limitations**: data từ ít trung tâm, nhãn histopathology không đồng đều, map nhãn PAD xấp xỉ, không phải công cụ chẩn đoán.
8. **Conclusion**.
9. **References** + **AI-assistance disclosure** (nếu dùng AI dịch/sửa văn).

---

## 7. Checklist trước khi nộp

- [ ] Split theo `lesion_id`, có unit test, manifest được commit (không commit ảnh).
- [ ] Không có quyết định nào dùng test set; test chạy 1 lần cuối cho mỗi cấu hình đã chốt.
- [ ] Mọi kết quả trong báo cáo truy vết được tới `outputs/<run>` + commit hash, ghi trong `experiments/results.csv`.
- [ ] 3 seed cho thí nghiệm chính, báo mean ± std.
- [ ] Ablation cho từng thành phần đề xuất.
- [ ] Báo cáo cả kết quả xấu/thất bại.
- [ ] README: tải data, cài môi trường, lệnh tái lập bảng chính, link checkpoint (Google Drive/HF Hub).
- [ ] `docs/ATTRIBUTION.md` ghi nguồn timm, grad-cam, code tham khảo; license dataset (HAM10000 CC BY-NC).
- [ ] Báo cáo LNCS đúng template, tiếng Anh, có khai báo AI.
- [ ] Không dùng API model đóng để dự đoán.

---

## 8. Prompt mẫu để giao cho model code (từng giai đoạn)

1. *"Đọc `PLAN_BTL.md` mục 0, 5.3–5.4. Viết `src/dl_project/data/splits.py`, `scripts/make_splits.py` và `tests/test_splits.py` để tạo split theo lesion_id cho HAM10000, lưu `data/splits/manifest.json`."*
2. *"Theo mục 5.5–5.6, viết dataset HAM10000 có metadata và model B0/B1/FiLM. Thêm test kiểm tra shape đầu ra và FiLM khởi tạo identity."*
3. *"Theo mục 5.7–5.9, viết losses, trainer (AMP, early stopping theo val macro-F1, seed), metrics, và `SkinLesionTask`, đăng ký vào `TASKS`. Thêm chế độ `--debug` chạy 2 batch để sanity check."*
4. *"Tạo các config E1–E12 theo mục 5.10 và `scripts/aggregate_results.py` gom mean ± std ra CSV + LaTeX."*
5. *"Viết `scripts/gradcam.py`, notebook phân tích lỗi, đo efficiency; cập nhật README tái lập."*
