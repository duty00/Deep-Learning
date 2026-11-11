# Results draft — RQ1 and RQ2 (validation only)

> **Ghi chú cho nhóm (C):** bản nháp này do AI soạn từ số liệu thật trong repo, dùng làm khung cho mục
> *Results and analysis*. Trước khi đưa vào báo cáo: kiểm tra lại từng số với các file nguồn bên dưới, viết lại
> bằng lời của nhóm, và **khai báo việc dùng AI** theo yêu cầu của đề.
> Mọi số ở đây là **validation**; số chính thức trong báo cáo phải là **test**, chấm một lần cho cấu hình đã chốt.
>
> Nguồn số liệu (tái tạo bằng `python scripts/aggregate_results.py` và `python scripts/analyze_results.py`):
> `reports/tables/results_validation.csv`, `reports/tables/paired_bootstrap_validation.csv`,
> `reports/figures/{recall_heatmap,reliability,confusion_e02_vs_e05}_validation.png`, run gốc trong `outputs/`.

## Setup recap

ConvNeXt-Tiny (ImageNet-pretrained), HAM10000 split by `lesion_id` (70/15/15, 7,011 / 1,502 / 1,502 images),
30 epochs max with early stopping on validation macro-F1 (patience 5), 3 seeds (42, 43, 44) per configuration,
Tesla T4 on Kaggle. Each row changes one component relative to its comparison partner.

## Table 1 — Validation results (mean ± std over 3 seeds)

| ID | Metadata fusion | Loss | Macro-F1 | Balanced acc. | Rare recall† | Mel recall | nv recall | ECE ↓ |
|---|---|---|---|---|---|---|---|---|
| E1 | none | CE | 0.742 ± 0.009 | 0.746 ± 0.016 | 0.720 ± 0.018 | 0.586 ± 0.062 | 0.937 ± 0.006 | 0.076 ± 0.027 |
| E2 | concat | CE | 0.754 ± 0.036 | 0.752 ± 0.037 | 0.720 ± 0.049 | 0.641 ± 0.117 | 0.923 ± 0.034 | 0.059 ± 0.022 |
| E3 | concat | weighted CE | 0.763 ± 0.022 | 0.771 ± 0.014 | 0.760 ± 0.017 | 0.711 ± 0.038 | 0.920 ± 0.035 | 0.081 ± 0.019 |
| E4 | concat | focal | **0.777 ± 0.021** | 0.757 ± 0.030 | 0.717 ± 0.053 | 0.530 ± 0.116 | **0.968 ± 0.011** | **0.045 ± 0.020** |
| E5 | concat | logit-adjusted | 0.764 ± 0.017 | **0.801 ± 0.017** | **0.800 ± 0.039** | **0.737 ± 0.018** | 0.907 ± 0.028 | 0.088 ± 0.038 |
| E6 | FiLM (stages 2–3) | CE | 0.744 ± 0.015 | 0.750 ± 0.024 | 0.721 ± 0.036 | 0.582 ± 0.072 | 0.958 ± 0.008 | 0.096 ± 0.033 |
| E7 | FiLM (stages 2–3) | logit-adjusted | 0.737 ± 0.010 | 0.774 ± 0.012 | 0.750 ± 0.017 | 0.612 ± 0.057 | 0.930 ± 0.011 | 0.086 ± 0.022 |

† Mean recall over akiec, bcc, df, mel, vasc.

## Table 2 — Paired differences (A − B), validation

Paired bootstrap over **lesions** (2,000 resamples of the 1,122 validation lesions; the same resample is applied
to both models and all three seeds, and metrics are averaged over seeds). The interval reflects sampling of the
validation set only, **not** seed-to-seed training variance, which Table 1 reports separately.

| Comparison | Macro-F1 | Balanced acc. | Rare recall | Mel recall | nv recall |
|---|---|---|---|---|---|
| E2 − E1 (concat vs no metadata) | +0.012 [−0.038, +0.056] | +0.005 [−0.047, +0.057] | −0.000 [−0.070, +0.072] | +0.054 [−0.004, +0.111] | **−0.014 [−0.028, −0.001]** |
| E6 − E2 (FiLM vs concat, CE) | −0.009 [−0.053, +0.035] | −0.002 [−0.041, +0.040] | +0.002 [−0.050, +0.058] | −0.058 [−0.121, +0.007] | **+0.035 [+0.023, +0.046]** |
| E7 − E5 (FiLM vs concat, LA) | −0.027 [−0.056, +0.002] | −0.027 [−0.057, +0.005] | **−0.050 [−0.091, −0.005]** | **−0.125 [−0.182, −0.072]** | **+0.023 [+0.009, +0.038]** |
| E3 − E2 (weighted CE vs CE) | +0.010 [−0.027, +0.046] | +0.020 [−0.013, +0.051] | +0.040 [−0.004, +0.083] | **+0.070 [+0.014, +0.133]** | −0.003 [−0.015, +0.007] |
| E4 − E2 (focal vs CE) | +0.023 [−0.012, +0.059] | +0.006 [−0.023, +0.036] | −0.003 [−0.043, +0.039] | **−0.110 [−0.173, −0.052]** | **+0.044 [+0.033, +0.056]** |
| E5 − E2 (LA vs CE) | +0.010 [−0.035, +0.057] | **+0.050 [+0.001, +0.098]** | **+0.081 [+0.015, +0.148]** | **+0.096 [+0.036, +0.159]** | **−0.016 [−0.028, −0.006]** |
| E5 − E3 (LA vs weighted CE) | +0.001 [−0.035, +0.039] | +0.030 [−0.008, +0.070] | +0.040 [−0.012, +0.097] | +0.026 [−0.008, +0.062] | **−0.013 [−0.025, −0.001]** |
| E5 − E4 (LA vs focal) | −0.013 [−0.049, +0.023] | **+0.044 [+0.003, +0.085]** | **+0.084 [+0.028, +0.141]** | **+0.207 [+0.136, +0.282]** | **−0.061 [−0.074, −0.047]** |
| E7 − E2 (proposed vs baseline) | −0.017 [−0.057, +0.026] | +0.023 [−0.014, +0.063] | +0.031 [−0.019, +0.087] | −0.028 [−0.082, +0.026] | +0.007 [−0.005, +0.018] |

Bold: 95% interval excludes zero.

## Findings (draft text)

**No configuration is separable on macro-F1.** Every macro-F1 difference in Table 2 has an interval that
contains zero, and the spread across seeds (up to ±0.036 for E2) is as large as the gaps between methods. Macro-F1
alone therefore cannot rank these configurations on HAM10000 validation; the differences appear in per-class
recall instead.

**RQ1 — FiLM does not outperform concatenation.** With CE, FiLM (E6) and concat (E2) are indistinguishable on
macro-F1, balanced accuracy and rare recall; FiLM shifts predictions towards the majority class (nv recall
+0.035) at the expense of melanoma (−0.058, interval just crossing zero). With the logit-adjusted loss, FiLM (E7)
is worse than concat (E5) on rare recall (−0.050) and melanoma recall (−0.125), both with intervals excluding
zero. A plausible mechanism, to be discussed in the Method/Analysis section: each ConvNeXt stage is followed by a
LayerNorm in the downsampling layer, which removes any channel-uniform component of the FiLM shift and scale, so
the modulation can only act through between-channel differences. The hypothesis that FiLM improves macro-F1 over
concatenation is **not supported**.

**Metadata helps only marginally.** Adding concatenated metadata (E2 vs E1) changes macro-F1 by +0.012 with an
interval spanning zero; its clearest effect is a small trade from nv towards melanoma.

**RQ2 — the logit-adjusted loss gives the best rare-class recall, at a cost to nv and calibration.** Relative to
CE, logit adjustment (E5) raises balanced accuracy (+0.050), rare recall (+0.081) and melanoma recall (+0.096),
all with intervals excluding zero, while nv recall drops by 0.016. Against focal loss the contrast is sharper:
focal loss attains the highest macro-F1 (0.777) and best calibration (ECE 0.045), but it does so by favouring
nv (recall 0.968) and has the lowest melanoma recall of all runs (0.530); E5 recovers +0.207 melanoma recall over
focal loss. Weighted CE (E3) lies between CE and logit adjustment and is not significantly different from E5 on
any rare-class metric. The confusion matrices (Fig. `confusion_e02_vs_e05`) show where the gain comes from: the
share of melanomas predicted as nv falls from 0.25 (E2) to 0.13 (E5), and dermatofibroma recall rises from 0.71
to 0.88, while more akiec and bkl are now predicted as mel.

**Calibration.** Logit adjustment and FiLM both worsen calibration (ECE 0.086–0.096 vs 0.059 for E2); the
reliability diagram (Fig. `reliability`) shows E5 is over-confident in the 0.6–0.9 confidence range. Focal loss
is the best calibrated. If calibrated probabilities matter for the use case, post-hoc temperature scaling on
validation should be evaluated before the test run.

**Implication for the proposed method.** The planned combination (E7, FiLM + logit adjustment) is not better
than the CE baseline on any metric with an interval excluding zero, and is worse than concat + logit adjustment
(E5) on rare-class recall. Following the plan's safe-stop rule, the report should keep E7 as the proposed method
and present this as a negative result for FiLM, while carrying E5 forward as the best configuration for the
backbone (E13), data-size (E12e/f) and test-set evaluations.

## Caveats to state in Limitations

- Validation only; the test set has not been used. All selection decisions above use validation.
- Three seeds per configuration; seed variance on some per-class recalls is large (e.g. akiec ±0.17 for E5),
  because the validation split contains few df and vasc images.
- The bootstrap interval captures validation-set sampling, not training randomness.
- The logit-adjustment temperature τ was fixed at 1.0; the planned τ sweep (`t07_*`) has not been run.
- No multiple-comparison correction across the nine comparisons; treat single borderline intervals with care.
