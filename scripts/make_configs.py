"""Generate configs/experiments/e*.json for the experiment matrix (PLAN_BTL.md section 5.10).

All configs share configs/baseline.json except the fields listed below, so a comparison changes only
the component under study. Seeds are passed at run time (--seed 42/43/44).
Usage: python scripts/make_configs.py   (overwrites configs/experiments/ and experiments/matrix.csv)
"""
import csv
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / "configs/baseline.json").read_text(encoding="utf-8"))

# id, research question, components, model overrides, training overrides, note
EXPERIMENTS = [
    ("e01_image_only_ce", "baseline", {"meta_fusion": "none", "loss": "ce"}, {}, {}, "image only"),
    ("e02_concat_ce", "baseline (main)", {"meta_fusion": "concat", "loss": "ce"}, {}, {}, "B1"),
    ("e03_concat_wce", "RQ2", {"meta_fusion": "concat", "loss": "weighted_ce"}, {}, {}, ""),
    ("e04_concat_focal", "RQ2", {"meta_fusion": "concat", "loss": "focal"}, {}, {}, ""),
    ("e05_concat_la", "RQ2", {"meta_fusion": "concat", "loss": "logit_adjusted", "tau": 1.0}, {}, {}, ""),
    ("e06_film_ce", "RQ1", {"meta_fusion": "film", "film_stages": [2, 3], "loss": "ce"}, {}, {}, ""),
    ("e07_film_la", "proposed", {"meta_fusion": "film", "film_stages": [2, 3], "loss": "logit_adjusted",
                                 "tau": 1.0}, {}, {}, "main method"),
    ("e08_film_la_domainaug", "RQ3", {"meta_fusion": "film", "film_stages": [2, 3], "loss": "logit_adjusted",
                                      "tau": 1.0, "domain_aug": True}, {}, {}, ""),
    ("e09_film3_la", "ablation FiLM", {"meta_fusion": "film", "film_stages": [3], "loss": "logit_adjusted",
                                       "tau": 1.0}, {}, {}, "FiLM stage 3 only"),
    ("e10_effb0_concat_ce", "RQ4", {"meta_fusion": "concat", "loss": "ce"}, {"name": "efficientnet_b0"},
     {"micro_batch_size": 32}, "BatchNorm: run on Kaggle with micro 32"),
    ("e11_effb0_film_la", "RQ4", {"meta_fusion": "film", "film_stages": [2, 3], "loss": "logit_adjusted",
                                  "tau": 1.0}, {"name": "efficientnet_b0"}, {"micro_batch_size": 32},
     "BatchNorm: run on Kaggle with micro 32"),
    ("e12a_concat_ce_frac25", "data size", {"meta_fusion": "concat", "loss": "ce", "train_fraction": 0.25},
     {}, {}, ""),
    ("e12b_concat_ce_frac50", "data size", {"meta_fusion": "concat", "loss": "ce", "train_fraction": 0.5},
     {}, {}, ""),
    ("e12c_film_la_frac25", "data size", {"meta_fusion": "film", "film_stages": [2, 3],
                                          "loss": "logit_adjusted", "tau": 1.0, "train_fraction": 0.25}, {}, {}, ""),
    ("e12d_film_la_frac50", "data size", {"meta_fusion": "film", "film_stages": [2, 3],
                                          "loss": "logit_adjusted", "tau": 1.0, "train_fraction": 0.5}, {}, {}, ""),
    # tau tuning on VALIDATION only (seed 42); pick tau before running E5/E7-E12 with 3 seeds.
    ("t07_film_la_tau05", "tune tau", {"meta_fusion": "film", "film_stages": [2, 3], "loss": "logit_adjusted",
                                       "tau": 0.5}, {}, {}, "validation only"),
    ("t07_film_la_tau15", "tune tau", {"meta_fusion": "film", "film_stages": [2, 3], "loss": "logit_adjusted",
                                       "tau": 1.5}, {}, {}, "validation only"),
]
DEFAULT_COMPONENTS = {"meta_fusion": "concat", "film_stages": [], "loss": "ce", "tau": None,
                      "domain_aug": False, "train_fraction": 1.0}


def main():
    out_dir = ROOT / "configs/experiments"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for exp_id, question, components, model, training, note in EXPERIMENTS:
        config = deepcopy(BASE)
        config["experiment_name"] = exp_id
        config["components"] = {**DEFAULT_COMPONENTS, **components}
        config["model"].update(model)
        config["training"].update(training)
        path = out_dir / f"{exp_id}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        seeds = "42" if exp_id.startswith("t") else "42;43;44"
        rows.append({"id": exp_id, "question": question, "config": path.relative_to(ROOT).as_posix(),
                     "seed": seeds, "owner": "A", "status": "todo", "run_path": "", "notes": note})
    with (ROOT / "experiments/matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} configs to {out_dir} and experiments/matrix.csv")


if __name__ == "__main__":
    main()
