"""SkinLesionTask: HAM10000 7-class classification (PLAN_BTL.md section 5).

train:    fits on train, selects the checkpoint on validation (never test), writes run artefacts.
evaluate: loads ``model.checkpoint`` and scores ``evaluation.split`` (validation while developing; test
          only for final, already-chosen configurations).
Run artefacts in run_dir: train.log, history.csv, best.pt, last.pt, encoder.json, val_metrics_best.json,
train_summary.json | metrics.json, predictions.csv.
"""
import csv
import json
import platform
from pathlib import Path

from .base import Task

CONFIG_DEFAULTS = {
    "image_size": 224, "num_workers": 2, "head_learning_rate": 1e-3, "weight_decay": 0.05,
    "warmup_epochs": 1, "early_stopping_patience": 5, "amp": True, "grad_clip": 1.0,
}
DEBUG_LIMITS = {"epochs": 2, "max_train_batches": 8, "max_eval_batches": 4}


def _logger(run_dir):
    path = Path(run_dir) / "train.log"

    def log(message):
        print(message, flush=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    return log


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _hardware(device):
    import torch

    info = {"device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": platform.python_version()}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, gpu_memory_mb=props.total_memory // 2**20)
    return info


def _loader(dataset, batch_size, shuffle, num_workers, seed):
    import torch
    from torch.utils.data import DataLoader
    from ..engine.seed import make_generator, worker_init_fn

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0, worker_init_fn=worker_init_fn,
                      generator=make_generator(seed) if shuffle else None)


class SkinLesionTask(Task):
    def train(self, config, project_root, run_dir):
        import torch
        from ..data.ham10000 import CLASSES, RARE_CLASSES, MetadataEncoder, build_datasets, class_counts
        from ..engine.losses import build_loss
        from ..engine.seed import set_seed
        from ..engine.trainer import Trainer
        from ..models.classifier import build_model
        from ..metrics.efficiency import count_parameters

        log = _logger(run_dir)
        train_cfg = {**CONFIG_DEFAULTS, **config["training"]}
        components = config.get("components", {})
        debug = bool(config.get("debug"))
        seed = config["seed"]
        set_seed(seed)
        device = _device()
        log(f"device: {_hardware(device)}")

        root = Path(project_root)
        datasets, encoder, train_records = build_datasets(
            root / config["dataset"]["root"], root / config["dataset"]["split_manifest"],
            image_size=train_cfg["image_size"], domain_aug=bool(components.get("domain_aug")),
            train_fraction=float(components.get("train_fraction", 1.0)), subsample_seed=seed)
        encoder.save(Path(run_dir) / "encoder.json")
        counts = class_counts(train_records)
        log(f"train images: {len(datasets['train'])} | val: {len(datasets['validation'])} | "
            f"train class counts {dict(zip(CLASSES, counts))}")

        micro = train_cfg.get("micro_batch_size", train_cfg["batch_size"])
        workers = 0 if debug else train_cfg["num_workers"]
        train_loader = _loader(datasets["train"], micro, True, workers, seed)
        val_loader = _loader(datasets["validation"], max(micro, 32), False, workers, seed)

        model = build_model(config, MetadataEncoder.NUM_NUMERIC, MetadataEncoder.NUM_LOCALIZATIONS).to(device)
        log(f"model: {config['model']['name']} fusion={model.meta_fusion} film_stages={model.film_stages} "
            f"params={count_parameters(model):,}")
        loss_fn = build_loss(components.get("loss", "ce"), counts, tau=components.get("tau")).to(device)
        optimizer = torch.optim.AdamW(model.parameter_groups(train_cfg["learning_rate"],
                                                             train_cfg["head_learning_rate"],
                                                             train_cfg["weight_decay"]))
        limits = DEBUG_LIMITS if debug else {}
        trainer = Trainer(
            model, loss_fn, optimizer, device, run_dir, CLASSES, RARE_CLASSES,
            batch_size=train_cfg["batch_size"], micro_batch_size=micro,
            epochs=limits.get("epochs", train_cfg["epochs"]), warmup_epochs=train_cfg["warmup_epochs"],
            patience=train_cfg["early_stopping_patience"], amp=train_cfg["amp"], grad_clip=train_cfg["grad_clip"],
            max_train_batches=limits.get("max_train_batches"), max_eval_batches=limits.get("max_eval_batches"),
            extra_checkpoint={"config": config, "encoder": encoder.state_dict(), "classes": list(CLASSES)},
            log=log)
        summary = trainer.fit(train_loader, val_loader)
        summary.update(hardware=_hardware(device), debug=debug, micro_batch_size=micro,
                       train_class_counts=dict(zip(CLASSES, counts)), parameters=count_parameters(model))
        (Path(run_dir) / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log(f"done: best val macro-F1 {summary['best_val_macro_f1']:.4f} at epoch {summary['best_epoch']} "
            f"| peak GPU memory {summary['peak_gpu_memory_mb']} MB | checkpoint {Path(run_dir) / 'best.pt'}")

    def evaluate(self, config, project_root, run_dir):
        import torch
        from ..data.ham10000 import (CLASSES, RARE_CLASSES, HAM10000Dataset, MetadataEncoder, build_transforms,
                                     index_images, load_split_records)
        from ..engine.trainer import predict
        from ..metrics.classification import bootstrap_ci, classification_report
        from ..models.classifier import build_model

        log = _logger(run_dir)
        root = Path(project_root)
        split = config["evaluation"]["split"]
        device = _device()
        checkpoint = torch.load(root / config["model"]["checkpoint"], map_location="cpu", weights_only=False)
        trained_config = checkpoint["config"]
        encoder = MetadataEncoder.from_state_dict(checkpoint["encoder"])  # train-fitted statistics
        model = build_model({**trained_config, "model": {**trained_config["model"], "pretrained": None}},
                            MetadataEncoder.NUM_NUMERIC, MetadataEncoder.NUM_LOCALIZATIONS)
        model.load_state_dict(checkpoint["model"])
        model.to(device)

        image_size = {**CONFIG_DEFAULTS, **trained_config["training"]}["image_size"]
        dataset_root = root / config["dataset"]["root"]
        records = load_split_records(dataset_root, root / config["dataset"]["split_manifest"])[split]
        dataset = HAM10000Dataset(records, index_images(dataset_root), encoder, build_transforms(False, image_size))
        loader = _loader(dataset, 64, False, 0 if config.get("debug") else 2, 0)
        ids, y_true, probs = predict(model, loader, device,
                                     max_batches=DEBUG_LIMITS["max_eval_batches"] if config.get("debug") else None)
        report = classification_report(y_true, probs, list(CLASSES), RARE_CLASSES)
        report["macro_f1_ci95"] = bootstrap_ci(y_true, probs, "macro_f1", list(CLASSES), rare_classes=RARE_CLASSES)
        report.update(split=split, checkpoint=str(config["model"]["checkpoint"]),
                      checkpoint_epoch=checkpoint.get("epoch"), trained_experiment=trained_config["experiment_name"],
                      trained_seed=trained_config["seed"], hardware=_hardware(device))
        (Path(run_dir) / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        with (Path(run_dir) / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_id", "y_true", "y_pred"] + [f"p_{name}" for name in CLASSES])
            for image_id, label, row in zip(ids, y_true, probs):
                writer.writerow([image_id, CLASSES[label], CLASSES[row.argmax()]] + [f"{p:.6f}" for p in row])
        log(f"{split}: macro-F1 {report['macro_f1']:.4f} (95% CI {report['macro_f1_ci95'][0]:.3f}-"
            f"{report['macro_f1_ci95'][1]:.3f}) | bal-acc {report['balanced_accuracy']:.4f} | "
            f"rare-recall {report['rare_recall']:.4f} | mel AUROC {report['mel_auroc']}")
