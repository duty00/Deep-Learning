"""Training / evaluation loops (PLAN_BTL.md 5.8, docs/HARDWARE.md batch convention).

* ``batch_size`` is the EFFECTIVE batch; the DataLoader yields ``micro_batch_size`` images and gradients are
  accumulated over ``accum = batch_size // micro_batch_size`` micro-batches before one optimizer step.
  Each epoch uses only complete accumulation groups (the few leftover images change every epoch because
  the loader reshuffles), so every update averages exactly ``batch_size`` images.
* The LR schedule (linear warmup + cosine) is defined in optimizer steps.
* Checkpoint selection and early stopping use the VALIDATION macro-F1 only.
"""
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..metrics.classification import classification_report


def cosine_with_warmup(optimizer, warmup_steps, total_steps, min_factor=0.01):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _forward(model, batch, device):
    return model(batch["image"].to(device, non_blocking=True),
                 batch["meta_num"].to(device, non_blocking=True),
                 batch["meta_loc"].to(device, non_blocking=True))


@torch.no_grad()
def predict(model, loader, device, amp=True, max_batches=None):
    """Return (image_ids, y_true, probs) for a loader, in loader order."""
    model.eval()
    ids, labels, probs = [], [], []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = _forward(model, batch, device)
        probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        labels.append(batch["label"].numpy())
        ids.extend(batch["image_id"])
    return ids, np.concatenate(labels), np.concatenate(probs)


class Trainer:
    def __init__(self, model, loss_fn, optimizer, device, run_dir, class_names, rare_classes,
                 batch_size, micro_batch_size, epochs, warmup_epochs=1, patience=5, amp=True,
                 grad_clip=1.0, max_train_batches=None, max_eval_batches=None, extra_checkpoint=None,
                 log=print):
        if batch_size % micro_batch_size:
            raise ValueError("batch_size must be a multiple of micro_batch_size.")
        self.model, self.loss_fn, self.optimizer = model, loss_fn, optimizer
        self.device, self.run_dir = device, Path(run_dir)
        self.class_names, self.rare_classes = list(class_names), list(rare_classes)
        self.accum = batch_size // micro_batch_size
        self.epochs, self.warmup_epochs, self.patience = epochs, warmup_epochs, patience
        self.amp = amp and device.type == "cuda"
        self.grad_clip = grad_clip
        self.max_train_batches, self.max_eval_batches = max_train_batches, max_eval_batches
        self.extra_checkpoint = extra_checkpoint or {}
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.log = log

    def _steps_per_epoch(self, loader):
        micro = len(loader) if self.max_train_batches is None else min(len(loader), self.max_train_batches)
        steps = micro // self.accum
        if steps == 0:
            raise ValueError(f"Not enough micro-batches ({micro}) for one optimizer step (accum={self.accum}).")
        return steps

    def save_checkpoint(self, path, epoch, val_report):
        torch.save({"model": self.model.state_dict(), "epoch": epoch, "val_macro_f1": val_report["macro_f1"],
                    **self.extra_checkpoint}, path)

    def fit(self, train_loader, val_loader):
        steps_per_epoch = self._steps_per_epoch(train_loader)
        scheduler = cosine_with_warmup(self.optimizer, self.warmup_epochs * steps_per_epoch,
                                       self.epochs * steps_per_epoch)
        history, best, best_epoch, bad_epochs = [], -1.0, None, 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_start, running, seen = time.time(), 0.0, 0
            self.optimizer.zero_grad(set_to_none=True)
            micro_limit = steps_per_epoch * self.accum
            for index, batch in enumerate(train_loader):
                if index >= micro_limit:
                    break
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                    logits = _forward(self.model, batch, self.device)
                loss = self.loss_fn(logits, batch["label"].to(self.device, non_blocking=True))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch}, micro-batch {index}.")
                self.scaler.scale(loss / self.accum).backward()
                running += loss.item() * len(batch["label"])
                seen += len(batch["label"])
                if (index + 1) % self.accum == 0:
                    if self.grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
            _, y_val, p_val = predict(self.model, val_loader, self.device, self.amp, self.max_eval_batches)
            val = classification_report(y_val, p_val, self.class_names, self.rare_classes)
            val_loss = float(F.nll_loss(torch.log(torch.from_numpy(p_val).clamp_min(1e-12)),
                                        torch.from_numpy(y_val)).item())
            row = {"epoch": epoch, "train_loss": running / max(1, seen), "val_loss": val_loss,
                   "val_macro_f1": val["macro_f1"], "val_balanced_accuracy": val["balanced_accuracy"],
                   "val_rare_recall": val["rare_recall"], "val_accuracy": val["accuracy"],
                   "lr_backbone": self.optimizer.param_groups[0]["lr"], "epoch_seconds": time.time() - epoch_start}
            history.append(row)
            self._write_history(history)
            self.save_checkpoint(self.run_dir / "last.pt", epoch, val)
            improved = val["macro_f1"] > best
            if improved:
                best, best_epoch, bad_epochs = val["macro_f1"], epoch, 0
                self.save_checkpoint(self.run_dir / "best.pt", epoch, val)
                (self.run_dir / "val_metrics_best.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
            else:
                bad_epochs += 1
            self.log(f"epoch {epoch:02d} | train_loss {row['train_loss']:.4f} | val_loss {val_loss:.4f} | "
                     f"val_macroF1 {val['macro_f1']:.4f} | val_rare_recall {val['rare_recall']:.4f} | "
                     f"{row['epoch_seconds']:.0f}s" + (" *" if improved else ""))
            if bad_epochs >= self.patience:
                self.log(f"early stopping: no val macro-F1 improvement for {self.patience} epochs")
                break
        summary = {"best_epoch": best_epoch, "best_val_macro_f1": best, "epochs_run": len(history),
                   "steps_per_epoch": steps_per_epoch, "accumulation_steps": self.accum,
                   "train_minutes": (time.time() - start) / 60,
                   "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20
                   if self.device.type == "cuda" else None}
        return summary

    def _write_history(self, history):
        with (self.run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
