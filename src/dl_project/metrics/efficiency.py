"""Parameter count, latency and peak memory (PLAN_BTL.md 5.9). Report GPU name, input size, batch size."""
import time


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def measure_latency(model, make_inputs, batch_sizes=(1, 32), warmup=20, iters=100, device="cuda", amp=True):
    """Mean/std forward latency (ms per batch) and peak memory; ``make_inputs(batch) -> tuple of tensors``."""
    import torch

    model = model.to(device).eval()
    results = {"device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "amp": amp,
               "warmup": warmup, "iters": iters, "batches": {}}
    for batch in batch_sizes:
        inputs = tuple(t.to(device) for t in make_inputs(batch))
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        times = []
        with torch.no_grad(), torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                             dtype=torch.float16 if device == "cuda" else torch.bfloat16,
                                             enabled=amp):
            for step in range(warmup + iters):
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                model(*inputs)
                if device == "cuda":
                    torch.cuda.synchronize()
                if step >= warmup:
                    times.append((time.perf_counter() - start) * 1000)
        mean = sum(times) / len(times)
        std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
        results["batches"][str(batch)] = {
            "ms_mean": mean, "ms_std": std, "images_per_s": batch * 1000 / mean,
            "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
        }
    return results
