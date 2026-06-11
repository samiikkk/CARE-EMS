import time
import numpy as np
import torch


# PROFILING HELPERS
def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_inference(model, loader, device, n_warmup_batches=5):
    model.eval()
    use_cuda = device.type == "cuda"

    latencies     = []
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (batch_x, _) in enumerate(loader):
            batch_x = batch_x.to(device)
            B = batch_x.size(0)

            # warm-up
            if batch_idx < n_warmup_batches:
                _ = model(batch_x)
                if use_cuda:
                    torch.cuda.synchronize(device)
                continue

            if use_cuda:
                torch.cuda.synchronize(device)

            t0 = time.perf_counter()
            _  = model(batch_x)

            if use_cuda:
                torch.cuda.synchronize(device)  

            t1 = time.perf_counter()

            latencies.append((t1 - t0) / B) 
            total_samples += B

    avg_latency_ms = np.mean(latencies) * 1_000   # convert to ms


    return avg_latency_ms, total_samples


def print_profiling_report(model, test_loader, device, peak_memory_mb):
    print("\n" + "=" * 60, flush=True)
    print("  MODEL PROFILING REPORT", flush=True)
    print("=" * 60, flush=True)

    # Parameters 
    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters     : {total_params:,}", flush=True)
    print(f"  Trainable parameters : {trainable_params:,}", flush=True)

    # Inference time & memory 
    print("\n  Measuring inference latency and peak memory ...", flush=True)
    avg_lat_ms, n_samples = measure_inference(model, test_loader, device)

    print(f"  Samples measured     : {n_samples:,}", flush=True)
    print(f"  Avg latency / sample : {avg_lat_ms:.4f} ms", flush=True)


    print(f"  GPU peak memory: {peak_memory_mb:.2f} MB", flush=True)
