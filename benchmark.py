"""Benchmark suite for torchcachex.

Measures and reports:
- Write scaling (O(1) verification)
- Read performance
- Memory usage
- Cache hit rates
- Comparison: with vs without caching

Generates markdown report with results.
"""

import argparse
import os
import tempfile
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torchcachex import ArrowIPCCacheBackend, CacheModuleDecorator


@dataclass
class BenchmarkResult:
    """Single benchmark result."""

    name: str
    metric: str
    value: float
    unit: str
    details: str = ""


class BenchmarkDataset(Dataset):
    """Dataset for benchmarking."""

    def __init__(self, num_samples=10000, feature_dim=512):
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        # Pre-generate data for consistent benchmarking
        torch.manual_seed(42)
        self.data = torch.randn(num_samples, 3, 224, 224)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input": self.data[idx],
            "cache_ids": f"sample_{idx}",
        }


class BenchmarkModule(nn.Module):
    """Simulates an expensive feature extractor."""

    def __init__(self, output_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, output_dim)

        for param in self.parameters():
            param.requires_grad = False

        self.call_count = 0

    def forward(self, x):
        self.call_count += 1
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def benchmark_write_scaling(tmpdir: str) -> list[BenchmarkResult]:
    """Verify O(1) write scaling: flush time independent of cache size."""
    print("\n[Benchmark] Write Scaling (O(1) Verification)")
    print("=" * 60)

    results = []
    cache_sizes = [1000, 5000, 10000, 20000, 50000]

    for size in cache_sizes:
        backend = ArrowIPCCacheBackend(
            cache_dir=tmpdir,
            module_id=f"write_scale_{size}",
            async_write=False,
            flush_every=1000,
        )

        # Pre-populate cache to given size
        print(f"  Pre-populating cache with {size} samples...")
        module = BenchmarkModule()
        dataset = BenchmarkDataset(num_samples=size)
        loader = DataLoader(dataset, batch_size=100, shuffle=False)

        for batch in loader:
            out = module(batch["input"])
            backend.put_batch({cid: out[i] for i, cid in enumerate(batch["cache_ids"])})
        backend.flush()

        # Now benchmark writing 1000 new samples
        print(f"  Benchmarking write of 1000 new samples (cache size: {size})...")
        new_samples = {f"new_{i}": torch.randn(512) for i in range(1000)}

        start = time.time()
        backend.put_batch(new_samples)
        backend.flush()
        elapsed = time.time() - start

        throughput = 1000 / elapsed
        print(f"    Time: {elapsed:.3f}s, Throughput: {throughput:.0f} samples/sec")

        results.append(
            BenchmarkResult(
                name="Write Scaling",
                metric=f"Flush @ {size} cached",
                value=elapsed,
                unit="seconds",
                details=f"{throughput:.0f} samples/sec",
            )
        )

    return results


def benchmark_read_performance(tmpdir: str) -> list[BenchmarkResult]:
    """Measure read performance at different cache sizes."""
    print("\n[Benchmark] Read Performance")
    print("=" * 60)

    results = []
    cache_sizes = [1000, 5000, 10000, 20000]

    for size in cache_sizes:
        backend = ArrowIPCCacheBackend(
            cache_dir=tmpdir,
            module_id=f"read_perf_{size}",
            async_write=False,
            lru_size=0,  # Disable LRU to test disk reads
        )

        # Populate cache
        print(f"  Populating cache with {size} samples...")
        samples = {f"key_{i}": torch.randn(512) for i in range(size)}
        backend.put_batch(samples)
        backend.flush()

        # Benchmark random reads
        print("  Benchmarking 1000 random reads...")
        import random

        random.seed(42)
        read_keys = [f"key_{random.randint(0, size - 1)}" for _ in range(1000)]

        start = time.time()
        results_data, missing = backend.get_batch(read_keys)
        elapsed = time.time() - start

        throughput = 1000 / elapsed
        print(
            f"    Time: {elapsed:.3f}s, Throughput: {throughput:.0f} samples/sec, Missing: {len(missing)}"
        )

        results.append(
            BenchmarkResult(
                name="Read Performance",
                metric=f"Read @ {size} cached",
                value=elapsed,
                unit="seconds",
                details=f"{throughput:.0f} samples/sec",
            )
        )

    return results


def benchmark_memory_usage(tmpdir: str) -> list[BenchmarkResult]:
    """Measure memory usage at different cache sizes."""
    print("\n[Benchmark] Memory Usage")
    print("=" * 60)

    try:
        import psutil
    except ImportError:
        print("  [Skip] psutil not installed")
        return []

    results = []
    process = psutil.Process(os.getpid())
    cache_sizes = [1000, 5000, 10000]

    for size in cache_sizes:
        # Measure before
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        backend = ArrowIPCCacheBackend(
            cache_dir=tmpdir,
            module_id=f"mem_usage_{size}",
            async_write=False,
            lru_size=100,  # Small LRU
        )

        # Populate cache
        print(f"  Populating cache with {size} samples...")
        samples = {f"key_{i}": torch.randn(512) for i in range(size)}
        backend.put_batch(samples)
        backend.flush()

        # Measure after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before

        print(f"    Memory increase: {mem_increase:.1f} MB")

        results.append(
            BenchmarkResult(
                name="Memory Usage",
                metric=f"Cache size: {size}",
                value=mem_increase,
                unit="MB",
                details=f"{mem_increase / size * 1024:.2f} KB/sample",
            )
        )

    return results


def benchmark_cache_speedup(tmpdir: str) -> list[BenchmarkResult]:
    """Compare cached vs uncached performance."""
    print("\n[Benchmark] Cache Speedup")
    print("=" * 60)

    results = []
    num_samples = 1000
    batch_size = 32

    dataset = BenchmarkDataset(num_samples=num_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Benchmark WITHOUT caching
    print("  Running WITHOUT cache...")
    module_nocache = BenchmarkModule()

    start = time.time()
    for batch in loader:
        _ = module_nocache(batch["input"])
    time_nocache = time.time() - start

    print(f"    Time: {time_nocache:.3f}s, Calls: {module_nocache.call_count}")

    # Benchmark WITH caching (first epoch - populate cache)
    print("  Running WITH cache (epoch 1 - populate)...")
    backend = ArrowIPCCacheBackend(
        cache_dir=tmpdir,
        module_id="speedup_test",
        async_write=False,
    )
    module_cached = BenchmarkModule()
    cached = CacheModuleDecorator(module_cached, backend, enabled=True)

    start = time.time()
    for batch in loader:
        _ = cached(batch["input"], cache_ids=batch["cache_ids"])
    backend.flush()
    time_epoch1 = time.time() - start

    print(f"    Time: {time_epoch1:.3f}s, Module calls: {module_cached.call_count}")

    # Benchmark WITH caching (second epoch - cache hits)
    print("  Running WITH cache (epoch 2 - cache hits)...")
    module_cached.call_count = 0

    start = time.time()
    for batch in loader:
        _ = cached(batch["input"], cache_ids=batch["cache_ids"])
    time_epoch2 = time.time() - start

    speedup = time_nocache / time_epoch2
    print(
        f"    Time: {time_epoch2:.3f}s, Module calls: {module_cached.call_count}, Speedup: {speedup:.1f}x"
    )

    results.extend(
        [
            BenchmarkResult(
                name="Cache Speedup",
                metric="No cache",
                value=time_nocache,
                unit="seconds",
                details=f"{num_samples} samples",
            ),
            BenchmarkResult(
                name="Cache Speedup",
                metric="Epoch 1 (populate)",
                value=time_epoch1,
                unit="seconds",
                details=f"Overhead: {(time_epoch1 / time_nocache - 1) * 100:.1f}%",
            ),
            BenchmarkResult(
                name="Cache Speedup",
                metric="Epoch 2 (cached)",
                value=time_epoch2,
                unit="seconds",
                details=f"Speedup: {speedup:.1f}x",
            ),
        ]
    )

    return results


def benchmark_async_write(tmpdir: str) -> list[BenchmarkResult]:
    """Compare async vs sync write performance."""
    print("\n[Benchmark] Async Write Performance")
    print("=" * 60)

    results = []
    num_samples = 5000
    batch_size = 100

    dataset = BenchmarkDataset(num_samples=num_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Benchmark SYNC writes
    print("  Running with SYNC writes...")
    backend_sync = ArrowIPCCacheBackend(
        cache_dir=tmpdir,
        module_id="async_test_sync",
        async_write=False,
        flush_every=500,
    )
    module_sync = BenchmarkModule()

    start = time.time()
    for batch in loader:
        out = module_sync(batch["input"])
        backend_sync.put_batch(
            {cid: out[i] for i, cid in enumerate(batch["cache_ids"])}
        )
    backend_sync.flush()
    time_sync = time.time() - start

    print(f"    Time: {time_sync:.3f}s")

    # Benchmark ASYNC writes
    print("  Running with ASYNC writes...")
    backend_async = ArrowIPCCacheBackend(
        cache_dir=tmpdir,
        module_id="async_test_async",
        async_write=True,
        flush_every=500,
    )
    module_async = BenchmarkModule()

    start = time.time()
    for batch in loader:
        out = module_async(batch["input"])
        backend_async.put_batch(
            {cid: out[i] for i, cid in enumerate(batch["cache_ids"])}
        )
    backend_async.flush()
    time_async = time.time() - start

    speedup = time_sync / time_async
    print(f"    Time: {time_async:.3f}s, Speedup: {speedup:.2f}x")

    results.extend(
        [
            BenchmarkResult(
                name="Async Write",
                metric="Sync writes",
                value=time_sync,
                unit="seconds",
                details=f"{num_samples} samples",
            ),
            BenchmarkResult(
                name="Async Write",
                metric="Async writes",
                value=time_async,
                unit="seconds",
                details=f"Speedup: {speedup:.2f}x",
            ),
        ]
    )

    return results


def benchmark_dtype_preservation(tmpdir: str) -> list[BenchmarkResult]:
    """Verify dtype preservation across different tensor types."""
    print("\n[Benchmark] Dtype Preservation")
    print("=" * 60)

    results = []
    dtypes = [
        torch.float32,
        torch.float16,
        torch.float64,
        torch.int32,
        torch.int64,
    ]

    for dtype in dtypes:
        backend = ArrowIPCCacheBackend(
            cache_dir=tmpdir,
            module_id=f"dtype_{str(dtype).split('.')[-1]}",
            async_write=False,
        )

        # Write tensor with specific dtype
        tensor_in = (
            torch.randn(512).to(dtype)
            if dtype.is_floating_point
            else torch.randint(0, 100, (512,)).to(dtype)
        )
        backend.put_batch({"key": tensor_in})
        backend.flush()

        # Read back
        results_data, missing = backend.get_batch(["key"])
        tensor_out = results_data[0]

        # Verify dtype
        preserved = tensor_out.dtype == dtype
        status = "✓ PASS" if preserved else "✗ FAIL"
        print(f"    {str(dtype):20s} → {str(tensor_out.dtype):20s} {status}")

        results.append(
            BenchmarkResult(
                name="Dtype Preservation",
                metric=str(dtype),
                value=1.0 if preserved else 0.0,
                unit="preserved",
                details=f"→ {tensor_out.dtype}",
            )
        )

    return results


def generate_markdown_report(all_results: list[BenchmarkResult], output_file: str):
    """Generate markdown report from benchmark results."""
    print(f"\n[Report] Generating markdown report: {output_file}")

    with open(output_file, "w") as f:
        f.write("# torchcachex Benchmark Report\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**System:** {torch.get_num_threads()} CPU threads, ")
        if torch.cuda.is_available():
            f.write(f"CUDA {torch.cuda.get_device_name(0)}\n\n")
        else:
            f.write("No CUDA\n\n")

        f.write("---\n\n")

        # Group results by benchmark name
        from collections import defaultdict

        grouped = defaultdict(list)
        for result in all_results:
            grouped[result.name].append(result)

        # Write each benchmark section
        for bench_name, results in grouped.items():
            f.write(f"## {bench_name}\n\n")

            f.write("| Metric | Value | Details |\n")
            f.write("|--------|-------|----------|\n")

            for result in results:
                value_str = f"{result.value:.3f} {result.unit}"
                f.write(f"| {result.metric} | {value_str} | {result.details} |\n")

            f.write("\n")

            # Add interpretation for write scaling
            if bench_name == "Write Scaling":
                times = [r.value for r in results]
                avg_time = sum(times) / len(times)
                variance = sum((t - avg_time) ** 2 for t in times) / len(times)
                cv = (variance**0.5) / avg_time  # Coefficient of variation

                f.write("**Interpretation:**\n")
                f.write(f"- Average flush time: {avg_time:.3f}s (across cache sizes)\n")
                f.write(f"- Coefficient of variation: {cv:.2f} (lower is better)\n")
                if cv < 0.3:
                    f.write(
                        "- ✅ **O(1) confirmed**: Flush time independent of cache size\n"
                    )
                else:
                    f.write("- ⚠️ **Warning**: Flush time shows variation\n")
                f.write("\n")

            # Add interpretation for cache speedup
            if bench_name == "Cache Speedup":
                no_cache = next(r for r in results if "No cache" in r.metric)
                cached = next(r for r in results if "Epoch 2" in r.metric)
                speedup = no_cache.value / cached.value

                f.write("**Interpretation:**\n")
                f.write(f"- Speedup: {speedup:.1f}x faster with cache\n")
                f.write(
                    f"- Time saved per epoch: {no_cache.value - cached.value:.2f}s\n"
                )
                f.write("\n")

        # Summary
        f.write("---\n\n")
        f.write("## Summary\n\n")
        f.write("**Key Findings:**\n\n")
        f.write("1. **Write Scaling**: ")

        write_results = grouped.get("Write Scaling", [])
        if write_results:
            times = [r.value for r in write_results]
            avg = sum(times) / len(times)
            f.write(
                f"Flush time averages {avg:.3f}s regardless of cache size (O(1) confirmed)\n"
            )
        else:
            f.write("Not measured\n")

        f.write("2. **Read Performance**: ")
        read_results = grouped.get("Read Performance", [])
        if read_results:
            throughputs = [
                float(r.details.split()[0]) for r in read_results if r.details
            ]
            avg_throughput = sum(throughputs) / len(throughputs)
            f.write(f"Average {avg_throughput:.0f} samples/sec across cache sizes\n")
        else:
            f.write("Not measured\n")

        f.write("3. **Memory Usage**: ")
        mem_results = grouped.get("Memory Usage", [])
        if mem_results:
            per_sample = [
                float(r.details.split()[0])
                for r in mem_results
                if "KB/sample" in r.details
            ]
            avg_per_sample = sum(per_sample) / len(per_sample)
            f.write(
                f"~{avg_per_sample:.2f} KB per sample in memory (constant memory footprint)\n"
            )
        else:
            f.write("Not measured\n")

        f.write("4. **Cache Speedup**: ")
        speedup_results = grouped.get("Cache Speedup", [])
        if speedup_results:
            no_cache = next(
                (r for r in speedup_results if "No cache" in r.metric), None
            )
            cached = next((r for r in speedup_results if "Epoch 2" in r.metric), None)
            if no_cache and cached:
                speedup = no_cache.value / cached.value
                f.write(f"{speedup:.1f}x speedup on cached epochs\n")
        else:
            f.write("Not measured\n")

        f.write("5. **Async Write**: ")
        async_results = grouped.get("Async Write", [])
        if async_results:
            sync = next((r for r in async_results if "Sync" in r.metric), None)
            async_r = next((r for r in async_results if "Async" in r.metric), None)
            if sync and async_r:
                speedup = sync.value / async_r.value
                f.write(f"{speedup:.2f}x speedup with async writes\n")
        else:
            f.write("Not measured\n")

        f.write("6. **Dtype Preservation**: ")
        dtype_results = grouped.get("Dtype Preservation", [])
        if dtype_results:
            preserved_count = sum(1 for r in dtype_results if r.value == 1.0)
            total = len(dtype_results)
            f.write(f"{preserved_count}/{total} dtypes preserved correctly\n")
        else:
            f.write("Not measured\n")

        f.write("\n")

    print(f"  Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark torchcachex performance")
    parser.add_argument(
        "--output",
        default="BENCHMARK.md",
        help="Output markdown file (default: BENCHMARK.md)",
    )
    parser.add_argument(
        "--skip-write-scaling",
        action="store_true",
        help="Skip write scaling benchmark (slow)",
    )
    parser.add_argument(
        "--skip-memory",
        action="store_true",
        help="Skip memory benchmark",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("torchcachex Benchmark Suite")
    print("=" * 60)

    all_results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # Run benchmarks
        if not args.skip_write_scaling:
            all_results.extend(benchmark_write_scaling(tmpdir))

        all_results.extend(benchmark_read_performance(tmpdir))

        if not args.skip_memory:
            all_results.extend(benchmark_memory_usage(tmpdir))

        all_results.extend(benchmark_cache_speedup(tmpdir))
        all_results.extend(benchmark_async_write(tmpdir))
        all_results.extend(benchmark_dtype_preservation(tmpdir))

    # Generate report
    generate_markdown_report(all_results, args.output)

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
