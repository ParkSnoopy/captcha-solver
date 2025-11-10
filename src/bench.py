# src/bench.py
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

# 로컬 레포의 모듈 사용 (train.py/evalu.py와 동일한 import 경로 가정)
from model import CaptchaModelV22


# -----------------------------
# FLOPs 계산용 훅 (Conv2d / Linear)
# -----------------------------
def _count_conv2d(m: nn.Conv2d, x, y):
    # x: (N, Cin, Hin, Win), y: (N, Cout, Hout, Wout)
    x = x[0]
    N, Cin, Hin, Win = x.shape
    N, Cout, Hout, Wout = y.shape
    kh, kw = (
        m.kernel_size
        if isinstance(m.kernel_size, tuple)
        else (m.kernel_size, m.kernel_size)
    )
    groups = m.groups
    # MACs = Cout * Hout * Wout * (Cin/groups) * kh * kw * N
    macs = Cout * Hout * Wout * (Cin // groups) * kh * kw * N
    # FLOPs ≈ 2 * MACs (multiply + add), bias add는 보통 무시
    return 2 * macs


def _count_linear(m: nn.Linear, x, y):
    # x: (N, Fin), y: (N, Fout)
    x = x[0]
    N = x.shape[0]
    Fin = m.in_features
    Fout = m.out_features
    macs = N * Fin * Fout
    return 2 * macs


def compute_flops(train_args, model: nn.Module, input_shape, device="cpu"):
    # 모듈별 FLOPs 합산을 위한 훅 등록
    flops_total = 0

    def make_hook(counter_fn):
        def _hook(m, x, y):
            nonlocal flops_total
            try:
                flops_total += counter_fn(m, x, y)
            except Exception:
                pass

        return _hook

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(make_hook(_count_conv2d)))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(make_hook(_count_linear)))
        # BatchNorm/GELU/Dropout 등은 FLOPs ~0으로 간주

    model.eval()
    dummy = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        _ = model(dummy)

    for h in hooks:
        h.remove()

    return flops_total  # 정수 (FLOPs)


# -----------------------------
# 레이턴시 측정
# -----------------------------
def measure_latency(
    train_args, model: nn.Module, device="cpu", batch_size=1, warmup=20, runs=200
):
    model.eval()
    x = torch.randn(batch_size, 3, train_args.height, train_args.width, device=device)
    timings = []

    # sync util
    def _sync():
        if device == "cuda":
            torch.cuda.synchronize()

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            _sync()

    # measure
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            _sync()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # ms

    timings.sort()
    avg = sum(timings) / len(timings)
    p50 = timings[len(timings) // 2]
    p90 = timings[int(len(timings) * 0.90)]
    p95 = timings[int(len(timings) * 0.95)]
    return {
        "bs": batch_size,
        "runs": runs,
        "avg_ms": avg,
        "p50_ms": p50,
        "p90_ms": p90,
        "p95_ms": p95,
    }


# -----------------------------
# 파라미터 카운트
# -----------------------------
def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# -----------------------------
# 체크포인트 로드 & 모델 구성
# -----------------------------
def sanitize_blocks(blocks):
    # CaptchaModelV22.__init__에서 blocks.insert(0, 3)를 수행함.
    # 저장된 config에 3이 이미 포함돼 있으면 중복을 피하도록 제거.
    if len(blocks) > 0 and blocks[0] == 3:
        return blocks[1:]
    return blocks


def find_latest_ckpt(ckpt_dir: Path):
    cands = sorted(ckpt_dir.glob("*.pth"))
    if not cands:
        raise FileNotFoundError(f"No checkpoints under: {ckpt_dir}")
    # 파일명에 타임스탬프가 포함되므로 사전순 정렬로도 최신이 뒤쪽일 가능성 큼
    return cands[-1]


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Benchmark CaptchaModelV22")
    ap.add_argument("--ckpt", type=str, default=None, help="checkpoint path (.pth)")
    ap.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
    )
    ap.add_argument("--bs", type=int, default=1, help="batch size for latency")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    ckpt_path = (
        Path(args.ckpt) if args.ckpt else find_latest_ckpt(Path("./checkpoints"))
    )
    state = torch.load(str(ckpt_path), weights_only=False)
    train_conf = state["model_config"]
    train_args = state["cli_args"]
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    device = torch.device("cuda" if args.use_gpu else "cpu")
    print(f"[INFO] Device: {device}")

    model = model = CaptchaModelV22(**train_conf).to(device)

    # Params
    total_params, trainable_params = count_parameters(model)
    print(f"[PARAMS] total={total_params:,d}  trainable={trainable_params:,d}")

    # FLOPs (bs=1 입력 기준)
    flops = compute_flops(
        train_args=train_args,
        model=model,
        input_shape=(1, 3, train_args.height, train_args.width),
        device=device,
    )
    gflops = flops / 1e9
    print(
        f"[FLOPS] bs=1 input ({3}x{train_args.height}x{train_args.width}) → {gflops:.3f} GFLOPs"
    )

    # Latency
    lat = measure_latency(
        train_args=train_args,
        model=model,
        device=device,
        batch_size=args.bs,
        warmup=args.warmup,
        runs=args.runs,
    )
    print(f"[LATENCY] bs={lat['bs']}  runs={lat['runs']}")
    print(
        f"  avg={lat['avg_ms']:.3f} ms  p50={lat['p50_ms']:.3f} ms  p90={lat['p90_ms']:.3f} ms  p95={lat['p95_ms']:.3f} ms"
    )


if __name__ == "__main__":
    main()
