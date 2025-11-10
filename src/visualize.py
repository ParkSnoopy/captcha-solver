# visualize.py
# Usage:
#   python visualize.py \
#     --model ./checkpoints/20250101_1200_epoch10_on_v2.2.pth \
#     --data  ./dist \
#     --num 1000 \
#     --examples 24 \
#     --save-dir ./viz \
#     --use-gpu

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

# local modules (repo 내부 파일)
from helper import TRANSFORM, fit_image, I2C, C2I
from config import CHARSET, MAX_H, MAX_W, SEED
from train import USE_MODEL  # CaptchaModelV22


# -----------------------------
# Matplotlib (plots)
# -----------------------------
import matplotlib

matplotlib.use("Agg")  # 파일 저장용 백엔드
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Visualize results of captcha model")
    ap.add_argument("-m", "--model", type=Path, required=True, help="checkpoint (.pth)")
    ap.add_argument("-d", "--data", type=Path, required=True, help="image dir")
    ap.add_argument("-n", "--num", type=int, default=500, help="num images to sample")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--use-gpu", action="store_true", default=False)
    ap.add_argument("--save-dir", type=Path, default=Path("./viz"))
    ap.add_argument("--examples", type=int, default=16, help="# of annotated examples")
    ap.add_argument(
        "--grid-cols", type=int, default=8, help="grid columns for examples"
    )
    ap.add_argument("--topk", type=int, default=3, help="top-k to record per position")
    ap.add_argument(
        "--profile-model",
        action="store_true",
        default=True,
        help="measure params/FLOPs/latency and save a txt",
    )
    ap.add_argument("--latency-batch", type=int, default=1)
    ap.add_argument("--latency-runs", type=int, default=200)
    ap.add_argument("--latency-warmup", type=int, default=20)
    return ap.parse_args()


# -----------------------------
# Utils
# -----------------------------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _load_images(data_dir: Path) -> List[Path]:
    paths = [*data_dir.glob("*.png"), *data_dir.glob("*.jpg"), *data_dir.glob("*.jpeg")]
    return sorted(paths)


def _filename_to_label(p: Path) -> str:
    # train/dataset과 동일한 규칙: stem의 첫 '.' 이전까지
    return p.stem.split(".")[0].upper()


def _softmax_logits(logits: torch.Tensor) -> torch.Tensor:
    # [B, L, C] -> softmax over C
    return F.softmax(logits, dim=-1)


def _make_grid(images: List[Image.Image], cols: int) -> Image.Image:
    if not images:
        return None
    w, h = images[0].size
    cols = max(1, cols)
    rows = math.ceil(len(images) / cols)
    grid = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(im, (c * w, r * h))
    return grid


def _annotate(img: Image.Image, text: str, ok: bool) -> Image.Image:
    # 간단한 상단 바와 텍스트 박스
    im = img.copy().convert("RGB")
    draw = ImageDraw.Draw(im)
    W, H = im.size
    bar_h = max(16, H // 10)
    draw.rectangle([0, 0, W, bar_h], fill=(220, 220, 220))
    label = f"{text}"
    # 시스템 폰트가 보장되지 않으니 기본 폰트 사용
    draw.text((6, 2), label, fill=(0, 128, 0) if ok else (192, 0, 0))
    return im


# -----------------------------
# Optional: profile (params/FLOPs/latency)
# -----------------------------
def _count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _compute_flops(
    model: torch.nn.Module, height: int, width: int, device: torch.device
) -> int:
    flops_total = 0

    def _count_conv2d(m: torch.nn.Conv2d, x, y):
        nonlocal flops_total
        x = x[0]
        N, Cin, Hin, Win = x.shape
        N, Cout, Hout, Wout = y.shape
        kh, kw = (
            m.kernel_size
            if isinstance(m.kernel_size, tuple)
            else (m.kernel_size, m.kernel_size)
        )
        groups = m.groups
        macs = Cout * Hout * Wout * (Cin // groups) * kh * kw * N
        flops_total += 2 * macs

    def _count_linear(m: torch.nn.Linear, x, y):
        nonlocal flops_total
        x = x[0]
        N = x.shape[0]
        macs = N * m.in_features * m.out_features
        flops_total += 2 * macs

    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(_count_conv2d))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(_count_linear))

    model.eval()
    dummy = torch.randn(1, 3, height, width, device=device)
    with torch.no_grad():
        _ = model(dummy)
    for h in hooks:
        h.remove()
    return int(flops_total)


def _measure_latency(model, device, height, width, batch_size=1, warmup=20, runs=200):
    x = torch.randn(batch_size, 3, height, width, device=device)
    times = []

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            _sync()
        for _ in range(runs):
            t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if device.type == "cuda":
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                _ = model(x)
                t1.record()
                torch.cuda.synchronize()
                ms = t0.elapsed_time(t1)
            else:
                import time

                t0f = time.perf_counter()
                _ = model(x)
                _sync()
                t1f = time.perf_counter()
                ms = (t1f - t0f) * 1000.0
            times.append(ms)

    times.sort()
    avg = sum(times) / len(times)
    p50 = times[len(times) // 2]
    p90 = times[int(len(times) * 0.90)]
    p95 = times[int(len(times) * 0.95)]
    return {
        "avg_ms": avg,
        "p50_ms": p50,
        "p90_ms": p90,
        "p95_ms": p95,
        "bs": batch_size,
        "runs": runs,
    }


# -----------------------------
# Main eval/visualize
# -----------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    _ensure_dir(args.save_dir)

    # 1) load checkpoint & model
    ckpt = torch.load(str(args.model), weights_only=False, map_location="cpu")
    model_conf = ckpt["model_config"]
    train_args = ckpt["cli_args"]
    device = torch.device(
        "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    )

    model = USE_MODEL(**model_conf).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # 2) collect images
    paths_all = _load_images(args.data)
    if len(paths_all) == 0:
        raise FileNotFoundError(f"No images in: {args.data}")

    if args.num > 0 and args.num < len(paths_all):
        paths = random.sample(paths_all, k=args.num)
    else:
        paths = paths_all

    L = model_conf["len_captcha"]
    C = len(CHARSET)

    # accumulators
    total = 0
    string_ok = 0
    pos_correct = np.zeros(L, dtype=np.int64)
    pos_total = np.zeros(L, dtype=np.int64)
    confusion = np.zeros((C, C), dtype=np.int64)  # [true, pred]
    conf_correct = []
    conf_incorrect = []

    ex_images_ok, ex_images_ng = [], []

    # 3) iterate
    pbar = tqdm(paths, desc="Eval", unit="img")
    for p in pbar:
        label = _filename_to_label(p)
        # 길이 mismatch 방지
        if len(label) != L:
            continue

        img = Image.open(p)
        img_fit = fit_image(img)
        x = TRANSFORM(img_fit).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)  # [1, L, C]
            prob = _softmax_logits(logits)[0].cpu().numpy()  # [L, C]
            pred_idx = prob.argmax(-1)  # [L]
            pred_str = "".join(I2C[int(i)] for i in pred_idx)

        total += 1
        ok = pred_str.upper() == label.upper()
        if ok:
            string_ok += 1

        # per-position
        for i, (t_char, p_i) in enumerate(zip(label, pred_idx)):
            t_i = C2I[t_char]
            pos_total[i] += 1
            if p_i == t_i:
                pos_correct[i] += 1
                conf_correct.append(prob[i, p_i])
            else:
                conf_incorrect.append(prob[i, p_i])
            confusion[t_i, p_i] += 1

        # Annotated examples
        ann = _annotate(img_fit.resize((MAX_W, MAX_H)), f"{label} -> {pred_str}", ok)
        if ok and len(ex_images_ok) < args.examples // 2:
            ex_images_ok.append(ann)
        elif not ok and len(ex_images_ng) < args.examples // 2:
            ex_images_ng.append(ann)

        acc_live = 100.0 * string_ok / max(1, total)
        pbar.set_postfix(string_acc=f"{acc_live:.2f}%")

    # 4) metrics
    string_acc = 100.0 * string_ok / max(1, total)
    pos_acc = (pos_correct / np.maximum(1, pos_total)).tolist()

    metrics = {
        "total_images": total,
        "string_accuracy_percent": string_acc,
        "position_accuracy_percent": [
            float(a * 100.0) for a in (pos_correct / np.maximum(1, pos_total))
        ],
        "len_captcha": L,
        "charset": CHARSET,
        "checkpoint": str(args.model),
    }
    (args.save_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # 5) plots
    # (a) per-position accuracy
    plt.figure(figsize=(max(6, L), 4))
    plt.title("Per-position accuracy")
    plt.bar(np.arange(L), np.array(pos_acc) * 100.0)
    plt.xlabel("Position index")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()
    plt.savefig(args.save_dir / "per_position_accuracy.png", dpi=160)
    plt.close()

    # (b) confusion matrix
    cm = confusion.astype(np.float64)
    cm_row = cm / np.maximum(1e-9, cm.sum(axis=1, keepdims=True))  # row-normalized
    plt.figure(figsize=(10, 10))
    plt.title("Confusion Matrix (row-normalized)")
    plt.imshow(cm_row, aspect="auto")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=np.arange(C), labels=list(CHARSET), fontsize=6, rotation=90)
    plt.yticks(ticks=np.arange(C), labels=list(CHARSET), fontsize=6)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(args.save_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    # (c) confidence hist (correct vs incorrect)
    plt.figure(figsize=(6, 4))
    plt.title("Top-1 confidence distribution")
    if len(conf_correct) > 0:
        plt.hist(conf_correct, bins=30, alpha=0.6, label="correct")
    if len(conf_incorrect) > 0:
        plt.hist(conf_incorrect, bins=30, alpha=0.6, label="incorrect")
    plt.xlabel("Top-1 probability")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.save_dir / "confidence_hist.png", dpi=160)
    plt.close()

    # (d) examples grid
    ex_images = ex_images_ok + ex_images_ng
    if ex_images:
        grid = _make_grid(ex_images, cols=args.grid_cols)
        if grid is not None:
            grid.save(args.save_dir / "examples_grid.png")

    # 6) optional: model profile
    profile_txt = []
    if args.profile_model:
        H = getattr(train_args, "height", MAX_H)
        W = getattr(train_args, "width", MAX_W)
        total_params, trainable_params = _count_parameters(model)
        profile_txt.append(
            f"[PARAMS] total={total_params:,d}, trainable={trainable_params:,d}"
        )
        try:
            flops = _compute_flops(model, height=H, width=W, device=device)
            profile_txt.append(
                f"[FLOPs] bs=1 input (3x{H}x{W}) -> {flops / 1e9:.3f} GFLOPs"
            )
        except Exception as e:
            profile_txt.append(f"[FLOPs] failed: {e}")

        try:
            lat = _measure_latency(
                model,
                device=device,
                height=H,
                width=W,
                batch_size=args.latency_batch,
                warmup=args.latency_warmup,
                runs=args.latency_runs,
            )
            profile_txt.append(f"[LATENCY] bs={lat['bs']} runs={lat['runs']}")
            profile_txt.append(
                f"  avg={lat['avg_ms']:.3f} ms  p50={lat['p50_ms']:.3f} ms  p90={lat['p90_ms']:.3f} ms  p95={lat['p95_ms']:.3f} ms"
            )
        except Exception as e:
            profile_txt.append(f"[LATENCY] failed: {e}")

    if profile_txt:
        (args.save_dir / "model_profile.txt").write_text(
            "\n".join(profile_txt), encoding="utf-8"
        )

    # 7) console summary
    print("\n=== SUMMARY ===")
    print(f"Total images: {total}")
    print(f"String accuracy: {string_acc:.2f}%")
    print(
        f"Per-position accuracy (%): {[round(x, 2) for x in metrics['position_accuracy_percent']]}"
    )
    if profile_txt:
        print("\n".join(profile_txt))
    print(f"\nSaved figures → {args.save_dir.resolve()}")


if __name__ == "__main__":
    main()
