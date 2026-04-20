from __future__ import annotations

"""Lab 04 (skeleton): Markov Random Field (MRF) image restoration."""

import argparse
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

PenaltyType = Literal["quadratic", "huber"]


def mrf_energy(
    x: np.ndarray,
    y: np.ndarray,
    lambda_smooth: float,
    penalty: PenaltyType = "quadratic",
    huber_delta: float = 1.0,
) -> float:
    """
    Compute pairwise MRF energy for grayscale image restoration.

    Energy:
        E(x) = sum_p (x_p - y_p)^2 + lambda * sum_(p,q in N) rho(x_p - x_q)

    Args:
        x: Restored image candidate `(H,W)`.
        y: Observed noisy image `(H,W)`.
        lambda_smooth: Smoothness weight.
        penalty: `"quadratic"` or `"huber"`.
        huber_delta: Delta parameter for Huber penalty.

    Returns:
        Scalar energy.
    """
    # Data term
    data_term = np.sum((x - y) ** 2)

    # Differences (4-neighborhood)
    dx = x[:, 1:] - x[:, :-1]
    dy = x[1:, :] - x[:-1, :]

    if penalty == "quadratic":
        smooth = np.sum(dx**2) + np.sum(dy**2)

    elif penalty == "huber":
        def huber(z):
            return np.where(
                np.abs(z) <= huber_delta,
                0.5 * z**2,
                huber_delta * (np.abs(z) - 0.5 * huber_delta)
            )
        smooth = np.sum(huber(dx)) + np.sum(huber(dy))

    else:
        raise ValueError("Unknown penalty")

    return float(data_term + lambda_smooth * smooth)


def mrf_denoise(
    y: np.ndarray,
    lambda_smooth: float,
    num_iters: int,
    step: float = 0.1,
    penalty: PenaltyType = "quadratic",
    huber_delta: float = 1.0,
) -> np.ndarray:
    """
    Restore grayscale image by minimizing MRF energy.

    Args:
        y: Observed noisy image `(H,W)`.
        lambda_smooth: Smoothness weight.
        num_iters: Number of optimization iterations.
        step: Optimization step size.
        penalty: `"quadratic"` or `"huber"`.
        huber_delta: Delta parameter for Huber penalty.

    Returns:
        Restored image with the same shape as `y`.
    """
    x = y.copy().astype(np.float32)

    for _ in range(num_iters):

        # Data gradient
        grad = 2 * (x - y)

        # Neighbor differences
        dx_f = np.zeros_like(x)
        dx_b = np.zeros_like(x)
        dy_f = np.zeros_like(x)
        dy_b = np.zeros_like(x)

        dx_f[:, :-1] = x[:, :-1] - x[:, 1:]
        dx_b[:, 1:] = x[:, 1:] - x[:, :-1]
        dy_f[:-1, :] = x[:-1, :] - x[1:, :]
        dy_b[1:, :] = x[1:, :] - x[:-1, :]

        def grad_penalty(d):
            if penalty == "quadratic":
                return d
            elif penalty == "huber":
                return np.where(
                    np.abs(d) <= huber_delta,
                    d,
                    huber_delta * np.sign(d)
                )
            else:
                raise ValueError("Unknown penalty")

        grad += lambda_smooth * (
            grad_penalty(dx_f) +
            grad_penalty(dx_b) +
            grad_penalty(dy_f) +
            grad_penalty(dy_b)
        )

        x = x - step * grad

    return x


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    arr = x.astype(np.float32)
    mn, mx = np.min(arr), np.max(arr)

    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)

    arr = (arr - mn) * 255.0 / (mx - mn)
    return np.clip(arr, 0, 255).astype(np.uint8)


def main() -> int:
    """
    Lab 04 demo (skeleton).

    Expected behavior after implementation:
    - load grayscale image from `./imgs/`
    - add Gaussian noise (deterministic seed)
    - denoise with MRF (quadratic and/or huber)
    - save side-by-side result to `./out/lab04/mrf_denoise.png`
    """
    parser = argparse.ArgumentParser(description="Lab 04 skeleton (implement functions first).")
    parser.add_argument("--img", type=str, default="lenna.png", help="Input image from ./imgs/")
    parser.add_argument("--out", type=str, default="out/lab04", help="Output directory (relative to repo root)")
    args, _ = parser.parse_known_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save_figure(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    try:
      repo_root = Path(__file__).resolve().parents[1]
    except NameError:
      repo_root = Path.cwd()
    imgs_dir = repo_root / "imgs"
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(imgs_dir / args.img), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(imgs_dir / args.img))

    missing: list[str] = []

    try:
        clean = img.astype(np.float32)
        rng = np.random.default_rng(0)
        noisy = clean + rng.normal(0.0, 18.0, size=clean.shape).astype(np.float32)
        noisy = np.clip(noisy, 0.0, 255.0)

        den_quad = mrf_denoise(noisy, lambda_smooth=0.25, num_iters=80, step=0.1, penalty="quadratic")
        den_hub = mrf_denoise(noisy, lambda_smooth=0.25, num_iters=80, step=0.1, penalty="huber", huber_delta=8.0)

        e_noisy_q = mrf_energy(noisy, noisy, lambda_smooth=0.25, penalty="quadratic")
        e_quad = mrf_energy(den_quad, noisy, lambda_smooth=0.25, penalty="quadratic")
        e_noisy_h = mrf_energy(noisy, noisy, lambda_smooth=0.25, penalty="huber", huber_delta=8.0)
        e_hub = mrf_energy(den_hub, noisy, lambda_smooth=0.25, penalty="huber", huber_delta=8.0)

        plt.figure(figsize=(12, 4))
        panels = [
            ("Original", clean),
            ("Noisy (seed=0)", noisy),
            (f"MRF quadratic\nE: {e_noisy_q:.1f} -> {e_quad:.1f}", den_quad),
            (f"MRF huber\nE: {e_noisy_h:.1f} -> {e_hub:.1f}", den_hub),
        ]
        for i, (title, im) in enumerate(panels, start=1):
            plt.subplot(1, 4, i)
            plt.title(title)
            plt.imshow(normalize_to_uint8(im), cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "mrf_denoise.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    if missing:
        (out_dir / "STATUS.txt").write_text(
            "Lab 04 demo is incomplete. Implement the TODO functions in labs/lab04_mrf_restoration.py.\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_dir / 'STATUS.txt'}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
