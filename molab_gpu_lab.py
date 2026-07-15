"""Bounded synthetic GPU lab used by the Molab tutorial.

This is a teaching experiment, not evidence for the paper reproduction.  It
keeps the LeWM failure mode—jointly training an encoder and predictor makes a
constant representation an easy solution—and uses a compact differentiable
random-projection Gaussianity penalty inspired by SIGReg.
"""

from __future__ import annotations

import argparse
import json
import math
import time


def run_gpu_lab(
    selected_lambda: float = 0.08,
    steps: int = 240,
    batch_size: int = 256,
    image_size: int = 64,
    latent_dim: int = 64,
    seed: int = 3072,
) -> dict:
    import torch
    from torch import nn
    from torch.nn import functional as F

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        effective_steps = int(steps)
        effective_batch = int(batch_size)
    else:
        device = torch.device("cpu")
        device_name = "CPU fallback"
        effective_steps = min(int(steps), 30)
        effective_batch = min(int(batch_size), 32)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    grid = torch.linspace(0.0, 1.0, image_size, device=device)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    xx = xx[None, None]
    yy = yy[None, None]

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2),
                nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 96, 3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.project = nn.Sequential(
                nn.Flatten(), nn.Linear(96 * 4 * 4, 256), nn.GELU(), nn.Linear(256, latent_dim)
            )

        def forward(self, image):
            return self.project(self.features(image))

    class Predictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim + 2, 256),
                nn.GELU(),
                nn.Linear(256, 256),
                nn.GELU(),
                nn.Linear(256, latent_dim),
            )

        def forward(self, z, action):
            return self.net(torch.cat((z, action), dim=-1))

    def render(position):
        px = position[:, 0, None, None, None]
        py = position[:, 1, None, None, None]
        sigma2 = 2.0 * 0.055**2
        blob = torch.exp(-((xx - px) ** 2 + (yy - py) ** 2) / sigma2)
        mirror = torch.exp(-((xx - (1.0 - px)) ** 2 + (yy - py) ** 2) / sigma2)
        return torch.cat((blob, mirror, 0.45 * blob + 0.25 * xx.expand_as(blob)), dim=1)

    def make_batch(n):
        position = 0.2 + 0.6 * torch.rand(n, 2, device=device)
        action = 0.20 * (torch.rand(n, 2, device=device) - 0.5)
        next_position = (position + action).clamp(0.08, 0.92)
        return render(position), action, render(next_position)

    projections = torch.randn(48, latent_dim, device=device)
    projections = F.normalize(projections, dim=1)
    frequencies = torch.tensor((0.5, 1.0, 1.5, 2.0), device=device)

    def gaussian_projection_loss(z):
        projected = z @ projections.T
        angles = projected[:, :, None] * frequencies
        real = torch.cos(angles).mean(0)
        imag = torch.sin(angles).mean(0)
        target = torch.exp(-0.5 * frequencies.square())[None, :]
        return (real - target).square().mean() + imag.square().mean()

    @torch.no_grad()
    def evaluate(encoder, predictor):
        image, action, next_image = make_batch(max(1024, effective_batch))
        z = encoder(image)
        next_z = encoder(next_image)
        pred = predictor(z, action)
        centered = next_z - next_z.mean(0, keepdim=True)
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(1e-12)
        probabilities = eigenvalues / eigenvalues.sum()
        effective_rank = torch.exp(-(probabilities * probabilities.log()).sum())
        return {
            "prediction_mse": float(F.mse_loss(pred, next_z)),
            "gaussian_projection_loss": float(gaussian_projection_loss(next_z)),
            "latent_std": float(next_z.std(0).mean()),
            "effective_rank": float(effective_rank),
            "latent_dim": latent_dim,
        }

    def train_variant(weight):
        torch.manual_seed(seed)
        encoder = Encoder().to(device)
        predictor = Predictor().to(device)
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(predictor.parameters()), lr=3e-4, weight_decay=1e-4
        )
        started = time.perf_counter()
        checkpoints = []
        marks = {1, max(1, effective_steps // 4), max(1, effective_steps // 2), effective_steps}
        for step in range(1, effective_steps + 1):
            image, action, next_image = make_batch(effective_batch)
            z = encoder(image)
            next_z = encoder(next_image)
            prediction = predictor(z, action)
            prediction_loss = F.mse_loss(prediction, next_z)
            regularizer = gaussian_projection_loss(torch.cat((z, next_z), dim=0))
            loss = prediction_loss + weight * regularizer
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(predictor.parameters()), 5.0)
            optimizer.step()
            if step in marks:
                checkpoints.append(
                    {
                        "step": step,
                        "prediction_mse": float(prediction_loss.detach()),
                        "gaussian_projection_loss": float(regularizer.detach()),
                    }
                )
        if device.type == "cuda":
            torch.cuda.synchronize()
        metrics = evaluate(encoder, predictor)
        metrics.update(
            {
                "lambda": float(weight),
                "steps": effective_steps,
                "batch_size": effective_batch,
                "runtime_seconds": time.perf_counter() - started,
                "trajectory": checkpoints,
            }
        )
        return metrics

    started = time.perf_counter()
    variants = [train_variant(0.0), train_variant(float(selected_lambda))]
    total_runtime = time.perf_counter() - started
    return {
        "label": "synthetic teaching experiment; not reproduction evidence",
        "device": str(device),
        "device_name": device_name,
        "cuda_available": torch.cuda.is_available(),
        "requested_steps": int(steps),
        "executed_steps": effective_steps,
        "requested_batch_size": int(batch_size),
        "executed_batch_size": effective_batch,
        "image_size": int(image_size),
        "total_runtime_seconds": total_runtime,
        "variants": variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-lambda", type=float, default=0.08)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3072)
    args = parser.parse_args()
    result = run_gpu_lab(
        selected_lambda=args.selected_lambda,
        steps=args.steps,
        batch_size=args.batch_size,
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    for variant in result["variants"]:
        print(
            "ORX_METRIC "
            + json.dumps(
                {
                    "kind": "molab_gpu_lab",
                    "device": result["device_name"],
                    "total_runtime_seconds": result["total_runtime_seconds"],
                    **{key: value for key, value in variant.items() if key != "trajectory"},
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
