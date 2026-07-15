# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "altair==5.5.0",
#   "marimo==0.23.14",
#   "pandas==2.3.3",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(
    width="full",
    app_title="LeWorldModel: a 96% PushT reproduction",
)


@app.cell
def _():
    import math

    import altair as alt
    import marimo as mo
    import pandas as pd

    return alt, math, mo, pd


@app.cell
def _(mo):
    mo.md(r"""
    # Can a 15M-parameter world model really solve 96% of PushT goals?

    This executable tutorial tests one clear claim from
    [**LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels**](https://arxiv.org/abs/2603.19312):
    the released pixel-only LeWM reaches **96% success on PushT**.

    The answer is **yes—for the released-checkpoint evaluation**. Our supervised
    ORX run solved **48 of 50 reachable goals (96%)**, exactly matching the
    paper's point estimate. The notebook then shows why that headline needs an
    anti-collapse check: a jointly trained encoder can make prediction trivial
    by mapping every frame to nearly the same vector.

    > **Evidence boundary.** Frozen ORX measurements below are reproduction
    evidence. The optional GPU lab at the end is a synthetic teaching experiment,
    not an additional paper result.
    """)
    return


@app.cell
def _(math, pd):
    headline = pd.DataFrame(
        [
            {"source": "Paper (Figure 6)", "success_rate": 96.0, "successes": None, "trials": None},
            {"source": "ORX reproduction", "success_rate": 96.0, "successes": 48, "trials": 50},
        ]
    )

    def wilson_interval(successes, trials, z=1.959963984540054):
        proportion = successes / trials
        denominator = 1 + z * z / trials
        center = (proportion + z * z / (2 * trials)) / denominator
        margin = z * math.sqrt(
            proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
        ) / denominator
        return 100 * (center - margin), 100 * (center + margin)

    reproduced_ci = wilson_interval(48, 50)
    return headline, reproduced_ci


@app.cell
def _(alt, headline, mo, reproduced_ci):
    headline_chart = (
        alt.Chart(headline)
        .mark_bar(cornerRadiusEnd=4, size=34)
        .encode(
            x=alt.X("success_rate:Q", title="Goal success (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("source:N", title=None, sort=["Paper (Figure 6)", "ORX reproduction"]),
            color=alt.Color("source:N", legend=None),
            tooltip=["source:N", alt.Tooltip("success_rate:Q", format=".1f")],
        )
        .properties(height=130, title="Like-for-like PushT result")
    )
    mo.vstack(
        [
            mo.md("## 1. Paper result and reproduction verdict"),
            mo.hstack(
                [
                    mo.stat(value="96%", label="Paper", caption="Figure 6 · PushT", bordered=True),
                    mo.stat(value="48 / 50", label="Reproduced", caption="96% goal success", bordered=True),
                    mo.stat(
                        value=f"{reproduced_ci[0]:.1f}–{reproduced_ci[1]:.1f}%",
                        label="95% Wilson interval",
                        caption="finite-sample uncertainty",
                        bordered=True,
                    ),
                ],
                widths="equal",
            ),
            mo.ui.altair_chart(headline_chart),
            mo.callout(
                mo.md(
                    "**Verdict: reproduced (released-checkpoint evaluation).** The run used the authors' "
                    "released PushT checkpoint, evaluation seed 42, 50 reachable goals, CEM with 300 "
                    "candidates / 30 elites / 30 iterations, horizon 5, and action block 5. Evaluation "
                    "took 35.18 s inside a 3 min 13 s one-GPU job."
                ),
                kind="success",
            ),
        ]
    )
    return


@app.cell
def _(pd):
    collapse_rows = []
    collapse_trajectories = {
        "No SIGReg (λ=0)": {
            "Prediction loss": [0.068495, 6.24628e-5, 9.92368e-6, 5.33978e-6, 3.34471e-6, 2.61433e-6, 2.32247e-6, 1.84505e-6],
            "Gaussianity diagnostic": [55.5, 51.4943, 51.4943, 51.4943, 51.4943, 51.4943, 51.4943, 51.4943],
        },
        "SIGReg (λ=0.05)": {
            "Prediction loss": [0.068495, 0.0161022, 0.00923039, 0.00642990, 0.00538097, 0.00437342, 0.00371254, 0.00302290],
            "Gaussianity diagnostic": [55.5, 2.2594, 1.7473, 1.5726, 1.5261, 1.4797, 1.4483, 1.4220],
        },
    }
    for variant_name, metric_map in collapse_trajectories.items():
        for metric_name, metric_values in metric_map.items():
            for epoch_number, metric_value in enumerate(metric_values):
                collapse_rows.append(
                    {
                        "variant": variant_name,
                        "metric": metric_name,
                        "epoch": epoch_number,
                        "value": metric_value,
                    }
                )
    collapse_data = pd.DataFrame(collapse_rows)
    seed_data = pd.DataFrame(
        [
            {"seed": "3072", "epoch7_prediction_loss": 0.00428108},
            {"seed": "123", "epoch7_prediction_loss": 0.00423116},
            {"seed": "42", "epoch7_prediction_loss": 0.00443598},
        ]
    )
    return collapse_data, seed_data


@app.cell
def _(mo):
    inspect_epoch = mo.ui.slider(
        start=0,
        stop=7,
        step=1,
        value=7,
        label="Inspect completed epoch",
        show_value=True,
    )
    inspect_epoch
    return (inspect_epoch,)


@app.cell
def _(alt, collapse_data, inspect_epoch, mo):
    selected_epoch = collapse_data[collapse_data["epoch"] == inspect_epoch.value]
    collapse_chart = (
        alt.Chart(collapse_data)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("epoch:Q", title="Completed epoch", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("value:Q", title=None, scale=alt.Scale(type="log")),
            color=alt.Color("variant:N", title="Training objective"),
            strokeDash=alt.StrokeDash("variant:N", legend=None),
            tooltip=[
                "variant:N",
                "metric:N",
                "epoch:Q",
                alt.Tooltip("value:Q", format=".6g"),
            ],
        )
        .properties(width=330, height=245)
        .facet(column=alt.Column("metric:N", title=None))
        .properties(title="Low prediction error can be a collapse symptom")
    )
    selected_table = selected_epoch.pivot(index="variant", columns="metric", values="value").reset_index()
    mo.vstack(
        [
            mo.md("## 2. Why the anti-collapse term matters"),
            mo.md(
                "A prediction-only JEPA has a shortcut: encode every image as the same vector. "
                "The prediction error then looks excellent even though the latent contains no usable state. "
                "SIGReg instead pushes random one-dimensional projections of the latent toward a standard Gaussian."
            ),
            mo.ui.altair_chart(collapse_chart),
            mo.md(f"**Snapshot at epoch {inspect_epoch.value}.**"),
            mo.ui.table(selected_table, selection=None),
            mo.callout(
                mo.md(
                    "At epoch 7, λ=0 reduced prediction error **37,124×** from initialization, "
                    "yet its Gaussianity diagnostic remained **51.49**. With λ=0.05, the diagnostic "
                    "fell to **1.422** while prediction learning continued. A useful falsifier is simple: "
                    "if λ=0 retained latent scale and downstream control as well as the regularized run, "
                    "the anti-collapse interpretation would fail."
                ),
                kind="warn",
            ),
        ]
    )
    return


@app.cell
def _(alt, mo, seed_data):
    seed_chart = (
        alt.Chart(seed_data)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "epoch7_prediction_loss:Q",
                title="Epoch-7 validation prediction loss",
                scale=alt.Scale(domain=[0, 0.005]),
            ),
            y=alt.Y("seed:N", title="Seed"),
            tooltip=["seed:N", alt.Tooltip("epoch7_prediction_loss:Q", format=".6f")],
        )
        .properties(height=135, title="Three-seed convergence is tightly grouped")
    )
    mo.vstack(
        [
            mo.md("## 3. Robustness, leakage checks, and interpretation"),
            mo.ui.altair_chart(seed_chart),
            mo.ui.table(
                [
                    {
                        "check": "Independent reachable goals",
                        "result": "48/50; no training-goal reuse in the control evaluator",
                    },
                    {
                        "check": "Latent covariance",
                        "result": "diagonal 1.0027 ± 0.1346; off-diagonal RMS 0.0786",
                    },
                    {
                        "check": "Effective rank",
                        "result": "93.5 of 192 dimensions on 8,192 held-out samples",
                    },
                    {
                        "check": "Continuity negative control",
                        "result": "target-swap surprise 2.002 vs normal 0.00818 (paired p≈0)",
                    },
                ],
                selection=None,
            ),
            mo.md(
                "The three PushT training seeds finish epoch 7 within **0.00423–0.00444** prediction loss. "
                "That supports stable optimization, but it does not replace final control evaluation for each seed."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md("## 4. Limitations and provenance"),
            mo.md(
                "The 96% verdict confirms the **released-checkpoint evaluation**, not end-to-end PushT "
                "training from random initialization. The overnight training grid reached seven of the paper's "
                "ten epochs before cancellation, so its loss curves are supporting evidence only. Hardware was "
                "an NVIDIA RTX PRO 6000 rather than the paper's unspecified single-GPU setup. The selected "
                "formal run cost about **0.054 GPU-hours** (3 min 13 s wall time); the control loop itself took 35.18 s."
            ),
            mo.md(
                """
                | Experiment | Exact code and configuration | Verdict | Compute |
                |---|---|---|---|
                | Confirmatory PushT evaluation | [Confirmatory reproduction code](https://github.com/rehaanahmad2013/le-wm-b7f421c1/tree/orx/released-pusht-paper-seed-reference) — runner, fixed evaluation config, k8s manifest, and diagnostics | **Reproduced:** 48/50 = 96% | 1× RTX PRO 6000 · 3m13s job |
                | Prediction-only control | [Collapse-control code](https://github.com/rehaanahmad2013/le-wm-b7f421c1/tree/orx/pusht-lambda-0-0-collapse-control) — same runner with SIGReg weight fixed to zero | **Collapsed:** 37,124× prediction decrease without Gaussianity | 1× RTX PRO 6000 · 7 epochs |
                | Regularized comparison | [λ=0.05 training code](https://github.com/rehaanahmad2013/le-wm-b7f421c1/tree/orx/pusht-lambda-0-05) — fixed training config, manifest, and validation metrics | **Partial:** stable through 7/10 epochs | 1× RTX PRO 6000 · 7 epochs |
                | Optional GPU lab validation | [Validated teaching-lab code](https://github.com/rehaanahmad2013/le-wm-b7f421c1/tree/orx/molab-gpu-lab-validated-variance-surrogate) — synthetic runner, fixed config, one-GPU manifest, and evaluation | **Validated teaching path; not reproduction evidence** | 1× RTX PRO 6000 · 19.63s workload |
                """
            ),
            mo.callout(
                mo.md(
                    "A full claim reproduction would still retrain PushT for all 10 paper epochs across multiple "
                    "seeds, run 50-goal control evaluation for every final checkpoint, and compare planning speed "
                    "against PLDM and DINO-WM under matched hardware."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.function
def run_mini_jepa_lab(
    selected_lambda=0.08,
    steps=900,
    batch_size=512,
    image_size=96,
    latent_dim=64,
    seed=3072,
):
    import time

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
        def __init__(self):
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
                nn.Flatten(),
                nn.Linear(96 * 4 * 4, 256),
                nn.GELU(),
                nn.Linear(256, latent_dim),
            )

        def forward(self, image):
            return self.project(self.features(image))

    class Predictor(nn.Module):
        def __init__(self):
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

    def make_batch(number):
        position = 0.2 + 0.6 * torch.rand(number, 2, device=device)
        action = 0.20 * (torch.rand(number, 2, device=device) - 0.5)
        next_position = (position + action).clamp(0.08, 0.92)
        return render(position), action, render(next_position)

    projections = torch.randn(48, latent_dim, device=device)
    projections = F.normalize(projections, dim=1)
    frequencies = torch.tensor((0.5, 1.0, 1.5, 2.0), device=device)

    def gaussian_projection_loss(z):
        projected = z @ projections.T
        centered = projected - projected.mean(0, keepdim=True)
        std = (centered.square().mean(0) + 1e-4).sqrt()
        normalized = centered / std[None, :]
        covariance = normalized.T @ normalized / max(len(normalized) - 1, 1)
        off_diagonal = covariance - torch.diag(torch.diag(covariance))
        angles = normalized[:, :, None] * frequencies
        real = torch.cos(angles).mean(0)
        imag = torch.sin(angles).mean(0)
        target = torch.exp(-0.5 * frequencies.square())[None, :]
        characteristic = (real - target).square().mean() + imag.square().mean()
        variance = F.relu(1.0 - std).mean()
        decorrelation = off_diagonal.square().mean()
        mean = projected.mean(0).square().mean()
        return 2.0 * variance + 0.10 * decorrelation + 0.10 * characteristic + 0.01 * mean

        @torch.no_grad()
        def evaluate(encoder, predictor):
            image, action, next_image = make_batch(max(1024, effective_batch))
            z = encoder(image)
            next_z = encoder(next_image)
            prediction = predictor(z, action)
            centered = next_z - next_z.mean(0, keepdim=True)
            covariance = centered.T @ centered / max(len(centered) - 1, 1)
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(1e-12)
            probabilities = eigenvalues / eigenvalues.sum()
            effective_rank = torch.exp(-(probabilities * probabilities.log()).sum())
            return {
                "prediction_mse": float(F.mse_loss(prediction, next_z)),
                "gaussian_projection_loss": float(gaussian_projection_loss(next_z)),
                "latent_std": float(next_z.std(0).mean()),
                "effective_rank": float(effective_rank),
                "latent_dim": latent_dim,
            }

    def train_variant(weight):
        torch.manual_seed(seed)
        encoder = Encoder().to(device)
        predictor = Predictor().to(device)
        parameters = list(encoder.parameters()) + list(predictor.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-4)
        started = time.perf_counter()
        trajectory = []
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
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            if step in marks:
                trajectory.append(
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
                "trajectory": trajectory,
            }
        )
        return metrics

    started = time.perf_counter()
    variants = [train_variant(0.0), train_variant(float(selected_lambda))]
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
            "total_runtime_seconds": time.perf_counter() - started,
            "variants": variants,
    }


@app.cell
def _(mo):
    lab_lambda = mo.ui.slider(
        start=0.02,
        stop=0.20,
        step=0.02,
        value=0.08,
        label="Projection regularizer λ",
        show_value=True,
    )
    lab_steps = mo.ui.slider(
        start=300,
        stop=1200,
        step=300,
        value=900,
        label="GPU training steps",
        show_value=True,
    )
    lab_run = mo.ui.run_button(label="Run the mini-JEPA GPU lab")
    mo.vstack(
        [
            mo.md("## 5. Optional interactive GPU lab"),
            mo.callout(
                mo.md(
                    "**Synthetic teaching experiment—not reproduction evidence.** Train two compact "
                    "convolutional JEPAs on procedurally generated 96×96 moving-blob frames: a λ=0 control "
                    "and your selected projection-regularized variant. Molab supplies PyTorch; no model "
                    "download is needed. The expensive work starts only when you press the button."
                ),
                kind="warn",
            ),
            mo.hstack([lab_lambda, lab_steps, lab_run], widths="equal"),
            mo.md(
                "On the validated RTX PRO 6000 path, the default 900-step comparison took **19.63 s**. "
                "It produced latent standard deviation **0.000061** for λ=0 versus **0.578** for λ=0.08. "
                "If CUDA is unavailable, the notebook reports the CPU fallback and caps work at 30 steps / batch 32."
            ),
        ]
    )
    return lab_lambda, lab_run, lab_steps


@app.cell
def _(alt, lab_lambda, lab_run, lab_steps, mo, pd):
    if lab_run.value:
        try:
            lab_result = run_mini_jepa_lab(
                selected_lambda=lab_lambda.value,
                steps=lab_steps.value,
            )
            lab_metrics = pd.DataFrame(lab_result["variants"])
            lab_long = lab_metrics.melt(
                id_vars=["lambda"],
                value_vars=["latent_std", "gaussian_projection_loss", "prediction_mse"],
                var_name="metric",
                value_name="value",
            )
            lab_chart = (
                alt.Chart(lab_long)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("lambda:N", title="Regularizer λ"),
                    y=alt.Y("value:Q", title=None, scale=alt.Scale(type="log")),
                    color=alt.Color("lambda:N", legend=None),
                    tooltip=["metric:N", "lambda:N", alt.Tooltip("value:Q", format=".6g")],
                )
                .properties(width=190, height=190)
                .facet(column=alt.Column("metric:N", title=None))
                .properties(title="What changed after joint training?")
            )
            lab_output = mo.vstack(
                [
                    mo.callout(
                        mo.md(
                            f"Selected device: **{lab_result['device_name']}** · CUDA "
                            f"**{lab_result['cuda_available']}** · runtime "
                            f"**{lab_result['total_runtime_seconds']:.2f} s** · "
                            f"{lab_result['executed_steps']} steps / batch {lab_result['executed_batch_size']}."
                        ),
                        kind="success" if lab_result["cuda_available"] else "info",
                    ),
                    mo.ui.altair_chart(lab_chart),
                    mo.ui.table(
                        lab_metrics[
                            [
                                "lambda",
                                "prediction_mse",
                                "gaussian_projection_loss",
                                "latent_std",
                                "runtime_seconds",
                            ]
                        ],
                        selection=None,
                    ),
                ]
            )
        except Exception as lab_error:
            lab_output = mo.callout(
                mo.md(
                    f"The lab could not start: `{type(lab_error).__name__}: {lab_error}`. "
                    "Molab's attached GPU image must provide a CUDA-enabled PyTorch installation."
                ),
                kind="danger",
            )
    else:
        lab_output = mo.callout(
            mo.md("Ready. Adjust λ or the step budget, then press **Run the mini-JEPA GPU lab**."),
            kind="info",
        )
    lab_output
    return


if __name__ == "__main__":
    app.run()
