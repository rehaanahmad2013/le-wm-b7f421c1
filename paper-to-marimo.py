import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full", app_title="LeWorldModel reproduction")


@app.cell
def _():
    import math

    import altair as alt
    import marimo as mo
    import pandas as pd

    return alt, math, mo, pd


@app.cell
def _(alt, mo, pd):
    ablation = pd.DataFrame(
        [
            ["SIGReg weight", 0.00, 0.00006246, 51.4943],
            ["SIGReg weight", 0.01, 0.012691, 4.6153],
            ["SIGReg weight", 0.05, 0.016102, 2.2594],
            ["SIGReg weight", 0.09, 0.025065, 2.3143],
            ["SIGReg weight", 0.10, 0.025154, 2.1534],
            ["SIGReg weight", 0.20, 0.065518, 3.2960],
            ["SIGReg weight", 0.50, 0.156785, 9.4106],
            ["Predictor dropout", 0.00, 0.022017, 1.9751],
            ["Predictor dropout", 0.10, 0.025065, 2.3143],
            ["Predictor dropout", 0.20, 0.025897, 2.1558],
            ["Predictor dropout", 0.50, 0.024467, 2.0745],
        ],
        columns=["sweep", "value", "prediction_loss", "sigreg_loss"],
    )
    ablation_long = ablation.melt(
        id_vars=["sweep", "value"],
        value_vars=["prediction_loss", "sigreg_loss"],
        var_name="metric",
        value_name="validation_value",
    )
    ablation_chart = (
        alt.Chart(ablation_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("value:Q", title="Hyperparameter value"),
            y=alt.Y("validation_value:Q", title=None),
            color=alt.Color("metric:N", title=None),
            tooltip=["sweep:N", "metric:N", "value:Q", alt.Tooltip("validation_value:Q", format=".5f")],
        )
        .properties(width=300, height=210)
        .facet(column=alt.Column("sweep:N", title=None))
        .properties(title="One-epoch regularization frontier")
    )
    mo.vstack(
        [
            mo.md("## Early ablation signal"),
            mo.ui.altair_chart(ablation_chart),
            mo.callout(mo.md("λ=0 collapses: prediction loss reaches 6.25×10⁻⁵ after one epoch and 5.34×10⁻⁶ after three, while the SIGReg diagnostic stays at 51.49. λ=0.01 minimizes one-epoch prediction loss but becomes non-finite in epoch 3 at the paper learning rate; its stabilized rerun uses half the learning rate and tighter clipping. λ≈0.05–0.10 is the early balance region. Dropout has much smaller first-epoch effects."), kind="warn"),
        ]
    )
    return


@app.cell
def _(alt, mo, pd):
    collapse = pd.DataFrame(
        [
            {"epoch": 0, "metric": "Prediction loss", "value": 0.06849},
            {"epoch": 1, "metric": "Prediction loss", "value": 0.0000624628},
            {"epoch": 2, "metric": "Prediction loss", "value": 0.00000992368},
            {"epoch": 3, "metric": "Prediction loss", "value": 0.00000533978},
            {"epoch": 0, "metric": "SIGReg diagnostic", "value": 55.5},
            {"epoch": 1, "metric": "SIGReg diagnostic", "value": 51.4943},
            {"epoch": 2, "metric": "SIGReg diagnostic", "value": 51.4943},
            {"epoch": 3, "metric": "SIGReg diagnostic", "value": 51.4943},
        ]
    )
    collapse_chart = (
        alt.Chart(collapse)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("epoch:Q", title="Completed epoch", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("value:Q", title=None, scale=alt.Scale(type="log")),
            color=alt.Color("metric:N", title=None),
            tooltip=["metric:N", "epoch:Q", alt.Tooltip("value:Q", format=".6g")],
        )
        .properties(width=640, height=260, title="Prediction-only training converges to a collapsed latent")
    )
    mo.vstack(
        [
            mo.md("## Collapse trajectory"),
            mo.ui.altair_chart(collapse_chart),
            mo.md("Prediction error falls by **12,826×** from initialization through epoch 3, but the held-out SIGReg diagnostic remains **51.49** instead of approaching the Gaussian target. Low prediction error alone is therefore not evidence of a useful world model."),
        ]
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md("## Full from-scratch result: TwoRoom"),
            mo.hstack(
                [
                    mo.stat(value="46/50 = 92%", label="Control success", caption="paper 87%; released checkpoint 90%", bordered=True),
                    mo.stat(value="0.00716", label="One-step latent MSE", caption="after 10 epochs", bordered=True),
                    mo.stat(value="1.015", label="Covariance diagonal", caption="SIGReg target = 1", bordered=True),
                    mo.stat(value=".9987 / .9998", label="Position probe r", caption="linear / MLP", bordered=True),
                ],
                widths="equal",
            ),
            mo.callout(mo.md("Training from pixels with the paper's history size 1 took 11,010 seconds and exceeded both the paper and released-checkpoint control point estimates."), kind="success"),
            mo.ui.table(
                [
                    {"evaluation": "Final validation", "measurement": "loss .16841 · prediction .00798 · SIGReg 1.7826"},
                    {"evaluation": "Latent health", "measurement": "diag 1.015 ± .126 · off-diagonal RMS .119 · rank 55.2/192"},
                    {"evaluation": "Position probes", "measurement": "linear r .99870 · MLP r .99975 (2-D proprio)"},
                    {"evaluation": "Offline surprise", "measurement": "normal .00716 · visual proxy 1.13661 · target swap 2.02884"},
                ],
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(alt, mo, pd):
    trajectories = {
        "Random seed": {
            "3072": [0.0250648, 0.0134171, 0.00925220, 0.00772082, 0.00597508],
            "123": [0.0244867, 0.0133568, 0.00993717, 0.00767666, 0.00616885],
            "42": [0.0280995, 0.0130376, 0.00890300, 0.00745580, 0.00640973],
        },
        "SIGReg weight": {
            "0.05": [0.0161022, 0.00923039, 0.00642990, 0.00538097, 0.00419310],
            "0.09": [0.0250648, 0.0134171, 0.00925220, 0.00772082, 0.00597508],
            "0.10": [0.0251537, 0.0141873, 0.00969312, 0.00797197, 0.00623122],
            "0.20": [0.0655181, 0.0206285, 0.0155480, 0.0116433, 0.00973398],
            "0.50": [0.156785, 0.0566378, 0.0394593, 0.0292177, 0.0231041],
        },
        "Predictor dropout": {
            "0.0": [0.0220167, 0.0121709, 0.00929167, 0.00717766, 0.00559076],
            "0.1": [0.0250648, 0.0134171, 0.00925220, 0.00772082, 0.00597508],
            "0.2": [0.0258967, 0.0145559, 0.00962948, 0.00793779, 0.00633766],
            "0.5": [0.0244672, 0.0153039, 0.0112242, 0.00953380, 0.00798399],
        },
    }
    training_rows = [
        {"sweep": sweep, "variant": variant, "epoch": epoch, "prediction_loss": value}
        for sweep, variants in trajectories.items()
        for variant, values in variants.items()
        for epoch, value in enumerate(values, start=1)
    ]
    training_progress = pd.DataFrame(training_rows)
    progress_chart = (
        alt.Chart(training_progress)
        .mark_line(point=True)
        .encode(
            x=alt.X("epoch:Q", title="Completed epoch", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("prediction_loss:Q", title="Validation prediction loss", scale=alt.Scale(type="log")),
            color=alt.Color("variant:N", title="Variant"),
            tooltip=["sweep:N", "variant:N", "epoch:Q", alt.Tooltip("prediction_loss:Q", format=".6f")],
        )
        .properties(width=250, height=220)
        .facet(column=alt.Column("sweep:N", title=None))
        .properties(title="Five-epoch PushT convergence snapshot (stopped on user request)")
    )
    mo.vstack(
        [
            mo.md("## Training convergence across parallel runs"),
            mo.ui.altair_chart(progress_chart),
            mo.callout(mo.md("The three seeds remain tight after epoch 5 (0.00598–0.00641). SIGReg weight 0.05 gives the lowest prediction loss (0.00419), while 0.2–0.5 learns substantially more slowly. Dropout 0–0.2 is similar; dropout 0.5 trails. These PushT runs were stopped after five checkpoints when the user returned."), kind="info"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # LeWorldModel, reproduced

        An executable companion to **LeWorldModel: Stable End-to-End Joint-Embedding
        Predictive Architecture from Pixels** (arXiv:2603.19312). It keeps the
        paper's reported targets separate from measurements reproduced on the
        16-GPU Kubernetes cluster and links every measurement to its supervised
        ORX run.

        The released PushT checkpoint reaches **48/50 = 96%**, exactly matching
        Figure 6. More importantly, the first full model trained from pixels reaches
        **46/50 = 92% on TwoRoom**, above the paper's 87% point estimate. Use the
        environment selector to inspect control fidelity, latent Gaussianity,
        physical-state probes, and surprise.
        """
    )
    return


@app.cell
def _(pd):
    control = pd.DataFrame(
        [
            {"environment": "PushT", "paper": 96.0, "reproduced": 96.0, "successes": 48, "trials": 50, "run_id": "db4d0e7d-1397-4013-8710-3c40e0dd6e4c"},
            {"environment": "TwoRoom", "paper": 87.0, "reproduced": 90.0, "successes": 45, "trials": 50, "run_id": "9bd8039d-e568-4efc-9140-d8ac65416612"},
            {"environment": "Reacher", "paper": 86.0, "reproduced": 74.0, "successes": 37, "trials": 50, "run_id": "db9e902a-cf4c-4600-ad10-395f11293cf1"},
            {"environment": "Cube", "paper": 74.0, "reproduced": 74.0, "successes": 37, "trials": 50, "run_id": "7e9fd669-b5ee-461d-8578-8bf6eac91c63"},
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {"environment": "PushT", "one_step_mse": 0.008176, "cov_diag": 1.00269, "effective_rank": 93.53, "linear_probe_r": 0.81410, "mlp_probe_r": 0.86094, "normal_surprise": 0.008176, "visual_surprise": 1.37091, "teleport_surprise": 2.00234},
            {"environment": "TwoRoom", "one_step_mse": 0.063974, "cov_diag": 0.96840, "effective_rank": 93.91, "linear_probe_r": 0.99646, "mlp_probe_r": 0.99949, "normal_surprise": 0.063974, "visual_surprise": 2.47055, "teleport_surprise": 1.98513},
            {"environment": "Reacher", "one_step_mse": 0.006533, "cov_diag": 0.99026, "effective_rank": 79.12, "linear_probe_r": 0.49426, "mlp_probe_r": 0.48544, "normal_surprise": 0.006533, "visual_surprise": 1.86891, "teleport_surprise": 1.98026},
            {"environment": "Cube", "one_step_mse": 0.004270, "cov_diag": 1.00998, "effective_rank": 108.63, "linear_probe_r": 0.49234, "mlp_probe_r": 0.46136, "normal_surprise": 0.004270, "visual_surprise": 1.60468, "teleport_surprise": 2.01824},
        ]
    )
    return control, diagnostics


@app.cell
def _(control, mo):
    environment = mo.ui.dropdown(
        options=control["environment"].tolist(),
        value="PushT",
        label="Environment",
    )
    environment
    return (environment,)


@app.cell
def _(alt, control, mo):
    control_long = control.melt(
        id_vars=["environment", "run_id"],
        value_vars=["paper", "reproduced"],
        var_name="series",
        value_name="success_rate",
    ).dropna()
    control_chart = (
        alt.Chart(control_long)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("success_rate:Q", title="Success rate (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("environment:N", title=None, sort=["PushT", "TwoRoom", "Reacher", "Cube"]),
            yOffset="series:N",
            color=alt.Color("series:N", title=None, scale=alt.Scale(domain=["paper", "reproduced"], range=["#8b95a5", "#5b7cfa"])),
            tooltip=["environment:N", "series:N", alt.Tooltip("success_rate:Q", format=".1f")],
        )
        .properties(height=240, title="Goal-conditioned control: paper vs reproduction")
    )
    mo.ui.altair_chart(control_chart)
    return


@app.cell
def _(control, environment, math, mo):
    selected_control = control.loc[control["environment"] == environment.value].iloc[0]

    def wilson(successes, trials, z=1.959963984540054):
        if successes is None or math.isnan(successes):
            return None
        p = successes / trials
        den = 1 + z * z / trials
        center = (p + z * z / (2 * trials)) / den
        margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
        return 100 * (center - margin), 100 * (center + margin)

    selected_ci = wilson(selected_control["successes"], selected_control["trials"])
    reproduced_text = "pending" if selected_ci is None else f"{selected_control['reproduced']:.1f}%"
    ci_text = "awaiting run" if selected_ci is None else f"95% Wilson CI {selected_ci[0]:.1f}–{selected_ci[1]:.1f}%"
    gap_text = "—" if selected_ci is None else f"{selected_control['reproduced'] - selected_control['paper']:+.1f} pp"
    mo.hstack(
        [
            mo.stat(value=f"{selected_control['paper']:.1f}%", label="Paper target", bordered=True),
            mo.stat(value=reproduced_text, label="Reproduced", caption=ci_text, bordered=True),
            mo.stat(value=gap_text, label="Reproduction gap", caption=f"ORX {selected_control['run_id'][:8]}", bordered=True),
        ],
        widths="equal",
    )
    return


@app.cell
def _(diagnostics, environment, mo):
    selected_diag_rows = diagnostics.loc[diagnostics["environment"] == environment.value]
    mo.stop(selected_diag_rows.empty, mo.callout(mo.md("Diagnostics are pending for this environment."), kind="info"))
    selected_diag = selected_diag_rows.iloc[0]
    mo.hstack(
        [
            mo.stat(value=f"{selected_diag['one_step_mse']:.4f}", label="One-step latent MSE", bordered=True),
            mo.stat(value=f"{selected_diag['cov_diag']:.3f}", label="Mean covariance diagonal", caption="SIGReg target = 1", bordered=True),
            mo.stat(value=f"{selected_diag['effective_rank']:.1f} / 192", label="Effective latent rank", bordered=True),
            mo.stat(value=f"{selected_diag['linear_probe_r']:.3f} / {selected_diag['mlp_probe_r']:.3f}", label="Probe Pearson r", caption="linear / MLP", bordered=True),
        ],
        widths="equal",
    )
    return (selected_diag,)


@app.cell
def _(alt, environment, mo, pd, selected_diag):
    surprise = pd.DataFrame(
        {
            "condition": ["Normal", "Visual proxy", "Continuity violation"],
            "prediction_mse": [selected_diag["normal_surprise"], selected_diag["visual_surprise"], selected_diag["teleport_surprise"]],
        }
    )
    surprise_chart = (
        alt.Chart(surprise)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("condition:N", title=None, sort=["Normal", "Visual proxy", "Continuity violation"]),
            y=alt.Y("prediction_mse:Q", title="Next-latent prediction MSE"),
            color=alt.Color("condition:N", legend=None, scale=alt.Scale(range=["#8b95a5", "#e3a857", "#d95d78"])),
            tooltip=["condition:N", alt.Tooltip("prediction_mse:Q", format=".5f")],
        )
        .properties(height=250, title=f"{environment.value}: matched offline surprise test")
    )
    mo.vstack(
        [
            mo.ui.altair_chart(surprise_chart),
            mo.callout(
                mo.md("The physical proxy replaces the target with an unrelated trajectory state. The visual proxy permutes color channels and is deliberately labeled separately from the paper's simulator-level color intervention."),
                kind="neutral",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("## What was actually reproduced")
    mo.mermaid(
        """
        flowchart LR
          P[224×224 pixels] --> E[ViT-Tiny encoder]
          E --> Z[192-D latent]
          A[5-action block] --> AE[Action encoder]
          Z --> T[6-layer AdaLN predictor]
          AE --> T
          T --> ZN[Predicted next latent]
          ZN --> M[MSE prediction]
          Z --> S[SIGReg → N(0,I)]
          ZN --> C[CEM terminal goal cost]
        """
    )
    return


@app.cell
def _(mo, pd):
    protocol_ledger = pd.DataFrame(
        [
            ["Epochs", "10", "released config said 100", "10 for paper-faithful runs"],
            ["SIGReg λ", "0.1 prose / 0.09 best", "released config 0.09", "0.09 headline; explicit ablations"],
            ["TwoRoom history", "1", "released checkpoint 3", "checkpoint eval 3; train variants explicit"],
            ["CEM iterations", "30 PushT / 10 others", "released config 30 all", "paper protocol"],
            ["Eval RNG", "42", "training seed 3072", "separate eval_seed=42"],
            ["Dependencies", "not pinned", "latest resolver breaks ABI", "NumPy 1.26.4 + SWM 0.1.1"],
        ],
        columns=["Decision", "Paper", "Released code", "This reproduction"],
    )
    mo.vstack([mo.md("## Reproducibility ledger"), mo.ui.table(protocol_ledger, selection=None)])
    return


@app.cell
def _(mo, pd):
    smoke = pd.DataFrame(
        [
            ["Total validation loss", 5.01090, 0.20633, "24.3× lower"],
            ["Prediction loss", 0.07340, 0.02689, "2.7× lower"],
            ["SIGReg diagnostic", 55.00000, 1.99357, "27.6× lower"],
        ],
        columns=["Metric", "Initialization", "After one epoch", "Change"],
    )
    mo.vstack(
        [
            mo.md("## From-scratch smoke result"),
            mo.ui.table(smoke, selection=None),
            mo.callout(
                mo.md("The 18.0M-parameter model trained end-to-end for 13,933 steps; every parameter received gradients. Its 5-case control check reached 1/5, which is a pipeline smoke test rather than a paper-comparable estimate."),
                kind="success",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Interpretation

        - The exact PushT headline confirms that the released model, data, and
          paper-seed CEM stack are internally consistent.
        - Reacher reaches 74% rather than the paper's 86%; with only 50 trials,
          its Wilson interval is wide, but the point estimate is a meaningful
          discrepancy that the from-scratch runs should help diagnose.
        - Cube reaches 37/50 = 74%, exactly matching the paper, and has the
          highest effective rank of the four released checkpoints measured.
        - PushT and TwoRoom covariance diagonals are close to one, directly
          supporting the claimed SIGReg Gaussianization; their effective rank is
          about half the 192-D ambient space, so isotropy is useful but not exact.
        - Physical state is highly recoverable in TwoRoom (`r≈.996` linear), while
          the seven-dimensional PushT state is more entangled (`r≈.814` linear,
          `.861` MLP) in this joint probe.
        - The offline target-swap test robustly detects continuity violations, but
          it is not a substitute for the paper's simulator-level teleport/color
          intervention. Those two protocols remain labeled separately.

        **Primary sources:** [paper](https://arxiv.org/abs/2603.19312),
        [official code](https://github.com/lucas-maes/le-wm), and
        [official checkpoints/data](https://huggingface.co/collections/quentinll/lewm).
        """
    )
    return


if __name__ == "__main__":
    app.run()
