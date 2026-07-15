"""Branch-configured LeWM paper reproduction entrypoint.

Every ORX node invokes this file with the same command. Experimental choices
live in config/repro.json and therefore remain visible in the git diff.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config" / "repro.json").read_text())
HOME = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))
DATA = HOME / "datasets"
ARCHIVES = HOME / "archives"
LOCKS = HOME / "locks"


DATASETS = {
    "pusht": {
        "repo": "quentinll/lewm-pusht",
        "file": "pusht_expert_train.h5.zst",
        "sha256": "7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318",
        "expected": "pusht_expert_train.h5",
        "train_config": "pusht",
        "eval_config": "pusht.yaml",
        "policy": "quentinll/lewm-pusht",
    },
    "tworoom": {
        "repo": "quentinll/lewm-tworooms",
        "file": "tworoom.tar.zst",
        "sha256": "494b1a02f0765cd9a0d9daf1786c419ced1009977fc45d01e3158932f8d080ca",
        "expected": "tworoom.h5",
        "train_config": "tworoom",
        "eval_config": "tworoom.yaml",
        "policy": "quentinll/lewm-tworooms",
    },
    "cube": {
        "repo": "quentinll/lewm-cube",
        "file": "cube_single_expert.tar.zst",
        "sha256": "3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600",
        "expected": "ogbench/cube_single_expert.h5",
        "train_config": "ogb",
        "eval_config": "cube.yaml",
        "policy": "quentinll/lewm-cube",
    },
    "reacher": {
        "repo": "quentinll/lewm-reacher",
        "file": "reacher.tar.zst",
        "sha256": "4ff238f7370bf8c3fd1882ccf604f0e3538375c2636ddbbf45eeb7544d492668",
        "expected": "dmc/reacher_random.h5",
        "train_config": "dmc",
        "eval_config": "reacher.yaml",
        "policy": "quentinll/lewm-reacher",
    },
}


def emit(kind: str, **values) -> None:
    print("ORX_METRIC " + json.dumps({"kind": kind, **values}, sort_keys=True), flush=True)


def run(cmd: list[str], **kwargs) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


@contextlib.contextmanager
def exclusive_lock(name: str):
    LOCKS.mkdir(parents=True, exist_ok=True)
    with (LOCKS / f"{name}.lock").open("w") as handle:
        print(f"Waiting for shared lock: {name}", flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def locate_expected(expected: str) -> Path | None:
    exact = DATA / expected
    if exact.exists():
        return exact
    name = Path(expected).name
    matches = sorted(DATA.rglob(name))
    return matches[0] if matches else None


def prepare_dataset(environment: str) -> Path:
    spec = DATASETS[environment]
    DATA.mkdir(parents=True, exist_ok=True)
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(f"dataset-{environment}"):
        found = locate_expected(spec["expected"])
        if found is not None:
            print(f"Using cached dataset: {found}", flush=True)
            return found

        archive = ARCHIVES / spec["file"]
        url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/main/{spec['file']}"
        if not archive.exists() or sha256(archive) != spec["sha256"]:
            partial = archive.with_suffix(archive.suffix + ".partial")
            run(["curl", "-L", "--fail", "--retry", "12", "--retry-all-errors", "--continue-at", "-", "-o", str(partial), url])
            partial.replace(archive)
        actual_hash = sha256(archive)
        if actual_hash != spec["sha256"]:
            raise RuntimeError(f"Checksum mismatch for {archive}: {actual_hash}")

        if archive.name.endswith(".h5.zst"):
            output = DATA / archive.name.removesuffix(".zst")
            run(["zstd", "-T0", "-d", "-f", str(archive), "-o", str(output)])
        else:
            run(["tar", "--zstd", "-xf", str(archive), "-C", str(DATA)])

        found = locate_expected(spec["expected"])
        if found is None:
            candidates = sorted(DATA.rglob("*.h5"))
            same_env = [p for p in candidates if environment in str(p).lower() or Path(spec["expected"]).stem in p.stem]
            if len(same_env) == 1:
                found = same_env[0]
                target = DATA / spec["expected"]
                target.parent.mkdir(parents=True, exist_ok=True)
                if target != found and not target.exists():
                    target.symlink_to(found)
                found = target
        if found is None:
            raise FileNotFoundError(f"Could not find {spec['expected']} after extracting {archive}")
        print(f"Prepared dataset: {found}", flush=True)
        return found


def set_determinism(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        import torch
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def train(environment: str) -> str:
    spec = DATASETS[environment]
    name = CFG["name"]
    overrides = [
        f"data={spec['train_config']}",
        f"seed={CFG['seed']}",
        f"trainer.max_epochs={CFG['epochs']}",
        "trainer.devices=1",
        "wandb.enabled=false",
        f"history_size={CFG['history_size']}",
        f"loss.sigreg.weight={CFG['sigreg_weight']}",
        f"model.predictor.dropout={CFG['predictor_dropout']}",
        f"subdir={name}",
        f"output_model_name={name}",
    ]
    started = time.time()
    run([sys.executable, "-u", "train.py", *overrides], cwd=ROOT)
    elapsed = time.time() - started
    checkpoint = f"{name}/weights_epoch_{CFG['epochs']}.pt"
    emit("training", environment=environment, seed=CFG["seed"], epochs=CFG["epochs"], elapsed_seconds=elapsed, checkpoint=checkpoint)
    return checkpoint


def analyze_offline(environment: str, policy: str) -> None:
    """Measure latent health, probes, and matched visual/continuity surprise."""
    import numpy as np
    import torch
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from scipy.stats import pearsonr, ttest_rel
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from omegaconf import OmegaConf
    from utils import get_column_normalizer, get_img_preprocessor

    spec = DATASETS[environment]
    cfg = OmegaConf.load(ROOT / "config" / "train" / "data" / f"{spec['train_config']}.yaml")
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    dataset_cfg["num_steps"] = int(CFG["history_size"]) + 1
    dataset = swm.data.load_dataset(dataset_name, **dataset_cfg)
    transforms = [get_img_preprocessor("pixels", "pixels", 224)]
    for col in cfg.dataset.keys_to_load:
        if not col.startswith("pixels"):
            transforms.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms)

    model = swm.wm.utils.load_pretrained(policy).cuda().eval().requires_grad_(False)
    n = min(int(CFG["probe_samples"]), len(dataset))
    rng = np.random.default_rng(int(CFG["seed"]))
    indices = rng.choice(len(dataset), size=n, replace=False)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices.tolist()), batch_size=128, num_workers=4)
    embeddings, targets, errors, visual_errors, teleport_errors = [], [], [], [], []
    target_key = next((k for k in ("state", "proprio", "observation") if k in dataset.column_names), None)

    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model.encode(dict(batch))
            history = int(CFG["history_size"])
            pred = model.predict(out["emb"][:, :history], out["act_emb"][:, :history])[:, -1]
            target = out["emb"][:, history]
            normal = (pred - target).pow(2).mean(-1)

            visual_batch = dict(batch)
            visual_pixels = batch["pixels"].clone()
            visual_pixels[:, history] = visual_pixels[:, history][:, [1, 2, 0]]
            visual_batch["pixels"] = visual_pixels
            visual_target = model.encode(visual_batch)["emb"][:, history]
            visual = (pred - visual_target).pow(2).mean(-1)

            shuffled = target.roll(1, dims=0)
            teleport = (pred - shuffled).pow(2).mean(-1)
            embeddings.append(target.cpu())
            errors.append(normal.cpu())
            visual_errors.append(visual.cpu())
            teleport_errors.append(teleport.cpu())
            if target_key:
                targets.append(batch[target_key][:, history].flatten(1).cpu())

    z = torch.cat(embeddings).numpy()
    normal = torch.cat(errors).numpy()
    visual = torch.cat(visual_errors).numpy()
    teleport = torch.cat(teleport_errors).numpy()
    centered = z - z.mean(0, keepdims=True)
    cov = centered.T @ centered / max(len(z) - 1, 1)
    eig = np.linalg.eigvalsh(cov).clip(min=1e-12)
    p = eig / eig.sum()
    effective_rank = float(np.exp(-(p * np.log(p)).sum()))
    offdiag = cov - np.diag(np.diag(cov))
    emit("latent_health", environment=environment, policy=policy, samples=len(z), one_step_mse=float(normal.mean()), covariance_diag_mean=float(np.diag(cov).mean()), covariance_diag_std=float(np.diag(cov).std()), covariance_offdiag_rms=float(np.sqrt(np.mean(offdiag ** 2))), effective_rank=effective_rank, embedding_dim=z.shape[1])
    emit("surprise", environment=environment, policy=policy, protocol="offline_target_swap", normal_mean=float(normal.mean()), visual_mean=float(visual.mean()), teleport_mean=float(teleport.mean()), visual_paired_p=float(ttest_rel(visual, normal).pvalue), teleport_paired_p=float(ttest_rel(teleport, normal).pvalue), samples=len(normal))

    if targets:
        y = torch.cat(targets).numpy()
        x_train, x_test, y_train, y_test = train_test_split(z, y, test_size=0.2, random_state=int(CFG["seed"]))
        for probe_name, probe in {
            "linear": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "mlp": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=100, early_stopping=True, random_state=int(CFG["seed"]))),
        }.items():
            probe.fit(x_train, y_train)
            y_pred = probe.predict(x_test)
            rs = [pearsonr(y_test[:, i], y_pred[:, i]).statistic for i in range(y.shape[1]) if np.std(y_test[:, i]) > 1e-8]
            emit("probe", environment=environment, policy=policy, target=target_key, probe=probe_name, dimensions=y.shape[1], mse=float(mean_squared_error(y_test, y_pred)), pearson_r=float(np.nanmean(rs)))


def evaluate_control(environment: str, policy: str) -> None:
    spec = DATASETS[environment]
    cmd = [
        sys.executable,
        "-u",
        "eval.py",
        f"--config-name={spec['eval_config']}",
        f"policy={policy}",
        f"eval.num_eval={CFG['num_eval']}",
        f"solver.n_steps={CFG['cem_steps']}",
        f"seed={CFG['seed']}",
        f"output.filename={CFG['name']}-results.txt",
    ]
    run(cmd, cwd=ROOT)


def main() -> None:
    environment = CFG["environment"]
    if environment not in DATASETS:
        raise ValueError(f"Unknown environment: {environment}")
    print(json.dumps({"reproduction_config": CFG}, indent=2, sort_keys=True), flush=True)
    print(f"python={sys.version}", flush=True)
    run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    set_determinism(int(CFG["seed"]))
    dataset = prepare_dataset(environment)
    emit("dataset", environment=environment, path=str(dataset), size_bytes=dataset.stat().st_size)

    if CFG["mode"] == "released":
        policy = DATASETS[environment]["policy"]
    elif CFG["mode"] == "train":
        policy = train(environment)
    else:
        raise ValueError(f"Unknown mode: {CFG['mode']}")

    if CFG.get("run_diagnostics", True):
        analyze_offline(environment, policy)
    if CFG.get("run_control", True):
        evaluate_control(environment, policy)
    emit("complete", name=CFG["name"], environment=environment, mode=CFG["mode"], seed=CFG["seed"])


if __name__ == "__main__":
    main()
