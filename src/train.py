"""Training loop and experiment runner.

Six experiments are defined in ``config.EXPERIMENTS`` (A: U-Net from scratch, B: ResNet18
U-Net with pre-trained encoder, C: the same network with random weights, D: B with strong
augmentation and a higher weight decay, E: B with a ResNet50 encoder, F: the final network, the
recipe of B trained on all the labelled frames). An experiment may override the number of epochs,
the weight decay and the augmentation profile, the other settings come from the global values of
``config``. Which one runs is decided by the flag ``config.EXPERIMENT`` or by the argument of ``train``.

    from src.train import train
    train()            # the experiment of config.EXPERIMENT
    train("B")         # a specific experiment
    # or from a terminal:  python -m src.train --experiment B

All hyperparameters are read from ``config``. Everything produced by an experiment is stored in
``config.OUTPUT_DIR / "experiment_<name>"``:

- ``best.pt``:      weights of the epoch with the best validation score, plus everything needed
                    for inference (model name, mean and std of the normalisation, image size).
                    A final-training run (experiment F) writes ``final.pt`` instead, with the same
                    content: the weights of the last epoch, since it has no held-out data to choose from;
- ``last.pt``:      full state of the last epoch (optimiser, scheduler, history), to resume;
- ``history.json``: losses, validation scores and timings of every epoch;
- ``results.json``: summary of the run (best score, per-class IoU, parameters, time, GPU memory).
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR
from . import config
from .data import CLASS_NAMES, download_dataset
from .dataset import create_dataloaders
from .losses import create_loss
from .metrics import MeanIoU
from .models import build_model, count_parameters
from .split import load_split


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed every random number generator used by the project."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_experiment(name: str | None = None) -> tuple[str, dict]:
    """Return the name and the settings of an experiment (default: config.EXPERIMENT)."""
    name = name or config.EXPERIMENT
    if name not in config.EXPERIMENTS:
        raise ValueError(f"unknown experiment '{name}', available: {sorted(config.EXPERIMENTS)}")
    return name, config.EXPERIMENTS[name]


def resolve_settings(settings: dict) -> dict:
    """The settings an experiment can override, with the global ``config`` value where it does not.

    This is the only place where the fallback to ``EPOCHS``, ``WEIGHT_DECAY`` and ``AUGMENTATION``
    is decided, so that ``train`` and ``compare_experiments`` always agree. ``final_training`` is
    False unless the experiment asks for it (see ``train``).
    """
    return {"epochs": settings.get("epochs", config.EPOCHS),
            "weight_decay": settings.get("weight_decay", config.WEIGHT_DECAY),
            "augmentation": settings.get("augmentation", config.AUGMENTATION),
            "encoder_lr_factor": settings["encoder_lr_factor"],
            "final_training": bool(settings.get("final_training", False))}


def lr_factor(epoch: int, epochs: int, warmup_epochs: int = config.WARMUP_EPOCHS, min_factor: float = config.MIN_LR_FACTOR) -> float:
    """Multiplier of the base learning rate at a given epoch (0-based).

    Linear warmup for ``warmup_epochs`` epochs, then a cosine decay from 1 down to ``min_factor``.
    The warmup avoids large, damaging updates while the decoder is still random, the decay
    lets the network settle into a minimum at the end of the training.
    """
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: torch.nn.Module, lr: float = config.LEARNING_RATE,
                    weight_decay: float = config.WEIGHT_DECAY,
                    encoder_lr_factor: float = 1.0) -> torch.optim.Optimizer:
    """AdamW. With ``encoder_lr_factor != 1`` the encoder gets its own, scaled learning rate.

    The model must then provide ``encoder_parameters()`` and ``decoder_parameters()``. The decoder
    group is always the last one, so ``optimizer.param_groups[-1]`` is the reference learning rate.
    """
    if encoder_lr_factor != 1.0:
        if not hasattr(model, "encoder_parameters"):
            raise ValueError("encoder_lr_factor needs a model with encoder_parameters()")
        groups = [{"params": list(model.encoder_parameters()), "lr": lr * encoder_lr_factor}, {"params": list(model.decoder_parameters()), "lr": lr}]
    else:
        groups = [{"params": list(model.parameters()), "lr": lr}]
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def _make_grad_scaler(enabled: bool):
    """Gradient scaler for mixed precision (the constructor changed between PyTorch versions)."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _to_device(images: torch.Tensor, masks: torch.Tensor, device: torch.device):
    # channels_last memory layout is faster on GPUs with tensor cores
    images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
    return images, masks.to(device, non_blocking=True)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp: bool,
                    grad_clip: float | None) -> dict[str, float]:
    """One pass over the training data. Returns the mean loss and its two terms."""
    model.train()
    sums = torch.zeros(3, device=device)        # accumulated on the device: no GPU sync at every step
    count = 0
    for images, masks in loader:
        images, masks = _to_device(images, masks, device)
        optimizer.zero_grad(set_to_none=True)
        # Forward pass in float16 where it is safe (faster, less memory)...
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        # ...but the loss is computed in float32 (see the loss module)
        parts = criterion.components(logits, masks)
        # The scaler multiplies the loss to keep small float16 gradients from underflowing
        scaler.scale(parts["total"]).backward()
        if grad_clip:
            scaler.unscale_(optimizer)          # gradients back to their real scale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch = images.shape[0]
        sums += torch.stack([parts["total"].detach(), parts["ce"].detach(), parts["dice"].detach()]) * batch
        count += batch
    loss, ce, dice = (sums / count).tolist()
    return {"loss": loss, "ce": ce, "dice": dice}


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool) -> dict:
    """Validation pass: loss and the official metric (see ``MeanIoU.compute`` for the keys)."""
    model.eval()
    metric = MeanIoU()
    sums = torch.zeros(3, device=device)
    count = 0
    for images, masks in loader:
        images, masks = _to_device(images, masks, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        parts = criterion.components(logits, masks)
        metric.update(logits, masks)
        batch = images.shape[0]
        sums += torch.stack([parts["total"], parts["ce"], parts["dice"]]) * batch
        count += batch
    loss, ce, dice = (sums / count).tolist()
    results = metric.compute()
    results.update({"loss": loss, "ce": ce, "dice": dice})
    return results


def _clean(values) -> list[float | None]:
    """NaN-safe list of floats for JSON (a class absent from the whole validation set is NaN)."""
    return [None if np.isnan(v) else float(v) for v in values]


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, output_dir: str | Path,
        epochs: int, device: torch.device, metadata: dict, use_amp: bool = config.USE_AMP,
        grad_clip: float | None = config.GRAD_CLIP_NORM,
        patience: int | None = config.EARLY_STOPPING_PATIENCE, resume: bool = False,
        verbose: bool = True, final_training: bool = False) -> dict:
    """Train ``model`` for up to ``epochs`` epochs, validating and checkpointing after each one.

    Args:
        metadata: saved in the checkpoints (experiment name, model name and arguments, mean and std
            of the normalisation, image size). ``load_checkpoint`` uses it to rebuild the model.
        patience: stop when the validation score has not improved for this many epochs.
        resume: continue from ``last.pt`` in ``output_dir`` instead of starting from scratch.
        final_training: the run has no held-out data (the "validation" loader contains frames that are
            also in the training set), so the score cannot be used to choose anything: there is no early
            stopping, every epoch replaces the saved weights, and the weights of the last epoch are
            written to ``final.pt`` (instead of ``best.pt``). The score is still computed and logged:
            it is a sanity check on frames the network has seen, not an estimate of its quality.

    Returns:
        A dict with the history, the best score and epoch, and the paths of the checkpoints.
    """
    print(f"Image size = {metadata.get('image_size')}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = output_dir / "last.pt", output_dir / ("final.pt" if final_training else "best.pt")
    if final_training:
        patience = None
    use_amp = use_amp and device.type == "cuda"
    scaler = _make_grad_scaler(use_amp)

    history: list[dict] = []
    best_score, best_epoch, start_epoch = -1.0, 0, 0
    if resume:
        if not last_path.exists():
            raise FileNotFoundError(f"nothing to resume: {last_path} does not exist")
        state = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, best_score, best_epoch = state["history"], state["best_score"], state["best_epoch"]
        start_epoch = state["epoch"]
        if verbose:
            print(f"Resuming from epoch {start_epoch} (best score so far {best_score:.4f})")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    run_started = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_started = time.time()
        lr = optimizer.param_groups[-1]["lr"]            # learning rate used during this epoch
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp, grad_clip)
        val = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        record = {
            "epoch": epoch + 1, "lr": lr,
            "train_loss": train_stats["loss"], "train_ce": train_stats["ce"], "train_dice": train_stats["dice"],
            "val_loss": val["loss"], "val_ce": val["ce"], "val_dice": val["dice"],
            "val_score": val["score"], "val_mean_class_iou": val["mean_class_iou"],
            "val_per_class": _clean(val["per_class"]),
            "seconds": time.time() - epoch_started,
            "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0,
        }
        history.append(record)

        improved = final_training or val["score"] > best_score      # a final run always keeps the latest epoch
        if improved:
            best_score, best_epoch = val["score"], epoch + 1
            torch.save({**metadata, "epoch": best_epoch, "val_score": best_score,
                        "val_per_class": record["val_per_class"], "model": model.state_dict()}, best_path)

        torch.save({**metadata, "epoch": epoch + 1, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "history": history,
                    "best_score": best_score, "best_epoch": best_epoch}, last_path)
        (output_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")

        if verbose and final_training:
            # The frames of the "validation" loader are part of the training set: the score is a sanity
            # check (it should be high and keep rising), not an estimate of the quality of the network
            print(f"epoch {epoch + 1:3d}/{epochs} | lr {lr:.2e} | train {train_stats['loss']:.3f} "
                  f"(ce {train_stats['ce']:.3f}, dice {train_stats['dice']:.3f}) | seen-frames loss "
                  f"{val['loss']:.3f} | seen-frames score {val['score']:.4f} | {record['seconds']:.0f}s")
        elif verbose:
            print(f"epoch {epoch + 1:3d}/{epochs} | lr {lr:.2e} | train {train_stats['loss']:.3f} "
                f"(ce {train_stats['ce']:.3f}, dice {train_stats['dice']:.3f}) | val {val['loss']:.3f} "
                f"| score {val['score']:.4f}{' *' if improved else '  '} (best {best_score:.4f} "
                f"@ {best_epoch}) | {record['seconds']:.0f}s")

        if patience is not None and epoch + 1 - best_epoch >= patience:
            if verbose:
                print(f"Early stopping: no improvement for {patience} epochs")
            break

    best = next(r for r in history if r["epoch"] == best_epoch)
    results = {
        "experiment": metadata.get("experiment"), "description": metadata.get("description"),
        "final_training": final_training,
        "model": metadata.get("model_name"), "settings": metadata.get("settings"),
        "best_score": best_score, "best_epoch": best_epoch, "epochs_run": len(history),
        "best_val_mean_class_iou": best["val_mean_class_iou"],
        "best_val_per_class": dict(zip(CLASS_NAMES[1:], best["val_per_class"])),
        "parameters": count_parameters(model),
        "seconds_this_run": time.time() - run_started,
        "seconds_per_epoch": float(np.mean([r["seconds"] for r in history])),
        "gpu_peak_gb": max(r["gpu_peak_gb"] for r in history),
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return {"history": history, "best_score": best_score, "best_epoch": best_epoch,
            "best_path": best_path, "last_path": last_path, "output_dir": output_dir,
            "results": results}


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def train(experiment: str | None = None, epochs: int | None = None, resume: bool = False,
        device: str | None = None, batch_size: int = config.BATCH_SIZE,
        num_workers: int = config.NUM_WORKERS, output_dir: str | Path | None = None,
        verbose: bool = True) -> dict:
    """Run one complete experiment: data, model, loss, optimiser and training loop.

    Args:
        experiment: a key of ``config.EXPERIMENTS``. Default: ``config.EXPERIMENT``.
        epochs: number of epochs. Default: the "epochs" of the experiment, else ``config.EPOCHS``.
        resume: continue an interrupted run from its ``last.pt``.
        output_dir: where to write the results. Default: ``config.OUTPUT_DIR / experiment_<name>``.

    With ``final_training`` in the settings of the experiment (experiment F) the network is trained
    on every labelled frame: the training frames of the split, the validation ones and the frames
    that the split purged (there is no validation left to protect from leakage). Nothing is held out,
    so there is no early stopping and no best epoch: the weights of the last epoch are saved in
    ``final.pt``. The validation frames are still evaluated at every epoch, as a sanity check only.
    """
    name, settings = get_experiment(experiment)
    resolved = resolve_settings(settings)
    if epochs is not None:
        resolved["epochs"] = epochs             # an explicit argument wins over the experiment
    epochs, weight_decay, augmentation = resolved["epochs"], resolved["weight_decay"], resolved["augmentation"]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(config.SEED)
    torch.backends.cudnn.benchmark = True       # let cuDNN pick the fastest kernels for our fixed sizes

    # Data: the saved split, so that every experiment sees exactly the same samples
    samples = download_dataset(config.TRAIN_DIR)         # only data/train, never data/test
    train_samples, val_samples = load_split(config.SPLIT_PATH, samples)
    if resolved["final_training"]:
        train_samples = list(samples)
    if settings["normalization"] == "imagenet":
        mean, std = config.IMAGENET_MEAN, config.IMAGENET_STD
    else:
        mean, std = None, None                  # computed on the training samples by create_dataloaders
    train_loader, val_loader, mean, std = create_dataloaders(
        train_samples, val_samples, image_size=config.DEFAULT_IMAGE_SIZE, batch_size=batch_size,
        mean=mean, std=std, num_workers=num_workers, seed=config.SEED, augmentation=augmentation)

    # Model, loss (class weights from the training samples), optimiser and learning-rate schedule
    model = build_model(settings["model"], **settings["model_kwargs"])
    model = model.to(device).to(memory_format=torch.channels_last)
    criterion = create_loss(train_samples, config.DEFAULT_IMAGE_SIZE, device)
    optimizer = build_optimizer(model, weight_decay=weight_decay, encoder_lr_factor=settings["encoder_lr_factor"])
    scheduler = LambdaLR(optimizer, lambda epoch: lr_factor(epoch, epochs))

    output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR / f"experiment_{name}"
    metadata = {
        "experiment": name, "description": settings["description"],
        "model_name": settings["model"], "model_kwargs": dict(settings["model_kwargs"]),
        "mean": tuple(mean), "std": tuple(std), "image_size": tuple(config.DEFAULT_IMAGE_SIZE),
        "settings": resolved,                   # epochs, weight decay, augmentation, encoder learning rate
    }
    if verbose:
        print(f"Experiment {name}: {settings['description']}")
        print(f"{count_parameters(model):,} parameters | device {device} | " f"{len(train_samples)} training and {len(val_samples)} validation samples")
        print(f"{epochs} epochs | weight decay {weight_decay} | augmentation '{augmentation}'")
        if resolved["final_training"]:
            print("Final training: every labelled frame is used for training, the last epoch is kept and the "
                  "validation score below is measured on frames the network is trained on")

    return fit(model, train_loader, val_loader, criterion, optimizer, scheduler, output_dir, epochs, device,
               metadata, resume=resume, verbose=verbose, final_training=resolved["final_training"])


def run_experiments(names=None, **kwargs) -> dict[str, dict]:
    """Run several experiments one after the other, e.g. ``run_experiments(("A", "B"))``.

    Without ``names`` every experiment defined in ``config.EXPERIMENTS`` is run, in alphabetical order.
    """
    return {name: train(name, **kwargs) for name in names or sorted(config.EXPERIMENTS)}


# --------------------------------------------------------------------------- #
# Using the results
# --------------------------------------------------------------------------- #
def load_checkpoint(path: str | Path, device: str = "cpu"):
    """Rebuild a trained model from a checkpoint (``best.pt`` or ``last.pt``).

    Pre-trained weights are never downloaded here: the weights come from the checkpoint.

    Returns:
        (model in eval mode, checkpoint dict with mean, std, image_size, scores, ...).
    """
    state = torch.load(path, map_location=device, weights_only=True)
    kwargs = dict(state["model_kwargs"])
    if "pretrained" in kwargs:
        kwargs["pretrained"] = False
    model = build_model(state["model_name"], **kwargs)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state


def load_history(name: str, output_dir: str | Path | None = None) -> list[dict]:
    """The per-epoch history of an experiment that has been run (``history.json``)."""
    output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
    return json.loads((output_dir / f"experiment_{name}" / "history.json").read_text(encoding="utf-8"))


def _is_final_run(name: str, output_dir: Path) -> bool:
    """Whether an experiment was trained on all the frames (the "final_training" setting).

    Read from the ``results.json`` of the run, or from ``config.EXPERIMENTS`` for a run that is still
    in progress or was made before the key existed.
    """
    path = output_dir / f"experiment_{name}" / "results.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8")).get("final_training")
        if stored is not None:
            return bool(stored)
    return bool(config.EXPERIMENTS.get(name, {}).get("final_training", False))


def _finished_experiments(names, output_dir: Path) -> list[str]:
    """The requested experiments (default: all the defined ones) that have a ``history.json``.

    Final-training runs are left out: their score is measured on frames they were trained on, so it
    cannot be compared with the score of the experiments that have a held-out validation set.
    """
    return [n for n in names or sorted(config.EXPERIMENTS)
            if (output_dir / f"experiment_{n}" / "history.json").exists() and not _is_final_run(n, output_dir)]


def compare_experiments(names=None, output_dir: str | Path | None = None) -> pd.DataFrame:
    """Table with the results of the experiments that have been run, one row each.

    Columns: best validation score and its epoch, epochs run, parameters, time and GPU memory, what
    distinguishes the experiments (model, augmentation profile, weight decay, planned epochs),
    the training and validation loss at the best epoch (a large gap means overfitting) and the
    IoU of every class. Experiments that have not been run are simply missing.

    The column "trained on" is "train" for the experiments with a held-out validation set and
    "train+val" for a final-training run (experiment F). The score and the losses of such a run are
    measured on frames it was trained on: they are a sanity check and must not be compared with the
    other rows.

    The settings come from the ``results.json`` of the run, or from ``config.EXPERIMENTS`` for runs
    made before they were stored there.
    """
    output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
    rows = {}
    for name in names or sorted(config.EXPERIMENTS):
        path = output_dir / f"experiment_{name}" / "results.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        settings = r.get("settings")
        if not settings and name in config.EXPERIMENTS:
            settings = resolve_settings(config.EXPERIMENTS[name])
        settings = settings or {}
        model = r.get("model") or config.EXPERIMENTS.get(name, {}).get("model")
        losses = {"train loss": np.nan, "val loss": np.nan}
        history_path = output_dir / f"experiment_{name}" / "history.json"
        if history_path.exists():
            best = next((e for e in json.loads(history_path.read_text(encoding="utf-8"))
                         if e["epoch"] == r["best_epoch"]), None)
            if best:
                losses = {"train loss": best["train_loss"], "val loss": best["val_loss"]}
        rows[name] = {"score": r["best_score"], "best_epoch": r["best_epoch"], "epochs": r["epochs_run"],
                      "params (M)": r["parameters"] / 1e6, "s/epoch": r["seconds_per_epoch"],
                      "gpu GB": r["gpu_peak_gb"], "model": model,
                      "trained on": "train+val" if _is_final_run(name, output_dir) else "train",
                      "augmentation": settings.get("augmentation"),
                      "weight decay": settings.get("weight_decay"), "planned epochs": settings.get("epochs"),
                      **losses, **r["best_val_per_class"]}
    # The table mixes numbers and text: infer_objects gives the numeric columns their dtype back,
    # otherwise round() would leave them untouched
    return pd.DataFrame(rows).T.infer_objects().round(4)


def plot_comparison(names=None, output_dir: str | Path | None = None):
    """Curves of several experiments on the same axes: validation score, validation loss, training loss.

    It is the quickest way to see whether an experiment overfits less than another: with less
    overfitting the validation loss stays lower and the validation score higher while the training
    loss does not fall as much. Experiments that have not been run are skipped, and so are the
    final-training runs (see ``compare_experiments``).
    """
    output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
    fig, (ax_score, ax_val, ax_train) = plt.subplots(1, 3, figsize=(16, 4))
    for name in _finished_experiments(names, output_dir):
        history = load_history(name, output_dir)
        epochs = [r["epoch"] for r in history]
        ax_score.plot(epochs, [r["val_score"] for r in history], label=f"{name}")
        ax_val.plot(epochs, [r["val_loss"] for r in history], label=f"{name}")
        ax_train.plot(epochs, [r["train_loss"] for r in history], label=f"{name}")
    for ax, title in ((ax_score, "Validation score (official)"), (ax_val, "Validation loss"),
                      (ax_train, "Training loss")):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return fig


def plot_history(history: list[dict], final_training: bool = False):
    """Training and validation curves of one experiment: losses, validation score, learning rate.

    With ``final_training=True`` (experiment F) the "validation" frames are part of the training set, so
    the curves are labelled as measured on seen frames, and no best epoch is marked (the last one is kept).
    """
    epochs = [r["epoch"] for r in history]
    held_out = "seen frames (in the training set)" if final_training else "validation"
    fig, (ax_loss, ax_score, ax_lr) = plt.subplots(1, 3, figsize=(16, 4))
    ax_loss.plot(epochs, [r["train_loss"] for r in history], label="train")
    ax_loss.plot(epochs, [r["val_loss"] for r in history], label=held_out)
    ax_loss.set_title("Loss (weighted CE + Dice)")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()
    ax_score.plot(epochs, [r["val_score"] for r in history], label="score (official)")
    ax_score.plot(epochs, [r["val_mean_class_iou"] for r in history], label="mean class IoU")
    if not final_training:
        best = max(history, key=lambda r: r["val_score"])
        ax_score.axvline(best["epoch"], color="gray", ls=":", label=f"best (epoch {best['epoch']})")
    ax_score.set_title("IoU on seen frames, not a validation" if final_training else "Validation IoU")
    ax_score.set_xlabel("epoch")
    ax_score.legend()
    ax_lr.plot(epochs, [r["lr"] for r in history])
    ax_lr.set_yscale("log")
    ax_lr.set_title("Learning rate")
    ax_lr.set_xlabel("epoch")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one of the experiments defined in config.EXPERIMENTS")
    parser.add_argument("--experiment", default=config.EXPERIMENT, choices=sorted(config.EXPERIMENTS))
    parser.add_argument("--epochs", type=int, default=None,
                        help="default: the epochs of the experiment, else config.EPOCHS")
    parser.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    args = parser.parse_args()
    train(args.experiment, epochs=args.epochs, resume=args.resume)


if __name__ == "__main__":
    main()