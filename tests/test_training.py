"""Tests of src/train.py.

The training loop is tested on tiny synthetic data and a tiny network (a few seconds on the
CPU): the tests check that the pieces fit together, that the files are written, that a run can
be resumed and stopped early, and that the experiment configuration is consistent.
"""
import json

import numpy as np
import pytest
import torch
from torch.optim.lr_scheduler import LambdaLR

from src import config
from src.dataset import create_dataloaders
from src.losses import create_loss
from src.models import MODELS, ResNet50UNet, ResNetUNet, UNet, build_model
from src.train import (build_optimizer, compare_experiments, evaluate, fit, get_experiment,
                       load_checkpoint, load_history, lr_factor, plot_comparison, resolve_settings,
                       plot_history, run_experiments, set_seed, train, train_one_epoch)

SIZE = (32, 48)
CPU = torch.device("cpu")


@pytest.fixture
def setup(synthetic_samples):
    """Tiny network, loaders on 8 training + 4 validation samples, loss and optimiser."""
    set_seed(0)
    train_samples, val_samples = synthetic_samples[:8], synthetic_samples[8:]
    train_loader, val_loader, mean, std = create_dataloaders(
        train_samples, val_samples, image_size=SIZE, batch_size=4, num_workers=0, seed=0)
    model = UNet(base_channels=4, depth=2)
    criterion = create_loss(train_samples, SIZE, CPU)
    metadata = {"experiment": "T", "description": "test", "model_name": "unet",
                "model_kwargs": {"base_channels": 4, "depth": 2}, "mean": tuple(mean),
                "std": tuple(std), "image_size": SIZE}

    def make(lr=5e-3, epochs=4):
        optimizer = build_optimizer(model, lr=lr)
        scheduler = LambdaLR(optimizer, lambda e: lr_factor(e, epochs, warmup_epochs=1))
        return optimizer, scheduler

    return model, train_loader, val_loader, criterion, metadata, make


# --- Learning-rate schedule ------------------------------------------------------------------
def test_warmup_increases_the_learning_rate_linearly():
    factors = [lr_factor(e, 100, warmup_epochs=4) for e in range(4)]
    assert factors == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_cosine_decay_goes_from_one_down_to_the_minimum():
    factors = [lr_factor(e, 50, warmup_epochs=5, min_factor=0.01) for e in range(5, 50)]
    assert factors[0] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(factors, factors[1:]))            # never increases
    assert factors[-1] < 0.05 and factors[-1] >= 0.01


# --- Optimiser ---------------------------------------------------------------------------------
def test_single_group_optimizer_covers_every_parameter():
    model = UNet(base_channels=4, depth=2)
    optimizer = build_optimizer(model, lr=1e-3)
    assert len(optimizer.param_groups) == 1
    assert sum(len(g["params"]) for g in optimizer.param_groups) == len(list(model.parameters()))


def test_encoder_gets_a_scaled_learning_rate():
    model = ResNetUNet(pretrained=False)
    optimizer = build_optimizer(model, lr=1e-3, encoder_lr_factor=0.1)
    encoder_group, decoder_group = optimizer.param_groups
    assert encoder_group["lr"] == pytest.approx(1e-4)
    assert decoder_group["lr"] == pytest.approx(1e-3)
    assert sum(len(g["params"]) for g in optimizer.param_groups) == len(list(model.parameters()))


def test_scaled_encoder_lr_needs_a_model_with_an_encoder():
    with pytest.raises(ValueError):
        build_optimizer(UNet(base_channels=4, depth=2), encoder_lr_factor=0.1)


# --- Experiment configuration ----------------------------------------------------------------
def test_the_six_experiments_are_defined():
    assert {"A", "B", "C", "D", "E", "F"} <= set(config.EXPERIMENTS)


@pytest.mark.parametrize("name", sorted(config.EXPERIMENTS))
def test_every_experiment_points_to_an_existing_model_and_valid_settings(name):
    settings = config.EXPERIMENTS[name]
    assert settings["model"] in MODELS
    assert settings["normalization"] in {"dataset", "imagenet"}
    assert settings["encoder_lr_factor"] > 0
    assert isinstance(settings["model_kwargs"], dict)


def test_b_and_c_differ_only_in_the_weights_and_the_encoder_learning_rate():
    b, c = config.EXPERIMENTS["B"], config.EXPERIMENTS["C"]
    assert b["model"] == c["model"] and b["normalization"] == c["normalization"]
    assert b["model_kwargs"] == {"pretrained": True} and c["model_kwargs"] == {"pretrained": False}


def test_d_is_b_with_stronger_regularisation_and_nothing_else():
    b, d = config.EXPERIMENTS["B"], config.EXPERIMENTS["D"]
    same = ("model", "model_kwargs", "normalization", "encoder_lr_factor")
    assert all(b[k] == d[k] for k in same)
    assert d["augmentation"] in config.AUGMENTATIONS and d["augmentation"] != config.AUGMENTATION
    assert d["weight_decay"] > config.WEIGHT_DECAY
    assert resolve_settings(d)["epochs"] > 0                  # whatever the schedule length chosen for D


def test_e_is_b_with_a_resnet50_encoder_and_nothing_else():
    b, e = config.EXPERIMENTS["B"], config.EXPERIMENTS["E"]
    assert e["model"] == "resnet50_unet" and b["model"] == "resnet18_unet"
    same = ("model_kwargs", "normalization", "encoder_lr_factor")
    assert all(b[k] == e[k] for k in same)
    assert resolve_settings(e) == resolve_settings(b)         # same epochs, weight decay and augmentation


def test_f_is_the_recipe_of_b_trained_on_all_the_frames():
    b, f = config.EXPERIMENTS["B"], config.EXPERIMENTS["F"]
    same = ("model", "model_kwargs", "normalization", "encoder_lr_factor")
    assert all(b[k] == f[k] for k in same)
    assert resolve_settings(f).pop("final_training") is True
    assert {**resolve_settings(f), "final_training": False} == resolve_settings(b)   # same epochs, decay, augmentation
    assert not any(resolve_settings(config.EXPERIMENTS[n])["final_training"] for n in "ABCDE")


def test_experiments_that_do_not_override_anything_use_the_global_values():
    for name in ("A", "B", "C", "E"):
        assert not {"epochs", "weight_decay", "augmentation"} & set(config.EXPERIMENTS[name])


def test_get_experiment_uses_the_global_flag_by_default(monkeypatch):
    monkeypatch.setattr(config, "EXPERIMENT", "C")
    assert get_experiment()[0] == "C"
    assert get_experiment("A")[0] == "A"


def test_unknown_experiment_is_rejected():
    with pytest.raises(ValueError):
        get_experiment("Z")


# --- One epoch -------------------------------------------------------------------------------
def test_training_epoch_updates_the_weights_and_returns_finite_losses(setup):
    model, train_loader, _, criterion, _, make = setup
    optimizer, _ = make()
    before = [p.detach().clone() for p in model.parameters()]
    stats = train_one_epoch(model, train_loader, criterion, optimizer,
                            torch.cuda.amp.GradScaler(enabled=False), CPU, False, 1.0)
    assert set(stats) == {"loss", "ce", "dice"} and all(np.isfinite(v) for v in stats.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_evaluation_is_deterministic_and_does_not_change_the_model(setup):
    model, _, val_loader, criterion, _, _ = setup
    before = [p.detach().clone() for p in model.parameters()]
    first = evaluate(model, val_loader, criterion, CPU, False)
    second = evaluate(model, val_loader, criterion, CPU, False)
    assert first["score"] == second["score"] and 0.0 <= first["score"] <= 1.0
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert {"loss", "ce", "dice", "score", "mean_class_iou", "per_class"} <= set(first)


# --- Full loop -------------------------------------------------------------------------------
def test_fit_trains_validates_and_writes_all_the_files(setup, tmp_path):
    model, train_loader, val_loader, criterion, metadata, make = setup
    optimizer, scheduler = make()
    out = fit(model, train_loader, val_loader, criterion, optimizer, scheduler, tmp_path, epochs=4,
              device=CPU, metadata=metadata, use_amp=False, patience=None, verbose=False)

    history = out["history"]
    assert [r["epoch"] for r in history] == [1, 2, 3, 4]
    assert min(r["train_loss"] for r in history) < history[0]["train_loss"]       # it learns
    assert out["best_score"] == pytest.approx(max(r["val_score"] for r in history))
    for name in ("best.pt", "last.pt", "history.json", "results.json"):
        assert (tmp_path / name).exists()
    results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert results["best_score"] == pytest.approx(out["best_score"]) and results["parameters"] > 0
    assert len(json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))) == 4


def test_learning_rate_follows_the_schedule(setup, tmp_path):
    model, train_loader, val_loader, criterion, metadata, make = setup
    optimizer, scheduler = make(lr=1e-2, epochs=3)
    out = fit(model, train_loader, val_loader, criterion, optimizer, scheduler, tmp_path, epochs=3,
              device=CPU, metadata=metadata, use_amp=False, patience=None, verbose=False)
    expected = [1e-2 * lr_factor(e, 3, warmup_epochs=1) for e in range(3)]
    assert [r["lr"] for r in out["history"]] == pytest.approx(expected)


def test_best_checkpoint_rebuilds_the_model_and_keeps_the_inference_settings(setup, tmp_path):
    model, train_loader, val_loader, criterion, metadata, make = setup
    optimizer, scheduler = make()
    out = fit(model, train_loader, val_loader, criterion, optimizer, scheduler, tmp_path, epochs=3,
              device=CPU, metadata=metadata, use_amp=False, patience=None, verbose=False)

    loaded, state = load_checkpoint(out["best_path"])
    assert not loaded.training and isinstance(loaded, UNet)
    assert tuple(state["mean"]) == pytest.approx(metadata["mean"])
    assert tuple(state["std"]) == pytest.approx(metadata["std"])
    assert tuple(state["image_size"]) == SIZE and state["epoch"] == out["best_epoch"]

    # The loaded model reproduces the validation score recorded when the checkpoint was saved
    replay = evaluate(loaded, val_loader, criterion, CPU, False)
    assert replay["score"] == pytest.approx(state["val_score"])


def test_early_stopping_halts_a_run_that_does_not_improve(setup, tmp_path):
    model, train_loader, val_loader, criterion, metadata, make = setup
    optimizer, scheduler = make(lr=0.0)                      # learning rate zero: the score never changes
    out = fit(model, train_loader, val_loader, criterion, optimizer, scheduler, tmp_path, epochs=10,
              device=CPU, metadata=metadata, use_amp=False, patience=2, verbose=False)
    assert len(out["history"]) == 3                          # best at epoch 1, then 2 epochs without improvement


def test_an_interrupted_run_can_be_resumed(setup, tmp_path):
    model, train_loader, val_loader, criterion, metadata, make = setup
    optimizer, scheduler = make(epochs=4)
    fit(model, train_loader, val_loader, criterion, optimizer, scheduler, tmp_path, epochs=2,
        device=CPU, metadata=metadata, use_amp=False, patience=None, verbose=False)
    weights_at_stop = {k: v.clone() for k, v in model.state_dict().items()}

    # A fresh model, optimiser and scheduler, as after restarting the program
    fresh = UNet(base_channels=4, depth=2)
    fresh_optimizer = build_optimizer(fresh, lr=5e-3)
    fresh_scheduler = LambdaLR(fresh_optimizer, lambda e: lr_factor(e, 4, warmup_epochs=1))
    out = fit(fresh, train_loader, val_loader, criterion, fresh_optimizer, fresh_scheduler, tmp_path,
              epochs=4, device=CPU, metadata=metadata, use_amp=False, patience=None, resume=True,
              verbose=False)

    assert [r["epoch"] for r in out["history"]] == [1, 2, 3, 4]               # continues, does not restart
    assert out["results"]["epochs_run"] == 4
    # The resumed learning rate is the one the schedule prescribes for epoch 3, not the initial one
    assert out["history"][2]["lr"] == pytest.approx(5e-3 * lr_factor(2, 4, warmup_epochs=1))
    assert not all(torch.equal(weights_at_stop[k], v) for k, v in fresh.state_dict().items())


def test_resuming_without_a_checkpoint_fails_clearly(setup, tmp_path):
    model, train_loader, val_loader, criterion, metadata, make = setup
    optimizer, scheduler = make()
    with pytest.raises(FileNotFoundError):
        fit(model, train_loader, val_loader, criterion, optimizer, scheduler, tmp_path, epochs=2,
            device=CPU, metadata=metadata, resume=True, verbose=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_mixed_precision_epoch_on_gpu(synthetic_samples, tmp_path):
    train_loader, val_loader, mean, std = create_dataloaders(
        synthetic_samples[:8], synthetic_samples[8:], image_size=SIZE, batch_size=4, seed=0)
    device = torch.device("cuda")
    model = UNet(base_channels=4, depth=2).to(device).to(memory_format=torch.channels_last)
    criterion = create_loss(synthetic_samples[:8], SIZE, device)
    optimizer = build_optimizer(model)
    scheduler = LambdaLR(optimizer, lambda e: 1.0)
    out = fit(model, train_loader, val_loader, criterion, optimizer, scheduler, tmp_path, epochs=2,
              device=device, metadata={"experiment": "T", "model_name": "unet", "model_kwargs": {}},
              use_amp=True, patience=None, verbose=False)
    assert all(np.isfinite(r["train_loss"]) for r in out["history"])
    assert out["results"]["gpu_peak_gb"] > 0


# --- The whole experiment runner, on a synthetic dataset -------------------------------------
@pytest.fixture
def synthetic_project(monkeypatch, synthetic_samples, tmp_path):
    """Point the configuration to a synthetic dataset and split, and shrink the images."""
    split_path = tmp_path / "split.json"
    names = [p.name for p in synthetic_samples]
    split_path.write_text(json.dumps({"train": names[:8], "val": names[8:]}), encoding="utf-8")
    monkeypatch.setattr(config, "TRAIN_DIR", synthetic_samples[0].parent)
    monkeypatch.setattr(config, "SPLIT_PATH", split_path)
    monkeypatch.setattr(config, "DEFAULT_IMAGE_SIZE", SIZE)
    monkeypatch.setattr(config, "EXPERIMENTS", {
        **config.EXPERIMENTS,
        # B without the download of the weights, same code path (two parameter groups)
        "B_offline": {**config.EXPERIMENTS["B"], "model_kwargs": {"pretrained": False}},
        # D without the download of the weights: strong augmentation and weight decay, two parameter groups
        "D_offline": {**config.EXPERIMENTS["D"], "model_kwargs": {"pretrained": False}},
        # E without the download of the weights: the ResNet50 encoder with its wider skip connections
        "E_offline": {**config.EXPERIMENTS["E"], "model_kwargs": {"pretrained": False}},
        # F without the download of the weights: final training on all the frames
        "F_offline": {**config.EXPERIMENTS["F"], "model_kwargs": {"pretrained": False}},
    })
    return tmp_path


@pytest.mark.parametrize("name", ["A", "B_offline", "C", "D_offline", "E_offline", "F_offline"])
def test_train_runs_every_experiment_with_the_same_code(synthetic_project, name):
    out = train(name, epochs=2, device="cpu", batch_size=4, num_workers=0,
                output_dir=synthetic_project / f"out_{name}", verbose=False)

    assert len(out["history"]) == 2 and np.isfinite(out["best_score"])
    model, state = load_checkpoint(out["best_path"])
    assert state["experiment"] == name
    assert isinstance(model, UNet if name == "A" else ResNetUNet)
    if name == "E_offline":
        assert isinstance(model, ResNet50UNet)
    expected_mean = config.EXPERIMENTS[name]["normalization"]
    if expected_mean == "imagenet":
        assert tuple(state["mean"]) == config.IMAGENET_MEAN and tuple(state["std"]) == config.IMAGENET_STD


def test_experiment_overrides_reach_the_optimiser_the_dataloaders_and_the_schedule(synthetic_project, monkeypatch):
    """An experiment that sets epochs, weight decay and augmentation must really use them, while one
    that sets nothing must fall back to the global values."""
    import src.train as train_module
    monkeypatch.setitem(config.EXPERIMENTS, "override", {
        **config.EXPERIMENTS["A"], "epochs": 2, "weight_decay": 0.123, "augmentation": "strong"})
    seen = {}
    real_optimizer, real_loaders = train_module.build_optimizer, train_module.create_dataloaders

    def spy_optimizer(model, **kwargs):
        seen["weight_decay"] = kwargs["weight_decay"]
        return real_optimizer(model, **kwargs)

    def spy_loaders(*args, **kwargs):
        seen["augmentation"] = kwargs["augmentation"]
        return real_loaders(*args, **kwargs)

    monkeypatch.setattr(train_module, "build_optimizer", spy_optimizer)
    monkeypatch.setattr(train_module, "create_dataloaders", spy_loaders)

    out = train("override", device="cpu", batch_size=4, num_workers=0,
                output_dir=synthetic_project / "out_override", verbose=False)
    assert len(out["history"]) == 2                          # epochs of the experiment, not config.EPOCHS
    assert seen == {"weight_decay": 0.123, "augmentation": "strong"}

    train("A", epochs=1, device="cpu", batch_size=4, num_workers=0,
          output_dir=synthetic_project / "out_default", verbose=False)
    assert seen == {"weight_decay": config.WEIGHT_DECAY, "augmentation": config.AUGMENTATION}


def test_an_explicit_epochs_argument_wins_over_the_experiment_setting(synthetic_project, monkeypatch):
    monkeypatch.setitem(config.EXPERIMENTS, "override", {**config.EXPERIMENTS["A"], "epochs": 5})
    out = train("override", epochs=1, device="cpu", batch_size=4, num_workers=0,
                output_dir=synthetic_project / "out_epochs", verbose=False)
    assert len(out["history"]) == 1


def test_compare_experiments_collects_the_results(synthetic_project):
    for name in ("A", "C"):
        train(name, epochs=1, device="cpu", batch_size=4, num_workers=0,
              output_dir=synthetic_project / f"experiment_{name}", verbose=False)
    table = compare_experiments(["A", "B", "C"], output_dir=synthetic_project)
    assert list(table.index) == ["A", "C"]                    # B was not run: it is simply missing
    assert {"score", "best_epoch", "params (M)", "gpu GB"} <= set(table.columns)


def test_resolve_settings_falls_back_to_the_global_values():
    assert resolve_settings(config.EXPERIMENTS["B"]) == {
        "epochs": config.EPOCHS, "weight_decay": config.WEIGHT_DECAY, "augmentation": config.AUGMENTATION,
        "encoder_lr_factor": 0.1, "final_training": False}
    d = resolve_settings(config.EXPERIMENTS["D"])
    assert d["epochs"] == config.EXPERIMENTS["D"]["epochs"] and d["augmentation"] == "strong"


def test_run_experiments_without_names_runs_every_defined_experiment(monkeypatch):
    import src.train as train_module
    ran = []
    monkeypatch.setattr(train_module, "train", lambda name, **kwargs: ran.append(name) or name)
    assert run_experiments() == {name: name for name in sorted(config.EXPERIMENTS)}
    assert ran == sorted(config.EXPERIMENTS) and "D" in ran


def test_comparison_table_shows_what_distinguishes_the_experiments(synthetic_project):
    for name in ("B_offline", "D_offline"):
        train(name, epochs=2, device="cpu", batch_size=4, num_workers=0,
              output_dir=synthetic_project / f"experiment_{name}", verbose=False)
    table = compare_experiments(["B_offline", "D_offline"], output_dir=synthetic_project)

    assert table.loc["B_offline", "model"] == "resnet18_unet" and table.loc["D_offline", "model"] == "resnet18_unet"
    assert table.loc["B_offline", "augmentation"] == "standard" and table.loc["D_offline", "augmentation"] == "strong"
    assert table.loc["D_offline", "weight decay"] > table.loc["B_offline", "weight decay"]
    assert table.loc["B_offline", "planned epochs"] == 2      # the epochs passed to train() are the ones stored
    assert table.loc["D_offline", "planned epochs"] == 2
    # losses at the best epoch come from the history of the run
    history = load_history("D_offline", synthetic_project)
    best = next(e for e in history if e["epoch"] == table.loc["D_offline", "best_epoch"])
    assert table.loc["D_offline", "train loss"] == pytest.approx(best["train_loss"], abs=1e-4)
    assert table.loc["D_offline", "val loss"] == pytest.approx(best["val_loss"], abs=1e-4)
    assert table["score"].dtype.kind == "f"                   # numeric columns are not left as text


def test_runs_stored_before_the_settings_existed_still_appear_in_the_table(synthetic_project):
    """results.json of experiments A, B and C was written without the "settings" key."""
    out = train("D_offline", epochs=1, device="cpu", batch_size=4, num_workers=0,
                output_dir=synthetic_project / "experiment_D", verbose=False)
    path = out["output_dir"] / "results.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    del results["settings"], results["model"]
    path.write_text(json.dumps(results), encoding="utf-8")

    row = compare_experiments(["D"], output_dir=synthetic_project).loc["D"]
    assert row["augmentation"] == "strong"                    # taken from config.EXPERIMENTS["D"]
    assert row["model"] == "resnet18_unet"


def test_comparison_plot_skips_experiments_that_have_not_run(synthetic_project):
    import matplotlib
    matplotlib.use("Agg")
    train("A", epochs=2, device="cpu", batch_size=4, num_workers=0,
          output_dir=synthetic_project / "experiment_A", verbose=False)
    fig = plot_comparison(["A", "B", "C", "D"], output_dir=synthetic_project)
    assert len(fig.axes) == 3
    assert all(len(ax.get_lines()) == 1 for ax in fig.axes)   # only A has curves
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ["A"]


def test_evaluation_runs_in_eval_mode_and_leaves_the_batch_norm_statistics_alone(setup):
    model, _, val_loader, criterion, _, _ = setup
    model.train()                                            # evaluate must switch to eval mode by itself
    buffers_before = [b.clone() for b in model.buffers()]
    evaluate(model, val_loader, criterion, CPU, False)
    assert not model.training
    assert all(torch.equal(a, b) for a, b in zip(buffers_before, model.buffers()))


def test_loading_a_checkpoint_never_asks_for_the_pretrained_weights(monkeypatch, tmp_path):
    """Experiment B stores pretrained=True in its settings. Loading must still build the network with
    random weights, because the real ones come from the checkpoint (no download, works offline)."""
    import torchvision
    model = ResNetUNet(pretrained=False)
    path = tmp_path / "best.pt"
    torch.save({"model_name": "resnet18_unet", "model_kwargs": {"pretrained": True},
                "model": model.state_dict()}, path)

    requested = {}
    real_resnet18 = torchvision.models.resnet18

    def recorder(weights=None, **kwargs):
        requested["weights"] = weights
        return real_resnet18(weights=None)

    monkeypatch.setattr(torchvision.models, "resnet18", recorder)
    loaded, _ = load_checkpoint(path)
    assert requested["weights"] is None
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), loaded.state_dict().values()))


# --- Final training (experiment F) -------------------------------------------------------------
def test_final_training_keeps_the_last_epoch_and_never_stops_early(setup, tmp_path):
    model, train_loader, val_loader, criterion, metadata, make = setup
    optimizer, scheduler = make(lr=0.0)                      # the score never changes: a normal run would stop
    out = fit(model, train_loader, val_loader, criterion, optimizer, scheduler, tmp_path, epochs=4,
              device=CPU, metadata=metadata, use_amp=False, patience=1, verbose=False, final_training=True)

    assert len(out["history"]) == 4                          # patience is ignored
    assert out["best_epoch"] == 4 and out["results"]["final_training"] is True
    assert out["best_path"].name == "final.pt"
    assert (tmp_path / "final.pt").exists() and not (tmp_path / "best.pt").exists()


def test_final_weights_are_those_of_the_last_epoch_even_if_an_earlier_epoch_scored_higher(setup, tmp_path):
    """A normal run keeps the weights of the best score. Here the score is made to peak at epoch 1."""
    import src.train as train_module
    model, train_loader, val_loader, criterion, metadata, make = setup
    optimizer, scheduler = make(lr=5e-3, epochs=3)
    scores = iter([0.9, 0.1, 0.2])
    real_evaluate = train_module.evaluate
    train_module.evaluate = lambda *args, **kwargs: {**real_evaluate(*args, **kwargs), "score": next(scores)}
    try:
        out = fit(model, train_loader, val_loader, criterion, optimizer, scheduler, tmp_path, epochs=3,
                  device=CPU, metadata=metadata, use_amp=False, patience=None, verbose=False, final_training=True)
    finally:
        train_module.evaluate = real_evaluate

    saved = torch.load(out["best_path"], weights_only=True)
    assert saved["epoch"] == 3
    assert all(torch.equal(saved["model"][k], v) for k, v in model.state_dict().items())


def test_final_training_uses_every_labelled_frame_including_the_purged_ones(synthetic_project, synthetic_samples, monkeypatch):
    import src.train as train_module
    names = [p.name for p in synthetic_samples]
    split_path = synthetic_project / "split_with_purged.json"
    split_path.write_text(json.dumps({"train": names[:6], "val": names[8:]}), encoding="utf-8")   # 2 frames purged
    monkeypatch.setattr(config, "SPLIT_PATH", split_path)
    seen = {}
    real_loaders = train_module.create_dataloaders

    def spy(train_samples, val_samples, **kwargs):
        seen["train"], seen["val"] = len(train_samples), len(val_samples)
        return real_loaders(train_samples, val_samples, **kwargs)

    monkeypatch.setattr(train_module, "create_dataloaders", spy)

    out = train("F_offline", epochs=1, device="cpu", batch_size=4, num_workers=0,
                output_dir=synthetic_project / "out_final", verbose=False)
    assert seen == {"train": len(synthetic_samples), "val": 4}          # all the frames; the validation ones only for monitoring
    model, state = load_checkpoint(out["output_dir"] / "final.pt")
    assert state["experiment"] == "F_offline" and state["settings"]["final_training"] is True

    train("B_offline", epochs=1, device="cpu", batch_size=4, num_workers=0,
          output_dir=synthetic_project / "out_normal", verbose=False)
    assert seen == {"train": 6, "val": 4}                               # a normal experiment is not affected
    assert not (synthetic_project / "out_normal" / "final.pt").exists()


def test_comparison_marks_final_runs_and_the_plot_leaves_them_out(synthetic_project):
    import matplotlib
    matplotlib.use("Agg")
    for name in ("B_offline", "F_offline"):
        train(name, epochs=2, device="cpu", batch_size=4, num_workers=0,
              output_dir=synthetic_project / f"experiment_{name}", verbose=False)
    table = compare_experiments(["B_offline", "F_offline"], output_dir=synthetic_project)
    assert table.loc["B_offline", "trained on"] == "train" and table.loc["F_offline", "trained on"] == "train+val"

    fig = plot_comparison(["B_offline", "F_offline"], output_dir=synthetic_project)
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ["B_offline"]


def test_history_plot_does_not_call_the_frames_of_a_final_run_a_validation_set(setup, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    model, train_loader, val_loader, criterion, metadata, make = setup
    optimizer, scheduler = make()
    history = fit(model, train_loader, val_loader, criterion, optimizer, scheduler, tmp_path, epochs=2,
                  device=CPU, metadata=metadata, use_amp=False, patience=None, verbose=False)["history"]

    def legends(fig):
        return [t.get_text() for ax in fig.axes[:2] for t in ax.get_legend().get_texts()]

    normal, final = legends(plot_history(history)), legends(plot_history(history, final_training=True))
    assert "validation" in normal and any(text.startswith("best") for text in normal)
    assert "validation" not in final and not any(text.startswith("best") for text in final)
    assert "seen frames (in the training set)" in final
