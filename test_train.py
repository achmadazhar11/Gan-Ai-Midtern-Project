"""
tests/test_train.py

Tes untuk train.py. Semua jalan di CPU dengan model kecil (ngf/ndf = 8, 128x128)
dan data sintetis format CMP (label .png palet + foto .jpg).
Jalankan: pytest tests/test_train.py -v
"""

import csv
import dataclasses

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

import train as tr
from config import Config
from tests.helpers import make_pairs


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Data dan config kecil
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_cfg(tmp_path):
    make_pairs(tmp_path / "data")
    return Config(data_root=str(tmp_path / "data"), img_size=128, load_size=144, batch_size=2,
                  num_workers=0, pin_memory=False, manifest_path=str(tmp_path / "manifest.json"),
                  ngf=8, ndf=8, epochs=2, log_every=1, device="cpu",
                  checkpoint_dir=str(tmp_path / "runs"), run_name="tes")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# 1. Config dipetakan benar ke konstruktor (nama dan nilai)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("over", [
    {},
    {"use_dropout": False},
    {"n_dropout": 1, "dropout_p": 0.3},
    {"norm": "instance"},
    {"norm": "none", "use_skip": False},
    {"n_layers": 1, "max_mult": 4},
])
def test_build_models_maps_every_field(tiny_cfg, over):
    cfg = dataclasses.replace(tiny_cfg, **over)
    G, D = tr.build_models(cfg, torch.device("cpu"))

    dropouts = [b.dropout for b in G.decoder if isinstance(b.dropout, nn.Dropout)]
    assert len(dropouts) == (cfg.n_dropout if cfg.use_dropout else 0)
    assert all(d.p == cfg.dropout_p for d in dropouts)
    assert G.use_skip == cfg.use_skip and G.img_size == cfg.img_size
    assert G.enc_channels[0] == cfg.ngf and max(G.enc_channels) == cfg.ngf * min(cfg.max_mult, 2 ** 6)

    norm_type = {"batch": nn.BatchNorm2d, "instance": nn.InstanceNorm2d, "none": nn.Identity}[cfg.norm]
    assert isinstance(G.encoder[1].norm, norm_type)
    assert D.n_layers == cfg.n_layers
    assert D.net[0].out_channels == cfg.ndf
    assert any(isinstance(m, norm_type) for m in D.net) or cfg.norm == "none"
    leaky = [m for m in G.modules() if isinstance(m, nn.LeakyReLU)]
    assert all(m.negative_slope == cfg.leaky_slope for m in leaky)


def test_optimizers_use_config_values(tiny_cfg):
    cfg = dataclasses.replace(tiny_cfg, lr=1e-3, beta1=0.4, beta2=0.99)
    G, D = tr.build_models(cfg, torch.device("cpu"))
    for opt in tr.build_optimizers(cfg, G, D):
        g = opt.param_groups[0]
        assert g["lr"] == 1e-3 and g["betas"] == (0.4, 0.99)


# ---------------------------------------------------------------------------
# 2. D dipanggil TERPISAH; fake di-detach HANYA saat update D
# ---------------------------------------------------------------------------
def test_d_called_separately_and_detach_only_for_d_update(tiny_cfg):
    G, D = tr.build_models(tiny_cfg, torch.device("cpu"))
    opt_g, opt_d = tr.build_optimizers(tiny_cfg, G, D)
    cond = torch.randn(2, 3, 128, 128)
    real = torch.randn(2, 3, 128, 128)

    calls = []
    def spy(module, inputs):
        _, img = inputs
        calls.append({"batch": img.shape[0], "img_requires_grad": img.requires_grad,
                      "d_params_trainable": next(module.parameters()).requires_grad})
    handle = D.register_forward_pre_hook(spy)
    tr.train_step(G, D, opt_g, opt_d, None, None, cond, real, amp=False, lambda_l1=100.0, d_loss_scale=0.5)
    handle.remove()

    assert len(calls) == 3
    assert [c["batch"] for c in calls] == [2, 2, 2]               # tidak pernah real+fake digabung (4)
    assert [c["img_requires_grad"] for c in calls] == [False, False, True]   # real, fake.detach(), fake
    assert [c["d_params_trainable"] for c in calls] == [True, True, False]   # D dibekukan saat update G
    assert all(p.requires_grad for p in D.parameters())          # dipulihkan setelahnya


def test_train_step_updates_both_and_returns_losses(tiny_cfg):
    G, D = tr.build_models(tiny_cfg, torch.device("cpu"))
    opt_g, opt_d = tr.build_optimizers(tiny_cfg, G, D)
    g0 = [p.detach().clone() for p in G.parameters()]
    d0 = [p.detach().clone() for p in D.parameters()]
    parts = tr.train_step(G, D, opt_g, opt_d, None, None, torch.randn(2, 3, 128, 128),
                          torch.randn(2, 3, 128, 128), amp=False, lambda_l1=100.0, d_loss_scale=0.5)
    assert set(parts) == {"d_loss", "d_real", "d_fake", "g_loss", "g_adv", "g_l1"}
    assert all(v.dim() == 0 and torch.isfinite(v) and not v.requires_grad for v in parts.values())
    assert any(not torch.equal(a, b) for a, b in zip(g0, G.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(d0, D.parameters()))


def test_d_step_does_not_touch_g(tiny_cfg):
    """Isolasi langkah 1: setelah update D, gradien G harus tetap kosong."""
    G, D = tr.build_models(tiny_cfg, torch.device("cpu"))
    _, opt_d = tr.build_optimizers(tiny_cfg, G, D)
    calls = {"n": 0}
    real_step = tr._backward_and_step
    def stop_after_d(loss, opt, scaler):
        real_step(loss, opt, scaler)
        calls["n"] += 1
        if calls["n"] == 1:
            assert all(p.grad is None for p in G.parameters())
            raise StopIteration
    tr._backward_and_step, saved = stop_after_d, tr._backward_and_step
    try:
        with pytest.raises(StopIteration):
            tr.train_step(G, D, None, opt_d, None, None, torch.randn(1, 3, 128, 128),
                          torch.randn(1, 3, 128, 128), amp=False, lambda_l1=100.0, d_loss_scale=0.5)
    finally:
        tr._backward_and_step = saved
    assert all(p.requires_grad for p in D.parameters())          # finally memulihkan D


# ---------------------------------------------------------------------------
# 3. Mode train/eval, shape, rentang
# ---------------------------------------------------------------------------
def test_evaluate_and_samples_restore_train_mode(tiny_cfg, tmp_path):
    from dataset import get_dataloader
    G, _ = tr.build_models(tiny_cfg, torch.device("cpu"))
    G.train()
    val_l1 = tr.evaluate_val(G, get_dataloader(tiny_cfg, "val"), torch.device("cpu"))
    assert G.training and 0.0 <= val_l1 <= 2.0          # |fake - real| <= 2 di rentang [-1, 1]

    a = torch.rand(3, 3, 128, 128) * 2 - 1
    path = tr.save_sample_grid(G, a, a, tmp_path / "grid.png")
    assert G.training
    with Image.open(path) as im:
        assert im.size == (3 * 128, 3 * 128) and im.mode == "RGB"   # 3 kolom A|G(A)|B, 3 baris


def test_generator_output_range_inside_train_step(tiny_cfg):
    G, D = tr.build_models(tiny_cfg, torch.device("cpu"))
    seen = []
    G.register_forward_hook(lambda m, i, o: seen.append(o.detach()))
    opt_g, opt_d = tr.build_optimizers(tiny_cfg, G, D)
    tr.train_step(G, D, opt_g, opt_d, None, None, torch.randn(2, 3, 128, 128) * 5,
                  torch.rand(2, 3, 128, 128) * 2 - 1, amp=False, lambda_l1=100.0, d_loss_scale=0.5)
    assert seen[0].shape == (2, 3, 128, 128)
    assert seen[0].min() >= -1 and seen[0].max() <= 1


# ---------------------------------------------------------------------------
# 4. Run kecil end-to-end, file output, resume
# ---------------------------------------------------------------------------
def test_full_tiny_run_writes_everything(tiny_cfg):
    summary = tr.train(tiny_cfg)
    out = tr.run_dir(tiny_cfg)
    # 12 pasang -> train 10 (round 9.6), batch 2 -> 5 iterasi/epoch, 2 epoch
    assert summary["last_epoch"] == 2 and summary["steps"] == 10
    for name in ("config.json", "last.pt", "best.pt", "log_train.csv", "log_val.csv",
                 "samples/epoch_001.png", "samples/epoch_002.png"):
        assert (out / name).exists(), name
    train_rows, val_rows = read_csv(out / "log_train.csv"), read_csv(out / "log_val.csv")
    assert len(train_rows) == 10 and [r["epoch"] for r in val_rows] == ["1", "2"]
    assert all(np.isfinite(float(r["g_l1"])) for r in train_rows)
    ckpt = tr.load_checkpoint(out / "last.pt", torch.device("cpu"))
    assert ckpt["epoch"] == 2 and ckpt["step"] == 10
    assert ckpt["config"]["ngf"] == 8


def test_log_rows_are_window_averages(tiny_cfg, monkeypatch):
    """Setiap baris log = rata-rata log_every iterasi; sisa di akhir ikut dicatat."""
    counter = {"i": 0}

    def fake_step(*args, **kwargs):
        counter["i"] += 1
        v = torch.tensor(float(counter["i"]))      # nilai loss = nomor iterasi
        return {k: v for k in ("d_loss", "d_real", "d_fake", "g_loss", "g_adv", "g_l1")}

    monkeypatch.setattr(tr, "train_step", fake_step)
    tr.train(dataclasses.replace(tiny_cfg, log_every=3))       # 10 iterasi total
    rows = read_csv(tr.run_dir(tiny_cfg) / "log_train.csv")
    assert [int(r["step"]) for r in rows] == [3, 6, 9, 10]
    assert [int(r["n_iter"]) for r in rows] == [3, 3, 3, 1]
    assert [float(r["g_l1"]) for r in rows] == [2.0, 5.0, 8.0, 10.0]   # (1+2+3)/3, (4+5+6)/3, ...
    assert [r["epoch"] for r in rows] == ["1", "2", "2", "2"]          # jendela boleh melintasi epoch


def test_resume_continues_epochs(tiny_cfg):
    tr.train(dataclasses.replace(tiny_cfg, epochs=1))
    summary = tr.train(dataclasses.replace(tiny_cfg, epochs=2), resume=True)
    assert summary["last_epoch"] == 2 and summary["steps"] == 10
    assert [r["epoch"] for r in read_csv(tr.run_dir(tiny_cfg) / "log_val.csv")] == ["1", "2"]


def test_resume_rejects_changed_architecture(tiny_cfg):
    tr.train(dataclasses.replace(tiny_cfg, epochs=1))
    with pytest.raises(ValueError, match="ngf"):
        tr.train(dataclasses.replace(tiny_cfg, epochs=2, ngf=16), resume=True)


def test_resume_without_checkpoint_fails(tiny_cfg):
    with pytest.raises(FileNotFoundError):
        tr.train(tiny_cfg, resume=True)


def test_overfit_tiny_subset_reduces_l1(tiny_cfg):
    """Sanity check: 2 pasangan tanpa augmentasi, L1 harus turun jelas."""
    # 150 iterasi: rasio akhir/awal sekitar 0,41 di seed 0, 1, 2 (diukur), jadi
    # ambang 0,6 punya margin lebar dan tidak rapuh terhadap seed.
    cfg = dataclasses.replace(tiny_cfg, subset_size=2, epochs=150, batch_size=2, lr=1e-3,
                              use_dropout=False, log_every=1)
    tr.train(cfg)
    rows = read_csv(tr.run_dir(cfg) / "log_train.csv")
    first = np.mean([float(r["g_l1"]) for r in rows[:5]])
    last = np.mean([float(r["g_l1"]) for r in rows[-5:]])
    assert last < 0.6 * first, (first, last)


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------
def test_parse_overrides_types():
    got = tr.parse_overrides(["epochs=5", "use_amp=false", "lambda_l1=10.5",
                              "data_root=C:/Users/x y/base", "subset_size=null",
                              "split_ratios=[0.7, 0.2, 0.1]"])
    assert got == {"epochs": 5, "use_amp": False, "lambda_l1": 10.5,
                   "data_root": "C:/Users/x y/base", "subset_size": None,
                   "split_ratios": (0.7, 0.2, 0.1)}
    with pytest.raises(ValueError, match="tidak dikenal"):
        tr.parse_overrides(["lamda_l1=5"])
    with pytest.raises(ValueError, match="key=value"):
        tr.parse_overrides(["epochs"])


def test_main_cli_preset_and_max_steps(tiny_cfg):
    sets = [f"{k}={v}" for k, v in {
        "data_root": tiny_cfg.data_root, "img_size": 128, "load_size": 144, "num_workers": 0,
        "pin_memory": "false", "manifest_path": tiny_cfg.manifest_path, "ngf": 8, "ndf": 8,
        "device": "cpu", "checkpoint_dir": tiny_cfg.checkpoint_dir}.items()]
    argv = ["--preset", "noskip", "--max-steps", "3"]
    for s in sets:
        argv += ["--set", s]
    summary = tr.main(argv)
    assert summary["steps"] == 3 and summary["run_dir"].endswith("noskip")
    saved = tr.load_checkpoint(tr.run_dir(dataclasses.replace(tiny_cfg, run_name="noskip")) / "last.pt",
                               torch.device("cpu"))
    assert saved["config"]["use_skip"] is False and saved["config"]["epochs"] == 30
