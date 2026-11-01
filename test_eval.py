"""
tests/test_eval.py

Tes untuk eval.py: metrik dibandingkan dengan hitungan tangan, lalu evaluasi
end-to-end pada run kecil di CPU.
Jalankan: pytest tests/test_eval.py -v
"""

import csv
import dataclasses
import math
import shutil

import pytest
import torch
import torch.nn as nn

import eval as ev
import train as tr
from config import Config
from tests.helpers import make_pairs


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


def const(v, size=32):
    """Gambar [1, 3, size, size] bernilai konstan v (rentang [-1, 1])."""
    return torch.full((1, 3, size, size), float(v))


# ---------------------------------------------------------------------------
# 1. Metrik vs hitungan tangan
# ---------------------------------------------------------------------------
def test_l1_manual():
    assert ev.l1_per_image(const(0.2), const(0.0)).item() == pytest.approx(0.2)


def test_psnr_manual_and_identical():
    # 0.2 dan 0.0 di [-1, 1] -> 0.6 dan 0.5 di [0, 1] -> MSE 0.01 -> 10*log10(1/0.01) = 20 dB
    assert ev.psnr_per_image(const(0.2), const(0.0)).item() == pytest.approx(20.0, abs=1e-4)
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    assert torch.isinf(ev.psnr_per_image(x, x)).all()


def test_ssim_identical_is_one():
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    assert torch.allclose(ev.ssim_per_image(x, x), torch.ones(2), atol=1e-5)


@pytest.mark.parametrize("va,vb", [(0.2, 0.0), (-0.6, 0.4), (1.0, -1.0)])
def test_ssim_constant_images_closed_form(va, vb):
    """Dua gambar konstan: varians dan kovarians 0, jadi
    SSIM = (2ab + C1) / (a^2 + b^2 + C1), dengan a, b di [0, 1]."""
    a, b = (va + 1) / 2, (vb + 1) / 2
    c1 = 0.01 ** 2
    expected = (2 * a * b + c1) / (a * a + b * b + c1)
    assert ev.ssim_per_image(const(va), const(vb)).item() == pytest.approx(expected, rel=1e-4)


def test_ssim_symmetric_bounded_and_monotone_in_noise():
    x = torch.rand(1, 3, 64, 64) * 2 - 1
    y = torch.rand(1, 3, 64, 64) * 2 - 1
    assert ev.ssim_per_image(x, y).item() == pytest.approx(ev.ssim_per_image(y, x).item(), abs=1e-6)
    assert -1.0 <= ev.ssim_per_image(x, y).item() < 0.2           # dua noise acak: hampir tak mirip
    scores = [ev.ssim_per_image((x + s * torch.randn_like(x)).clamp(-1, 1), x).item()
              for s in (0.01, 0.1, 0.5)]
    assert scores[0] > scores[1] > scores[2]                      # makin berisik makin rendah


def test_ssim_rejects_tiny_image():
    with pytest.raises(ValueError, match="jendela"):
        ev.ssim_per_image(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8))


# ---------------------------------------------------------------------------
# 2. Mode dropout saat evaluasi
# ---------------------------------------------------------------------------
def test_enable_dropout_only_keeps_batchnorm_in_eval():
    G, _ = tr.build_models(Config(ngf=8, img_size=128, device="cpu"), torch.device("cpu"))
    G.eval()
    ev.enable_dropout_only(G)
    assert all(m.training for m in G.modules() if isinstance(m, nn.Dropout))
    assert not any(m.training for m in G.modules() if isinstance(m, nn.BatchNorm2d))


# ---------------------------------------------------------------------------
# 3. End-to-end pada run kecil
# ---------------------------------------------------------------------------
@pytest.fixture
def trained_runs(tmp_path):
    make_pairs(tmp_path / "data", n=20)           # split 16 / 2 / 2
    base = Config(data_root=str(tmp_path / "data"), img_size=128, load_size=144, batch_size=2,
                  num_workers=0, pin_memory=False, manifest_path=str(tmp_path / "manifest.json"),
                  ngf=8, ndf=8, epochs=1, log_every=4, device="cpu",
                  checkpoint_dir=str(tmp_path / "runs"))
    runs = []
    for name, over in (("main", {}), ("noskip", {"use_skip": False})):
        cfg = dataclasses.replace(base, run_name=name, **over)
        tr.train(cfg)
        runs.append(tr.run_dir(cfg))
    return tmp_path, runs


def test_evaluate_run_outputs(trained_runs):
    _, runs = trained_runs
    s = ev.evaluate_run(runs[0], device=torch.device("cpu"))
    assert s["n"] == 2 and s["split"] == "test" and s["epoch"] == 1
    assert 0.0 <= s["l1_mean"] <= 2.0 and math.isfinite(s["psnr_mean"]) and -1 <= s["ssim_mean"] <= 1
    with open(runs[0] / "eval_test_best.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 2
    from PIL import Image
    with Image.open(runs[0] / "eval_test_best_grid.png") as im:
        assert im.size == (3 * 128, 2 * 128)          # 2 gambar test, 3 kolom


def test_best_and_last_written_separately(trained_runs):
    _, runs = trained_runs
    ev.evaluate_run(runs[0], ckpt="best", device=torch.device("cpu"))
    ev.evaluate_run(runs[0], ckpt="last", device=torch.device("cpu"))
    for name in ("eval_test_best.csv", "eval_test_last.csv",
                 "eval_test_best_grid.png", "eval_test_last_grid.png"):
        assert (runs[0] / name).exists(), name


def test_main_compares_runs_and_reports_config(trained_runs):
    tmp_path, runs = trained_runs
    out = tmp_path / "perbandingan.csv"
    summaries = ev.main(["--runs", *map(str, runs), "--device", "cpu", "--out", str(out)])
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["run"] for r in rows] == ["main", "noskip"]
    assert [r["use_skip"] for r in rows] == ["True", "False"]
    assert summaries[0]["l1_mean"] != summaries[1]["l1_mean"]


def test_dropout_eval_is_active_and_reproducible(trained_runs, monkeypatch):
    """Catatan: di model mini yang baru dilatih beberapa iterasi, efek dropout ke
    output hanya ~1e-7 (jalur bottleneck nyaris nol, skip connection dominan).
    Karena itu yang diuji adalah modul Dropout-nya sendiri, bukan selisih L1."""
    _, runs = trained_runs
    seen = []
    original = nn.Dropout.forward

    def spy(self, x):
        out = original(self, x)
        seen.append((self.training, (out == 0).float().mean().item()))
        return out

    monkeypatch.setattr(nn.Dropout, "forward", spy)
    a = ev.evaluate_run(runs[0], dropout_eval=True, device=torch.device("cpu"))
    active = list(seen)
    seen.clear()
    ev.evaluate_run(runs[0], dropout_eval=False, device=torch.device("cpu"))
    inactive = list(seen)
    b = ev.evaluate_run(runs[0], dropout_eval=True, device=torch.device("cpu"))

    assert active and all(training for training, _ in active)
    assert sum(z for _, z in active) / len(active) > 0.3          # sekitar p = 0.5 nilai dinolkan
    assert inactive and not any(training for training, _ in inactive)
    assert a["l1_mean"] == b["l1_mean"]                           # seed tetap: bisa diulang


def test_data_root_override_after_moving_data(trained_runs):
    """Checkpoint dari Colab menyimpan path Colab; --data-root menggantinya."""
    tmp_path, runs = trained_runs
    shutil.move(str(tmp_path / "data"), str(tmp_path / "data_pindah"))
    with pytest.raises(FileNotFoundError):
        ev.evaluate_run(runs[0], device=torch.device("cpu"))
    s = ev.evaluate_run(runs[0], device=torch.device("cpu"),
                        overrides={"data_root": str(tmp_path / "data_pindah")})
    assert s["n"] == 2


def test_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="Checkpoint"):
        ev.evaluate_run(tmp_path / "tidak_ada", device=torch.device("cpu"))
