"""
tests/test_measure_vram.py

Tes jalan-kode untuk scripts/measure_vram.py. Tes ini jalan di CPU dengan
model kecil (ngf=8), jadi TIDAK mengukur memori. Angka VRAM yang sebenarnya
hanya bisa didapat dengan menjalankan skripnya di GPU kalian.
"""

import csv

import pytest
import torch

from models.discriminator import PatchDiscriminator
from models.generator import UNetGenerator
from scripts import measure_vram as mv


def _tiny_cpu_args(tmp_path, *extra):
    return ["--device", "cpu", "--img-sizes", "128", "--batches", "1", "2",
            "--ngfs", "8", "--ndf", "8", "--out", str(tmp_path / "vram.csv"), *extra]


def test_main_runs_on_cpu_and_writes_csv(tmp_path):
    results = mv.main(_tiny_cpu_args(tmp_path))
    # 1 img_size x 2 batch x 2 AMP x 1 ngf = 4 kombinasi
    assert len(results) == 4
    statuses = sorted(r.status for r in results)
    assert statuses == ["ok", "ok", "skipped", "skipped"]  # AMP dilewati di CPU
    for r in results:
        if r.status == "ok":
            assert r.g_params > 0 and r.d_params > 0 and r.iter_ms > 0
            assert r.peak_alloc_mib is None    # memori tidak diukur di CPU
            # Tanpa AMP, 1 iterasi pemanasan cukup untuk membentuk state Adam;
            # lalu 3 iterasi terukur (default) -> 4 step Adam untuk G dan D.
            assert r.warmup_iters == 1 and r.adam_ready is True
            assert r.g_steps == 4 and r.d_steps == 4
            assert r.note == ""

    with open(tmp_path / "vram.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert {"img_size", "batch", "amp", "ngf", "status", "peak_alloc_mib", "peak_reserved_mib",
            "steady_alloc_mib", "warmup_iters", "adam_ready", "g_steps", "d_steps"} <= set(rows[0])


def test_adam_ready_and_steps_track_real_optimizer_steps():
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    assert mv._adam_ready(opt) is False and mv._adam_steps(opt) == 0
    for _ in range(2):
        model(torch.randn(4, 3)).sum().backward()
        opt.step()
    assert mv._adam_ready(opt) is True and mv._adam_steps(opt) == 2


def test_skipped_steps_are_reported_not_hidden(tmp_path, monkeypatch):
    """Simulasi GradScaler yang selalu melewati step: state Adam tidak pernah
    terbentuk. Pemanasan harus berhenti di --max-warmup dan hasilnya ditandai."""
    calls = []

    def never_steps(*args, **kwargs):
        calls.append(1)
        return torch.tensor(0.0), torch.tensor(0.0)

    monkeypatch.setattr(mv, "train_step", never_steps)
    results = mv.main(_tiny_cpu_args(tmp_path, "--batches", "1", "--max-warmup", "3", "--iters", "2"))
    r = next(r for r in results if r.status == "ok")
    assert len(calls) == 3 + 2                   # 3 pemanasan (batas) + 2 terukur
    assert r.warmup_iters == 3 and r.adam_ready is False
    assert r.g_steps == 0 and "state Adam belum terbentuk" in r.note


def test_train_step_updates_both_models():
    torch.manual_seed(0)
    G = UNetGenerator(ngf=8, img_size=128).train()
    D = PatchDiscriminator(ndf=8).train()
    g_before = [p.detach().clone() for p in G.parameters()]
    d_before = [p.detach().clone() for p in D.parameters()]
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    cond, real = torch.randn(2, 3, 128, 128), torch.randn(2, 3, 128, 128)

    loss_d, loss_g = mv.train_step(G, D, opt_g, opt_d, None, None, cond, real,
                                   amp=False, lambda_l1=100.0)
    assert torch.isfinite(loss_d) and torch.isfinite(loss_g)
    assert any(not torch.equal(a, b) for a, b in zip(g_before, G.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(d_before, D.parameters()))


def test_uses_the_same_training_step_as_train_py():
    """Satu sumber kode: yang diukur adalah langkah training yang sesungguhnya."""
    import train as tr
    assert mv._train_step is tr.train_step
    assert mv.make_grad_scaler is tr.make_grad_scaler


def test_oom_is_recorded_and_sweep_continues(tmp_path, monkeypatch):
    calls = []

    def fake_oom(*args, **kwargs):
        calls.append(1)
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(mv, "train_step", fake_oom)
    results = mv.main(_tiny_cpu_args(tmp_path))
    ok_combos = [r for r in results if r.status != "skipped"]
    assert len(calls) == 2                       # sweep tidak berhenti di OOM pertama
    assert all(r.status == "OOM" for r in ok_combos)
    assert all("out of memory" in r.note for r in ok_combos)


def test_non_oom_error_is_not_swallowed(tmp_path, monkeypatch):
    def bug(*args, **kwargs):
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied")

    monkeypatch.setattr(mv, "train_step", bug)
    with pytest.raises(RuntimeError, match="shapes"):
        mv.main(_tiny_cpu_args(tmp_path))


@pytest.mark.parametrize("err,expected", [
    (torch.cuda.OutOfMemoryError("CUDA out of memory"), True),
    (RuntimeError("CUDA error: out of memory"), True),
    (RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"), True),
    (RuntimeError("size mismatch"), False),
    (ValueError("out of memory"), False),
])
def test_is_oom(err, expected):
    assert mv._is_oom(err) is expected


def test_table_shows_dash_for_missing_values():
    r = mv.ComboResult(256, 4, True, 64, "OOM", None, None, None, None, None,
                       None, None, None, None, None, None, "")
    table = mv.format_table([r])
    assert "OOM" in table and " - " in table


def test_cuda_required_by_default(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="CUDA tidak tersedia"):
        mv.main(["--out", "unused.csv"])
