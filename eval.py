"""
eval.py

Evaluasi generator yang sudah dilatih pada split TEST (yang tidak dipakai
saat training maupun pemilihan checkpoint).

Metrik per gambar, dihitung sendiri tanpa jaringan pretrained:
- L1   : rata-rata |G(A) - B| di rentang [-1, 1] (sama dengan val L1 saat training).
- PSNR : 10 * log10(1 / MSE), dengan gambar dipetakan ke [0, 1]. Makin tinggi makin baik.
- SSIM : structural similarity (Wang dkk., 2004), jendela Gaussian 11x11,
         sigma 1.5, C1 = (0.01)^2, C2 = (0.03)^2, dihitung per channel lalu dirata-rata.

Penting: ketiga metrik membandingkan piksel per piksel dengan foto asli. Model
yang menghasilkan gambar buram sering mendapat L1/PSNR bagus walaupun tidak
realistis. Karena itu grid gambar tetap wajib dilihat bersama angka-angka ini.

Contoh:
    python eval.py --runs runs/main
    python eval.py --runs runs/main runs/lambda10 runs/noskip --ckpt best
    python eval.py --runs runs/main --data-root "C:/.../base" --dropout-eval

Output per run (di folder run): eval_<split>_<ckpt>.csv (per gambar) dan
eval_<split>_<ckpt>_grid.png, misalnya eval_test_best.csv dan eval_test_last.csv.

Pilih checkpoint dengan hati-hati: best.pt dipilih dari val L1 terendah, dan
pada GAN model awal yang buram sering punya L1 paling rendah. Bandingkan best
dan last, bersama grid gambarnya.
Output gabungan: --out (default runs/perbandingan_test.csv), satu baris per run.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
import numpy as np
from PIL import Image

from dataset import denormalize, get_dataloader
from train import build_models, load_checkpoint

N_GRID = 6   # jumlah gambar test di grid


# ===========================================================================
# Metrik
# ===========================================================================
def to_unit(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1], float32."""
    return (x.float().clamp(-1, 1) + 1) / 2


def l1_per_image(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """[B] rata-rata |fake - real| per gambar, di rentang [-1, 1]."""
    return (fake.float() - real.float()).abs().mean(dim=(1, 2, 3))


def psnr_per_image(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """[B] PSNR dalam dB dengan rentang data 1 (gambar di [0, 1]).
    Gambar identik (MSE = 0) menghasilkan +inf."""
    mse = ((to_unit(fake) - to_unit(real)) ** 2).mean(dim=(1, 2, 3))
    return 10 * torch.log10(1.0 / mse)


def _gaussian_window(size: int, sigma: float, channels: int, device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(coords.double() ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g[:, None] * g[None, :]                       # [size, size], jumlah = 1
    return window_2d.expand(channels, 1, size, size).contiguous()


def ssim_per_image(fake: torch.Tensor, real: torch.Tensor, window: int = 11,
                   sigma: float = 1.5) -> torch.Tensor:
    """[B] SSIM rata-rata per gambar (gambar dipetakan ke [0, 1]).

    mu = rata-rata lokal berbobot Gaussian; sigma^2 dan kovarians dihitung dari
    E[x^2] - mu^2 dan E[xy] - mu_x mu_y. Konvolusi tanpa padding (hanya posisi
    di mana jendela muat penuh), dilakukan per channel (groups = channel).

    Dihitung dalam float64: E[x^2] - mu^2 adalah pengurangan dua angka yang hampir
    sama, dan di float32 galatnya (~1e-7) tidak kecil dibanding C2 = 9e-4.
    """
    x, y = to_unit(fake).double(), to_unit(real).double()
    c = x.shape[1]
    if min(x.shape[-2:]) < window:
        raise ValueError(f"Gambar {tuple(x.shape[-2:])} lebih kecil dari jendela SSIM {window}")
    w = _gaussian_window(window, sigma, c, x.device)

    def blur(t):
        return F.conv2d(t, w, groups=c)

    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x ** 2
    var_y = blur(y * y) - mu_y ** 2
    cov = blur(x * y) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return ssim_map.mean(dim=(1, 2, 3)).float()


# ===========================================================================
# Evaluasi satu run
# ===========================================================================
def enable_dropout_only(model: nn.Module) -> None:
    """Aktifkan HANYA modul Dropout (BatchNorm tetap memakai running stats).
    Untuk mencoba perilaku paper Pix2Pix yang memakai dropout juga saat test."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def load_run(run: Path, ckpt: str, device, overrides: dict) -> tuple:
    """Muat config dan generator dari checkpoint sebuah run."""
    path = Path(run) / f"{ckpt}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ada: {path}")
    state = load_checkpoint(path, device)
    cfg = dataclasses.replace(Config(**state["config"]), **overrides)
    G, _ = build_models(cfg, device)
    G.load_state_dict(state["G"])
    return cfg, G, state


@torch.no_grad()
def evaluate_run(run: Path, *, split: str = "test", ckpt: str = "best", dropout_eval: bool = False,
                 device: torch.device | None = None, overrides: dict | None = None) -> dict:
    """Hitung metrik per gambar untuk satu run; simpan CSV per gambar dan grid."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overrides = dict(overrides or {})
    overrides.setdefault("subset_size", None)           # evaluasi selalu memakai split penuh
    cfg, G, state = load_run(run, ckpt, device, overrides)
    G.eval()
    if dropout_eval:
        enable_dropout_only(G)
        torch.manual_seed(0)       # dropout acak: seed tetap agar hasil bisa diulang

    loader = get_dataloader(cfg, split)
    rows, grid = [], []
    for batch in loader:
        a, b = batch["A"].to(device), batch["B"].to(device)
        fake = G(a)
        for path, l1, ps, ss in zip(batch["path"], l1_per_image(fake, b),
                                    psnr_per_image(fake, b), ssim_per_image(fake, b)):
            rows.append({"path": Path(path).name, "l1": round(l1.item(), 5),
                         "psnr": round(ps.item(), 3), "ssim": round(ss.item(), 4)})
        for i in range(min(a.shape[0], N_GRID - len(grid))):
            # baris grid dari output yang SAMA dengan yang dinilai metrik
            grid.append(torch.cat([denormalize(t[i]) for t in (a, fake, b)], dim=2))

    run = Path(run)
    with open(run / f"eval_{split}_{ckpt}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "l1", "psnr", "ssim"])
        w.writeheader()
        w.writerows(rows)
    img = torch.cat(grid, dim=1).permute(1, 2, 0).numpy()      # baris: A | G(A) | B
    Image.fromarray(np.ascontiguousarray(img)).save(run / f"eval_{split}_{ckpt}_grid.png")

    def mean_std(key):
        vals = [r[key] for r in rows if math.isfinite(r[key])]
        if not vals:
            return float("nan"), float("nan")
        m = sum(vals) / len(vals)
        return m, (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

    summary = {"run": run.name, "ckpt": ckpt, "epoch": state["epoch"], "step": state["step"],
               "split": split, "n": len(rows), "dropout_eval": dropout_eval,
               "lambda_l1": cfg.lambda_l1, "use_skip": cfg.use_skip, "n_layers": cfg.n_layers,
               "lr": cfg.lr, "batch_size": cfg.batch_size}
    for key in ("l1", "psnr", "ssim"):
        m, s = mean_std(key)
        summary[f"{key}_mean"], summary[f"{key}_std"] = round(m, 5), round(s, 5)
    summary["best_val_l1"] = round(state["best_val"], 5)
    return summary


def write_summary(summaries: list, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)
    return out


# ===========================================================================
# CLI
# ===========================================================================
def main(argv=None) -> list:
    p = argparse.ArgumentParser(description="Evaluasi Pix2Pix di split test.")
    p.add_argument("--runs", nargs="+", required=True, help="folder run, misalnya runs/main")
    p.add_argument("--ckpt", default="best", choices=["best", "last"])
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--dropout-eval", action="store_true", help="dropout aktif saat inferensi")
    p.add_argument("--data-root", default=None, help="ganti data_root dari checkpoint (pindah mesin)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None, choices=["cpu", "cuda"])
    p.add_argument("--out", default="runs/perbandingan_test.csv")
    args = p.parse_args(argv)

    overrides = {"num_workers": args.num_workers, "pin_memory": False}
    if args.data_root:
        overrides["data_root"] = args.data_root
    device = torch.device(args.device) if args.device else None

    summaries = []
    for run in args.runs:
        s = evaluate_run(Path(run), split=args.split, ckpt=args.ckpt, dropout_eval=args.dropout_eval,
                         device=device, overrides=overrides)
        summaries.append(s)
        print(f"{s['run']:>12} (epoch {s['epoch']}, {s['ckpt']}): L1 {s['l1_mean']:.4f} | "
              f"PSNR {s['psnr_mean']:.2f} dB | SSIM {s['ssim_mean']:.4f} | n={s['n']}")
    out = write_summary(summaries, Path(args.out))
    print(f"Ringkasan disimpan ke {out}")
    return summaries


if __name__ == "__main__":
    main()
