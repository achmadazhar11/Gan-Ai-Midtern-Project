"""
train.py

Loop training Pix2Pix. Satu iterasi:

    1. Update D: D(cond, real) dan D(cond, fake.detach()), dipanggil TERPISAH.
    2. Update G: D(cond, fake) TANPA detach, plus L1(fake, real).

Contoh (dari root repo):
    python train.py --set data_root="C:/.../CMP_facade_DB_base/base"
    python train.py --preset lambda10 --set data_root=data/base
    python train.py --config runs/main/config.json --resume
    python train.py --set epochs=5 --set subset_size=8          # overfit test

Isi folder run (runs/<run_name>/):
    config.json        config lengkap run ini
    log_train.csv      rata-rata loss per log_every iterasi (kolom n_iter)
    log_val.csv        L1 validasi per epoch
    samples/epoch_XXX.png   grid tetap A | G(A) | B dari 4 gambar val pertama
    last.pt, best.pt   checkpoint terakhir dan dengan L1 validasi terbaik

Catatan Windows: DataLoader dengan num_workers > 0 memakai proses "spawn", yang
meng-import ulang file ini di setiap worker. Karena itu semua kode yang
menjalankan training berada di dalam main(), dan main() hanya dipanggil dari
`if __name__ == "__main__":`.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import (
    Config,
    amp_enabled,
    get_device,
    get_preset,
    load_config,
    run_dir,
    save_config,
    set_seed,
)
from dataset import denormalize, get_dataloader
from losses import discriminator_loss, generator_loss
from models.discriminator import PatchDiscriminator
from models.generator import UNetGenerator

N_SAMPLES = 4   # jumlah pasangan val di grid sampel

# Field yang boleh berbeda saat melanjutkan run (resume). Field lain, misalnya
# arsitektur, lr, atau lambda, harus sama dengan isi checkpoint.
RESUME_MUTABLE = {"epochs", "num_workers", "pin_memory", "log_every", "device",
                  "checkpoint_dir", "data_root", "manifest_path"}


# ===========================================================================
# Membangun model dan optimizer dari Config
# ===========================================================================
def build_models(cfg: Config, device: torch.device) -> tuple:
    """G dan D dengan argumen yang dipetakan satu per satu dari Config."""
    G = UNetGenerator(
        in_ch=3, out_ch=3, ngf=cfg.ngf, img_size=cfg.img_size,
        use_dropout=cfg.use_dropout, use_skip=cfg.use_skip, norm=cfg.norm,
        n_dropout=cfg.n_dropout, dropout_p=cfg.dropout_p, max_mult=cfg.max_mult,
        leaky_slope=cfg.leaky_slope, init_std=cfg.init_std,
    )
    D = PatchDiscriminator(
        in_ch=6, ndf=cfg.ndf, n_layers=cfg.n_layers, norm=cfg.norm,
        max_mult=cfg.max_mult, leaky_slope=cfg.leaky_slope, init_std=cfg.init_std,
    )
    return G.to(device), D.to(device)


def build_optimizers(cfg: Config, G, D) -> tuple:
    betas = (cfg.beta1, cfg.beta2)
    return (torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=betas),
            torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=betas))


def make_grad_scaler(enabled: bool):
    """GradScaler untuk AMP, atau None jika AMP mati.

    torch.amp.GradScaler("cuda") ada di PyTorch baru; torch.cuda.amp.GradScaler
    (nama lama, sudah usang) dipakai sebagai cadangan untuk versi lama.
    """
    if not enabled:
        return None
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()


# ===========================================================================
# Satu langkah training
# ===========================================================================
def _backward_and_step(loss, optimizer, scaler):
    """Tanpa AMP: backward + step. Dengan AMP: loss diskalakan agar gradien fp16
    tidak underflow; step dilewati otomatis jika gradien berisi inf/NaN."""
    if scaler is None:
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def train_step(G, D, opt_g, opt_d, scaler_g, scaler_d, cond, real, *,
               amp: bool, lambda_l1: float, d_loss_scale: float) -> dict:
    """Satu iterasi: update D lalu update G. Mengembalikan komponen loss
    (tensor 0-dim yang sudah di-detach; .item() dipanggil hanya saat logging)."""
    device_type = cond.device.type

    # ----- 1. Update D -----
    opt_d.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp):
        fake = G(cond)                          # graph G disimpan untuk langkah 2
        d_real = D(cond, real)                  # panggilan TERPISAH: real ...
        d_fake = D(cond, fake.detach())         # ... dan fake; detach -> G tidak ikut
        loss_d, parts_d = discriminator_loss(d_real, d_fake, scale=d_loss_scale)
    _backward_and_step(loss_d, opt_d, scaler_d)

    # ----- 2. Update G -----
    # Parameter D dibekukan sementara: gradien tetap mengalir LEWAT D ke fake
    # (ke G), tetapi tidak dihitung UNTUK bobot D. Hemat memori dan waktu, dan
    # gradien D dari langkah G tidak tercampur.
    D.requires_grad_(False)
    try:
        opt_g.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp):
            d_fake_for_g = D(cond, fake)        # TANPA detach
            loss_g, parts_g = generator_loss(d_fake_for_g, fake, real, lambda_l1)
        _backward_and_step(loss_g, opt_g, scaler_g)
    finally:
        D.requires_grad_(True)

    return {"d_loss": loss_d.detach(), **parts_d, "g_loss": loss_g.detach(), **parts_g}


# ===========================================================================
# Evaluasi dan sampel
# ===========================================================================
@torch.no_grad()
def evaluate_val(G, loader, device) -> float:
    """Rata-rata L1 per piksel antara G(A) dan B di split val.

    G memakai mode eval (BatchNorm memakai running stats, Dropout mati).
    Mode G dikembalikan seperti semula setelahnya.
    """
    was_training = G.training
    G.eval()
    total, count = 0.0, 0
    for batch in loader:
        a, b = batch["A"].to(device), batch["B"].to(device)
        fake = G(a)
        total += (fake.float() - b).abs().sum().item()
        count += b.numel()
    G.train(was_training)
    return total / max(count, 1)


@torch.no_grad()
def save_sample_grid(G, a, b, path: Path) -> Path:
    """Simpan grid: setiap baris A | G(A) | B (uint8 PNG). Mode G dikembalikan."""
    was_training = G.training
    G.eval()
    fake = G(a)
    G.train(was_training)
    rows = [torch.cat([denormalize(x) for x in (a[i], fake[i], b[i])], dim=2)   # sambung lebar
            for i in range(a.shape[0])]
    grid = torch.cat(rows, dim=1).permute(1, 2, 0).numpy()                      # CHW -> HWC
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(grid)).save(path)
    return path


# ===========================================================================
# Checkpoint
# ===========================================================================
def save_checkpoint(path: Path, *, G, D, opt_g, opt_d, scaler_g, scaler_d,
                    epoch: int, step: int, best_val: float, cfg: Config) -> None:
    state = {
        "G": G.state_dict(), "D": D.state_dict(),
        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
        "scaler_g": scaler_g.state_dict() if scaler_g else None,
        "scaler_d": scaler_d.state_dict() if scaler_d else None,
        "epoch": epoch, "step": step, "best_val": best_val,
        "config": dataclasses.asdict(cfg),
    }
    # Tulis ke file sementara lalu ganti nama: checkpoint tidak pernah setengah
    # jadi walaupun proses terhenti saat menyimpan.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    # weights_only=True: hanya tensor dan tipe dasar yang dimuat, tanpa
    # menjalankan kode sembarang dari file checkpoint.
    return torch.load(path, map_location=device, weights_only=True)


def check_resume_compatible(saved: dict, cfg: Config) -> None:
    current = dataclasses.asdict(cfg)
    diff = {k: (saved.get(k), current[k]) for k in current
            if k not in RESUME_MUTABLE and _norm(saved.get(k)) != _norm(current[k])}
    if diff:
        raise ValueError(f"Config berbeda dari checkpoint untuk field {diff}. "
                         "Resume hanya boleh mengubah: " + ", ".join(sorted(RESUME_MUTABLE)))


def _norm(v):
    return list(v) if isinstance(v, tuple) else v   # JSON/torch.save: tuple vs list


# ===========================================================================
# Log CSV
# ===========================================================================
TRAIN_FIELDS = ["epoch", "step", "n_iter", "d_loss", "d_real", "d_fake",
                "g_loss", "g_adv", "g_l1", "sec"]
VAL_FIELDS = ["epoch", "step", "val_l1", "epoch_sec"]


class LossWindow:
    """Kumpulkan loss beberapa iterasi lalu catat RATA-RATANYA.

    Satu titik di grafik = rata-rata `n_iter` iterasi (biasanya log_every), bukan
    nilai satu batch. Dengan batch 2, nilai satu batch sangat berisik sehingga
    grafiknya sulit dibaca. Penjumlahan dilakukan pada tensor (tetap di GPU,
    tanpa .item()), jadi GPU hanya perlu sinkron saat baris log ditulis.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sums, self.n, self.t0 = {}, 0, time.perf_counter()

    def add(self, parts: dict):
        for k, v in parts.items():
            self.sums[k] = self.sums.get(k, 0) + v
        self.n += 1

    def pop_row(self, epoch: int, step: int) -> dict:
        row = {"epoch": epoch, "step": step, "n_iter": self.n,
               **{k: round(float(v / self.n), 5) for k, v in self.sums.items()},
               "sec": round(time.perf_counter() - self.t0, 2)}
        self.reset()
        return row


def _append_csv(path: Path, fields: list, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


# ===========================================================================
# Loop training
# ===========================================================================
def train(cfg: Config, *, resume: bool = False, deterministic: bool = False,
          max_steps: int | None = None) -> dict:
    """Jalankan training sesuai cfg. Mengembalikan ringkasan run.

    max_steps: batasi jumlah iterasi total (untuk uji cepat); None = semua epoch.
    """
    set_seed(cfg.seed, deterministic=deterministic)   # sebelum model dan loader dibuat
    device = get_device(cfg)
    amp = amp_enabled(cfg, device)
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)

    G, D = build_models(cfg, device)
    opt_g, opt_d = build_optimizers(cfg, G, D)
    scaler_g, scaler_d = make_grad_scaler(amp), make_grad_scaler(amp)
    train_loader = get_dataloader(cfg, "train")
    val_loader = get_dataloader(cfg, "val")

    # Sampel tetap: N_SAMPLES pasangan val pertama (tanpa augmentasi), sama tiap epoch.
    val_ds = val_loader.dataset
    idx = range(min(N_SAMPLES, len(val_ds)))
    sample_a = torch.stack([val_ds[i]["A"] for i in idx]).to(device)
    sample_b = torch.stack([val_ds[i]["B"] for i in idx]).to(device)

    start_epoch, step, best_val = 1, 0, float("inf")
    if resume:
        ckpt_path = out / "last.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Tidak ada checkpoint untuk resume: {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path, device)
        check_resume_compatible(ckpt["config"], cfg)
        G.load_state_dict(ckpt["G"]); D.load_state_dict(ckpt["D"])
        opt_g.load_state_dict(ckpt["opt_g"]); opt_d.load_state_dict(ckpt["opt_d"])
        if scaler_g and ckpt["scaler_g"]:
            scaler_g.load_state_dict(ckpt["scaler_g"]); scaler_d.load_state_dict(ckpt["scaler_d"])
        start_epoch, step, best_val = ckpt["epoch"] + 1, ckpt["step"], ckpt["best_val"]
        print(f"Resume dari epoch {ckpt['epoch']} (step {step}, val L1 terbaik {best_val:.4f})")
    save_config(cfg, out)

    print(f"Run '{cfg.run_name}' di {out} | device={device} | AMP={amp} | "
          f"train={len(train_loader.dataset)} val={len(val_ds)} | "
          f"G={G.num_parameters():,} D={D.num_parameters():,} parameter")

    def log_window(epoch_now):
        row = window.pop_row(epoch_now, step)
        _append_csv(out / "log_train.csv", TRAIN_FIELDS, row)
        print(f"  epoch {epoch_now} step {step} (rata-rata {row['n_iter']} iterasi): "
              f"D {row['d_loss']:.3f} | G adv {row['g_adv']:.3f} L1 {row['g_l1']:.3f}")

    window = LossWindow()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.epochs + 1):
        G.train(); D.train()
        t_epoch = time.perf_counter()
        for batch in train_loader:
            cond = batch["A"].to(device, non_blocking=True)
            real = batch["B"].to(device, non_blocking=True)
            parts = train_step(G, D, opt_g, opt_d, scaler_g, scaler_d, cond, real,
                               amp=amp, lambda_l1=cfg.lambda_l1, d_loss_scale=cfg.d_loss_scale)
            step += 1
            window.add(parts)
            if window.n == cfg.log_every:      # jendela penuh: catat rata-ratanya
                log_window(epoch)
            if max_steps is not None and step >= max_steps:
                break

        val_l1 = evaluate_val(G, val_loader, device)
        epoch_sec = time.perf_counter() - t_epoch
        _append_csv(out / "log_val.csv", VAL_FIELDS,
                    {"epoch": epoch, "step": step, "val_l1": round(val_l1, 5),
                     "epoch_sec": round(epoch_sec, 1)})
        save_sample_grid(G, sample_a, sample_b, out / "samples" / f"epoch_{epoch:03d}.png")

        ckpt_kwargs = dict(G=G, D=D, opt_g=opt_g, opt_d=opt_d, scaler_g=scaler_g,
                           scaler_d=scaler_d, epoch=epoch, step=step, cfg=cfg)
        if val_l1 < best_val:
            best_val = val_l1
            save_checkpoint(out / "best.pt", best_val=best_val, **ckpt_kwargs)
        save_checkpoint(out / "last.pt", best_val=best_val, **ckpt_kwargs)
        print(f"Epoch {epoch}/{cfg.epochs}: val L1 {val_l1:.4f} (terbaik {best_val:.4f}), {epoch_sec:.1f} dtk")

        if max_steps is not None and step >= max_steps:
            break

    if window.n:   # sisa iterasi yang belum genap satu jendela di akhir training
        log_window(epoch)
    return {"run_dir": str(out), "last_epoch": epoch, "steps": step, "best_val_l1": best_val}


# ===========================================================================
# CLI
# ===========================================================================
def parse_overrides(pairs: list) -> dict:
    """['epochs=5', 'use_amp=false', 'data_root=C:/x'] -> dict bertipe.

    Nilai dibaca sebagai JSON jika bisa (angka, true/false, null, list),
    selain itu dianggap string. Nama field divalidasi oleh Config.
    """
    known = {f.name for f in dataclasses.fields(Config)}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set harus berbentuk key=value, didapat {pair!r}")
        key, raw = pair.split("=", 1)
        key = key.strip()
        if key not in known:
            raise ValueError(f"Field config tidak dikenal: {key!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        if key == "split_ratios" and isinstance(value, list):
            value = tuple(value)
        out[key] = value
    return out


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="Training Pix2Pix.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--config", help="muat Config dari JSON (misalnya runs/main/config.json)")
    src.add_argument("--preset", help="preset ablation dari config.get_preset")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="ganti satu field config; boleh berulang")
    p.add_argument("--resume", action="store_true", help="lanjutkan dari runs/<run_name>/last.pt")
    p.add_argument("--deterministic", action="store_true", help="cuDNN deterministik (lebih lambat)")
    p.add_argument("--max-steps", type=int, default=None, help="batas iterasi total (uji cepat)")
    args = p.parse_args(argv)

    if args.config:
        cfg = load_config(args.config)
    elif args.preset:
        cfg = get_preset(args.preset)
    else:
        cfg = Config()
    cfg = dataclasses.replace(cfg, **parse_overrides(args.set))
    return train(cfg, resume=args.resume, deterministic=args.deterministic, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
