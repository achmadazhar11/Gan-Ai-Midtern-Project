"""
scripts/measure_vram.py

Ukur puncak memori GPU satu langkah training Pix2Pix (update D lalu update G)
dengan data dummy (torch.randn), untuk setiap kombinasi:

    img_size {256, 128} x batch {1, 2, 4} x AMP {off, on} x ngf {64, 32}

Contoh (dari root repo, di laptop dengan GPU):
    python scripts/measure_vram.py
    python scripts/measure_vram.py --out hasil_vram.csv --iters 3
    python scripts/measure_vram.py --img-sizes 128 --batches 4 --amp off --cudnn-deterministic

Alur per kombinasi:
1. Pemanasan: ulangi langkah training sampai state Adam G dan D terbentuk
   (maks --max-warmup). Dengan AMP, GradScaler bisa melewati beberapa step awal.
2. Iterasi terukur (--iters), dengan state Adam sudah ada seperti saat training.

Yang dicatat per kombinasi:
- peak_alloc_mib    : torch.cuda.max_memory_allocated() sejak model dibuat,
                      termasuk pemanasan (training sungguhan juga harus lolos ini).
- peak_reserved_mib : torch.cuda.max_memory_reserved(), memori yang dipegang
                      caching allocator. Ini lebih dekat ke angka yang terlihat
                      di nvidia-smi / Task Manager, tetapi keduanya BELUM
                      termasuk CUDA context (beberapa ratus MiB, di luar PyTorch).
- steady_alloc_mib  : puncak khusus iterasi terukur. Jika jauh di bawah
                      peak_alloc_mib, lonjakan terjadi sekali di awal saja.
- adam_ready, warmup_iters, g_steps, d_steps: bukti bahwa state Adam sudah ada.

Skrip ini hanya meng-import model; file model tidak diubah.
Langkah training diambil dari train.train_step (yang memakai losses.py), jadi
yang diukur persis langkah yang dijalankan saat training. Di langkah update G,
parameter D dibekukan (requires_grad False), seperti di train.py.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch

# Agar `import models...` berhasil saat skrip dijalankan dari mana pun.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.discriminator import PatchDiscriminator  # noqa: E402
from models.generator import UNetGenerator  # noqa: E402
from train import make_grad_scaler  # noqa: E402
from train import train_step as _train_step  # noqa: E402

MIB = 2 ** 20


@dataclass
class ComboResult:
    img_size: int
    batch: int
    amp: bool
    ngf: int
    status: str                       # "ok", "OOM", atau "skipped"
    peak_alloc_mib: float | None      # puncak sejak model dibuat (termasuk pemanasan)
    peak_reserved_mib: float | None
    pct_of_total: float | None        # peak_reserved / total VRAM GPU, dalam %
    steady_alloc_mib: float | None    # puncak khusus iterasi terukur (setelah state Adam ada)
    iter_ms: float | None             # waktu iterasi terukur terakhir
    warmup_iters: int | None          # iterasi pemanasan sampai state Adam G dan D terbentuk
    adam_ready: bool | None           # True jika state Adam G dan D sudah terbentuk
    g_steps: int | None               # langkah Adam G yang benar-benar terjadi
    d_steps: int | None               # langkah Adam D yang benar-benar terjadi
    g_params: int | None
    d_params: int | None
    note: str


# ---------------------------------------------------------------------------
# Satu langkah training: SAMA PERSIS dengan train.py (satu sumber kode)
# ---------------------------------------------------------------------------
def train_step(G, D, opt_g, opt_d, scaler_g, scaler_d, cond, real, amp, lambda_l1,
               d_loss_scale=0.5):
    """Memanggil train.train_step, lalu mengembalikan (loss_d, loss_g).

    Fungsi pembungkus ini tetap ada (dengan nama yang sama) supaya tes bisa
    menggantinya dengan versi palsu, misalnya untuk mensimulasikan OOM.
    """
    parts = _train_step(G, D, opt_g, opt_d, scaler_g, scaler_d, cond, real,
                        amp=amp, lambda_l1=lambda_l1, d_loss_scale=d_loss_scale)
    return parts["d_loss"], parts["g_loss"]


def _adam_ready(opt) -> bool:
    """True jika SEMUA parameter optimizer sudah punya state Adam (exp_avg).

    State dibuat pada optimizer.step() pertama yang benar-benar berjalan. Dengan
    AMP, GradScaler melewati step jika gradien berisi inf/NaN (biasanya di iterasi
    awal, saat faktor skala masih terlalu besar), sehingga state belum ada.
    """
    params = [p for group in opt.param_groups for p in group["params"]]
    return all(p in opt.state and "exp_avg" in opt.state[p] for p in params)


def _adam_steps(opt) -> int:
    """Jumlah step Adam yang benar-benar terjadi (minimum di antara parameter)."""
    steps = [float(st["step"]) for st in opt.state.values() if "step" in st]
    return int(min(steps)) if steps else 0


def _is_oom(err: BaseException) -> bool:
    """OOM resmi, plus RuntimeError dari cuDNN/cuBLAS yang sebenarnya kehabisan memori."""
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    msg = str(err)
    return isinstance(err, RuntimeError) and ("out of memory" in msg or "ALLOC_FAILED" in msg)


def _cleanup(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Satu kombinasi
# ---------------------------------------------------------------------------
def _run_combo_inner(args, img_size, batch, amp, ngf, device, result: dict):
    """Semua tensor dan model hidup di dalam fungsi ini, sehingga otomatis
    dibebaskan saat fungsi selesai (termasuk ketika terjadi OOM)."""
    torch.manual_seed(args.seed)
    G = UNetGenerator(ngf=ngf, img_size=img_size, use_dropout=args.use_dropout,
                      use_skip=args.use_skip, norm=args.g_norm).to(device).train()
    D = PatchDiscriminator(in_ch=6, ndf=args.ndf, n_layers=args.n_layers,
                           norm=args.d_norm).to(device).train()
    result["g_params"] = G.num_parameters()
    result["d_params"] = D.num_parameters()

    betas = (args.beta1, args.beta2)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=betas)
    # Dua scaler terpisah: setiap optimizer punya siklus scale->step->update sendiri.
    scaler_g = make_grad_scaler(amp)
    scaler_d = make_grad_scaler(amp)

    cond = torch.randn(batch, 3, img_size, img_size, device=device).clamp_(-1, 1)
    real = torch.randn(batch, 3, img_size, img_size, device=device).clamp_(-1, 1)

    def step():
        return train_step(G, D, opt_g, opt_d, scaler_g, scaler_d, cond, real, amp, args.lambda_l1)

    # ----- Pemanasan: ulangi sampai state Adam G dan D terbentuk -----
    # Tanpa AMP biasanya cukup 1 iterasi. Dengan AMP bisa beberapa iterasi,
    # karena GradScaler melewati step selama gradien yang diskalakan masih inf.
    warm = 0
    while warm < args.max_warmup and not (_adam_ready(opt_g) and _adam_ready(opt_d)):
        step()
        warm += 1
    result["warmup_iters"] = warm

    if device.type == "cuda":
        torch.cuda.synchronize()
        # Simpan puncak sejauh ini, lalu reset agar puncak iterasi terukur tercatat terpisah.
        result["warm_peak"] = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
        torch.cuda.reset_peak_memory_stats()

    # ----- Iterasi terukur (state Adam sudah ada) -----
    for _ in range(args.iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss_d, loss_g = step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        result["iter_ms"] = (time.perf_counter() - t0) * 1000

    result["adam_ready"] = _adam_ready(opt_g) and _adam_ready(opt_d)
    result["g_steps"] = _adam_steps(opt_g)
    result["d_steps"] = _adam_steps(opt_d)
    notes = []
    if not result["adam_ready"]:
        notes.append(f"state Adam belum terbentuk setelah {warm} iterasi pemanasan; angka meremehkan memori")
    if not (torch.isfinite(loss_d) and torch.isfinite(loss_g)):
        notes.append("loss tidak finite")
    result["note"] = "; ".join(notes)


def run_combo(args, img_size, batch, amp, ngf, device) -> ComboResult:
    result = {"g_params": None, "d_params": None, "iter_ms": None, "warmup_iters": None,
              "adam_ready": None, "g_steps": None, "d_steps": None, "note": ""}
    if amp and device.type != "cuda":
        return ComboResult(img_size, batch, amp, ngf, "skipped", None, None, None, None,
                           None, None, None, None, None, None, None,
                           "AMP fp16 hanya diukur di CUDA")

    _cleanup(device)
    if device.type == "cuda":
        start_alloc = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()   # puncak dihitung ulang dari sini
        if start_alloc > 0:
            result["note"] = f"sisa memori sebelum mulai: {start_alloc / MIB:.1f} MiB"

    status = "ok"
    try:
        _run_combo_inner(args, img_size, batch, amp, ngf, device, result)
    except Exception as err:  # noqa: BLE001 - hanya OOM yang ditelan, lainnya dilempar ulang
        if not _is_oom(err):
            raise
        status = "OOM"
        result["note"] = str(err).splitlines()[0][:120]

    peak_alloc = peak_reserved = pct = steady_alloc = None
    if device.type == "cuda":
        torch.cuda.synchronize()
        # Puncak sejak reset terakhir: iterasi terukur, atau seluruh run jika OOM
        # terjadi saat pemanasan. Puncak total = maksimum dengan puncak pemanasan.
        since_reset = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
        warm_alloc, warm_resv = result.get("warm_peak", (0, 0))
        peak_alloc = max(warm_alloc, since_reset[0]) / MIB
        peak_reserved = max(warm_resv, since_reset[1]) / MIB
        if "warm_peak" in result:
            steady_alloc = since_reset[0] / MIB
        total = torch.cuda.get_device_properties(device).total_memory / MIB
        pct = 100 * peak_reserved / total
    _cleanup(device)

    return ComboResult(img_size, batch, amp, ngf, status,
                       peak_alloc, peak_reserved, pct, steady_alloc, result["iter_ms"],
                       result["warmup_iters"], result["adam_ready"],
                       result["g_steps"], result["d_steps"],
                       result["g_params"], result["d_params"], result["note"])


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_csv(results: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(ComboResult)])
        writer.writeheader()
        for r in results:
            row = asdict(r)
            for key in ("peak_alloc_mib", "peak_reserved_mib", "pct_of_total",
                        "steady_alloc_mib", "iter_ms"):
                if row[key] is not None:
                    row[key] = round(row[key], 1)
            writer.writerow(row)


def format_table(results: list) -> str:
    def num(v, fmt):
        return "-" if v is None else format(v, fmt)

    def adam(r):
        if r.adam_ready is None:
            return "-"
        return f"{'ya' if r.adam_ready else 'TIDAK'}/{r.warmup_iters}"

    header = f"{'img':>4} {'batch':>5} {'amp':>4} {'ngf':>4} {'status':>7} " \
             f"{'alloc MiB':>10} {'resv MiB':>9} {'steady':>7} {'% VRAM':>7} " \
             f"{'ms/iter':>8} {'adam/warm':>10}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.img_size:>4} {r.batch:>5} {'on' if r.amp else 'off':>4} {r.ngf:>4} "
            f"{r.status:>7} {num(r.peak_alloc_mib, '.0f'):>10} "
            f"{num(r.peak_reserved_mib, '.0f'):>9} {num(r.steady_alloc_mib, '.0f'):>7} "
            f"{num(r.pct_of_total, '.0f'):>7} {num(r.iter_ms, '.0f'):>8} {adam(r):>10}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ukur puncak VRAM satu langkah training Pix2Pix.")
    # Grid kombinasi
    p.add_argument("--img-sizes", type=int, nargs="+", default=[256, 128])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--amp", nargs="+", choices=["off", "on"], default=["off", "on"])
    p.add_argument("--ngfs", type=int, nargs="+", default=[64, 32])
    # Parameter model (default = yang sudah disepakati)
    p.add_argument("--g-norm", default="batch", choices=["batch", "instance", "none"])
    p.add_argument("--use-skip", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-dropout", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--d-norm", default="batch", choices=["batch", "instance", "none"])
    # Parameter training
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--lambda-l1", type=float, default=100.0)
    p.add_argument("--iters", type=int, default=3,
                   help="iterasi TERUKUR per kombinasi, setelah state Adam terbentuk")
    p.add_argument("--max-warmup", type=int, default=50,
                   help="batas iterasi pemanasan untuk menunggu state Adam terbentuk")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cudnn-benchmark", action="store_true",
                   help="samakan dengan train.py jika nanti memakai cudnn.benchmark=True")
    p.add_argument("--cudnn-deterministic", action="store_true",
                   help="paksa algoritma cuDNN deterministik (untuk menguji lonjakan memori)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                   help="cpu hanya untuk uji jalan kode; memori tidak diukur")
    p.add_argument("--out", default="vram_results.csv")
    args = p.parse_args(argv)
    if args.iters < 1:
        p.error("--iters harus >= 1")
    if args.max_warmup < 1:
        p.error("--max-warmup harus >= 1")
    return args


def main(argv=None) -> list:
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA tidak tersedia. Jalankan di laptop dengan GPU, "
                         "atau pakai --device cpu hanya untuk uji jalan kode.")
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.deterministic = args.cudnn_deterministic

    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        print(f"GPU: {prop.name}, total {prop.total_memory / MIB:.0f} MiB | "
              f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"G: norm={args.g_norm}, skip={args.use_skip}, dropout={args.use_dropout} | "
          f"D: ndf={args.ndf}, n_layers={args.n_layers}, norm={args.d_norm} | "
          f"iters={args.iters}, max_warmup={args.max_warmup} | "
          f"cudnn benchmark={args.cudnn_benchmark}, deterministic={args.cudnn_deterministic}\n")

    results = []
    for img_size in args.img_sizes:
        for batch in args.batches:
            for amp_flag in args.amp:
                for ngf in args.ngfs:
                    r = run_combo(args, img_size, batch, amp_flag == "on", ngf, device)
                    results.append(r)
                    print(f"  img={img_size} batch={batch} amp={amp_flag} ngf={ngf}: {r.status}"
                          + (f", {r.peak_reserved_mib:.0f} MiB" if r.peak_reserved_mib else ""))

    out = Path(args.out)
    write_csv(results, out)
    print("\n" + format_table(results))
    print(f"\nCSV disimpan ke: {out.resolve()}")
    return results


if __name__ == "__main__":
    main()
