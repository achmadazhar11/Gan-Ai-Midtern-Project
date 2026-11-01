"""scripts/smoke_real_data.py - uji cepat get_dataloader pada data asli.

Contoh (Windows, dari root repo):
  python scripts/smoke_real_data.py --module dataset --root "D:\\...\\CMP_facade_DB_base\\base" --out cek_lama.png
  python scripts/smoke_real_data.py --module dataset_baru --root "D:\\...\\base" --out cek_baru.png
Manifest ditulis ke folder sementara (tidak mengotori folder data).
"""
import argparse
import dataclasses
import importlib
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="dataset")
    ap.add_argument("--root", required=True, help="folder yang LANGSUNG berisi .jpg/.png (subfolder base)")
    ap.add_argument("--direction", default="AtoB", choices=["AtoB", "BtoA"])
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--load-size", type=int, default=286)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--out", default=None, help="simpan grid A|B (4 pasangan val) ke file ini")
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from config import Config

    mod = importlib.import_module(args.module)
    print(f"modul={mod.__name__} torch={torch.__version__} cuda={torch.cuda.is_available()}")

    with tempfile.TemporaryDirectory() as tmp:
        cfg = dataclasses.replace(
            Config(), data_root=args.root, direction=args.direction,
            img_size=args.img_size, load_size=args.load_size,
            num_workers=0, pin_memory=False, manifest_path=str(Path(tmp) / "manifest.json"))
        torch.manual_seed(cfg.seed)  # num_workers=0: crop memakai RNG global

        for split in ("train", "val", "test"):
            ds = mod.get_dataloader(cfg, split).dataset
            print(f"{split}: {len(ds)} pasang, label_mode={getattr(ds, 'label_mode', '(tidak ada)')}")

        loader = mod.get_dataloader(cfg, "train")
        t0, n = time.perf_counter(), 0
        for batch in loader:
            if n == 0:
                a, b = batch["A"], batch["B"]
                print(f"batch pertama: A{tuple(a.shape)} B{tuple(b.shape)} {a.dtype} "
                      f"rentang A[{a.min():.2f},{a.max():.2f}] B[{b.min():.2f},{b.max():.2f}]")
            n += 1
            if n >= args.batches:
                break
        dt = time.perf_counter() - t0
        print(f"{n} batch: {dt:.1f} dtk ({dt / n:.2f} dtk/batch, CPU, num_workers=0)")

        if args.out:
            val = mod.get_dataloader(cfg, "val").dataset
            rows = []
            for i in range(min(4, len(val))):
                s = val[i]
                pair = [mod.denormalize(s[k]).permute(1, 2, 0).numpy() for k in ("A", "B")]
                rows.append(np.concatenate(pair, axis=1))
            Image.fromarray(np.concatenate(rows, axis=0)).save(args.out)
            print(f"grid disimpan: {args.out} (kiri = A, kanan = B)")


if __name__ == "__main__":
    main()
