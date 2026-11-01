"""
scripts/preview_pairs.py

Simpan grid contoh pasangan (kolom kiri = A/kondisi, kolom kanan = B/target)
untuk memastikan secara visual bahwa A memang LABEL dan B memang FOTO.

Contoh (dari root repo):
    python scripts/preview_pairs.py --root data/facades --split train --direction AtoB
    python scripts/preview_pairs.py --root data/facades --split train --direction BtoA --augment

Skrip juga mencetak:
- "flatness" tiap kolom: proporsi piksel yang hampir sama dengan tetangga
  kanannya. Peta label (blok warna rata) biasanya jauh lebih rata daripada
  foto. Ini hanya PETUNJUK heuristik; keputusan akhir tetap dari melihat grid.
- hasil pemeriksaan tumpang tindih antar split yang tersedia.
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

# Agar `import dataset` berhasil saat skrip dijalankan dari mana pun.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import (  # noqa: E402
    SPLITS,
    PairedImageDataset,
    denormalize,
    find_split_overlap,
)

HEADER_H = 20  # tinggi baris judul kolom (piksel)


def flatness(x: torch.Tensor, tol: int = 8) -> float:
    """Proporsi piksel yang selisihnya dengan tetangga kanan <= tol (skala 0..255).
    Toleransi menyerap noise kompresi JPEG pada peta label."""
    u8 = denormalize(x).int()                          # [3, H, W]
    diff = (u8[:, :, 1:] - u8[:, :, :-1]).abs().amax(dim=0)
    return (diff <= tol).float().mean().item()


def build_grid(samples: list, img_size: int) -> Image.Image:
    """samples: list dict {"A","B"}. Hasil: gambar [header + n baris] x 2 kolom."""
    gap = 4
    width = 2 * img_size + gap
    height = HEADER_H + len(samples) * (img_size + gap)
    grid = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    draw.text((4, 4), "A (kondisi)", fill=(0, 0, 0))
    draw.text((img_size + gap + 4, 4), "B (target)", fill=(0, 0, 0))
    for row, s in enumerate(samples):
        y = HEADER_H + row * (img_size + gap)
        for col, key in enumerate(("A", "B")):
            arr = denormalize(s[key]).permute(1, 2, 0).numpy()   # CHW -> HWC
            grid.paste(Image.fromarray(arr), (col * (img_size + gap), y))
    return grid


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description="Simpan grid pasangan A|B untuk dicek visual.")
    p.add_argument("--root", required=True, help="folder dataset")
    p.add_argument("--split", default="train", choices=SPLITS)
    p.add_argument("--direction", default="AtoB", choices=("AtoB", "BtoA"))
    p.add_argument("--n", type=int, default=6, help="jumlah pasangan")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--load-size", type=int, default=286)
    p.add_argument("--augment", action="store_true",
                   help="tampilkan hasil augmentasi train (cek crop/flip sinkron)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="preview_pairs.png")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    ds = PairedImageDataset(args.root, args.split, args.direction, args.img_size,
                            args.load_size, augment=args.augment)
    n = min(args.n, len(ds))
    samples = [ds[i] for i in range(n)]

    out = Path(args.out)
    build_grid(samples, args.img_size).save(out)

    flat_a = sum(flatness(s["A"]) for s in samples) / n
    flat_b = sum(flatness(s["B"]) for s in samples) / n
    print(f"Grid disimpan ke: {out.resolve()}")
    print(f"Split '{args.split}' ({ds.split_source}), {len(ds)} pasangan, direction={args.direction}")
    print(f"Flatness rata-rata  A: {flat_a:.3f}   B: {flat_b:.3f}")
    if flat_a > flat_b:
        print("Petunjuk: A lebih rata -> kemungkinan A = label (sesuai tujuan).")
    else:
        print("Petunjuk: B lebih rata -> kemungkinan B = label. Coba direction sebaliknya.")
    print("Ini hanya heuristik. Pastikan dengan MELIHAT gambar grid.")

    # Pemeriksaan overlap antar split yang tersedia
    available = {}
    for split in SPLITS:
        try:
            available[split] = PairedImageDataset(args.root, split, args.direction,
                                                  args.img_size, args.load_size)
        except (ValueError, FileNotFoundError):
            pass
    if len(available) > 1:
        overlap = find_split_overlap(available)
        print(f"Cek overlap {list(available)}: path sama = {len(overlap['same_path'])}, "
              f"isi file identik = {len(overlap['same_content'])}")
    return out


if __name__ == "__main__":
    main()
