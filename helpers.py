"""
tests/helpers.py

Fungsi bantu bersama untuk tes. Dipisah dari file tes supaya tidak ada file
tes yang meng-import file tes lain.
"""

import numpy as np
from PIL import Image


def make_pairs(root, n=12, size=64):
    """n pasangan sintetis format CMP: label .png mode palet (indeks kelas +
    peta warna) dan foto .jpg RGB dengan nama dasar sama."""
    rng = np.random.default_rng(0)
    root.mkdir(parents=True)
    palette = [0, 0, 0] + [c for k in range(1, 13) for c in (k * 20, 255 - k * 20, 128)]
    for i in range(n):
        idx = np.kron(rng.integers(1, 13, (4, 4)), np.ones((size // 4, size // 4))).astype(np.uint8)
        label = Image.fromarray(idx, mode="P")
        label.putpalette(palette)
        label.save(root / f"cmp_b{i:04d}.png")
        Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8)).save(root / f"cmp_b{i:04d}.jpg")
