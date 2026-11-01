"""
dataset.py

Dataset berpasangan untuk Pix2Pix (target: CMP Facades, label -> foto).

Dua format file didukung dan dideteksi otomatis per folder:

1. "berdampingan" (combined): satu file berisi dua gambar kiri|kanan,
   lebar = 2 x tinggi. Mentahnya: A_raw = setengah KIRI, B_raw = setengah KANAN.
2. "terpisah" (separate): dua file dengan nama dasar sama, label .png dan
   foto .jpg/.jpeg. Mentahnya: A_raw = label (.png), B_raw = foto (.jpg).

`direction="AtoB"` memakai urutan mentah apa adanya, `"BtoA"` menukarnya.
Sisi mana yang label pada format berdampingan TIDAK diasumsikan di kode:
cek dengan scripts/preview_pairs.py lalu pilih direction yang membuat
A = label dan B = foto.

Split:
- Jika root punya subfolder train/ val/ test/ -> pakai split bawaan.
- Jika tidak (semua file langsung di root) -> split deterministik
  (default seed 42, 80/10/10) yang disimpan ke file manifest JSON. Run
  berikutnya (dan anggota kelompok lain) membaca manifest yang sama.

Semua tensor keluaran: float32, CPU, rentang [-1, 1], shape [3, H, W].
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")
DIRECTIONS = ("AtoB", "BtoA")
# Format terpisah: label disimpan sebagai .png, foto sebagai .jpg/.jpeg
LABEL_EXT = ".png"
PHOTO_EXTS = (".jpg", ".jpeg")

# Pillow >= 9.1 menyediakan enum ini (nama lama Image.BICUBIC sudah usang).
BICUBIC = Image.Resampling.BICUBIC
FLIP_LR = Image.Transpose.FLIP_LEFT_RIGHT


# ===========================================================================
# Konversi gambar <-> tensor
# ===========================================================================
def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB uint8 [H, W, 3] -> tensor float32 [3, H, W] di [-1, 1].

    x / 127.5 - 1 memetakan 0 -> -1 dan 255 -> +1, cocok dengan Tanh generator.
    """
    arr = np.asarray(img, dtype=np.float32)          # [H, W, 3], nilai 0..255
    t = torch.from_numpy(arr).permute(2, 0, 1)       # HWC -> CHW
    return (t / 127.5 - 1.0).contiguous()


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Tensor [-1, 1] -> uint8 [0, 255], shape tetap ([C,H,W] atau [B,C,H,W]).

    clamp dulu karena output model bisa sedikit keluar rentang; round agar
    konversi bolak-balik uint8 -> [-1,1] -> uint8 tepat sama.
    """
    x = x.detach().to("cpu", torch.float32).clamp(-1.0, 1.0)
    return ((x + 1.0) * 127.5).round().to(torch.uint8)


def _open_rgb(path: str) -> Image.Image:
    # `with` menutup file segera setelah dibaca. convert() membuat salinan di
    # memori, sehingga tidak ada file yang terkunci (penting di Windows).
    with Image.open(path) as im:
        return im.convert("RGB")


# ===========================================================================
# Menemukan pasangan file
# ===========================================================================
def scan_pairs(folder: Path) -> list:
    """Kembalikan daftar pasangan terurut dari satu folder.

    Setiap pasangan berupa tuple (path_a, path_b):
    - format berdampingan: (file, None)
    - format terpisah    : (label.png, foto.jpg)
    File non-gambar (misalnya .xml anotasi) diabaikan.
    """
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXTS)

    groups: dict = {}
    for p in files:
        groups.setdefault(p.stem, []).append(p)

    pairs = []
    for stem in sorted(groups):
        group = groups[stem]
        if len(group) == 1:
            pairs.append((str(group[0]), None))
            continue
        labels = [p for p in group if p.suffix.lower() == LABEL_EXT]
        photos = [p for p in group if p.suffix.lower() in PHOTO_EXTS]
        if len(group) == 2 and len(labels) == 1 and len(photos) == 1:
            pairs.append((str(labels[0]), str(photos[0])))
        else:
            raise ValueError(
                f"Nama dasar '{stem}' di {folder} punya file yang tidak bisa "
                f"dipasangkan: {[p.name for p in group]}"
            )

    kinds = {b is None for _, b in pairs}
    if len(kinds) > 1:
        raise ValueError(
            f"Folder {folder} mencampur format berdampingan dan terpisah; "
            "pisahkan dulu agar pasangan tidak salah."
        )
    return pairs


def _pair_key(root: Path, pair: tuple) -> str:
    """ID stabil sebuah pasangan untuk manifest: path relatif file pertama,
    pakai '/' agar manifest sama di Windows dan Linux."""
    return Path(pair[0]).relative_to(root).as_posix()


def _make_split(keys: list, seed: int, ratios: tuple) -> dict:
    """Acak urutan dengan seed tetap, lalu potong sesuai rasio."""
    keys = sorted(keys)                 # urutan awal tidak bergantung OS
    random.Random(seed).shuffle(keys)   # Random(seed) lokal: tidak mengganggu RNG global
    n = len(keys)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    return {
        "train": keys[:n_train],
        "val": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
    }


def load_or_create_manifest(root: Path, manifest_path: Path, seed: int,
                            ratios: tuple) -> dict:
    """Baca manifest jika sudah ada; jika belum, buat dan simpan."""
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("seed") != seed or tuple(manifest.get("ratios", ())) != tuple(ratios):
            raise ValueError(
                f"Manifest {manifest_path} dibuat dengan seed/rasio berbeda "
                f"({manifest.get('seed')}, {manifest.get('ratios')}). Hapus file itu "
                "atau samakan config agar split tidak berubah diam-diam."
            )
        return manifest

    keys = [_pair_key(root, p) for p in scan_pairs(root)]
    if not keys:
        raise FileNotFoundError(f"Tidak ada gambar di {root}")
    manifest = {
        "seed": seed,
        "ratios": list(ratios),
        "n_total": len(keys),
        "splits": _make_split(keys, seed, ratios),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Tulis ke file sementara lalu ganti nama: file manifest tidak pernah
    # setengah jadi walaupun proses terhenti di tengah jalan.
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, manifest_path)
    return manifest


def resolve_split(root, split: str, manifest_path=None, split_seed: int = 42,
                  split_ratios: tuple = (0.8, 0.1, 0.1)) -> tuple:
    """Kembalikan (daftar pasangan, sumber split: "bawaan" atau "manifest")."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {root}")

    builtin = [s for s in SPLITS if (root / s).is_dir()]
    if builtin:
        if split not in builtin:
            raise ValueError(
                f"Dataset di {root} punya split bawaan {builtin}, tetapi tidak ada "
                f"'{split}'. Pakai split yang tersedia."
            )
        return scan_pairs(root / split), "bawaan"

    if abs(sum(split_ratios) - 1.0) > 1e-6 or len(split_ratios) != 3:
        raise ValueError(f"split_ratios harus 3 angka berjumlah 1, didapat {split_ratios}")
    if manifest_path is None:
        manifest_path = root / f"split_manifest_seed{split_seed}.json"
    manifest = load_or_create_manifest(root, Path(manifest_path), split_seed,
                                       tuple(split_ratios))

    lookup = {_pair_key(root, p): p for p in scan_pairs(root)}
    keys = manifest["splits"][split]
    missing = [k for k in keys if k not in lookup]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} file di manifest tidak ditemukan (contoh: {missing[:3]}). "
            "Isi folder berubah sejak manifest dibuat."
        )
    return [lookup[k] for k in keys], "manifest"


# ===========================================================================
# Dataset
# ===========================================================================
class PairedImageDataset(Dataset):
    """Dataset pasangan (A = kondisi, B = target) untuk Pix2Pix.

    Args:
        root:      folder dataset (lihat docstring modul untuk tata letaknya).
        split:     "train", "val", atau "test".
        direction: "AtoB" (urutan mentah) atau "BtoA" (ditukar).
        img_size:  ukuran sisi gambar keluaran (crop akhir).
        load_size: ukuran resize sebelum random crop (hanya saat augment).
        augment:   None -> otomatis True untuk "train", False untuk lainnya.

    Keyword tambahan:
        subset_size:   ambil N pasangan pertama saja dan MATIKAN augmentasi
                       (untuk overfit test: input harus identik tiap epoch).
        manifest_path: lokasi file manifest jika split dibuat sendiri.
        split_seed, split_ratios: pengaturan split deterministik.

    __getitem__ -> {"A": [3,img,img] float32, "B": [3,img,img] float32, "path": str}
    """

    def __init__(self, root, split: str, direction: str = "AtoB",
                 img_size: int = 256, load_size: int = 286, augment=None, *,
                 subset_size=None, manifest_path=None, split_seed: int = 42,
                 split_ratios: tuple = (0.8, 0.1, 0.1)):
        super().__init__()
        if split not in SPLITS:
            raise ValueError(f"split harus salah satu dari {SPLITS}, didapat {split!r}")
        if direction not in DIRECTIONS:
            raise ValueError(f"direction harus salah satu dari {DIRECTIONS}, didapat {direction!r}")
        if img_size < 1 or load_size < img_size:
            raise ValueError(f"Butuh 1 <= img_size <= load_size, didapat {img_size}, {load_size}")

        pairs, self.split_source = resolve_split(root, split, manifest_path,
                                                 split_seed, split_ratios)
        if not pairs:
            raise FileNotFoundError(f"Split '{split}' kosong di {root}")

        if subset_size is not None:
            if subset_size < 1:
                raise ValueError(f"subset_size harus >= 1, didapat {subset_size}")
            pairs = pairs[:subset_size]
            augment = False  # overfit test: data harus sama persis tiap epoch

        # Hanya string dan angka yang disimpan sebagai atribut, sehingga
        # objek dataset mudah di-pickle ke worker DataLoader (wajib di Windows).
        self.pairs = pairs
        self.split = split
        self.direction = direction
        self.img_size = img_size
        self.load_size = load_size
        self.augment = (split == "train") if augment is None else bool(augment)

    def __len__(self) -> int:
        return len(self.pairs)

    # ----- membaca pasangan mentah -----
    def _load_raw_pair(self, idx: int) -> tuple:
        path_a, path_b = self.pairs[idx]
        if path_b is None:
            # Format berdampingan: belah dua tepat di tengah.
            img = _open_rgb(path_a)
            w, h = img.size
            if w != 2 * h:
                raise ValueError(
                    f"{path_a}: ukuran {w}x{h}, bukan 2:1. Bukan file berdampingan?"
                )
            raw_a = img.crop((0, 0, w // 2, h))   # kiri
            raw_b = img.crop((w // 2, 0, w, h))   # kanan
        else:
            raw_a, raw_b = _open_rgb(path_a), _open_rgb(path_b)
            if raw_a.size != raw_b.size:
                raise ValueError(
                    f"Ukuran label {raw_a.size} != foto {raw_b.size} untuk {path_a}; "
                    "pasangan mungkin tidak sejajar."
                )
        if self.direction == "BtoA":
            raw_a, raw_b = raw_b, raw_a
        return raw_a, raw_b

    # ----- augmentasi -----
    def _sample_aug_params(self) -> tuple:
        """Ambil SATU set parameter acak (top, left, flip) untuk satu pasangan.

        Memakai RNG torch karena DataLoader memberi seed berbeda ke RNG torch
        di setiap worker. (RNG NumPy tidak di-seed per worker, sehingga worker
        bisa menghasilkan augmentasi kembar.)
        """
        max_off = self.load_size - self.img_size
        top = int(torch.randint(0, max_off + 1, (1,)))
        left = int(torch.randint(0, max_off + 1, (1,)))
        flip = bool(torch.rand(1) < 0.5)
        return top, left, flip

    def _crop_flip(self, img: Image.Image, top: int, left: int, flip: bool) -> Image.Image:
        s = self.img_size
        img = img.crop((left, top, left + s, top + s))
        return img.transpose(FLIP_LR) if flip else img

    def _transform_pair(self, raw_a: Image.Image, raw_b: Image.Image) -> tuple:
        if self.augment:
            # Train: resize ke load_size, lalu crop + flip dengan parameter yang
            # SAMA untuk A dan B. Jika parameternya diambil terpisah, piksel
            # label tidak lagi sejajar dengan piksel foto -> model belajar salah.
            size = (self.load_size, self.load_size)
            a = raw_a.resize(size, BICUBIC)
            b = raw_b.resize(size, BICUBIC)
            top, left, flip = self._sample_aug_params()   # dipanggil SEKALI
            a = self._crop_flip(a, top, left, flip)
            b = self._crop_flip(b, top, left, flip)
        else:
            # Val/test (dan subset): resize langsung ke img_size, deterministik.
            size = (self.img_size, self.img_size)
            a = raw_a.resize(size, BICUBIC)
            b = raw_b.resize(size, BICUBIC)
        return to_tensor(a), to_tensor(b)

    def __getitem__(self, idx: int) -> dict:
        raw_a, raw_b = self._load_raw_pair(idx)
        a, b = self._transform_pair(raw_a, raw_b)
        # "path": file berdampingan, atau file label untuk format terpisah
        return {"A": a, "B": b, "path": self.pairs[idx][0]}


# ===========================================================================
# Pemeriksaan tumpang tindih antar split
# ===========================================================================
def _file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_split_overlap(datasets: dict) -> dict:
    """Cari pasangan yang muncul di lebih dari satu split.

    datasets: {"train": PairedImageDataset, "val": ..., "test": ...}
    Mengembalikan {"same_path": [...], "same_content": [...]}; keduanya
    harus kosong. "same_content" menangkap file yang isinya identik byte demi
    byte walaupun nama/foldernya berbeda (tidak menangkap gambar yang mirip
    tetapi dikompres ulang).
    """
    seen_path: dict = {}
    seen_hash: dict = {}
    same_path, same_content = [], []
    for split, ds in datasets.items():
        for pair in ds.pairs:
            for path in (p for p in pair if p is not None):
                key = str(Path(path).resolve())
                if key in seen_path and seen_path[key] != split:
                    same_path.append((path, seen_path[key], split))
                seen_path.setdefault(key, split)
                digest = _file_md5(path)
                if digest in seen_hash and seen_hash[digest][1] != split:
                    same_content.append((seen_hash[digest][0], path))
                seen_hash.setdefault(digest, (path, split))
    return {"same_path": same_path, "same_content": same_content}


# ===========================================================================
# DataLoader
# ===========================================================================
def get_dataloader(cfg, split: str) -> DataLoader:
    """Buat DataLoader dari config.

    Field cfg yang dibaca (nanti didefinisikan di config.py):
        data_root, direction, img_size, load_size, batch_size, num_workers,
        pin_memory, seed, subset_size, manifest_path, split_seed, split_ratios

    Catatan Windows: num_workers > 0 memakai proses "spawn", yang meng-import
    ulang skrip utama di setiap worker. Karena itu kode training WAJIB berada
    di dalam `if __name__ == "__main__":`. Dataset ini sendiri aman di-pickle.
    """
    dataset = PairedImageDataset(
        cfg.data_root, split, cfg.direction, cfg.img_size, cfg.load_size,
        augment=None,
        subset_size=cfg.subset_size,
        manifest_path=cfg.manifest_path,
        split_seed=cfg.split_seed,
        split_ratios=cfg.split_ratios,
    )
    # Generator ber-seed: urutan shuffle dan seed RNG tiap worker (yang
    # menentukan crop/flip) dapat direproduksi.
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        # pin_memory hanya berguna jika ada GPU; tanpa CUDA PyTorch memberi peringatan.
        pin_memory=bool(cfg.pin_memory) and torch.cuda.is_available(),
        generator=gen,
        # Worker tidak dibuat ulang tiap epoch: di Windows, spawn proses lambat.
        persistent_workers=cfg.num_workers > 0,
    )
