"""
tests/test_dataset.py

Tes untuk dataset.py. Semua tes memakai gambar SINTETIS yang dibuat di folder
sementara (tmp_path), sehingga bisa dijalankan tanpa mengunduh dataset asli:
- "label" sintetis = blok-blok warna rata (meniru peta label facade)
- "foto" sintetis  = noise acak (meniru tekstur foto; setiap patch unik)

Jalankan: pytest tests/test_dataset.py -v
"""

import json
import pickle
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from dataset import (
    BICUBIC,
    PairedImageDataset,
    denormalize,
    find_split_overlap,
    get_dataloader,
    to_tensor,
)

HALF = 64  # sisi setiap setengah gambar sintetis


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Pembuat data sintetis
# ---------------------------------------------------------------------------
def label_like(rng, size=HALF, blocks=4):
    """Blok warna rata: blocks x blocks petak, tiap petak satu warna."""
    colors = rng.integers(0, 256, (blocks, blocks, 3), dtype=np.uint8)
    return np.kron(colors, np.ones((size // blocks, size // blocks, 1), dtype=np.uint8))


def photo_like(rng, size=HALF):
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


def write_combined(path, left, right):
    Image.fromarray(np.concatenate([left, right], axis=1)).save(path)


def make_flat_combined(root, n=50, seed=0, left_fn=label_like, right_fn=photo_like):
    """n file berdampingan langsung di root (tanpa split bawaan). PNG = lossless."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        write_combined(root / f"{i:03d}.png", left_fn(rng), right_fn(rng))
    return root


@pytest.fixture
def flat_root(tmp_path):
    return make_flat_combined(tmp_path / "flat")


@pytest.fixture
def builtin_root(tmp_path):
    root = tmp_path / "builtin"
    rng = np.random.default_rng(1)
    for split, n in (("train", 10), ("val", 3), ("test", 3)):
        (root / split).mkdir(parents=True)
        for i in range(n):
            write_combined(root / split / f"{split}_{i}.png", label_like(rng), photo_like(rng))
    return root


@pytest.fixture
def separate_root(tmp_path):
    """Format terpisah ala CMP asli: label .png + foto .jpg + file .xml (diabaikan)."""
    root = tmp_path / "separate"
    root.mkdir()
    rng = np.random.default_rng(2)
    for i in range(20):
        Image.fromarray(label_like(rng)).save(root / f"cmp_{i:04d}.png")
        Image.fromarray(photo_like(rng)).save(root / f"cmp_{i:04d}.jpg", quality=95)
        (root / f"cmp_{i:04d}.xml").write_text("<anotasi/>")
    return root


def halves(path):
    """Belah file berdampingan secara manual (independen dari dataset.py)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
    w, h = im.size
    return im.crop((0, 0, w // 2, h)), im.crop((w // 2, 0, w, h))


# ---------------------------------------------------------------------------
# 1. Shape, dtype, device, rentang
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_shape_dtype_range(flat_root, split):
    ds = PairedImageDataset(flat_root, split, img_size=256, load_size=286)
    item = ds[0]
    assert set(item) == {"A", "B", "path"}
    assert isinstance(item["path"], str)
    for key in ("A", "B"):
        x = item[key]
        assert x.shape == (3, 256, 256)
        assert x.dtype == torch.float32 and x.device.type == "cpu"
        assert x.min() >= -1.0 and x.max() <= 1.0


def test_augment_default_per_split(flat_root):
    assert PairedImageDataset(flat_root, "train").augment is True
    assert PairedImageDataset(flat_root, "val").augment is False
    assert PairedImageDataset(flat_root, "test").augment is False


def test_val_is_deterministic_plain_resize(flat_root):
    """Val: resize bicubic langsung ke img_size, tanpa unsur acak."""
    ds = PairedImageDataset(flat_root, "val", img_size=128, load_size=143)
    i1, i2 = ds[0], ds[0]
    assert torch.equal(i1["A"], i2["A"]) and torch.equal(i1["B"], i2["B"])
    left, right = halves(i1["path"])
    assert torch.equal(i1["A"], to_tensor(left.resize((128, 128), BICUBIC)))
    assert torch.equal(i1["B"], to_tensor(right.resize((128, 128), BICUBIC)))


# ---------------------------------------------------------------------------
# 2. direction
# ---------------------------------------------------------------------------
def test_direction_swaps_a_and_b(flat_root):
    ab = PairedImageDataset(flat_root, "val", "AtoB")[0]
    ba = PairedImageDataset(flat_root, "val", "BtoA")[0]
    assert torch.equal(ab["A"], ba["B"]) and torch.equal(ab["B"], ba["A"])


def test_separate_format_label_is_a(separate_root):
    """Format terpisah: AtoB -> A = label (.png), B = foto (.jpg)."""
    from scripts.preview_pairs import flatness
    ds = PairedImageDataset(separate_root, "val", "AtoB", img_size=64, load_size=72)
    item = ds[0]
    assert item["path"].endswith(".png")
    assert flatness(item["A"]) > flatness(item["B"])


# ---------------------------------------------------------------------------
# 3. BUKTI crop/flip sinkron antara A dan B
# ---------------------------------------------------------------------------
def recover_params(out, ref, size):
    """Cari (top, left, flip) yang membuat `out` sama persis dengan potongan
    `ref` (gambar setelah resize ke load_size). Pencarian brute-force, tidak
    memakai kode augmentasi di dataset.py, jadi independen."""
    n_off = ref.shape[-1] - size + 1
    found = []
    for flip in (False, True):
        x = out.flip(-1) if flip else out           # batalkan flip
        for top in range(n_off):
            for left in range(n_off):
                crop = ref[:, top:top + size, left:left + size]
                if torch.equal(x[:, :4, :4], crop[:, :4, :4]) and torch.equal(x, crop):
                    found.append((top, left, flip))
    assert len(found) == 1, f"parameter tidak unik/tidak ditemukan: {found}"
    return found[0]


def test_crop_and_flip_are_synchronised(tmp_path):
    # Kedua sisi berupa noise INDEPENDEN, sehingga A dan B berbeda isinya dan
    # setiap potongan hanya cocok di satu posisi.
    root = make_flat_combined(tmp_path / "sync", n=10, left_fn=photo_like)
    size, load = 32, 40
    ds = PairedImageDataset(root, "train", img_size=size, load_size=load)
    assert ds.augment

    params_seen = []
    for draw in range(30):
        idx = draw % len(ds)
        item = ds[idx]
        left, right = halves(item["path"])
        ref_a = to_tensor(left.resize((load, load), BICUBIC))
        ref_b = to_tensor(right.resize((load, load), BICUBIC))
        pa = recover_params(item["A"], ref_a, size)
        pb = recover_params(item["B"], ref_b, size)
        assert pa == pb, f"A dan B tidak sinkron: A={pa}, B={pb}"
        params_seen.append(pa)

    # Augmentasi benar-benar acak: kedua kondisi flip muncul, offset bervariasi.
    assert {p[2] for p in params_seen} == {False, True}
    assert len({p[:2] for p in params_seen}) > 5


def test_random_params_sampled_once_per_item(flat_root, monkeypatch):
    ds = PairedImageDataset(flat_root, "train", img_size=32, load_size=40)
    calls = []
    original = ds._sample_aug_params
    monkeypatch.setattr(ds, "_sample_aug_params", lambda: calls.append(1) or original())
    ds[0]
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 4. Split: tidak ada tumpang tindih
# ---------------------------------------------------------------------------
def _keys(ds):
    return {p[0] for p in ds.pairs}


def test_manifest_split_disjoint_and_complete(flat_root):
    dss = {s: PairedImageDataset(flat_root, s) for s in ("train", "val", "test")}
    assert [len(dss[s]) for s in ("train", "val", "test")] == [40, 5, 5]
    k = {s: _keys(d) for s, d in dss.items()}
    assert not (k["train"] & k["val"]) and not (k["train"] & k["test"]) and not (k["val"] & k["test"])
    assert len(k["train"] | k["val"] | k["test"]) == 50
    assert find_split_overlap(dss) == {"same_path": [], "same_content": []}
    assert all(d.split_source == "manifest" for d in dss.values())


def test_manifest_saved_and_deterministic(flat_root):
    PairedImageDataset(flat_root, "train")
    manifest_file = flat_root / "split_manifest_seed42.json"
    assert manifest_file.exists()
    first = json.loads(manifest_file.read_text())
    assert first["seed"] == 42 and first["n_total"] == 50

    manifest_file.unlink()                     # buat ulang dari nol
    PairedImageDataset(flat_root, "train")
    assert json.loads(manifest_file.read_text())["splits"] == first["splits"]


def test_manifest_custom_path_and_seed_mismatch(flat_root, tmp_path):
    path = tmp_path / "manifests" / "facades.json"
    PairedImageDataset(flat_root, "train", manifest_path=path, split_seed=42)
    assert path.exists()
    with pytest.raises(ValueError, match="seed/rasio berbeda"):
        PairedImageDataset(flat_root, "train", manifest_path=path, split_seed=7)


def test_different_seed_gives_different_split(flat_root, tmp_path):
    a = PairedImageDataset(flat_root, "val", manifest_path=tmp_path / "s42.json", split_seed=42)
    b = PairedImageDataset(flat_root, "val", manifest_path=tmp_path / "s7.json", split_seed=7)
    assert _keys(a) != _keys(b)


def test_builtin_split_used_and_disjoint(builtin_root):
    dss = {s: PairedImageDataset(builtin_root, s) for s in ("train", "val", "test")}
    assert [len(dss[s]) for s in ("train", "val", "test")] == [10, 3, 3]
    assert all(d.split_source == "bawaan" for d in dss.values())
    assert find_split_overlap(dss) == {"same_path": [], "same_content": []}
    assert not (builtin_root / "split_manifest_seed42.json").exists()


def test_overlap_checker_detects_duplicate_file(builtin_root):
    """Kontrol negatif: salin satu file train ke val -> harus terdeteksi."""
    src = sorted((builtin_root / "train").iterdir())[0]
    shutil.copy(src, builtin_root / "val" / "duplikat.png")
    dss = {s: PairedImageDataset(builtin_root, s) for s in ("train", "val")}
    assert len(find_split_overlap(dss)["same_content"]) == 1


def test_missing_builtin_split_raises(tmp_path):
    root = tmp_path / "only_train"
    make_flat_combined(root / "train", n=3)
    with pytest.raises(ValueError, match="split bawaan"):
        PairedImageDataset(root, "val")


# ---------------------------------------------------------------------------
# 5. subset_size untuk overfit test
# ---------------------------------------------------------------------------
def test_subset_size(flat_root):
    full = PairedImageDataset(flat_root, "train")
    sub = PairedImageDataset(flat_root, "train", subset_size=8)
    assert len(sub) == 8
    assert sub.augment is False                  # walaupun split train
    assert sub.pairs == full.pairs[:8]
    i1, i2 = sub[3], sub[3]
    assert torch.equal(i1["A"], i2["A"])         # identik tiap kali dibaca


# ---------------------------------------------------------------------------
# 6. Validasi input
# ---------------------------------------------------------------------------
def test_rejects_non_side_by_side(tmp_path):
    root = tmp_path / "bad"
    root.mkdir()
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(root / "persegi.png")
    ds = PairedImageDataset(root, "train", split_ratios=(1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="bukan 2:1"):
        ds[0]


def test_rejects_mixed_formats(flat_root):
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(flat_root / "000.jpg")
    with pytest.raises(ValueError, match="mencampur format|tidak bisa"):
        PairedImageDataset(flat_root, "train", manifest_path=flat_root / "m.json")


@pytest.mark.parametrize("kwargs", [
    {"split": "dev"}, {"direction": "AB"}, {"img_size": 300, "load_size": 286},
])
def test_rejects_bad_arguments(flat_root, kwargs):
    args = {"split": "train", **kwargs}
    with pytest.raises(ValueError):
        PairedImageDataset(flat_root, **args)


# ---------------------------------------------------------------------------
# 7. DataLoader
# ---------------------------------------------------------------------------
def make_cfg(root, **over):
    cfg = dict(data_root=str(root), direction="AtoB", img_size=64, load_size=72,
               batch_size=4, num_workers=2, pin_memory=True, seed=123,
               subset_size=None, manifest_path=None, split_seed=42,
               split_ratios=(0.8, 0.1, 0.1))
    cfg.update(over)
    return SimpleNamespace(**cfg)


def test_dataloader_shuffle_only_train(flat_root):
    cfg = make_cfg(flat_root, num_workers=0)
    assert isinstance(get_dataloader(cfg, "train").sampler, RandomSampler)
    assert isinstance(get_dataloader(cfg, "val").sampler, SequentialSampler)
    assert isinstance(get_dataloader(cfg, "test").sampler, SequentialSampler)


def test_dataloader_batch_shapes_with_workers(flat_root):
    loader = get_dataloader(make_cfg(flat_root), "train")
    assert loader.num_workers == 2
    batch = next(iter(loader))
    assert batch["A"].shape == batch["B"].shape == (4, 3, 64, 64)
    assert len(batch["path"]) == 4


def test_dataloader_reproducible_with_seed(flat_root):
    b1 = next(iter(get_dataloader(make_cfg(flat_root), "train")))
    b2 = next(iter(get_dataloader(make_cfg(flat_root), "train")))
    assert b1["path"] == b2["path"]                # urutan shuffle sama
    assert torch.equal(b1["A"], b2["A"])           # crop/flip sama


def test_windows_style_spawn_workers(flat_root):
    """Windows memakai 'spawn': dataset harus bisa di-pickle dan dibaca ulang
    di proses baru. Kita paksa spawn agar perilaku Windows teruji di Linux."""
    ds = PairedImageDataset(flat_root, "train", img_size=64, load_size=72)
    pickle.loads(pickle.dumps(ds))
    loader = DataLoader(ds, batch_size=2, num_workers=2, multiprocessing_context="spawn")
    batch = next(iter(loader))
    assert batch["A"].shape == (2, 3, 64, 64)


def test_dataloader_subset(flat_root):
    loader = get_dataloader(make_cfg(flat_root, subset_size=8, num_workers=0), "train")
    assert len(loader.dataset) == 8 and loader.dataset.augment is False


# ---------------------------------------------------------------------------
# 8. denormalize
# ---------------------------------------------------------------------------
def test_denormalize_endpoints_and_dtype():
    x = torch.tensor([-1.0, 0.0, 1.0, -5.0, 5.0]).view(1, 1, 5)
    out = denormalize(x)
    assert out.dtype == torch.uint8
    assert out.flatten().tolist() == [0, 128, 255, 0, 255]  # di luar rentang di-clamp


@pytest.mark.parametrize("shape", [(3, 16, 16), (2, 3, 16, 16)])
def test_denormalize_roundtrip_exact(shape):
    u8 = torch.randint(0, 256, shape, dtype=torch.uint8)
    normalized = u8.float() / 127.5 - 1.0          # sama dengan to_tensor
    assert torch.equal(denormalize(normalized), u8)


# ---------------------------------------------------------------------------
# 9. Skrip grid pratinjau (smoke test)
# ---------------------------------------------------------------------------
def test_preview_script_writes_grid(flat_root, tmp_path, capsys):
    from scripts.preview_pairs import HEADER_H, main
    out = main(["--root", str(flat_root), "--split", "train", "--n", "3",
                "--img-size", "64", "--load-size", "72", "--augment",
                "--out", str(tmp_path / "grid.png")])
    with Image.open(out) as grid:
        assert grid.size == (2 * 64 + 4, HEADER_H + 3 * (64 + 4))
    printed = capsys.readouterr().out
    assert "A lebih rata" in printed            # label sintetis ada di kiri
    assert "isi file identik = 0" in printed
