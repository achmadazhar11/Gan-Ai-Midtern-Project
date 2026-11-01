# Pix2Pix dari Nol: Label Fasad → Foto Gedung

Proyek kelompok **Vibe Coding & Generative AI from Scratch**. Model Pix2Pix
(conditional GAN, Generator U-Net + Discriminator PatchGAN, loss BCE + L1)
dibangun dari nol dengan PyTorch, tanpa bobot pretrained, untuk menerjemahkan
peta label fasad (A) menjadi foto gedung (B) pada dataset CMP Facades.

Semua kode ditulis bersama LLM (Claude) dari spesifikasi kelompok. Riwayat
interaksi ada di `docs/` (log interaksi) dan di tautan sesi chat.

## Struktur repo

```
config.py                 semua hyperparameter, seed, device, preset ablation/tuning
dataset.py                dataset berpasangan, augmentasi sinkron, split + manifest
losses.py                 loss adversarial (BCE non-saturating) dan L1
train.py                  loop training, validasi, checkpoint, resume
eval.py                   evaluasi split test: L1, PSNR, SSIM, perbandingan run
models/generator.py       Generator U-Net
models/discriminator.py   Discriminator PatchGAN
scripts/preview_pairs.py  grid pratinjau A|B (cek arah label -> foto)
scripts/smoke_real_data.py uji cepat dataloader pada data asli
scripts/measure_vram.py   pengukuran puncak memori GPU
tests/                    tes pytest untuk setiap modul
notebooks/Pix2Pix_Colab.ipynb  notebook Colab siap "Run all"
docs/PEMBAGIAN_TUGAS.md   pembagian file per anggota dan cara commit
```

## Menjalankan

### 1. Instalasi

```
pip install -r requirements.txt
pytest -q
```

### 2. Data

Unduh **CMP Facade Database (base)** dari halaman resmi
<https://cmp.felk.cvut.cz/~tylecr1/facade/> (`CMP_facade_DB_base.zip`), ekstrak,
lalu arahkan `data_root` ke subfolder `base` yang berisi `cmp_bNNNN.jpg`,
`.png`, dan `.xml`. Data **tidak** disimpan di repo.

Cek arah label → foto:

```
python scripts/preview_pairs.py --root "PATH/base" --split val --direction AtoB --out cek_AtoB.png
```

### 3. Training dan evaluasi

```
python train.py --set data_root="PATH/base"                       # run utama (100 epoch)
python train.py --preset lambda10 --set data_root="PATH/base"     # ablation / tuning
python train.py --config runs/main/config.json --resume           # lanjutkan run
python eval.py --runs runs/main runs/lambda10                     # metrik split test
```

Preset yang tersedia: `lambda10`, `lambda100` (baseline), `lambda200`,
`noskip`, `nlayers1`, `lr1e-4`, `lr4e-4`, `batch1`, `batch4`.

Di Windows, jalankan dari terminal biasa (bukan sel notebook) karena
DataLoader dengan `num_workers > 0` memakai proses spawn.

### 4. Colab

Buka `notebooks/Pix2Pix_Colab.ipynb` di Colab, pilih runtime T4 GPU, atur sel
Pengaturan, lalu *Runtime → Run all*. Dataset diunduh otomatis.

## Hasil

> **[ISI setelah training penuh]** tabel `runs/perbandingan_test.csv`
> (L1, PSNR, SSIM per run), kurva val L1, dan grid hasil test.

## Keputusan konfigurasi (dari pengukuran Colab T4)

| Field | Nilai | Alasan |
|---|---|---|
| `img_size`, `ngf` | 256, 64 | semua kombinasi 256 ≤ 1,34 GB reserved |
| `batch_size` | 2 | muat sampai 4; tuning lewat preset `batch1`, `batch4` |
| `use_amp` | False | AMP tidak menghemat memori (didominasi bobot + Adam fp32); hanya lebih cepat |

Batas memori RTX 4050 (4 GB) perlu dikonfirmasi dengan
`python scripts/measure_vram.py` di laptop.

## Dataset, lisensi, dan sitasi

CMP Facade Database dirilis dengan lisensi Creative Commons
Attribution-ShareAlike (menurut halaman resmi). Sitasi yang diminta:

> R. Tyleček dan R. Šára. *Spatial Pattern Templates for Recognition of Objects
> with Regular Structure*. Proc. GCPR, Saarbrücken, 2013.

Arsitektur mengikuti paper Pix2Pix (Isola dkk., 2017, *Image-to-Image
Translation with Conditional Adversarial Networks*).

## Tim

| Peran | Nama |
|---|---|
| Anggota 1: Prompt Engineer & LLM Navigator (Ketua) | Dirgantara Mulyanda |
| Anggota 2: Data Pipeline & Preprocessing Engineer | Achamd Azhar Faiz Sabiq |
| Anggota 3: Training Loop & Hyperparameter Tuning Engineer | Khaiyir Hibrizi Shakthi |
| Anggota 4: Lead Documentarian & Error Handler | Adhi Suryo Nugroho Suhardjo |
