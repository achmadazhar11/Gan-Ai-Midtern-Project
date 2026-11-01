# Pembagian tugas dan cara commit

Rubrik panduan (Kerjasama Tim & Manajemen Integrasi, 20%) menilai keselarasan
tugas antar anggota yang **terlihat pada Git commit atau log kerja**. Karena
itu setiap anggota meng-commit bagiannya **sendiri, dari akun Git-nya sendiri**.
Jangan meng-commit atas nama anggota lain, dan jangan memalsukan tanggal commit:
riwayat commit adalah bukti kerja, dan dosen bisa menanyakannya saat presentasi.

## Pembagian file (usulan, sesuai job desk di panduan)

| Anggota | Peran | File yang di-commit dan dijelaskan saat presentasi |
|---|---|---|
| 1 | Prompt Engineer & LLM Navigator (Ketua) | `config.py`, `models/__init__.py`, `models/generator.py`, `models/discriminator.py`, `tests/test_config.py`, `tests/test_generator.py`, `tests/test_discriminator.py` |
| 2 | Data Pipeline & Preprocessing | `dataset.py`, `scripts/preview_pairs.py`, `scripts/smoke_real_data.py`, `tests/test_dataset.py` |
| 3 | Training Loop & Hyperparameter Tuning | `losses.py`, `train.py`, `eval.py`, `scripts/measure_vram.py`, `tests/test_losses.py`, `tests/test_train.py`, `tests/test_eval.py`, `tests/test_measure_vram.py` |
| 4 | Lead Documentarian & Error Handler | `README.md`, `requirements.txt`, `.gitignore`, `pytest.ini`, `docs/`, `notebooks/Pix2Pix_Colab.ipynb`, ekspor log interaksi LLM |

Pembagian ini boleh diubah kelompok. Yang penting: setiap orang benar-benar
memahami file yang ia commit.

## Langkah

### 0. Ketua: buat repo kosong

1. Buat repo baru di GitHub (misalnya `pix2pix-cmp-facades`), private atau public sesuai aturan dosen.
2. *Settings → Collaborators*: undang ketiga anggota lain.

### 1. Setiap anggota: atur identitas Git (sekali per laptop)

```
git config --global user.name "Nama Lengkap"
git config --global user.email "email-yang-terdaftar-di-github@contoh.com"
```

### 2. Anggota 4: commit pertama (kerangka repo)

```
git clone https://github.com/<akun>/<repo>.git
cd <repo>
# salin isi folder pix2pix ke sini
git add README.md requirements.txt .gitignore pytest.ini docs/
git commit -m "Kerangka repo: README, requirements, gitignore, pembagian tugas"
git push
```

### 3. Anggota 1, 2, 3: commit bagian masing-masing, bergantian

Urutan yang disarankan mengikuti ketergantungan kode (model dan config dulu):

```
git pull
# salin file bagian Anda (lihat tabel) ke folder repo
git add <file-file bagian Anda>
git commit -m "<deskripsi singkat, misalnya: Generator U-Net + tes shape>"
git push
```

Contoh pesan commit yang jelas:

- Anggota 1: `Config frozen + validasi + preset ablation/tuning`, `Generator U-Net dengan bottleneck tanpa norm`, `Discriminator PatchGAN + fungsi receptive field`
- Anggota 2: `Dataset berpasangan + augmentasi sinkron + manifest split`, `Skrip grid pratinjau dan smoke test data asli`
- Anggota 3: `Loss non-saturating + L1 (fp32)`, `Loop training dengan checkpoint dan resume`, `Evaluasi L1/PSNR/SSIM`, `Pengukuran VRAM dengan pemanasan state Adam`

### 4. Anggota 4: notebook dan log

```
git pull
git add notebooks/Pix2Pix_Colab.ipynb docs/
git commit -m "Notebook Colab Run all + log interaksi LLM"
git push
```

### 5. Setelah semua file masuk: cek bersama

```
git pull
pip install -r requirements.txt
pytest -q          # harus lulus semua
```

## Log kerja

Selain commit, setiap perubahan berikutnya (perbaikan bug, hasil training,
ablation) sebaiknya punya commit sendiri dengan pesan yang menjelaskan **apa**
dan **kenapa**, dan dicatat di log interaksi LLM jika melibatkan LLM.
