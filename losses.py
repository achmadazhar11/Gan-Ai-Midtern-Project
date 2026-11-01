"""
losses.py

Fungsi loss Pix2Pix:

    loss_D = d_loss_scale * [ BCE(D(x, y), 1) + BCE(D(x, G(x)), 0) ]
    loss_G = BCE(D(x, G(x)), 1) + lambda_l1 * L1(G(x), y)

x = kondisi (label), y = target (foto), G(x) = foto palsu.
Nilai default lambda_l1 = 100 dan d_loss_scale = 0.5 diatur di config.py.

Pembagian tanggung jawab dengan train.py:
- losses.py hanya MENGHITUNG loss dari logit dan gambar yang diberikan.
- train.py yang memastikan D dipanggil terpisah untuk real dan fake, dan
  bahwa fake di-detach() saat update D. losses.py tidak meng-detach apa pun,
  supaya aliran gradien tetap terlihat jelas di train.py.

Semua loss dihitung dalam float32 (input di-cast dengan .float()). Di bawah
AMP, logit dan gambar dari model bisa berupa float16; menghitung log/sigmoid
dan selisih kecil dalam float16 rawan kehilangan presisi. Dengan cast eksplisit,
hasilnya tidak bergantung pada daftar operasi autocast PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialLoss(nn.Module):
    """BCE antara logit PatchGAN dan target "real" (1) atau "fake" (0).

    Target dibuat dengan ukuran yang SAMA dengan peta logit (misalnya
    [B, 1, 30, 30] untuk n_layers=3 di 256, atau [B, 1, 126, 126] untuk
    n_layers=1), sehingga setiap patch dinilai terhadap targetnya sendiri.
    Rata-rata diambil atas batch dan semua patch.

    BCEWithLogitsLoss menggabungkan sigmoid + BCE dalam satu rumus yang stabil
    secara numerik, jadi D tidak boleh memakai Sigmoid di output.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if not isinstance(target_is_real, bool):
            raise TypeError(f"target_is_real harus bool, didapat {type(target_is_real).__name__}")
        logits = logits.float()
        target = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
        return self.bce(logits, target)   # skalar (0-dim)


# Satu instance dipakai bersama: AdversarialLoss tidak punya parameter atau state.
_adv = AdversarialLoss()


def discriminator_loss(d_real_logits: torch.Tensor, d_fake_logits: torch.Tensor,
                       scale: float = 0.5) -> tuple:
    """Loss D: pasangan asli harus dinilai real, pasangan palsu dinilai fake.

    Args:
        d_real_logits: D(cond, real), shape [B, 1, h, w].
        d_fake_logits: D(cond, fake.detach()), shape sama. detach dilakukan di train.py.
        scale: pengali total (d_loss_scale). 0.5 memperlambat belajar D relatif terhadap G.

    Returns:
        (total, {"d_real", "d_fake"}). Nilai dict berupa tensor 0-dim yang sudah
        di-detach (bukan float): train.py baru memanggil .item() saat logging,
        sehingga GPU tidak perlu sinkron setiap iterasi.
    """
    if d_real_logits.shape != d_fake_logits.shape:
        raise ValueError(f"Shape logit real {tuple(d_real_logits.shape)} != "
                         f"fake {tuple(d_fake_logits.shape)}")
    if scale <= 0:
        raise ValueError(f"scale harus > 0, didapat {scale}")
    loss_real = _adv(d_real_logits, True)
    loss_fake = _adv(d_fake_logits, False)
    total = scale * (loss_real + loss_fake)
    return total, {"d_real": loss_real.detach(), "d_fake": loss_fake.detach()}


def generator_loss(d_fake_logits: torch.Tensor, fake: torch.Tensor, real: torch.Tensor,
                   lambda_l1: float) -> tuple:
    """Loss G: menipu D + mendekati target piksel demi piksel.

    Bagian adversarial memakai bentuk NON-SATURATING: fake diberi label "real",
    yaitu meminimalkan -log D(x, G(x)). Bentuk asli minimax, log(1 - D(x, G(x))),
    gradiennya hampir nol justru saat D yakin gambar palsu (awal training), sehingga
    G sulit belajar. Bentuk non-saturating memberi gradien besar di kondisi itu.

    Args:
        d_fake_logits: D(cond, fake) TANPA detach, supaya gradien mengalir ke G.
        fake, real: [B, C, H, W] di rentang [-1, 1].
        lambda_l1: bobot L1 (config.lambda_l1). 0 = hanya loss adversarial.

    Returns:
        (total, {"g_adv", "g_l1"}), nilai dict berupa tensor 0-dim yang di-detach.
        g_l1 adalah L1 SEBELUM dikali lambda, supaya mudah dibandingkan antar-run.
    """
    if fake.shape != real.shape:
        raise ValueError(f"Shape fake {tuple(fake.shape)} != real {tuple(real.shape)}")
    if lambda_l1 < 0:
        raise ValueError(f"lambda_l1 harus >= 0, didapat {lambda_l1}")
    loss_adv = _adv(d_fake_logits, True)            # non-saturating
    loss_l1 = F.l1_loss(fake.float(), real.float())  # rata-rata |fake - real| per piksel
    total = loss_adv + lambda_l1 * loss_l1
    return total, {"g_adv": loss_adv.detach(), "g_l1": loss_l1.detach()}
