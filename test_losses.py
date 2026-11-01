"""
tests/test_losses.py

Tes untuk losses.py. Jalankan dari root repo:
    pytest tests/test_losses.py -v
"""

import math

import pytest
import torch
import torch.nn.functional as F

from losses import AdversarialLoss, discriminator_loss, generator_loss
from models.discriminator import PatchDiscriminator
from models.generator import UNetGenerator

LN2 = math.log(2.0)


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. Shape: target mengikuti peta patch, output skalar
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("patch", [30, 126, 14, 62])   # 256/128 x n_layers 3/1
@pytest.mark.parametrize("target_is_real", [True, False])
def test_adversarial_loss_any_patch_size(patch, target_is_real):
    logits = torch.randn(2, 1, patch, patch)
    loss = AdversarialLoss()(logits, target_is_real)
    assert loss.dim() == 0 and torch.isfinite(loss)


@pytest.mark.parametrize("n_layers,expected", [(3, 30), (1, 126)])
def test_losses_with_real_discriminator_output(n_layers, expected):
    D = PatchDiscriminator(ndf=8, n_layers=n_layers)
    cond, real, fake = (torch.rand(2, 3, 256, 256) * 2 - 1 for _ in range(3))
    d_real, d_fake = D(cond, real), D(cond, fake)
    assert d_real.shape == (2, 1, expected, expected)
    total_d, parts_d = discriminator_loss(d_real, d_fake)
    total_g, parts_g = generator_loss(d_fake, fake, real, lambda_l1=100.0)
    for t in (total_d, total_g, *parts_d.values(), *parts_g.values()):
        assert t.dim() == 0 and torch.isfinite(t)


# ---------------------------------------------------------------------------
# 2. Nilai dibandingkan dengan hitungan manual
# ---------------------------------------------------------------------------
def test_zero_logits_give_ln2():
    """Logit 0 -> sigmoid 0.5 -> BCE = -ln(0.5) = ln 2, untuk target 1 maupun 0."""
    z = torch.zeros(2, 1, 30, 30)
    total, parts = discriminator_loss(z, z, scale=0.5)
    assert parts["d_real"].item() == pytest.approx(LN2)
    assert parts["d_fake"].item() == pytest.approx(LN2)
    assert total.item() == pytest.approx(0.5 * (LN2 + LN2))
    total1, _ = discriminator_loss(z, z, scale=1.0)
    assert total1.item() == pytest.approx(2 * LN2)


def test_generator_loss_manual_value():
    z = torch.zeros(2, 1, 30, 30)
    fake = torch.zeros(2, 3, 8, 8)
    real = torch.full((2, 3, 8, 8), 0.5)        # |fake - real| = 0.5 di setiap piksel
    total, parts = generator_loss(z, fake, real, lambda_l1=100.0)
    assert parts["g_adv"].item() == pytest.approx(LN2)
    assert parts["g_l1"].item() == pytest.approx(0.5)
    assert total.item() == pytest.approx(LN2 + 100.0 * 0.5)


def test_confident_correct_logits_give_near_zero_loss():
    big = torch.full((1, 1, 30, 30), 20.0)
    assert AdversarialLoss()(big, True).item() < 1e-6     # yakin real, memang real
    assert AdversarialLoss()(-big, False).item() < 1e-6   # yakin fake, memang fake
    assert AdversarialLoss()(big, False).item() == pytest.approx(20.0, rel=1e-3)  # yakin tapi salah


def test_lambda_zero_and_l1_reported_before_weighting():
    z = torch.zeros(1, 1, 30, 30)
    fake, real = torch.zeros(1, 3, 8, 8), torch.ones(1, 3, 8, 8)
    total0, parts0 = generator_loss(z, fake, real, lambda_l1=0.0)
    assert total0.item() == pytest.approx(parts0["g_adv"].item())
    _, parts200 = generator_loss(z, fake, real, lambda_l1=200.0)
    assert parts200["g_l1"].item() == pytest.approx(1.0)   # g_l1 tidak dikali lambda


# ---------------------------------------------------------------------------
# 3. Non-saturating: gradien tetap besar saat D yakin gambar palsu
# ---------------------------------------------------------------------------
def test_non_saturating_gradient_is_large_when_d_rejects_fake():
    logits = torch.full((1, 1, 30, 30), -10.0, requires_grad=True)   # D: "jelas palsu"
    n = logits.numel()

    # Bentuk yang dipakai: -log D(G(x)), lewat generator_loss
    total, _ = generator_loss(logits, torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8), lambda_l1=0.0)
    grad_ns = torch.autograd.grad(total, logits)[0]
    # Bentuk minimax asli: minimalkan log(1 - D(G(x))) = -BCE(logit, 0)
    saturating = -F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    grad_sat = torch.autograd.grad(saturating, logits)[0]

    assert abs(grad_ns.sum().item()) > 0.99            # turunan per logit ~ -1/n, totalnya ~1
    assert abs(grad_sat.sum().item()) < 1e-3           # hampir nol: G tidak belajar
    assert grad_ns.sum().item() < 0                    # arah: menaikkan logit (lebih "real")


# ---------------------------------------------------------------------------
# 4. Detach adalah tanggung jawab train.py (losses.py tidak meng-detach)
# ---------------------------------------------------------------------------
def test_detach_controls_whether_g_receives_gradient_from_d_loss():
    G = UNetGenerator(ngf=8, img_size=128)
    D = PatchDiscriminator(ndf=8)
    cond, real = torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128)
    fake = G(cond)

    loss_d, _ = discriminator_loss(D(cond, real), D(cond, fake.detach()))
    loss_d.backward()
    assert all(p.grad is None for p in G.parameters())           # benar: G tidak tersentuh
    assert any(p.grad is not None for p in D.parameters())

    D.zero_grad(set_to_none=True)
    loss_wrong, _ = discriminator_loss(D(cond, real), D(cond, fake))   # LUPA detach
    loss_wrong.backward()
    assert any(p.grad is not None for p in G.parameters())       # G ikut menerima gradien D


def test_generator_loss_sends_gradient_to_g():
    G = UNetGenerator(ngf=8, img_size=128)
    D = PatchDiscriminator(ndf=8)
    cond, real = torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128)
    fake = G(cond)
    total, parts = generator_loss(D(cond, fake), fake, real, lambda_l1=100.0)
    total.backward()
    assert all(p.grad is not None for p in G.parameters())
    assert total.requires_grad and not parts["g_adv"].requires_grad   # dict = nilai untuk log saja


# ---------------------------------------------------------------------------
# 5. Presisi: selalu float32, juga dengan input float16 / autocast
# ---------------------------------------------------------------------------
def test_half_precision_inputs_give_float32_loss():
    logits = torch.randn(2, 1, 30, 30).half()
    fake = (torch.rand(2, 3, 16, 16) * 2 - 1).half()
    real = torch.rand(2, 3, 16, 16) * 2 - 1
    total_g, _ = generator_loss(logits, fake, real, 100.0)
    total_d, _ = discriminator_loss(logits, logits)
    assert total_g.dtype == torch.float32 and total_d.dtype == torch.float32


def test_under_cpu_autocast():
    G = UNetGenerator(ngf=8, img_size=128)
    D = PatchDiscriminator(ndf=8)
    cond, real = torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        fake = G(cond)
        total, _ = generator_loss(D(cond, fake), fake, real, 100.0)
    assert total.dtype == torch.float32 and torch.isfinite(total)
    total.backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="butuh GPU")
def test_under_cuda_autocast_fp16():
    G = UNetGenerator(ngf=8, img_size=128).cuda()
    D = PatchDiscriminator(ndf=8).cuda()
    cond, real = torch.randn(2, 3, 128, 128, device="cuda"), torch.randn(2, 3, 128, 128, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        fake = G(cond)
        loss_d, _ = discriminator_loss(D(cond, real), D(cond, fake.detach()))
        loss_g, _ = generator_loss(D(cond, fake), fake, real, 100.0)
    for loss in (loss_d, loss_g):
        assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss_g.backward()


# ---------------------------------------------------------------------------
# 6. Validasi input
# ---------------------------------------------------------------------------
def test_rejects_invalid_inputs():
    z = torch.zeros(1, 1, 30, 30)
    with pytest.raises(ValueError, match="Shape logit"):
        discriminator_loss(z, torch.zeros(1, 1, 14, 14))
    with pytest.raises(ValueError, match="scale"):
        discriminator_loss(z, z, scale=0.0)
    with pytest.raises(ValueError, match="Shape fake"):
        generator_loss(z, torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 16, 16), 100.0)
    with pytest.raises(ValueError, match="lambda_l1"):
        generator_loss(z, torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8), -1.0)
    with pytest.raises(TypeError, match="bool"):
        AdversarialLoss()(z, 1)
