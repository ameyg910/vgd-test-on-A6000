"""Transformer denoiser: shapes, conditioning, masking and CFG behaviour."""

from __future__ import annotations

from typing import Tuple

import pytest
import torch
from torch import Tensor

from diffusion.base.preconditioning import EDMConfig, EDMPreconditioner
from diffusion.base.transformer import (RMSNorm, SwiGLU, TransformerDenoiser,
                                        TransformerDenoiserConfig)


def small_config(**overrides: object) -> TransformerDenoiserConfig:
    """A fast configuration for tests; the real defaults are checked separately."""
    base = dict(input_dim=64, num_tokens=8, hidden_dim=32, depth=2, num_heads=4,
                context_dim=64, time_embed_dim=32)
    base.update(overrides)
    return TransformerDenoiserConfig(**base)  # type: ignore[arg-type]


@pytest.fixture
def model() -> TransformerDenoiser:
    """A model whose output head is *not* zero-initialised.

    The real initialisation zeroes ``token_out`` so training starts from the
    identity denoiser (asserted separately). That makes every output identically
    zero, which would make the wiring tests below vacuously pass, so here we
    perturb the head to stand in for a partially trained network.
    """
    torch.manual_seed(0)
    model = TransformerDenoiser(small_config())
    torch.nn.init.normal_(model.token_out.weight, std=0.05)
    torch.nn.init.normal_(model.token_out.bias, std=0.05)
    return model


def test_rmsnorm_produces_unit_rms() -> None:
    norm = RMSNorm(32)
    x = torch.randn(4, 8, 32) * 5.0
    out = norm(x)
    assert torch.allclose(out.pow(2).mean(dim=-1).sqrt(), torch.ones(4, 8), atol=1e-3)


def test_swiglu_shape_and_nonlinearity() -> None:
    ffn = SwiGLU(32)
    x = torch.randn(4, 32)
    assert ffn(x).shape == (4, 32)
    assert not torch.allclose(ffn(2 * x), 2 * ffn(x), atol=1e-4)


def test_forward_shapes_unconditional(model: TransformerDenoiser) -> None:
    x = torch.randn(6, 64)
    out = model(x, torch.randn(6))
    assert out.shape == (6, 64) and bool(torch.isfinite(out).all())


def test_forward_with_issue_and_episodic(model: TransformerDenoiser) -> None:
    x, c_noise = torch.randn(3, 64), torch.randn(3)
    issue = torch.randn(3, 64)
    episodic = torch.randn(3, 5, 64)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]], dtype=torch.bool)
    out = model(x, c_noise, issue=issue, episodic=episodic, episodic_mask=mask)
    assert out.shape == (3, 64) and bool(torch.isfinite(out).all())


def test_episodic_mask_blocks_padded_slots(model: TransformerDenoiser) -> None:
    """Padded memory slots must not change the output, whatever they contain."""
    torch.manual_seed(0)
    x, c_noise, issue = torch.randn(2, 64), torch.randn(2), torch.randn(2, 64)
    episodic = torch.randn(2, 4, 64)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    first = model(x, c_noise, issue=issue, episodic=episodic, episodic_mask=mask)

    polluted = episodic.clone()
    polluted[0, 2:] = 1e3
    polluted[1, 1:] = -1e3
    second = model(x, c_noise, issue=issue, episodic=polluted, episodic_mask=mask)
    assert torch.allclose(first, second, atol=1e-5)


def test_variable_length_memory_changes_output(model: TransformerDenoiser) -> None:
    """Unmasking a real slot must change the prediction, or the mask is inert."""
    torch.manual_seed(0)
    x, c_noise, issue = torch.randn(2, 64), torch.randn(2), torch.randn(2, 64)
    episodic = torch.randn(2, 4, 64)
    short = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    long = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.bool)
    assert not torch.allclose(
        model(x, c_noise, issue=issue, episodic=episodic, episodic_mask=short),
        model(x, c_noise, issue=issue, episodic=episodic, episodic_mask=long), atol=1e-5)


def test_conditioning_changes_output(model: TransformerDenoiser) -> None:
    torch.manual_seed(0)
    x, c_noise = torch.randn(4, 64), torch.randn(4)
    unconditional = model(x, c_noise)
    conditional = model(x, c_noise, issue=torch.randn(4, 64))
    assert not torch.allclose(unconditional, conditional, atol=1e-6)


def test_drop_conditioning_recovers_unconditional_path(model: TransformerDenoiser) -> None:
    """A dropped row must be identical to the no-issue forward pass.

    This is the invariant classifier-free guidance rests on: if dropping does
    not fully remove the conditioning, the unconditional branch is not
    unconditional and the CFG extrapolation is meaningless.
    """
    torch.manual_seed(0)
    x, c_noise = torch.randn(4, 64), torch.randn(4)
    issue = torch.randn(4, 64)
    dropped = model(x, c_noise, issue=issue,
                    drop_conditioning=torch.ones(4, dtype=torch.bool))
    plain = model(x, c_noise)
    assert torch.allclose(dropped, plain, atol=1e-5)


def test_drop_conditioning_is_per_row(model: TransformerDenoiser) -> None:
    torch.manual_seed(0)
    x, c_noise = torch.randn(4, 64), torch.randn(4)
    issue = torch.randn(4, 64)
    drop = torch.tensor([True, False, True, False])
    mixed = model(x, c_noise, issue=issue, drop_conditioning=drop)
    plain = model(x, c_noise)
    full = model(x, c_noise, issue=issue)
    assert torch.allclose(mixed[drop], plain[drop], atol=1e-5)
    assert torch.allclose(mixed[~drop], full[~drop], atol=1e-5)


def test_output_projection_starts_at_zero() -> None:
    """Zero-init output means the wrapped denoiser starts as the identity map."""
    model = TransformerDenoiser(small_config())
    x = torch.randn(4, 64)
    assert torch.allclose(model(x, torch.randn(4)), torch.zeros(4, 64), atol=1e-8)
    pre = EDMPreconditioner(model, EDMConfig(sigma_data=1.0))
    tiny = torch.full((4,), 1e-4)
    assert torch.allclose(pre(x, tiny), x, atol=1e-3)


def test_rejects_indivisible_token_split() -> None:
    with pytest.raises(ValueError):
        TransformerDenoiserConfig(input_dim=100, num_tokens=7)


def test_rejects_wrong_input_dim(model: TransformerDenoiser) -> None:
    with pytest.raises(ValueError):
        model(torch.randn(2, 63), torch.randn(2))


def test_default_configuration_is_within_parameter_budget() -> None:
    """~50M parameters, per the workplan's single-A6000 budget."""
    model = TransformerDenoiser()
    assert 40e6 < model.num_parameters < 60e6
    assert model.config.input_dim == 768


def test_forward_is_deterministic_in_eval(model: TransformerDenoiser) -> None:
    model.eval()
    x, c_noise = torch.randn(3, 64), torch.randn(3)
    with torch.no_grad():
        assert torch.equal(model(x, c_noise), model(x, c_noise))
