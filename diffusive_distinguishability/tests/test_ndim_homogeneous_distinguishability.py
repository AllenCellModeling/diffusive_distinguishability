import pytest
import diffusive_distinguishability.ndim_homogeneous_distinguishability as hd


def test_sanity_check():
    assert True


def test_zero_diffusivity():
    df = hd.simulate_diffusion_df(1, 0, 100, 1, 0)
    for dr in df['dr']:
        assert dr == 0
