"""Tests for pyspec.twodim.Spectrum2D / IsotropicSpectrum. Requires xarray."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pyspec import Spectrum2D


def test_plain_array_input_requires_d1_d2():
    with pytest.raises(ValueError, match="d1 and d2 must both be given"):
        Spectrum2D(np.random.randn(8, 8))


def test_plain_array_output_dims():
    rng = np.random.default_rng(0)
    phi = rng.standard_normal((32, 32))
    s = Spectrum2D(phi, d1=1.0, d2=1.0)

    assert isinstance(s.psd, xr.DataArray)
    assert s.psd.dims == ("k2", "k1")
    assert isinstance(s.kappa, xr.DataArray)


def test_dataarray_input_infers_spacing_but_output_dims_are_k2_k1_not_input_names():
    # The input's own dimension names ("y", "x") describe the space
    # domain. Fourier transforming it produces a *wavenumber*-domain
    # output, so the output must not be labeled "y"/"x" even though those
    # are exactly what's used to infer d1/d2.
    rng = np.random.default_rng(1)
    x = np.arange(0, 64, 2.0)
    y = np.arange(0, 32, 1.0)
    phi = xr.DataArray(
        rng.standard_normal((y.size, x.size)),
        dims=["y", "x"],
        coords={"y": y, "x": x},
    )
    s = Spectrum2D(phi)
    assert s.d1 == pytest.approx(2.0)
    assert s.d2 == pytest.approx(1.0)
    assert s.psd.dims == ("k2", "k1")


def test_dataarray_input_dims_kwarg_overrides_output_names_only():
    # The `dims` kwarg names the *output* dimensions; it never changes
    # which input coordinates d1/d2 are inferred from -- that's always
    # the DataArray's own dims, in (dim2, dim1) order.
    rng = np.random.default_rng(1)
    x = np.arange(0, 64, 2.0)
    y = np.arange(0, 32, 1.0)
    phi = xr.DataArray(
        rng.standard_normal((y.size, x.size)),
        dims=["y", "x"],
        coords={"y": y, "x": x},
    )
    s = Spectrum2D(phi, dims=("l", "k"))
    assert s.d1 == pytest.approx(2.0)
    assert s.d2 == pytest.approx(1.0)
    assert s.psd.dims == ("l", "k")


def test_does_not_mutate_input():
    rng = np.random.default_rng(2)
    phi = rng.standard_normal((16, 16))
    original = phi.copy()
    Spectrum2D(phi, d1=1.0, d2=1.0)
    np.testing.assert_array_equal(phi, original)


def test_isotropic_returns_1d_dataarray():
    rng = np.random.default_rng(3)
    phi = rng.standard_normal((64, 64))
    s = Spectrum2D(phi, d1=1.0, d2=1.0)
    iso = s.isotropic()
    assert isinstance(iso.psd, xr.DataArray)
    assert iso.psd.ndim == 1
    assert iso.psd.dims == ("kr",)


def test_isotropic_custom_dim_name():
    rng = np.random.default_rng(4)
    phi = rng.standard_normal((48, 48))
    s = Spectrum2D(phi, d1=1.0, d2=1.0)
    iso = s.isotropic(dim="wavenumber")
    assert iso.psd.dims == ("wavenumber",)


def test_var_matches_core_computation():
    from pyspec import _core

    rng = np.random.default_rng(5)
    phi = rng.standard_normal((40, 40))
    s = Spectrum2D(phi, d1=1.0, d2=1.0, detrend="linear", window="hann")

    k1 = s.psd.coords[s._k1_name].values
    k2 = s.psd.coords[s._k2_name].values
    expected = _core.variance_2d(k1, k2, s.psd.values, s.dk1, s.dk2, n1=s.n1)
    assert s.var == pytest.approx(expected)
