# pyspec

Spectral estimates from in-situ oceanographic data and models, rebuilt on
`xarray`.

Originally by Cesar B Rocha; developed for a project on horizontal
wavenumber spectra in Drake Passage.

## Install

```
pip install -e ".[test]"
```

Requires Python >= 3.10, numpy, scipy, and xarray.

## Quickstart

```python
import numpy as np
from pyspec import Spectrum1D, Spectrum2D, spec_helm_decomp

# 1-D spectrum (numpy or xarray input; always returns xarray output)
phi = np.random.randn(1000)
s = Spectrum1D(phi, dt=1.0)
s.psd    # xarray.DataArray, dims ("freq",)
s.var    # float, Parseval-theorem-consistent variance

# 2-D spectrum + isotropic (azimuthally-averaged) spectrum
field = np.random.randn(128, 128)
s2 = Spectrum2D(field, d1=1.0, d2=1.0)
iso = s2.isotropic()   # IsotropicSpectrum, .psd is 1-D DataArray vs radial k

# Helmholtz (rotational/divergent) decomposition
k = np.logspace(-3, 0, 60)
result = spec_helm_decomp(k, Cu=k**-2, Cv=k**-2)
result.psi   # rotational velocity-variance spectrum (multiply by 0.5 for KE)
result.phi   # divergent velocity-variance spectrum (multiply by 0.5 for KE)
```

If you pass an `xarray.DataArray` with a coordinate instead of a plain
array, `dt`/`d1`/`d2` are inferred automatically from the coordinate
spacing -- but the output dimension is *not* named after your input's
dimension. A Fourier transform changes the domain (time -> frequency,
space -> wavenumber), so the output always gets a frequency/wavenumber
name (`"freq"` for `Spectrum1D`, `("k2", "k1")` for `Spectrum2D`) unless
you override it explicitly with `dim`/`dims`:

```python
import xarray as xr
da = xr.DataArray(phi, dims=["time"], coords={"time": np.arange(1000) * 0.5})
s = Spectrum1D(da)             # dt=0.5 inferred from "time"; output dim is "freq"
s2 = Spectrum1D(da, dim="omega")  # output dim named "omega" instead
```

## Package layout

- `pyspec._core` -- pure-numpy computational routines (periodograms,
  isotropic averaging, confidence intervals, binning, slope fitting). No
  xarray dependency; fast to import and test.
- `pyspec.onedim` / `pyspec.twodim` -- `Spectrum1D`, `Spectrum2D`,
  `IsotropicSpectrum`: the xarray-facing object API.
- `pyspec.helmholtz` -- `spec_helm_decomp`, returning a
  `HelmholtzDecomposition` dataclass of `xarray.DataArray`s.
- `pyspec.errors` / `pyspec.binning` -- confidence intervals, spectral
  slope fitting, log-decade binning.

## Testing

```
pytest
```

`tests/test_core.py` covers the numpy computational core (including a
regression check of the vectorized isotropic averaging against a direct
port of the original loop). The rest of the suite exercises the
xarray-facing API and requires xarray; `tests/conftest.py` skips it
automatically if xarray isn't installed.

See `MIGRATION.md` for what changed relative to the pre-refactor version,
and why.

## License

MIT
