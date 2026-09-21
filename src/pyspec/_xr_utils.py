"""Helpers for accepting either numpy arrays or xarray.DataArrays as input,
while the public API always emits xarray.DataArrays as output.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _infer_spacing(coord: np.ndarray, name: str) -> float:
    if coord.size < 2:
        raise ValueError(
            f"coordinate {name!r} has fewer than 2 points; can't infer spacing"
        )
    diffs = np.diff(coord.astype(float))
    spacing = diffs[0]
    if spacing == 0 or not np.allclose(diffs, spacing, rtol=1e-6):
        raise ValueError(
            f"coordinate {name!r} is not evenly spaced; pass the sample "
            "spacing explicitly instead of relying on inference"
        )
    return float(spacing)


def as_1d_array_and_dt(
    data, dt: float | None = None, dim: str | None = None
) -> tuple[np.ndarray, float, str]:
    """Return ``(values, dt, out_dim_name)`` for a 1-D input.

    ``data`` may be a plain 1-D array (``dt`` then required) or a 1-D
    ``xarray.DataArray`` (``dt`` inferred from its coordinate unless given).

    ``dim`` names the *output* dimension (e.g. ``"freq"``) and is never
    used to pick which input coordinate to read ``dt`` from -- for a 1-D
    array there is only one dimension to read, so that's unambiguous. The
    output is never named after the input's own dimension: a Fourier
    transform changes the domain (time -> frequency, space -> wavenumber),
    so reusing the input's name (e.g. "time") for a frequency-domain
    output would be misleading, even though it's the input's name that
    determines ``dt``.
    """
    if isinstance(data, xr.DataArray):
        if data.ndim != 1:
            raise ValueError(f"expected a 1-D DataArray, got dims {data.dims}")
        input_dim = data.dims[0]
        out_dim = dim if dim is not None else "freq"
        arr = np.asarray(data.values, dtype=float)
        if dt is None:
            if input_dim not in data.coords:
                raise ValueError(
                    f"DataArray has no coordinate for dim {input_dim!r}; "
                    "pass dt explicitly"
                )
            dt = _infer_spacing(data.coords[input_dim].values, input_dim)
        return arr, dt, out_dim

    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D array, got shape {arr.shape}")
    if dt is None:
        raise ValueError("dt must be given explicitly for plain numpy input")
    return arr, dt, (dim or "freq")


def as_2d_array_and_spacing(
    data,
    d1: float | None = None,
    d2: float | None = None,
    dims: tuple[str, str] | None = None,
) -> tuple[np.ndarray, float, float, tuple[str, str]]:
    """Return ``(values, d1, d2, (out_dim2_name, out_dim1_name))`` for a 2-D input.

    Axis -1 is treated as the "1" axis (spacing ``d1``), axis -2 as the
    "2" axis (spacing ``d2``), matching pyspec's original convention.

    ``dims`` names the *output* dimensions and is never used to pick
    which of the input's own dimensions to read ``d1``/``d2`` from -- for
    a 2-D ``DataArray`` that's always ``data.dims`` in ``(dim2, dim1)``
    order. The output is never named after the input's own dimension
    names: a Fourier transform changes the domain (space -> wavenumber),
    so reusing input names (e.g. "y", "x") for a wavenumber-domain output
    would be misleading, even though it's the input's names that
    determine ``d1``/``d2``.
    """
    if isinstance(data, xr.DataArray):
        if data.ndim != 2:
            raise ValueError(f"expected a 2-D DataArray, got dims {data.dims}")
        input_dim2, input_dim1 = data.dims
        out_dim2, out_dim1 = dims if dims is not None else ("k2", "k1")
        arr = np.asarray(data.values, dtype=float)
        if d1 is None:
            if input_dim1 not in data.coords:
                raise ValueError(
                    f"DataArray has no coordinate for dim {input_dim1!r}; "
                    "pass d1 explicitly"
                )
            d1 = _infer_spacing(data.coords[input_dim1].values, input_dim1)
        if d2 is None:
            if input_dim2 not in data.coords:
                raise ValueError(
                    f"DataArray has no coordinate for dim {input_dim2!r}; "
                    "pass d2 explicitly"
                )
            d2 = _infer_spacing(data.coords[input_dim2].values, input_dim2)
        return arr, d1, d2, (out_dim2, out_dim1)

    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {arr.shape}")
    if d1 is None or d2 is None:
        raise ValueError("d1 and d2 must both be given explicitly for plain numpy input")
    out_dim2, out_dim1 = dims if dims is not None else ("k2", "k1")
    return arr, d1, d2, (out_dim2, out_dim1)
