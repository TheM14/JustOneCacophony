"""Orthonormal real spherical harmonics in ACN order, through fifth order."""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial import SphericalVoronoi


def _associated_legendre(order: int, degree: int, x: np.ndarray) -> np.ndarray:
    """P_degree^order(x), including the Condon-Shortley phase."""
    m = int(order)
    l = int(degree)
    if not 0 <= m <= l:
        raise ValueError("associated Legendre indices require 0 <= m <= l")
    x = np.asarray(x, dtype=np.float64)
    p_mm = np.ones_like(x)
    if m:
        double_factorial = 1.0
        for value in range(1, 2 * m, 2):
            double_factorial *= value
        p_mm = ((-1.0) ** m) * double_factorial * np.power(
            np.maximum(0.0, 1.0 - x * x), 0.5 * m)
    if l == m:
        return p_mm
    p_m1 = x * (2 * m + 1) * p_mm
    if l == m + 1:
        return p_m1
    previous_previous = p_mm
    previous = p_m1
    for current_degree in range(m + 2, l + 1):
        current = (
            (2 * current_degree - 1) * x * previous
            - (current_degree + m - 1) * previous_previous
        ) / float(current_degree - m)
        previous_previous, previous = previous, current
    return previous


def real_spherical_harmonics(directions, order: int = 5) -> np.ndarray:
    """Return [directions,(order+1)^2] ACN/N3D real harmonics.

    Coordinates use SOFA listener axes: +X front, +Y left, +Z up.  The basis is
    orthonormal over the sphere and includes the Condon-Shortley phase.
    """
    maximum_order = int(order)
    if not 0 <= maximum_order <= 12:
        raise ValueError("supported spherical-harmonic orders are 0..12")
    vectors = np.asarray(directions, dtype=np.float64)
    one = vectors.ndim == 1
    if one:
        vectors = vectors[None, :]
    if vectors.ndim != 2 or vectors.shape[1] != 3 or not np.isfinite(vectors).all():
        raise ValueError("directions must have finite shape [M,3]")
    length = np.linalg.norm(vectors, axis=1)
    if np.any(length <= 1.0e-15):
        raise ValueError("spherical-harmonic directions must be non-zero")
    unit = vectors / length[:, None]
    azimuth = np.arctan2(unit[:, 1], unit[:, 0])
    cos_colatitude = np.clip(unit[:, 2], -1.0, 1.0)
    result = np.empty((len(unit), (maximum_order + 1) ** 2), dtype=np.float64)
    column = 0
    for degree in range(maximum_order + 1):
        for m in range(-degree, degree + 1):
            absolute = abs(m)
            normalization = math.sqrt(
                (2 * degree + 1) / (4.0 * math.pi)
                * math.factorial(degree - absolute)
                / math.factorial(degree + absolute))
            legendre = _associated_legendre(absolute, degree, cos_colatitude)
            if m < 0:
                value = math.sqrt(2.0) * normalization * legendre * np.sin(
                    absolute * azimuth)
            elif m > 0:
                value = math.sqrt(2.0) * normalization * legendre * np.cos(
                    m * azimuth)
            else:
                value = normalization * legendre
            result[:, column] = value
            column += 1
    return result[0] if one else result


def spherical_voronoi_weights(directions) -> np.ndarray:
    """Area weights for an irregular full-sphere grid, with uniform fallback."""
    vectors = np.asarray(directions, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("directions must have shape [M,3]")
    unit = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    if len(unit) < 4:
        return np.full(len(unit), 1.0 / len(unit), dtype=np.float64)
    try:
        voronoi = SphericalVoronoi(unit, radius=1.0, center=np.zeros(3))
        areas = np.asarray(voronoi.calculate_areas(), dtype=np.float64)
        if not np.isfinite(areas).all() or np.any(areas <= 0.0):
            raise ValueError("invalid spherical Voronoi areas")
        return areas / np.sum(areas, dtype=np.float64)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return np.full(len(unit), 1.0 / len(unit), dtype=np.float64)


def fit_real_spherical_harmonics(directions, values, *, order: int = 5,
                                 ridge: float = 1.0e-6,
                                 weights=None) -> np.ndarray:
    """Weighted ridge fit.  Output shape is [terms,...value trailing axes]."""
    basis = real_spherical_harmonics(directions, order=order)
    target = np.asarray(values)
    if target.shape[0] != basis.shape[0]:
        raise ValueError("spherical-harmonic target count does not match directions")
    if target.dtype.kind == "c":
        target = np.asarray(target, dtype=np.complex128)
        solve_dtype = np.complex128
    else:
        target = np.asarray(target, dtype=np.float64)
        solve_dtype = np.float64
    if weights is None:
        weight = spherical_voronoi_weights(directions)
    else:
        weight = np.asarray(weights, dtype=np.float64)
        if weight.shape != (len(basis),) or np.any(weight < 0.0) or not np.isfinite(weight).all():
            raise ValueError("weights must be finite non-negative [M]")
        total = float(np.sum(weight))
        if total <= 0.0:
            raise ValueError("weights must have positive sum")
        weight = weight / total
    flat = target.reshape(len(target), -1)
    weighted_basis = basis * weight[:, None]
    gram = basis.T @ weighted_basis
    regularization = float(ridge)
    if not math.isfinite(regularization) or regularization < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    scale = float(np.trace(gram)) / gram.shape[0]
    system = gram + np.eye(gram.shape[0], dtype=np.float64) * regularization * scale
    right = basis.T @ (weight[:, None] * flat)
    coefficients = np.linalg.solve(system.astype(solve_dtype), right.astype(solve_dtype))
    return coefficients.reshape((basis.shape[1],) + target.shape[1:])


def evaluate_real_spherical_harmonics(coefficients, directions,
                                      *, order: int = 5) -> np.ndarray:
    basis = real_spherical_harmonics(directions, order=order)
    coeff = np.asarray(coefficients)
    if coeff.shape[0] != (int(order) + 1) ** 2:
        raise ValueError("coefficient term count does not match order")
    return np.tensordot(basis, coeff, axes=([-1], [0]))
