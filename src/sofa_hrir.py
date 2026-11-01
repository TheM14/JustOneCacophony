"""Local spherical interpolation of full complex HRIR spectra."""
from __future__ import annotations

from collections import OrderedDict
import math
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import ConvexHull, cKDTree

from sofa_canonical import CanonicalHrtf, load_simple_free_field_hrir


class FullHrirField:
    response_kind = "hrir"

    def __init__(self, source, *, method="lf_logmag_phase", shell_radius_m=None,
                 fft_size=None):
        if method not in ("lf_logmag_phase", "lf_mca_zero"):
            raise ValueError("unsupported full HRIR interpolation method")
        self.canonical = (source if isinstance(source, CanonicalHrtf) else
                          load_simple_free_field_hrir(source, target_sample_rate_hz=48000))
        c = self.canonical
        if c.sample_rate_hz != 48000:
            raise ValueError("full HRIR renderer requires 48000 Hz")
        shells = c.shells_m
        if shell_radius_m is not None and (not math.isfinite(shell_radius_m) or shell_radius_m <= 0):
            raise ValueError("shell radius must be finite and positive")
        if shell_radius_m is None and len(shells) != 1:
            raise ValueError("multi-radius SOFA requires an explicit --hrtf-radius-m")
        selected = c.shell_indices(shells[0] if shell_radius_m is None else shell_radius_m)
        if np.any(np.sum(c.hrir[selected]**2, axis=-1) <= 0):
            raise ValueError("each selected HRIR ear must contain signal energy")
        self.measurement_radius_m = float(np.median(c.measurement_radius_m[selected]))
        support = c.taps + int(math.ceil(float(np.max(c.delay_samples[selected])))) + 1
        if fft_size is None:
            fft_size = max(2048, 1 << (2*support-1).bit_length())
            if fft_size > 16384:
                raise ValueError("HRIR and Data.Delay exceed the automatic FFT budget")
        self.method, self.fft_size = method, int(fft_size)
        if self.fft_size < 64 or self.fft_size % 2 or c.taps + np.max(c.delay_samples[selected]) >= self.fft_size//2:
            raise ValueError("HRIR and Data.Delay exceed the selected FFT time budget")
        receiver = c.receiver_position_cartesian_m[selected]
        if not np.allclose(receiver, receiver[0], atol=1e-9, rtol=0):
            raise ValueError("moving receiver geometry is unsupported")
        self.receivers = receiver[0].copy()
        unit = c.unit_directions[selected]
        _, first, inverse = np.unique(np.round(unit, 10), axis=0, return_index=True, return_inverse=True)
        for group in np.flatnonzero(np.bincount(inverse) > 1):
            members = selected[inverse == group]
            if not (np.allclose(c.hrir[members], c.hrir[members[0]], atol=1e-10, rtol=1e-8)
                    and np.allclose(c.delay_samples[members], c.delay_samples[members[0]], atol=1e-10, rtol=0)):
                raise ValueError("coincident SOFA measurements disagree")
        selected = selected[np.sort(first)]
        self.directions = c.unit_directions[selected].copy()
        self.tree = cKDTree(self.directions)
        hull = ConvexHull(self.directions)
        if np.any(hull.equations[:, 3] >= -1e-9):
            raise ValueError("SOFA must enclose the listener; extrapolation is unsupported")
        self.faces = hull.simplices
        vertices = self.directions[self.faces]
        dots = np.einsum("fij,fkj->fik", vertices, vertices)
        self.max_triangle_edge_degrees = math.degrees(math.acos(float(np.clip(dots.min(), -1, 1))))
        matrix = self.directions[self.faces].transpose(0, 2, 1)
        if np.max(np.linalg.cond(matrix)) > 1e8:
            raise ValueError("ill-conditioned spherical interpolation mesh")
        self.inverse = np.linalg.inv(matrix)
        self.frequencies = np.fft.rfftfreq(self.fft_size, 1/c.sample_rate_hz)
        self.omega = 2*np.pi*self.frequencies/c.sample_rate_hz
        raw = np.fft.rfft(c.hrir[selected], self.fft_size)
        fit = (self.frequencies >= 200) & (self.frequencies <= 1500)
        x = self.omega[fit]-self.omega[fit].mean()
        phase = np.unwrap(np.angle(raw), axis=-1)[..., fit]
        self.delays = -np.sum(x*phase, axis=-1)/np.sum(x*x) + c.delay_samples[selected]
        residual_phase = phase+(self.delays-c.delay_samples[selected])[..., None]*self.omega[fit]
        residual_phase -= residual_phase.mean(axis=-1, keepdims=True)
        self.phase_fit_rms_rad = np.sqrt(np.mean(residual_phase**2, axis=-1))
        self.recording_origin_samples = int(round(float(np.median(self.delays.mean(axis=1)))))
        self.common_latency_samples = self.fft_size//2
        physical = raw*np.exp(-1j*self.omega*c.delay_samples[selected, :, None])
        self.floor = max(float(np.max(np.abs(physical)))*1e-6, 1e-15)
        self.aligned = physical*np.exp(1j*self.omega*self.delays[..., None])
        self.log_magnitude = None
        self.auditory = None
        if method == "lf_logmag_phase":
            self.log_magnitude = np.log(np.maximum(np.abs(physical), self.floor))
        else:
            erb = lambda f: 21.4*np.log10(1+.00437*f)
            self.centres = (10**(np.linspace(erb(50), erb(20000), 41)/21.4)-1)/.00437
            bw = 1.019*24.7*(1+.00437*self.centres)
            self.bank = (1+((self.frequencies[None, :]-self.centres[:, None])/bw[:, None])**2)**-4
            self.bank[:, 1:-1] *= 2
            self.bank /= self.bank.sum(axis=1, keepdims=True)
            self.auditory = self._auditory_db(physical)
        self._cache = OrderedDict()

    def _auditory_db(self, spectrum):
        return 10*np.log10(np.maximum(np.abs(spectrum)**2 @ self.bank.T, 1e-24))

    def weights(self, direction):
        q = np.asarray(direction, dtype=np.float64)
        if q.shape != (3,) or not np.isfinite(q).all() or np.linalg.norm(q) <= 1e-12:
            raise ValueError("query direction must be a finite nonzero vector")
        q = q/np.linalg.norm(q)
        distance, nearest = self.tree.query(q)
        if distance < 1e-10:
            return np.array([nearest]), np.array([1.]), 0.
        g = self.inverse @ q
        valid = np.flatnonzero(np.min(g, axis=1) >= -1e-10)
        if not len(valid):
            raise ValueError("no containing spherical triangle")
        face = int(valid[0])
        weights = np.maximum(g[face], 0)
        weights /= weights.sum()
        ids = self.faces[face]
        vertices = self.directions[ids]
        edge = math.degrees(math.acos(float(np.clip(np.min(vertices @ vertices.T), -1, 1))))
        return ids, weights, edge

    def spectrum(self, direction):
        q = np.asarray(direction, dtype=np.float64)
        if q.shape != (3,) or not np.isfinite(q).all() or np.linalg.norm(q) <= 1e-12:
            raise ValueError("query direction must be a finite nonzero vector")
        q = q/np.linalg.norm(q)
        key = tuple(q)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key].copy()
        ids, weights, _ = self.weights(q)
        delay = weights @ self.delays[ids]
        spectrum = np.einsum("i,ief->ef", weights, self.aligned[ids])
        spectrum *= np.exp(-1j*self.omega*delay[:, None])
        if len(ids) > 1:
            if self.method == "lf_logmag_phase":
                magnitude = np.exp(np.einsum("i,ief->ef", weights, self.log_magnitude[ids]))
                spectrum = magnitude*spectrum/np.maximum(np.abs(spectrum), self.floor)
            else:
                target = np.einsum("i,ieb->eb", weights, self.auditory[ids])
                difference = target-self._auditory_db(spectrum)
                correction = PchipInterpolator(self.centres, difference, axis=-1)(
                    np.clip(self.frequencies, self.centres[0], self.centres[-1]))
                spectrum *= 10**(np.clip(correction, -12, 12)/20)
        self._cache[key] = spectrum.copy()
        if len(self._cache) > 128:
            self._cache.popitem(last=False)
        return spectrum

    def info(self):
        arrays = [self.aligned, self.log_magnitude, self.auditory, self.delays,
                  self.inverse, self.directions, self.faces]
        return dict(source_sha256=self.canonical.source_sha256,
                    sample_rate_hz=self.canonical.sample_rate_hz,
                    convention=self.canonical.convention,
                    measurement_radius_m=self.measurement_radius_m,
                    method=self.method, fft_size=self.fft_size,
                    retained_filter_taps=self.fft_size,
                    recording_origin_samples=self.recording_origin_samples,
                    interpolation_storage_bytes=sum(x.nbytes for x in arrays if x is not None),
                    max_triangle_edge_degrees=self.max_triangle_edge_degrees,
                    phase_fit_rms_rad_p95=float(np.percentile(self.phase_fit_rms_rad, 95)),
                    ear_model="measured-SOFA-receivers",
                    receiver_positions_m=self.receivers.tolist())
