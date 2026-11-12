import numpy as np
import pytest

from src.utility import signal_metrics as sm


def gaussian(t, mu, sigma, height=1.0):
    return height * np.exp(-0.5 * ((t - mu) / sigma) ** 2)


T = np.linspace(0.0, 100.0, 2001)
REF = gaussian(T, 50.0, 5.0)


def test_identical_signals_have_zero_metrics():
    m = sm.signal_metrics(T, REF, REF)
    assert set(m) == set(sm.METRIC_NAMES)
    for name, value in m.items():
        assert value == pytest.approx(0.0, abs=1e-12), name


def test_peak_is_interpolated_between_samples():
    t = np.linspace(0.0, 100.0, 101)
    tp, hp = sm.peak(t, gaussian(t, 50.3, 5.0, height=2.0))
    assert tp == pytest.approx(50.3, abs=2e-3)
    assert hp == pytest.approx(2.0, rel=1e-4)


def test_shift_changes_times_only():
    m = sm.signal_metrics(T, gaussian(T, 52.0, 5.0), REF, window=0.05)
    for name in ('Peak time err.', 'Elution start err.', 'Front position err.',
                 'Tail position err.', 'Elution end err.', 'Ret. time err.'):
        assert m[name] == pytest.approx(2.0, abs=1e-3), name
    for name in ('Peak height err.', 'Peak width err.', 'Front width err.', 'Mass err.',
                 'Variance err.', 'Excess TV', 'Spurious extrema', 'Negative mass',
                 'Local overshoot', 'Local undershoot'):
        assert m[name] == pytest.approx(0.0, abs=1e-4), name


def test_smearing_lowers_peak_and_widens_it():
    # same mass, twice the standard deviation
    m = sm.signal_metrics(T, gaussian(T, 50.0, 10.0, height=0.5), REF)
    assert m['Peak height err.'] == pytest.approx(-0.5, abs=1e-4)
    assert m['Mass err.'] == pytest.approx(0.0, abs=1e-6)
    assert m['Variance err.'] == pytest.approx(3.0, abs=1e-3)
    assert m['Excess TV'] == pytest.approx(-0.5, abs=1e-4)
    # rise from 5% to 95% of a Gaussian takes 2.1275 sigma
    assert m['Front width err.'] == pytest.approx(2.1275 * 5.0, abs=5e-2)
    assert m['Peak width err.'] == pytest.approx(1.0, abs=1e-2)
    assert m['Elution start err.'] < 0.0
    assert m['Peak time err.'] == pytest.approx(0.0, abs=1e-6)


def test_oscillation_is_detected():
    wiggle = 0.05 * np.sin(2.0 * np.pi * T / 4.0) * (np.abs(T - 20.0) < 10.0)
    m = sm.signal_metrics(T, REF + wiggle, REF)
    assert m['Excess TV'] > 0.1
    assert m['Spurious extrema'] >= 8
    assert m['Local undershoot'] == pytest.approx(0.05, abs=5e-3)
    assert m['Negative mass'] < 0.0
    # the peak itself is untouched
    assert m['Peak height err.'] == pytest.approx(0.0, abs=1e-12)


def test_overshoot_at_front_is_local_overshoot_not_shift():
    t = np.linspace(0.0, 10.0, 1001)
    ref = np.where((t > 3.0) & (t < 7.0), 1.0, 0.0)
    over = ref.copy()
    over[(t > 3.0) & (t < 3.3)] = 1.1
    m = sm.signal_metrics(t, over, ref)
    assert m['Local overshoot'] == pytest.approx(0.1, abs=1e-12)
    assert m['Local undershoot'] == pytest.approx(0.0, abs=1e-12)
    assert m['Peak height err.'] == pytest.approx(0.1, abs=1e-12)

    shifted = np.where((t > 3.1) & (t < 7.1), 1.0, 0.0)
    m = sm.signal_metrics(t, shifted, ref)
    assert m['Local overshoot'] == pytest.approx(0.0, abs=1e-12)
    assert m['Local undershoot'] == pytest.approx(0.0, abs=1e-12)
    assert m['Front position err.'] == pytest.approx(0.1, abs=1e-2)


def test_rectangular_pulse_peak_is_not_extrapolated():
    t = np.linspace(0.0, 10.0, 101)
    ref = np.where((t > 3.0) & (t <= 7.0), 1.0, 0.0)
    assert sm.peak(t, ref)[1] == pytest.approx(1.0, abs=1e-14)
    m = sm.signal_metrics(t, ref, ref)
    assert m['Peak height err.'] == pytest.approx(0.0, abs=1e-14)
    assert m['Front width err.'] == pytest.approx(0.0, abs=1e-14)


def test_turning_points_ignore_noise():
    c = np.array([0.0, 1.0, 0.999, 1.0, 0.0, 0.5, 0.0])
    assert sm.count_turning_points(c, noise=0.01) == 3
    assert sm.count_turning_points(c, noise=1e-6) == 5
    assert sm.count_turning_points(np.linspace(0.0, 1.0, 10), noise=0.01) == 0


def test_breakthrough_curve_has_no_falling_crossings():
    t = np.linspace(0.0, 10.0, 501)
    ref = 0.5 * (1.0 + np.tanh(t - 5.0))
    m = sm.signal_metrics(t, 0.5 * (1.0 + np.tanh(t - 5.5)), ref)
    assert np.isnan(m['Tail position err.'])
    assert np.isnan(m['Peak width err.'])
    assert m['Front position err.'] == pytest.approx(0.5, abs=1e-3)


def test_columns_for_refinement_series():
    ref = np.column_stack((REF, gaussian(T, 60.0, 4.0)))
    sols = [ref * 0.9, ref]
    header, values = sm.signal_metric_columns(T, sols, ref)
    n = len(sm.METRIC_NAMES)
    assert len(header) == 2 * n
    assert header[0] == sm.METRIC_NAMES[0] + ' (comp 0)'
    assert values.shape == (2, 2 * n)
    i = header.index('Peak height err. (comp 1)')
    assert values[0, i] == pytest.approx(-0.1, abs=1e-12)
    assert np.allclose(values[1], 0.0, atol=1e-12)

    header, values = sm.signal_metric_columns(T, [REF], REF)
    assert header == sm.METRIC_NAMES

    with pytest.raises(ValueError):
        sm.signal_metric_columns(T, [ref[:-1]], ref)
