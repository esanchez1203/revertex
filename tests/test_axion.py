from __future__ import annotations

import itertools

import awkward as ak
import lh5
import numpy as np
import pytest
from scipy.integrate import quad

from revertex.cli import cli
from revertex.generators import axion

ME = axion._ME  # keV
E_PP = 5490.0  # keV, p + d -> 3He + a


def _momenta(arr, i):
    """3-momenta (keV) of particle i of every event."""
    ekin = ak.to_numpy(arr.ekin[:, i])
    pid = ak.to_numpy(arr.g4_pid[:, i])
    mass = np.where(pid == 22, 0.0, ME)
    p = np.sqrt(ekin**2 + 2 * ekin * mass)
    d = np.column_stack([ak.to_numpy(arr[c][:, i]) for c in ("px", "py", "pz")])
    return p[:, None] * d


def test_compton_cross_section_matches_literature():
    # Borexino, PRD 85 092003 (2012), Eq. 10: 4.3e-25 g^2 cm^2 at 5.49 MeV
    assert axion.compton_cross_section(5.49, eunit="MeV") == pytest.approx(
        4.3e-25, rel=0.01
    )


@pytest.mark.parametrize(
    ("energy", "mass"), [(5490.0, 0.0), (1000.0, 0.0), (2000.0, 1000.0)]
)
def test_compton_differential_integrates_to_total(energy, mass):
    alpha = 1 / 137.035999084
    hbarc_cm = 1.973269804e-8  # keV cm
    total = quad(
        lambda c: axion.compton_dsigma_dcos(energy, mass, c), -1, 1, limit=200
    )[0]
    assert total * alpha * hbarc_cm**2 == pytest.approx(
        axion.compton_cross_section(energy, mass), rel=1e-6
    )


def test_compton_kinematics_and_distribution():
    arr = axion.generate_axion_kinematics(
        200_000, process="compton", energy=E_PP, seed=1
    )
    assert ak.all(arr.g4_pid[:, 0] == 11)
    assert ak.all(arr.g4_pid[:, 1] == 22)

    # energy and momentum conservation: the electron and photon momenta add up to
    # the axion momentum (massless here), which gives the axion direction per event
    assert np.allclose(ak.to_numpy(ak.sum(arr.ekin, axis=1)), E_PP)
    p_tot = _momenta(arr, 0) + _momenta(arr, 1)
    assert np.allclose(np.linalg.norm(p_tot, axis=1), E_PP)
    axis = p_tot / E_PP

    # the axions are isotropic
    assert np.all(np.abs(axis.mean(axis=0)) < 0.01)

    # photon angle to the axion follows d sigma / d cos
    k = _momenta(arr, 1)
    cos_g = np.sum(k * axis, axis=1) / np.linalg.norm(k, axis=1)
    edges = np.linspace(-1, 1, 41)
    hist, _ = np.histogram(cos_g, edges)
    expected = np.array(
        [
            quad(lambda c: axion.compton_dsigma_dcos(E_PP, 0.0, c), a, b)[0]
            for a, b in itertools.pairwise(edges)
        ]
    )
    expected *= len(cos_g) / expected.sum()
    ok = expected > 100
    pull = (hist[ok] - expected[ok]) / np.sqrt(expected[ok])
    assert np.max(np.abs(pull)) < 5
    assert np.sum(pull**2) / ok.sum() < 2


def test_compton_massive_axion_kinematics():
    arr = axion.generate_axion_kinematics(
        10_000, process="compton", energy=2000, mass=1000, seed=2
    )
    p_a = np.sqrt(2000.0**2 - 1000.0**2)
    p_tot = _momenta(arr, 0) + _momenta(arr, 1)
    assert np.allclose(np.linalg.norm(p_tot, axis=1), p_a)
    assert np.allclose(ak.to_numpy(ak.sum(arr.ekin, axis=1)), 2000)


def test_pair_kinematics():
    arr = axion.generate_axion_kinematics(20_000, process="pair", energy=E_PP, seed=3)
    assert ak.all(arr.g4_pid[:, 0] == 11)
    assert ak.all(arr.g4_pid[:, 1] == -11)
    assert np.allclose(ak.to_numpy(ak.sum(arr.ekin, axis=1)), E_PP - 2 * ME)
    assert ak.all(arr.ekin > 0)

    # leptons are isotropic overall but emitted close to each other (both forward
    # along the axion), and the pair is symmetric in Born approximation
    pm, pp = _momenta(arr, 0), _momenta(arr, 1)
    d_m = pm / np.linalg.norm(pm, axis=1, keepdims=True)
    d_p = pp / np.linalg.norm(pp, axis=1, keepdims=True)
    assert np.all(np.abs(d_m.mean(axis=0)) < 0.03)
    assert np.median(np.degrees(np.arccos(np.sum(d_m * d_p, axis=1)))) < 40
    e_m, e_p = ak.to_numpy(arr.ekin[:, 0]), ak.to_numpy(arr.ekin[:, 1])
    assert abs(np.mean(e_m) - np.mean(e_p)) < 5 * np.std(e_m - e_p) / np.sqrt(len(e_m))


def test_pair_energy_spectrum_matches_matrix_element():
    """Sampled T+ spectrum vs the matrix element weighted proposal."""
    arr = axion.generate_axion_kinematics(100_000, process="pair", energy=E_PP, seed=4)
    t_plus = ak.to_numpy(arr.ekin[:, 1])
    edges = np.linspace(0, E_PP - 2 * ME, 11)
    hist, _ = np.histogram(t_plus, edges)

    # independent estimate: importance-weighted proposals, without rejection
    rng = np.random.default_rng(99)
    k = np.tile([E_PP, 0, 0, E_PP], (400_000, 1))
    p_minus, p_plus, density = axion._pair_proposal(400_000, E_PP, rng)
    weights = axion._pair_weights(k, p_minus, p_plus, density)
    expected, _ = np.histogram(p_plus[:, 0] - ME, edges, weights=weights)
    expected *= len(t_plus) / expected.sum()

    assert np.allclose(hist, expected, rtol=0.05)


def test_axioelectric_events():
    arr = axion.generate_axion_kinematics(
        100_000, process="axioelectric", energy=E_PP, seed=5
    )
    # one electron per event with the full axion energy
    assert ak.all(ak.num(arr.ekin) == 1)
    assert ak.all(arr.g4_pid == 11)
    assert np.allclose(ak.to_numpy(arr.ekin[:, 0]), E_PP)

    # isotropic, like the axions
    d = np.column_stack([ak.to_numpy(arr[c][:, 0]) for c in ("px", "py", "pz")])
    assert np.allclose(np.linalg.norm(d, axis=1), 1)
    assert np.all(np.abs(d.mean(axis=0)) < 0.01)


def test_primakoff_events():
    arr = axion.generate_axion_kinematics(
        100_000, process="primakoff", energy=E_PP, seed=8
    )
    # one photon per event with the full axion energy, isotropic
    assert ak.all(ak.num(arr.ekin) == 1)
    assert ak.all(arr.g4_pid == 22)
    assert np.allclose(ak.to_numpy(arr.ekin[:, 0]), E_PP)
    d = np.column_stack([ak.to_numpy(arr[c][:, 0]) for c in ("px", "py", "pz")])
    assert np.allclose(np.linalg.norm(d, axis=1), 1)
    assert np.all(np.abs(d.mean(axis=0)) < 0.01)


def test_screening_factor():
    # 1 - F(q): -> 1 for large momentum transfer, -> 0 for small, monotonic
    q = np.logspace(-2, 4, 200)  # keV
    s = axion._screening(q)
    assert s[-1] == pytest.approx(1, abs=1e-3)
    assert s[0] < 1e-4
    assert np.all(np.diff(s) > 0)
    # Thomas-Fermi scale of Ge: half screened around ~10 keV
    assert 3 < q[np.argmin(np.abs(s - 0.5))] < 30


def test_isotropic_axions():
    arr = axion.generate_axion_kinematics(
        100_000, process="compton", energy=E_PP, seed=6
    )
    p_tot = _momenta(arr, 0) + _momenta(arr, 1)
    d = p_tot / np.linalg.norm(p_tot, axis=1, keepdims=True)
    assert np.all(np.abs(d.mean(axis=0)) < 0.01)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"process": "bremsstrahlung", "energy": E_PP}, "process must be"),
        ({"process": "pair", "energy": 1000.0}, "above"),
        ({"process": "compton", "energy": 100.0, "mass": 200.0}, "mass < energy"),
        ({"process": "compton", "energy": 1.0, "eunit": "GeV"}, "eunits"),
    ],
)
def test_invalid(kwargs, match):
    with pytest.raises(ValueError, match=match):
        axion.generate_axion_kinematics(10, **kwargs)


def test_reproducible():
    a = axion.generate_axion_kinematics(50, process="pair", energy=E_PP, seed=7)
    b = axion.generate_axion_kinematics(50, process="pair", energy=E_PP, seed=7)
    assert ak.all(a.ekin == b.ekin)


def test_save_and_cli(tmp_path):
    out = tmp_path / "axion.lh5"
    cli(
        [
            "--seed",
            "1",
            "axion-kin",
            "-o",
            str(out),
            "-n",
            "1000",
            "--process",
            "axioelectric",
            "--energy",
            "5490",
        ]
    )
    tbl = lh5.read("vtx/kin", str(out))
    n_part = tbl["n_part"].nda
    assert np.sum(n_part) == len(n_part)
    assert np.sum(n_part > 0) == 1000
    assert tbl["ekin"].attrs["units"] == "keV"
