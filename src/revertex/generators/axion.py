from __future__ import annotations

import logging

import awkward as ak
import lh5
import numpy as np
from numpy.typing import ArrayLike, NDArray

from revertex.core import _get_chunks, convert_output_kin

log = logging.getLogger(__name__)

# electron rest mass in keV
_ME = 510.99895

# fine structure constant
_ALPHA = 1 / 137.035999084

# hbar * c in keV cm
_HBARC_KEV_CM = 1.973269804e-8

# PDG codes of the emitted particles
_PID_ELECTRON = 11
_PID_POSITRON = -11
_PID_GAMMA = 22

# default chunk size (events per write)
_CHUNK_SIZE = 1_000_000

# number of proposals per batch in the pair production rejection sampling
_PAIR_BATCH = 200_000

# fraction of isotropic proposals for the pair production lepton angles
_PAIR_ISO_FRAC = 0.2

# atomic screening for pair production in germanium: Thomas-Fermi radius and
# Moliere's three-Yukawa fit of the Thomas-Fermi form factor
_Z_GE = 32
_A_TF_KEV = 0.885 * 0.529177e-8 * _Z_GE ** (-1 / 3) / _HBARC_KEV_CM  # in 1/keV
_MOLIERE_ALPHA = (0.10, 0.55, 0.35)
_MOLIERE_BETA = (6.0, 1.2, 0.3)

# above this axion mass (keV) the pair production sampling becomes inefficient
_PAIR_MASS_WARN = 900.0

# Dirac matrices (Dirac representation) and metric for the pair production traces
_I2 = np.eye(2)
_Z2 = np.zeros((2, 2))
_PAULI = [
    np.array([[0, 1], [1, 0]]),
    np.array([[0, -1j], [1j, 0]]),
    np.array([[1, 0], [0, -1]]),
]
_GAMMA = np.array(
    [np.block([[_I2, _Z2], [_Z2, -_I2]])]
    + [np.block([[_Z2, s], [-s, _Z2]]) for s in _PAULI]
).astype(complex)
_GAMMA5 = np.block([[_Z2, _I2], [_I2, _Z2]]).astype(complex)
_I4 = np.eye(4, dtype=complex)
_METRIC = np.array([1.0, -1.0, -1.0, -1.0])

PROCESSES = ("axioelectric", "compton", "pair", "primakoff")


def save_axion_kinematics(
    n_gen: int,
    out_file: str,
    *,
    process: str,
    energy: float,
    mass: float = 0.0,
    seed: int | None = None,
    eunit: str = "keV",
) -> None:
    """Save the final states of axion interactions to a file.

    Parameters
    ----------
    n_gen
        number of interactions to generate.
    out_file
        path to the output file.
    process
        the axion interaction, one of `axioelectric`, `compton`, `pair` or
        `primakoff`.
    energy
        total energy of the axion.
    mass
        mass of the axion.
    seed
        random seed, for each chunk will be multiplied by 7.
    eunit
        the unit for `energy` and `mass`, default keV.
    """
    chunks = _get_chunks(n_gen, _CHUNK_SIZE)

    for idx, chunk in enumerate(chunks):
        # generate kinematics
        kin_ak = generate_axion_kinematics(
            chunk,
            process=process,
            energy=energy,
            mass=mass,
            seed=seed,
            eunit=eunit,
        )
        msg = f"Generated {process} axion kinematics {kin_ak}"
        log.debug(msg)

        # update the seed
        seed = seed * 7 if seed is not None else None

        # convert
        kin_lh5 = convert_output_kin(kin_ak, eunit="keV")

        # write
        mode = "of" if idx == 0 else "append"
        lh5.write(kin_lh5, "vtx/kin", out_file, wo_mode=mode)


def generate_axion_kinematics(
    size: int,
    *,
    process: str,
    energy: float,
    mass: float = 0.0,
    seed: int | None = None,
    eunit: str = "keV",
) -> ak.Array:
    """Generate the final-state particles of axion interactions in germanium.

    The supported processes are:

    - `axioelectric`: a + atom -> e- + ion, a single electron with the full axion
      energy, emitted along the axion direction. The binding energy released by
      the atomic relaxation is deposited within micrometres, so it is included in
      the electron energy.
    - `compton`: a + e- -> e- + gamma on an electron at rest, the photon angle is
      sampled from the tree-level cross section and the electron follows from
      energy-momentum conservation.
    - `pair`: a + Z -> Z + e- + e+ in the Coulomb field of the nucleus, the energies
      and directions of both leptons are sampled from the Born-level matrix
      element (Born approximation with atomic screening, no Coulomb corrections).
    - `primakoff`: a + Z -> Z + gamma, inverse Primakoff conversion in the
      Coulomb field of the nucleus (via the axion-photon coupling), a single
      photon with the full axion energy emitted along the axion direction (the
      nuclear recoil and the photon angle, of order m_a^2 / E^2 and
      1 / (E r_0), are negligible).

    The axions arrive isotropically and all particles of an event are emitted at
    time zero.

    Parameters
    ----------
    size
        number of interactions to generate.
    process
        the axion interaction, one of `axioelectric`, `compton`, `pair` or
        `primakoff`.
    energy
        total energy of the axion.
    mass
        mass of the axion.
    seed
        random seed.
    eunit
        the unit for `energy` and `mass`, default keV.

    Returns
    -------
    An awkward array with the kinematics of every event, energies in keV.
    """
    energy = _to_kev(energy, eunit)
    mass = _to_kev(mass, eunit)

    if process not in PROCESSES:
        msg = f"process must be one of {PROCESSES} not {process}"
        raise ValueError(msg)

    if mass < 0 or energy <= mass:
        msg = f"axion mass and energy must satisfy 0 <= mass < energy not mass = {mass} keV and energy = {energy} keV"
        raise ValueError(msg)

    rng = np.random.default_rng(seed=seed)

    # isotropic direction of flight of each axion
    axis = _sample_isotropic(size, seed=rng)

    if process == "axioelectric":
        return _generate_axioelectric(size, energy, axis)
    if process == "compton":
        return _generate_compton(size, energy, mass, axis, rng)
    if process == "primakoff":
        return _generate_primakoff(size, energy, axis)

    return _generate_pair(size, energy, mass, axis, rng)


def compton_cross_section(
    energy: float, mass: float = 0.0, eunit: str = "keV"
) -> float:
    """Total cross section of axio-Compton conversion (a + e- -> e- + gamma).

    Closed form of Zhitnitsky & Skovpen, Sov. J. Nucl. Phys. 29, 513 (1979), as
    given in Borexino, Phys. Rev. D 85, 092003 (2012), Eq. 8.

    Parameters
    ----------
    energy
        total energy of the axion.
    mass
        mass of the axion.
    eunit
        the unit for `energy` and `mass`, default keV.

    Returns
    -------
    The cross section per electron in cm^2, divided by the coupling `g_ae^2`.
    """
    energy = _to_kev(energy, eunit)
    mass = _to_kev(mass, eunit)

    p_a = np.sqrt(energy**2 - mass**2)
    y = 2 * _ME * energy + mass**2

    bracket = (
        2 * _ME**2 * (_ME + energy) * y / (_ME**2 + y) ** 2
        + 4
        * _ME
        * (mass**4 + 2 * mass**2 * _ME**2 - 4 * _ME**2 * energy**2)
        / (y * (_ME**2 + y))
        + (4 * _ME**2 * p_a**2 + mass**4)
        / (p_a * y)
        * np.log((_ME + energy + p_a) / (_ME + energy - p_a))
    )
    return float(_ALPHA / (8 * _ME**2 * p_a) * bracket * _HBARC_KEV_CM**2)


def compton_dsigma_dcos(energy: float, mass: float, cos_theta: ArrayLike) -> NDArray:
    r"""Differential cross section of axio-Compton conversion on an electron at rest.

    Based on the spin-averaged matrix element of the s- and u-channel diagrams for
    a pseudoscalar coupling, with :math:`S = s - m_e^2` and :math:`U = u - m_e^2`:

    .. math::

        |M|^2 = -4 - 2 (S/U + U/S) + 4 m_a^2 (1/S + 1/U)
                + 4 m_a^2 m_e^2 (1/S + 1/U)^2 - 4 m_a^4 / (S U)

    in units of :math:`e^2 g_{ae}^2`. Its integral is :func:`compton_cross_section`.

    Parameters
    ----------
    energy
        total energy of the axion in keV.
    mass
        mass of the axion in keV.
    cos_theta
        cosine of the angle between the photon and the axion.

    Returns
    -------
    `d sigma / d cos(theta)` in units of `g_ae^2 alpha / keV^2`.
    """
    cos_theta = np.asarray(cos_theta, dtype=float)

    p_a = np.sqrt(energy**2 - mass**2)
    e_gamma = _compton_photon_energy(energy, mass, cos_theta)

    s_red = mass**2 + 2 * energy * _ME
    u_red = -2 * _ME * e_gamma
    inv = 1 / s_red + 1 / u_red

    msq = (
        -4
        - 2 * (s_red / u_red + u_red / s_red)
        + 4 * mass**2 * inv
        + 4 * mass**2 * _ME**2 * inv**2
        - 4 * mass**4 / (s_red * u_red)
    )

    # e^2 = 4 pi alpha, and 2 pi from the azimuth
    return (
        8
        * np.pi**2
        * msq
        * e_gamma
        / (64 * np.pi**2 * _ME * p_a * (energy + _ME - p_a * cos_theta))
    )


def _to_kev(value: float, eunit: str) -> float:
    """Convert an energy to keV."""
    if eunit == "MeV":
        factor = 1000.0
    elif eunit == "keV":
        factor = 1.0
    elif eunit == "eV":
        factor = 1 / 1000.0
    else:
        msg = f"Only eunits keV, MeV or eV are supported not {eunit}"
        raise ValueError(msg)

    return float(value) * factor


def _compton_photon_energy(energy: float, mass: float, cos_theta: ArrayLike) -> NDArray:
    """Photon energy (keV) of axio-Compton conversion as a function of its angle."""
    p_a = np.sqrt(energy**2 - mass**2)
    return (mass**2 + 2 * energy * _ME) / (2 * (energy + _ME - p_a * cos_theta))


def _make_output(directions: NDArray, ekin: NDArray, g4_pid: NDArray) -> ak.Array:
    """Pack the particles into the kinematics array.

    The inputs have shape `(n_events, n_particles)`, with an extra last axis of
    length 3 for the directions.
    """
    out = {
        "px": directions[..., 0],
        "py": directions[..., 1],
        "pz": directions[..., 2],
        "ekin": ekin,
        "time": np.zeros_like(ekin),
        "g4_pid": g4_pid.astype(np.int64),
    }
    # variable-length lists, so every event gets its own n_part entry
    out = {key: ak.from_regular(val, axis=1) for key, val in out.items()}

    return ak.Array(out)


def _generate_axioelectric(size: int, energy: float, axis: NDArray) -> ak.Array:
    """Generate the electron of the axioelectric effect.

    The electron gets the full axion energy: `energy - E_b` from the absorption
    plus `E_b` from the relaxation of the vacancy, which is deposited locally.
    It is emitted along the axion direction.
    """
    directions = axis[:, None, :]
    ekin = np.full((size, 1), energy)
    pid = np.full((size, 1), _PID_ELECTRON)

    return _make_output(directions, ekin, pid)


def _generate_primakoff(size: int, energy: float, axis: NDArray) -> ak.Array:
    """Generate the photon of inverse Primakoff conversion.

    The photon gets the full axion energy and is emitted along the axion
    direction.
    """
    directions = axis[:, None, :]
    ekin = np.full((size, 1), energy)
    pid = np.full((size, 1), _PID_GAMMA)

    return _make_output(directions, ekin, pid)


def _generate_compton(
    size: int, energy: float, mass: float, axis: NDArray, rng: np.random.Generator
) -> ak.Array:
    """Generate the electron and photon of axio-Compton conversion."""
    p_a = np.sqrt(energy**2 - mass**2)

    # tabulate d sigma / d E_gamma on a uniform photon energy grid, E_gamma is
    # monotonic in cos(theta) and this grid resolves the narrow forward region
    e_grid = np.linspace(
        _compton_photon_energy(energy, mass, -1.0),
        _compton_photon_energy(energy, mass, 1.0),
        20_001,
    )
    y = mass**2 + 2 * energy * _ME
    cos_grid = np.clip((energy + _ME - y / (2 * e_grid)) / p_a, -1, 1)
    pdf = compton_dsigma_dcos(energy, mass, cos_grid) * y / (2 * p_a * e_grid**2)
    pdf = np.clip(pdf, 0, None)

    # inverse transform sampling from the tabulated cdf
    cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(e_grid))])
    e_gamma = np.interp(rng.random(size), cdf / cdf[-1], e_grid)

    cos_theta = np.clip((energy + _ME - y / (2 * e_gamma)) / p_a, -1, 1)
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = rng.uniform(low=0, high=2 * np.pi, size=size)

    # momenta in the frame with the axion along +z
    k_gamma = e_gamma[:, None] * np.column_stack(
        (sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta)
    )
    p_e = np.array([0.0, 0.0, p_a]) - k_gamma

    directions = np.stack(
        [
            _rotate_to_axis(_unit(p_e), axis),
            _rotate_to_axis(_unit(k_gamma), axis),
        ],
        axis=1,
    )
    ekin = np.column_stack([energy - e_gamma, e_gamma])
    pid = np.tile([_PID_ELECTRON, _PID_GAMMA], (size, 1))

    return _make_output(directions, ekin, pid)


def _generate_pair(
    size: int, energy: float, mass: float, axis: NDArray, rng: np.random.Generator
) -> ak.Array:
    """Generate the electron and positron of axion pair production.

    Uses rejection sampling from the matrix element with the proposal of
    :func:`_pair_proposal`, the envelope is set by a pilot batch.
    """
    if energy <= 2 * _ME:
        msg = f"pair production needs an axion energy above {2 * _ME:.1f} keV not {energy} keV"
        raise ValueError(msg)

    if mass > _PAIR_MASS_WARN:
        msg = (
            f"pair production with an axion mass of {mass:.0f} keV: close to 2 m_e the "
            "cross section is very peaked and the sampling becomes inefficient and "
            "less reliable, above 2 m_e the axion decays (a -> e+ e-)"
        )
        log.warning(msg)

    k = np.tile([energy, 0.0, 0.0, np.sqrt(energy**2 - mass**2)], (_PAIR_BATCH, 1))

    # pilot batch to set the envelope
    w_max = 1.2 * np.max(_pair_weights(k, *_pair_proposal(_PAIR_BATCH, energy, rng)))

    p_minus, p_plus = [], []
    n_acc, n_over = 0, 0

    while n_acc < size:
        pm, pp, density = _pair_proposal(_PAIR_BATCH, energy, rng)
        weights = _pair_weights(k, pm, pp, density)

        n_over += np.sum(weights > w_max)
        accept = rng.random(_PAIR_BATCH) * w_max < weights

        p_minus.append(pm[accept])
        p_plus.append(pp[accept])
        n_acc += np.sum(accept)

    if n_over > 0:
        msg = f"{n_over} pair production weights exceeded the envelope, this region is slightly under-sampled"
        log.warning(msg)

    p_minus = np.concatenate(p_minus)[:size]
    p_plus = np.concatenate(p_plus)[:size]

    directions = np.stack(
        [
            _rotate_to_axis(_unit(p_minus[:, 1:]), axis),
            _rotate_to_axis(_unit(p_plus[:, 1:]), axis),
        ],
        axis=1,
    )
    ekin = np.column_stack([p_minus[:, 0] - _ME, p_plus[:, 0] - _ME])
    pid = np.tile([_PID_ELECTRON, _PID_POSITRON], (size, 1))

    return _make_output(directions, ekin, pid)


def _pair_msq(k: NDArray, p_minus: NDArray, p_plus: NDArray) -> NDArray:
    """Matrix element of a + Z -> Z + e- + e+ in a static Coulomb field.

    Born approximation, summed over the lepton spins and with couplings
    stripped, with the screening of the nuclear charge by the atomic electrons
    (Thomas-Fermi form factor, see :func:`_screening`). The Coulomb photon
    couples via `gamma^0` with a propagator `1 / |q|^2`. With a vector instead of a pseudoscalar vertex the
    same code reproduces the Bethe-Heitler cross section.

    Parameters
    ----------
    k, p_minus, p_plus
        arrays with shape `(n, 4)` of the 4-momenta of the axion, electron and
        positron.
    """
    n = len(k)
    q = p_minus + p_plus - k
    q3_sq = np.sum(q[:, 1:] ** 2, axis=1)

    g0 = np.broadcast_to(_GAMMA[0], (n, 4, 4))
    g5 = np.broadcast_to(_GAMMA5, (n, 4, 4))

    # electron propagators of the two diagrams
    prop_1 = (_slash(p_minus - q) + _ME * _I4) / (
        _mdot(p_minus - q, p_minus - q) - _ME**2
    )[:, None, None]
    prop_2 = (_slash(p_minus - k) + _ME * _I4) / (
        _mdot(p_minus - k, p_minus - k) - _ME**2
    )[:, None, None]

    amp = g0 @ prop_1 @ g5 + g5 @ prop_2 @ g0
    amp_bar = _GAMMA[0] @ np.conj(np.swapaxes(amp, -1, -2)) @ _GAMMA[0]

    trace = np.trace(
        (_slash(p_minus) + _ME * _I4) @ amp @ (_slash(p_plus) - _ME * _I4) @ amp_bar,
        axis1=1,
        axis2=2,
    ).real

    # atomic screening of the nuclear charge: the Coulomb field is multiplied by
    # 1 - F(q), with the Thomas-Fermi form factor in Moliere's approximation
    return trace / q3_sq**2 * _screening(np.sqrt(q3_sq)) ** 2


def _screening(q: NDArray) -> NDArray:
    """Screening factor 1 - F(q) of the nuclear Coulomb field (q in keV).

    Thomas-Fermi atom in Moliere's approximation: the potential is a sum of
    Yukawa terms, so 1 - F(q) = sum_i alpha_i q^2 / (q^2 + (beta_i / a_TF)^2).
    It is ~1 for momentum transfers well above ~10 keV and suppresses the
    long-range (small q) part of the field.
    """
    q2 = q**2
    return sum(
        a * q2 / (q2 + (b / _A_TF_KEV) ** 2)
        for a, b in zip(_MOLIERE_ALPHA, _MOLIERE_BETA)
    )


def _pair_weights(
    k: NDArray, p_minus: NDArray, p_plus: NDArray, density: NDArray
) -> NDArray:
    """Ratio of the differential cross section to the proposal density."""
    p_p = np.linalg.norm(p_plus[:, 1:], axis=1)
    p_m = np.linalg.norm(p_minus[:, 1:], axis=1)

    # d sigma ~ |M|^2 p+ p- dE+ dOmega+ dOmega-
    return _pair_msq(k, p_minus, p_plus) * p_p * p_m / density


def _pair_proposal(
    size: int, energy: float, rng: np.random.Generator
) -> tuple[NDArray, NDArray, NDArray]:
    """Proposal for the electron and positron 4-momenta, axion along +z.

    The positron energy is uniform, both polar angles follow
    :func:`_sample_forward_angle`.

    Returns
    -------
    the electron and positron 4-momenta and the proposal density in
    `(E+, cos+, phi+, cos-, phi-)`.
    """
    t_tot = energy - 2 * _ME
    e_plus = _ME + rng.uniform(low=0, high=t_tot, size=size)
    e_minus = energy - e_plus

    cos_p, density_p = _sample_forward_angle(e_plus, rng)
    cos_m, density_m = _sample_forward_angle(e_minus, rng)

    p_plus = _four_momenta(e_plus, cos_p, rng.uniform(0, 2 * np.pi, size))
    p_minus = _four_momenta(e_minus, cos_m, rng.uniform(0, 2 * np.pi, size))

    return p_minus, p_plus, density_p * density_m / (t_tot * (2 * np.pi) ** 2)


def _sample_forward_angle(
    e_tot: NDArray, rng: np.random.Generator
) -> tuple[NDArray, NDArray]:
    r"""Sample the polar angle of a lepton from pair production.

    A mixture of a forward peak

    .. math::

        P(\theta) \propto \frac{\theta \theta_0^2}{(\theta^2 + \theta_0^2)^2}

    with :math:`\theta_0 = m_e / E` and an isotropic part covering the full sphere.

    Returns
    -------
    the cosine of the angles and the proposal density in `cos(theta)`.
    """
    size = len(e_tot)
    theta_0 = _ME / e_tot

    # inverse transform of the forward peak, replaced by isotropic samples
    r = rng.random(size)
    theta = theta_0 * np.sqrt(r / (1 - r))
    iso = (rng.random(size) < _PAIR_ISO_FRAC) | (theta > np.pi)

    cos_theta = np.cos(theta)
    cos_theta[iso] = rng.uniform(low=-1, high=1, size=np.sum(iso))
    theta = np.arccos(cos_theta)

    # density in cos(theta) of the mixture
    peak = 2 * theta * theta_0**2 / (theta**2 + theta_0**2) ** 2
    peak /= np.maximum(np.sin(theta), 1e-300)
    peak_beyond_pi = theta_0**2 / (np.pi**2 + theta_0**2)

    density = (1 - _PAIR_ISO_FRAC) * peak + 0.5 * (
        _PAIR_ISO_FRAC + (1 - _PAIR_ISO_FRAC) * peak_beyond_pi
    )
    return cos_theta, density


def _four_momenta(e_tot: NDArray, cos_theta: NDArray, phi: NDArray) -> NDArray:
    """4-momenta of electrons from their total energy and direction."""
    p = np.sqrt(np.clip(e_tot**2 - _ME**2, 0, None))
    sin_theta = np.sqrt(np.clip(1 - cos_theta**2, 0, None))

    return np.column_stack(
        (e_tot, p * sin_theta * np.cos(phi), p * sin_theta * np.sin(phi), p * cos_theta)
    )


def _slash(p: NDArray) -> NDArray:
    """Contract 4-vectors of shape `(n, 4)` with the Dirac matrices."""
    return np.einsum("na,aij->nij", p * _METRIC, _GAMMA)


def _mdot(a: NDArray, b: NDArray) -> NDArray:
    """Minkowski product of 4-vectors."""
    return np.sum(a * b * _METRIC, axis=-1)


def _unit(v: NDArray) -> NDArray:
    """Normalize the rows of an array of vectors."""
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _sample_isotropic(
    size: int, seed: int | np.random.Generator | None = None
) -> NDArray:
    """Generate isotropically distributed unit vectors with shape `(size, 3)`."""
    rng = np.random.default_rng(seed=seed)

    cos_theta = rng.uniform(low=-1, high=1, size=size)
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = rng.uniform(low=0, high=2 * np.pi, size=size)

    return np.column_stack(
        (sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta)
    )


def _rotate_to_axis(vectors: NDArray, axis: NDArray) -> NDArray:
    """Rotate vectors given in a frame with `axis` along +z to the global frame.

    Parameters
    ----------
    vectors
        array with shape `(n, 3)` of vectors in the local frame.
    axis
        array with shape `(n, 3)` of unit vectors, the local +z axis of each row.
    """
    # orthonormal basis (e1, e2, axis) for every row, with a helper vector that is
    # never (nearly) parallel to the axis
    helper = np.tile([1.0, 0.0, 0.0], (len(axis), 1))
    helper[np.abs(axis[:, 0]) > 0.9] = [0.0, 1.0, 0.0]

    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(axis, e1)

    return vectors[:, :1] * e1 + vectors[:, 1:2] * e2 + vectors[:, 2:3] * axis