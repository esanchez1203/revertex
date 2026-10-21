# Axion interaction generator

Axions of a fixed energy, such as the 5.49 MeV solar axions from the p + d →
³He + a reaction, can be detected in germanium through several interaction
channels. Each channel produces a characteristic set of particles at the
interaction point. This generator samples those particles from the physics of
each process, with energies and directions consistent with energy and momentum
conservation, and writes them in the
{ref}`remage kinematics format <remage:manual-input>`.

| Process        | Reaction            | Particles per event | Coupling |
| -------------- | ------------------- | ------------------- | -------- |
| `axioelectric` | a + atom → e⁻ + ion | 1 electron          | g_ae     |
| `compton`      | a + e⁻ → e⁻ + γ     | electron + photon   | g_ae     |
| `pair`         | a + Z → Z + e⁻ + e⁺ | electron + positron | g_ae     |
| `primakoff`    | a + Z → Z + γ       | 1 photon            | g_aγ     |

The inputs are the total energy $E_a$ and the mass $m_a$ of the axion, with
momentum $p_a = \sqrt{E_a^2 - m_a^2}$. The axions arrive isotropically: every
event gets a random direction of flight and the outgoing particles are oriented
relative to it. For solar axions this approximates the changing position of the
Sun over a long exposure. All particles of an event are emitted at time zero.

The generator describes a single interaction of a given type. The rate of each
channel (flux, cross section, couplings) is not part of the event generation and
is applied in the analysis, see [Normalization](#normalization).

Natural units ($\hbar = c = 1$) are used below; $m_e$ is the electron mass,
$\alpha$ the fine-structure constant and $Z = 32$ for germanium.

## Usage

### Command line

```console
> revertex --seed 42 axion-kin \
    --process compton \
    --energy 5490 \
    --mass 0 \
    --n-events 1000000 \
    --out-file compton.lh5
```

| option             | default | description                                       |
| ------------------ | ------- | ------------------------------------------------- |
| `--process`, `-p`  | —       | `axioelectric`, `compton`, `pair` or `primakoff`  |
| `--energy`, `-E`   | —       | total energy of the axion                         |
| `--mass`, `-m`     | 0       | mass of the axion                                 |
| `--eunit`, `-e`    | `keV`   | unit of the energy and mass: `keV`, `MeV` or `eV` |
| `--n-events`, `-n` | —       | number of events                                  |
| `--out-file`, `-o` | —       | output LH5 file                                   |

The seed is the global option `--seed`, given before `axion-kin`. Full option
reference:

```console
> revertex axion-kin --help
```

### Python

```python
from revertex.generators import axion

# write to a file, in chunks
axion.save_axion_kinematics(
    1_000_000, "pair.lh5", process="pair", energy=5490, mass=0, seed=42
)

# or get the events as an awkward array
events = axion.generate_axion_kinematics(10_000, process="compton", energy=5490, seed=1)
```

## Physics and energy distributions

### Axioelectric effect

The axion is absorbed by a bound electron, $a + \mathrm{Ge} \to e^- +
\mathrm{Ge}^+$ \[[1](#ref1),[6](#ref6),[7](#ref7)\]. This is a two-body final
state with a recoiling ion, so the electron energy is fixed:

$$
T_e = E_a - E_b ,
$$

where $E_b \le 11.1$ keV is the binding energy of the shell (K shell of Ge). The
vacancy is filled by X-rays and Auger electrons of total energy $E_b$, which are
absorbed within micrometres of the vertex in germanium. The generator therefore
emits a single electron with

$$
T_e = E_a ,
$$

along the axion direction. At MeV energies the angular distribution of the
electron is strongly forward-peaked (opening angle ~ $m_e/E_a$), so this is a
good approximation.

### Compton conversion

The axion converts on an electron, $a + e^- \to e^- + \gamma$
\[[1](#ref1),[2](#ref2),[3](#ref3)\]. For an electron at rest the photon energy
is fixed by its angle $\theta$ with respect to the axion direction (two-body
kinematics):

$$
E_\gamma(\theta) = \frac{m_a^2 + 2 m_e E_a}{2\,(E_a + m_e - p_a \cos\theta)},
\qquad T_e = E_a - E_\gamma .
$$

The angular distribution follows from the tree-level matrix element of the s-
and u-channel diagrams for a pseudoscalar coupling $g_{ae}$, summed over spins.
With $S = s - m_e^2 = m_a^2 + 2 m_e E_a$ and $U = u - m_e^2 = -2 m_e E_\gamma$:

$$
\overline{|\mathcal{M}|^2} = e^2 g_{ae}^2 \left[ -4 - 2\left(\frac{S}{U} +
\frac{U}{S}\right) + 4 m_a^2 \left(\frac{1}{S} + \frac{1}{U}\right) + 4 m_a^2
m_e^2 \left(\frac{1}{S} + \frac{1}{U}\right)^2 - \frac{4 m_a^4}{S U} \right],
$$

$$
\frac{d\sigma}{d\cos\theta} = \frac{\overline{|\mathcal{M}|^2}}{32 \pi\, m_e
p_a} \, \frac{E_\gamma}{E_a + m_e - p_a\cos\theta} ,
$$

implemented in {func}`.axion.compton_dsigma_dcos`. The photon energy spectrum is

$$
\frac{d\sigma}{dE_\gamma} = \frac{d\sigma}{d\cos\theta}\,
\frac{m_a^2 + 2 m_e E_a}{2 p_a E_\gamma^2},
\qquad E_\gamma \in \left[E_\gamma(\pi),\, E_\gamma(0)\right],
$$

which the generator tabulates on a grid of 20 000 points and samples by inverse
transform. The electron momentum is $\vec p_e = \vec p_a - \vec k_\gamma$ and
the azimuth is uniform. Integrating gives the closed form of Zhitnitsky and
Skovpen \[[2](#ref2)\], as quoted by Borexino \[[3](#ref3), eq. 8\]:

$$
\sigma_{C} = \frac{\alpha g_{ae}^2}{8 m_e^2 p_a} \left[ \frac{2 m_e^2 (m_e +
E_a)\, y}{(m_e^2 + y)^2} + \frac{4 m_e (m_a^4 + 2 m_a^2 m_e^2 - 4 m_e^2
E_a^2)}{y\,(m_e^2 + y)} + \frac{4 m_e^2 p_a^2 + m_a^4}{p_a\, y} \ln\frac{m_e +
E_a + p_a}{m_e + E_a - p_a} \right],
$$

with $y = 2 m_e E_a + m_a^2$, available as {func}`.axion.compton_cross_section`.
For $E_a = 5.49$ MeV and $m_a = 0$ it is $4.3 \times 10^{-25}\,\mathrm{cm}^2
\times g_{ae}^2$ per electron. The binding of the electron (≤ 11 keV) is
neglected.

### Pair production

The axion converts into an electron-positron pair in the Coulomb field of the
nucleus, $a + Z \to Z + e^- + e^+$. This is the axion analogue of the
Bethe-Heitler process \[[8](#ref8)\], with the photon vertex $\gamma^\mu$
replaced by the pseudoscalar vertex $g_{ae}\gamma_5$. In Born approximation the
nucleus is a static source and the nuclear recoil energy is neglected, so

$$
T_- + T_+ = E_a - 2 m_e ,
$$

with momentum transfer $\vec q = \vec p_- + \vec p_+ - \vec p_a$ absorbed by the
nucleus. The amplitude of the two diagrams (axion attached to the electron or to
the positron line) is

$$
\Gamma = \gamma^0 \frac{\not p_- - \not q + m_e}{(p_- - q)^2 - m_e^2}\gamma_5 +
\gamma_5 \frac{\not p_- - \not k + m_e}{(p_- - k)^2 - m_e^2} \gamma^0 ,
$$

where $k$ is the axion 4-momentum and $\gamma^0$ the coupling to the static
Coulomb field. Summed over the lepton spins:

$$
\overline{|\mathcal{M}|^2} \propto
\frac{\mathrm{Tr}\left[(\not p_- + m_e)\,\Gamma\,(\not p_+ - m_e)\,\bar\Gamma\right]}{|\vec q|^4}
\,\left[1 - F(|\vec q|)\right]^2 ,
\qquad \bar\Gamma = \gamma^0 \Gamma^\dagger \gamma^0 ,
$$

and the fully differential cross section is

$$
d\sigma \propto \overline{|\mathcal{M}|^2}\, |\vec p_+|\, |\vec p_-|\;
dE_+\, d\Omega_+\, d\Omega_- .
$$

The traces are evaluated numerically with explicit Dirac matrices (`_pair_msq`).
The generator samples $(E_+, \Omega_+, \Omega_-)$ jointly by rejection sampling:
$E_+$ is proposed uniformly in $[m_e, E_a - m_e]$ and the polar angles from a
forward peak $P(\theta) \propto \theta\theta_0^2 / (\theta^2 + \theta_0^2)^2$,
$\theta_0 = m_e/E$, mixed with 20 % isotropic proposals. The positron energy
spectrum $d\sigma/dT_+$, the angular distributions and the correlations between
the two leptons therefore follow the matrix element; they are not parametrized
separately. The normalization of $d\sigma$ is not needed for the sampling; the
total cross section is obtained by numerical integration of the same expression
(see [Normalization](#normalization)).

**Screening.** The atomic electrons screen the nuclear charge at distances above
the Thomas-Fermi radius $a_{TF} = 0.885\, a_0 Z^{-1/3}$. The Coulomb field is
multiplied by $1 - F(q)$, with the Thomas-Fermi form factor in Molière's
approximation \[[9](#ref9),[10](#ref10)\]:

$$
1 - F(q) = \sum_{i=1}^{3} \alpha_i \frac{q^2}{q^2 + (\beta_i / a_{TF})^2},
\qquad \alpha_i = (0.10, 0.55, 0.35), \quad \beta_i = (6.0, 1.2, 0.3),
$$

implemented in `_screening`. For Ge, $\hbar c / a_{TF} \approx 13$ keV:
screening only affects momentum transfers below ~10 keV. It lowers the rate by
about 1 % at $m_a = 0$, but becomes important for $m_a$ close to $2 m_e$, where
the minimum momentum transfer becomes small.

**Checks.** With the vertex $\gamma_5$ replaced by $\gamma^\mu$ (a photon), the
same code reproduces the Bethe-Heitler pair cross section in germanium as
tabulated by Maximon \[[11](#ref11)\], and screening lowers it by 0.7 % at 5.49
MeV, in line with the known size of the effect \[[10](#ref10)\].

**Not included:** the Coulomb correction beyond the Born approximation ($Z\alpha
= 0.23$ for Ge, a few-percent effect \[[10](#ref10)\]) and pair production on
the atomic electrons (triplet production, about $1/Z$ of the nuclear rate).

:::{warning}

For axion masses approaching $2 m_e = 1022$ keV the cross section is dominated
by very small momentum transfers and is strongly peaked. Above ~900 keV the
sampling becomes inefficient and less reliable, and a warning is printed. Above
$2 m_e$ the axion can decay ($a \to e^+e^-$) and pair production in the nuclear
field is no longer the relevant process.

:::

### Inverse Primakoff conversion

The axion converts into a photon in the Coulomb field of the nucleus through the
axion-photon coupling, $a + Z \to Z + \gamma$
\[[3](#ref3),[4](#ref4),[5](#ref5),[12](#ref12)\]. The nuclear recoil energy is
~ $q^2/2M \lesssim$ eV and the typical photon angle is of order $q/E_a$, a few
mrad, so

$$
E_\gamma = E_a ,
$$

and the generator emits a single photon with the full axion energy along the
axion direction.

## Output format

The generator writes one table to the LH5 file, with one row per particle:

`vtx/kin` : Kinematics of the particles at the interaction point.

| field    | unit | description                                                       |
| -------- | ---- | ----------------------------------------------------------------- |
| `ekin`   | keV  | kinetic energy                                                    |
| `px`     | —    | x-component of the unit direction                                 |
| `py`     | —    | y-component of the unit direction                                 |
| `pz`     | —    | z-component of the unit direction                                 |
| `time`   | ns   | always 0                                                          |
| `g4_pid` | —    | PDG code: 11 (e⁻), −11 (e⁺) or 22 (γ)                             |
| `n_part` | —    | particles in the event (1 or 2) on its first row, 0 on the others |

| process        | particles (in this order) |
| -------------- | ------------------------- |
| `axioelectric` | e⁻                        |
| `compton`      | e⁻, γ                     |
| `pair`         | e⁻, e⁺                    |
| `primakoff`    | γ                         |

No positions are written: the vertices are sampled by remage (see below). No
rate is written either: the normalization depends on the flux and couplings, see
[Normalization](#normalization).

## Using the output in remage

The file contains kinematics only, so the interaction point is chosen by remage,
for example uniformly in the germanium detectors:

```text
/RMG/Generator/Confine Volume
/RMG/Generator/Confinement/Physical/AddVolume <detector>
/RMG/Generator/Select FromFile
/RMG/Generator/FromFile/FileName compton.lh5
```

In [legend-simflow](https://github.com/legend-exp/legend-simflow) the generator
is used as a `vtx` tier with `product: kinematics`:

```yaml
# tier/vtx/<experiment>/simconfig.yaml
axion_compton:
  command: >-
    revertex axion-kin --process compton --energy 5490 --mass 0 --n-events
    {N_EVENTS} --out-file {OUTPUT_FILE}
  product: kinematics

# tier/stp/<experiment>/simconfig.yaml
axion_compton_hpge:
  template: $_/templates/default.mac
  generator: ~vertices:axion_compton
  confinement: ~volumes.bulk:^[BCPV][0-9][0-9].*
  primaries_per_job: 500000
  number_of_jobs: 20
```

## Validation

The unit tests (`tests/test_axion.py`) check:

- energy conservation for every process ($T_- + T_+ = E_a - 2 m_e$ for pairs);
- momentum conservation $|\vec p_e + \vec k_\gamma| = p_a$ and the photon
  spectrum against $d\sigma/dE_\gamma$ for Compton conversion, also for massive
  axions;
- the integral of $d\sigma/d\cos\theta$ against $\sigma_C$ and the Borexino
  value $4.3 \times 10^{-25}\,\mathrm{cm}^2 \times g_{ae}^2$;
- the pair energy sharing against the matrix element;
- the screening factor ($\to 1$ at large momentum transfer, $\to 0$ at small);
- the single full-energy photon of Primakoff events;
- isotropy, reproducibility with a seed, the command line and the LH5 format.

## API reference

```{eval-rst}
.. autofunction:: revertex.generators.axion.generate_axion_kinematics
   :no-index:

.. autofunction:: revertex.generators.axion.save_axion_kinematics
   :no-index:

.. autofunction:: revertex.generators.axion.compton_dsigma_dcos
   :no-index:

.. autofunction:: revertex.generators.axion.compton_cross_section
   :no-index:
```

## References

- [1] F. T. Avignone III et al., _Laboratory limits on solar axions from an
  ultralow-background germanium spectrometer_, Phys. Rev. D 37, 618 (1988).

- [2] A. R. Zhitnitsky and Yu. I. Skovpen, _Problem of axion detection_, Sov. J.
  Nucl. Phys. 29, 513 (1979).

- [3] G. Bellini et al. (Borexino), _Search for solar axions produced in the
  p(d,³He)A reaction with Borexino detector_, Phys. Rev. D 85, 092003 (2012),
  [arXiv:1203.6258](https://arxiv.org/abs/1203.6258).

- [4] W. Buchmüller and F. Hoogeveen, _Coherent production of light scalar or
  pseudoscalar particles in Bragg scattering_, Phys. Lett. B 237, 278 (1990).

- [5] H. Primakoff, _Photo-production of neutral mesons in nuclear electric
  fields and the mean life of the neutral meson_, Phys. Rev. 81, 899 (1951).

- [6] M. Pospelov, A. Ritz and M. Voloshin, _Bosonic super-WIMPs as keV-scale
  dark matter_, Phys. Rev. D 78, 115012 (2008).

- [7] A. Derevianko et al., _Axio-electric effect_, Phys. Rev. D 82, 065006
  (2010).

- [8] H. Bethe and W. Heitler, _On the stopping of fast particles and on the
  creation of positive electrons_, Proc. R. Soc. London A 146, 83 (1934).

- [9] G. Molière, _Theorie der Streuung schneller geladener Teilchen I_, Z.
  Naturforsch. A 2, 133 (1947).

- [10] Y.-S. Tsai, _Pair production and bremsstrahlung of charged leptons_, Rev.
  Mod. Phys. 46, 815 (1974).

- [11] L. C. Maximon, _Simple analytic representations of the total cross
  section for electron-positron pair production_, J. Res. Natl. Bur. Stand. 72B,
  79 (1968).

- [12] M. Agostini et al. (Borexino), _Search for solar axions with Borexino_,
  [arXiv:2504.19135](https://arxiv.org/abs/2504.19135) (2025).
