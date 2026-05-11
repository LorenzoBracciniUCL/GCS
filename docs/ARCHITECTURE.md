# QCAT / GCS — Architecture & Conventions

> Reference paper: **Braccini, Bose, Serafini, *Superpositions of Quantum Gaussian Processes*, arXiv:2510.01156** ([HTML](https://arxiv.org/html/2510.01156)). All equation / table / section numbers below refer to this paper.

This document is the canonical map between the paper's formalism and the code. It is written so a future Claude session (or any reader who already knows the physics from the paper) can locate any object in the codebase and verify it line-by-line against the paper.

---

## 1. What the library computes

A **Gaussian-Branched Cat State (GCS)** is a joint state of `N` qubits and `n` bosonic modes that can be written as a superposition of Gaussian processes indexed by the qubit branch labels `J, K ∈ {±1}^N` (Sec. III, Eq. 21). Every branch is fully characterized by four objects evolved in time:

| Object | Shape | Meaning | Paper |
|---|---|---|---|
| `σ_JK(t)` | `(2n × 2n)` complex | branched covariance matrix | Eq. 25, Tab. 1/3 |
| `r_JK(t)` | `(2n × 1)` complex | branched first moment | Eq. 25, Tab. 1/3 |
| `C_JK(t)` | scalar real | contrast / decoherence rate | Eq. 25 (= −Re r⁽⁰⁾_JK) |
| `φ_JK(t)` | scalar real | branch relative phase | Eq. 25 (= Im r⁽⁰⁾_JK) |

From these the **qubit reduced density matrix** is reconstructed as
`ϱ^q_JK(t) = exp(-C_JK(t) + i·φ_JK(t)) · ϱ^q_JK(0)` (Eq. 25 / Eq. 38).

Hermitian-conjugate symmetry on the index pair (Sec. III, after Eq. 25):
`σ_KJ = σ*_JK,  r_KJ = r*_JK,  C_KJ = C_JK,  φ_KJ = -φ_JK`.

Only the upper triangle `j ≤ k` is computed; the lower triangle is set by conjugation.

---

## 2. Conventions

### 2.1 Symplectic form (paper Eq. 1–2)

`Ω_n = ⊕_{i=1}^n Ω₁`, `Ω₁ = [[0, 1], [-1, 0]]`. Canonical ordering `r̂ = (x₁, p₁, x₂, p₂, …, x_n, p_n)ᵀ`. `[r̂, r̂ᵀ] = i Ω_n`. `Ω⁻¹ = Ωᵀ = -Ω`.

Implemented in `src/Gaussian.py:Omega_N`. Ladder-basis transform `α = U r` (Eq. 3) in `src/Gaussian.py:U_N`.

### 2.2 Branch labels

`J ∈ {±1}^N` (Sec. V, Eq. 39). Encoded as an integer `j ∈ {0, …, 2^N - 1}` via the binary-to-±1 map `2·bit - 1`, implemented in `src/QIT_Functions.py:Extract_Qubit_Labels_Array`. So `j = 0 → (-1,…,-1)`, `j = 2^N - 1 → (+1,…,+1)`.

### 2.3 Hamiltonian structure (paper Eq. 28–30)

`H_J = H_m + Σ_i J_i · H_q,i`, `r_J = r_m + Σ_i J_i · r_q,i`. The library carries three named cases:

| Case | `H_q,i` | `r_q,i` | Module |
|---|---|---|---|
| **Gaussian** | 0 | 0 | `src/Gaussian.py` (no branch label) |
| **Force** | 0 | ≠ 0 | `src/Unitary_Operator_Force.py`, `src/Open_Operator_Force.py` |
| **General** | ≠ 0 | ≠ 0 | `src/Unitary_Operator_Gaussian.py`, `src/Open_Operator_Gaussian.py` |

> ⚠️ **Naming trap**: the file `Unitary_Operator_Gaussian.py` implements the **General** Hamiltonian case, *not* the pure-Gaussian one. The pure-Gaussian (single-mode) Hamiltonian path is in `src/Gaussian.py`. See BUGS.md §5 for the rename suggestion.

### 2.4 Decoherence (paper Eq. 43–46)

The master equation is `∂ρ/∂τ = i[ρ, H_g] + i[ρ, dᵀr̂] + ℒ_r̂(ρ) + ℒ_σ_z(ρ)`. The bosonic dissipator is parameterized by a `2n × 2n` complex matrix `B`, decomposed as

`B = ½ Ωᵀ D Ω − i E`  (Eq. 46)

with `E` antisymmetric (drift) and `D` symmetric positive (diffusion). The mapping `B → (E, D)` is in `src/Gaussian.py:Get_Decoherence_Rates`:
- `D = 2 Ω · Re(B) · Ωᵀ`
- `E = − Im(B)`

`Decoherence(N, n, basis, B)` in `src/Objects.py` accepts `basis ∈ {'Canonical', 'Ladder'}`; ladder-basis input is rotated via `U_N` before decomposition.

### 2.5 Qubit dephasing (paper Tab. 3)

Single-qubit dephasing rate Γ_z enters the r⁽⁰⁾_JK ODE as a `Γ_z · (JK − 1) / 2` term. **This term is not currently implemented anywhere in the code** — `H_q_0_array` is the unitary `H^0_q` term only.

---

## 3. Data shapes

Stored on `Quantum_State` after dynamics:

```
sigma_JK_t : (T, 2^N, 2^N, 2n, 2n)        complex
r_JK_t     : (T, 2^N, 2^N, 2n, 1)         complex
C_JK_t     : (T, 2^N, 2^N)                real
phi_JK_t   : (T, 2^N, 2^N)                real
r_JK_0_t   : (T, 2^N, 2^N)                complex   (= -C + iφ)
rho_q_t    : (T, 2^N, 2^N)                complex   QRDM time series
```

In the Gaussian-only (no qubit) code path, `r_t : (T, 2n, 1)` and `sigma_t : (T, 2n, 2n)` live instead.

---

## 4. Module map (paper → code)

| Module | Role | Paper anchor |
|---|---|---|
| `src/Gaussian.py` | Single-Gaussian ODEs, Ω_N, U_N, S_m = exp(ΩHτ), open-dynamics utilities, `Get_Decoherence_Rates`, log-negativity (2 modes) | Eq. 1–3, Eq. 43–46 |
| `src/QIT_Functions.py` | `Extract_Qubit_Labels_Array` (index ↔ ±1 vector), `Compute_r_tilde` | Sec. V |
| `src/Symplectic_Known.py` | Closed-form `S_m(τ)` for QHO_1, QHO_N (broken — see BUGS.md), 2-coupled-QHO with `xx` coupling | Tab. 2 |
| `src/Unitary_Operator_Force.py` | **Force** case, closed-form / numerical. Table 2 integral forms for σ, r, C, φ | Tab. 2 |
| `src/Unitary_Operator_Gaussian.py` | **General** case, three nested ODEs (σ ⟶ r ⟶ r⁽⁰⁾) | Tab. 1 |
| `src/Open_Operator_Force.py` | Force + Lindblad. `Dynamics_Symetric_Numerical` (E = 0 path) is the only working entry point | Tab. 2 with D |
| `src/Open_Operator_Gaussian.py` | General + Lindblad. Adds 2E to drift, +D to σ̇, but **misses Ω·d driving and Γ_z dephasing** | Tab. 3 |
| `src/Unitary_Operator_Time_Depentent.py` | Time-dependent unitary General — currently sign-flipped vs constant version (BUGS.md §1) | Tab. 1 generalized to H(t) |
| `src/Open_Operator_Time_Depentent.py` | Misnamed: actually implements time-dependent **unitary** ODEs (no E/D), with sign and coefficient errors | — |
| `src/Measurements.py` | Qubit projective measurement, generaldyne / homodyne POVM (Eq. 51), PPT log-negativity. Hardcoded N = 2 in `Negativity_*` | Eq. 51, Sec. VII |
| `src/Wigner_Functions.py` | Wigner functions: diagonal sum, PMS (post-measurement), single Gaussian | Sec. III–IV |
| `src/Plots_Functions.py` | All matplotlib code: phase-space, Wigner 2-D/4-times/fringes, 3-D vectors/ellipsoids, animations | — |
| `src/Objects.py` | OOP front: `Hamiltonian`, `Symplectic`, `Decoherence`, `Quantum_State`. Dispatches to one of the operator-valued modules based on `Hamiltonian.type` | — |

---

## 5. ODEs — exact line references

### 5.1 Unitary General σ̇_JK (paper Tab. 1)

`src/Unitary_Operator_Gaussian.py:53`
```
σ̇ = ½[Ω(H_J + H_K)σ − σ(H_J + H_K)Ω] − i[σ(H_J − H_K)σ + Ω(H_J − H_K)Ω]
```
✓ matches Tab. 1.

### 5.2 Unitary General ṙ_JK

`src/Unitary_Operator_Gaussian.py:83`
```
ṙ = ½ Ω(H_J + H_K) r − ½i σ(H_J − H_K) r − ½ Ω(r_J + r_K) + ½i σ(r_J − r_K)
```
✓ matches Tab. 1.

### 5.3 Unitary General ṙ⁽⁰⁾_JK

`src/Unitary_Operator_Gaussian.py:112`
```
ṙ⁽⁰⁾ = i·[ -½ rᵀ(H_J − H_K) r + (r_J − r_K)ᵀ r − ¼ Tr((H_J − H_K) σ) − ½ (J − K)ᵀ H^0_q ]
```
✓ matches Tab. 1.

### 5.4 Open General σ̇_JK (paper Tab. 3)

`src/Open_Operator_Gaussian.py:53`
```
σ̇ = ½[Ω(H_J + H_K + 2E) σ − σ(H_J + H_K + 2Eᵀ) Ω] + D − i[σ(H_J − H_K)σ + Ω(H_J − H_K)Ω]
```
✓ matches Tab. 3.

### 5.5 Open General ṙ_JK

`src/Open_Operator_Gaussian.py:83`
```
ṙ = ½ Ω(H_J + H_K + 2E) r − ½i σ(H_J − H_K) r − ½ Ω(r_J + r_K) + ½i σ(r_J − r_K)
```
⚠ Missing the Ω·d driving term that appears in Tab. 3 as `½ Ω(2d − r_J − r_K)`. Acceptable only when d = 0.

### 5.6 Force-case closed forms (paper Tab. 2)

C_JK (real part of r⁽⁰⁾) — `src/Unitary_Operator_Force.py:40`:
```
C_JK = ¼ (Δr̃)ᵀ Ωᵀ σ(τ) Ω Δr̃     where  Δr̃ = (r̃_J(τ) − r̃_J) − (r̃_K(τ) − r̃_K)
```
✓ matches Tab. 2 verbatim.

φ_JK — `src/Unitary_Operator_Force.py:45`:
```
φ = −(r̃_J − r̃_K)ᵀ Ω (r̃_0(τ) − r̃_0)
    + ½ (r̃_J − r̃_K)ᵀ Ω [(r̃_J(τ) − r̃_J) + (r̃_K(τ) − r̃_K)]
    + (τ/2) [ (r̃_J − r̃_K)ᵀ H_m (r̃_J + r̃_K) − H^0_qᵀ (J − K) ]
```
✓ matches Tab. 2. (The trailing comment `#### Meno a Caso` at line 49 is a TODO marker, not a code issue.)

### 5.7 Hermitian-conjugate fill

`src/Unitary_Operator_Gaussian.py:170–175` and analogous blocks in every operator-valued module. Sets `σ_KJ = σ*_JK`, `r_KJ = r*_JK`, `C_KJ = C_JK`, `φ_KJ = −φ_JK` (paper Sec. III).

---

## 6. Computational flow

A typical user pipeline is encoded in `Objects.Quantum_State`:

```
1. Hamiltonian(N, n).Initialize_Constant_Hamiltonians(H_array, r_array, H_q_0_array)
       — `H_array[0] = H_m`, `H_array[i+1] = H_q,i`; same shape for r_array.
       — `.type[0]` = 'Constant'|'Time'; `.type[1]` = 'Gaussian'|'Force'|'General'.

2. (optional) Symplectic(N, n).Initialize_Constant_Symplectic(name=…)         # for Analytical paths
   (optional) Decoherence(N, n, basis, B)                                     # for Open paths

3. Quantum_State(N, n).Initialize_Gaussian_State(r_0, sigma_0, rho_q_0)

4. .Unitary_Dynamics_{Numerical|Analytical}(H[, Symplectic], t_array)
   .Open_Dynamics_{Numerical|Analytical}(H, Decoherence, t_array)

5. Reads `sigma_JK_t`, `r_JK_t`, `C_JK_t`, `phi_JK_t`, `rho_q_t` off `self`.

6. .Qubit_Ideal_Measurament(...), .Mode_Measurament(...), .Negativity_Qubit(...)
   .Plot_Wigner_Function_{Diag|PMS|Gauss}(...), .Plot_Vectors_*(...), .Animate_*(...)
```

Internally, the General case solves three nested IVPs with `scipy.integrate.solve_ivp(method='DOP853')`:

```
σ_JK(t)  ← solve_ivp on equation_dot_sigma_jk    (first; no dependencies)
r_JK(t)  ← solve_ivp on equation_dot_r_jk        (needs cubic interp of σ_JK)
r⁽⁰⁾_JK(t)← solve_ivp on equation_dot_r_jk_0     (needs cubic interp of σ_JK, r_JK)
```

The Force case uses the closed-form integral expressions (Tab. 2) directly — no IVPs.

---

## 7. Extension points (generalizations)

### 7.1 Multi-qubit (N > 2)

- ODE loops already use `range(2**N_qubits)` → scale automatically.
- ❌ `src/Plots_Functions.py:_INDEX_MAP_3D` (line 755) hardcodes the four 2-qubit entries; needs to be generated as `[((j,k), label_jk, role_jk) for j,k in product(range(2**N), range(2**N))]`.
- ❌ `src/Measurements.py:Negativity_Qubits` hardcodes `dimensions_list = [dim, dim]`, `mask = [0, 1]` — only 2-qubit. For N > 2 needs `dimensions_list = [2] * N` and a mask describing the bipartition.

### 7.2 Multi-mode (n > 1)

- All ODE code uses `2*n_modes` correctly.
- ❌ `src/Measurements.py` hardcodes `r_measure = np.array([[v],[v]])` (lines 75, 148) — only n = 1. Needs `np.tile`/`np.repeat` over modes.
- ❌ `src/Symplectic_Known.py:Symplectic_QHO_N` is structurally broken (see BUGS.md §9).
- `src/Wigner_Functions.py` already takes a `mode_number` — picks one mode at a time. To plot multi-mode Wigners, decide on slice convention.

### 7.3 Non-Gaussian initial states

Major: the entire `sigma_0` / `r_0` parameterization assumes Gaussianity. Supporting non-Gaussian initial states would require either:
- Expanding the initial state in a Gaussian basis (sum of Gaussians) and propagating each term — every `Initialize_Gaussian_State` would become `Initialize_Mixture_Of_Gaussians`, and every downstream object would carry an extra "Gaussian-component index".
- Or: keeping the dynamics Gaussian but storing a non-Gaussian QRDM and reconstructing observables via Wigner-function integration.

### 7.4 Time-dependent decoherence (B(t))

- `Decoherence` currently stores `E_mat`, `D_mat` once at construction.
- For B(t): make `Decoherence` carry `B_func` and recompute `(E(t), D(t))` inside the ODE integrand. `src/Open_Operator_Time_Depentent.py` needs to be rewritten correctly first (see BUGS.md §1).

---

## 8. Known broken / dead code paths

Detailed list in `docs/BUGS.md`. Headline items:

1. Both `*_Time_Depentent.py` files (note the typo) have sign errors on `return -X.flatten()` lines. The "Open" variant additionally lacks all decoherence terms.
2. `src/Open_Operator_Force.py:Dynamics_General_Numerical` and `Dy_Analytical` reference functions that do not exist in scope (`Compute_R_JK_Unitary_Gaussian_Force_Numerical` and friends). Any call path that lands there raises `NameError`.
3. `src/Gaussian.py:Compute_Sigma_Open_Gaussian_Numerical_not_working` is named honestly; `Compute_Sigma_Open_Gaussian_Analytical` is also broken (undefined `y0`).
4. `src/Objects.py` lines 224, 249 dispatch to `Time_Depentent.Dynamics_Numerical` / `Dynamics_Analytical`, neither of which exists in those modules.

---

## 9. Style and naming notes

- Filename typos: **Depentent** → Dependent; **Indipendent** → Independent. Variable and string typos throughout: **Simpletic** → Symplectic, **Generadyne** → Generaldyne, **Measurament** → Measurement, **Caovariant** / **Covariant** mix, **eficency** → efficiency.
- `Hamiltonian.type[1] == 'General'` dispatches to `Unitary_Operator_Gaussian` — the "Gaussian" module name refers to *Gaussian-process-valued operators*, **not** the no-qubit case. Renaming this file (e.g., `Unitary_Operator_General.py`) would remove the most-asked-about confusion.
- Many modules pin `complex64` for storage; intermediate computations are `complex128`. Cast-down at storage loses precision in long-time integrations.
- Debug `print()` calls inside hot loops: `src/Unitary_Operator_Gaussian.py:162, 176, 177`, `src/Measurements.py:158`, `src/Gaussian.py:151`, several in `Objects.py`.

---

## 10. References inside the repo

- `notebooks/Examples_Unitary.ipynb`, `notebooks/Examples_Open.ipynb` — minimal-example demos.
- `notebooks/Examples_Paper.ipynb` — reproduces the two worked examples in paper Sec. VIII (qubit-resonator + Stern-Gerlach interferometer).
- `notebooks/Examples_Animation_Wigner.ipynb` — animations.
- `Mathematica_Analytical.nb` — symbolic crosscheck of analytical formulas (mostly the Force-case Table 2 integrals).
