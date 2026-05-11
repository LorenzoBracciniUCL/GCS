# QCAT / GCS — Project context for Claude

This is a Python library for **Gaussian-Branched Cat States** (`N` qubits × `n` bosonic modes, paper: [arXiv:2510.01156](https://arxiv.org/abs/2510.01156)). State = superposition of Gaussian processes indexed by branch labels `J, K ∈ {±1}^N`. Each branch carries `(σ_JK, r_JK, C_JK, φ_JK)`; the QRDM is reconstructed as `ϱ^q_JK(t) = exp(−C_JK + iφ_JK) · ϱ^q_JK(0)`.

For a deep dive, read `docs/ARCHITECTURE.md` (paper-to-code map) and `docs/BUGS.md` (known issues). The brief below is the load-bearing context only.

## Source tree (`src/`)

- `Gaussian.py` — `Omega_N`, `U_N`, `S_m = exp(ΩHτ)`, no-qubit ODEs, `Get_Decoherence_Rates(B) → (E, D)`.
- `QIT_Functions.py` — `Extract_Qubit_Labels_Array(N, j)` returns `J ∈ {±1}^N` (map `2·bit − 1`).
- `Symplectic_Known.py` — closed-form `S_m(τ)`: `QHO_1` ✓, `QHO_n` ❌ broken, `QHO_2_XX` ✓.
- `Unitary_Operator_Force.py` — Force case (H_q=0, r_q≠0); closed-form Table-2 integrals.
- `Unitary_Operator_Gaussian.py` — **General** case (despite the name). Three nested ODEs σ → r → r⁽⁰⁾, Tab. 1.
- `Open_Operator_Force.py` — only `Dynamics_Symetric_Numerical` (E=0) is reliable; `Dynamics_General_Numerical` calls undefined functions.
- `Open_Operator_Gaussian.py` — General + Lindblad, Tab. 3. Missing Ω·d driving and Γ_z dephasing terms.
- `*_Time_Depentent.py` (sic — typo in filename) — sign-flipped vs constant versions; "Open" variant lacks E/D entirely.
- `Measurements.py` — projective qubit measurements, generaldyne/homodyne (Eq. 51), PPT log-negativity (hardcoded N=2).
- `Wigner_Functions.py` — Wigner functions: diagonal, PMS, Gaussian.
- `Plots_Functions.py` — matplotlib: 2-D Wigner, 3-D vectors/ellipsoids, animations. 3-D plotting hardcodes 2-qubit (`_INDEX_MAP_3D`).
- `Objects.py` — `Hamiltonian`, `Symplectic`, `Decoherence`, `Quantum_State` (OOP front-end).

## Conventions

- Canonical ordering `r̂ = (x₁, p₁, …, x_n, p_n)ᵀ`; `Ω₁ = [[0,1],[-1,0]]`.
- `H_J = H_m + Σ_i J_i·H_q,i`, `r_J = r_m + Σ_i J_i·r_q,i`. `H_array = [H_m, H_q,1, …, H_q,N]`; same for `r_array`.
- Hermitian-conjugate symmetry: `σ_KJ = σ*_JK`, `r_KJ = r*_JK`, `C_KJ = C_JK`, `φ_KJ = −φ_JK`. Only `j ≤ k` is solved; lower triangle is filled by conjugation.
- Time-axis is always **leading** index. Shapes on `Quantum_State`:
  - `sigma_JK_t : (T, 2^N, 2^N, 2n, 2n)`, `r_JK_t : (T, 2^N, 2^N, 2n, 1)`
  - `C_JK_t, phi_JK_t : (T, 2^N, 2^N)`, `rho_q_t : (T, 2^N, 2^N)`
- ODE solver: `scipy.integrate.solve_ivp` with `method='DOP853'`. σ ODE solved first, then `interp1d(kind='cubic')` feeds r ODE, then both feed r⁽⁰⁾ ODE.
- Decoherence input `B` (complex `2n × 2n`); `E = −Im(B)`, `D = 2Ω·Re(B)·Ωᵀ`. Pass `basis='Canonical'` or `'Ladder'`.

## Hamiltonian-type dispatch (`Objects.Hamiltonian.type`)

| `.type[1]` | Routes to | When |
|---|---|---|
| `'Gaussian'` | `src/Gaussian.py` | No qubit coupling at all — `len(H_array) == 1`, `len(r_array) == 1`. |
| `'Force'` | `Unitary_Operator_Force.py` / `Open_Operator_Force.py` | `len(H_array) == 1` but `len(r_array) > 1`. |
| `'General'` | `Unitary_Operator_Gaussian.py` / `Open_Operator_Gaussian.py` | `len(H_array) > 1`. |

## Gotchas

- **Name confusion**: file `Unitary_Operator_Gaussian.py` is the **General** case, *not* the pure-Gaussian path. The `"Gaussian"` in the filename means "Gaussian-process-valued operator".
- **Off-diagonal σ_JK is complex and not Hermitian** for `J ≠ K`. Plot code that treats `Im[σ_00]` as a real radius must `abs()` before `sqrt` (already fixed in `Draw_Ellipsoids`).
- Many sub-paths in `Objects.Open_Dynamics_*` dispatch to functions that **do not exist** in the target modules (see `docs/BUGS.md` for the list). Most likely to bite: `Open_Dynamics_Numerical` with `Hamiltonian.type[0] == 'Time'`.
- Mixed `complex64`/`complex128` dtypes across modules — be careful with precision-sensitive computations.

## When in doubt

- ODE form / coefficients → check Tab. 1 (unitary), Tab. 2 (force closed form), Tab. 3 (open) of the paper.
- Branch indexing → `QIT_Functions.Extract_Qubit_Labels_Array`.
- Decoherence convention → paper Eq. 46 and `Gaussian.Get_Decoherence_Rates`.
- For 3-D phase-space plots, `_INDEX_MAP_3D` and `_DEFAULT_COLOR_MAP_3D` in `Plots_Functions.py` define how (j,k) maps to colors/labels.

Default to writing no comments unless the *why* is non-obvious. Treat the paper as the authoritative reference; treat existing code as suggestive but verify against the paper before reusing a formula.
