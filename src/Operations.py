import numpy as np
import Gaussian as Gaussian


def Conditional_Displacement(N_qubits, n_modes, sigma_JK, r_JK, r_array):
    """
    Apply a conditional displacement to a GCS snapshot.

    Parameters
    ----------
    sigma_JK : (2^N, 2^N, 2n, 2n)
    r_JK     : (2^N, 2^N, 2n, 1)
    r_array  : (2^N, 2n, 1)  — displacement vector r_j for each branch J

    Returns
    -------
    sigma_new : (2^N, 2^N, 2n, 2n)  — unchanged
    r_new     : (2^N, 2^N, 2n, 1)
    delta_C   : (2^N, 2^N)
    delta_phi : (2^N, 2^N)

    Update rules (Eq. 5):
      sigma_jk  unchanged
      r_jk  -> r_jk - 1/2 (r_j + r_k) - i/2 sigma_jk Omega (r_j - r_k)
      C_jk  -> C_jk + 1/4 (r_j-r_k)^T Omega^T sigma_jk Omega (r_j-r_k)
                     + (r_j-r_k)^T Omega r_jk^I
      phi_jk -> phi_jk + (r_j-r_k)^T Omega r_jk^R - 1/2 r_j^T Omega r_k
    """
    N2 = 2**N_qubits
    Omega = Gaussian.Omega_N(n_modes)

    sigma_new = sigma_JK.copy()
    r_new = r_JK.copy().astype(complex)
    delta_C = np.zeros((N2, N2))
    delta_phi = np.zeros((N2, N2))

    for j in range(N2):
        r_j = r_array[j]
        for k in range(j, N2):
            r_k = r_array[k]
            r_jk = r_JK[j, k]
            diff = r_j - r_k
            r_jk_R = np.real(r_jk)
            r_jk_I = np.imag(r_jk)
            sigma_jk = sigma_JK[j, k]

            r_new[j, k] = r_jk - 0.5*(r_j + r_k) - 0.5j * sigma_jk @ Omega @ diff
            delta_C[j, k] = np.real(
                0.25 * (diff.T @ Omega.T @ sigma_jk @ Omega @ diff)[0, 0]
                + (diff.T @ Omega @ r_jk_I)[0, 0]
            )
            delta_phi[j, k] = np.real(
                (diff.T @ Omega @ r_jk_R)[0, 0]
                - 0.5 * (r_j.T @ Omega @ r_k)[0, 0]
            )

            r_new[k, j] = np.conjugate(r_new[j, k])
            delta_C[k, j] = delta_C[j, k]
            delta_phi[k, j] = -delta_phi[j, k]

    return sigma_new, r_new, delta_C, delta_phi


def Gaussian_Operation(N_qubits, n_modes, sigma_JK, r_JK, X, Y, r_c):
    """
    Apply a Gaussian operation described by matrices X, Y and displacement r_c.

    Parameters
    ----------
    sigma_JK : (2^N, 2^N, 2n, 2n)
    r_JK     : (2^N, 2^N, 2n, 1)
    X, Y     : (2n, 2n)
    r_c      : (2n, 1)

    Returns
    -------
    sigma_new : (2^N, 2^N, 2n, 2n)
    r_new     : (2^N, 2^N, 2n, 1)
    delta_C   : (2^N, 2^N)  — constant ln(det X) for all (j,k)
    delta_phi : (2^N, 2^N)  — zero

    Update rules:
      sigma_jk -> Omega X^{-1} Omega^T sigma_jk Omega X^{-1T} Omega^T
                  + Omega X^{-1} Y X^{-1T} Omega^T
      r_jk     -> Omega X^{-1T} Omega^T r_jk + r_c
      C_jk     -> C_jk + ln(det X)
      phi_jk   unchanged
    """
    N2 = 2**N_qubits
    Omega = Gaussian.Omega_N(n_modes)

    X_inv = np.linalg.inv(X)
    M = Omega @ X_inv @ Omega.T          # appears in sigma update: M sigma M^T + N
    M_r = Omega @ X_inv.T @ Omega.T     # = M^T, appears in r update: M_r r + r_c
    N_mat = Omega @ X_inv @ Y @ X_inv.T @ Omega.T
    log_det_X = np.real(np.log(np.linalg.det(X)))

    sigma_new = np.zeros_like(sigma_JK, dtype=complex)
    r_new = np.zeros_like(r_JK, dtype=complex)
    delta_C = np.full((N2, N2), log_det_X)
    delta_phi = np.zeros((N2, N2))

    for j in range(N2):
        for k in range(j, N2):
            sigma_new[j, k] = M @ sigma_JK[j, k] @ M.T + N_mat
            r_new[j, k] = M_r @ r_JK[j, k] + r_c

            sigma_new[k, j] = np.conjugate(sigma_new[j, k])
            r_new[k, j] = np.conjugate(r_new[j, k])

    return sigma_new, r_new, delta_C, delta_phi
