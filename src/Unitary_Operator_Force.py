import numpy as np 
import math as math 
import cmath as cmath
from scipy import linalg as linalg

import QIT_Functions as QIT_Functions

import Gaussian as Gaussian



def Compute_tilde_r_J(N_qubits, J_labels, H_m_inv, r_ham_array):
    """
    Function which computes the vector of translated first moments with qubit interaction f_q_i and additional force f_u
    Input:
    J     List of label for the qubits
    f_q   Strength force qubit
    f_u   Strength unkonwn force
    H_mat Hamiltonian Matrix (2n x 2n)
    Return: Vector of first moment r_jk (2n)
    """
    r_J = r_ham_array[0].copy()
    for i in range(N_qubits):
        r_J += J_labels[i]*r_ham_array[i+1]
    return H_m_inv@r_J

def Compute_Sigma_JK_Unitary_Force(N_qubits, n_modes, sigma_0, S_m):
    sigma_JK_t = np.zeros((2**N_qubits, 2**N_qubits, 2*n_modes, 2*n_modes))
    sigma_t = Gaussian.Compute_sigma_t_S_m(sigma_0, S_m)
    for i in range(2**N_qubits):
        for j in range(2**N_qubits):
            sigma_JK_t[i,j] = sigma_t
    return sigma_JK_t

def Compute_R_JK_Unitary_Force(n_modes, r_0_t, sigma_t, r_J_t,r_J, r_K_t, r_K):
    omega = Gaussian.Omega_N(n_modes)
    R_JK = r_0_t  - 1/2*(r_J_t - r_J + r_K_t - r_K ) - 1J/2* sigma_t@omega@(r_J_t - r_J - r_K_t + r_K)
    return R_JK

def Compute_C_JK_Unitary_Force(n_modes, sigma_t, r_J_t,r_J, r_K_t, r_K):
    omega = Gaussian.Omega_N(n_modes)
    C_JK =  1/4*np.transpose(r_J_t - r_J - r_K_t + r_K)@np.transpose(omega)@sigma_t@omega@(r_J_t - r_J - r_K_t + r_K)
    return np.real(C_JK[0,0])

def Compute_phi_JK_Unitary_Force(n_modes, r_0_t, r_0, r_J_t,r_J, r_K_t, r_K, H_m, H_q_0_array,J_labels, K_labels, time):
    omega = Gaussian.Omega_N(n_modes)
    phi_1 = - np.transpose(r_J - r_K)@omega@(r_0_t  -r_0)
    phi_2 =  1/2*np.transpose(r_J - r_K)@omega@(r_J_t - r_J + r_K_t - r_K)
    phi_3 =  time/2*( np.transpose(r_J - r_K)@H_m@(r_J+r_K) - np.transpose(H_q_0_array)@(J_labels - K_labels)) #### Meno a Caso
    return np.real((phi_1 + phi_2 + phi_3)[0,0])



def Dynamics_Numerical(N_qubits, n_modes, r_0, sigma_0, H_m, r_ham_array, H_q_0_array, t_array):
    H_m_inv = np.linalg.inv(H_m)
    
    sigma_JK_t = np.zeros((len(t_array), 2**N_qubits, 2**N_qubits, 2*n_modes, 2*n_modes))
    r_JK_t = np.zeros((len(t_array), 2**N_qubits, 2**N_qubits, 2*n_modes, 1 ), dtype =np.complex64)
    C_JK_t = np.zeros((len(t_array), 2**N_qubits, 2**N_qubits))
    phi_JK_t = np.zeros((len(t_array), 2**N_qubits, 2**N_qubits))

    for i in range(len(t_array)):
        # Compute simplectic transformation and covariant matrix (all equal)

        S_m_t = Gaussian.Compute_S_m(n_modes, H_m, t_array[i])
        sigma_JK_t[i] = Compute_Sigma_JK_Unitary_Force(N_qubits, n_modes, sigma_0, S_m_t)
        r_0_t = S_m_t@r_0
        
        for j in range(2**N_qubits):
            J_labels = QIT_Functions.Extract_Qubit_Labels_Array(N_qubits, j)
            r_J = Compute_tilde_r_J(N_qubits, J_labels, H_m_inv, r_ham_array)
            r_J_t = S_m_t@r_J
            
            for k in range(2**N_qubits):
                K_labels = QIT_Functions.Extract_Qubit_Labels_Array(N_qubits, k)
                r_K = Compute_tilde_r_J(N_qubits, K_labels, H_m_inv, r_ham_array)
                r_K_t = S_m_t@r_K
                # Compute quantities 
                r_JK_t[i,j,k] = Compute_R_JK_Unitary_Force(n_modes, r_0_t, sigma_JK_t[i,j,k], r_J_t, r_J, r_K_t, r_K)
                C_JK_t[i,j,k] = Compute_C_JK_Unitary_Force(n_modes, sigma_JK_t[i,j,k], r_J_t, r_J, r_K_t, r_K)
                phi_JK_t[i,j,k] = Compute_phi_JK_Unitary_Force(n_modes, r_0_t, r_0, r_J_t,r_J, r_K_t, r_K, H_m, H_q_0_array,J_labels, K_labels, t_array[i])
                            
    return sigma_JK_t, r_JK_t, C_JK_t, phi_JK_t 


def Dynamics_Analytical(N_qubits, n_modes, r_0, sigma_0, S_func, H_m, r_ham_array, H_q_0_array, t_array):
    H_m_inv = np.linalg.inv(H_m)
    
    sigma_JK_t = np.zeros((len(t_array), 2**N_qubits, 2**N_qubits, 2*n_modes, 2*n_modes))
    r_JK_t = np.zeros((len(t_array), 2**N_qubits, 2**N_qubits, 2*n_modes, 1 ), dtype =np.complex64)
    C_JK_t = np.zeros((len(t_array), 2**N_qubits, 2**N_qubits))
    phi_JK_t = np.zeros((len(t_array), 2**N_qubits, 2**N_qubits))

    for i in range(len(t_array)):
        # Compute simplectic transformation and covariant matrix (all equal)

        S_m_t = S_func(t_array[i])
        sigma_JK_t[i] = Compute_Sigma_JK_Unitary_Force(N_qubits, n_modes, sigma_0, S_m_t)
        r_0_t = S_m_t@r_0
        
        for j in range(2**N_qubits):
            J_labels = QIT_Functions.Extract_Qubit_Labels_Array(N_qubits, j)
            r_J = Compute_tilde_r_J(N_qubits, J_labels, H_m_inv, r_ham_array)
            r_J_t = S_m_t@r_J
            
            for k in range(j, 2**N_qubits):
                K_labels = QIT_Functions.Extract_Qubit_Labels_Array(N_qubits, k)
                r_K = Compute_tilde_r_J(N_qubits, K_labels, H_m_inv, r_ham_array)
                r_K_t = S_m_t@r_K
                # Compute quantities 
                r_JK_t[i,j,k] = Compute_R_JK_Unitary_Force(n_modes, r_0_t, sigma_JK_t[i,j,k], r_J_t, r_J, r_K_t, r_K)
                C_JK_t[i,j,k] = Compute_C_JK_Unitary_Force(n_modes, sigma_JK_t[i,j,k], r_J_t, r_J, r_K_t, r_K)
                phi_JK_t[i,j,k] = Compute_phi_JK_Unitary_Force(n_modes, r_0_t, r_0, r_J_t,r_J, r_K_t, r_K, H_m, H_q_0_array,J_labels, K_labels, t_array[i])

                # Set complex conjugate  
                sigma_JK_t[:,k,j] = np.conjugate(sigma_JK_t[:,j,k]) 
                r_JK_t[:,k,j] = np.conjugate(r_JK_t[:,j,k]) 
                C_JK_t[:,k,j] = C_JK_t[:,j,k]
                phi_JK_t[:,k,j] = - phi_JK_t[:,j,k]
                            
    return sigma_JK_t, r_JK_t, C_JK_t, phi_JK_t 