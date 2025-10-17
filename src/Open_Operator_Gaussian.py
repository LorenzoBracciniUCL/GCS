import numpy as np 
import math as math 
import cmath as cmath
from scipy import linalg as linalg
from scipy import integrate as integ
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import Gaussian as Gaussian

import QIT_Functions as QIT_Functions

def Compute_r_J(N_qubits, J_labels,  r_ham_array):
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
    return r_J

def Compute_H_J(N_qubits, J_labels,  H_array):
    """
    Function which computes the vector of translated first moments with qubit interaction f_q_i and additional force f_u
    Input:
    J     List of label for the qubits
    f_q   Strength force qubit
    f_u   Strength unkonwn force
    H_mat Hamiltonian Matrix (2n x 2n)
    Return: Vector of first moment r_jk (2n)
    """
    H_J = H_array[0].copy()
    
    for i in range(N_qubits):
        H_J += J_labels[i]* H_array[i+1]

    return H_J

def equation_dot_sigma_jk(t, y, H_J, H_K,  E, D, n_modes):
    """
    Defines the first ODE for sigma_jk when sigma_jk is a 2n x 2n matrix.
    """
    Omega = Gaussian.Omega_N(n_modes) 
    sigma_jk = y.reshape((2*n_modes, 2*n_modes))  # Reshape to 2n x 2n matrix
    
    # Compute derivative
    sigma_jk_dot = 0.5 * (Omega @ (H_J + H_K +2*E) @ sigma_jk - sigma_jk @ (H_J + H_K+2*E.T) @ Omega ) + D - 1j * (sigma_jk@(H_J - H_K) @ sigma_jk + Omega @ (H_J - H_K) @ Omega)

    return sigma_jk_dot.flatten()

def Compute_sigma_JK_Open_General_Numerical(n_modes, sigma_0, H_J, H_K,  E,D, t_array):
    # Flatten initial conditions
    y0 = sigma_0.flatten()
    
    # Time span and points
    t_span = (t_array[0], t_array[-1])
    
    # Solve the ODE system
    sigma_jk_sol = solve_ivp(equation_dot_sigma_jk, t_span, y0, t_eval=t_array, args=(H_J, H_K, E,D, n_modes))
    
    
    
    # Extract solutions
    sigma_jk_t = sigma_jk_sol.y.T.reshape(-1,2*n_modes, 2*n_modes)  # Reshape back to matrices
    return sigma_jk_t, sigma_jk_sol


def equation_dot_r_jk(t, y, sigma_jk_interp, H_j, H_k, r_j, r_k,  E, d, n_modes):
    """
    Defines the ODE for r_jk when r_jk is a 2n x 2n matrix.
    """
    Omega = Gaussian.Omega_N(n_modes) 
    r_jk = y.reshape((2*n_modes, 1))  # Reshape to 2n x 1 vector
    sigma_jk = sigma_jk_interp(t).reshape(2*n_modes, 2*n_modes)  # Properly reshape interpolated sigma_jk
    
    # Compute derivative
    r_jk_dot = 0.5 * Omega @ (H_j + H_k+ 2*E) @ r_jk - 0.5j * sigma_jk @ (H_j - H_k) @ r_jk - 0.5 * Omega @ (r_j + r_k) + 0.5j * sigma_jk @ (r_j - r_k)
    
    return  r_jk_dot.flatten()
    
def Compute_R_JK_Open_General_Numerical(n_modes,r_0, sigma_JK_sol, H_j, H_k, r_j, r_k,  E,D, t_array):
    
    sigma_JK_interp = interp1d(sigma_JK_sol.t, sigma_JK_sol.y.T, axis=0, kind='cubic', fill_value='extrapolate')
    # Flatten initial conditions
    y0 = r_0.flatten()
    
    # Time span and points
    t_span = (t_array[0], t_array[-1])
    
    # Solve for r_jk
    sol_r = solve_ivp(equation_dot_r_jk, t_span, y0, t_eval=t_array, args=(sigma_JK_interp, H_j, H_k, r_j, r_k, E,D, n_modes))
    
    # Extract solutioon
    r_jk_sol = sol_r.y.T.reshape(-1, 2*n_modes, 1)  # Reshape back to vectors
   
    return r_jk_sol, sol_r

def equation_dot_r_jk_0(t, y, sigma_JK_interp,r_JK_interp, H_j, H_k, r_j, r_k, J_labels, K_labels, H_q_0_array,  n_modes):
    """
    Defines the ODE for r_0_jk when r_0_jk can be complex.
    """
    r_jk = r_JK_interp(t).reshape(2*n_modes, 1)  # Interpolated r_jk
    sigma_jk = sigma_JK_interp(t).reshape(2*n_modes, 2*n_modes)  # Interpolated sigma_jk
    
    # Compute derivative
    r_0_jk_dot =  1j *( - 0.5 * r_jk.T @ (H_j - H_k ) @ r_jk + (r_j - r_k).T @ r_jk
                - 0.25 * np.trace((H_j - H_k) @ sigma_jk) - 0.5 * np.transpose(J_labels- K_labels)@H_q_0_array)
    
    return r_0_jk_dot[0,0]


def Compute_R_JK_0_Open_General_Numerical(n_modes, r_JK_sol, sigma_JK_sol,  H_j, H_k, r_j, r_k, J_labels, K_labels, H_q_0_array, rho_q_0_JK, t_array):
    
    r_JK_interp = interp1d(r_JK_sol.t, r_JK_sol.y.T, axis=0, kind='cubic', fill_value='extrapolate')
    sigma_JK_interp = interp1d(sigma_JK_sol.t, sigma_JK_sol.y.T, axis=0, kind='cubic', fill_value='extrapolate')
    # Flatten initial conditions
    y0 = [np.log(rho_q_0_JK)]

    # Time span and points
    t_span = (t_array[0], t_array[-1])
    
    # Solve for r_jk
    sol_r_0 = solve_ivp(equation_dot_r_jk_0, t_span, y0, t_eval=t_array, 
                        args=(sigma_JK_interp,r_JK_interp, H_j, H_k, r_j, r_k, J_labels, K_labels, H_q_0_array,  n_modes))
    # Extract solutioon
    r_0_jk_sol = sol_r_0.y.T # Reshape back to vectors
      
    return r_0_jk_sol[:,0]

def Dynamics_Numerical(N_qubits, n_modes, r_0, sigma_0, H_array, r_ham_array, H_q_0_array, rho_q_0, E,D, t_array):
    
    sigma_JK_t = np.zeros((len(t_array), 2**N_qubits, 2**N_qubits, 2*n_modes, 2*n_modes),dtype =np.complex64)
    r_JK_t = np.zeros((len(t_array), 2**N_qubits, 2**N_qubits, 2*n_modes, 1 ), dtype =np.complex64)
    r_JK_0_t = np.zeros((len(t_array), 2**N_qubits, 2**N_qubits), dtype =np.complex64)
    C_JK_t = np.zeros((len(t_array), 2**N_qubits, 2**N_qubits), dtype =np.complex64)
    phi_JK_t = np.zeros((len(t_array), 2**N_qubits, 2**N_qubits), dtype =np.complex64)
    
    r_J = np.zeros((2*n_modes, 1), dtype =np.complex64)
    H_J = np.zeros((2*n_modes, 2*n_modes), dtype =np.complex64)
    r_K = np.zeros((2*n_modes, 1), dtype =np.complex64)
    H_K = np.zeros((2*n_modes, 2*n_modes), dtype =np.complex64)

    for j in range(2**N_qubits):
        
        J_labels = QIT_Functions.Extract_Qubit_Labels_Array(N_qubits, j)
        r_J = Compute_r_J(N_qubits, J_labels, r_ham_array)
        H_J = Compute_H_J(N_qubits, J_labels,  H_array)

        for k in range(j, 2**N_qubits):
            K_labels = QIT_Functions.Extract_Qubit_Labels_Array(N_qubits, k)
            r_K = Compute_r_J(N_qubits, K_labels, r_ham_array)
            H_K = Compute_H_J(N_qubits, K_labels,  H_array)
            
            # Compute quantities 
            sigma_JK_t[:,j,k], sigma_JK_sol = Compute_sigma_JK_Open_General_Numerical(n_modes, sigma_0, H_J, H_K,  E, D, t_array)
            r_JK_t[:,j,k], r_JK_sol = Compute_R_JK_Open_General_Numerical(n_modes, r_0, sigma_JK_sol, H_J, H_K, r_J, r_K,  E, D, t_array)
            r_JK_0_t[:,j,k] = Compute_R_JK_0_Open_General_Numerical(n_modes, r_JK_sol, sigma_JK_sol,  H_J, H_K, r_J, r_K, J_labels, K_labels, H_q_0_array,rho_q_0[j,k], t_array)
            C_JK_t[:,j,k] =  - np.real(r_JK_0_t[:,j,k] - np.log(rho_q_0[j,k]))
            phi_JK_t[:,j,k] = np.imag(r_JK_0_t[:,j,k] - np.log(rho_q_0[j,k]))

            # Set complex conjugate  
            sigma_JK_t[:,k,j] = np.conjugate(sigma_JK_t[:,j,k]) 
            r_JK_t[:,k,j] = np.conjugate(r_JK_t[:,j,k]) 
            r_JK_0_t[:,k,j] = np.conjugate(r_JK_0_t[:,j,k]) 
            C_JK_t[:,k,j] = C_JK_t[:,j,k]
            phi_JK_t[:,k,j] = - phi_JK_t[:,j,k]
                       
    return sigma_JK_t, r_JK_t, C_JK_t, phi_JK_t
