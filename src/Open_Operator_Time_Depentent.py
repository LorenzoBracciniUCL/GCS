import numpy as np 
import matplotlib.pyplot as plt
import math as math 
import cmath as cmath
from scipy import linalg as linalg
from scipy import integrate as integ
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import QIT_Functions as QIT_Functions
import Gaussian as Gaussian

def Compute_r_J_func(J_labels,  r_array_func):
    """
    Function which computes the vector of translated first moments with qubit interaction f_q_i and additional force f_u
    Input:
    J     List of label for the qubits
    f_q   Strength force qubit
    f_u   Strength unkonwn force
    H_mat Hamiltonian Matrix (2n x 2n)
    Return: Vector of first moment r_jk (2n)
    """
    def r_J_func(t):
        r_J_val = r_array_func[0](t)
        for i in range(len(r_array_func)-1):
            r_J_val += J_labels[i]*r_array_func[i+1](t)

        return r_J_val

    return r_J_func

def Compute_H_J_func(J_labels,  H_array_func):
    """
    Function which computes the vector of translated first moments with qubit interaction f_q_i and additional force f_u
    Input:
    J     List of label for the qubits
    f_q   Strength force qubit
    f_u   Strength unkonwn force
    H_mat Hamiltonian Matrix (2n x 2n)
    Return: Vector of first moment r_jk (2n)
    """
    def H_J_func(t):
        H_J_val = H_array_func[0](t)
        for i in range(len(H_array_func)-1):
            H_J_val += J_labels[i]*H_array_func[i+1](t)

        return H_J_val

    return H_J_func


def equation_dot_sigma_jk_Time_Dep(t, y, H_J_func, H_K_func, n_modes):
    """
    Defines the first ODE for sigma_jk when sigma_jk is a 2n x 2n matrix.
    """
    Omega = Gaussian.Omega_N(n_modes) 
    sigma_jk = y.reshape((2*n_modes, 2*n_modes))  # Reshape to 2n x 2n matrix
    H_J = H_J_func(t)
    H_K = H_K_func(t)
    
    # Compute derivative
    sigma_jk_dot = 0.5 * (sigma_jk @ (H_J + H_K) @ Omega - Omega @ (H_J + H_K) @ sigma_jk) - 1j * (sigma_jk @ (H_J - H_K) @ sigma_jk + Omega @ (H_J - H_K) @ Omega)
    
    return - sigma_jk_dot.flatten()

def Compute_sigma_JK_Unitary_Gaussian_General_Numerical_Time_Dep(n_modes, sigma_0, H_J_func, H_K_func, t_array):
    # Flatten initial conditions
    y0 = sigma_0.flatten()
    
    # Time span and points
    t_span = (t_array[0], t_array[-1])
    
    # Solve the ODE system
    sigma_jk_sol = solve_ivp(equation_dot_sigma_jk_Time_Dep, t_span, y0, t_eval=t_array, args=(H_J_func, H_K_func, n_modes))
    
    # Extract solutions
    sigma_jk_t = sigma_jk_sol.y.T.reshape(-1, 2*n_modes, 2*n_modes)  # Reshape back to matrices
    return sigma_jk_t, sigma_jk_sol


def equation_dot_r_jk_Time_Dep(t, y, sigma_jk_interp, H_J_func, H_K_func, r_J_func, r_K_func, n_modes):
    """
    Defines the ODE for r_jk when r_jk is a 2n x 2n matrix.
    """
    Omega = Gaussian.Omega_N(n_modes) 
    r_jk = y.reshape((2*n_modes, 1))  # Reshape to 2n x 1 vector
    sigma_jk = sigma_jk_interp(t).reshape(2*n_modes, 2*n_modes)  # Properly reshape interpolated sigma_jk
    H_j = H_J_func(t)
    H_k = H_K_func(t)
    r_j = r_J_func(t)
    r_k = r_K_func(t)

  
    # Compute derivative
    r_jk_dot = -0.5 * Omega @ (H_j + H_k) @ r_jk + 0.5j * sigma_jk @ (H_j - H_k) @ r_jk + 0.5 * Omega @ (r_j + r_k) + 0.5j * sigma_jk @ (r_j - r_k)
    
    return - r_jk_dot.flatten()
    
def Compute_R_JK_Unitary_Gaussian_General_Numerical_Time_Dep(n_modes,r_0, sigma_JK_sol, H_J_func, H_K_func, r_J_func, r_K_func, t_array):
    
    sigma_JK_interp = interp1d(sigma_JK_sol.t, sigma_JK_sol.y.T, axis=0, kind='cubic', fill_value='extrapolate')
    # Flatten initial conditions
    y0 = r_0.flatten()
    
    # Time span and points
    t_span = (t_array[0], t_array[-1])
    
    # Solve for r_jk
    sol_r = solve_ivp(equation_dot_r_jk_Time_Dep, t_span, y0, t_eval=t_array, args=(sigma_JK_interp, H_J_func, H_K_func, r_J_func, r_K_func, n_modes))
    # Extract solutioon
    r_jk_sol = sol_r.y.T.reshape(-1, 2*n_modes, 1)  # Reshape back to vectors
   
    return r_jk_sol, sol_r

def equation_dot_r_jk_0_Time_Dep(t, y, sigma_JK_interp,r_JK_interp, H_J_func, H_K_func, r_J_func, r_K_func, J_labels, K_labels, H_q_0_array_func, n_modes):
    """
    Defines the ODE for r_0_jk when r_0_jk can be complex.
    """
    r_jk = r_JK_interp(t).reshape(2*n_modes, 1)  # Interpolated r_jk
    sigma_jk = sigma_JK_interp(t).reshape(2*n_modes, 2*n_modes)  # Interpolated sigma_jk
    H_j = H_J_func(t)
    H_k = H_K_func(t)
    r_j = r_J_func(t)
    r_k = r_K_func(t)
    H_q_0_array = np.zeros(len(H_q_0_array_func))
    for i in range(len(H_q_0_array_func)):
        H_q_0_array[i] = H_q_0_array_func[i](t)

    # Compute derivative
    r_0_jk_dot = (0.5 * r_jk.T @ (H_j - H_k) @ r_jk - (r_j - r_k).T @ r_jk
                   + 0.5 * np.trace((H_j - H_k) @ sigma_jk) + 0.5 * np.transpose(J_labels- K_labels)@H_q_0_array)

    
    return - (-1j * r_0_jk_dot).flatten()


def Compute_R_JK_0_Unitary_Gaussian_General_Numerical_Time_Dep(n_modes, r_JK_sol, sigma_JK_sol,  H_J_func, H_K_func, r_J_func, r_K_func, J_labels, K_labels, H_q_0_array_func, rho_q_0_JK, t_array):
    
    r_JK_interp = interp1d(r_JK_sol.t, r_JK_sol.y.T, axis=0, kind='cubic', fill_value='extrapolate')
    sigma_JK_interp = interp1d(sigma_JK_sol.t, sigma_JK_sol.y.T, axis=0, kind='cubic', fill_value='extrapolate')
    # Flatten initial conditions
    y0 = [np.log(rho_q_0_JK)]

    # Time span and points
    t_span = (t_array[0], t_array[-1])
    
    # Solve for r_jk
    sol_r_0 = solve_ivp(equation_dot_r_jk_0_Time_Dep, t_span, y0, t_eval=t_array, 
                        args=(sigma_JK_interp,r_JK_interp, H_J_func, H_K_func, r_J_func, r_K_func, J_labels, K_labels, H_q_0_array_func, n_modes))
    # Extract solutioon
    r_0_jk_sol = sol_r_0.y.T # Reshape back to vectors
      
    return r_0_jk_sol[:,0]

def Unitary_Gaussian_General_Time_Dep_Numerical(N_qubits, n_modes, r_0, sigma_0, H_array_func, r_array_func, H_q_0_array_func, rho_q_0, t_array):
    
    sigma_JK_t = np.zeros((len(t_array), 2**N_qubits, 2**N_qubits, 2*n_modes, 2*n_modes),dtype =np.complex64)
    r_JK_t = np.zeros((len(t_array), 2**N_qubits, 2**N_qubits, 2*n_modes, 1 ), dtype =np.complex64)
    r_JK_0_t = np.zeros((len(t_array), 2**N_qubits, 2**N_qubits), dtype =np.complex64)

    C_JK_t = np.zeros((len(t_array), 2**N_qubits, 2**N_qubits), dtype =np.complex64)
    phi_JK_t = np.zeros((len(t_array), 2**N_qubits, 2**N_qubits), dtype =np.complex64)

    for j in range(2**N_qubits):
        
        J_labels = QIT_Functions.Extract_Qubit_Labels_Array(N_qubits, j)
        r_J_func = Compute_r_J_func(J_labels, r_array_func)
        H_J_func = Compute_H_J_func(J_labels,  H_array_func)

        
        for k in range(j,2**N_qubits):
            K_labels = QIT_Functions.Extract_Qubit_Labels_Array(N_qubits, k)
            r_K_func = Compute_r_J_func(K_labels, r_array_func)
            H_K_func = Compute_H_J_func(K_labels,  H_array_func)
            
            # Compute quantities 

            sigma_JK_t[:,j,k], sigma_JK_sol = Compute_sigma_JK_Unitary_Gaussian_General_Numerical_Time_Dep(n_modes, sigma_0, H_J_func, H_K_func, t_array)
            r_JK_t[:,j,k], r_JK_sol = Compute_R_JK_Unitary_Gaussian_General_Numerical_Time_Dep(n_modes, r_0, sigma_JK_sol, H_J_func, H_K_func, r_J_func, r_K_func, t_array)
            r_JK_0_t[:,j,k] = Compute_R_JK_0_Unitary_Gaussian_General_Numerical_Time_Dep(n_modes, r_JK_sol, sigma_JK_sol,  H_J_func, H_K_func, r_J_func, r_K_func, J_labels, K_labels, H_q_0_array_func,rho_q_0[j,k], t_array)
            C_JK_t[:,j,k] = - (np.real(r_JK_0_t[:,j,k] - np.log(rho_q_0[j,k])))
            phi_JK_t[:,j,k] = np.imag(r_JK_0_t[:,j,k] - np.log(rho_q_0[j,k]))

            # Set complex conjugate  
            sigma_JK_t[:,k,j] = np.conjugate(sigma_JK_t[:,j,k]) 
            r_JK_t[:,k,j] = np.conjugate(r_JK_t[:,j,k]) 
            r_JK_0_t[:,k,j] = np.conjugate(r_JK_0_t[:,j,k]) 
            C_JK_t[:,k,j] = C_JK_t[:,j,k]
            phi_JK_t[:,k,j] = - phi_JK_t[:,j,k]

    return sigma_JK_t, r_JK_t, C_JK_t, phi_JK_t