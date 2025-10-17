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

def Create_XX_PP(GCS, sigma_para, steps):

    if len(np.shape(GCS.r_JK_t)) > 4:
        max_x = np.max(np.real(GCS.r_JK_t[:,:,:,0,0] + sigma_para*GCS.sigma_JK_t[:,:,:,0,0]))
        max_p = np.max(np.real(GCS.r_JK_t[:,:,:,1,0] + sigma_para*GCS.sigma_JK_t[:,:,:,1,1]))
        max_max = int(np.max(np.array([max_x, max_p])))+1
    else:
        max_x = np.max(np.real(GCS.r_t[:,0] + sigma_para*GCS.sigma_t[:,0,0]))
        max_p = np.max(np.real(GCS.r_t[:,1] + sigma_para*GCS.sigma_t[:,1,1]))
        max_max = int(np.max(np.array([max_x, max_p])))+3

    x_array = np.linspace(-max_max, max_max, steps)
    p_array = np.linspace(-max_max, max_max, steps)
    XX, PP = np.meshgrid(x_array, p_array)

    return x_array, p_array,  XX, PP

def Create_r_tilde_array(GCS,sigma_para, mode_number, steps):
    """creates the Matrix of vectors"""

    x_array, p_array,  XX, PP = Create_XX_PP(GCS, sigma_para, steps)
    r_tilde_array  = np.zeros((steps, steps, 2*GCS.n_modes, 1))
    for i in range(GCS.n_modes):
        if i == mode_number:
            for j in range(steps):
                for k in range(steps):
                    r_tilde_array[j,k,2*i,0] = XX[j,k]
                    r_tilde_array[j,k,2*i+1,0] = PP[j,k]
        else:
            pass

    return r_tilde_array, x_array, p_array,  XX, PP

def Wigner_PMS_func(GCS, time_index, r_tilde_array):
    """
    Computes the Wigner function W_jk at time t.
    
    Parameters:
    - GCS: Cat Gaussian State
    - time_index: time index
    - r_tilde_array: matrix of phase space vectors 
    - 
    Returns:
    - Matrix of the Wigner function of diagonal 
    """

    W_PMS = np.zeros((len(GCS.POVM_array[0]), len(r_tilde_array), len(r_tilde_array)), dtype =np.complex64)

    sigma_JK = GCS.sigma_JK_t[time_index]
    r_JK = GCS.r_JK_t[time_index] 
    rho_q = GCS.rho_q_t[time_index]  
    N_qubits = GCS.N_qubits
    n_modes = GCS.n_modes

    for i in range(2**(N_qubits)):
        for j in range(2**(N_qubits)):
            det_sigma = np.linalg.det(sigma_JK[i,j])
            sigma_inv = np.linalg.inv(sigma_JK[i,j])
            for m in range(len(r_tilde_array)):
                for n in range(len(r_tilde_array)):
                    for a in range(len(W_PMS)):
                        W_PMS[a,m,n] += GCS.POVM_array[a,i,j]*(2**n_modes)*rho_q[i,j]/(np.pi**n_modes * np.sqrt(det_sigma))*np.exp(-np.transpose(r_tilde_array[m,n] - r_JK[i,j])@sigma_inv@(r_tilde_array[m,n] - r_JK[i,j]))
    for a in range(len(W_PMS)):
        if GCS.prob_t[a,time_index] == 0:
            W_PMS[a] = np.zeros((len(r_tilde_array), len(r_tilde_array)))
        else:
            W_PMS[a] = W_PMS[a]/GCS.prob_t[a,time_index]
    return np.real(W_PMS)

def Wigner_Sum_Diagonal(GCS, time_index, r_tilde_array):
    """
    Computes the Wigner function W_jk using NumPy.
    
    Parameters:
    - GCS: Cat Gaussian State
    - time_index: time index
    - r_tilde_array: matrix of phase space vectors 
    - 
    Returns:
    - Matrix of the Wigner function of diagonal 
    """
    sigma_JK = GCS.sigma_JK_t[time_index]
    r_JK = GCS.r_JK_t[time_index] 
    rho_q = GCS.rho_q_t[time_index]  
    N_qubits = GCS.N_qubits
    n_modes = GCS.n_modes

    W_Diag = np.zeros((len(r_tilde_array), len(r_tilde_array)))

    for i in range(2**(N_qubits)):
        det_sigma = np.linalg.det(sigma_JK[i,i])
        sigma_inv = np.linalg.inv(sigma_JK[i,i])
        for m in range(len(r_tilde_array)):
            for n in range(len(r_tilde_array)):
                W_Diag[m,n] += (2**n_modes)*rho_q[i,i]/(np.pi**n_modes * np.sqrt(det_sigma))*np.exp(-np.transpose(r_tilde_array[m,n] - r_JK[i,i])@sigma_inv@(r_tilde_array[m,n] - r_JK[i,i]))[0,0]
    return W_Diag

def Wigner_Gaussian(GCS, time_index, r_tilde_array):
    n_modes = GCS.n_modes
    sigma = GCS.sigma_t[time_index]
    r_first = GCS.r_t[time_index]

    W_Gausss = np.zeros((len(r_tilde_array), len(r_tilde_array)))

    det_sigma = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    for m in range(len(r_tilde_array)):
        for n in range(len(r_tilde_array)):
            W_Gausss[m,n] += (2**n_modes)/(np.pi**n_modes * np.sqrt(det_sigma))*np.exp(-np.transpose(r_tilde_array[m,n] -r_first)@sigma_inv@(r_tilde_array[m,n] - r_first))[0,0]

    return W_Gausss


def Wigner_t_Sum_Diag(GCS, r_tilde_array):
    """
    Computes the time evolution of the diagonal Wigner function
    - W_jk: float -> Computed Wigner function value
    """

    W_D_t = np.zeros((len(GCS.t_array), len(r_tilde_array), len(r_tilde_array)))

    for i in range(len(GCS.t_array)):
        W_D_t[i] = Wigner_Sum_Diagonal(GCS, i, r_tilde_array)

    return W_D_t

def Wigner_t_PMS(GCS, r_tilde_array):
    """
    Computes the time evolution of the Wigner function of PMS
    """

    W_PMS_t = np.zeros((len(GCS.t_array), len(GCS.POVM_array[0]), len(r_tilde_array), len(r_tilde_array)))

    for i in range(len(GCS.t_array)):
        W_PMS_t[i] = Wigner_PMS_func(GCS, i, r_tilde_array)

    return W_PMS_t

def Wigner_t_Gauss(GCS, r_tilde_array):
    """
    Computes the time evolution of the Wigner function of PMS
    """

    W_t_Gausss = np.zeros((len(GCS.t_array),  len(r_tilde_array), len(r_tilde_array)))

    for i in range(len(GCS.t_array)):
        W_t_Gausss[i] = Wigner_Gaussian(GCS, i, r_tilde_array)

    return W_t_Gausss



def Wigner_jk_t_Func(GCS, r_tilde_array):
    """
    Computes the Time evolition of the Wigner function W_jk using NumPy.
    - W_jk: float -> Computed Wigner function value
    """

    W_jk_t = np.zeros((len(GCS.t_array), 2**(GCS.N_qubits),2**(GCS.N_qubits),len(r_tilde_array), len(r_tilde_array)))

    for i in range(len(GCS.t_array)):
        W_jk_t[i] = Wigner_jk_func(GCS, i, r_tilde_array)

    return W_jk_t

def Wigner_jk_func(GCS, time_index, r_tilde_array):
    """
    Computes the Wigner function W_jk at time t.
    
    Parameters:
    - GCS: Cat Gaussian State
    - time_index: time index
    - r_tilde_array: matrix of phase space vectors 
    - 
    Returns:
    - Matrix of the Wigner function of diagonal 
    """
    sigma_JK = GCS.sigma_JK_t[time_index]
    r_JK = GCS.r_JK_t[time_index] 
    rho_q = GCS.rho_q_t[time_index]  
    N_qubits = GCS.N_qubits
    n_modes = GCS.n_modes

    W_jk = np.zeros((2**(N_qubits),2**(N_qubits),len(r_tilde_array), len(r_tilde_array)), dtype =np.complex64)

    for i in range(2**(N_qubits)):
        for j in range(2**(N_qubits)):
            det_sigma = np.linalg.det(sigma_JK[i,j])
            sigma_inv = np.linalg.inv(sigma_JK[i,j])
            for m in range(len(r_tilde_array)):
                for n in range(len(r_tilde_array)):
                    W_jk[i,j,m,n] = (2**n_modes)*rho_q[i,j]/(np.pi**n_modes * np.sqrt(det_sigma))*np.exp(-np.transpose(r_tilde_array[m,n] - r_JK[i,j])@sigma_inv@(r_tilde_array[m,n] - r_JK[i,j]))[0,0]
    return 