import numpy as np 
import math as math 
import cmath as cmath
from scipy import linalg as linalg

from scipy import linalg as linalg
from scipy import integrate as integ
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from scipy.integrate import quad, quad_vec

import QIT_Functions as QIT_Functions

def Omega_N(n_modes):
    """Generates the Simpletic Form of n modes"""
    omega_n = np.zeros((2*n_modes, 2*n_modes))
    for i in range(n_modes):
        omega_n[2*i + 1, 2 * i] = -1
        omega_n[2*i , 2 * i+ 1] = 1
    return omega_n

def U_N(n_modes):
    """Transformation from ladder to canonical, \alpha = U r"""
    u_n = np.zeros((2*n_modes, 2*n_modes), dtype =np.complex64)
    for i in range(n_modes):
        u_n[int(2*i), int(2*i)] = 1
        u_n[int(2*i+1), int(2*i)] = 1
        u_n[int(2*i), int(2*i+1)] = 1j
        u_n[int(2*i+1) , int(2 * i+ 1)] = -1j
    return u_n/np.sqrt(2)

def Compute_r_tilde(H_m, r_m):
    return np.linalg.inv(H_m)@r_m

def Compute_S_m(n_modes, H_mat, time):
    """
    Compute the INVERSE of the symplectic transformation matrix
    Input:
    H_mat    Hamiltonian Matrix (2n x 2n)
    time     Time 
    Output:  Symplectic Transformation Matrix (2n x 2n)
    """
    Omega = Omega_N(n_modes)
    S_matrix = linalg.expm(Omega@H_mat*time)
    return S_matrix

def Compute_r_t_S_m(r_0, r_tilde_m, S_m,):
    """
    Compute the unitary time evolution of the first moments
    Input:
    r_0         Vector of first moment at t=0 (2n)
    S_matrix    Symplectic Transformation Matrix (2n x 2n)
    Output:  Vector of first moment at t (2n)
    """
    r_t =np.real(S_m@(r_0 + r_tilde_m) - r_tilde_m)
    return r_t

def Compute_sigma_t_S_m(sigma_0, S_matrix):
    """
    Compute the unitary time evolution of the covariant matrix 
    Input:
    sigma_0     Covariant Matrix at t=0 (2n x 2n)
    S_matrix    Symplectic Transformation Matrix (2n x 2n)
    Output:  Covariant Matrix at time t (2n x 2n)
    """
    sigma_t = np.real(S_matrix@sigma_0@np.transpose(S_matrix))
    return sigma_t
 
def log_nega_2_modes(sigma):
    det = linalg.det(sigma)
    sigma_a = np.array([[sigma[0,0],sigma[0,1]],[sigma[1,0],sigma[1,1]]])
    sigma_b = np.array([[sigma[2,2],sigma[2,3]],[sigma[3,2],sigma[3,3]]])
    sigma_ab = np.array([[sigma[0,2],sigma[0,3]],[sigma[1,2],sigma[1,3]]])
    delta_tilde = linalg.det(sigma_a) + linalg.det(sigma_b) - 2* linalg.det(sigma_ab)
    mu_minus = np.real(np.sqrt((delta_tilde- np.sqrt(delta_tilde**2-4*det))/2))
    if mu_minus>0:
        nega = np.max(np.array([0, - math.log2(mu_minus)]))
    else:
        nega = 0
    return nega

def Unitary_Numerical(n_modes, r_0, sigma_0, H_m, r_m, t_array):
    """
    Compute the unitary time evolution of the first and second moments 
    Input:
    n_modes     Number of Modes
    r_0         Initial First Moment (2n, 1)
    sigma_0     Covariant Matrix at t=0 (2n x 2n)
    N_m         Hamiltonian Matrix (2n x 2n)
    r_m         Force (2n x 1)
    Output:  First and Second Moments at an array of time t (2n x 2n)
    """
    r_t = np.zeros((len(t_array),  2*n_modes, 1))
    sigma_t = np.zeros((len(t_array), 2*n_modes, 2*n_modes))
    
    for i in range(len(t_array)):
        S_m_t = Compute_S_m(n_modes, H_m, t_array[i])
        r_t[i] = Compute_r_t_S_m(r_0, r_m, S_m_t)
        sigma_t[i] = Compute_sigma_t_S_m(sigma_0, S_m_t)
        
    return r_t, sigma_t

def Unitary_Analytical(n_modes, r_0, sigma_0, symplectic_func, H_m, r_m, t_array):
    """
    Compute the unitary time evolution of the first and second moments 
    Input:
    n_modes     Number of Modes
    r_0         Initial First Moment (2n x 1)
    sigma_0     Covariant Matrix at t=0 (2n x 2n)
    S_m_func    Analytical Function of S_m
    r_m         Force (2n x 1)
    Output:  First and Second Moments at an array of time t (2n x 2n)
    """
    
    r_t = np.zeros((len(t_array),  2*n_modes, 1))
    sigma_t = np.zeros((len(t_array), 2*n_modes, 2*n_modes))
    
    r_tilde_m = Compute_r_tilde(H_m, r_m)
    
    for i in range(len(t_array)):
        
        S_m_t = symplectic_func(t_array[i])
        r_t[i] = Compute_r_t_S_m(r_0, r_tilde_m, S_m_t)
        sigma_t[i] = Compute_sigma_t_S_m(sigma_0, S_m_t)
        
    return r_t, sigma_t

################################ Open Dynamcis ################################

def Get_Decoherence_Rates(n_modes, B_mat):
    D_mat = 2*Omega_N(n_modes)@np.real(B_mat)@Omega_N(n_modes).T
    E_mat = - np.imag(B_mat)
    return E_mat, D_mat

def Compute_Sigma_Open_Gaussian_Numerical_not_working(n_modes, sigma_0, H_m, E, D, t_array):
    sigma_t = np.zeros((len(t_array),2*n_modes,2*n_modes))
    Omega = Omega_N(n_modes)
    
    for i in range(len(t_array)):
        S_tau = np.exp(t_array[i]*Omega@(H_m+E))
        
        def Integrand(t):
            S_m_t_tau = np.exp(Omega@(H_m+E)*(t_array[i] - t))
            term = S_m_t_tau@D@S_m_t_tau.T
            return term.flatten() 
        
        additional_term  = quad_vec(Integrand, 0, t_array[i],epsabs=1e-20)
        
        sigma_t[i] = S_tau@sigma_0@np.transpose(S_tau) + additional_term[0].reshape((2*n_modes, 2*n_modes))
        print(additional_term[0].reshape((2*n_modes, 2*n_modes)))

    return sigma_t



def equation_dot_sigma(t, y, H_m, E, D,n_modes):
    """
    Defines the first ODE for sigma_jk when sigma_jk is a 2n x 2n matrix.
    """
    Omega = Omega_N(n_modes) 
    sigma= y.reshape((2*n_modes, 2*n_modes))  # Reshape to 2n x 2n matrix
    
    # Compute derivative
    sigma_dot =  Omega @ (H_m + E) @ sigma - sigma @ (H_m + E.T) @ Omega + D
    return sigma_dot.flatten()

def Compute_Sigma_Open_Gaussian_Numerical(n_modes, sigma_0, H_m, E, D, t_array):
    # Flatten initial conditions
    y0 = sigma_0.flatten()
    
    # Time span and points
    t_span = (t_array[0], t_array[-1])
    
    # Solve the ODE system
    sigma_sol = solve_ivp(equation_dot_sigma, t_span, y0, t_eval=t_array, args=(H_m, E, D,n_modes))
    
    # Extract solutions
    sigma_t = sigma_sol.y.T.reshape(-1,2*n_modes, 2*n_modes)  # Reshape back to matrices
    return sigma_t

def Compute_Sigma_Open_Gaussian_Analytical(n_modes, sigma_0, H_m, E, D, t_array):
    # Flatten initial conditions
    Omega = Omega_N(n_modes) 
    
    def integrand(t):
        return np.exp(Omega@(H_m + E)*(3))
    
    # Time span and points
    t_span = (t_array[0], t_array[-1])
    
    # Solve the ODE system
    sigma_sol = solve_ivp(equation_dot_sigma, t_span, y0, t_eval=t_array, args=(H_m, E, D,n_modes))
    
    # Extract solutions
    sigma_t = sigma_sol.y.T.reshape(-1,2*n_modes, 2*n_modes)  # Reshape back to matrices
    return sigma_t


def equation_dot_r(t, y, H_m, r_m, E, n_modes):
    """
    Defines the ODE for r_jk when r_jk is a 2n x 2n matrix.
    """
    Omega = Omega_N(n_modes) 
    r = y.reshape((2*n_modes, 1))  # Reshape to 2n x 1 vector
    
    # Compute derivative
    r_dot = Omega @ (H_m + E) @ r - r_m
    
    return  r_dot.flatten()
    
def Compute_R_Open_Gaussian_Numerical(n_modes, r_0, H_m, r_m, E, t_array):

    y0 = r_0.flatten()
    
    # Time span and points
    t_span = (t_array[0], t_array[-1])
    
    # Solve for r_jk
    sol_r = solve_ivp(equation_dot_r, t_span, y0, t_eval=t_array, args=( H_m, r_m, E, n_modes))
    
    # Extract solutioon
    r_t = sol_r.y.T.reshape(-1, 2*n_modes, 1)  # Reshape back to vectors
   
    return r_t



def Open_Numerical(n_modes, r_0, sigma_0, H_m, r_m, E, D, t_array):
    """
    Compute the unitary time evolution of the first and second moments 
    Input:
    n_modes     Number of Modes
    r_0         Initial First Moment (2n, 1)
    sigma_0     Covariant Matrix at t=0 (2n x 2n)
    N_m         Hamiltonian Matrix (2n x 2n)
    r_m         Force (2n x 1)
    Output:  First and Second Moments at an array of time t (2n x 2n)
    """
    r_t = Compute_R_Open_Gaussian_Numerical(n_modes,r_0, H_m, r_m, E, t_array)
    sigma_t = Compute_Sigma_Open_Gaussian_Numerical(n_modes, sigma_0, H_m, E, D, t_array)
        
    return r_t, sigma_t

def Open_Dynamics_Analytical(n_modes, r_0, sigma_0, symplectic_func, H_m, r_m, t_array):

    """
    Compute the unitary time evolution of the first and second moments 
    Input:
    n_modes     Number of Modes
    r_0         Initial First Moment (2n x 1)
    sigma_0     Covariant Matrix at t=0 (2n x 2n)
    S_m_func    Analytical Function of S_m
    r_m         Force (2n x 1)
    Output:  First and Second Moments at an array of time t (2n x 2n)
    """
    
    r_t = np.zeros((len(t_array),  2*n_modes, 1))
    sigma_t = np.zeros((len(t_array), 2*n_modes, 2*n_modes))
    
    r_tilde_m = QIT_Functions.Compute_r_tilde(H_m, r_m)
    
    for i in range(len(t_array)):
        
        S_m_t = symplectic_func(t_array[i])
        r_t[i] = QIT_Functions.Compute_r_t_S_m(r_0, r_tilde_m, S_m_t)
        sigma_t[i] = QIT_Functions.Compute_sigma_t_S_m(sigma_0, S_m_t)
        
    return r_t, sigma_t









