import numpy as np 
import matplotlib.pyplot as plt
import math as math 
import cmath as cmath
from scipy import linalg as linalg
from scipy import integrate as integ
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import QIT_Functions as QIT_Functions



def Symplectic_QHO_1(n_modes, omega):
    
    def Symplectic_t(t):
        s_m_t = np.array([
                    [np.cos(omega*t), np.sin(omega*t)],
                    [-np.sin(omega*t),  np.cos(omega*t)]
                ])
        return s_m_t
    
    return Symplectic_t
    

def Symplectic_QHO_N(n_modes, omegas_array):

    def Symplectic_QHO_t_1(omega, t):
        s_m_t = np.array([
                    [np.cos(omega*t), np.sin(omega*t)],
                    [-np.sin(omega*t),  np.cos(omega*t)]
                ])
    
    def Symplectic_QHO_t(t):
        s_m_t_1 = np.zeros((2*n_modes, 2*n_modes))
        vec_s_t = np.zeros((n_modes, ))

        for i in range(n_modes):
            vec_s_t[i] = Symplectic_QHO_t_1(omegas_array[i], t)

        s_m_t = np.block([[mat if i == j else np.zeros_like(mat) for j, mat in enumerate(vec_s_t)] for i, mat in enumerate(vec_s_t) ])
        
        return s_m_t
    
    return Symplectic_QHO_t

def Symplectic_2QHO_XX(g):
    
    def S_Hg_inv(t):
        omega_g = np.sqrt(1 - 2*g)
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        cos_og_t = np.cos(omega_g * t)
        sin_og_t = np.sin(omega_g * t)

        return 0.5 * np.array([
            [cos_t + cos_og_t,       sin_t + (1/omega_g) * sin_og_t,  cos_t - cos_og_t,       sin_t - (1/omega_g) * sin_og_t],
            [-sin_t - omega_g * sin_og_t, cos_t + cos_og_t,           -sin_t + omega_g * sin_og_t, cos_t - cos_og_t],
            [cos_t - cos_og_t,       sin_t - (1/omega_g) * sin_og_t,  cos_t + cos_og_t,       sin_t + (1/omega_g) * sin_og_t],
            [-sin_t + omega_g * sin_og_t, cos_t - cos_og_t,           -sin_t - omega_g * sin_og_t, cos_t + cos_og_t]
        ])
    
    return S_Hg_inv

