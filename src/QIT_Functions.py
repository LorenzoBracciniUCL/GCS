import numpy as np 
import matplotlib.pyplot as plt
import math as math 
import cmath as cmath
from scipy import linalg as linalg
from scipy import integrate as integ
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def Compute_r_tilde(H_m, r_m):
    return np.linalg.inv(H_m)@r_m


def Extract_Qubit_Labels_Array_old(N_qubits, index):
    """
    From the indecex of the 2^n x 2^n density matrix to two arrays of label of the qubits
    """
    index = index 
    J_labels = np.zeros(N_qubits).astype(float)

    ranges = np.zeros(N_qubits)


    if N_qubits ==1:
        ranges[0] = 0
    else:
        for i in range(N_qubits):
            ranges[i] = 2**(N_qubits - 1 - i) - 1

    for i in range(N_qubits):
        if index > np.sum(ranges*J_labels) + ranges[i] + 0.1:
            J_labels[i] = 1.0
        else:
            J_labels[i] = -1.0
    return J_labels 


def Extract_Qubit_Labels_Array(N_qubits, index):
    """
    From the indecex of the 2^n x 2^n density matrix to two arrays of label of the qubits
    """
    number = format(index,'0{0}b'.format(N_qubits))
    array = 2*np.array([float(digit) for digit in str(number)]) - 1
    return array