import numpy as np 
import matplotlib.pyplot as plt
import math as math 
import cmath as cmath
from scipy import linalg as linalg
from scipy import integrate as integ
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import QIT_Functions as QIT_Functions 


########################################## Expectation Values ##########################################

def Qubit_Expecation_Value(operator, rho_q):
    return np.trace(operator@rho_q)

def Qubit_Expecation_Value_Time_Evolution(operator, rho_q_t, t_array):
    exp_t = np.zeros(len(t_array))
    for i in range(len(t_array)):
        exp_t[i] = Qubit_Expecation_Value(operator, rho_q_t[i])
    return exp_t

########################################## Ideal Measurements ##########################################

def Qubit_Ideal_Measurament(N_qubits, operator, rho_q):

        outcomes, eve = linalg.eig(operator)
        dim = len(outcomes)
        pointer = np.transpose(eve)
        
        POVM_array = np.zeros((dim, 2**(N_qubits), 2**(N_qubits)), dtype =np.complex64)
        prob_outcome = np.zeros(dim)
        rho_q_PMS_array = np.zeros((dim, 2**(N_qubits), 2**(N_qubits)), dtype =np.complex64)
        
        for i in range(dim):
            POVM_array[i] = np.outer(pointer[i], pointer[i])
            rho_q_PMS_unnormalize = POVM_array[i]@rho_q
            prob_outcome[i] = np.trace(rho_q_PMS_unnormalize)
            rho_q_PMS_array[i] = rho_q_PMS_unnormalize/prob_outcome[i] 
            
        return outcomes, prob_outcome, POVM_array, rho_q_PMS_array

def Qubit_Ideal_Measurament_Time_Evolution(N_qubits, operator, rho_q_t,t_array):

    outcomes, eve = linalg.eig(operator)
    dim = len(outcomes)
    pointer = np.transpose(eve)

    POVM_array = np.zeros((dim, 2**(N_qubits), 2**(N_qubits)), dtype =np.complex64)

    for i in range(dim):
        POVM_array[i] = np.outer(pointer[i], pointer[i])
    
    prob_t = np.zeros((dim,  len(t_array)))
    rho_q_t_PMS_array = np.zeros((dim,  len(t_array), 2**(N_qubits), 2**(N_qubits)), dtype =np.complex64)

    for j in range(len(t_array)):
        for i in range(dim):
            rho_q_PMS_unnormalize = POVM_array[i]@rho_q_t[j]
            prob_t[i,j] = np.trace(rho_q_PMS_unnormalize)
            rho_q_t_PMS_array[i,j] = rho_q_PMS_unnormalize/prob_t[i,j] 

    return outcomes, prob_t, POVM_array, rho_q_t_PMS_array

########################################## Noisy Measurements ##########################################

def Generadyne_Qubit_Post_Measurement_State(N_qubits, n_modes, sigma_jk, sigma_measure, r_jk, rho_q, r_measure_array):
    
    P_jk = np.zeros((2**(N_qubits), 2**(N_qubits), len(r_measure_array)), dtype = np.complex128)

    Prob_dis = np.zeros(len(r_measure_array), dtype = np.complex128)

    for k in range(len(r_measure_array)):
        r_measure = np.array([[r_measure_array[k]],[r_measure_array[k]]])
        for i in range(2**(N_qubits)):
            for j in range(2**(N_qubits)):
                inv = linalg.inv(sigma_jk[i,j]+ sigma_measure)
                det = linalg.det(sigma_jk[i,j]+ sigma_measure)
                P_jk[i,j,k] = rho_q[i,j]/(np.pi**(n_modes)*np.sqrt(det))*np.exp(-np.transpose(r_jk[i,j] - r_measure)@inv@(r_jk[i,j] - r_measure))[0,0]
        for i in range(2**N_qubits):
            Prob_dis[k] += P_jk[i,i,k]

    norm = np.sum(Prob_dis)

    return P_jk/norm, Prob_dis/norm

def Generadyne_Qubit_Post_Measurement_State_Time_Evolution(N_qubits, n_modes, sigma_jk_t, sigma_measure, r_jk_t, rho_q_t, r_measure_array, t_array):
    
    P_jk_t = np.zeros((len(t_array), 2**(N_qubits), 2**(N_qubits), len(r_measure_array)))
    Prob_dis_t = np.zeros((len(t_array), len(r_measure_array)))

    for i in range(len(t_array)):
        P_jk_t[i], Prob_dis_t[i] = Generadyne_Qubit_Post_Measurement_State(N_qubits, n_modes, sigma_jk_t[i], sigma_measure, r_jk_t[i], rho_q_t[i], r_measure_array)
    
    return P_jk_t,Prob_dis_t

def CM_Homodyne_Noisy_1Mode(n_modes, mode_number, axis_angle, efficency):

    z = 10**(-5)

    efficency_angle = np.arccos(np.sqrt(efficency))
    sigma_m = 1/z**2*np.eye(2*n_modes)
    
    rotation_mat_1_mode = np.array([[np.cos(axis_angle), - np.sin(axis_angle)],
                             [np.sin(axis_angle), np.cos(axis_angle)]])
    
    for i in range(n_modes):
        if i == mode_number:
            sigma_homo = z**2*np.eye(2)
            sigma_homo[1,1] = 1/z**2 
                        
            sigma_m_1mode = np.transpose(rotation_mat_1_mode)@sigma_homo@rotation_mat_1_mode + np.tan(efficency_angle)**2*np.eye(2)

            sigma_m[2*i,2*i] = sigma_m_1mode[0,0]
            sigma_m[2*i,2*i+1] = sigma_m_1mode[0,1]
            sigma_m[2*i+1,2*i] = sigma_m_1mode[1,0]
            sigma_m[2*i+1,2*i+1] = sigma_m_1mode[1,1]

        else:
            pass

    return sigma_m

def CM_Heterodyne_Noisy_1Mode(n_modes, mode_number, axis_angle, efficency):
    
    efficency_angle = np.sqrt(np.arccos(efficency))

    sigma_m = np.zeros((2*n_modes, 2*n_modes))
    heterodyne = (1  + 2*np.tan(efficency_angle)**2)*np.eye(2,2)
    for i in range(n_modes):
        if i == mode_number:
            sigma_m[2*i,2*i] = heterodyne[0,0]
            sigma_m[2*i,2*i+1] = heterodyne[0,1]
            sigma_m[2*i+1,2*i] = heterodyne[1,0]
            sigma_m[2*i+1,2*i+1] = heterodyne[1,1]
        else:
            pass
    return sigma_m


def Homodyne_Qubit_Post_Measurement_State(N_qubits, n_modes, sigma_jk, sigma_measure, r_jk, rho_q, r_measure_array, mode_number, quadrature):
    
    P_jk = np.zeros((2**(N_qubits), 2**(N_qubits), len(r_measure_array)), dtype = np.complex128)
    Prob_dis = np.zeros(len(r_measure_array), dtype = np.complex128)

    for k in range(len(r_measure_array)):
        r_measure = np.array([[r_measure_array[k]],[r_measure_array[k]]])
        for i in range(2**(N_qubits)):
            for j in range(2**(N_qubits)):
                inv = linalg.inv(sigma_jk[i,j]+ sigma_measure)
                if quadrature == 'Position':
                    det =(sigma_jk[i,j]+ sigma_measure)[2*mode_number,2*mode_number]
                elif quadrature == 'Momentum':
                    det =(sigma_jk[i,j]+ sigma_measure)[2*mode_number+1,2*mode_number+1]
                else:
                    pass
                print(det)
                P_jk[i,j,k] = rho_q[i,j]/(np.pi**(n_modes)*np.sqrt(det))*np.exp(-np.transpose(r_jk[i,j] - r_measure)@inv@(r_jk[i,j] - r_measure))[0,0]
        
        for i in range(2**N_qubits):
            Prob_dis[k] += P_jk[i,i,k]

    norm = np.sum(Prob_dis)

    return P_jk/norm, Prob_dis/norm


def Homodyne_Qubit_Post_Measurement_State_Time_Evolution(N_qubits, n_modes, sigma_jk_t, sigma_measure, r_jk_t, rho_q_t, r_measure_array, t_array, mode_number, quadrature):
    
    P_jk_t = np.zeros((len(t_array), 2**(N_qubits), 2**(N_qubits), len(r_measure_array)))
    Prob_dis_t = np.zeros((len(t_array), len(r_measure_array)))

    for i in range(len(t_array)):
        P_jk_t[i], Prob_dis_t[i] = Homodyne_Qubit_Post_Measurement_State(N_qubits, n_modes, sigma_jk_t[i], sigma_measure, r_jk_t[i], rho_q_t[i], r_measure_array, mode_number, quadrature)

    return P_jk_t,Prob_dis_t
########################################## Entanlgement Measurements ##########################################


def Partial_Transpose_Dense(rho, mask, dimensions_list):
        """Computes the Partial Transposition"""

        dims = [dimensions_list, dimensions_list]
        nsys = len(mask)
        pt_dims = np.arange(2 * nsys).reshape(2, nsys).T
        pt_idx = np.concatenate([[pt_dims[n, mask[n]] for n in range(nsys)],
                                [pt_dims[n, 1 - mask[n]] for n in range(nsys)]])

        partial_transpose = rho.reshape(
            np.array(dims).flatten()).transpose(pt_idx).reshape(rho.shape)

        return partial_transpose

def Negativity_Qubits(rho):
        """Compute the witness (PPT) both as value (given the state) and as operator """
        J = (len(rho)-1)/4 
        dim = int(2*J+1)
        dimensions_list = [dim,dim]
        mask = [0,1]
        
        rho_pt = Partial_Transpose_Dense(rho, mask, dimensions_list)
        
        eva, eve = linalg.eig(rho_pt)
        eve = np.transpose(eve)
        
        nega = 0
        witness_matrix = np.zeros((dim**2, dim**2),dtype =np.complex128)
        
        for i in range(len(eva)):
            num = 0
            if eva[i] < 0:
                nega += eva[i]
                witness_matrix += np.outer(eve[i], np.conj(eve[i].T))
                num += 1
        
        witness_matrix = Partial_Transpose_Dense(witness_matrix, mask, dimensions_list)
        return nega, witness_matrix


def Negativity_Qubits_Time(rho_t, t_array):
        """Compute the witness (PPT) both as value (given the state) and as operator """
        J = (len(rho_t[0])-1)/4 
        dim = int(2*J+1)
        dimensions_list = [dim,dim]
        mask = [0,1]
        witness_matrix = np.zeros((len(t_array), dim**2, dim**2),dtype =np.complex128)
        nega_array = np.zeros(len(t_array))
        
        for j in range(len(t_array)):
            
            rho_pt = Partial_Transpose_Dense(rho_t[j], mask, dimensions_list)
            
            eva, eve = linalg.eig(rho_pt)
            eve = np.transpose(eve)
            
            nega = 0
            
            
            for i in range(len(eva)):
                num = 0
                if eva[i] < 0:
                    nega_array[j] += eva[i]
                    witness_matrix[j] += np.outer(eve[i], np.conj(eve[i].T))
                    num += 1
            
            witness_matrix[j] = Partial_Transpose_Dense(witness_matrix[j], mask, dimensions_list)
        return nega_array, witness_matrix
