import numpy as np 
import matplotlib.pyplot as plt
import math as math 
import cmath as cmath
from scipy import linalg as linalg


from matplotlib import cm
from matplotlib import rcParams
from matplotlib import colors


import Gaussian as Gaussian
import Unitary_Operator_Force as Unitary_Operator_Force
import Unitary_Operator_Gaussian as Unitary_Operator_Gaussian
import Unitary_Operator_Time_Depentent as Unitary_Operator_Time_Depentent

import Open_Operator_Force as Open_Operator_Force
import Open_Operator_Gaussian as Open_Operator_Gaussian
import Open_Operator_Time_Depentent as Open_Operator_Time_Depentent

import Symplectic_Known as Symplectic_Known
import Measurements as Measurements
import Plots_Functions as Plots_Functions
import Wigner_Functions as Wigner_Functions


class Hamiltonian:
    
    def __init__(self, N_qubits, n_modes):

        # Parameters for Initiation
        self.N_qubits = N_qubits
        self.n_modes = n_modes
        #self.evolution_type = evolution_type # Gaussian, Force, General
        #self.sim_type = sim_type # Numerical or Analytical

    def Initialize_Constant_Hamiltonians(self, H_array, r_array, H_q_0_array):
        """
        H_array       Array of 1 to N+1 Matrix Hamiltonians (2nx2n)
        r_array       Array of 0 to N+1 Force Vectors (2nx1)
        H_q_0_array   Array of free evolution of the qubit (2Nx1)
        """
        
        self.H_array = H_array
        self.r_array = r_array
        self.H_q_0_array = H_q_0_array
        
        if len(H_array) == 1:
            
            if len(r_array) == 1:
                self.type = ['Constant','Gaussian']
                print('This is a time independent gaussian Hamiltonian')
            else:
                self.type = ['Constant','Force']           
                print('This is a time independent operator-valued force Hamiltonian')
        else:
            self.type = ['Constant','General']
            print('This is a time independent operator-valued general Hamiltonian')
        return 

    def Initialize_Time_Dep_Hamiltonian(self, H_array_func, r_array_func, H_q_0_array_func, t_array):
        """
        H_array_func       Function which returns an array of 1 to N+1 Matrix Hamiltonians (2nx2n) at each t
        r_array_func       Function which returns an array of 0 to N+1 Force Vectors (2nx1) at each t 
        H_q_0_array_func   Function which returns an array of free evolution of the qubit (2Nx1) at each t
        """

        self.H_array_func = H_array_func
        self.r_array_func = r_array_func
        self.H_q_0_array_func = H_q_0_array_func
        
        if len(H_array_func) == 1:
            
            if len(r_array_func) == 1:
                self.type = ['Time','Gaussian']
                print('This is a time dependent gaussian Hamiltonian')
            else:
                self.type = ['Time','Force']    
                print('This is a time dependent operator-valued force Hamiltonian')
        else:
            self.type = ['Time','General']
            print('This is a time dependent operator-valued general Hamiltonian')

        H_array_t = np.zeros((len(H_array_func), len(t_array), 2*self.n_modes, 2*self.n_modes))
        r_array_t = np.zeros((len(r_array_func), len(t_array), 2*self.n_modes, 1))
        H_q_0_array_t = np.zeros((len(H_q_0_array_func), len(t_array)))
        
        for j in range(len(H_array_func)):
            for i in range(len(t_array)):
                H_array_t[j,i] = H_array_func[j](t_array[i])
                r_array_t[j,i] = r_array_func[j](t_array[i])
        
        for j in range(len(H_q_0_array_t)):
            for i in range(len(t_array)):
                H_q_0_array_t[j,i] = H_q_0_array_func[j](t_array[i])

        self.H_array_t = H_array_t
        self.r_array_t = r_array_t
        self.H_q_0_array_t = H_q_0_array_t
        
        return
    
class Symplectic:
    
    def __init__(self, N_qubits, n_modes):

        # Parameters for Initiation
        self.N_qubits = N_qubits
        self.n_modes = n_modes
        #self.evolution_type = evolution_type # Gaussian, Force, General
        #self.sim_type = sim_type # Numerical or Analytical

    def Initialize_Constant_Symplectic(self, symplectic_name = False,  symplectic_parameters = False, symplectic_func = False):
        
        self.type = 'Constant'

        if symplectic_name == False:
            self.symplectic_transformation = symplectic_func
        else:
            symplectic_transformation_known = {
                "QHO_1": Symplectic_Known.Symplectic_QHO_1,
                "QHO_n": Symplectic_Known.Symplectic_QHO_N,
                "QHO_2_XX": Symplectic_Known.Symplectic_2QHO_XX}
            self.symplectic_transformation = symplectic_transformation_known[symplectic_name](self.n_modes, symplectic_parameters)
        
        return 

class Decoherence:

    def __init__(self, N_qubits, n_modes, basis, B_matrix):

        # Parameters for Initiation
        self.N_qubits = N_qubits
        self.n_modes = n_modes

        if basis == 'Canonical':
            self.E_mat, self.D_mat = Gaussian.Get_Decoherence_Rates(n_modes, B_matrix)

        elif basis == 'Ladder':
            U_N = Gaussian.U_N(n_modes) 
            self.E_mat, self.D_mat = Gaussian.Get_Decoherence_Rates(n_modes,  U_N.T@B_matrix@U_N)
        
        else:
            print('Input Valid Basis NAme: Canonical or Ladder')

class Quantum_State:

################################## Initialization ################################## 

    def __init__(self, N_qubits, n_modes):
        self.N_qubits = N_qubits
        self.n_modes = n_modes
        self.sigma_JK_t_PMS = None
        self.Measurament = None

    def Initialize_Gaussian_State(self, r_0, sigma_0,  rho_q_0):
        
        print('Initializing a Gaussian State')
        
        self.sigma_0 = sigma_0.astype(complex)
        self.r_0 = r_0.astype(complex)
        self.rho_q_0 = rho_q_0.astype(complex)
        self.type = 'Gaussian' # Gaussian or Cat
        
        return

    def Start_Dynamics_Unitary(self):
        
        self.r_t = 0 
        self.sigma_t = 0 
        self.sigma_JK_t = 0
        self.r_JK_t = 0 
        self.rho_q_t = 0
        self.r_JK_0_t = 0
        self.C_JK_t = 0
        self.phi_JK_t = 0 
        
        return 

    def Start_Dynamics_Noise(self):
        
        self.r_t_D = 0 
        self.sigma_t_D = 0 
        self.sigma_JK_t_D = 0
        self.r_JK_t_D = 0 
        self.rho_q_t_D = 0
        self.r_JK_0_t_D = 0
        self.C_JK_t_D = 0
        self.phi_JK_t_D = 0 
        
        return 

################################## Dynamics ################################## 

    def Unitary_Dynamics_Numerical(self, Hamiltonian, t_array):
        """Given an Hamiltonian, Computes the unitary Dyanmics Numerically:
        1. Operator Valued Gaussian -> Solves the ODEs of Table 1
        2. Operator Valued Force -> Numerically Exponentiate the Symplectic Transformation for Table 2
        """

        self.t_array = t_array

        self.Start_Dynamics_Unitary()
        
        if Hamiltonian.type[0] == 'Constant':
            if Hamiltonian.type[1] == 'Gaussian':
                self.r_t, self.sigma_t = Gaussian.Unitary_Numerical(self.n_modes, self.r_0, self.sigma_0, 
                                                                        Hamiltonian.H_array[0], Hamiltonian.r_array[0], t_array)
    
            elif Hamiltonian.type[1] == 'Force':
                self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t  = Unitary_Operator_Force.Dynamics_Numerical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0,  Hamiltonian.H_array[0], Hamiltonian.r_array, Hamiltonian.H_q_0_array, t_array)
                self.r_JK_0_t =  - self.C_JK_t + 1j*self.phi_JK_t
                self.rho_q_t =  np.exp(- self.C_JK_t + 1j*self.phi_JK_t)*self.rho_q_0
                
            elif Hamiltonian.type[1] == 'General':
                self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t = Unitary_Operator_Gaussian.Dynamics_Numerical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0, 
                                                                                                    Hamiltonian.H_array, Hamiltonian.r_array, Hamiltonian.H_q_0_array, self.rho_q_0, t_array)
                self.r_JK_0_t =  - self.C_JK_t + 1j*self.phi_JK_t
                self.rho_q_t =  np.exp(- self.C_JK_t + 1j*self.phi_JK_t)*self.rho_q_0
                    
    
        elif Hamiltonian.type[0] == 'Time':
            self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t  = Unitary_Operator_Time_Depentent.Dynamics_Numerical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0, Hamiltonian.H_array_func, 
                                                                                                            Hamiltonian.r_array_func, Hamiltonian.H_q_0_array_func, self.rho_q_0, self.t_array)
            self.r_JK_0_t =  - self.C_JK_t + 1j*self.phi_JK_t
            self.rho_q_t =  np.exp(- self.C_JK_t + 1j*self.phi_JK_t)*self.rho_q_0
        
        return 
    
    def Unitary_Dynamics_Analytical(self, Hamiltonian, Symplectic, t_array):
        
        self.t_array = t_array
        self.Start_Dynamics_Unitary()
        
        if Symplectic.type == 'Constant':
            if Hamiltonian.type[1] == 'Gaussian':
                self.r_t, self.sigma_t = Gaussian.Unitary_Analytical(self.n_modes, self.r_0, self.sigma_0, 
                                                                                            Symplectic.symplectic_transformation, 
                                                                                            Hamiltonian.H_array[0], Hamiltonian.r_array[0],  t_array)
            elif Hamiltonian.type[1] == 'Force':
                self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t  = Unitary_Operator_Force.Dynamics_Analytical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0, 
                                                                                                                    Symplectic.symplectic_transformation,
                                                                                                                    Hamiltonian.H_array[0], Hamiltonian.r_array, Hamiltonian.H_q_0_array, t_array)
                self.r_JK_0_t =  - self.C_JK_t + 1j*self.phi_JK_t
                self.rho_q_t =  np.exp(- self.C_JK_t + 1j*self.phi_JK_t)*self.rho_q_0
    
        elif Symplectic.type == 'Time':
            self.sigma_JK_t, self.r_JK_t, self.r_JK_0_t = Unitary_Operator_Time_Depentent.Dynamics_Analytical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0, Hamiltonian.H_array_func, 
                                                                                                               Hamiltonian.r_array_func, Hamiltonian.H_q_0_array_func, self.rho_q_0, self.t_array)
        
        return 
    
    def Open_Dynamics_Numerical(self, Hamiltonian, Decoherence, t_array):

        self.t_array = t_array

        E = Decoherence.E_mat
        D = Decoherence.D_mat 

        self.Start_Dynamics_Noise()
        
        if Hamiltonian.type[0] == 'Constant':
            if Hamiltonian.type[1] == 'Gaussian':
                print('Computing the Gaussian Time-Indipendent Dynamics of a Gaussian State')
                self.r_t, self.sigma_t = Gaussian.Open_Numerical(self.n_modes, self.r_0, self.sigma_0, Hamiltonian.H_array[0], Hamiltonian.r_array[0], E, D, t_array)
                    

    
            elif Hamiltonian.type[1] == 'Force':
                if E.all() == 0.0:
                    self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t  = Open_Operator_Force.Dynamics_Symetric_Numerical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0,  Hamiltonian.H_array[0], Hamiltonian.r_array, Hamiltonian.H_q_0_array, D, t_array)
                    self.r_JK_0_t =  - self.C_JK_t + 1j*self.phi_JK_t
                    self.rho_q_t =  np.exp(- self.C_JK_t + 1j*self.phi_JK_t)*self.rho_q_0
                else: 
                    self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t  = Open_Operator_Force.Dynamics_General_Numerical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0,  Hamiltonian.H_array[0], Hamiltonian.r_array, Hamiltonian.H_q_0_array, E, D, t_array)
                    self.r_JK_0_t =  - self.C_JK_t + 1j*self.phi_JK_t
                    self.rho_q_t =  np.exp(- self.C_JK_t + 1j*self.phi_JK_t)*self.rho_q_0
                
                
            elif Hamiltonian.type[1] == 'General':
            
                self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t = Open_Operator_Gaussian.Dynamics_Numerical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0, 
                                                                                                Hamiltonian.H_array, Hamiltonian.r_array, Hamiltonian.H_q_0_array, self.rho_q_0,E, D, t_array)
                self.r_JK_0_t =  - self.C_JK_t + 1j*self.phi_JK_t
                self.rho_q_t =  np.exp(- self.C_JK_t + 1j*self.phi_JK_t)*self.rho_q_0
                
    
        elif Hamiltonian.type[0] == 'Time':
            self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t  = Open_Operator_Time_Depentent.Dynamics_Numerical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0, Hamiltonian.H_array_func, 
                                                                                                            Hamiltonian.r_array_func, Hamiltonian.H_q_0_array_func, self.rho_q_0,E, D, self.t_array)
            self.r_JK_0_t =  - self.C_JK_t + 1j*self.phi_JK_t
            self.rho_q_t =  np.exp(- self.C_JK_t + 1j*self.phi_JK_t)*self.rho_q_0
            
        return 

    def Open_Dynamics_Analytical(self, Hamiltonian, Symplectic, t_array):
        self.t_array = t_array

        self.Start_Dynamics_Unitary()
        
        if Symplectic.type == 'Constant':
            if Hamiltonian.type[1] == 'Gaussian':
                if self.type == 'Gaussian':
                    print('Computing the Gaussian Time-Indipendent Dynamics of a Gaussian State Semi-Analytically')
                    self.r_t, self.sigma_t = Gaussian.Unitary_Analytical(self.n_modes, self.r_0, self.sigma_0, 
                                                                                                Symplectic.symplectic_transformation, 
                                                                                                Hamiltonian.H_array[0], Hamiltonian.r_array[0],  t_array)
                    
                elif self.type == 'Cat':
                    print('Computing the Gaussian Time-Indipendent Dynamics of a Cat State')
                    print('to do')
                    
                else:
                    print('Please Initialise the State')
    
            elif Hamiltonian.type[1] == 'Force':
                if self.type == 'Gaussian':
                    print('Computing the Operator-Valued Force Time-Indipendent Dynamics of a Gaussian State')
                    self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t  = Unitary_Operator_Force.Unitary_Analytical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0, 
                                                                                                                    Symplectic.symplectic_transformation,
                                                                                                                    Hamiltonian.H_array[0], Hamiltonian.r_array, Hamiltonian.H_q_0_array, t_array)
                    self.r_JK_0_t =  - self.C_JK_t + 1j*self.phi_JK_t
                    self.rho_q_t =  np.exp(- self.C_JK_t + 1j*self.phi_JK_t)*self.rho_q_0
                    
                elif Quantum_State.type == 'Cat':
                    print('Computing the Operator-Valued Force Time-Indipendent Dynamics of a Cat State')
                    
                else:
                    print('Please Initialise the State')
    
        elif Symplectic.type == 'Time':
            if self.type == 'Gaussian':
                print('Computing the Genearl Opeator-Valued Time-Dependent Unitary Dynamics of a Gaussian State')
    
                self.sigma_JK_t, self.r_JK_t, self.r_JK_0_t = Unitary_Operator_Time_Depentent.Unitary_Gaussian_General_Time_Dep_Numerical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0, Hamiltonian.H_array_func, 
                                                                                                               Hamiltonian.r_array_func, Hamiltonian.H_q_0_array_func, self.rho_q_0, self.t_array)
                self.rho_q_t = np.exp(self.r_JK_0_t)
                #self.sigma_JK_t, , self.C_JK_t, self.phi_JK_t = Unitary_Gaussian_General_Numerical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0, self.H_array, self.r_array, self.H_q_0_array, t_array)
                    
            elif self.type == 'Cat':
                print('Computing the Genearl Opeator Time-Dependent Dynamics of a Cat State')
                
            else:
                print('Please Initialise the State')
        
        return 

################################## Measuraments ################################## 

    def Qubit_Expectation_Values(self, operator,type_exp):
        if type_exp == 'Final':
            self.exp = Measurements.Qubit_Expecation_Value(operator, self.rho_q_t[-1])
        elif type_exp == 'Time':
            print('hi')
            self.exp_t = Measurements.Qubit_Expecation_Value_Time_Evolution(operator, self.rho_q_t, self.t_array)
        else:
            print('Input Type')
        return 

    def Qubit_Ideal_Measurament(self, operator, type_measure):
        """Compute a qubit measurament
        If type == Ideal, parameter == index time 
        """
        self.Measurament = 'qubit'

        if type_measure == 'Final':
            self.outcomes, self.prob, self.POVM_array, self.rho_q_PMS_array = Measurements.Qubit_Ideal_Measurament(self.N_qubits, operator, self.rho_q_t[-1])
        
        elif type_measure == 'Time':
            self.outcomes, self.prob_t, self.POVM_array, self.rho_q_t_PMS_array =  Measurements.Qubit_Ideal_Measurament_Time_Evolution(self.N_qubits, operator,  self.rho_q_t,self.t_array)
        else:
            print('Input Type')
        return 
    
    def Negativity_Qubit(self, type_nega):
        if type_nega == 'Final':
            self.nega, self.witness_matrix = Measurements.Negativity_Qubits(self.rho_q_t[-1])
        elif type_nega == 'Time':
            self.nega_t, self.witness_matrix_t = Measurements.Negativity_Qubits_Time(self.rho_q_t, self.t_array)
        else:
            print('Input Type')
        return 
    
    def Mode_Measurament(self, type_measure, mode_number, axis_angle, efficency, r_measure_array):
        """Compute a qubit measurament
        If type == Ideal, parameter == index time 
        """
        print(type_measure, 'Measurement')
        if type_measure[0] == 'Final':
            if type_measure[1] == 'Heterodyne':
                sigma_measure = Measurements.CM_Heterodyne_Noisy_1Mode(self.n_modes, mode_number, axis_angle, efficency)
                self.Qubit_PMS, self.Prob_Dis = Measurements.Generadyne_Qubit_Post_Measurement_State(self.N_qubits, self.n_modes, self.sigma_JK_t[-1], sigma_measure, self.r_JK_t[-1], self.rho_q_t[-1], r_measure_array)
            elif type_measure[1] == 'Homodyne':
                sigma_measure = Measurements.CM_Homodyne_Noisy_1Mode(self.n_modes, mode_number, axis_angle, efficency)
                self.Qubit_PMS, self.Prob_Dis = Measurements.Generadyne_Qubit_Post_Measurement_State(self.N_qubits, self.n_modes, self.sigma_JK_t[-1], sigma_measure, self.r_JK_t[-1], self.rho_q_t[-1], r_measure_array)

        elif type_measure[0] == 'Time':
            if type_measure[1] == 'Heterodyne':
                sigma_measure = Measurements.CM_Heterodyne_Noisy_1Mode(self.n_modes, mode_number, axis_angle, efficency)
                self.Qubit_PMS_t, self.Prob_Dis_t = Measurements.Generadyne_Qubit_Post_Measurement_State_Time_Evolution(self.N_qubits, self.n_modes, self.sigma_JK_t, sigma_measure, self.r_JK_t, self.rho_q_t, r_measure_array, self.t_array)
            elif type_measure[1] == 'Homodyne':
                sigma_measure = Measurements.CM_Homodyne_Noisy_1Mode(self.n_modes, mode_number, axis_angle, efficency)
                self.Qubit_PMS_t, self.Prob_Dis_t = Measurements.Generadyne_Qubit_Post_Measurement_State_Time_Evolution(self.N_qubits, self.n_modes, self.sigma_JK_t, sigma_measure, self.r_JK_t, self.rho_q_t, r_measure_array, self.t_array)
        else:
            print('Input Type')
        return 
    
    def Homdyne_Mode_Measurament(self, type_measure, mode_number, axis_angle, quadrature, efficency, r_measure_array):
        """Compute a qubit measurament
        If type == Ideal, parameter == index time 
        """
        print(type_measure, 'Measurement')
        if type_measure == 'Final':
            sigma_measure = Measurements.CM_Heterodyne_Noisy_1Mode(self.n_modes, mode_number, axis_angle, efficency)
            self.Qubit_PMS, self.Prob_Dis = Measurements.Homodyne_Qubit_Post_Measurement_State(self.N_qubits, self.n_modes, self.sigma_JK_t[-1], sigma_measure, self.r_JK_t[-1], self.rho_q_t[-1], r_measure_array, mode_number, quadrature)
        elif type_measure == 'Time':
            sigma_measure = Measurements.CM_Heterodyne_Noisy_1Mode(self.n_modes, mode_number, axis_angle, efficency)
            self.Qubit_PMS_t, self.Prob_Dis_t = Measurements.Homodyne_Qubit_Post_Measurement_State_Time_Evolution(self.N_qubits, self.n_modes, self.sigma_JK_t, sigma_measure, self.r_JK_t, self.rho_q_t, r_measure_array, self.t_array, mode_number, quadrature)
        else:
            print('Input Type')
        return 
    
################################## Plot ################################## 

    def Plot_Phase_Space_First_QRDM(self, leged_lables, array_pi,save):
        Plots_Functions.Plot_Phase_Space_First_QRDM_Func(self, leged_lables, array_pi ,save)

    def Plot_Wigner_Function_Diag(self, steps, time_index, mode_number, sigma_para, array_tick, save):
            
        r_tilde_array, x_array, p_array, XX, PP = Wigner_Functions.Create_r_tilde_array(self,sigma_para, mode_number, steps)
        self.wigner_diag = Wigner_Functions.Wigner_Sum_Diagonal(self, time_index, r_tilde_array)
        bar_limits = [0, np.max(self.wigner_diag)]
        fig = Plots_Functions.Plot_Wigner_Diag(self.wigner_diag,  x_array, p_array, bar_limits, array_tick, save) 
        return

    def Plot_Wigner_4_Times(self, steps, time_index_array, mode_number, sigma_para, array_tick, save):
            
        r_tilde_array, x_array, p_array, XX, PP = Wigner_Functions.Create_r_tilde_array(self,sigma_para, mode_number, steps)
        wigner_array = np.zeros((4,len(x_array),len(p_array)))

        for i in range(len(time_index_array)):
            wigner_array[i] = Wigner_Functions.Wigner_Sum_Diagonal(self, time_index_array[i], r_tilde_array)
        
        fig = Plots_Functions.Plot_Wigner_4_Times(wigner_array, x_array, p_array, save)
        return

    def Plot_Wigner_Function_PMS(self, steps, time_index, mode_number, sigma_para, array_tick, save):
        r_tilde_array, x_array, p_array, XX, PP = Wigner_Functions.Create_r_tilde_array(self,sigma_para, mode_number, steps)
        self.wigner_PMS = Wigner_Functions.Wigner_PMS_func(self, time_index, r_tilde_array)

        bar_limits = [-np.max(self.wigner_PMS), np.max(self.wigner_PMS)]
        fig = Plots_Functions.Plot_Wigner_Fringes(self.wigner_PMS,  x_array, p_array, bar_limits,array_tick, save) 
        return
    
    def Plot_Wigner_Function_Gauss(self, steps, time_index, mode_number, sigma_para, array_tick, save):
            
        r_tilde_array, x_array, p_array, XX, PP = Wigner_Functions.Create_r_tilde_array(self,sigma_para, mode_number, steps)
        self.wigner_gauss = Wigner_Functions.Wigner_Gaussian(self, time_index, r_tilde_array)
        bar_limits = [0, np.max(self.wigner_gauss)]
        fig = Plots_Functions.Plot_Wigner_Diag(self.wigner_gauss,  x_array, p_array, bar_limits, array_tick, save) 
        return

################################## Animation ################################## 

    def Animate_Wigner_Function_Diag(self, steps, mode_number, sigma_para, array_tick, n_frames, save):
            
        r_tilde_array, x_array, p_array, XX, PP = Wigner_Functions.Create_r_tilde_array(self,sigma_para, mode_number, steps)
        self.wigner_diag_t = Wigner_Functions.Wigner_t_Sum_Diag(self, r_tilde_array)
        bar_limits = [0, np.max(self.wigner_diag_t)]
        fig = Plots_Functions.Animate_Diagonal(self.wigner_diag_t,  x_array, p_array, bar_limits, array_tick, n_frames, save)
        return

    def Animate_Wigner_Function_PMS(self, steps, mode_number, sigma_para, array_tick, n_frames, save):
            
        r_tilde_array, x_array, p_array, XX, PP = Wigner_Functions.Create_r_tilde_array(self,sigma_para, mode_number, steps)
        self.wigner_PMS_t = Wigner_Functions.Wigner_t_PMS(self, r_tilde_array)
        bar_limits = [-np.max(self.wigner_PMS_t), np.max(self.wigner_PMS_t)]
        print(np.shape(self.wigner_PMS_t))
        fig = Plots_Functions.Animate_Fringes(self.wigner_PMS_t,  x_array, p_array, bar_limits, array_tick, n_frames, save)
        return
    
    def Animate_Gaussian(self, steps, mode_number, sigma_para, array_tick, n_frames, save):
            
        r_tilde_array, x_array, p_array, XX, PP = Wigner_Functions.Create_r_tilde_array(self,sigma_para, mode_number, steps)
        self.wigner_gauss_t = Wigner_Functions.Wigner_t_Gauss(self, r_tilde_array)
        bar_limits = [0, np.max(self.wigner_gauss_t)]
        fig = Plots_Functions.Animate_Diagonal(self.wigner_gauss_t,  x_array, p_array, bar_limits, array_tick, n_frames, save)
        return
        
