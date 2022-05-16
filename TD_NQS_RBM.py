from multiprocessing.sharedctypes import Value
from unittest import result
import numpy as np
import os, sys
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import timeit

from NQS_RBM import *

class TD_NQS_RBM(NQS_RBM):
    '''
    Defines a Time Dependent (TD) simulation of an NQS RBM model
    '''
    def __init__(self, init_H, Nv,Nh, init_mode="ground_state", init_state_params={"kContrastDiv": 6000, "lrate": 0.4, "epochs": 75}):
        
        #Initialize a static NQS model
        super().__init__(init_H, Nv,Nh) 
        
        #Prepare the initial state of the system
        if init_mode=="ground_state":
            #
            #Prepare initial state in ground state of initial hamiltonian
            #
            
            #Try to load it from a saved file from an earlier run to save time
            #otherwise (re)compute it on the fly
            
            sampler = 'MetroSmoothed'
            k_samples = init_state_params["kContrastDiv"]
            epochs = init_state_params["epochs"]
            
            try:
                WORKDIR_PATH = sys.path[0]
                DATA_PATH = WORKDIR_PATH + "/GS_archive/" 
                filename = f'NQSdata_J{self.hamilt.J:01}_h{self.hamilt.h:01}_{sampler}_Cycles{k_samples}_Epochs{epochs}.pickle'
                
                with open(DATA_PATH + filename,'rb') as f:
                    
                    results = pickle.load(f) #(Convergence, Percentage, aRBM, cRBM, WRBM, E_exact_per_site)
                    
                    self.weights['a'] = results[2] 
                    self.weights['c'] = results[3]
                    self.weights['W'] = results[4]
                    
                    gs_energy_convergence = results[0]
                    gs_convergence_percentage = results[1]
                    E_exact_per_site = results[-1] #Current exact energy value
                    
            except:
                print("File with RBM Ground state not found or opening failed")
                print("(Re)Computing ground state...")
                self.get_RBM_GS(**init_state_params)

            else:
                self.init_state_folder = f'GS_J{self.hamilt.J:01}_h{self.hamilt.h:01}_{sampler}_Cycles{k_samples}_Epochs{epochs}/'
                print("Succesfully imported precomputed ground state!")
        
        elif init_mode=="random":
            pass
        else:
            raise ValueError("Unknown state initialization mode")
    
    def store_evol_to_file(self, results, evol_type, filename):
        
        #Store to file
        WORKDIR_PATH = sys.path[0]
        DATA_PATH = WORKDIR_PATH + "/EVOLUTION_archive/" 
        
        print('\nFile = ', filename)
        
        file_path = DATA_PATH+self.init_state_folder+filename
        if self.applied_ops != '':
            applied_ops_folder = f"{self.applied_ops}/"
            file_path = DATA_PATH+self.init_state_folder+applied_ops_folder+filename
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path,'wb') as f:
            pickle.dump(results,f)
        
        
    
    def TD_validation_error(self,train_gradients, val_gradients):
        #(paras_deriv, S_kk, gradE, EVarExpVal)
        thetadot_F_train = np.matmul(np.conj(train_gradients[0]), train_gradients[2]) #\dot(\theta)^\dagger F
        thetadot_S_thetadot_train = np.matmul(np.matmul(np.conj(train_gradients[0]), train_gradients[1]), train_gradients[0])
        Fdagger_thetadot_train = np.matmul(np.conj(train_gradients[2]).T , train_gradients[0])
        
        tdpv_error = 1 + (thetadot_S_thetadot_train + 1j*thetadot_F_train - 1j*Fdagger_thetadot_train) / train_gradients[-1]
        
        thetadot_F_val = np.matmul(np.conj(val_gradients[0]), train_gradients[2]) #\dot(\theta)^\dagger F
        thetadot_S_thetadot_val = np.matmul(np.matmul(np.conj(val_gradients[0]), train_gradients[1]), val_gradients[0])
        Fdagger_thetadot_val = np.matmul(np.conj(train_gradients[2]).T , val_gradients[0])
        
        val_error = 1 + (thetadot_S_thetadot_val + 1j*thetadot_F_val - 1j*Fdagger_thetadot_val) / train_gradients[-1]
        
        return tdpv_error, val_error
        
    def evolute_quench(self, target_H, delta_t,end_of_time, kContrastDiv, reg_mode, reg_strength, val_fraction, required_paulis, store_result=True):
        """
        Performing Quantum Quench on RBM NQS using Stochastic Reconfiguration
        """
        print("Starting a quench...")
        
        #Make a quench by changing to target hamiltonian
        self.hamilt = target_H
        
        #Initialize outputs
        energies = np.array([])
        pauli_exp_over_time = []
        tdvp_error_over_time = []
        val_error_over_time = []
        reject_percent = np.array([0])

        evol_phases = np.array([]) #Acquinted phases through evolution 
        
        old_ensemble_prob_amps = np.array([])
        
        print(required_paulis)
        
        #Run time
        for t in tqdm(np.arange(0, end_of_time,delta_t)):
            
            im_time_lrate = -1j*delta_t #Imaginary time learning rate for evolution
            
            #Sample ensemble of configurations for expectation evaluation and evolution
            Vensemble, prct = self.MetropolisSamp(self.weights['W'], self.weights['a'], self.weights['c'], self.V, kContrastDiv)
            
            #Split for validation and evaluate expect. on validation set:
            Vensemble_train, Vensemble_val = np.split(Vensemble, [int((1-val_fraction) * Vensemble.shape[0])], axis=0)
            val_expectations = self.evaluate_exp_vals_Vect(self.weights, Vensemble_val, paulis=[[None]])

            val_weights, val_gradients = self.WeightUpdateSmoothed(self.weights, im_time_lrate, t, val_expectations, reg_mode, reg_strength) 
            
            #Evaluate all expectations at once from (training) ensemble:
            tic = timeit.default_timer()
            expectations, pauliExpVals = self.evaluate_exp_vals_Vect(self.weights, Vensemble_train, paulis=required_paulis)
            toc = timeit.default_timer()
            print(f"Vectorized evaluation took {toc-tic} seconds")
            
            EExpVal = expectations[0]
            
            #Get Updated weights for next timestep: |psi(t + delta_t)>
            new_weights, train_gradients = self.WeightUpdateSmoothed(self.weights, im_time_lrate, t, expectations, reg_mode, reg_strength) 
            
            #Calculate the tdvp and validation errors:
            tdvp_error, val_error = self.TD_validation_error(train_gradients, val_gradients) 
              
            #Get acquinted phase change in evolution:
            new_ensemble_prob_amps = expectations[-1]
            if t==0.0:
                old_ensemble_prob_amps = new_ensemble_prob_amps.copy()
            overlap_psi_t_psi_t_dt = self.RMB_inner_product(old_ensemble_prob_amps, new_ensemble_prob_amps)
            print("overlap t, dt:", overlap_psi_t_psi_t_dt)
            evol_phase = np.angle(overlap_psi_t_psi_t_dt / ((1 - 1j*EExpVal)*delta_t )) / delta_t
            
            #Store the new weights and old probability amplitudes for next timestep
            self.weights = new_weights
            old_ensemble_prob_amps = new_ensemble_prob_amps.copy()
            
            
            print(val_error, tdvp_error)
            
            #Store outputs:
            E_per_site = np.real(EExpVal)/self.Nv
            energies = np.append(energies, E_per_site)
            reject_percent = np.append(reject_percent, prct)
            pauli_exp_over_time.append(pauliExpVals)
            evol_phases = np.append(evol_phases, evol_phase)
            tdvp_error_over_time.append(tdvp_error)
            val_error_over_time.append(val_error)
            
            #print(pauliExpVals)
        
        #Pack output for export and return
        pauli_output = (required_paulis, pauli_exp_over_time)
        evol_errors = (tdvp_error_over_time, val_error_over_time)
        
        WRBM = np.copy(self.weights['W'])
        aRBM = np.copy(self.weights['a'])
        cRBM = np.copy(self.weights['c'])
        results = (WRBM, aRBM, cRBM, energies, pauli_output, evol_phases, evol_errors)
        
        if store_result:
            filename = f'NQS_quench_targeth{target_H.h:01}_targetg{target_H.g:01}_dt{delta_t}_eot{end_of_time}_samples{kContrastDiv}_valfrac{val_fraction}_{reg_mode}{reg_strength}.pickle'
            self.store_evol_to_file(results, 'quench', filename)
        
        return results
    
    def run_time(self, delta_t, end_of_time, kContrastDiv, reg_mode, reg_strength, val_fraction, required_paulis):
        target_H = self.hamilt
        results = self.evolute_quench(target_H, delta_t, end_of_time, kContrastDiv, reg_mode, reg_strength, val_fraction, required_paulis, store_result=False)
        
        filename = f'NQS_runtime_h{self.hamilt.h:01}_g{self.hamilt.g:01}_dt{delta_t}_eot{end_of_time}_samples{kContrastDiv}_valfrac{val_fraction}_{reg_mode}{reg_strength}.pickle'
        self.store_evol_to_file(results, 'run_time', filename)
        self.applied_ops += f'U{delta_t}_'

        
        return results
    