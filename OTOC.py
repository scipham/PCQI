from mimetypes import init
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import timeit
from copy import deepcopy

from TD_NQS_RBM import *


class OTOC:
    '''
    Defines an OTOC for an NQS RBM evoluted state
    '''
    def __init__(self, hamilt, Nv,Nh, init_mode, init_state_params):

        #Initialize a time dependent NQS model
        self.psi_1 = TD_NQS_RBM(hamilt, Nv,Nh, init_mode, init_state_params)
        self.psi_2 = TD_NQS_RBM(hamilt, Nv,Nh, init_mode, init_state_params)
        
        #Define which pauli strings are required for the OTOC
        self.op_1 = [f"X{s}" for s in np.arange(1, Nv, step=1)]
        self.op_2 = [f"Y{s}" for s in np.arange(1, Nv, step=2)]
        
    def get_evolution(self, state, evol_params, required_paulis, reverse=False):
        #
        #Prepare time evolution of a state OR Try to load it from a saved file from an earlier run to save time
        #otherwise (re)compute it on the fly
        assert 'delta_t' in evol_params
        assert 'end_of_time' in evol_params
        assert 'kContrastDiv' in evol_params 
        assert 'reg_mode' in evol_params
        assert 'reg_strength' in evol_params
        assert 'val_fraction' in evol_params
        
        if reverse: #Inverse time evolution
            evol_params["delta_t"] = -1*abs(evol_params["delta_t"])
            
        applied_ops_folder = state.applied_ops
        '''
            delta_t = evol_params["delta_t"]
            end_of_time = evol_params["end_of_time"]
            kContrastDiv = evol_params['kContrastDiv']
            reg_mode = evol_params['reg_mode']
            reg_strength = evol_params['reg_strength']
            val_fraction = evol_params['val_fraction']
            
            WORKDIR_PATH = sys.path[0]
            DATA_PATH = WORKDIR_PATH + "/EVOLUTION_archive/" 
            
            init_state_folder = state.init_state_folder
            
            filename = f'NQS_runtime_h{self.hamilt.h:01}_g{self.hamilt.g:01}_dt{delta_t}_eot{end_of_time}_samples{kContrastDiv}_valfrac{val_fraction}_{reg_mode}{reg_strength}.pickle'
            file_path = DATA_PATH + init_state_folder + applied_ops_folder + filename
            
            with open(file_path,'rb') as f:
                
                results = pickle.load(f) # (WRBM, aRBM, cRBM, energies, pauli_output, evol_phases, evol_errors)
                state.weights['W'] = results[0]
                state.weights['a'] = results[1]
                state.weights['c'] = results[2]
                return state, results
        except:
            print("File with evoluted state not found or opening failed")
            print("(Re)Computing state evolution...")
        '''
        return state, state.run_time(**evol_params, required_paulis=required_paulis, store_result=False) 

        #else:
        #    print("Succesfully imported precomputed evolution!")
    
    
    def compute_efficient(self, evol_params, time_samples):
        """
        Computes the OTOC of a given time evolution on a RBM NQS
        """
        #OTOC is overlap between two states (operators acted and time evoluted from initial states)
        
        if any(np.diff(time_samples) < evol_params['delta_t']):
            raise ValueError("Provide only time samples with difference larger than the evolution timestep")
        elif 0.0 in time_samples:
            raise ValueError("Taking the OTOC at t=0.0 doesn't make sense. Give only time samples > 0.0")
        
        self.psi_1.apply_pauli_string(self.op_1)
        init_state_1 = deepcopy(self.psi_1)
        init_state_2 = deepcopy(self.psi_2)
        
        otoc_output = []
        tot_global_phases_1, tot_global_phases_2 = np.array([]), np.array([])
        
        for (it, t) in tqdm(enumerate(time_samples)):
            time_step = time_samples[it]-time_samples[it-1]
            if it == 0:
                time_step = time_samples[it]-0.0
            
            print(f"\n Busy with iteration {it} of {time_samples.shape[0]} \n \n")
            
            evol_params['end_of_time'] = time_step
            
            #Update current initial state to save time in next timestep
            init_state_1, results_1_forward = self.get_evolution(state=init_state_1, evol_params=evol_params, required_paulis=[self.op_1], reverse=False)
            #print(init_state_1.weights)
            tot_global_phases_1 = np.append(tot_global_phases_1, results_1_forward[-2])
            
            init_state_2, results_2_forward = self.get_evolution(state=init_state_2, evol_params=evol_params, required_paulis=[self.op_1], reverse=False)
            tot_global_phases_2 = np.append(tot_global_phases_2, results_2_forward[-2])
            
            #Inefficient part:
            temp_state_1 = deepcopy(init_state_1)
            temp_state_2 = deepcopy(init_state_2)
            temp_tot_global_phases_1 = deepcopy(tot_global_phases_1)
            temp_tot_global_phases_2 = deepcopy(tot_global_phases_2)
            
            evol_params['end_of_time'] = t #Set evolution time for backwards time evolution
            
            temp_state_1.apply_pauli_string(self.op_2)
            temp_state_1, results_1_backward = self.get_evolution(state=temp_state_1, evol_params=evol_params, required_paulis=[self.op_1], reverse=True)
            temp_tot_global_phases_1 = np.append(temp_tot_global_phases_1, results_1_backward[-2])
            
            temp_state_2.apply_pauli_string(self.op_2)
            temp_state_2, results_2_backward = self.get_evolution(state=temp_state_2, evol_params=evol_params, required_paulis=[self.op_1], reverse=True)
            temp_tot_global_phases_2 = np.append(temp_tot_global_phases_2, results_2_backward[-2])
            temp_state_2.apply_pauli_string(self.op_1)
            
            
            Vensemble, prct = temp_state_1.MetropolisSamp(temp_state_1.weights['W'], temp_state_1.weights['a'], temp_state_1.weights['c'], temp_state_1.init_V, evol_params['kContrastDiv'])
            expectations_1 = temp_state_1.evaluate_exp_vals_Vect(temp_state_1.weights, Vensemble)
            psi_1_prob_amps = expectations_1[-1]
            Vensemble, prct = temp_state_2.MetropolisSamp(temp_state_2.weights['W'], temp_state_2.weights['a'], temp_state_2.weights['c'], temp_state_2.init_V, evol_params['kContrastDiv'])
            expectations_2 = temp_state_2.evaluate_exp_vals_Vect(temp_state_2.weights, Vensemble)
            psi_2_prob_amps = expectations_2[-1]
            
            raw_otoc_val = temp_state_1.RMB_inner_product(psi_1_prob_amps, psi_2_prob_amps)
            
            #Cancel acquired evolution phases 
            temp_tot_global_phases_2 = np.conj(temp_tot_global_phases_2)
            
            otoc_val = raw_otoc_val * (np.prod(np.exp(1j*temp_tot_global_phases_1)) * np.prod(np.exp(1j*temp_tot_global_phases_2)))
            
            otoc_output.append(otoc_val)
        
        print("Finished OTOC computation ")
        
        return time_samples, np.array([otoc_output]).flatten()
            

    def compute_inefficient(self, evol_params, time_samples):
        """
        Computes the OTOC of a given time evolution on a RBM NQS
        """
        #OTOC is overlap between two states (operators acted and time evoluted from initial states)
        
        if any(np.diff(time_samples) < evol_params['delta_t']):
            raise ValueError("Provide only time samples with difference larger than the evolution timestep")
        
        
        otoc_output = []
        
        for (it, t) in tqdm(enumerate(time_samples)):
            
            temp_state_1 = self.psi_1.deepcopy()
            temp_state_2 = self.psi_2.deepcopy()
            
            evol_params['end_of_time'] = t
            
            tot_global_phases_1, tot_global_phases_2 = np.array([]), np.array([])
            
            temp_state_1.apply_pauli_string(self.op_1)
            temp_state_1, results_1_forward = self.get_evolution(state=temp_state_1, evol_params=evol_params, required_paulis=[self.op_1], reverse=False)
            tot_global_phases_1 = np.append(tot_global_phases_1, results_1_forward[-2])
            temp_state_1.apply_pauli_string(self.op_2)
            temp_state_1, results_1_backward = self.get_evolution(state=temp_state_1, evol_params=evol_params, required_paulis=[self.op_1], reverse=True)
            tot_global_phases_1 = np.append(tot_global_phases_1, results_1_backward[-2])
            
            self.get_evolution(state=temp_state_2, evol_params=evol_params, required_paulis=[self.op_1], reverse=False)
            temp_state_2, results_2_forward = self.get_evolution(state=temp_state_2, evol_params=evol_params, required_paulis=[self.op_1], reverse=False)
            tot_global_phases_2 = np.append(tot_global_phases_2, results_2_forward[-2])
            temp_state_2.apply_pauli_string(self.op_2)
            temp_state_2, results_2_backward = self.get_evolution(state=temp_state_2, evol_params=evol_params, required_paulis=[self.op_1], reverse=True)
            tot_global_phases_2 = np.append(tot_global_phases_2, results_2_backward[-2])
            temp_state_2.apply_pauli_string(self.op_1)
            
            Vensemble, prct = temp_state_1.MetropolisSamp(temp_state_1.weights['W'], temp_state_1.weights['a'], temp_state_1.weights['c'], temp_state_1.init_V, evol_params['kContrastDiv'])
            expectations_1 = temp_state_1.evaluate_exp_vals(temp_state_1.weights, Vensemble)
            psi_1_prob_amps = expectations_1[-1]
            Vensemble, prct = temp_state_2.MetropolisSamp(temp_state_2.weights['W'], temp_state_2.weights['a'], temp_state_2.weights['c'], temp_state_2.init_V, evol_params['kContrastDiv'])
            expectations_2 = temp_state_2.evaluate_exp_vals(temp_state_2.weights, Vensemble)
            psi_2_prob_amps = expectations_2[-1]
            
            raw_otoc_val = temp_state_1.RMB_inner_product(psi_1_prob_amps, psi_2_prob_amps)

            
            #Cancel acquired evolution phases 
            tot_global_phases_2 = np.conj(tot_global_phases_2)
            otoc_val = raw_otoc_val / (np.sum(tot_global_phases_1) + np.sum(tot_global_phases_2))
            
            otoc_output.append(otoc_val)
        
        print("Finished OTOC computation ")
        
        return time_samples, np.array(otoc_output)
                
   
