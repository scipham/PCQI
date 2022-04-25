import os
import numpy as np
from NQS_RBM import NQS_RBM
import pickle
from tqdm import tqdm


class TD_NQS_RBM(NQS_RBM):
    '''
    Defines a Time Dependent (TD) simulation of an NQS RBM model
    '''
    def __init__(self, init_H, Nv,Nh, init_mode="ground_state", init_state_params={"kContrastDiv": 6000, "lrate": 0.4, "epochs": 52}):
        
        #Initialize a static NQS model
        super().__init__(init_H, Nv,Nh) 
        
        #Prepare the initial state of the system
        if init_mode=="ground_state":
            #
            #Prepare initial state in ground state of initial hamiltonian
            #
            
            #Try to load it from a saved file from an earlier run to save time
            #otherwise (re)compute it on the fly
            try:
                sampler = 'MetroSmoothed'
                k_samples = init_state_params["kContrastDiv"]
                epochs = init_state_params["epochs"]
                
                WORKDIR_PATH =os.getcwd()
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
                print("Succesfully imported precomputed ground state!")
        
        elif init_mode=="random":
            pass
        else:
            raise ValueError("Unknown state initialization mode")
        
        
    def evolute_quench(self, target_H, delta_t,end_of_time, time_lrate, kContrastDiv):
        """
        Performing Quantum Quench on RBM NQS using Stochastic Reconfiguration
        """
        print("Starting a quench...")
        
        #Make a quench by changing to target hamiltonian
        self.hamilt = target_H
        
        #Initialize outputs
        energies = np.array([])
        paulix_mean = np.array([])
        
        reject_percent = np.array([0])
        evol_phases = np.array([]) #Acquinted phases through evolution 
        
        old_ensemble_prob_amps = np.array([])
        
        #Run time
        for t in tqdm(np.arange(0, end_of_time,delta_t)):

            
            #Sample ensemble of configurations for expectation evaluation and evolution
            Vensemble, prct = self.MetropolisSamp(self.weights['W'], self.weights['a'], self.weights['c'], self.V, kContrastDiv)
            
            #Evaluate all expectations at once from ensemble:
            #required_paulis = [[f"X{s}" for s in range(self.Nv)],["X0", "Y1"], ["Y0", "X1"]]
            required_paulis = [[f"X{s}" for s in range(self.Nv)]]
            expectations, pauliExpVals = self.evaluate_exp_vals(self.weights, Vensemble, paulis=required_paulis)
            EExpVal = expectations[0]
            
            #Get Updated weights for next timestep: |psi(t + delta_t)>
            im_time_lrate = -1j*time_lrate #Imaginary time learning rate for evolution
            new_weights = self.WeightUpdateSmoothed(self.weights, im_time_lrate, t, expectations, True) #Note: disabled regularization of covar-matrix.
            
            #Get acquinted phase change in evolution:
            new_ensemble_prob_amps = expectations[-1]
            overlap_psi_t_psi_t_dt = self.RMB_inner_product(old_ensemble_prob_amps, new_ensemble_prob_amps)
            evol_phase = np.angle(overlap_psi_t_psi_t_dt / ((1 - 1j*EExpVal)*delta_t )) / delta_t
            evol_phases = np.append(evol_phases, evol_phase)
            
            #Store the new weights and old probability amplitudes for next timestep
            self.weights = new_weights
            old_ensemble_prob_amps = new_ensemble_prob_amps.copy()
            
            #Store outputs:
            E_per_site = np.real(EExpVal)/self.Nv
            energies = np.append(energies, E_per_site)
            reject_percent = np.append(reject_percent, prct)
            paulix_mean = np.append(paulix_mean, np.mean(pauliExpVals))

            #print(pauliExpVals)

        return energies, paulix_mean
            
        #--------Store and return results
        #--------Calculate other fun quantities with the obtained expectation values
        #--------Apply phase corrections with evol_phases from time evolution where needed (for assymetric overlap of states)
