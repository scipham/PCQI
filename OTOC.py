import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import timeit

from TD_NQS_RBM import *


class OTOC(NQS_RBM):
    '''
    Defines an OTOC for an NQS RBM evoluted state
    '''
    def __init__(self, init_H, Nv,Nh, init_mode, init_state_params, evol_type, evol_params):

        #Initialize a time dependent NQS model
        super().__init__(init_H, Nv,Nh,  init_H, Nv,Nh, init_mode, init_state_params)
        
        #Define which pauli strings are required for the OTOC
        self.required_paulis = [[f"X{s}" for s in range(self.Nv)],[f"Z{s}" for s in range(self.Nv)]]
        
        if evol_type=="quench":
            #
            #Prepare time evolution of a state OR Try to load it from a saved file from an earlier run to save time
            #otherwise (re)compute it on the fly
            assert 'target_H' in evol_params
            assert 'delta_t' in evol_params
            assert 'end_of_time' in evol_params
            assert 'kContrastDiv' in evol_params 
            assert 'reg_mode' in evol_params
            assert 'reg_strength' in evol_params
            assert 'val_fraction' in evol_params
            
            try:
                target_H = evol_params["target_H"]
                delta_t = evol_params["delta_t"]
                end_of_time = evol_params["end_of_time"]
                kContrastDiv = evol_params['kContrastDiv']
                reg_mode = evol_params['reg_mode']
                reg_strength = evol_params['reg_strength']
                val_fraction = evol_params['val_fraction']
                
                WORKDIR_PATH = sys.path[0]
                DATA_PATH = WORKDIR_PATH + "/EVOLUTION_archive/" 
                filename = f'NQS_quench_data_targeth{target_H.h:01}_targetg{target_H.g:01}_dt{delta_t}_eot{end_of_time}_samples{kContrastDiv}_valfrac{val_fraction}_{reg_mode}{reg_strength}.pickle'
                
                with open(DATA_PATH + filename,'rb') as f:
                    
                    results = pickle.load(f) #(Convergence, Percentage, aRBM, cRBM, WRBM, E_exact_per_site)
                    
                    self.evol_pauli_exp = results[2]
                    #...
            except:
                print("File with evoluted state not found or opening failed")
                print("(Re)Computing ground state...")
                self.evolute_quench(**evol_params, self.required_paulis)

            else:
                print("Succesfully imported precomputed evolution!")
                        
        elif evol_type=="random":
            pass
        else:
            raise ValueError("Unknown type of evolution")
    
    
    
    def compute(self):
        """
        Computes the OTOC of a given time evolution on a RBM NQS
        """
        #OTOC is overlap between two states (operators acted and time evoluted from initial states)
        
        #--------Apply phase corrections with evol_phases from time evolution where needed (for assymetric overlap of states)
    
   
