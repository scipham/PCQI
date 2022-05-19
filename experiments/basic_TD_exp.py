#--------Always include this heading in experiments
from pickle import NEXT_BUFFER
import sys, os
from time import time
sys.path.append(os.path.join(sys.path[0],'..'))

from TFIM import *
from NQS_RBM import *
from TD_NQS_RBM import *
from PLOTTING import *

np.random.seed(12)

EXPERIMENTS_PATH = sys.path[0]
RESULTS_PATH = EXPERIMENTS_PATH + "/data/" 
#-------------------------------------------------------

init_H = TFIM(h=0.5, g=0.5)
target_H = TFIM(h=0.6, g=0.6)     
#init_H = TFIM(h=4, g=4)
#target_H = TFIM(h=2, g=2)

N_V = 10
N_H = 40 

init_state_params = {"kContrastDiv": 6000, 
                     "lrate": 0.1, 
                     "epochs": 70}

evol_params = {'target_H': target_H,
            'delta_t': 0.005,
            'end_of_time':0.05,
            'kContrastDiv': 9000,
            'reg_mode':'diag_shift',
            'reg_strength':1e-7,
            'val_fraction':0.2}

td_nqs_model = TD_NQS_RBM(init_H = init_H,
                            Nv = N_V,
                            Nh = N_H,
                            init_mode="ground_state",
                            init_state_params=init_state_params)

time_evol_output = td_nqs_model.evolute_quench(**evol_params, required_paulis = [[f"X{s}" for s in range(N_V)],[f"Z{s}" for s in range(N_V)]])

#with open(RESULTS_PATH+"temp_result.pickle",'wb') as f:
#    pickle.dump(time_evol_output,f)

#time_evol_output = None
#with open(RESULTS_PATH + "temp_result.pickle",'rb') as f:
#    time_evol_output = pickle.load(f) #(Convergence, Percentage, aRBM, cRBM, WRBM, E_exact_per_site)

time = np.arange(0,evol_params['end_of_time'], evol_params['delta_t'] )

plot_time_dependent_exp_vals(time, time_evol_output[3],  time_evol_output[4]) 
plot_time_evolution_errors(time, time_evol_output[6])