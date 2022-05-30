#--------Always include this heading in experiments
from pickle import NEXT_BUFFER
import sys, os
from time import time
sys.path.append(os.path.join(sys.path[0],'..'))

from TFIM import *
from NQS_RBM import *
from TD_NQS_RBM import *
from PLOTTING import *
from netket_f.netket import Netket

np.random.seed(12)

EXPERIMENTS_PATH = sys.path[0]
RESULTS_PATH = EXPERIMENTS_PATH + "/data/" 
#-------------------------------------------------------

init_H = TFIM(h=0.5, g=0.5)
target_H = TFIM(h=0.55, g=0.55)

dt_array = np.array([0.2e-3, 0.15e-3, 0.1e-3, 0.08e-3, 0.05e-3])
mean_rel_error = np.array([])

N_V = 10
N_H = 40 

init_state_params = {"kContrastDiv": 6000, 
                     "lrate": 0.4, 
                     "epochs": 70}

evol_params = {'target_H': target_H,
            'delta_t': 0.08e-3, #0.2e-3,
            'end_of_time':0.18,
            'kContrastDiv': 8000,
            'reg_mode':'diag_shift',
            'reg_strength':1e-5,
            'val_fraction':0.1}


for dt in tqdm(dt_array):
    evol_params['delta_t'] = dt
    evol_params['end_of_time'] = 10*dt
    
    td_nqs_model = TD_NQS_RBM(init_H = init_H,
                            Nv = N_V,
                            Nh = N_H,
                            init_mode="ground_state",
                            init_state_params=init_state_params)

    time_evol_output = td_nqs_model.evolute_quench(**evol_params, required_paulis = [[f"X{s}" for s in range(N_V)],[f"Z{s}" for s in range(N_V)]])
    evol_errors = time_evol_output[6]

    mean_rel_error = np.append(mean_rel_error, np.mean(np.array(evol_errors[0]) / np.array(evol_errors[1])))
    

sweep_dt_error_result = (dt_array, mean_rel_error)

with open(RESULTS_PATH+"sweep_dt_error_result.pickle",'wb') as f:
    pickle.dump(sweep_dt_error_result,f)

sweep_dt_error_result = None
with open(RESULTS_PATH + "sweep_dt_error_result.pickle",'rb') as f:
    sweep_dt_error_result = pickle.load(f) #(Convergence, Percentage, aRBM, cRBM, WRBM, E_exact_per_site)

print(sweep_dt_error_result)
plot_time_evolution_rel_error(sweep_dt_error_result[0],sweep_dt_error_result[1])