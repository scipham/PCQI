#--------Always include this heading in experiments
import sys, os
from time import time
sys.path.append(os.path.join(sys.path[0],'..'))

from TFIM import *
from NQS_RBM import *
from TD_NQS_RBM import *
from PLOTTING import *
from OTOC import *

np.random.seed(12)

EXPERIMENTS_PATH = sys.path[0]
RESULTS_PATH = EXPERIMENTS_PATH + "/data/" 
#-------------------------------------------------------

init_H = TFIM(h=4.0, g=4.0)
target_H = TFIM(h=3.0, g=3.0)   
#init_H = TFIM(h=4, g=4)
#target_H = TFIM(h=2, g=2)

N_V = 10
N_H = 40 


init_state_params = {"kContrastDiv": 10000, 
                     "lrate": 0.1, 
                     "epochs": 100}

evol_params = {'target_H': target_H,
            'delta_t': 0.001,
            'end_of_time':0.005,
            'kContrastDiv': 9000,
            'reg_mode':'diag_shift',
            'reg_strength':0.0005,
            'val_fraction':0.2}

otoc = OTOC(init_H = init_H,
            Nv = N_V,
            Nh = N_H,
            init_mode="ground_state",
            init_state_params=init_state_params,
            evol_type='quench', 
            evol_params=evol_params)


otoc_output = OTOC.compute()

#with open(RESULTS_PATH+"temp_result.pickle",'wb') as f:
#    pickle.dump(time_evol_output,f)

#time_evol_output = None
#with open(RESULTS_PATH + "temp_result.pickle",'rb') as f:
#    time_evol_output = pickle.load(f) #(Convergence, Percentage, aRBM, cRBM, WRBM, E_exact_per_site)

time = np.arange(0,evol_params['end_of_time'], evol_params['delta_t'] )
