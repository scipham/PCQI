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

hamilt = TFIM(h=1/2, g=1/2)

N_V = 10
N_H = 40 


init_state_params = {"kContrastDiv": 6000, 
                     "lrate": 0.4, 
                     "epochs": 70}

evol_params = {'delta_t': 0.001,
            'end_of_time':0.005,
            'kContrastDiv': 9000,
            'reg_mode':'diag_shift',
            'reg_strength':0.0005,
            'val_fraction':0.2}

otoc_calculator = OTOC(hamilt = hamilt,
            Nv = N_V,
            Nh = N_H,
            init_mode="ground_state",
            init_state_params=init_state_params)


time_samples = np.array([0.002, 0.004, 0.006, 0.008]) #np.linspace(0.0, 1.0, 5)
print(time_samples)

otoc_output = otoc_calculator.compute_efficient(evol_params=evol_params,
                                     time_samples=time_samples)
print(otoc_output[1])
plt.plot(otoc_output[0], otoc_output[1])
