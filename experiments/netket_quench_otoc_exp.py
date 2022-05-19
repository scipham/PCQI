import sys, os
import numpy as np

sys.path.append(os.path.join(sys.path[0],'..'))

from netket_f.netket import Netket
from PLOTTING import plot_netket_otoc

N_chain = 10

h_init = 3

N_V = 10
N_H = 40 

time_samples = np.arange(0,1,0.01)

netket_obj = Netket(N_chain = N_chain)

netket_obj.calc_ground_state(h_init=h_init)
otocs_exact = netket_obj.calc_otoc_exact(time_samples)



if __name__ == "__main__":


    plot_netket_otoc(time_samples, otocs_exact)