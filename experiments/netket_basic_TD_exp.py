import sys, os
import numpy as np

sys.path.append(os.path.join(sys.path[0],'..'))

from PLOTTING import plot_netket_quench
from netket_f.netket import Netket

N_chain = 10

h_init = 0.5
h_quench = 1

dt = 0.005
end_of_time = 2 + 1e-8

netket_obj = Netket(N_chain = N_chain)

netket_obj.calc_ground_state(h_init=h_init)
Sx_approx = netket_obj.quench_evolve(h_quench=h_quench, dt=dt, end_of_time=end_of_time)
Sx_exact = netket_obj.quench_evolve_exact(dt=dt, end_of_time=end_of_time)



if __name__ == "__main__":

    time = np.arange(0,end_of_time,dt)

    plot_netket_quench(time, Sx_approx, Sx_exact)