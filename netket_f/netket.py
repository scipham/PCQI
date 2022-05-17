from copy import deepcopy
import netket as nk
import netket_dynamics as nkd
import qutip
import numpy as np
from tqdm import tqdm

class Netket():

    def __init__(self, N_chain):

        self.N_chain = N_chain

        self.graph = nk.graph.Hypercube(length=N_chain, n_dim=1, pbc=True)

        self.hilbert = nk.hilbert.Spin(s=1 / 2, N=self.graph.n_nodes)

        self.Sx = sum([nk.operator.spin.sigmax(self.hilbert, i) for i in range(self.N_chain)])
        self.Sy = sum([nk.operator.spin.sigmay(self.hilbert, i) for i in range(self.N_chain)])

        self.gs_weigths = None
        self.qutip_gs = None

    def calc_ground_state(self, h_init = 0.5):

        self.h_init = h_init

        self.H_init = nk.operator.Ising(hilbert=self.hilbert, graph=self.graph, h=self.h_init)

        RBM = nk.models.RBM(alpha=1, use_visible_bias=True, dtype=complex)

        sampler = nk.sampler.MetropolisHamiltonian(self.hilbert, self.H_init, n_chains=16)

        self.vs = nk.vqs.MCState(sampler, RBM, n_samples=1000, n_discard_per_chain=100, seed=34)

        optim = nk.optimizer.Sgd(0.01)
        sr = nk.optimizer.SR(diag_shift=1e-4)

        gs = nk.VMC(self.H_init, optim, variational_state=self.vs)

        gs.run(n_iter=300, out="example_ising1d_GS", obs={"Sx": self.Sx})


        self.gs_weigths = self.vs.parameters
        self.qutip_gs = self.vs.to_qobj()

    def quench_evolve(self, h_quench, dt, end_of_time):

        if self.gs_weigths == None:
            print('No initial ground state to calculate time evolution on!')

        else:

            self.H_quench = nk.operator.Ising(hilbert=self.hilbert, graph=self.graph, h=h_quench)
            self.vs.parameters = self.gs_weigths

            time_evolution = nkd.TimeEvolution(self.H_quench, variational_state=self.vs, algorithm=nkd.Euler(), dt=dt)

            log = nk.logging.JsonLog("example_ising1d_TE")
            time_evolution.run(end_of_time, out=log, show_progress=True, obs={"SX": self.Sx})
