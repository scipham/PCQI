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

        return(log.data["SX"])

        

    def quench_evolve_exact(self, dt , end_of_time):

        if self.qutip_gs == None:
            print('No initial ground state to calculate exact time evolution on!')

        else:
            tvals = np.arange(0.0, end_of_time, dt)
            Sx_dyn = qutip.sesolve(self.H_quench.to_qobj(), self.qutip_gs, tvals, e_ops=[self.Sx.to_qobj()]).expect[0]

        return(Sx_dyn)

    def calc_otoc(self):

        pass

    def calc_otoc_exact(self, time_samples):

        assert(self.qutip_gs != None)

        psi0_1 = deepcopy(self.qutip_gs)
        psi0_2 = deepcopy(self.qutip_gs)

        V1 = self.Sx.to_qobj()
        V2 = self.Sy.to_qobj()

        H = self.H_init.to_qobj()

        V1psi = V1 * psi0_1

        otocs = []

        for i,t in tqdm(enumerate(time_samples)):

            HV1psi = qutip.sesolve(H,V1psi ,[0,t]).states[0]
            Hpsi = qutip.sesolve(H, psi0_2, [0,t]).states[0]

            V2Hpsi =  V2 * Hpsi 
            V2HV1psi =  V2 * HV1psi

            H_dagV2Hpsi = qutip.sesolve(H, V2Hpsi, [0,-t]).states[0]
            H_dagV2HV1psi = qutip.sesolve(H, V2HV1psi, [0,-t]).states[0]

            V1H_dagV2Hpsi = V1 * H_dagV2Hpsi

            V1H_dagV2Hpsi_n = V1H_dagV2Hpsi.norm()
            H_dagV2HV1psi_n = H_dagV2HV1psi.norm()


            otocs.append(H_dagV2HV1psi_n.overlap(V1H_dagV2Hpsi_n))

        return(otocs)