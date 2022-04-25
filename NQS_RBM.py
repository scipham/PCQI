import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

def sigmoid(X):
    return 1./(np.exp(X)+1)

class NQS_RBM:
    
    def __init__(self, hamilt, Nv,Nh):
        self.hamilt = hamilt
        self.Nv = Nv
        self.Nh = Nh
        self.weights = {}
        
    
        # Service message
        print("""\
            Neural Quantum State of the transverse field Ising model:
            Ising model parameters J, h: %f, %f
            Number of visible spins: %i
            Number of hidden spins: %i
            """ %(self.hamilt.J,self.hamilt.h, self.Nv,self.Nh))

        #
        # Initialing visible spins with either 0 or 1
        #
        self.V = np.random.choice([0,1],self.Nv)
        
        magnetization = np.sum(self.V)-self.Nv/2

        print('Magnetization of Initial state: ', magnetization)

        #Initialize weights:
        self.initialize_weights()
    
    def initialize_weights(self):
        #
        # Initializing weights with real values between -1 and 1
        # The system is VERY sensitive to initial conditions. 
        # E.g. it will not converge if all weights are negative.
        #
        W0 = (0.2)*(2*np.random.rand(self.Nh,self.Nv)-1. +np.random.rand(self.Nh,self.Nv)*1j)
        a0 = (0.1)*(2*np.random.rand(self.Nv)-1. + np.random.rand(self.Nv)*1j)
        c0 = (0.1)*(2*np.random.rand(self.Nh)-1. + np.random.rand(self.Nh)*1j)
 
        self.weights['W'] = np.real(W0)
        self.weights['a'] = np.real(a0)
        self.weights['c'] = np.real(c0)
    
    def LnRMBWavefunction(self, W,a,c,V):
        #
        # Golden rule of numerics: avoid exponentials.
        # Use ln's instead.
        #
        Wsummed = 0
        LnPreFactor = 0
        L = V.shape[0]
        for s in range(L):
            Wsummed = Wsummed + W[:,s]*V[s]
            LnPreFactor = LnPreFactor - a[s]*V[s]
        
        # Difference between bits 0 and 1 and spins -1 and 1
        LnPrePreFactor = np.sum(a)/2 + np.sum(c)/2+np.sum(W)/4
        AngleFactor = np.prod(1+np.exp(-c - Wsummed))
        LnPsiRMB = LnPrePreFactor + LnPreFactor + np.log(AngleFactor)
        return LnPsiRMB
    
    def MetropolisCycle(self, W,a,c,Vt):
        rejectvalue = 0   
        LnPsiOld = self.LnRMBWavefunction(W,a,c,Vt)
        #
        # Flip a random spin
        # 
        L = Vt.shape[0] 
        site = np.random.randint(L)
        Vt[site] = - Vt[site] +1
        LnPsiNew = self.LnRMBWavefunction(W,a,c,Vt)
        #
        acceptanceratio = np.exp(np.real(np.conj(LnPsiNew)+LnPsiNew-np.conj(LnPsiOld)-LnPsiOld))
        #if acceptanceratio #MISSING INEQUALITY SIGN# 1:
        if acceptanceratio >= 1:
            return Vt,rejectvalue
        else:
            p = np.random.rand()
            #if p #MISSING INEQUALITY SIGN# acceptanceratio:
            if p >= acceptanceratio:
                rejectvalue = 1
                Vt[site] = - Vt[site] + 1
                
            return Vt,rejectvalue

    def MetropolisSamp(self, W,a,c,V,k):
        #
        # Burn-in to get rid of initial condition dependence
        #
        rejections = 0
        rejectvalue = 0
        burn_in = 10000

        for z in range(burn_in):
            Vt = V
            V,rejectvalue = self.MetropolisCycle(W,a,c,Vt)
            rejections = rejections + rejectvalue
        
        print('Percentage Rejections in Burn-in: %.2f %%' %(rejections/burn_in*100))
        #
        #
        # We collect the full sequence of spin configurations V
        # Together they form a efficient short representation of the full distribution
        # 
        rejections = 0
        rejectvalue = 0
        Vensemble = np.copy(V)
        L = np.shape(V)[0]
        for z in range(k):
            # initiate sweep, i.e. cycle over # visible spins between appending
            for zz in range(L):
                V,rejectvalue = self.MetropolisCycle(W,a,c,V)
            Vensemble = np.append(Vensemble,V)
            rejections = rejections + rejectvalue
        
        prctrej = rejections/(L*k) * 100
        #print('Percentage Rejections in Ensemble: %.1f %% (%i/%i)' %(prctrej,rejections,k))
        Vensemble_reshape = Vensemble.reshape((k+1,L))
        # print(Vensemble_reshape)
        return Vensemble_reshape, prctrej 
    
    def Elocal(self, W,a,c,V):
        #
        # Computing the wavefunction for state V
        #
        L = V.shape[0]
        LnPsi = self.LnRMBWavefunction(W,a,c,V)
        LnPsiBar = np.conj(LnPsi)
        #
        # Computing the energy for state V
        # First the Ising term
        #
        Vshift = np.array([V[(i+1)%L] for i in range(L)])
        One = np.ones(L)
        ElocalJ = -self.hamilt.J*(np.sum((2*V-One)*(2*Vshift-One)))
        #
        # Next the magnetic term -B\sum_i \sigma^x_i
        # Because this is not diagonal on the
        # states, we compute 
        # <V|EB|Psi> instead
        # The action of Sigma^x_i is
        # to flip the spin on site i:
        # i.e. map V[i] to -V[i]+1
        #
        EBlocalPsi = 0
        for i in range(L):
            V[i] = -V[i]+1
            EBlocalPsi = EBlocalPsi - self.hamilt.h*np.exp(self.LnRMBWavefunction(W,a,c,V)-LnPsi) #Compare flipped with unflipped (sigma_x applied)
            V[i] = -V[i]+1
        
        ElocalPsi = ElocalJ + EBlocalPsi
        
        return ElocalPsi, LnPsi
    
    def RMB_inner_product(self, left_prob_amps, right_prob_amps):
        v1 = np.mean(left_prob_amps / right_prob_amps)
        v2 = np.mean(right_prob_amps / left_prob_amps)
        overlap = np.sqrt(np.conj(v1) * v2) 
        return overlap
    
    def eval_pauli(self, operator, site, V):
        LnPsi = self.LnRMBWavefunction(self.weights['W'], self.weights['a'], self.weights['c'], V)
        V = V.copy()
        
        if operator == "X":
            temp_a = np.array(self.weights['a'], dtype = complex)
            temp_a[site] = -temp_a[site]
            temp_W = np.array(self.weights['W'], dtype = complex)
            temp_W[:,site] = -temp_W[:, site] 
            return np.exp(self.LnRMBWavefunction(temp_W, temp_a, self.weights['c'], V)-LnPsi) #Compare flipped with unflipped (sigma_x applied)
        
        elif operator == "Z":
            temp_a = np.array(self.weights['a'], dtype = complex)
            temp_a[site] += 1j*np.pi/2
            return np.exp(self.LnRMBWavefunction(self.weights['W'], temp_a, self.weights['c'], V)-LnPsi) #Compare flipped with unflipped (sigma_z applied)
        
        elif operator == "Y":
            #Apply first X than Z + ignore global phase factor -i
            temp_a = np.array(self.weights['a'], dtype = complex)
            temp_a[site] = -temp_a[site]
            temp_W = np.array(self.weights['W'], dtype = complex)
            temp_W[:,site] = -temp_W[:, site] 
            
            temp_a[site] += 1j*np.pi/2
            return np.exp(self.LnRMBWavefunction(temp_W, temp_a, self.weights['c'], V)-LnPsi) #Compare flipped with unflipped (sigma_y applied)

        else:
            raise ValueError("Unknown Pauli operator")
    
    def evaluate_exp_vals(self, o_weights, Vensemble, paulis=[[None]]):
               # 
        # <Psi|Operator|Psi> = \sum_{all S,S'} <Psi|S><S|Operator|S'><S'|Psi>
        # is approximated by ensemble average
        # <Psi|Operator|Psi> \simeq \sum_{Gibbs S,S'} <Psi|S><S|Operator|S'><S'|Psi>
        # For L large dim(S)=2^L, whereas we only need a finite number of Gibbs samples
        # So this will help greatly at large L
        #
        #o_weights = old weights
        
        LenEnsemb = Vensemble.shape[0]
        L = self.Nv
        H = self.Nh
        #
        # Initializing for ensemble Exp(ectation)Val(ue)
        #
        LnNormPsi = 0
        EExpVal = 0
        ElocalExpVal = 0
        ElocalVExpVal = 0
        ElocalHExpVal = 0
        ElocalWExpVal = 0
        derivsExpVal = 0
        moment2ExpVal = 0
        
        ensemble_prob_amps = np.array([])
        
        pauliExpVals = [[0 for pauli in pauli_str] for pauli_str in paulis ]
        
        for l in range(LenEnsemb):
            V = Vensemble[l]
            #
            # V now labels a particular state
            #
            # Computing the energy for state V
            #
            ElocalPsi, LnPsi = self.Elocal(o_weights['W'],o_weights['a'],o_weights['c'],V)
            #
            # Next we compute 
            # <V|EV|V> = Elocal*V
            # <V|EH|V> = <Esigmoid(WV+c)> =Elocal*
            # <V|EHV|V> = <EVsigmoid(WV+c)>
            #
            ElocalVPsi = ElocalPsi*V 
            ElocalHPsi = ElocalPsi*sigmoid(o_weights['c'] + np.matmul(o_weights['W'],V))  #sigmoid = current h vector
            ElocalWPsi = ElocalPsi*np.outer(sigmoid(o_weights['c'] + np.matmul(o_weights['W'],V)),V)
            # 
            # Next we compute 
            # <V>
            # <H>
            # <HV>
            #
            derivs = np.concatenate((V,np.real(sigmoid(o_weights['c']+np.matmul(o_weights['W'],V))),np.real(np.outer(sigmoid(o_weights['c']+np.matmul(o_weights['W'],V)),V)).reshape(L*H)))
            #
            # Matrix of conj.derivs \times derivs
            #
            moment2 = np.outer(np.conj(derivs),derivs)
            #
            # Computing ensemble averages (uniform distrib. over all sampled configs)
            #
            ElocalExpVal = ElocalExpVal + ElocalPsi/LenEnsemb
            ElocalVExpVal = ElocalVExpVal + np.real(ElocalVPsi)/(LenEnsemb)
            ElocalHExpVal = ElocalHExpVal + np.real(ElocalHPsi)/(LenEnsemb)
            ElocalWExpVal = ElocalWExpVal + np.real(ElocalWPsi)/(LenEnsemb)
            derivsExpVal = derivsExpVal + derivs/LenEnsemb
            moment2ExpVal = moment2ExpVal + moment2/LenEnsemb
            
            #Evaluate Ensemble probability amplitudes <s|psi>
            np.append(ensemble_prob_amps, LnPsi)
            
            #Evaluate additional pauli strings:
            if paulis[0][0] != None:
                for (i_str, pauli_str) in enumerate(paulis):
                    for (i, pauli) in enumerate(pauli_str):
                        operator, site = pauli[0], int(pauli[1])

                        pauli_exp = self.eval_pauli(operator, site, V)
                        pauliExpVals[i_str][i] += pauli_exp / (LenEnsemb)
        
        if paulis[0][0] == None:
            return (ElocalExpVal, ElocalVExpVal, ElocalHExpVal, ElocalWExpVal, derivsExpVal, moment2ExpVal, ensemble_prob_amps)
        else:
            return (ElocalExpVal, ElocalVExpVal, ElocalHExpVal, ElocalWExpVal, derivsExpVal, moment2ExpVal, ensemble_prob_amps), pauliExpVals
            
    def WeightUpdateSmoothed(self, o_weights,lrate,ep, expectations, reg_strength=1.0):   
 
        L = self.Nv
        H = self.Nh
        
        VExpVal = 0
        HExpVal = 0
        WExpVal = 0
        agradientEExpVal = 0
        cgradientEExpVal = 0
        WgradientEExpVal = 0
        
        ElocalExpVal, ElocalVExpVal, ElocalHExpVal, ElocalWExpVal, derivsExpVal, moment2ExpVal, ensemble_prob_amps = expectations
        #
        # Statistical local gradients, ignoring the quantum mechanical term
        #
        VExpVal = derivsExpVal[:L]
        HExpVal = derivsExpVal[L:L+H]
        WExpVal = derivsExpVal[L+H:].reshape(H,L)
        agradientEStat = - ElocalVExpVal + ElocalExpVal*VExpVal
        cgradientEStat = - ElocalHExpVal + ElocalExpVal*HExpVal
        WgradientEStat = - ElocalWExpVal + ElocalExpVal*WExpVal
        #
        # Computing metric on Probability space
        #
        #   - Cartesian metric as default
        #
        S_kkCartesian = np.diag(np.ones(L*H+L+H))
        #
        #   - Sorella version
        #
        S_kkSorella = moment2ExpVal - np.outer(np.conj(derivsExpVal),derivsExpVal)
        
        #S_kk = S_kkCartesian
        
        #
        #   - Regulator necessary to ensure inverse exists
        #
        lreg = reg_strength * np.max(np.array([100*(0.9)**ep,0.01]))    
        S_kkSorellaReg =  lreg * np.diag(np.diag(S_kkCartesian))
        
        #
        #S_kk = S_kkCartesian
        S_kk = S_kkSorella + S_kkSorellaReg #Sorella = use variance in parameters/their derivates to adjust learning rate individually (per parameter type, per parameter)!
    
        agrad = np.copy(agradientEStat)
        cgrad = np.copy(cgradientEStat)
        Wgrad = np.copy(WgradientEStat)
        #
        # Print out average length-squared of gradients as diagnostic
        # (finding good initial guess of model parameters manually)
        #
        GradAAbsSq = np.real(np.inner(np.conj(agrad),agrad))/L
        GradCAbsSq = np.real(np.inner(np.conj(cgrad),cgrad))/H
        GradWAbsSq = np.real(np.sum(np.conj(Wgrad)*Wgrad))/(L*H)
        print('\rGradient absval-squared: a: %.4f, c: %.4f, W: %.4f. ' %(GradAAbsSq,GradCAbsSq,GradWAbsSq), end='')
        #
        #
        Wgradtemp = Wgrad.reshape(L*H)
        paras = np.concatenate((o_weights['a'],o_weights['c'],o_weights['W'].reshape(L*H)))
        gradE = np.conj(np.concatenate((agrad,cgrad,Wgradtemp)))
        #
        #print('output',np.mean(S_kk), np.mean(gradE))
        deltaparas = lrate * np.einsum('ij,j->i',np.linalg.pinv(S_kk),gradE) #Learning rate in metric x gradient
        paras = paras - deltaparas #Update parameters (collectively in one big array)
        print('average weight update size:', np.average(deltaparas))
        #
        #
        
        n_weights = {}
        n_weights['a'] = paras[:L]
        n_weights['c'] = paras[L:L+H]
        n_weights['W'] = paras[L+H:].reshape(H,L)
        #
        #print('Local Energy: ', ElocalExpVal)
        #
        return n_weights
    
    def get_exact_GS(self):
        # The transverse field Ising model happens to
        # be exactly solvable through other means.
        # We secretly know the exact GS energy:
        #

        free_fermion_modes = np.sqrt(1 + self.hamilt.g**2-2*self.hamilt.g*np.cos(2*np.pi*np.arange(self.Nv)/self.Nv)) 
        E_exact_per_site = -self.hamilt.J*np.sum(free_fermion_modes)/self.Nv #Number of modes on each site * energy of occupation = interaction energy
        return E_exact_per_site
        
    def get_RBM_GS(self, kContrastDiv, lrate,epochs, reg_strength=1.0):
        # Service message
        print("""\
            Performing variational ground state search with:
            Monte Carlo sequence size: %i
            Learning Rate: %f
            Epochs: %i
            """ %(kContrastDiv, lrate, epochs))

        #
        # Variable Initialization for plotting results
        #
        Convergence = np.array([[1,1]])
        Percentage = np.array([0])
        prct = 0
        E_exact_per_site = 0
        
        #
        # Learning/Variational Minimization cycle
        #
        for ep in tqdm(range(epochs)):
            #
            Vensemble, prct = self.MetropolisSamp(self.weights['W'], self.weights['a'], self.weights['c'], self.V, kContrastDiv) #Get  representative samples
            
            expectations = self.evaluate_exp_vals(self.weights, Vensemble)
            self.weights = self.WeightUpdateSmoothed(self.weights, lrate, ep, expectations, reg_strength) #Update paramters by fixed paramter gradients on ensemble
            
            EExpVal = expectations[0]
            EVarPerSite = np.real(EExpVal)/self.Nv
            Convergence = np.append(Convergence,np.array([[ep,EVarPerSite]]),axis=0)
            Percentage = np.append(Percentage,np.array([prct]),axis=0)
            #lrate = lrate * 0.95 
            
            E_exact_per_site = self.get_exact_GS()
            print('\rEpoch %i/%i: Variational Energy: %f, Exact Energy: %f ' %(ep+1,epochs,EVarPerSite, E_exact_per_site), end='')
            if not np.abs(EVarPerSite) < 10e6:
                print('\nNumerical Runaway: discontinuing...')
                break
            #print('Weights updated: Started learning epoch %i out of %i\n' %(ep+1,epochs))
        
        WRBM = np.copy(self.weights['W'])
        aRBM = np.copy(self.weights['a'])
        cRBM = np.copy(self.weights['c'])
        sampler = 'MetroSmoothed'
        results = (Convergence, Percentage, aRBM, cRBM, WRBM, E_exact_per_site)
                
        WORKDIR_PATH = sys.path[0]
        DATA_PATH = WORKDIR_PATH + "/GS_archive/" 
        filename = f'NQSdata_J{self.hamilt.J:01}_h{self.hamilt.h:01}_{sampler}_Cycles{kContrastDiv}_Epochs{epochs}.pickle'
        print('\nFile = ', filename)
        
        with open(DATA_PATH+filename,'wb') as f:
            pickle.dump(results,f)
            
        return results