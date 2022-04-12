import numpy as np

from .lnrmbwavefunction import LnRMBWavefunction


def MetropolisCycle(W,a,c,Vt):
    rejectvalue = 0   
    LnPsiOld = LnRMBWavefunction(W,a,c,Vt)
    #
    # Flip a random spin
    # 
    L = Vt.shape[0] 
    site = np.random.randint(L)
    Vt[site] = - Vt[site] +1
    LnPsiNew = LnRMBWavefunction(W,a,c,Vt)
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

def MetropolisSamp(W,a,c,V,k):
    #
    # Burn-in to get rid of initial condition dependence
    #
    rejections = 0
    rejectvalue = 0
    burn_in = 10000

    for z in range(burn_in):
        Vt = V
        V,rejectvalue = MetropolisCycle(W,a,c,Vt)
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
            V,rejectvalue = MetropolisCycle(W,a,c,V)
        Vensemble = np.append(Vensemble,V)
        rejections = rejections + rejectvalue
    
    prctrej = 100*rejections/k
    #print('Percentage Rejections in Ensemble: %.1f %% (%i/%i)' %(prctrej,rejections,k))
    Vensemble_reshape = Vensemble.reshape((k+1,L))
    # print(Vensemble_reshape)
    return Vensemble_reshape, prctrej 


