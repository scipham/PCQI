import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

def plot_time_independent_convergence(Convergence,Percentage, EexactPerSite):
    Eexc = EexactPerSite*np.ones(Convergence.shape[0]-1)
    fig, ax = plt.subplots()
    
    ax.plot(Convergence[1:,0],Convergence[1:,1], label="Simulated energy per site")
    ax.plot(Convergence[1:,0],Eexc, label="Exact energy per site")
    ax2 = ax.twinx()
    ax2.plot(Convergence[1:,0],Percentage[1:],color='red',linestyle=':', label="Rejection rate")
    ax2.set_ylim(0,100)
    
    ax.set_title('Convergence')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'${E_{loc}}/{L}$')
    ax2.set_ylabel("Rejection rate")
    ax.legend()
    plt.show()
    
def plot_time_dependent_exp_vals(time,energies, pauli_result):
    x_over_time = []
    z_over_time = []
    pauli_exp_vals = pauli_result[1]
    
    for t in range(time.shape[0]):
        all_x_at_t = np.array(pauli_exp_vals[t][0], dtype=complex)
        average_x_at_t = np.mean(all_x_at_t)
        x_over_time.append(average_x_at_t)
    
        all_z_at_t = np.array(pauli_exp_vals[t][1], dtype=complex)
        average_z_at_t = np.mean(all_z_at_t)
        z_over_time.append(average_z_at_t)
        
    
    plt.plot(time, np.abs(np.array(x_over_time)))
    plt.xlabel('time (s)')
    plt.ylabel(r'\sigma_x')
    plt.ylim((-1.1, 1.1))
    plt.show()
    plt.plot(time, np.real(np.array(z_over_time)))
    plt.xlabel('time (s)')
    plt.ylabel(r'\sigma_z')
    plt.ylim((-1.1, 1.1))
    plt.show()
    plt.plot(time,energies)
    plt.ylabel(r'Energy per site')
    plt.show()
    
def plot_time_evolution_errors(time, evol_errors):
    tdvp_error_over_time = np.array(evol_errors[0])
    val_error_over_time = np.array(evol_errors[1])

    plt.plot(time, tdvp_error_over_time, label=r'r^2_{tr}')
    plt.plot(time, val_error_over_time, label=r'r^2_{val}')
    
    plt.xlabel('time (s)')
    plt.ylabel(r'r^2-error')
    plt.yscale('log')
    plt.legend()
    plt.show()


def plot_netket_quench(time, Sx_netket, Sx_qutip):

    plt.plot(time,Sx_netket, label = 'Netket approximation')
    plt.plot(time,Sx_qutip, label = 'Qutip exact solution')

    plt.xlabel('time in (a.u.)')
    plt.ylabel('<$S_x$>')

    plt.grid()

    plt.show()