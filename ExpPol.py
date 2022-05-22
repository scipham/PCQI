import matplotlib.pyplot as plt
import numpy as np

n = np.linspace(1,30,500)

exp = np.exp(n)
pol = n**2

plt.semilogy(n, exp, color = 'blue' , label = 'Exact')
plt.semilogy(n, pol, color = 'red', label = 'NQS')

plt.xticks(ticks = np.arange(0,31,5))

plt.xlabel('Number of particles', fontsize = 13)
plt.ylabel('Degrees of freedom', fontsize = 13)
plt.title('Advantage of NQS versus exact description', fontsize = 16)
plt.grid()
plt.legend()

plt.savefig('NQSadvantage.png')
plt.show()

