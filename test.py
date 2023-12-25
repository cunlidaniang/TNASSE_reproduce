from score.Score import *
from counter import eval_init
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

score_init()
eval_init()

x=[]
x_eta=[]
x_gamma=[]
x_rho=[]
y=[]
z=[]

for _ in range(1000):
    tt=get_rand_code()
    Psi, eta, gamma, rho=get_score(tt)
    t,v,a=get_train_valid_test(tt)
    x.append(Psi)
    x_eta.append(eta)
    x_gamma.append(gamma)
    x_rho.append(rho)
    y.append(a)
    if(a<0.5):
        z.append(tt)

x=np.array(x)
y=np.array(y)
x_eta=np.array(x_eta)
x_gamma=np.array(x_gamma)
x_rho=np.array(x_rho)

tau, p_value = stats.kendalltau(x, y)
tau_eta, p_value_eta = stats.kendalltau(x_eta, y)
tau_gamma, p_value_gamma = stats.kendalltau(x_gamma, y)
tau_rho, p_value_rho = stats.kendalltau(x_rho, y)

with open('correlation_xy.txt','w') as file:
    file.write('Kendall\'s Tau: ' + str(tau) + '\n')
    file.write('p-value: ' + str(p_value) + '\n')
    file.write('Kendall\'s Tau: ' + str(tau_eta) + '\n')
    file.write('p-value: ' + str(p_value_eta) + '\n')
    file.write('Kendall\'s Tau: ' + str(tau_gamma) + '\n')
    file.write('p-value: ' + str(p_value_gamma) + '\n')
    file.write('Kendall\'s Tau: ' + str(tau_rho) + '\n')
    file.write('p-value: ' + str(p_value_rho) + '\n')


plt.scatter(x, y)
plt.savefig('scatterplot.png')

with open('z.txt', 'w') as file:
    for zz in z:
        print(zz, file=file)