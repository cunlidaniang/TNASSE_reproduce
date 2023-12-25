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
    x.append([eta, gamma, rho])
    y.append(a)

x=np.array(x)
y=np.array(y)

from sklearn.neural_network import MLPRegressor

regr = MLPRegressor(random_state=1, max_iter=500).fit(x, y)

x_test=[]
y_test=[]
for _ in range(100):
    tt=get_rand_code()
    Psi, eta, gamma, rho=get_score(tt)
    t,v,a=get_train_valid_test(tt)
    x_test.append([eta, gamma, rho])
    y_test.append(a)

x_test=np.array(x_test)
y_test=np.array(y_test)

y_pred=regr.predict(x_test)
tau, p_value = stats.kendalltau(y, y_pred)

with open('correlation.txt','w') as file:
    file.write('Kendall\'s Tau: ' + str(tau) + '\n')
    file.write('p-value: ' + str(p_value) + '\n')