from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from score.OON import *
from score.get_data import get_cifar_10_loader
from score.nasbench import api
import random

import os

def score_init():
    global inputs
    inputs=None
    global nasbench
    path=os.path.abspath(__file__)
    path=os.path.dirname(path)
    path=os.path.dirname(path)
    nasbench=api.NASBench(path+'/dataset/nasbench_only108.tfrecord')

def get_score(code):
    spec=CodeToSpec(code)
    network=SpecToNet(spec)
    global inputs
    if(inputs==None):
        loader=get_cifar_10_loader()
        data_iterator=iter(loader)
        inputs, label=next(data_iterator)
    score=NetToScore(network,inputs)
    return score

def get_train_valid_test(code):
    global nasbench
    model_spec=CodeToSpec(code)
    data = nasbench.query(model_spec)
    return data['train_accuracy'],data['validation_accuracy'],data['test_accuracy']

def get_rand_code():
    matrix=np.zeros((7,7),dtype=int)
    l1=np.random.randint(2,size=5)
    previous=-1
    edge=0
    for i,v in enumerate(l1):
        if v==0:
            continue
        if previous==-1:
            matrix[0][i+1] = 1
        else:
            matrix[previous+1][i+1] = 1
        previous = i
        edge+=1
    matrix[previous+1][-1] = 1
    edge+=1
    edge=9-edge
    edge=random.randint(1,edge)
    x_positions, y_positions = np.where(matrix == 0)
    positions = np.where(x_positions < y_positions)
    random_position = np.random.choice(positions[0],size=edge,replace=False)
    x_positions=x_positions[random_position]
    y_positions=y_positions[random_position]
    while edge>0:
        matrix[x_positions[edge-1],y_positions[edge-1]]=1
        edge-=1
    l2=np.zeros(21,dtype=int)
    te=0
    for i in range(0,6):
        for j in range(i+1,7):
            l2[te]=matrix[i,j]
            te+=1
    l3=np.random.randint(3, size=5)
    return (l1,l2,l3)

