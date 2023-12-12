from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from score.OON import *
from score.get_data import get_cifar_10_loader

def score_init():
    global inputs
    inputs=None

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
