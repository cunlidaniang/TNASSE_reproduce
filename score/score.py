from OON import *
from get_data import get_cifar_10_loader
def get_score(code):
    spec=CodeToSpec(code)
    network=SpecToNet(spec)
    loader=get_cifar_10_loader()
    data_iterator=iter(loader)
    input, label=next(data_iterator)
    print('here')
    score=NetToScore(network,input)
    return score

l1=[2,3,1,1,2]
l2=[1,0,1,0,1,1,1,1,0,0,0,0,1,0,0,0,1,0,1,0,1]
l3=[0,0,1,0,1]
code=(l1,l2,l3)
print(get_score(code))
