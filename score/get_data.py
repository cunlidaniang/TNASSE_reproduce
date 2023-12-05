import os
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms

def get_cifar_10_loader():
    root=os.path.abspath(__file__)
    root=os.path.dirname(root)
    root=root+'/cifar-10'
    transform=transforms.Compose([transforms.ToTensor()])
    train_data=dset.CIFAR10(root,train=True,download=True,transform=transform)
    data_loader=data.DataLoader(dataset=train_data,batch_size=16,shuffle=True,pin_memory=True)
    return data_loader

data_loader=get_cifar_10_loader()
data_iterator = iter(data_loader)
input , label = next(data_iterator)
print(input.shape,label.shape)
print(label)