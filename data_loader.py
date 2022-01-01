import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils import data

from pkl_loader import get_emnist_m, get_fmnist_m,get_mnist_m




def get_loader_m(conf):

    trans= transforms.Compose([
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    transform2 = transforms.Compose([
                    transforms.Scale(conf.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))
                    ])

    if conf.m_dataset=='f':
        temp_dataset=get_fmnist_m(trans)
    elif conf.m_dataset=='e':
        temp_dataset=get_emnist_m(trans)

    elif conf.m_dataset=='m':
        temp_dataset=get_mnist_m(trans)


    return  data.DataLoader(dataset=temp_dataset,
                                   batch_size=conf.batch_size,
                                   shuffle=True,
                                   num_workers=0)







def spilit_dataset(dataset):

    dataset_size1=int(0.5*len(dataset))
    dataset_size2=len(dataset)-dataset_size1
    dataset1, dataset2 = torch.utils.data.random_split(dataset, [dataset_size1, dataset_size2])

    return dataset1,dataset2





