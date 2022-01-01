import pickle

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from matplotlib import cm
import numpy as np
import imgaug.augmenters as iaa


def read_pkl(pkl_path):
    with open(pkl_path,'rb') as f:
        pkl_data=pickle.load(f)
        X_Train=pkl_data['train']
        Y_Train=pkl_data['train_label']
        X_Ori=pkl_data['original']

    return X_Ori,X_Train,Y_Train


class PklDataset(Dataset):

    def __init__(self,X_ori,X_Train,Y_Train, transform=None):
        self.transform = transform
        self.X_Ori=X_ori
        self.X_Train=X_Train
        self.Y_Train=Y_Train

    def __len__(self):
        return len(self.X_Train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ori_x = np.expand_dims(self.X_Ori[idx], axis=0)

        x = np.expand_dims(self.X_Train[idx], axis=0)

        nptrans= iaa.Resize({"height":32,"width":32})
        x=nptrans(images=x)
        ori_x=nptrans(images=ori_x)
        x=torch.tensor(x).squeeze(0).float()/255.0
        ori_x=torch.tensor(ori_x).squeeze(0).float()/255.0
        if len(ori_x.size())==2:
            ori_x=ori_x.unsqueeze(axis=2)
            ori_x=ori_x.repeat(1,1,3)

        x=x.permute(2,0,1)
        ori_x=ori_x.permute(2,0,1)
        y = self.Y_Train[idx]
        if self.transform:
          x=self.transform(x)
          ori_x=self.transform(ori_x)
        return ori_x,x,y

def get_emnist_m(trans=None):
    pkl_path="./emnistm_data.pkl"
    X_ori,X_Train,Y_Train=read_pkl(pkl_path)

    return PklDataset(X_ori,X_Train,Y_Train,trans)

def get_mnist_m(trans=None):
    pkl_path="./mnistm_data.pkl"
    X_ori,X_Train,Y_Train=read_pkl(pkl_path)

    return PklDataset(X_ori,X_Train,Y_Train,trans)



def get_fmnist_m(trans=None):
    pkl_path="./fashion_m_data.pkl"
    X_ori,X_Train,Y_Train=read_pkl(pkl_path)

    return PklDataset(X_ori,X_Train,Y_Train,trans)




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    a=get_fmnist_m()
    for n,(x,y,z) in enumerate(a):
      plt.imshow( y.permute(1, 2, 0)  )
      plt.show()
      x=x.squeeze(0)
      print(x.size())
      plt.imshow( x)

      plt.show()
      if n>5:
        break