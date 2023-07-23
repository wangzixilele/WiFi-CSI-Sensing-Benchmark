import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader


def UT_HAR_dataset(root_dir):
    #the UT_HAR stands for the HAR(Human Activity Recognition, 人体行为识别) for University of Toronto
    #here defind a method to import the data of the dataset and regular it

    # glob is a moudle that can find the path of the file
    # the function of glob.glob is to find the path of the file that match the rule
    # In this case, the rule is root_dir+'/UT_HAR/data/*.csv'
    # the root_dir is the path of the dataset

    #this root_dir is set at util.py as data = UT_HAR_dataset(root) under the function"load_data_n_model"
    #and run in the run.py which set the root as './Data/'
    #so the structure of the file should be:
    #Benchmark(the running code)
    # ├── Data
    #     ├── NTU-Fi_HAR
    #     │   ├── test_amp
    #     │   ├── train_amp
    #     ├── NTU-Fi-HumanID
    #     │   ├── test_amp
    #     │   ├── train_amp
    #     ├── UT_HAR
    #     │   ├── data
    #     │   ├── label
    #     ├── Widardata
    #     │   ├── test
    #     │   ├── train
    data_list = glob.glob(root_dir+r'\UT_HAR\data\*.csv')
    label_list = glob.glob(root_dir+r'\UT_HAR\label\*.csv')
    WiFi_data = {}

    #the WiFi_data is a dictionary that store the data and label
    #the key is the name of the data and the value is the data be tensor

    for data_dir in data_list:
        data_name = data_dir.split('\\')[-1].split('.')[0]
        #this is different from the original because of the different split mark '\\' and '/'
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            #the readin form is the numpy.ndarray which is tensor in numpy
            #and the original shape of the data is 500, 250, 90
            #the 500 is the number of the data in the reshape is the len(data)
            #so in here the reshape is to add one dimension(1 choice) to the data
            data = data.reshape(len(data),1,250,90)
            #norm is a function that can normalize the data
            #this is call min-max normalization(最大最小值归一化:因为结果一定会在[0,1]之间)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        WiFi_data[data_name] = torch.Tensor(data_norm)
        #the tensor is already normalized
        #torch.Tensor is a function that can transform the data to tensor(张量)
    for label_dir in label_list:
        label_name = label_dir.split('\\')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        WiFi_data[label_name] = torch.Tensor(label)
    return WiFi_data


# dataset: /class_name/xx.mat
class CSI_Dataset(Dataset):
    """CSI dataset."""

    def __init__(self, root_dir, modal='CSIamp', transform=None, few_shot=False, k=5, single_trace=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            modal (CSIamp/CSIphase): CSI data modal
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.modal=modal
        self.transform = transform
        self.data_list = glob.glob(root_dir+'\\*\\*.mat')
        self.folder = glob.glob(root_dir+'\\*\\')
        self.category = {self.folder[i].split('\\')[-2]:i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('\\')[-2]]
        x = sio.loadmat(sample_dir)[self.modal]
        
        # normalize
        x = (x - 42.3199)/4.9802
        
        # sampling: 2000 -> 500
        x = x[:,::4]
        x = x.reshape(3, 114, 500)
        
        if self.transform:
            x = self.transform(x)
        
        x = torch.FloatTensor(x)

        return x,y


class Widar_Dataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.data_list = glob.glob(root_dir+'/*/*.csv')
        self.folder = glob.glob(root_dir+'/*/')
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = np.genfromtxt(sample_dir, delimiter=',')
        
        # normalize
        x = (x - 0.0025)/0.0119
        
        # reshape: 22,400 -> 22,20,20
        x = x.reshape(22,20,20)
        # interpolate from 20x20 to 32x32
        # x = self.reshape(x)
        x = torch.FloatTensor(x)

        return x,y

