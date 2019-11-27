import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
class DataGenerator(Dataset):
    'Generates data for Keras'
    def __init__(self, df, batch_size=1, dim=(480,640), n_channels=3,im_type = "std_cam",
                 class_list=['1PO-A', '2PO1-A', '2PO2-A', 'CR-A', 'KT-A', 'LO-A', 'PA-A', 'ST-A', 'TL-A'],time_steps = 32):
        super(DataGenerator, self).__init__()
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.im_type = im_type
        self.n_channels = n_channels
        self.class_list = class_list
        self.time_steps = time_steps

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.df)))

    def __getitem__(self, index):

        X, y = self.__data_generation(index)

        return X, y


    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        y = np.zeros(9)
        indices = np.arange(self.time_steps) + index*self.time_steps
        # Generate data
        im = Image.open(os.getcwd()+"/data/" +self.im_type +"/"+self.df.iloc[index,0])
        X =np.swapaxes(np.swapaxes(np.array(im),0,2),1,2)
        label = self.df.iloc[index,1]
        y[self.class_list.index(label)] = 1
        X = torch.from_numpy(X/255).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.IntTensor)
        return X, y

