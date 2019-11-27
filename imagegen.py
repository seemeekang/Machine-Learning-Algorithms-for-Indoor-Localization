import numpy as np
import keras
import pandas as pd
from PIL import Image
import os
class DataGenerator(keras.utils.Sequence):

    def __init__(self, df, batch_size=1, dim=(480,640), n_channels=3,im_type = "std_cam",
                 class_list=['1PO-A', '2PO1-A', '2PO2-A', 'CR-A', 'KT-A', 'LO-A', 'PA-A', 'ST-A', 'TL-A'],time_steps = 32):

        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.im_type = im_type
        self.n_channels = n_channels
        self.class_list = class_list
        self.time_steps = time_steps

    def __len__(self):
        'Denotes the number of steps per epoch'
        return int(np.floor(len(self.df) / self.time_steps))

    def __getitem__(self, index):

        X, y = self.__data_generation(index)

        return X, y


    def __data_generation(self, index):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.zeros(shape=(self.batch_size,self.time_steps, *self.dim, self.n_channels))
        y = np.zeros(shape = (self.batch_size,self.time_steps,len(self.class_list)))
        indices = np.arange(self.time_steps) + index*self.time_steps
        # Generate data
        for i, ind in enumerate(indices):
            im = im = Image.open(os.getcwd()+"/data/" +self.im_type +"/"+self.df.iloc[ind,0])
            im_array = np.array(im)
            X[0,i,:,:,:] = im_array
            label = self.df.iloc[ind,1]
            y[0,i,self.class_list.index(label)] = 1

        return X, y

