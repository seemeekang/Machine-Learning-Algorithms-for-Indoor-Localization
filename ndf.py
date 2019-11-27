import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F




class ColdFeatureLayer(nn.Sequential):
    def __init__(self,dropout_rate,shallow=False):
        super(ColdFeatureLayer, self).__init__()
        self.shallow = shallow
        if shallow:
            self.add_module('conv1', nn.Conv2d(3, 1, kernel_size=15,padding=1,stride=15))
        else:
            self.add_module('conv1', nn.Conv2d(3, 5, kernel_size=10, padding=1))
            self.add_module('relu1', nn.ReLU())
            self.add_module('bn1',nn.BatchNorm2d(5))
            self.add_module('pool1', nn.MaxPool2d(kernel_size=2))

            self.add_module('conv2', nn.Conv2d(5, 5, kernel_size=5, padding=1))
            self.add_module('relu2', nn.ReLU())
            self.add_module('bn2', nn.BatchNorm2d(5))
            self.add_module('pool2', nn.MaxPool2d(kernel_size=2))
            self.add_module('drop2', nn.Dropout(dropout_rate))

            self.add_module('conv3', nn.Conv2d(5, 5, kernel_size=5, padding=1))
            self.add_module('relu3', nn.ReLU())
            self.add_module('bn3', nn.BatchNorm2d(5))
            self.add_module('pool3', nn.MaxPool2d(kernel_size=2))
            self.add_module('drop3', nn.Dropout(dropout_rate))

    def get_out_feature_size(self):
        if self.shallow:
            return 1344
        else:
            return 21945
