
from sklearn.metrics import accuracy_score,recall_score
import pickle as pck
import random
import numpy as np
random.seed(1279)
import pandas as pd
import os
Illum  = True
im_type = "std_cam"


if Illum:
    test_df =  pd.read_csv(os.getcwd()+"/test_sc.txt",header=None)
    test_df = test_df.iloc[1:,:]
    test_df.columns = ["filename", "label"]

    model_dir = "i"+im_type
else:
    test_df = pd.read_csv(os.getcwd() + "/test_std.txt", header=None)
    test_df = test_df.iloc[1:, :]
    test_df.columns = ["filename", "label"]

    model_dir = im_type

#change path for different model predictions
predictions = pck.load(open(os.getcwd()+"/preds/istd_camCNN12.npy", "rb" ))

class_list = ['1PO-A','2PO1-A','2PO2-A','CR-A','KT-A','LO-A','PA-A','ST-A','TL-A']
predicted_indices= np.argmax(predictions,axis = 1)
predicted_labels = np.zeros(shape=predictions.shape)
for i in range(len(predicted_indices)):
    predicted_labels[i,int(predicted_indices[i])] = 1
labels = np.zeros(shape=(len(test_df["label"]),len(class_list)))
labels_str = list(test_df["label"])
for i in range(len(test_df["label"])):
    ind = class_list.index(labels_str[i])
    labels[i,ind] = 1

labels_ind = np.nonzero(labels>0)
labels_ind = list(labels_ind[1])

pred_ind = np.nonzero(predicted_labels>0)
pred_ind = list(pred_ind[1])


print(recall_score(labels,predicted_labels,average="micro"))
print(recall_score(labels,predicted_labels,average="macro"))
print(recall_score(labels,predicted_labels,average="weighted"))
print(accuracy_score(labels,predicted_labels))

