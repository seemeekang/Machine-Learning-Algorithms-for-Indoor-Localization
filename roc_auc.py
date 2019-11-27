import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd

from sklearn.metrics import roc_curve, auc
import pickle as pck

from scipy import interp
import os
lw = 2

Illum  = False
im_type = "omni_cam"

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

#change here for different data
predictions = pck.load(open(os.getcwd()+"/preds/omni_camCNN.npy", "rb" ))
class_list = ['1PO-A','2PO1-A','2PO2-A','CR-A','KT-A','LO-A','PA-A','ST-A','TL-A']
predicted_indices= np.argmax(predictions,axis = 1)
predicted_labels = np.zeros(shape=predictions.shape)
labels_str = list(test_df["label"])
for i in range(len(predicted_indices)):
    predicted_labels[i,int(predicted_indices[i])] = 1
labels = np.zeros(shape=(len(test_df["label"]),len(class_list)))
for i in range(len(test_df["label"])):
    ind = class_list.index(labels_str[i])
    labels[i,ind] = 1
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(9):
    fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), predictions.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(9)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(9):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 9

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()


plt.plot(fpr["macro"], tpr["macro"],
         label='CNN (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='deeppink', linestyle=':', linewidth=4)




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

#change here for different data
predictions = pck.load(open(os.getcwd()+"/preds/predtest38.npy", "rb" ))
class_list = ['1PO-A','2PO1-A','2PO2-A','CR-A','KT-A','LO-A','PA-A','ST-A','TL-A']
predicted_indices= np.argmax(predictions,axis = 1)
predicted_labels = np.zeros(shape=predictions.shape)
labels_str = list(test_df["label"])
for i in range(len(predicted_indices)):
    predicted_labels[i,int(predicted_indices[i])] = 1
labels = np.zeros(shape=(len(test_df["label"]),len(class_list)))
for i in range(len(test_df["label"])):
    ind = class_list.index(labels_str[i])
    labels[i,ind] = 1
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(9):
    fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), predictions.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(9)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(9):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 9

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves



plt.plot(fpr["macro"], tpr["macro"],
         label='CNN-NDF(area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])



plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for omnidirectional images with mixed training data')
plt.legend(loc="lower right")

plt.show()