import pandas as pd
import os

import random
random.seed(1279)

dir_name = "/data/data_shuffled.txt"
load_images = pd.read_csv(os.getcwd()+dir_name,header=None)
load_images = load_images.iloc[1:,:]
load_images.columns = ["filename","label"]

class_list = ['1PO-A','2PO1-A','2PO2-A','CR-A','KT-A','LO-A','PA-A','ST-A','TL-A']
label_list = list(load_images["label"])
indices_tr =[]
indices_val = []
indices_test = []
class_weights = {}
cnt = 0
for c in class_list:
    ind_list=[]
    for i in range(len(label_list)):
        if label_list[i] ==c:
            ind_list.append(i)
    random.shuffle(ind_list)

    indices_test.extend(ind_list[int(len(ind_list)*0.5):])
    ind_temp = ind_list[0:int(len(ind_list)*0.5)]
    indices_val.extend(ind_temp[int(len(ind_temp)*0.75):])
    indices_tr.extend(ind_temp[0:int(len(ind_temp)*0.75)])
    ## after seeing frequency of classes, we divide our class weights by the largest number to obtain normalized class weight
    class_weights[cnt] = (len(label_list)/(2*len(ind_temp)))/20.27900552486188
    cnt = cnt +1

print(class_weights)
val_df = load_images.iloc[indices_val,:]
train_df = load_images.iloc[indices_tr,:]
test_df = load_images.iloc[indices_test,:]

train_df.to_csv(os.getcwd()+"/train_std.txt",index=False)
val_df.to_csv(os.getcwd()+"/val_std.txt",index=False)
test_df.to_csv(os.getcwd()+"/test_std.txt",index=False)


