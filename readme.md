## Dataset
Download dataset using the link https://www.nada.kth.se/cas/COLD/downloads.php.
Freiburg Lab, Part A, Path 2 (extended), sequence 1 of cloud, night and sunny.
All text files listed below includes image file names and corresponding labels.
1. train_std.txt, val_std.txt, test_std.txt : shuffled mixed training, validation and test set
2. train_night.txt, val_night.txt, test_sc.txt: night data for training and validation, sunny dand cloudy data for test.

## Scripts
    prepData.py: For train, validation and test splits
    trainCNN.py,train_ConvLSTM.py,trainCNN-NDF.py: Run these files to train corresponding algorithms. Note that we obtain test results after training CNN-NDF, but we save models for CNN and ConvLSTM
    imagegen.py: data generator for convLSTM
    dataset.py: data generator for
    ndf.py: CNN-NDF model
    predict.py: produce predictions for chosen model and saves predictions to a specified file
    predFromArr.py: loads saved prediction arrays and calculates accuracy and recall
    roc_auc.py: Produce ROC plot and auc score
---
## Requirements

1. Pytorch
2. pandas
3. numpy
4. keras
5. sklearn
6. scipy
7. matplotlib
8. PIL
9. opencv
