import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import Sequential
from keras.layers import ConvLSTM2D,TimeDistributed,Dense,Flatten,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from imagegen2 import DataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping
import numpy as np
import pandas as pd


dir_name = "/train_night.txt"
df_train= pd.read_csv(os.getcwd()+dir_name,header=None)
df_train= df_train.iloc[1:,:]

dir_name = "/val_night.txt"
df_val= pd.read_csv(os.getcwd()+dir_name,header=None)
df_val= df_val.iloc[1:,:]
dir_name = "/test_sc.txt"
df_test= pd.read_csv(os.getcwd()+dir_name,header=None)
df_test= df_test.iloc[1:,:]

df_train = pd.concat([df_train,df_val])
training_generator = DataGenerator(df_train)
validation_generator = DataGenerator(df_val[0:2304])
class_weights = [0.837962962962963, 0.5709779179810726, 1.0, 0.11837802485284499, 0.7327935222672064, 0.7903930131004366,
          0.5041782729805013, 0.6830188679245283, 0.5552147239263803]

seq = Sequential()
seq.add(ConvLSTM2D(filters=5, kernel_size=(10, 10),
                   input_shape=(None, 480, 640, 3),padding='same',strides=(2,2), return_sequences=True,data_format="channels_last"))
seq.add(BatchNormalization())
seq.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format="channels_last")))

seq.add(ConvLSTM2D(filters=5, kernel_size=(5, 5),padding='same', return_sequences=True,data_format="channels_last"))
seq.add(BatchNormalization())
seq.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format="channels_last")))

seq.add(ConvLSTM2D(filters=5, kernel_size=(5, 5),padding='same', return_sequences=False,data_format="channels_last"))
seq.add(BatchNormalization())
seq.add(TimeDistributed(MaxPooling2D(pool_size=( 2, 2), strides=None, padding='valid', data_format="channels_last")))

seq.add(TimeDistributed(Flatten()))
seq.add(TimeDistributed(Dense(9,activation='softmax')))


seq.summary()


seq.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['categorical_accuracy'])
checkpointer = ModelCheckpoint(os.getcwd()+"/models/std/weights.{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5", monitor='val_categorical_accuracy',
                               verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
seq.fit_generator(generator=training_generator,
                    validation_data=validation_generator,epochs=100,validation_steps=int(np.floor(len(df_val)-32)),callbacks=[checkpointer],class_weight=class_weights)
