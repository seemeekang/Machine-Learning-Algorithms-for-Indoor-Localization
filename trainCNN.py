import pandas as pd
import os
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import rmsprop
from keras.callbacks import ModelCheckpoint,EarlyStopping
import random
import numpy as np
random.seed(1279)


Illum  = True
im_type = "omni_cam"
#prepare data set and class weights
if Illum:
    train_df =  pd.read_csv(os.getcwd()+"/train_night.txt",header=None)
    train_df = train_df.iloc[1:,:]
    train_df.columns = ["filename", "label"]
    val_df = pd.read_csv(os.getcwd() + "/val_night.txt", header=None)
    val_df = val_df.iloc[1:, :]
    val_df.columns = ["filename", "label"]
    class_weights = [0.837962962962963, 0.5709779179810726, 1.0, 0.11837802485284499, 0.7327935222672064, 0.7903930131004366,
          0.5041782729805013, 0.6830188679245283, 0.5552147239263803]
    model_dir = "i"+im_type
else:
    train_df = pd.read_csv(os.getcwd() + "/train_std.txt", header=None)
    train_df = train_df.iloc[1:, :]
    train_df.columns = ["filename", "label"]
    val_df = pd.read_csv(os.getcwd() + "/val_std.txt", header=None)
    val_df = val_df.iloc[1:, :]
    val_df.columns = ["filename", "label"]
    class_weights = [ 0.9202898550724637, 0.42905405405405406, 1.0, 0.11441441441441443, 0.4669117647058824,
                      0.8881118881118882, 0.4057507987220448, 0.5772727272727273, 0.49416342412451364]
    model_dir = im_type
class_list = ['1PO-A','2PO1-A','2PO2-A','CR-A','KT-A','LO-A','PA-A','ST-A','TL-A']


#Data generator
train_datagen = ImageDataGenerator(rescale=1./255,data_format="channels_last")
val_datagen = ImageDataGenerator(rescale=1./255,data_format="channels_last")
test_datagen = ImageDataGenerator(rescale=1./255,data_format="channels_last")
train_generator = train_datagen.flow_from_dataframe(
        dataframe = train_df,
        directory = os.getcwd()+"/data/"+im_type+"/",
        x_col = "filename",
        y_col = "label",
        batch_size=32,
        target_size=(480,640),
        class_mode='categorical',
        classes = ['1PO-A','2PO1-A','2PO2-A','CR-A','KT-A','LO-A','PA-A','ST-A','TL-A']
        )

val_generator = val_datagen.flow_from_dataframe(
        dataframe = val_df,
        directory = os.getcwd()+"/data/"+im_type+"/",
        x_col = "filename",
        y_col = "label",
        batch_size=32,
        target_size=(480,640),
        class_mode='categorical',
        classes=['1PO-A', '2PO1-A', '2PO2-A', 'CR-A', 'KT-A', 'LO-A', 'PA-A', 'ST-A', 'TL-A'])

#model
model = Sequential()
model.add(Conv2D(5, (10, 10), padding='same',input_shape=(480,640,3),data_format="channels_last"))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),data_format="channels_last"))

model.add(Conv2D(5, (5, 5),data_format="channels_last"))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),data_format="channels_last"))
model.add(Dropout(0.25))

model.add(Conv2D(5, (5, 5),data_format="channels_last"))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),data_format="channels_last"))
model.add(Dropout(0.25))



model.add(Flatten())
model.add(Dense(9))
model.add(Activation('softmax'))
model.summary()

opt = rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['categorical_accuracy'])

checkpointer = ModelCheckpoint(os.getcwd()+"/models/"+model_dir+"/weights.{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5", monitor='val_categorical_accuracy',
                               verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=100,callbacks=[checkpointer],class_weight=class_weights)

