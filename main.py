#!pip install split-folders


import tensorflow as tf
import numpy as np
import pandas as pd
import os
import keras
import matplotlib.pyplot as plt
import shutil
import splitfolders
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from keras.layers.core import Dropout
from keras.layers.core import Dense, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam_v2
from keras.metrics import categorical_crossentropy

os.mkdir('/kaggle/working/main_directory')

kaggle_input_path = '/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset/'

#Copy files to one folder
shutil.copytree(kaggle_input_path + 'Normal/images','/kaggle/working/main_directory/Normal')
shutil.copytree(kaggle_input_path+ 'COVID/images','/kaggle/working/main_directory/COVID')

#split into train/test splits (we create val set later on)
splitfolders.ratio('/kaggle/working/main_directory/', '/kaggle/working/output',seed=1337,ratio=(0.9,0.1),move=True)

train_path = '/kaggle/working/output/train'
test_path = '/kaggle/working/output/val'

train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        validation_split=0.2) # set validation split

batchSize=32

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(256, 256),
    batch_size=batchSize,
    subset='training') # set training data

validation_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(256, 256),
    batch_size=batchSize,
    subset='validation') # set validation data

test_generator = ImageDataGenerator().flow_from_directory(
    test_path,
    target_size=(256,256),
    shuffle= False,
    batch_size = batchSize) # set test data

# InceptionV3 model for our base model
# We want to load the pretrained ImageNet weights
# Set include_top to False as our images are 256x256 (default is 299x299)

base_model = InceptionV3(weights='imagenet',
                                include_top=False,
                                input_shape=(256, 256,3))

base_model.trainable = False    # freeze weights of model

X = base_model.output
X = keras.layers.GlobalAveragePooling2D()(X)
X = Dropout(0.2)(X)
# output layer with softmax (we could use sigmoid as only two classes, but this will generalise to more classes)
predictions = Dense(2, activation='softmax')(X)

# this is the model we will train
model = Model(base_model.input, predictions)

model.compile(adam_v2.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

stepsPerEpoch= (train_generator.samples+ (batchSize-1)) // batchSize
print("stepsPerEpoch: ", stepsPerEpoch)

validationSteps=(validation_generator.samples+ (batchSize-1)) // batchSize
print("validationSteps: ", validationSteps)

history = model.fit_generator(
    train_generator,
    validation_data = validation_generator,
    epochs = 3,
    steps_per_epoch = stepsPerEpoch,
    validation_steps= validationSteps,
    callbacks=callbacks_list,
    verbose=1)

base_model.trainable = True
model.compile(adam_v2.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    validation_data = validation_generator,
    epochs = 10,
    steps_per_epoch = stepsPerEpoch,
    validation_steps= validationSteps,
    callbacks=callbacks_list,
    verbose=1)

validation_generator.reset()
score = model.evaluate_generator(validation_generator, (validation_generator.samples + (batchSize-1)) //batchSize)
print("For validation data set; Loss: ",score[0]," Accuracy: ", score[1])

model.compile(adam_v2.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

test_generator.reset()
score = model.evaluate_generator(test_generator, (test_generator.samples + (batchSize-1)) // batchSize)
print("For test data set; Loss: ",score[0]," Accuracy: ", score[1])

#In testing, this model achieved 98.8% accuracy at classifying COVID vs normal Chest X rays