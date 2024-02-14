import cv2
import os
import numpy as np
import time
import scipy
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Input,Dropout,GlobalAveragePooling2D,Flatten,Conv2D,BatchNormalization,Activation,MaxPooling2D
from keras.models import Model,Sequential
from keras.optimizers import Adam,SGD,RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from keras.preprocessing import image
from keras.models import load_model

from pathlib import Path
from joblib import load


print('''****************************************************************
        \n***** PROCESAMIENTO DE IMAGENES - MOSQUITOS AEDES vs CULEX *****
        \n*****************************************************************''')

image_path = input('Ingrese la ubicacion de la imagen: ')
# E:/Estudios/DMC/Especializacion MLE/Dataset/images/val

model = tf.keras.models.load_model('E:/Estudios/DMC/Especializacion MLE/models/color/vgg16_model.h5')

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
picture_size = 96
no_of_classes = 1

image_val = datagen.flow_from_directory(    image_path, 
                                            target_size = (picture_size,picture_size),
                                            class_mode = 'categorical',
                                            batch_size = 1,
                                            shuffle = False)

for x_batch, y_batch in image_val:
    processed_image = x_batch
    break

predictions = model.predict(image_val)

first_column = predictions[:, 0]

if predictions[:, 0]>0.5:
    print('El mosquito es Aedes', round(float(first_column[0]*100), 2), '% de probabilidad')
else:
    print('El mosquito es Culex con un', round(float((1-first_column[0])*100), 2), '% de probabilidad')

# Mostrar la imagen procesada
plt.imshow(processed_image[0])  # processed_image es un tensor, tomamos el primer elemento
plt.axis('off')  # Desactiva los ejes
plt.show()