import tensorflow as tf
import pandas as pd
import os
import cv2
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, add, Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Flatten, Dropout, Lambda
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Convolution2D, Cropping2D, ELU
from tensorflow.keras.layers import BatchNormalization

def load_image(filename):
    img = cv2.imread(filename)
    return img


colnames = ['image', 'velocity', 'steer']
data = pd.read_csv('etykiety.csv', names=colnames, header=None, delimiter=' ')
batch_size = 16

# train path to one sections !
dataset = []
# dataset_path = '/home/legion/PycharmProjects/AV/src/av_04/images'
dataset_path = '/av_ws/src/av_04/images'
# dataset_path = '/home/legion/PycharmProjects/AV/src/av_04/images'
#380 #550
path_list = os.listdir(dataset_path)
path_list = sorted(path_list, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
# img_height = 100
# img_width = 100
for each in path_list:
    print(each)
    each = os.path.join(dataset_path, each)
    img = load_image(each)
    img = cv2.resize(img, (200, 42))
    # img = img[380:550, :]
    # cv2.imwrite(each, img)
    dataset.append(img)

X = dataset
X = np.array(X)
# data.loc[data['steer'] > 0, 'steer'] = 1
# data.loc[data['steer'] < 0, 'steer'] = -1
data['steer'] = np.round(data['steer'], 2)
data['velocity'] = np.round(data['velocity'], 2)
y = data[['velocity', 'steer']]
print(y)
num_output_classes = 2

X_train, X_valid, Y_train, Y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

train_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255, zoom_range=[0.75, 1])
train_generator_with_data = train_generator.flow(X_train, Y_train, batch_size=batch_size, shuffle=True)
valid_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
valid_generator_with_data = valid_generator.flow(X_valid, Y_valid, batch_size=32, shuffle=True)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Lambda, Cropping2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(42, 200, 3)))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(2))
model.compile(loss='mse', optimizer='adam')
learning_rate = 3e-4
optimizer = Adam(learning_rate)

model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['accuracy'])
nb_epochs = 2

training_samples = train_generator_with_data.n
validation_samples = valid_generator_with_data.n

model.summary()

model.fit_generator(
    train_generator_with_data,
    steps_per_epoch=training_samples // 32,
    validation_data=valid_generator_with_data,
    validation_steps=validation_samples // 32,
    epochs=nb_epochs)

# Save the weights
ResNet32.save('cnn_models/model_resnet32.h5')
