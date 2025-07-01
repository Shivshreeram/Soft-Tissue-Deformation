import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


#put your own path here
c_path = '/content/drive/MyDrive/Image dataset/compression'
r_path = '/content/drive/MyDrive/Image dataset/rigid'
t_path = '/content/drive/MyDrive/Image dataset/tension'

c = []
r = []
t = []

for i in glob.glob(c_path+'/*.png'):
  img = cv2.imread(i)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = np.array(img)
  img = img/255
  c.append(img)

for i in glob.glob(r_path+'/*.png'):
  img = cv2.imread(i)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = np.array(img)
  img = img/255
  r.append(img)

for i in glob.glob(t_path+'/*.png'):
  img = cv2.imread(i)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = np.array(img)
  img = img/255
  t.append(img)


r_final = []
c_final = []
t_final = []

x = 0
while x < 1000:
  c_final.append((c[x+1]-c[x]))
  r_final.append((r[x+1]-r[x]))
  t_final.append((t[x+1]-t[x]))

c_final = np.array(c_final)
r_final = np.array(r_final)
t_final = np.array(t_final)

from sklearn.utils import shuffle

c_labels = np.zeros(len(c_final))  # Label 0 for class c
r_labels = np.ones(len(r_final))   # Label 1 for class r
t_labels = np.full(len(t_final), 2) # Label 2 for class t

b = 0.1 # b is the proportion of images for train-val set
b = 500*b # 500 is the total no. of inputs per label (become half as we combine concecutive img to one)

X_train = np.concatenate((c_final[:b], r_final[:b], t_final[:b]))
y_train = np.concatenate((c_labels[:b], r_labels[:b], t_labels[:b])) 

X_train, y_train = shuffle(X_train, y_train, random_state=100)

X_val = np.concatenate((c_final[b:], r_final[b:], t_final[b:]))
y_val = np.concatenate((c_labels[b:], r_labels[b:], t_labels[b:])) 

X_train, y_train = shuffle(X_train, y_train, random_state=100)
X_val, y_val = shuffle(X_val, y_val, random_state=100)

#CNN model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.regularizers import l2

model = keras.Sequential([
    keras.Input(shape=(384, 192, 1)),

    layers.Conv2D(32, kernel_size=(4, 4), padding="same", kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    LeakyReLU(),
    layers.MaxPooling2D(pool_size=(3, 3)),
    layers.Dropout(0.5),

    layers.Conv2D(64, kernel_size=(4, 4), padding="same", kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    LeakyReLU(),
    layers.MaxPooling2D(pool_size=(3, 3)),
    layers.Dropout(0.5),

    layers.Conv2D(128, kernel_size=(4, 4), padding="same", kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    LeakyReLU(),
    layers.MaxPooling2D(pool_size=(3, 3)),
    layers.Dropout(0.5),

    layers.Conv2D(256, kernel_size=(4, 4), padding="same", kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    LeakyReLU(),
    layers.MaxPooling2D(pool_size=(3, 3)),
    layers.Dropout(0.5),

    layers.Flatten(),
    layers.Dense(128, kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    LeakyReLU(),
    layers.Dropout(0.5),

    layers.Dense(64, kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    LeakyReLU(),
    layers.Dropout(0.5),

    layers.Dense(3, activation="softmax"),
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["accuracy"],
)

# model.summary()

# Hyper parameters

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',    
    factor=0.5,          # Reduce learning rate by a factor of 0.5
    patience=5,             # Wait for 5 epochs without improvement to reduce LR
    verbose=1               # Show information about LR adjustments
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',     
    patience=10, # Stop training after 15 epochs without improvement

    restore_best_weights=True,  # Restore the best weights during training

    verbose=1               # Show information about early stopping
)

history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=200,             # Train for up to 200 epochs
    validation_data=(X_val, y_val),
    callbacks=[lr_scheduler, early_stopping] 
)


