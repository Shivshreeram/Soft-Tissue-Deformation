import cv2
import numpy as np
import glob
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and preprocess grayscale images (compression images)
c_path = '/content/drive/MyDrive/Image dataset/compression'
c = []
for i in glob.glob(c_path + '/*.png'):
    img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
    img = img / 255.0
    c.append(img)

# Compute difference between consecutive image pairs
c_final = []
x = 0
while x < 1000:
    c_final.append((c[x + 1] - c[x]))
    x += 2
c_final = np.array(c_final) * 100

# Load and preprocess displacement maps
disp_a = []
for d in glob.glob("/content/drive/MyDrive/Image dataset/com_strain_disp/displacements/*.npy"):
    array = np.load(d)
    array = cv2.resize(array, (array.shape[1] * 2, array.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
    array = array / 255.0
    disp_a.append(array)

# Separate x and y displacement maps
y_x, y_y = [], []
for i in range(len(disp_a)):
    if i % 2 == 0:
        y_x.append(disp_a[i] * 100)
    else:
        y_y.append(disp_a[i] * 100)
y_x = np.array(y_x)
y_y = np.array(y_y)

# Prepare input and output arrays
X = np.array(c_final)
y = np.stack([y_x, y_y], axis=-1)
X = X.reshape(-1, X.shape[1], X.shape[2], 1)
X, y = shuffle(X, y, random_state=42)

# U-Net Model Definition
def conv_block(x, filters, dropout_rate=0.5, l2_reg=0.0005):
    x = layers.Conv2D(filters, (3, 3), padding="same", kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.4)(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.4)(x)
    x = Dropout(dropout_rate)(x)
    return x

def encoder_block(x, filters):
    x = conv_block(x, filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(x, skip_connection, filters):
    x = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.4)(x)
    x = layers.Concatenate()([x, skip_connection])
    x = conv_block(x, filters)
    return x

def build_unet(input_shape=(384*2, 192*2, 1)):
    inputs = layers.Input(input_shape)
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)
    s5, p5 = encoder_block(p4, 512)
    s6, p6 = encoder_block(p5, 1024)
    bottleneck = conv_block(p6, 2048)
    d1 = decoder_block(bottleneck, s6, 1024)
    d2 = decoder_block(d1, s5, 512)
    d3 = decoder_block(d2, s4, 256)
    d4 = decoder_block(d3, s3, 128)
    d5 = decoder_block(d4, s2, 64)
    d6 = decoder_block(d5, s1, 32)
    output1 = layers.Conv2D(1, (1, 1), activation=LeakyReLU(alpha=0.2), name="output1")(d6)
    output2 = layers.Conv2D(1, (1, 1), activation=LeakyReLU(alpha=0.2), name="output2")(d6)
    model = Model(inputs, [output1, output2], name="U-Net_Two_Outputs")
    return model

unet_model = build_unet()

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Callbacks
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Compile model
unet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss={"output1": tf.keras.losses.Huber(), "output2": tf.keras.losses.Huber()},
    metrics={"output1": ["mae", "mse"], "output2": ["mae", "mse"]}
)

# Model training
history = unet_model.fit(
    X_train,
    {"output1": y_train[..., 0], "output2": y_train[..., 1]},
    batch_size=32,
    epochs=200,
    validation_data=(X_val, {"output1": y_val[..., 0], "output2": y_val[..., 1]}),
    callbacks=[lr_scheduler, early_stopping]
)

# Optional: Save model
# unet_model.save('/content/drive/MyDrive/unet_model.keras')
