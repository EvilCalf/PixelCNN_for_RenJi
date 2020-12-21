import os

import numpy as np
from keras.utils import to_categorical
from tensorflow.python import keras
from tensorflow.python.keras import layers
import pandas as pd
from glob import glob
from tensorflow.python.ops.gen_math_ops import Xlogy
from tqdm import tqdm
import PIL
from generate_data import generate_arrays_from_file

img_size = 400
num_classes = 2
input_shape = (img_size, img_size, 3)
n_residual_blocks = 5
batch_size = 2
train_path = 'train'  # 根据自己的目录修改
train_num = len(glob(train_path + '/*/*.jpg'))

X = np.zeros((train_num, img_size, img_size, 3), dtype=np.uint8)
Y = np.zeros((train_num,), dtype=np.uint8)
i = 0
for img_path in tqdm(glob(train_path + '/*/*.jpg')):
    img = PIL.Image.open(img_path)
    img = img.resize((img_size, img_size))  # 图片resize
    arr = np.asarray(img)  # 图片转array
    X[i, :, :, :] = arr  # 赋值
    if img_path.split('\\')[-2] == 'SSAP':
        Y[i] = 0
    else:
        Y[i] = 1
i += 1
Y = to_categorical(Y)

class PixelConvLayer(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        self.conv.build(input_shape)
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)


class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])


inputs = keras.Input(shape=input_shape)

x = PixelConvLayer(
    mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same"
)(inputs)

for _ in range(n_residual_blocks):
    x = ResidualBlock(filters=128)(x)

for _ in range(2):
    x = PixelConvLayer(
        mask_type="B",
        filters=128,
        kernel_size=1,
        strides=1,
        activation="relu",
        padding="valid",
    )(x)

x = keras.layers.Conv2D(
    filters=3, kernel_size=1, strides=1, activation="relu", padding="valid"
)(x)

x = keras.layers.Flatten()(x)

out = keras.layers.Dense(2, activation="softmax")(x)

pixel_cnn = keras.Model(inputs, out)

if os.path.exists('PixelCNN_Classify.h5'):
    pixel_cnn.load_weights('PixelCNN_Classify.h5')

adam = keras.optimizers.Adam(lr=0.001)
pixel_cnn.compile(
    optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
callbacks_list = [
    keras.callbacks.History(),
    keras.callbacks.ModelCheckpoint("PixelCNN_Classify.h5", monitor='val_acc', verbose=1,
                                    save_best_only=True, save_weights_only=True, mode='auto', period=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                      verbose=1, mode='auto', min_lr=0.00001),
    keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10,
                                  verbose=0, mode='auto', baseline=None, restore_best_weights=False)

]
pixel_cnn.summary()

history = pixel_cnn.fit(
    x=X, y=Y, batch_size=batch_size, validation_split=0.1, epochs=1000, verbose=1, callbacks=callbacks_list,shuffle=True
)
