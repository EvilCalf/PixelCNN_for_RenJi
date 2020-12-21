
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras import layers
import pandas as pd
from keras.utils import to_categorical
from tensorflow.python.keras.backend import argmax

num_classes = 10
input_shape = (28, 28, 1)
n_residual_blocks = 5


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
    filters=1, kernel_size=1, strides=1, activation="relu", padding="valid"
)(x)

x = keras.layers.Flatten()(x)

out = keras.layers.Dense(10, activation="softmax")(x)

pixel_cnn = keras.Model(inputs, out)

pixel_cnn.load_weights('PixelCNN_Classify.h5')
adam = keras.optimizers.Adam(lr=0.001)
pixel_cnn.compile(
    optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
data = pd.read_csv('test.csv', header=None).values
data = data.reshape(data.shape[0], 28, 28, 1)
data = np.where(data < (0.33 * 256), 0, 1)
data = data.astype(np.float32)
output = pixel_cnn.predict(data, batch_size=128, verbose=1)
output = pd.DataFrame(output)
output.to_csv('output.csv', header=None, index=None)
# output = argmax(output, axis=1)
# ans = pd.DataFrame(output)
# ans.to_csv('ans.csv')
