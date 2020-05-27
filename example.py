import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
n_classes = 10
input_shape = x_train.shape[1:]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
m = np.mean(x_train)
s = np.std(x_train)
x_train = (x_train-m)/(s + 1e-7)
x_test = (x_test-m)/(s + 1e-7)

# Training options
lr_schedule = lambda epoch: 10**(-3 - epoch//60)
n_epochs = 200
batch_size = 128
opt = keras.optimizers.Adam(lr=lr_schedule(0))
lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
callbacks = [lr_scheduler]
datagen = keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen.fit(x_train)
train_gen = datagen.flow(x_train, y_train, batch_size=batch_size)
test_gen = datagen.flow(x_test, y_test, batch_size=batch_size)

# Build model
reg = keras.regularizers.l2(1e-4)
init = keras.initializers.he_uniform()

def resnet_block(filters, conv, downsample, x, shortcuts='projection'):
    if downsample:
        y = conv(filters, 3, padding='same', strides=2, kernel_initializer=init, kernel_regularizer=reg)(x)
    else:
        y = conv(filters, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = conv(filters, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg)(y)
    y = layers.BatchNormalization()(y)
    if downsample:
        if shortcuts == 'projection':
            x = conv(filters, 1, padding='same', strides=2, kernel_initializer=init, kernel_regularizer=reg)(x)
        elif shortcuts == 'identity':
            x = layers.MaxPooling2D(1, strides=2)(x)
            x = layers.Lambda(lambda x: tf.pad(x, [[0,0], [0,0], [0,0], [0,filters//2]]))(x)
    x = layers.add([x, y])
    x = layers.Activation('relu')(x)
    return x

inputs = keras.Input(shape=input_shape)
x = layers.Conv2D(16, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg)(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = resnet_block(16, layers.Conv2D, False, x)
x = resnet_block(16, layers.Conv2D, False, x)
x = resnet_block(16, layers.Conv2D, False, x)
x = resnet_block(32, layers.Conv2D, True, x)
x = resnet_block(32, layers.Conv2D, False, x)
x = resnet_block(32, layers.Conv2D, False, x)
x = resnet_block(64, layers.Conv2D, True, x)
x = resnet_block(64, layers.Conv2D, False, x)
x = resnet_block(64, layers.Conv2D, False, x)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(n_classes)(x)
model = keras.Model(inputs, outputs)
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt, metrics=['accuracy'])
model.summary()

# Train model
model.fit(train_gen, steps_per_epoch=x_train.shape[0]//batch_size, epochs=n_epochs, callbacks=callbacks)
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
