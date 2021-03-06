import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gibbs_pruning

parser = argparse.ArgumentParser(description="Example script for using Gibbs pruning on ResNet")
parser.add_argument('-s', '--stretch', default=1, type=int, help="""Stretching
    factor to increase the length of the Gibbs annealing and learning rate
    schedule. E.g., stretch=10 increases the training time from 200 epochs to 2000
    epochs.""")
parser.add_argument('--ham', default='unstructured', choices=['unstructured',
    'kernel', 'filter'], help="""Hamiltonian to use for Gibbs pruning. Defaults
    to unstructured.""")
parser.add_argument('-b','--baseline', action='store_true', help="""Use
    ordinary convolutions, without Gibbs pruning""")
parser.add_argument('-p', default=0.9, type=float, help="""Target pruning fraction""")
args = parser.parse_args()
if args.baseline:
    args.stretch = 1

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
def lr_schedule(epoch):
    epoch //= args.stretch
    return 1e-3 if epoch <= 80 \
        else 1e-4 if epoch <= 120 \
        else 1e-5 if epoch <= 160 \
        else 1e-6
n_epochs = 200*args.stretch
batch_size = 128
opt = keras.optimizers.Adam(learning_rate=lr_schedule(0))
lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
callbacks = [lr_scheduler]

# Gibbs pruning options
test_pruning_mode = 'kernel' if args.ham == 'kernel' else \
        'filter' if args.ham == 'filter' else \
        'gibbs'
c = 0.01 if args.ham == 'kernel' else 1.0
if args.baseline:
    conv = layers.Conv2D
else:
    conv = lambda f, ks, **kwargs: gibbs_pruning.GibbsPrunedConv2D(f, ks, p=args.p,
            hamiltonian=args.ham, test_pruning_mode=test_pruning_mode, c=c, **kwargs)
    beta_schedule = np.logspace(0, 4, num=128*args.stretch) if args.ham == 'unstructured' or args.ham == 'kernel' else \
            np.logspace(-2.5, 0, num=128*args.stretch)
    callbacks.append(gibbs_pruning.GibbsPruningAnnealer(beta_schedule, verbose=1))

datagen = keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen.fit(x_train)
train_gen = datagen.flow(x_train, y_train, batch_size=batch_size)

# Build model
reg = keras.regularizers.l2(1e-4)
init = keras.initializers.he_normal()

def resnet_block(filters, conv, downsample, x, shortcuts='projection'):
    if downsample:
        y = conv(filters, 3, padding='same', strides=2, kernel_initializer=init, kernel_regularizer=reg, use_bias=False)(x)
    else:
        y = conv(filters, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg, use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = conv(filters, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg, use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    if downsample:
        if shortcuts == 'projection':
            x = conv(filters, 1, padding='same', strides=2, kernel_initializer=init, kernel_regularizer=reg, use_bias=False)(x)
        elif shortcuts == 'identity':
            x = layers.MaxPooling2D(1, strides=2)(x)
            x = layers.Lambda(lambda x: tf.pad(x, [[0,0], [0,0], [0,0], [0,filters//2]]))(x)
    x = layers.add([x, y])
    x = layers.Activation('relu')(x)
    return x

inputs = keras.Input(shape=input_shape)
x = layers.Conv2D(16, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg, use_bias=False)(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = resnet_block(16, conv, False, x)
x = resnet_block(16, conv, False, x)
x = resnet_block(16, conv, False, x)
x = resnet_block(32, conv, True, x)
x = resnet_block(32, conv, False, x)
x = resnet_block(32, conv, False, x)
x = resnet_block(64, conv, True, x)
x = resnet_block(64, conv, False, x)
x = resnet_block(64, conv, False, x)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(n_classes, activation='softmax', kernel_initializer=init)(x)
model = keras.Model(inputs, outputs)
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])
model.summary()

# Train model
# Use test data for validation just to track performance, this is okay since
# we're not using it for early stopping or anything else that affects training
model.fit(train_gen, steps_per_epoch=x_train.shape[0]//batch_size,
    epochs=n_epochs, validation_data=(x_test,y_test), callbacks=callbacks)
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
