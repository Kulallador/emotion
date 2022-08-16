from sched import scheduler
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
import random as python_random
import argparse
import datetime

np.random.seed(42)
python_random.seed(42)
tf.random.set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Path to input data")
parser.add_argument("--tflite", action="store_true", help="Save model to tflite")
parser.add_argument("--save_path", help="Path to save model")
parser.add_argument("--log_dir", help="Path to save logs")

def save_in_tflite(model, name):
    converter = tf.lite.TFLiteConverter.from_keras_model(model=model)
    tflite_model = converter.convert()

    with open(name, 'wb') as f:
        f.write(tflite_model)

def residual_block(x_in, channels, skip_conv=False, strides=(1,1)):
    x = tf.keras.layers.Conv2D(channels, 3, strides=strides, padding="same")(x_in)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Conv2D(channels, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if skip_conv:
        x_in = tf.keras.layers.Conv2D(channels, 1, strides)(x_in)

    x = tf.keras.layers.Add()([x_in, x])
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    return x

def resnetlike_EmotionCNN(n_classes, input_shape=(48, 48, 1)): 
    x_input = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(16, 3, padding="same")(x_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = residual_block(x, 16)

    x = residual_block(x, 32, skip_conv=True, strides=(2,2))

    x = residual_block(x, 64, skip_conv=True, strides=(2,2))

    x = residual_block(x, 128, skip_conv=True, strides=(2,2))

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(n_classes)(x)
    x = tf.keras.layers.Softmax()(x)

    return tf.keras.models.Model(inputs=x_input, outputs=x)

if __name__ == "__main__":
    args = parser.parse_args()

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        f"{args.input}/train",
        target_size=(48, 48),
        batch_size=128,
        class_mode='categorical',
        color_mode="grayscale"
    ) 

    test_generator = test_datagen.flow_from_directory(
        f"{args.input}/test",
        target_size=(48, 48),
        batch_size=128,
        class_mode='categorical',
        color_mode="grayscale"
    )

    model = resnetlike_EmotionCNN(6)
    opt = tf.keras.optimizers.Adam(0.001)
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(opt, loss, metrics=["accuracy"])

    scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr if (epoch+1) % 80 != 0 else lr*0.1)
    callbacks = [scheduler]

    if not args.log_dir is None:
        log_dir = args.log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)

    model.fit(x=train_generator,
            epochs=1,
            callbacks=callbacks)
    
    model.save(f"{args.save_path}.h5")
    
    save_in_tflite(model, f"{args.save_path}.tflite")