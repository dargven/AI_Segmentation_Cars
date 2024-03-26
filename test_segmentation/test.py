import os
import time
import glob
from imageio import imread

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from livelossplot.inputs.keras import PlotLossesCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from skimage import measure
from skimage.io import imread, imsave, imshow
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.morphology import dilation, disk
from skimage.draw import polygon, polygon_perimeter

# from livelossplot.tf_keras import PlotLossesCallback
import requests

images = sorted(glob.glob('../src/images/*.png'))
masks = sorted(glob.glob('../src/masks/*.png'))

# Зададим размер batch и размер изображения
BATCH_SIZE = 16
IMG_SHAPE = (256, 256)
OUTPUT_SIZE = (960, 1280)

# 7 классов + 1 фон
CLASSES = 5

# Цвета для отображения сегментации
Colors = ['red', 'blue', 'green', 'white', 'black']
rgb_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 255),
              (0, 0, 0)]


# Функция для загрузки изображения
def load_img(image, mask):
    # Загружаем и препроцессим изображения
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image)
    image = tf.image.resize(image, OUTPUT_SIZE)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255.0

    # Загружаем и препроцессим маски изображений
    mask = tf.io.read_file(mask)
    mask = tf.io.decode_png(mask)
    mask = tf.image.grayscale_to_rgb(mask)  # Если маски сохранены в GRAY
    mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.image.resize(mask, OUTPUT_SIZE)
    mask = tf.image.convert_image_dtype(mask, tf.float32)

    # Сохраним маски для каждого класса
    masks = []
    for i in range(CLASSES):
        masks.append(tf.where(tf.equal(mask, float(i)), 1.0, 0.0))
    masks = tf.stack(masks, axis=2)
    masks = tf.reshape(masks, OUTPUT_SIZE + (CLASSES,))
    return image, masks


# Аугментация изобажений
def aug_img(image, masks):
    random_crop = tf.random.uniform((), 0.3, 1)
    image = tf.image.central_crop(image, random_crop)
    masks = tf.image.central_crop(masks, random_crop)
    random_flip = tf.random.uniform((), 0, 1)
    if random_flip >= 0.5:
        image = tf.image.flip_left_right(image)
        masks = tf.image.flip_left_right(masks)
    image = tf.image.resize(image, IMG_SHAPE)
    masks = tf.image.resize(masks, IMG_SHAPE)
    return image, masks


# Функция для подсчёта DICE коэффициента
def dice_coef(y_pred, y_true):
    y_pred = tf.unstack(y_pred, axis=3)
    y_true = tf.unstack(y_true, axis=3)
    dice_summ = 0

    for i, (a_y_pred, b_y_true) in enumerate(zip(y_pred, y_true)):
        dice_calculate = (2 * tf.math.reduce_sum(a_y_pred * b_y_true) + 1) / \
                         (tf.math.reduce_sum(a_y_pred + b_y_true) + 1)

        dice_summ += dice_calculate
    avg_dice = dice_summ / CLASSES
    return avg_dice


# Функция для подсчета DICE loss
def dice_loss(y_pred, y_true):
    d_loss = 1 - dice_coef(y_pred, y_true)
    return d_loss


# binary_crossentropy - дает хорошую сходимость модели при сбалансированном наборе данных
# DICE - хорошо в задачах сегментации но плохая сходимость
# Плохо сбалансированные данные, но хорошие реузьтаты:
# Binary crossentropy + 0.25 * DICE

def dice_bce_loss(y_pred, y_true):
    total_loss = 0.25 * dice_loss(y_pred, y_true) + tf.keras.losses.binary_crossentropy(y_pred, y_true)
    return total_loss


# #Можно воспользоваться готовой DICE (но не очень удобно)
# preds = torch.tensor([[2, 0], [2, 1]])
# target = torch.tensor([[1, 1], [2, 0]])
# dice(y_pred, y_true, average='micro')


# Функция для подсчёта расстояния между точками
def distance_between_p(p1, p2):
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis


dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(images),
                               tf.data.Dataset.from_tensor_slices(masks)))

# Загружаем дату
dataset = dataset.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)

# Аугментация (увеличим датасет в 50 раз)
dataset = dataset.repeat(50)
dataset = dataset.map(aug_img, num_parallel_calls=tf.data.AUTOTUNE)

# Размер train выборки
train_size = 1800

# Делим на train и test
train_dataset = dataset.take(train_size).cache()
test_dataset = dataset.skip(train_size).take(len(dataset) - train_size).cache()

train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


def unet_model(image_size, output_classes):
    # Входной слой
    input_layer = tf.keras.layers.Input(shape=image_size + (3,))
    conv_1 = tf.keras.layers.Conv2D(64, 4, activation=tf.keras.layers.LeakyReLU(),
                                    strides=2, padding='same', kernel_initializer='glorot_normal',
                                    use_bias=False)(input_layer)
    # Сворачиваем
    conv_1_1 = tf.keras.layers.Conv2D(128, 4, activation=tf.keras.layers.LeakyReLU(), strides=2,
                                      padding='same', kernel_initializer='glorot_normal',
                                      use_bias=False)(conv_1)
    batch_norm_1 = tf.keras.layers.BatchNormalization()(conv_1_1)

    # 2
    conv_2 = tf.keras.layers.Conv2D(256, 4, activation=tf.keras.layers.LeakyReLU(), strides=2,
                                    padding='same', kernel_initializer='glorot_normal',
                                    use_bias=False)(batch_norm_1)
    batch_norm_2 = tf.keras.layers.BatchNormalization()(conv_2)

    # 3
    conv_3 = tf.keras.layers.Conv2D(512, 4, activation=tf.keras.layers.LeakyReLU(), strides=2,
                                    padding='same', kernel_initializer='glorot_normal',
                                    use_bias=False)(batch_norm_2)
    batch_norm_3 = tf.keras.layers.BatchNormalization()(conv_3)

    # 4
    conv_4 = tf.keras.layers.Conv2D(512, 4, activation=tf.keras.layers.LeakyReLU(), strides=2,
                                    padding='same', kernel_initializer='glorot_normal',
                                    use_bias=False)(batch_norm_3)
    batch_norm_4 = tf.keras.layers.BatchNormalization()(conv_4)

    # 5
    conv_5 = tf.keras.layers.Conv2D(512, 4, activation=tf.keras.layers.LeakyReLU(), strides=2,
                                    padding='same', kernel_initializer='glorot_normal',
                                    use_bias=False)(batch_norm_4)
    batch_norm_5 = tf.keras.layers.BatchNormalization()(conv_5)

    # 6
    conv_6 = tf.keras.layers.Conv2D(512, 4, activation=tf.keras.layers.LeakyReLU(), strides=2,
                                    padding='same', kernel_initializer='glorot_normal',
                                    use_bias=False)(batch_norm_5)

    # Разворачиваем
    # 1
    up_1 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2DTranspose(512, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer='glorot_normal',
                                                                          use_bias=False)(conv_6), conv_5])
    batch_up_1 = tf.keras.layers.BatchNormalization()(up_1)

    # Добавим Dropout от переобучения
    batch_up_1 = tf.keras.layers.Dropout(0.25)(batch_up_1)

    # 2
    up_2 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2DTranspose(512, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer='glorot_normal',
                                                                          use_bias=False)(batch_up_1), conv_4])
    batch_up_2 = tf.keras.layers.BatchNormalization()(up_2)
    batch_up_2 = tf.keras.layers.Dropout(0.25)(batch_up_2)

    # 3
    up_3 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2DTranspose(512, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer='glorot_normal',
                                                                          use_bias=False)(batch_up_2), conv_3])
    batch_up_3 = tf.keras.layers.BatchNormalization()(up_3)
    batch_up_3 = tf.keras.layers.Dropout(0.25)(batch_up_3)

    # 4
    up_4 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2DTranspose(256, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer='glorot_normal',
                                                                          use_bias=False)(batch_up_3), conv_2])
    batch_up_4 = tf.keras.layers.BatchNormalization()(up_4)

    # 5
    up_5 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2DTranspose(128, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer='glorot_normal',
                                                                          use_bias=False)(batch_up_4), conv_1_1])
    batch_up_5 = tf.keras.layers.BatchNormalization()(up_5)

    # 6
    up_6 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2DTranspose(64, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer='glorot_normal',
                                                                          use_bias=False)(batch_up_5), conv_1])
    batch_up_6 = tf.keras.layers.BatchNormalization()(up_6)

    # Выходной слой
    output_layer = tf.keras.layers.Conv2DTranspose(output_classes, 4, activation='sigmoid', strides=2,
                                                   padding='same',
                                                   kernel_initializer='glorot_normal')(batch_up_6)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


COT_model = unet_model(IMG_SHAPE, CLASSES)

tf.keras.utils.plot_model(COT_model, show_shapes=True)

# Компилируем модель
COT_model.compile(optimizer='adam',
                  loss=[dice_bce_loss],
                  metrics=[dice_coef])

COT_model.summary()

EPOCHS = 10

# Обучение
COT_model.fit(train_dataset,
              validation_data=test_dataset,
              epochs=EPOCHS, initial_epoch=0,
              callbacks=[PlotLossesCallback()])

# Сохранить веса модели
# COT_model.save('/content/gdrive/MyDrive/cat_data/networks/checkpoint_best_unet_like_model_10epochs')
COT_model.save('./cat_data/networks/checkpoint_best_unet_like_model_10epochs')
