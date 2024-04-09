# import os
# path = '/content/LessonTest/dataset/masks'
# imgs = os.listdir(path)
# for img in imgs:
#   if(img != ".ipynb_checkpoints"):
#       os.remove(f'{path}/{img}')

"""## Подключаем необходимые модули"""
import copy
# !git clone https://github.com/lyftzeigen/SemanticSegmentationLesson.git

import os
import glob
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from skimage import measure
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.morphology import dilation, disk
from skimage.draw import polygon_perimeter
from skimage import exposure

from metrics import metrics

print(f'Tensorflow version {tf.__version__}')
print(f'GPU is {"ON" if tf.config.list_physical_devices("GPU") else "OFF"}')

"""## Подготовим набор данных для обучения"""

CLASSES = 5

COLORS = ['black', 'white', 'red',
          'blue', 'green']

SAMPLE_SIZE = (256, 256)

OUTPUT_SIZE = (1080, 1920)


def load_images(image, mask):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image)
    image = image[:, :, :3]
    image = tf.image.resize(image, OUTPUT_SIZE)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255.0

    mask = tf.io.read_file(mask)
    mask = tf.io.decode_png(mask)
    mask = mask[:, :, :3]
    # mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.image.resize(mask, OUTPUT_SIZE)
    mask = tf.image.convert_image_dtype(mask, tf.float32)

    masks = []

    for i in range(CLASSES):
        masks.append(tf.where(tf.equal(mask, float(i)), 1.0, 0.0))

    masks = tf.stack(masks, axis=2)
    masks = tf.reshape(masks, OUTPUT_SIZE + (CLASSES,))

    return image, masks


def augmentate_images(image, masks):
    random_crop = tf.random.uniform((), 0.3, 1)
    image = tf.image.central_crop(image, random_crop)
    masks = tf.image.central_crop(masks, random_crop)

    random_flip = tf.random.uniform((), 0, 1)
    if random_flip >= 0.5:
        image = tf.image.flip_left_right(image)
        masks = tf.image.flip_left_right(masks)

    image = tf.image.resize(image, SAMPLE_SIZE)
    masks = tf.image.resize(masks, SAMPLE_SIZE)

    return image, masks


images = sorted(glob.glob('../../src/dataset/images/*.jpg'))
class_masks = sorted(glob.glob('../../src/dataset/class_masks2/*.png'))  # Используем для обучения
binary_masks = sorted(glob.glob('../../src/dataset/cuple_of_masks/*.png'))  # Используем для подсчета итоговой метрики
true_masks = []

images_dataset = tf.data.Dataset.from_tensor_slices(images)
masks_dataset = tf.data.Dataset.from_tensor_slices(class_masks)

dataset = tf.data.Dataset.zip((images_dataset, masks_dataset))

dataset = dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.repeat(60)
dataset = dataset.map(augmentate_images, num_parallel_calls=tf.data.AUTOTUNE)

"""## Посмотрим на содержимое набора данных"""

images_and_masks = list(dataset.take(5))

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(16, 6))

for i, (image, masks) in enumerate(images_and_masks):
    ax[0, i].set_title('Image')
    ax[0, i].set_axis_off()
    ax[0, i].imshow(image)

    ax[1, i].set_title('Mask')
    ax[1, i].set_axis_off()
    ax[1, i].imshow(image / 1.5)

    for channel in range(CLASSES):
        contours = measure.find_contours(np.array(masks[:, :, channel]))
        for contour in contours:
            ax[1, i].plot(contour[:, 1], contour[:, 0], linewidth=1, color=COLORS[channel])

plt.show()
plt.close()

"""## Разделим набор данных на обучающий и проверочный"""

train_dataset = dataset.take(7920).cache()
test_dataset = dataset.skip(7920).take(300).cache()

train_dataset = train_dataset.batch(8)
test_dataset = test_dataset.batch(8)

"""## Обозначим основные блоки модели"""


def input_layer():
    return tf.keras.layers.Input(shape=SAMPLE_SIZE + (3,))


def downsample_block(filters, size, batch_norm=True):
    initializer = tf.keras.initializers.GlorotNormal()

    result = tf.keras.Sequential()

    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if batch_norm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample_block(filters, size, dropout=False):
    initializer = tf.keras.initializers.GlorotNormal()

    result = tf.keras.Sequential()

    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                        kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if dropout:
        result.add(tf.keras.layers.Dropout(0.25))

    result.add(tf.keras.layers.ReLU())
    return result


def output_layer(size):
    initializer = tf.keras.initializers.GlorotNormal()
    return tf.keras.layers.Conv2DTranspose(CLASSES, size, strides=2, padding='same',
                                           kernel_initializer=initializer, activation='sigmoid')


"""## Построим U-NET подобную архитектуру"""

inp_layer = input_layer()

downsample_stack = [
    downsample_block(64, 4, batch_norm=False),
    downsample_block(128, 4),
    downsample_block(256, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
]

upsample_stack = [
    upsample_block(512, 4, dropout=True),
    upsample_block(512, 4, dropout=True),
    upsample_block(512, 4, dropout=True),
    upsample_block(256, 4),
    upsample_block(128, 4),
    upsample_block(64, 4)
]

out_layer = output_layer(4)

# Реализуем skip connections
x = inp_layer

downsample_skips = []

for block in downsample_stack:
    x = block(x)
    downsample_skips.append(x)

downsample_skips = reversed(downsample_skips[:-1])

for up_block, down_block in zip(upsample_stack, downsample_skips):
    x = up_block(x)
    x = tf.keras.layers.Concatenate()([x, down_block])

out_layer = out_layer(x)

unet_like = tf.keras.Model(inputs=inp_layer, outputs=out_layer)

tf.keras.utils.plot_model(unet_like, show_shapes=True, dpi=72)

"""## Определим метрики и функции потерь"""


def dice_mc_metric(a, b):
    a = tf.unstack(a, axis=3)
    b = tf.unstack(b, axis=3)

    dice_summ = 0

    for i, (aa, bb) in enumerate(zip(a, b)):
        numenator = 2 * tf.math.reduce_sum(aa * bb) + 1
        denomerator = tf.math.reduce_sum(aa + bb) + 1
        dice_summ += numenator / denomerator

    avg_dice = dice_summ / CLASSES

    return avg_dice


def dice_mc_loss(a, b):
    return 1 - dice_mc_metric(a, b)


def dice_bce_mc_loss(a, b):
    return 0.3 * dice_mc_loss(a, b) + tf.keras.losses.binary_crossentropy(a, b)


"""## Компилируем модель"""

# unet_like.compile(optimizer='adam', loss=[dice_bce_mc_loss], metrics=[dice_mc_metric])

"""## Обучаем нейронную сеть и сохраняем результат"""

# history_dice = unet_like.fit(train_dataset, validation_data=test_dataset, epochs=1, initial_epoch=0)

# unet_like.save_weights("../../src/networks/unet_like/test.weights.h5")

"""## Загружаем обученную модель"""
# модель
unet_like.load_weights('../../src/networks/unet_like/test.weights.h5')

"""## Проверим работу сети на всех кадрах из видео"""

rgb_colors = [
    (0, 0, 0),  # черный
    (255, 255, 255),  # белый
    (255, 0, 0),  # красный
    (0, 0, 255),  # синий
    (0, 255, 0),  # зеленый
]

# Получение списка файлов изображений
# frames = sorted(glob.glob('../../src/dataset/images/*.jpg'))
frames = sorted(glob.glob('../../src/dataset/cuple_of_images/*.jpg'))
predicted_masks = []

mask_files = sorted(glob.glob('../../src/dataset/cuple_of_masks/*.png'))

true_masks = []

for mask_file in mask_files:
    mask_image = Image.open(mask_file)

    mask = np.array(mask_image)

    true_masks.append(mask)

# Проход по каждому изображению и применение модели
for filename in frames:
    try:
        # Загрузка изображения
        frame = imread(filename)
        # Изменение размера изображения до размера выборки
        sample = resize(frame, SAMPLE_SIZE)

        # Получение предсказания модели для изображения
        predict = unet_like.predict(np.expand_dims(sample, axis=0))
        predict = predict.reshape(SAMPLE_SIZE + (CLASSES,))

        # Масштабирование координат
        scale = frame.shape[0] / SAMPLE_SIZE[0], frame.shape[1] / SAMPLE_SIZE[1]

        # Уменьшение яркости изображения
        frame = (frame / 1.5).astype(np.uint8)

        # Создание нового изображения для наложения масок
        overlay = np.zeros_like(frame)

        # Проход по каждому классу
        for channel in range(1, CLASSES):
            # Создание маски
            mask = np.array(predict[:, :, channel])
            mask = resize(mask, frame.shape[:2], anti_aliasing=True)

            # Применение маски к overlay без прозрачности
            overlay[mask > 0.3] = rgb_colors[channel]  # Попробуйте изменить порог бинаризации здесь

        # Сохранение результата
        predicted_masks.append(copy.deepcopy(overlay))
        imsave(f'../../src/dataset/test/{os.path.basename(filename)}', overlay)
    except Exception as e:
        continue
max_height_predicted = max(overlay.shape[0] for overlay in predicted_masks)
max_width_predicted = max(overlay.shape[1] for overlay in predicted_masks)
###////////////###
max_height_true = max(mask.shape[0] for mask in true_masks)
max_width_true = max(mask.shape[1] for mask in true_masks)

resized_masks = []
for overlay in predicted_masks:
    # Resize or pad each overlay image to match the maximum dimensions
    resized_overlay = np.zeros((max_height_predicted, max_width_predicted, 3), dtype=np.uint8)
    resized_overlay[:overlay.shape[0], :overlay.shape[1]] = overlay
    resized_masks.append(resized_overlay)
# Convert the list of resized overlay images to a single numpy array
predicted_masks = np.stack(resized_masks)
predicted_masks = np.squeeze(predicted_masks, axis=0)
print(predicted_masks.shape)
resized_true_masks = []
for mask in true_masks:
    # Resize or pad each true mask to match the maximum dimensions
    resized_mask = np.zeros((max_height_true, max_width_true), dtype=np.uint8)
    resized_mask[:mask.shape[0], :mask.shape[1]] = mask
    resized_true_masks.append(resized_mask)

# Convert the list of resized true masks to a single numpy array
true_masks = np.stack(resized_true_masks)

# Convert the list of resized true masks to a single numpy array
true_masks = np.stack(resized_true_masks)

# Подсчет метрик
predicted_masks = np.stack(predicted_masks)
print(predicted_masks)
for mask_file in binary_masks:
    mask_image = Image.open(mask_file)
    mask = np.array(mask_image)
    true_masks.append(mask)
print(true_masks)
true_masks = np.stack(true_masks)
print(f"Итоговые метрики: {metrics(true_masks, predicted_masks)}")
