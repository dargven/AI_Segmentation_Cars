import cv2
import numpy as np
import os
import concurrent.futures

folder_path = '../src/dataset/masks'
output_folder_path = '../src/dataset/class_masks2'

count = 0
count_img = os.listdir(folder_path)

# Классы
classes = {
    (0, 0, 0): 0,
    (255, 255, 255): 1,
    (0, 0, 255): 2,
    (255, 0, 0): 3,
    (0, 255, 0): 4,
}


# Преобразование изображения
def convert_image(image_path):
    image = cv2.imread(image_path)
    output_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_color = tuple(image[i, j])
            if pixel_color in classes:
                output_image[i, j] = classes[pixel_color]
    return output_image


# Функция для обработки отдельного изображения
def process_image(image_filename):
    image_path = os.path.join(folder_path, image_filename)
    output_path = os.path.join(output_folder_path, image_filename)
    converted_image = convert_image(image_path)
    cv2.imwrite(output_path, converted_image)
    return image_filename


if __name__ == '__main__':
    image_files = [filename for filename in os.listdir(folder_path) if
                   filename.endswith('.jpg') or filename.endswith('.png')]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_image, image_files))

    print('Все изображения обработаны.')
