import os
import cv2
import numpy as np
import concurrent.futures


def mask_to_classes(mask, class_colors):
    """
    Преобразует маску сегментации в классовый вид по заданным цветам классов.

    Аргументы:
    mask : numpy.ndarray
        Массив с маской сегментации, размером (высота, ширина, каналы).
    class_colors : dict
        Словарь, в котором ключи - названия классов, а значения - цвета классов в формате RGB.

    Возвращает:
    numpy.ndarray
        Массив с классовой разметкой, размером (высота, ширина).
    """
    height, width, _ = mask.shape
    num_classes = len(class_colors)
    class_mask = np.zeros((height, width), dtype=np.uint8)

    for idx, (class_name, color) in enumerate(class_colors.items()):
        class_mask[np.all(mask == color, axis=-1)] = idx + 1

    return class_mask


def process_mask(mask_path, output_dir, class_colors):
    """
    Обрабатывает маску сегментации, преобразуя ее в классовый вид и сохраняя результат.

    Аргументы:
    mask_path : str
        Путь к файлу с маской сегментации.
    output_dir : str
        Директория для сохранения результата.
    class_colors : dict
        Словарь, в котором ключи - названия классов, а значения - цвета классов в формате RGB.
    """
    mask = cv2.imread(mask_path)
    class_mask = mask_to_classes(mask, class_colors)

    filename = os.path.basename(mask_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, class_mask)


# Задаем цвета классов
class_colors = {
        'background': [0, 0, 0],
        'body': [255, 255, 255],
        'lights': [0, 255, 0],
        "wheels": [0, 0, 255],
        'windows': [255, 0, 0]
    }

# Директории для входных и выходных данных
input_dir = "../src/dataset/masks"
output_dir = "../src/dataset/class_masks"

# Создаем выходную директорию, если она не существует
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Получаем список файлов масок сегментации
mask_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

# Обрабатываем маски с использованием многопоточности
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for mask_path in mask_files:
        futures.append(executor.submit(process_mask, mask_path, output_dir, class_colors))

    # Дожидаемся завершения всех задач
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"An error occurred: {e}")
