from PIL import Image
import os


def resize_image(image_name):
    """
    Функция для изменения размера изображения.

    Args:
      image_name: имя файла изображения (например, "image.png").

    Returns:
      None.
    """

    # Путь к исходному изображению
    input_path = f"src/images/{image_name}"

    # Путь к папке для сохранения
    output_path = "resimages"

    # Открываем изображение
    image = Image.open(input_path)

    # Определяем желаемый размер
    desired_size = (960, 1280)

    # Изменяем размер изображения
    image = image.resize(desired_size)

    # Сохраняем изображение
    image.save(f"{output_path}/{image_name}")


def resize_all_images():
    """
    Функция для изменения размера всех изображений в папке.

    Args:
      None.

    Returns:
      None.
    """

    # Путь к папке с исходными изображениями
    input_path = "src/images"

    # Путь к папке для сохранения
    output_path = "resimages"

    # Создаем папку для сохранения, если она не существует
    os.makedirs(output_path, exist_ok=True)

    # Получаем список файлов в папке
    filenames = os.listdir(input_path)

    # Перебираем все файлы
    for filename in filenames:

        # Проверяем, является ли файл изображением
        if filename.lower().endswith(".png") or filename.lower().endswith(".jpg"):
            # Изменяем размер изображения
            resize_image(filename)


# Пример использования
resize_all_images()
