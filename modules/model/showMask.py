import cv2
import matplotlib.pyplot as plt
import numpy as np

# Путь к изображению и маске
image_path = "../../src/dataset/images/car_side1.jpg"
mask_path = "../../src/dataset/class_masks2/car_side1.png"

# Загрузка изображения и маски
image = cv2.imread(image_path)
mask = cv2.imread(mask_path)

# Отображение изображения и маски
cv2.imshow("Image", image)
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Проверка типа данных
print("Тип данных маски:", mask.dtype)

# Проверка размерности (если маска черно-белая, то shape будет иметь только два значения)
print("Размерность маски:", mask.shape)

# Проверка наличия каналов
if len(mask.shape) == 3:
    print("Маска цветная")
else:
    print("Маска черно-белая")

plt.imshow(mask)  # Используем серую цветовую карту для отображения маски в оттенках серого
plt.axis('off')
plt.show()
plt.close()

mask = cv2.imread("../../src/dataset/class_masks2/car_side1.png")

# Проверка, является ли маска полностью черной
is_black_mask = np.all(mask == 0)

if is_black_mask:
    print("Маска полностью черная")
else:
    print("Маска не полностью черная")
