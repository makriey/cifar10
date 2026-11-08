# Класифікація зображень CIFAR-10 за допомогою CNN

# CNN (Convolutional Neural Network) — згорткова нейронна мережа. Вона автоматично виділяє ознаки (features) із зображення та навчається класифікувати їх.

#1. Імпортуємо необхідні бібліотеки

import tensorflow as tf
import tensorflow.keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

#2. Завантаження та підготовка набору CIFAR-10

# CIFAR-10 — набір зображень розміром 32x32 пікселі в 10 класах
# ['літак', 'автомобіль', 'птах', 'кіт', 'олень', 'собака', 'жаба', 'кінь', 'корабель', 'вантажівка']

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

# Список назв класів українською мовою
class_name = ['літак', 'автомобіль', 'птах', 'кіт', 'олень', 'собака', 'жаба', 'кінь', 'корабель', 'вантажівка']

#3. Побудова моделі CNN

model = tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    tensorflow.keras.layers.MaxPooling2D((2, 2)),

    tensorflow.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tensorflow.keras.layers.MaxPooling2D((2, 2)),

    tensorflow.keras.layers.Conv2D(64, (3, 3), activation="relu"),

    tensorflow.keras.layers.Flatten(),
    tensorflow.keras.layers.Dense(64, activation="relu"),

    tensorflow.keras.layers.Dense(10, activation="softmax")
])

#4. Компіляція моделі

model.compile(
    optimizer='adam',  # оптимізатор, що адаптивно навчає ваги
    loss='sparse_categorical_crossentropy',  # функція втрат для багатокласової задачі
    metrics=['accuracy']  # точність — основна метрика
)

#5. Навчання моделі (тренування)

# Модель буде проходити 10 епох (повних переглядів всіх даних)

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

#6. Оцінка точності моделі на тестових даних

loss, acc = model.evaluate(X_test, y_test)
print(f"\nТочність на тестовому наборі: {acc:.2f}")

# Побудова графіка точності

plt.plot(history.history['accuracy'], label='Тренувальна')
plt.plot(history.history['val_accuracy'], label='Валідаційна')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.legend()
plt.title('Графік точності моделі')
plt.grid(True)

# Зберігаємо графік у файл (для README)
os.makedirs('images', exist_ok=True)
plt.savefig('images/training_curves.png', dpi=150, bbox_inches='tight')

plt.show()

#7. Класифікація ВЛАСНОГО зображення користувача

# Функція для підготовки зображення
def load_and_prepare_image(image_path):
    """
    Завантаження та підготовка зображення користувача:
    - читаємо зображення з диску
    - змінюємо розмір до 32x32 (як в CIFAR-10)
    - нормалізуємо значення пікселів
    - розширюємо розмірність для подачі в модель
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # → (1, 32, 32, 3)
    return img_array

# Шлях до вашого зображення (замініть на свій файл, наприклад, 'cat.jpg')
custom_image_path = 'images/my_image.jpg'

#8. Прогнозування класу зображення користувача

# Якщо файл існує, проводимо обробку і класифікацію
if os.path.exists(custom_image_path):
    image_path = custom_image_path
    img_ready = load_and_prepare_image(image_path)
    prediction = model.predict(img_ready)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"\nМодель передбачає: {class_name[predicted_class]} ({confidence*100:.2f}% впевненість)")

    # Показуємо зображення
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Передбачено: {class_name[predicted_class]}")
    plt.show()
else:
    print(f"\nФайл '{custom_image_path}' не знайдено на диску.")
