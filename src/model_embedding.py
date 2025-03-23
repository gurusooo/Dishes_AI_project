import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image

# Загрузка модели
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Модель успешно загружена: {model_path}")
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели: {str(e)}")
        return None

# Список категорий
food_classes = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare"
]

# Функция для получения имени класса по индексу
def get_class_name(class_id):
    if 0 <= class_id < len(food_classes):
        return food_classes[class_id]
    return "Неизвестное блюдо"

# Предварительная обработка изображения
def preprocess_image(img):
    img = img.resize((224, 224))  # MobileNetV2 использует входной размер 224x224
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность batch
    img_array = preprocess_input(img_array)  
    return img_array

# Функция предсказания
def predict(model, img_array):
    try:
        predictions = model.predict(img_array)
        top_3_preds = predictions[0].argsort()[-3:][::-1]  # Топ-3 предсказания
        results = [(idx, get_class_name(idx), predictions[0][idx]) for idx in top_3_preds]
        return results
    except Exception as e:
        print(f"Ошибка при предсказании: {str(e)}")
        return []

