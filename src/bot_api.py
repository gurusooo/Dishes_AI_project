import telebot
from telebot import types
from model_embedding import load_model, preprocess_image, predict
from PIL import Image
import io

MODEL_PATH = '../food101_model.h5'
model = load_model(MODEL_PATH)
bot = telebot.TeleBot('BOT-TOKEN')

@bot.message_handler(commands=['start'])
def start(message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Начало работы")
    markup.add(btn1)
    bot.send_message(message.from_user.id, f"👋 Привет {message.from_user.first_name}!", reply_markup=markup)

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        image = Image.open(io.BytesIO(downloaded_file))

        image_array = preprocess_image(image)
        
        predictions = predict(model, image_array)

        response = "Вероятно, это:\n"
        for i, (class_id, class_name, confidence) in enumerate(predictions):
            response += f"{i + 1}. {class_name} ({(confidence * 100):.2f}%)\n"

        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка: {str(e)}")

@bot.message_handler(content_types=['text'])
def get_text_messages(message):

    if message.text == 'Начало работы':
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True) 
        btn1 = types.KeyboardButton('Что ты умеешь?')
        btn2 = types.KeyboardButton('Определи еду по фото.')
        markup.add(btn1, btn2)
        bot.send_message(message.from_user.id, 'Что Вы хотите сделать?', reply_markup=markup)


    elif message.text == 'Что ты умеешь?':
        bot.send_message(message.from_user.id, 'Я небольшой бот, созданный дуетом студентов для определения блюд по фото, я вывожу три самых вероятных категории. Обращайтесь, буду рад помочь!', parse_mode='Markdown')

    elif message.text == 'Определи еду по фото.':
        bot.send_message(message.from_user.id, 'Пожалуйста, отправьте фото блюда, и я постараюсь его распознать.', parse_mode='Markdown')
        bot.register_next_step_handler(message, handle_photo)

bot.polling(none_stop=True, interval=0)