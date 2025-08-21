import os
import logging
import telebot
from product_evaluator import ProductEvaluator

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Инициализация оценщика
evaluator = ProductEvaluator()

# Создание экземпляра бота
bot = telebot.TeleBot("8302560859:AAEhaZx1O_s3iIg8lZCfM6e8noxJeXp01ns")

# Максимальная длина сообщения в Telegram
MAX_MESSAGE_LENGTH = 4000  # Немного меньше 4096 для безопасности


# Обработчик команды /start
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Обработчик команды /start"""
    welcome_message = (
        "Добро пожаловать в PDiddy checker!\n\n"
        "Отправьте фотографию этикетки детского питания, и я проверю, "
        "соответствует ли продукт требованиям Всемирной Организации Здравоохранения.\n\n"
        "Поддерживаются продукты для детей от 6 до 36 месяцев.\n\n"
        "📸 Для анализа отправьте четкое фото этикетки с хорошим освещением.\n"
        "🔍 Система проверит состав, маркировку и возрастные ограничения с использованием нейросети Qwen3."
    )
    bot.reply_to(message, welcome_message)


# Обработчик фотографий
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    """Обработчик фотографий"""
    try:
        # Получение информации о фото (берем самое большое разрешение)
        photo_file = bot.get_file(message.photo[-1].file_id)

        # Создание временного пути для сохранения
        image_path = f"temp_{message.message_id}.jpg"

        # Скачивание фото
        downloaded_file = bot.download_file(photo_file.file_path)
        with open(image_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        # Оценка продукта
        processing_msg = bot.reply_to(message, "🔍 Анализирую изображение. Анализ может длится до 1 минуты...")

        try:
            # Выполняем оценку через Qwen3 API
            analysis = evaluator.evaluate_product(image_path)

            # Проверка на пустой результат
            if not analysis or len(analysis.strip()) == 0:
                logger.error("Получен пустой результат от анализатора")
                analysis = (
                    "❌ Ошибка анализа: получен пустой результат.\n\n"
                    "Пожалуйста, попробуйте:\n"
                    "1. Отправить более четкое фото этикетки\n"
                    "2. Убедиться, что текст на этикетке хорошо виден\n"
                    "3. Проверить освещение на фото"
                )

            # Удаляем сообщение "Анализирую изображение"
            try:
                bot.delete_message(message.chat.id, processing_msg.message_id)
            except Exception as delete_error:
                logger.warning(f"Не удалось удалить сообщение: {str(delete_error)}")

            # Создаем временный TXT-файл
            txt_path = f"report_{message.message_id}.txt"
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(analysis)

            # Отправляем файл
            with open(txt_path, 'rb') as txt_file:
                bot.send_document(
                    message.chat.id,
                    txt_file,
                    caption="📄 Результат анализа детского питания"
                )

            # Удаляем временный TXT-файл
            if os.path.exists(txt_path):
                os.remove(txt_path)

        except Exception as eval_error:
            logger.error(f"Ошибка при оценке продукта: {str(eval_error)}")
            try:
                # Удаляем сообщение "Анализирую изображение", если оно существует
                try:
                    bot.delete_message(message.chat.id, processing_msg.message_id)
                except:
                    pass

                bot.send_message(message.chat.id, (
                    "⚠️ Произошла ошибка при анализе продукта.\n\n"
                    "Возможные причины:\n"
                    "- Сложная структура этикетки\n"
                    "- Слишком мелкий шрифт\n"
                    "- Низкое качество изображения\n\n"
                    "Попробуйте отправить более четкое фото этикетки."
                ))
            except Exception as send_error:
                logger.error(f"Не удалось отправить сообщение об ошибке: {str(send_error)}")

        # Удаление временного файла изображения
        if os.path.exists(image_path):
            os.remove(image_path)

    except Exception as e:
        logger.error(f"Критическая ошибка обработки: {str(e)}")
        try:
            # Пытаемся удалить сообщение "Анализирую изображение", если оно существует
            if 'processing_msg' in locals():
                try:
                    bot.delete_message(message.chat.id, processing_msg.message_id)
                except:
                    pass

            bot.reply_to(message, (
                "❌ Критическая ошибка при обработке изображения.\n\n"
                "Пожалуйста, попробуйте:\n"
                "1. Отправить фото в лучшем качестве\n"
                "2. Убедиться, что этикетка полностью видна\n"
                "3. Проверить освещение на фото\n\n"
                "Если проблема сохраняется, свяжитесь с администратором."
            ))
        except Exception as reply_error:
            logger.error(f"Не удалось отправить сообщение об ошибке: {str(reply_error)}")


# Обработчик всех остальных сообщений
@bot.message_handler(func=lambda message: True)
def handle_other_messages(message):
    """Обработчик текстовых сообщений"""
    response = (
        "Я могу анализировать только фотографии этикеток детского питания.\n\n"
        "📸 Чтобы проверить продукт:\n"
        "1. Сфотографируйте этикетку продукта\n"
        "2. Убедитесь, что текст хорошо виден\n"
        "3. Отправьте фото в этот чат\n\n"
        "Поддерживаются продукты для детей от 6 до 36 месяцев."
    )
    bot.reply_to(message, response)


def main():
    """Запуск бота"""
    logger.info("Запуск бота с использованием telebot...")

    # Проверка наличия файла требований
    if not os.path.exists('Инструкция_по_оценке_продукта_питания.doc'):
        logger.error("Файл с требованиями ВОЗ не найден!")
        logger.info(
            "Пожалуйста, поместите файл 'Инструкция_по_оценке_продукта_питания.doc' в корневую директорию проекта.")

    # Запуск бота
    logger.info("Бот запущен и готов к работе")
    bot.polling(none_stop=True)


if __name__ == "__main__":
    main()
