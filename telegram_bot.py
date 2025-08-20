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
        "Добро пожаловать в WHO Baby Food Checker!\n\n"
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
        processing_msg = bot.reply_to(message, "🔍 Анализирую изображение. Это может занять 15-30 секунд...")

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

            # Отправка результата по частям
            if len(analysis) > MAX_MESSAGE_LENGTH:
                # Разбиваем на логические части
                parts = split_into_logical_parts(analysis, MAX_MESSAGE_LENGTH)

                # Удаляем сообщение "Анализирую изображение"
                try:
                    bot.delete_message(message.chat.id, processing_msg.message_id)
                except:
                    pass

                # Отправляем каждую часть
                for i, part in enumerate(parts):
                    # Добавляем номер части, если их больше одной
                    if len(parts) > 1:
                        header = f"📌 РЕЗУЛЬТАТ АНАЛИЗА (часть {i + 1}/{len(parts)})\n\n"
                        part = header + part

                    # Отправляем часть
                    bot.send_message(message.chat.id, part)
            else:
                # Редактируем сообщение с результатом
                try:
                    bot.edit_message_text(
                        chat_id=message.chat.id,
                        message_id=processing_msg.message_id,
                        text=analysis
                    )
                except Exception as edit_error:
                    logger.warning(f"Не удалось отредактировать сообщение: {str(edit_error)}")
                    # Если редактирование не удалось, отправляем новое сообщение
                    bot.send_message(message.chat.id, analysis)
                    # И удаляем старое сообщение
                    try:
                        bot.delete_message(message.chat.id, processing_msg.message_id)
                    except:
                        pass

        except Exception as eval_error:
            logger.error(f"Ошибка при оценке продукта: {str(eval_error)}")
            try:
                bot.edit_message_text(
                    chat_id=message.chat.id,
                    message_id=processing_msg.message_id,
                    text=(
                        "⚠️ Произошла ошибка при анализе продукта.\n\n"
                        "Возможные причины:\n"
                        "- Сложная структура этикетки\n"
                        "- Слишком мелкий шрифт\n"
                        "- Низкое качество изображения\n\n"
                        "Попробуйте отправить более четкое фото этикетки."
                    )
                )
            except:
                # Если не удалось отредактировать, отправляем новое сообщение
                bot.reply_to(message, (
                    "⚠️ Произошла ошибка при анализе продукта.\n\n"
                    "Возможные причины:\n"
                    "- Сложная структура этикетки\n"
                    "- Слишком мелкий шрифт\n"
                    "- Низкое качество изображения\n\n"
                    "Попробуйте отправить более четкое фото этикетки."
                ))

        # Удаление временного файла
        if os.path.exists(image_path):
            os.remove(image_path)

    except Exception as e:
        logger.error(f"Критическая ошибка обработки: {str(e)}")
        try:
            # Пытаемся удалить сообщение "Анализирую изображение", если оно существует
            if 'processing_msg' in locals():
                bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass

        error_message = (
            "❌ Критическая ошибка при обработке изображения.\n\n"
            "Пожалуйста, попробуйте:\n"
            "1. Отправить фото в лучшем качестве\n"
            "2. Убедиться, что этикетка полностью видна\n"
            "3. Проверить освещение на фото\n\n"
            "Если проблема сохраняется, свяжитесь с администратором."
        )
        try:
            bot.reply_to(message, error_message)
        except Exception as reply_error:
            logger.error(f"Не удалось отправить сообщение об ошибке: {str(reply_error)}")


def split_into_logical_parts(text, max_length):
    """
    Разбивает текст на логические части, сохраняя структуру отчета
    """
    # Сначала попробуем разбить по заголовкам
    sections = re.split(r'(\n[А-Я][^:]+:\s*\n)', text)

    parts = []
    current_part = ""

    # Обрабатываем разделы
    for i in range(len(sections)):
        # Если это заголовок, объединяем с содержимым
        if i > 0 and re.match(r'\n[А-Я][^:]+:\s*\n', sections[i]):
            section = sections[i] + sections[i + 1] if i + 1 < len(sections) else sections[i]
            i += 1
        else:
            section = sections[i]

        # Если текущая часть + новый раздел не превышает лимит
        if len(current_part) + len(section) <= max_length:
            current_part += section
        else:
            # Если текущая часть не пустая, сохраняем ее
            if current_part.strip():
                parts.append(current_part.strip())
                current_part = ""

            # Если раздел сам по себе меньше лимита
            if len(section) <= max_length:
                current_part = section
            else:
                # Делим длинный раздел на абзацы
                paragraphs = section.split('\n\n')
                temp_section = ""

                for p in paragraphs:
                    if len(temp_section) + len(p) + 2 <= max_length:
                        temp_section += p + "\n\n"
                    else:
                        if temp_section.strip():
                            parts.append(temp_section.strip())
                        temp_section = p + "\n\n"

                if temp_section.strip():
                    current_part = temp_section

    # Добавляем оставшуюся часть
    if current_part.strip():
        parts.append(current_part.strip())

    # Если все же получился слишком длинный текст, делим на равные части
    if not parts:
        parts = [text[i:i + max_length] for i in range(0, len(text), max_length)]

    return parts


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