import easyocr
import cv2
import numpy as np
import re
import json
import requests
import os
import logging
import time
from collections import defaultdict

# Настройка логирования
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )


class ProductEvaluator:
    def __init__(self):
        """
        Инициализация оценщика
        """
        self.qwen_api_key = "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"

        # Инициализация EasyOCR (только русский язык)
        logger.info("Загрузка модели EasyOCR (русский язык)...")
        try:
            self.reader = easyocr.Reader(['ru'], gpu=False)
            logger.info("Модель EasyOCR успешно загружена.")
        except Exception as e:
            logger.error(f"Ошибка при загрузке EasyOCR: {str(e)}")
            raise

    def preprocess_image(self, image_path):
        """
        Подготавливает изображение для OCR с улучшенной обработкой
        """
        logger.info(f"Начало обработки изображения: {image_path}")

        # Загрузка изображения
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"Не удалось загрузить изображение: {image_path}")

        # Конвертация в оттенки серого
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        logger.debug("Изображение преобразовано в оттенки серого")

        # Удаление шума
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        logger.debug("Удален шум с помощью fastNlMeansDenoising")

        # Адаптивная бинаризация
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5
        )
        logger.debug("Применена адаптивная бинаризация")

        # Увеличение размера изображения для улучшения распознавания
        scale_percent = 150
        width = int(thresh.shape[1] * scale_percent / 100)
        height = int(thresh.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(thresh, dim, interpolation=cv2.INTER_CUBIC)
        logger.debug(f"Изображение увеличено до {scale_percent}%")

        return resized

    def extract_text_from_image(self, image_path):
        """
        Извлекает текст из изображения с помощью EasyOCR
        """
        logger.info("Начало распознавания текста с изображения")

        try:
            # Предварительная обработка
            processed_img = self.preprocess_image(image_path)

            # Распознавание текста
            logger.debug("Запуск распознавания текста через EasyOCR...")
            results = self.reader.readtext(
                processed_img,
                detail=0,
                paragraph=True,
                min_size=20,
                text_threshold=0.7,
                low_text=0.4
            )

            # Объединение в строку
            text = " ".join(results)
            logger.debug(f"Распознанный текст: {text}")

            # Удаление повторяющихся пробелов
            text = re.sub(r'\s+', ' ', text).strip()

            return text
        except Exception as e:
            logger.error(f"Ошибка при извлечении текста: {str(e)}")
            raise

    def get_who_requirements(self, requirements_path='who_requirements.json'):
        """
        Загружает требования ВОЗ из JSON-файла и формирует текстовое описание
        """
        logger.info(f"Извлечение требований ВОЗ из JSON: {requirements_path}")

        # Проверка существования файла
        if not os.path.exists(requirements_path):
            raise FileNotFoundError(f"Файл {requirements_path} не найден")

        try:
            with open(requirements_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Ошибка при чтении JSON: {str(e)}")

        # Формирование текстового описания требований
        text = "ТРЕБОВАНИЯ ВОЗ ПО ОЦЕНКЕ ПРОДУКТОВ ПИТАНИЯ ДЛЯ ДЕТЕЙ\n"
        text += "=" * 60 + "\n\n"

        # 1. Категории продуктов
        text += "КАТЕГОРИИ ПРОДУКТОВ:\n"
        for cat in data['categories']:
            text += f"\n• Код: {cat['code']}\n"
            text += f"  Категория: {cat['category']}\n"
            text += f"  Подкатегория: {cat['subcategory']}\n"
            text += f"  Описание: {cat['description']}\n"
            text += f"  Примеры: {', '.join(cat['examples'])}\n"

        # 2. Требования к составу
        text += "\nТРЕБОВАНИЯ К СОСТАВУ ПО КАТЕГОРИЯМ:\n"
        for code, reqs in data['content_requirements'].items():
            text += f"\n• Категория {code}:\n"
            for nutrient, limits in reqs.items():
                if nutrient == "added_sugars" and limits is True:
                    text += "  - Добавленные сахара: запрещены\n"
                elif isinstance(limits, dict):
                    parts = []
                    if 'min' in limits:
                        parts.append(f"мин {limits['min']}")
                    if 'max' in limits:
                        parts.append(f"макс {limits['max']}")
                    if parts:
                        text += f"  - {self._format_nutrient_name(nutrient)}: {' и '.join(parts)}\n"

        # 3. Требования к маркировке
        text += "\nОБЩИЕ ТРЕБОВАНИЯ К МАРКИРОВКЕ:\n"
        labeling_reqs = data['labeling_requirements']
        if labeling_reqs['no_health_claims']:
            text += "  - Запрещены заявления о пользе для здоровья\n"
        if labeling_reqs['clear_product_name']:
            text += "  - Четкое название продукта\n"
        if labeling_reqs['clear_ingredients_list']:
            text += "  - Четкий список ингредиентов\n"
        if labeling_reqs['no_sipping_instructions']:
            text += "  - Запрещены инструкции по питью через соску\n"
        if labeling_reqs['appropriate_preparation']:
            text += "  - Указания по приготовлению должны быть подходящими\n"
        if labeling_reqs['breastfeeding_support']:
            text += "  - Поддержка грудного вскармливания\n"

        # 4. Запрещенные заявления
        text += "\nЗАПРЕЩЕННЫЕ ЗАЯВЛЕНИЯ:\n"
        for category, statements in data['prohibited_statements'].items():
            category_name = {
                "composition_claims": "Заявления о составе",
                "health_benefit_claims": "Заявления о пользе для здоровья",
                "marketing_claims": "Маркетинговые заявления"
            }.get(category, category)
            text += f"\n• {category_name}:\n"
            for stmt in statements:
                text += f"  - {stmt}\n"

        logger.info("Требования ВОЗ успешно преобразованы в текст")
        return text

    def _format_nutrient_name(self, nutrient):
        """
        Форматирует название питательного вещества в читаемый вид
        """
        mapping = {
            "energy": "Энергетическая ценность (ккал/100г)",
            "sodium": "Натрий (мг/100г)",
            "total_sugar": "Сахара (г/100г)",
            "protein": "Белки (г/100г)",
            "fats": "Жиры (г/100г)",
            "fruit_content": "Содержание фруктов (% от веса)",
            "high_sugar_label": "Порог для предупреждения о высоком содержании сахара"
        }
        return mapping.get(nutrient, nutrient.replace('_', ' ').title())

    def analyze_with_qwen(self, who_requirements, product_text):
        """
        Отправляет запрос в Qwen3 API для анализа соответствия продукта требованиям ВОЗ
        """
        logger.info("Отправка запроса в Qwen3 API для анализа соответствия продукта требованиям ВОЗ")

        # Формируем промпт
        prompt = self._create_analysis_prompt(who_requirements, product_text)

        # Подготовка запроса к API
        headers = {
            "Authorization": f"Bearer {self.qwen_api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "qwen/qwen3-235b-a22b",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 2000,
            "stop": ["===КОНЕЦ ОТЧЕТА==="]
        }

        try:
            # Отправка запроса
            logger.debug("Отправка запроса к Qwen3 API...")
            start_time = time.time()
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            elapsed = time.time() - start_time
            logger.debug(f"Получен ответ от API за {elapsed:.2f} секунд")

            # Проверка статуса ответа
            if response.status_code != 200:
                logger.warning(f"Ошибка API: {response.status_code} - {response.text}")
                return self._create_fallback_report(product_text)

            # Извлечение ответа
            result = response.json()
            analysis = result['choices'][0]['message']['content'].strip()

            # Проверка на полноту отчета
            if not self._is_report_complete(analysis):
                logger.warning("Получен неполный отчет от API, добавляем недостающие элементы...")
                analysis = self._complete_report(analysis)

            logger.debug(f"Получен анализ от Qwen3: {analysis}")
            return analysis

        except Exception as e:
            logger.error(f"Ошибка при обращении к Qwen3 API: {str(e)}")
            return self._create_fallback_report(product_text)

    def _create_analysis_prompt(self, who_requirements, product_text):
        """
        Создает промпт для анализа соответствия продукта требованиям ВОЗ
        """
        prompt = (
            "Ты являешься экспертом по оценке продуктов детского питания в соответствии с требованиями ВОЗ. "
            "Тебе предоставлены:\n\n"

            "1. Требования ВОЗ по оценке продуктов питания для детей грудного и раннего возраста от 6 до 36 месяцев\n"
            "2. Текст этикетки продукта детского питания\n\n"

            "Твоя задача:\n"
            "1. Определить категорию продукта на основе текста этикетки\n"
            "2. Проверить соответствие продукта требованиям ВОЗ для этой категории\n"
            "3. Выявить все нарушения (если они есть)\n"
            "4. Сформировать отчет в строгом формате:\n\n"

            "РЕЗУЛЬТАТ ОЦЕНКИ СООТВЕТСТВИЯ ПРОДУКТА ТРЕБОВАНИЯМ ВОЗ\n"
            "============================================================\n\n"

            "ОПРЕДЕЛЕННАЯ КАТЕГОРИЯ: [код] - [название]\n"
            "НАЗВАНИЕ ПРОДУКТА: [название]\n"
            "ВОЗРАСТНАЯ МАРКИРОВКА: [возраст]\n\n"

            "РЕЗУЛЬТАТ: [соответствует/не соответствует]\n\n"

            "НАЙДЕННЫЕ НАРУШЕНИЯ:\n"
            "1. [нарушение 1]\n"
            "2. [нарушение 2]\n"
            "...\n\n"

            "РЕКОМЕНДАЦИИ:\n"
            "[рекомендации]\n\n"

            "ИНФОРМАЦИЯ О ПРОДУКТЕ:\n"
            "[краткое описание]\n\n"

            "===КОНЕЦ ОТЧЕТА===\n\n"

            "ВАЖНО:\n"
            "- Если продукт содержит добавленные сахара в категории, где они запрещены — это критическое нарушение\n"
            "- Если возрастная маркировка не соответствует требованиям категории — это критическое нарушение\n"
            "- Если на этикетке есть запрещенные заявления — это критическое нарушение\n\n"
            "- НЕ ИСПОЛЬЗУЙ ФОРМАТИРОВАНИЯ В СВОЕМ ОТВЕТЕ!\n\n"

            "Требования ВОЗ:\n\n"
            f"{who_requirements}\n\n"

            "Текст этикетки продукта:\n\n"
            f"{product_text}\n\n"

            "Проведи анализ и сформируй отчет в указанном формате. "
            "Если категория не определена, укажи 'Неизвестная категория'. "
            "Если нарушений нет, укажи 'Нарушения не обнаружены'."
        )

        return prompt

    def _is_report_complete(self, report):
        """
        Проверяет, является ли отчет полным
        """
        required_sections = [
            "РЕЗУЛЬТАТ ОЦЕНКИ СООТВЕТСТВИЯ ПРОДУКТА ТРЕБОВАНИЯМ ВОЗ",
            "ОПРЕДЕЛЕННАЯ КАТЕГОРИЯ:",
            "НАЗВАНИЕ ПРОДУКТА:",
            "ВОЗРАСТНАЯ МАРКИРОВКА:",
            "РЕЗУЛЬТАТ:",
            "НАЙДЕННЫЕ НАРУШЕНИЯ:",
            "РЕКОМЕНДАЦИИ:",
            "ИНФОРМАЦИЯ О ПРОДУКТЕ:"
        ]

        for section in required_sections:
            if section not in report:
                return False

        if not report.endswith("===КОНЕЦ ОТЧЕТА==="):
            return False

        return True

    def _complete_report(self, incomplete_report):
        """
        Дополняет неполный отчет недостающими элементами
        """
        if "НАЗВАНИЕ ПРОДУКТА:" in incomplete_report and "ВОЗРАСТНАЯ МАРКИРОВКА:" not in incomplete_report:
            incomplete_report += "\nВОЗРАСТНАЯ МАРКИРОВКА: Не указана\n\nРЕЗУЛЬТАТ: Критическое нарушение — отсутствует возрастная маркировка\n\nНАЙДЕННЫЕ НАРУШЕНИЯ:\n1. Отсутствует возрастная маркировка (требуется для всех категорий)"

        elif "РЕЗУЛЬТАТ:" in incomplete_report and "НАЙДЕННЫЕ НАРУШЕНИЯ:" not in incomplete_report:
            incomplete_report += "\n\nНАЙДЕННЫЕ НАРУШЕНИЯ:\n1. Не удалось определить категорию продукта"

        elif "НАЙДЕННЫЕ НАРУШЕНИЯ:" in incomplete_report and "РЕКОМЕНДАЦИИ:" not in incomplete_report:
            incomplete_report += "\n\nРЕКОМЕНДАЦИИ:\n❌ Не удалось завершить анализ. Проверьте качество фото этикетки."

        if not incomplete_report.endswith("===КОНЕЦ ОТЧЕТА==="):
            incomplete_report += "\n\n===КОНЕЦ ОТЧЕТА==="

        return incomplete_report

    def _create_fallback_report(self, product_text):
        """
        Создает резервный отчет в случае недоставности API
        """
        # Определение категории на основе ключевых слов
        category = "Неизвестная категория"
        if re.search(r'каша|крупа|мюсли|рис|макароны', product_text, re.IGNORECASE):
            category = "1 - Сухие каши и крахмалистые продукты"
        elif re.search(r'молоко|йогурт|сыр|творог|пудинг', product_text, re.IGNORECASE):
            category = "2 - Молочные продукты"
        elif re.search(r'пюре|фрукт|яблоко|груша|банан', product_text, re.IGNORECASE):
            category = "3.1 - Фруктовые и овощные пюре/коктейли"

        # Определение возраста
        age_match = re.search(r'(?:возраст|для|детей)\s*(?:от\s*)?(\d+)\s*(?:месяц|мес)', product_text, re.IGNORECASE)
        age_label = f"С {age_match.group(1)} месяцев" if age_match else "Не указана"

        # Формирование отчета
        report = "РЕЗУЛЬТАТ ОЦЕНКИ СООТВЕТСТВИЯ ПРОДУКТА ТРЕБОВАНИЯМ ВОЗ\n"
        report += "=" * 60 + "\n\n"
        report += f"ОПРЕДЕЛЕННАЯ КАТЕГОРИЯ: {category}\n"
        report += f"НАЗВАНИЕ ПРОДУКТА: {self._extract_product_name(product_text)}\n"
        report += f"ВОЗРАСТНАЯ МАРКИРОВКА: {age_label}\n\n"
        report += "РЕЗУЛЬТАТ: Упрощенный анализ (API недоступен)\n\n"
        report += "НАЙДЕННЫЕ НАРУШЕНИЯ:\n"
        report += "1. Невозможно проверить соответствие требованиям к составу\n"
        report += "2. Невозможно проверить запрещенные заявления\n\n"
        report += "РЕКОМЕНДАЦИИ:\n"
        report += "⚠️ Для полного анализа повторите проверку позже.\n"
        report += "Система временно использует упрощенный анализ на основе ключевых слов.\n\n"
        report += "ИНФОРМАЦИЯ О ПРОДУКТЕ:\n"
        report += f"Извлеченный текст: {product_text[:200]}{'...' if len(product_text) > 200 else ''}\n\n"
        report += "===КОНЕЦ ОТЧЕТА==="

        return report

    def _extract_product_name(self, text):
        """
        Извлекает название продукта из текста
        """
        lines = text.split('\n')
        for line in lines[:3]:
            if re.search(r'[а-яА-Я]', line) and len(line) > 5:
                return line.strip().title()
        return "Не определено"

    def evaluate_product(self, image_path):
        """
        Полная оценка продукта с использованием Qwen3 API
        """
        logger.info("Начало оценки продукта...")

        # Извлечение текста из изображения
        product_text = self.extract_text_from_image(image_path)
        logger.info(f"Извлеченный текст с этикетки: {product_text[:200]}{'...' if len(product_text) > 200 else ''}")

        # Извлечение требований ВОЗ из JSON
        try:
            who_requirements = self.get_who_requirements()
            logger.info(f"Требования ВОЗ загружены из JSON")
        except Exception as e:
            logger.error(f"Ошибка при извлечении требований ВОЗ: {str(e)}")
            who_requirements = "Не удалось загрузить требования ВОЗ из JSON"

        # Анализ с помощью Qwen3 API
        analysis = self.analyze_with_qwen(who_requirements, product_text)

        # Убедимся, что отчет полный
        if not self._is_report_complete(analysis):
            analysis = self._complete_report(analysis)

        logger.info("Оценка продукта завершена")
        return analysis

    def generate_evaluation_report(self, analysis):
        """
        Возвращает отчет об оценке (уже сформированный Qwen3)
        """
        return analysis
