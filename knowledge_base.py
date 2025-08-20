import easyocr
import cv2
import numpy as np
import re
import json
from collections import defaultdict
import os
import string
from difflib import SequenceMatcher


class ProductEvaluator:
    def __init__(self, requirements_path='who_requirements.json'):
        """
        Инициализация оценщика с загрузкой требований ВОЗ
        """
        self.requirements = self._load_requirements(requirements_path)
        self.current_product = None
        self.evaluation_results = None

        # Инициализация EasyOCR (русский + английский)
        print("Загрузка модели EasyOCR (русский + английский)...")
        self.reader = easyocr.Reader(['ru', 'en'], gpu=False)  # gpu=True если есть GPU
        print("Модель EasyOCR загружена.")

        # Добавляем словарь для исправления распространенных ошибок OCR
        self.ocr_correction_dict = {
            "упаЮВан": "упакован",
            "МОДИфИЦИЮОВАННЙ": "модифицированной",
            "аТМО": "атмосфере",
            "фере": "атмосфере",
            "ИЗГОТОВИIЕЛb": "Изготовитель",
            "ПаВеЛецкая": "Павелецкая",
            "ПЛ": "Пл",
            "Д": "Д",
            "УСЛОВИЯ": "Условия",
            "ХРАНЕНИЯ": "хранения",
            "ЗЖКрЫТУЮ": "Закрытую",
            "ПаЧКУ": "Пачку",
            "ХОаНИТb": "хранить",
            "ПаЛУЖНОМ": "Приятном",
            "МеСе": "месте",
            "ПрИТемПературе": "При температуре",
            "б0лее": "более",
            "ОТНОСИТеЛЬНОЙ": "относительной",
            "ВЛаЖНОСТИ": "влажности",
            "ВОЗДУХа": "воздуха",
            "б0Лее": "более",
            "В(крЫТЫЙ": "В закрытый",
            "ПаКеТ": "пакет",
            "ЛеДУеТ": "следует",
            "ПЛОТНО": "плотно",
            "ЗаКОЫТ": "закрыт",
            "ОтКОЫТУЮ": "открытую",
            "ПрОДЖКТОМ": "продуктом",
            "ПрОДЖКТОМ": "продуктом",
            "ЛеДУеТ": "следует",
            "ХраНИТЬ": "хранить",
            "ПрОХЛаДНОМ": "прохладном",
            "MeСТе": "месте",
            "бОЛее": "более",
            "НеДеЛЬ": "недель"
        }

    def _load_requirements(self, path):
        """Загружает требования из JSON файла"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise Exception(f"Файл требований {path} не найден. Сначала запустите knowledge_base.py")
        except json.JSONDecodeError:
            raise Exception(f"Ошибка при чтении файла требований {path}. Файл поврежден или имеет неправильный формат.")

    def preprocess_image(self, image_path):
        """
        Подготавливает изображение для OCR с улучшенной обработкой
        """
        # Загрузка изображения
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"Не удалось загрузить изображение: {image_path}")

        # Конвертация в оттенки серого
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Улучшение контраста
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)

        # Уменьшение шума
        denoised = cv2.fastNlMeansDenoising(contrast_enhanced, h=10)

        # Адаптивная бинаризация с настройкой для улучшения текста
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5
        )

        # Увеличение размера изображения для лучшего распознавания мелкого шрифта
        scale_percent = 150  # Увеличение на 50%
        width = int(thresh.shape[1] * scale_percent / 100)
        height = int(thresh.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(thresh, dim, interpolation=cv2.INTER_CUBIC)

        return resized

    def correct_ocr_errors(self, text):
        """
        Исправляет распространенные ошибки OCR с использованием словаря и fuzzy matching
        """
        # Сначала исправляем по словарю
        words = text.split()
        corrected_words = []

        for word in words:
            # Проверяем, есть ли слово в словаре исправлений
            if word in self.ocr_correction_dict:
                corrected_words.append(self.ocr_correction_dict[word])
            else:
                # Ищем близкие совпадения
                best_match = None
                best_ratio = 0

                for error, correction in self.ocr_correction_dict.items():
                    ratio = SequenceMatcher(None, word, error).ratio()
                    if ratio > 0.7 and ratio > best_ratio:
                        best_ratio = ratio
                        best_match = correction

                if best_match and best_ratio > 0.7:
                    corrected_words.append(best_match)
                else:
                    corrected_words.append(word)

        corrected_text = " ".join(corrected_words)

        # Дополнительные исправления для распространенных проблем
        corrected_text = re.sub(r'([а-яА-Яa-zA-Z])\s+([а-яА-Яa-zA-Z])', r'\1\2',
                                corrected_text)  # Убираем лишние пробелы внутри слов
        corrected_text = re.sub(r'(\d+)\s+(\d+)', r'\1.\2', corrected_text)  # Исправляем десятичные дроби
        corrected_text = corrected_text.replace("  ", " ")  # Убираем двойные пробелы

        return corrected_text

    def extract_text_from_image(self, image_path):
        """
        Извлекает текст из изображения с помощью EasyOCR с улучшенной обработкой
        """
        processed_img = self.preprocess_image(image_path)

        # Распознавание текста
        results = self.reader.readtext(processed_img, detail=0, paragraph=False)

        # Объединение в строку
        text = " ".join(results)

        # Исправление распространенных ошибок OCR
        corrected_text = self.correct_ocr_errors(text)

        # Дополнительная очистка текста
        cleaned_text = self.clean_extracted_text(corrected_text)

        return cleaned_text

    def clean_extracted_text(self, text):
        """
        Очищает извлеченный текст от шума и улучшает читаемость
        """
        # Удаление повторяющихся символов
        text = re.sub(r'(.)\1{2,}', r'\1', text)

        # Исправление распространенных проблем с распознаванием
        text = text.replace("₽", "Р").replace("©", "с").replace("®", "р")
        text = text.replace("а", "a").replace("е", "e").replace("о", "o").replace("р", "p").replace("с", "c")

        # Удаление непечатаемых символов
        text = ''.join(filter(lambda x: x in string.printable, text))

        # Замена нестандартных пробелов
        text = re.sub(r'\s+', ' ', text)

        # Исправление распространенных опечаток в ключевых словах
        text = re.sub(r'возр[ао]ст', 'возраст', text, flags=re.IGNORECASE)
        text = re.sub(r'месяц[аы]', 'месяца', text, flags=re.IGNORECASE)
        text = re.sub(r'дет[яй]', 'дети', text, flags=re.IGNORECASE)
        text = re.sub(r'питани[ея]', 'питание', text, flags=re.IGNORECASE)
        text = re.sub(r'состав', 'состав', text, flags=re.IGNORECASE)
        text = re.sub(r'ингредиенты', 'ингредиенты', text, flags=re.IGNORECASE)

        return text.strip()

    def identify_product_category(self, text):
        """
        Определяет категорию продукта на основе извлеченного текста с улучшенным fuzzy matching
        """
        text_lower = text.lower()
        category_scores = defaultdict(float)

        # Добавляем ключевые слова для каждой категории с весами
        category_keywords = {
            "1": ["каша", "крупа", "мюсли", "рис", "макаронные", "сухие", "крупы", "крупяные", "крупяной"],
            "2": ["йогурт", "молоко", "сыр", "молочный", "молочные", "творог", "сметана", "пудинг", "кефир"],
            "3.1": ["фрукт", "яблоко", "груша", "банан", "пюре", "десерт", "фруктовый", "фруктовые"],
            "3.2": ["овощ", "морковь", "картофель", "свекла", "капуста", "пюре", "овощной", "овощные"],
            "4.1": ["овощ", "крупа", "рис", "макароны", "овощи", "блюдо", "сочетание", "овощные"],
            "4.2": ["сыр", "сырный", "сыром", "сыра"],
            "4.3": ["курица", "говядина", "баранина", "рыба", "мясо", "птица"],
            "4.4": ["курица", "говядина", "баранина", "рыба", "мясо", "птица"],
            "4.5": ["курица", "говядина", "баранина", "рыба", "мясо", "птица"],
            "5.1": ["сухофрукт", "яблоко", "изюм", "фрукт", "фруктовый"],
            "5.2": ["печенье", "крекер", "сухарь", "блинчик", "выпечка", "снек", "перекус"],
            "6": ["масло", "бульон", "ингредиент", "ингредиенты"],
            "7": ["шоколад", "конфета", "лакрица", "марципан", "пастила"],
            "8": ["сок", "напиток", "смузи", "нектар", "напитки", "соки", "напитка"]
        }

        # Веса для разных ключевых слов
        keyword_weights = {
            "каша": 1.5, "крупа": 1.3, "мюсли": 1.2, "рис": 1.0,
            "йогурт": 2.0, "молоко": 1.5, "сыр": 1.3, "молочный": 1.2,
            "фрукт": 1.5, "яблоко": 1.3, "пюре": 1.2, "десерт": 1.0,
            "овощ": 1.5, "морковь": 1.3, "картофель": 1.2,
            "сок": 2.0, "напиток": 1.5
        }

        # Поиск ключевых слов в тексте
        for code, keywords in category_keywords.items():
            for keyword in keywords:
                # Используем fuzzy matching для поиска ключевых слов
                for match in re.finditer(r'\b\w+\b', text_lower):
                    word = match.group(0)
                    similarity = SequenceMatcher(None, word, keyword).ratio()

                    if similarity > 0.7:  # Порог схожести
                        weight = keyword_weights.get(keyword, 1.0)
                        category_scores[code] += similarity * weight

        # Добавляем проверку на наличие явных указаний категории
        age_match = re.search(r'(для|возраст)\s*(\d+)\s*[-–]\s*(\d+)\s*месяц', text_lower)
        if age_match:
            # Если указан возраст 6-36 месяцев, это может быть категория 1, 2 или 3.1
            category_scores["1"] += 0.5
            category_scores["2"] += 0.5
            category_scores["3.1"] += 0.5

        # Проверка на напитки (категория 8)
        if re.search(r'сок|напиток|нектар', text_lower):
            category_scores["8"] += 2.0

        # Проверка на молочные продукты (категория 2)
        if re.search(r'йогурт|молоко|сыр|творог', text_lower):
            category_scores["2"] += 1.5

        # Проверка на каши (категория 1)
        if re.search(r'каша|крупа|мюсли', text_lower):
            category_scores["1"] += 1.5

        # Проверка на фруктовые продукты (категория 3.1)
        if re.search(r'фрукт|яблоко|груша', text_lower):
            category_scores["3.1"] += 1.2

        # Выбор категории с наибольшим количеством совпадений
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            # Проверяем, достаточно ли уверенности в определении
            if category_scores[best_category] > 0.5:
                return best_category

        # Если не удалось определить с достаточной уверенностью, используем резервные правила
        if re.search(r'йогурт|молоко|сыр', text_lower):
            return "2"
        elif re.search(r'сок|напиток', text_lower):
            return "8"
        elif re.search(r'каша|крупа', text_lower):
            return "1"
        elif re.search(r'пюре|фрукт', text_lower):
            return "3.1"

        return None

    def extract_product_info(self, text):
        """
        Извлекает информацию о продукте из текста с улучшенными шаблонами
        """
        product_info = {
            "name": self._extract_product_name(text),
            "ingredients": self._extract_ingredients(text),
            "nutritional_values": self._extract_nutritional_values(text),
            "age_label": self._extract_age_label(text),
            "sugar_content": self._extract_sugar_content(text),
            "prohibited_claims": self._check_prohibited_claims(text),
            "original_text": text  # Сохраняем оригинальный распознанный текст для отладки
        }

        return product_info

    def _extract_product_name(self, text):
        """Извлекает название продукта с улучшенными шаблонами"""
        # Список возможных названий категорий для фильтрации
        category_names = ["каша", "йогурт", "пюре", "сок", "напиток", "молоко", "творог", "сыр"]

        # Паттерны для поиска названия
        patterns = [
            r'название\s*продукта\s*[:\-]?\s*([^\n]+)',
            r'продукт\s*[:\-]?\s*([^\n]+)',
            r'наименование\s*[:\-]?\s*([^\n]+)',
            r'марка\s*[:\-]?\s*([^\n]+)',
            r'торговая\s*марка\s*[:\-]?\s*([^\n]+)',
            r'^([А-ЯЁ][а-яё\s\-]+[а-яё])',  # Заголовок с заглавной буквы
            r'([А-ЯЁ][а-яё\-\s]+)\s*(?:для|питание|детское)',  # Название перед ключевыми словами
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Удаляем возможные лишние слова в конце
                name = re.sub(r'\s*(?:для|питание|детское|детей|малышей|малыша|ребенка|ребенку).*$', '', name,
                              flags=re.IGNORECASE)
                # Удаляем возможные лишние слова в начале
                name = re.sub(r'^(?:продукт|название|марка|торговая марка|детское)\s*', '', name, flags=re.IGNORECASE)
                # Удаляем возможные номера и коды
                name = re.sub(r'\s*[\d\-_]+$', '', name)
                # Удаляем дублирующие пробелы
                name = re.sub(r'\s+', ' ', name)
                return name.strip()

        # Если не найдено по шаблонам, попробуем найти наиболее вероятное название
        # Ищем последовательность слов, начинающуюся с заглавной буквы и содержащую ключевые слова
        for category in category_names:
            match = re.search(r'([А-ЯЁ][а-яё\-\s]+)\s*' + category, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Если все еще не найдено, возвращаем первые 3-5 слов, начинающиеся с заглавной буквы
        words = text.split()
        title_words = []
        for word in words:
            if word and word[0].isupper() and len(word) > 2:
                title_words.append(word)
                if len(title_words) >= 5:
                    break

        if title_words:
            return " ".join(title_words)

        return "Не определено"

    def _extract_ingredients(self, text):
        """Извлекает список ингредиентов с улучшенной обработкой"""
        ingredients = []

        # Поиск раздела "Состав" или "Ingredients" с более гибкими шаблонами
        ing_match = re.search(
            r'(состав|ингредиенты|ingredients|composition|содержит)[\s:]*([^\n.]+(?:\n[^\n.]+)*)',
            text, re.IGNORECASE
        )

        if ing_match:
            ing_text = ing_match.group(2).strip()
            # Разделение по запятым или точкам с запятой с учетом возможных ошибок OCR
            ingredients = [ing.strip() for ing in re.split(r'[,;]', ing_text) if ing.strip()]

            # Удаляем возможные номера и проценты в начале
            ingredients = [re.sub(r'^[\d\.\s%]+', '', ing) for ing in ingredients]
            # Удаляем возможные номера и проценты в конце
            ingredients = [re.sub(r'[\d\.\s%]+$', '', ing) for ing in ingredients]
            # Удаляем дублирующие пробелы
            ingredients = [re.sub(r'\s+', ' ', ing) for ing in ingredients]
            # Удаляем пустые элементы
            ingredients = [ing for ing in ingredients if ing]

        # Если не найдено по основному шаблону, ищем альтернативные варианты
        if not ingredients:
            # Ищем строки, содержащие возможные ингредиенты
            potential_ingredients = []
            for line in text.split('\n'):
                line = line.strip().lower()
                if "состав" in line or "ингредиент" in line:
                    continue  # Пропускаем заголовок
                # Если строка содержит возможные ингредиенты (без цифр в начале)
                if re.match(r'^[а-яёa-z]', line) and not re.match(r'^\d', line):
                    potential_ingredients.append(line)

            if potential_ingredients:
                ingredients = potential_ingredients[:5]  # Берем первые 5 подходящих строк

        return ingredients

    def _extract_nutritional_values(self, text):
        """Извлекает пищевую ценность с улучшенной обработкой"""
        nutrients = {}

        # Поиск информации о калориях с более гибкими шаблонами
        cal_match = re.search(r'(\d+)\s*(?:ккал|kcal|калории|энергетическая ценность|калории)', text, re.IGNORECASE)
        if cal_match:
            nutrients["calories"] = int(cal_match.group(1))

        # Поиск информации о белках
        protein_match = re.search(r'(\d+[.,]?\d*)\s*(?:г|грамм)\s*белк', text, re.IGNORECASE)
        if protein_match:
            nutrients["protein"] = float(protein_match.group(1).replace(',', '.'))

        # Поиск информации о жирах
        fat_match = re.search(r'(\d+[.,]?\d*)\s*(?:г|грамм)\s*жир', text, re.IGNORECASE)
        if fat_match:
            nutrients["fat"] = float(fat_match.group(1).replace(',', '.'))

        # Поиск информации об углеводах
        carb_match = re.search(r'(\d+[.,]?\d*)\s*(?:г|грамм)\s*углевод', text, re.IGNORECASE)
        if carb_match:
            nutrients["carbohydrates"] = float(carb_match.group(1).replace(',', '.'))

        # Поиск информации о сахаре
        sugar_match = re.search(r'(\d+[.,]?\d*)\s*(?:г|грамм)\s*(?:сахар|сахара)', text, re.IGNORECASE)
        if sugar_match:
            nutrients["sugar"] = float(sugar_match.group(1).replace(',', '.'))

        return nutrients

    def _extract_age_label(self, text):
        """Извлекает возрастную маркировку с улучшенными шаблонами"""
        # Более гибкие шаблоны для поиска возрастной маркировки
        age_patterns = [
            r'возраст\s*[:\-]?\s*(\d+)\s*[-–]\s*(\d+)\s*месяц',
            r'для\s*детей\s*от\s*(\d+)\s*до\s*(\d+)\s*месяц',
            r'с\s*(\d+)\s*месяц[аев]',
            r'от\s*(\d+)\s*месяц[аев]',
            r'(\d+)\s*[+]\s*месяц',
            r'(\d+)\s*месяц[аев]\s*и\s*старше',
            r'(\d+)\s*месяц[аев]',
            r'(\d+)\s*[-–]\s*(\d+)\s*мес',
            r'(\d+)\s*[-–]\s*(\d+)\s*м',
            r'(\d+)\s*[-–]\s*(\d+)\s*месяцев'
        ]

        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) >= 2:
                    return f"{match.group(1)}-{match.group(2)} месяцев"
                else:
                    return f"С {match.group(1)} месяцев"

        # Проверка на наличие явного упоминания "6 месяцев" или подобного
        if re.search(r'6\s*месяц', text, re.IGNORECASE):
            return "С 6 месяцев"
        if re.search(r'1\s*год', text, re.IGNORECASE):
            return "До 12 месяцев"

        # Поиск в тексте похожих формулировок
        similar_phrases = [
            "для малышей", "для детей", "рекомендуется с", "подходит с"
        ]

        for phrase in similar_phrases:
            if phrase in text.lower():
                # Проверяем, есть ли рядом цифры
                match = re.search(r'(\d+)\s*месяц', text, re.IGNORECASE)
                if match:
                    return f"С {match.group(1)} месяцев"

        return "Не указано"

    def _extract_sugar_content(self, text):
        """Извлекает информацию о содержании сахара с улучшенной обработкой"""
        # Поиск информации о содержании сахара
        sugar_match = re.search(r'сахар\s*[:\-]?\s*(\d+[.,]?\d*)\s*г', text, re.IGNORECASE)
        if sugar_match:
            return float(sugar_match.group(1).replace(',', '.'))

        # Поиск информации о добавленных сахара
        if re.search(r'без\s*добавления\s*сахара|без\s*сахара', text, re.IGNORECASE):
            return 0.0

        # Поиск информации о содержании сахара в процентах
        percent_match = re.search(r'сахар\s*[:\-]?\s*(\d+[.,]?\d*)\s*%', text, re.IGNORECASE)
        if percent_match:
            # Предполагаем, что процент относится к 100г продукта
            percent = float(percent_match.group(1).replace(',', '.'))
            # Приблизительный пересчет в граммы (для воды 1% = 1г/100мл, но для других продуктов может отличаться)
            return percent * 1.0  # Упрощенный расчет

        return None

    def _check_prohibited_claims(self, text):
        """Проверяет наличие запрещенных заявлений с fuzzy matching"""
        prohibited = []
        text_lower = text.lower()

        # Проверка заявления о составе
        for claim in self.requirements["prohibited_statements"]["composition_claims"]:
            claim_lower = claim.lower()
            # Используем fuzzy matching для поиска
            if self._is_claim_present(text_lower, claim_lower, threshold=0.75):
                prohibited.append(f"Запрещенное заявление о составе: {claim}")

        # Проверка заявления о пользе для здоровья
        for claim in self.requirements["prohibited_statements"]["health_benefit_claims"]:
            claim_lower = claim.lower()
            if self._is_claim_present(text_lower, claim_lower, threshold=0.75):
                prohibited.append(f"Запрещенное заявление о пользе для здоровья: {claim}")

        # Проверка маркетинговых тезисов
        for claim in self.requirements["prohibited_statements"]["marketing_claims"]:
            claim_lower = claim.lower()
            if self._is_claim_present(text_lower, claim_lower, threshold=0.75):
                prohibited.append(f"Запрещенный маркетинговый тезис: {claim}")

        return prohibited

    def _is_claim_present(self, text, claim, threshold=0.7):
        """
        Проверяет, присутствует ли заявление в тексте с использованием fuzzy matching
        """
        # Проверяем точное вхождение
        if claim in text:
            return True

        # Проверяем вхождение ключевых слов
        claim_words = claim.split()
        if len(claim_words) > 1:
            # Проверяем, есть ли хотя бы 70% ключевых слов
            words_found = sum(1 for word in claim_words if word in text)
            if words_found / len(claim_words) >= 0.7:
                return True

        # Используем SequenceMatcher для fuzzy matching
        matcher = SequenceMatcher(None, text, claim)
        if matcher.ratio() > threshold:
            return True

        # Проверяем вхождение частей заявления
        for i in range(len(claim_words)):
            for j in range(i + 1, min(i + 3, len(claim_words)) + 1):
                phrase = " ".join(claim_words[i:j])
                if phrase in text and len(phrase) > 5:  # Игнорируем короткие фразы
                    return True

        return False

    def evaluate_product(self, image_path):
        """
        Полная оценка продукта с улучшенной обработкой
        """
        # Извлечение текста из изображения
        text = self.extract_text_from_image(image_path)

        # Определение категории продукта
        category = self.identify_product_category(text)

        # Извлечение информации о продукте
        product_info = self.extract_product_info(text)

        # Проверка соответствия требованиям
        compliance_results = self._check_compliance(category, product_info)

        # Сохранение результатов
        self.current_product = {
            "image_path": image_path,
            "extracted_text": text,
            "category": category,
            "info": product_info
        }
        self.evaluation_results = {
            "compliance": compliance_results,
            "category": category,
            "product_info": product_info
        }

        return self.evaluation_results

    def _check_compliance(self, category, product_info):
        """
        Проверяет соответствие продукта требованиям ВОЗ с улучшенной логикой
        """
        # Если категория не определена, попробуем определить по другим признакам
        if not category:
            # Проверка на напиток (категория 8)
            if re.search(r'сок|напиток|нектар', product_info["original_text"].lower()):
                category = "8"
            # Проверка на молочный продукт (категория 2)
            elif re.search(r'йогурт|молоко|сыр|творог', product_info["original_text"].lower()):
                category = "2"
            # Проверка на кашу (категория 1)
            elif re.search(r'каша|крупа|мюсли', product_info["original_text"].lower()):
                category = "1"
            # Проверка на фруктовый продукт (категория 3.1)
            elif re.search(r'пюре|фрукт|яблоко|груша', product_info["original_text"].lower()):
                category = "3.1"

        if not category or category not in self.requirements["content_requirements"]:
            return {
                "status": "unknown",
                "message": "Не удалось определить категорию продукта. Пожалуйста, проверьте качество изображения.",
                "violations": []
            }

        requirements = self.requirements["content_requirements"][category]
        violations = []

        # Проверка энергетической ценности
        if "calories" in product_info["nutritional_values"]:
            cal = product_info["nutritional_values"]["calories"]

            # Проверяем, является ли значение подходящим
            if isinstance(requirements["energy"], dict):
                if "min" in requirements["energy"] and cal < requirements["energy"]["min"]:
                    violations.append(
                        f"Недостаточная энергетическая ценность: {cal} ккал/100г (требуется ≥{requirements['energy']['min']})")
                if "max" in requirements["energy"] and cal > requirements["energy"]["max"]:
                    violations.append(
                        f"Избыточная энергетическая ценность: {cal} ккал/100г (требуется ≤{requirements['energy']['max']})")

        # Проверка содержания сахара
        if requirements["added_sugars"]:
            # Проверяем ингредиенты на наличие сахара и подсластителей
            sugar_ingredients = ["сахар", "фруктоза", "глюкоза", "мед", "сироп", "подсластитель"]
            for ing in product_info["ingredients"]:
                if any(sugar in ing.lower() for sugar in sugar_ingredients):
                    violations.append(f"Продукт содержит добавленные сахара или подсластители: {ing} (запрещено)")
                    break

        # Проверка возрастной маркировки
        if requirements["age_labeling"]:
            if "Не указано" in product_info["age_label"] or "Не определено" in product_info["age_label"]:
                violations.append("Отсутствует возрастная маркировка (должна быть 6-36 месяцев)")
            else:
                # Проверяем, соответствует ли указанная возрастная группа требованиям
                age_match = re.search(r'(\d+)', product_info["age_label"])
                if age_match:
                    min_age = int(age_match.group(1))
                    if min_age > 6:
                        violations.append(
                            f"Возрастная маркировка начинается с {min_age} месяцев (требуется с 6 месяцев)")

        # Проверка запрещенных заявлений
        if product_info["prohibited_claims"]:
            violations.extend(product_info["prohibited_claims"])

        # Проверка для напитков (категория 8)
        if category == "8":
            violations.append("Напитки для детей до 36 месяцев не рекомендуются ВОЗ")

        # Проверка содержания фруктов
        if "fruit_content" in requirements and requirements["fruit_content"]:
            # Ищем упоминания фруктов в ингредиентах
            fruit_keywords = ["яблоко", "груша", "банан", "апельсин", "мандарин", "вишня", "черешня", "слива",
                              "абрикос", "персик"]
            fruit_count = sum(
                1 for ing in product_info["ingredients"] if any(fruit in ing.lower() for fruit in fruit_keywords))

            if fruit_count > 0 and "max" in requirements["fruit_content"]:
                # Если есть информация о процентном содержании
                if product_info["sugar_content"] and product_info["sugar_content"] > requirements["fruit_content"][
                    "max"]:
                    violations.append(
                        f"Содержание фруктов превышает допустимый предел {requirements['fruit_content']['max']}%")

        # Проверка на наличие маркировки высокого содержания сахара
        if "high_sugar_label" in requirements and requirements["high_sugar_label"]:
            if "sugar" in product_info["nutritional_values"]:
                sugar = product_info["nutritional_values"]["sugar"]
                # Проверяем, нужно ли добавлять маркировку
                if sugar > requirements["high_sugar_label"].get("threshold", 0):
                    # Проверяем, есть ли маркировка на упаковке
                    if not re.search(r'высокое содержание сахара|много сахара', product_info["original_text"],
                                     re.IGNORECASE):
                        violations.append("Отсутствует маркировка высокого содержания сахара")

        if not violations:
            return {
                "status": "compliant",
                "message": "Продукт полностью соответствует требованиям ВОЗ",
                "violations": []
            }
        else:
            return {
                "status": "non-compliant",
                "message": f"Продукт не соответствует {len(violations)} требованиям ВОЗ",
                "violations": violations
            }

    def generate_evaluation_report(self):
        """
        Генерирует отчет об оценке с улучшенным форматированием
        """
        if not self.evaluation_results:
            return "Сначала выполните оценку продукта с помощью метода evaluate_product()"

        report = "РЕЗУЛЬТАТ ОЦЕНКИ СООТВЕТСТВИЯ ПРОДУКТА ТРЕБОВАНИЯМ ВОЗ\n"
        report += "=" * 60 + "\n\n"

        # Информация о продукте
        product_info = self.evaluation_results["product_info"]
        category = self.evaluation_results["category"]

        # Определение названия категории
        category_name = "Неизвестная категория"
        if category and category in self.requirements["categories"]:
            cat_info = self.requirements["categories"][category]
            category_name = f"{category} - {cat_info['category']}"

        report += f"ОПРЕДЕЛЕННАЯ КАТЕГОРИЯ: {category_name}\n"
        report += f"НАЗВАНИЕ ПРОДУКТА: {product_info['name']}\n"
        report += f"ВОЗРАСТНАЯ МАРКИРОВКА: {product_info['age_label']}\n\n"

        # Результаты оценки
        result = self.evaluation_results["compliance"]
        report += f"РЕЗУЛЬТАТ: {result['message']}\n\n"

        # Нарушения (если есть)
        if result["violations"]:
            report += "НАЙДЕННЫЕ НАРУШЕНИЯ:\n"
            for i, violation in enumerate(result["violations"], 1):
                report += f"{i}. {violation}\n"

        # Рекомендации
        report += "\nРЕКОМЕНДАЦИИ:\n"
        if result["status"] == "compliant":
            report += "✅ Продукт полностью соответствует всем требованиям ВОЗ для детского питания."
        else:
            report += "❌ Продукт НЕ СООТВЕТСТВУЕТ требованиям ВОЗ для детского питания.\n"
            report += "Рекомендуется пересмотреть состав и маркировку продукта."

        # Добавляем подсказку для пользователя
        report += "\n\nℹ️ Примечание: Если результат кажется некорректным, возможно, распознавание текста было неточным. "
        report += "Попробуйте отправить более четкое изображение этикетки с хорошим освещением."

        return report