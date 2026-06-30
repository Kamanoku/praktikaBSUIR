"""
app.py — Flask-сервер для проверки соответствия ФИО выбранному полу.
Запуск: python app.py
Требует файлы gender_model.keras и char_vocab.json (создаёт train.py).
"""

import re
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# Загрузка модели и словаря при старте сервера
# ─────────────────────────────────────────────

print("Загрузка модели...")
model = tf.keras.models.load_model("gender_model.keras")

print("Загрузка словаря символов...")
with open("char_vocab.json", encoding="utf-8") as f:
    vocab_data = json.load(f)

CHAR2IDX = vocab_data["char2idx"]
MAX_LEN  = vocab_data["max_len"]

# ─────────────────────────────────────────────
# Константы валидации
# ─────────────────────────────────────────────

MIN_CYRILLIC_RATIO = 0.6
MIN_CYRILLIC_CHARS = 2
MIN_WORDS = 2
MAX_WORDS = 3

CYRILLIC_RE = re.compile(r'[а-яёА-ЯЁ]')
VALID_FIO_RE = re.compile(r"^[а-яёА-ЯЁ\s\-\'’]+$")

EASTERN_PARTICLES = {"оглы", "кызы", "ибн", "аль", "ад", "ас", "аш"}

# ─────────────────────────────────────────────
# TEMPERATURE SCALING — калибровка уверенности модели
# ─────────────────────────────────────────────
# Проблема: sigmoid на необученных краевых случаях часто выдаёт
# вероятности, близкие к 0.5 (полная неопределённость) или почти
# 0.99+ (сверхуверенность), и почти никогда — что-то среднее.
# Это типичный эффект "overconfidence" у LSTM/Dense сетей,
# обученных через binary_crossentropy без регуляризации вероятностей.
#
# Temperature scaling — стандартный метод калибровки (Guo et al., 2017):
# вместо sigmoid(logit) считаем sigmoid(logit / T), где T > 1 "сглаживает"
# распределение вероятностей, делая модель менее самоуверенной.
#
# T подбирается эмпирически. T=2.5–3.5 обычно даёт более плавный
# и реалистичный разброс уверенности (60-90% вместо 50%/100%).

TEMPERATURE = 3.0

def calibrate_probability(raw_prob: float, temperature: float = TEMPERATURE) -> float:
    """
    Пересчитывает вероятность модели через temperature scaling.
    raw_prob — исходная вероятность модели (0..1) класса "женщина".
    Возвращает сглаженную вероятность того же класса.
    """
    # Защита от log(0) / log(1)
    eps = 1e-7
    p = np.clip(raw_prob, eps, 1 - eps)

    # Переводим вероятность обратно в логит: logit = ln(p / (1-p))
    logit = np.log(p / (1 - p))

    # Делим логит на температуру — "сжимаем" уверенность к центру
    scaled_logit = logit / temperature

    # Возвращаем обратно в вероятность через sigmoid
    calibrated = 1 / (1 + np.exp(-scaled_logit))

    return float(calibrated)


# ─────────────────────────────────────────────
# Валидация ввода (проверка на "человечность" строки)
# ─────────────────────────────────────────────

def validate_fio(fio: str) -> dict | None:
    """
    Проверяет, похожа ли строка на настоящее ФИО.
    Возвращает None если всё хорошо, иначе {"reason": ..., "hint": ...}.
    """
    cyrillic_chars = CYRILLIC_RE.findall(fio)
    total_letters  = len(re.findall(r'\S', fio)) # Количество букв без пробелов
    # Разбиваем на слова и игнорируем служебные восточные частицы при подсчете
    words = [w for w in fio.split() if w.lower() not in EASTERN_PARTICLES]

    if len(cyrillic_chars) == 0:
        return {
            "reason": "В строке нет кириллических букв",
            "hint":   "ФИО должно быть написано на русском языке, например: Иванов Иван Иванович"
        }

    cyrillic_ratio = len(cyrillic_chars) / max(total_letters, 1)
    if cyrillic_ratio < MIN_CYRILLIC_RATIO:
        return {
            "reason": f"Мало кириллических символов ({int(cyrillic_ratio*100)}% — ожидается ≥{int(MIN_CYRILLIC_RATIO*100)}%)",
            "hint":   "Введите ФИО только русскими буквами без латиницы и цифр"
        }

    if not VALID_FIO_RE.match(fio):
        bad = set(re.findall(r"[^а-яёА-ЯЁ\s\-\'’]", fio))
        return {
            "reason": f"Недопустимые символы: {' '.join(sorted(bad))}",
            "hint":   "ФИО должно содержать только русские буквы, пробелы, дефисы и апострофы"
        }

    if len(words) < MIN_WORDS:
        return {
            "reason": "Введено слишком мало слов",
            "hint":   "Укажите хотя бы Фамилию и Имя, например: Иванов Иван"
        }

    # ─── НОВАЯ ПРОВЕРКА: ограничение на максимальное количество слов ───
    if len(words) > MAX_WORDS:
        return {
            "reason": f"Введено слишком много слов ({len(words)})",
            "hint": "ФИО должно состоять максимум из 3 слов: Фамилия Имя Отчество"
        }

    if len(cyrillic_chars) < MIN_CYRILLIC_CHARS:
        return {
            "reason": "Строка слишком короткая для ФИО",
            "hint":   "Введите полное ФИО"
        }

    # ─── НОВАЯ ПРОВЕРКА: Сверхкороткие ФИО (от 2 до 4 символов включительно) ───
    if 2 <= total_letters <= 4:
        return {
            "reason": f"Введённое ФИО подозрительно короткое ({total_letters} симв.)",
            "hint": "Убедитесь, что данные введены верно (например: «У И» или «Ли Ан»)"
        }

    suspicious_words = []
    for w in words:
        word_lower = w.lower()
        if word_lower in EASTERN_PARTICLES:
            continue
        letters = re.findall(r'[а-яёА-ЯЁ]', w)
        if len(letters) >= 3 and len(set(l.lower() for l in letters)) == 1:
            suspicious_words.append(w)

    words_to_analyze = [w for w in words if w.lower() not in EASTERN_PARTICLES]
    if words_to_analyze and len(suspicious_words) == len(words_to_analyze):
        return {
            "reason": "Все слова выглядят как заглушки (все буквы одинаковые)",
            "hint":   "Введите настоящее ФИО на русском языке"
        }

    return None


def clean_fio_for_model(fio: str) -> str:
    """Удаляет служебные восточные частицы перед подачей в модель."""
    particles = r'\b(оглы|кызы|ибн|аль|ад|ас|аш)\b'
    cleaned = re.sub(particles, '', fio, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def encode(text: str) -> np.ndarray:
    """Строка ФИО → числовой вектор для модели."""
    seq = [CHAR2IDX.get(c, 0) for c in text[:MAX_LEN]]
    seq += [0] * (MAX_LEN - len(seq))
    return np.array([seq], dtype=np.int32)


# ─────────────────────────────────────────────
# Эндпоинт /predict
# ─────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    Принимает:
    {
        "fio":            "Иванов Иван Иванович",
        "selected_gender": "male" | "female"   # пол, выбранный пользователем
    }

    Возвращает:
    {
        "selected_gender":  "male" | "female",
        "predicted_gender": "Мужской" | "Женский" | "Неизвестно",
        "match":            true | false | null,   # null если валидация не пройдена
        "confidence":       0.50..1.00,             # калиброванная уверенность модели
        "raw_prob":         0.00..1.00,             # вероятность до калибровки
        "warning":          null | { "reason": "...", "hint": "..." }
    }
    """
    body = request.get_json(silent=True)

    if not body or "fio" not in body:
        return jsonify({"error": "Поле 'fio' обязательно"}), 400

    fio = str(body["fio"]).strip()
    if not fio:
        return jsonify({"error": "ФИО не может быть пустым"}), 400

    selected_gender = body.get("selected_gender")
    if selected_gender not in ("male", "female"):
        return jsonify({"error": "Поле 'selected_gender' должно быть 'male' или 'female'"}), 400

    # ── Шаг 1: валидация строки ──────────────────────────────
    validation_error = validate_fio(fio)

    if validation_error:
        return jsonify({
            "selected_gender":  selected_gender,
            "predicted_gender": "Неизвестно",
            "match":            None,
            "confidence":       0.50,
            "raw_prob":         0.50,
            "warning":          validation_error
        })

    # ── Шаг 2: предсказание модели + калибровка ──────────────
    cleaned_fio = clean_fio_for_model(fio)
    x = encode(cleaned_fio)
    raw_prob = float(model.predict(x, verbose=0)[0][0])    # вероятность класса "женщина"

    # Применяем temperature scaling, чтобы избежать 50%/100% перекосов
    calibrated_prob = calibrate_probability(raw_prob)

    if calibrated_prob >= 0.5:
        predicted_gender = "Женский"
        confidence        = calibrated_prob
    else:
        predicted_gender = "Мужской"
        confidence        = 1.0 - calibrated_prob

    confidence = max(0.5, min(1.0, confidence))

    # ── Шаг 3: сравнение с выбором пользователя ──────────────
    predicted_code = "female" if predicted_gender == "Женский" else "male"
    match = (predicted_code == selected_gender)

    return jsonify({
        "selected_gender":  selected_gender,
        "predicted_gender": predicted_gender,
        "match":            match,
        "confidence":       round(confidence, 4),
        "raw_prob":         round(raw_prob, 4),
        "warning":          None
    })


# ─────────────────────────────────────────────
# Health-check
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "gender_model.keras", "temperature": TEMPERATURE})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
