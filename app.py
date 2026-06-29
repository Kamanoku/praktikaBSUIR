"""
app.py — Flask-сервер для предсказания пола по ФИО.
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
# Константы для валидации
# ─────────────────────────────────────────────

# Минимальная доля кириллических букв в строке
MIN_CYRILLIC_RATIO = 0.6

# Минимальное количество кириллических букв в строке
MIN_CYRILLIC_CHARS = 4

# Минимальное количество слов
MIN_WORDS = 2

# Паттерн кириллицы
CYRILLIC_RE = re.compile(r'[а-яёА-ЯЁ]')

# Допустимые символы в ФИО: кириллица, пробел, дефис и два вида апострофов
VALID_FIO_RE = re.compile(r"^[а-яёА-ЯЁ\s\-\'’]+$")


# ─────────────────────────────────────────────
# Валидация ввода
# ─────────────────────────────────────────────

def validate_fio(fio: str) -> dict | None:
    """
    Проверяет строку ФИО на «человечность».
    Возвращает None если всё хорошо,
    или словарь {"reason": ..., "hint": ...} если есть проблема.
    """
    cyrillic_chars = CYRILLIC_RE.findall(fio)
    total_letters  = len(re.findall(r'\S', fio))   # не-пробельных символов
    words          = fio.split()

    # 1. Нет кириллицы вообще (цифры, латиница и т.д.)
    if len(cyrillic_chars) == 0:
        return {
            "reason": "В строке нет кириллических букв",
            "hint":   "ФИО должно быть написано на русском языке, например: Иванов Иван Иванович"
        }

    # 2. Слишком мало кириллических букв относительно общей длины
    cyrillic_ratio = len(cyrillic_chars) / max(total_letters, 1)
    if cyrillic_ratio < MIN_CYRILLIC_RATIO:
        return {
            "reason": f"Мало кириллических символов ({int(cyrillic_ratio*100)}% — ожидается ≥{int(MIN_CYRILLIC_RATIO*100)}%)",
            "hint":   "Введите ФИО только русскими буквами без латиницы и цифр"
        }

    # 3. Недопустимые символы (латиница, цифры, спецсимволы)
    if not VALID_FIO_RE.match(fio):
        bad = set(re.findall(r"[^а-яёА-ЯЁ\s\-\'’]", fio))
        return {
            "reason": f"Недопустимые символы: {' '.join(sorted(bad))}",
            "hint":   "ФИО должно содержать только русские буквы, пробелы, дефисы и апострофы"
        }

    # 4. Слишком мало слов
    if len(words) < MIN_WORDS:
        return {
            "reason": "Введено слишком мало слов",
            "hint":   "Укажите хотя бы Фамилию и Имя, например: Иванов Иван"
        }

    # 5. Общая длина кириллических букв слишком маленькая
    if len(cyrillic_chars) < MIN_CYRILLIC_CHARS:
        return {
            "reason": "Строка слишком короткая для ФИО",
            "hint":   "Введите полное ФИО"
        }

    # 6. Слова-заглушки: все буквы одинаковые (аааа, бббб)
    # Игнорируем легитимные восточные частицы и служебные слова
    EASTERN_PARTICLES = {"оглы", "кызы", "ибн", "аль", "ад", "ас", "аш", "эр"}

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

    return None  # всё в порядке


def encode(text: str) -> np.ndarray:
    """Строка ФИО → числовой вектор для модели."""
    seq = [CHAR2IDX.get(c, 0) for c in text[:MAX_LEN]]
    seq += [0] * (MAX_LEN - len(seq))
    return np.array([seq], dtype=np.int32)


# ─────────────────────────────────────────────
# Вспомогательные функции для модели
# ─────────────────────────────────────────────

def clean_fio_for_model(fio: str) -> str:
    """Удаляет служебные восточные частицы, чтобы они не путали нейросеть."""
    particles = r'\b(оглы|кызы|ибн|аль|ад|ас|аш)\b'
    cleaned = re.sub(particles, '', fio, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# ─────────────────────────────────────────────
# Эндпоинт /predict
# ─────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    Принимает: { "fio": "Иванов Иван Иванович" }
    Возвращает:
    {
        "gender":     "Мужской" | "Женский" | "Неизвестно",
        "confidence": 0.50..1.00,
        "raw_prob":   0.00..1.00,
        "warning":    null | { "reason": "...", "hint": "..." }
    }
    """
    body = request.get_json(silent=True)

    if not body or "fio" not in body:
        return jsonify({"error": "Поле 'fio' обязательно"}), 400

    fio = str(body["fio"]).strip()
    if not fio:
        return jsonify({"error": "ФИО не может быть пустым"}), 400

    # ── Шаг 1: валидация ─────────────────────────────────────
    validation_error = validate_fio(fio)

    if validation_error:
        # Возвращаем «технический» ответ с низкой уверенностью
        # и объяснением проблемы — фронтенд покажет алерт
        return jsonify({
            "gender":     "Неизвестно",
            "confidence": 0.50,
            "raw_prob":   0.50,
            "warning":    validation_error   # {"reason": ..., "hint": ...}
        })

    # ── Шаг 2: предсказание модели ───────────────────────────
    cleaned_fio = clean_fio_for_model(fio)
    x = encode(cleaned_fio)
    raw_prob = float(model.predict(x, verbose=0)[0][0])

    if raw_prob > 0.5:
        gender     = "Женский"
        confidence = raw_prob
    elif raw_prob < 0.5:
        gender     = "Мужской"
        confidence = 1.0 - raw_prob
    else:
        gender     = "Неизвестно"
        confidence = 0.50

    confidence = max(0.5, min(1.0, confidence))

    return jsonify({
        "gender":     gender,
        "confidence": round(confidence, 4),
        "raw_prob":   round(raw_prob, 4),
        "warning":    None    # валидация пройдена
    })


# ─────────────────────────────────────────────
# Health-check
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "gender_model.keras"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)