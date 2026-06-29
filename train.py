"""
train.py — Обучение Character-level Bidirectional LSTM для определения пола по ФИО.

ИСТОЧНИКИ ДАТАСЕТА (выбираются автоматически, в порядке приоритета):
  1. Kaggle  — rai220/russian-cyrillic-names-and-sex  (ФИО целиком + пол)
  3. Синтетический fallback                           (встроенные списки имён)

Запуск:  python train.py
Результат: gender_model.keras  +  char_vocab.json
"""

import os
import json
import random
import numpy as np

# ─────────────────────────────────────────────────────────────────
# 0. ВСПОМОГАТЕЛЬНЫЕ ДАННЫЕ (используются в fallback и при сборке
#    ФИО из датасетов, содержащих только имена без фамилий)
# ─────────────────────────────────────────────────────────────────

MALE_LAST    = ["Иванов","Смирнов","Кузнецов","Попов","Васильев","Петров","Соколов",
                "Михайлов","Новиков","Фёдоров","Морозов","Волков","Алексеев","Лебедев",
                "Семёнов","Егоров","Павлов","Козлов","Степанов","Николаев","Орлов",
                "Андреев","Макаров","Никитин","Захаров","Зайцев","Соловьёв","Борисов",
                "Яковлев","Григорьев","Романов","Воробьёв","Сергеев","Кузьмин","Фролов",
                "Александров","Дмитриев","Королёв","Гусев","Киселёв"]

FEMALE_LAST  = [s + "а" if not s.endswith("а") else s for s in MALE_LAST]

MALE_MIDDLE  = ["Александрович","Алексеевич","Андреевич","Антонович","Артёмович",
                "Борисович","Вадимович","Васильевич","Викторович","Витальевич",
                "Владимирович","Владиславович","Вячеславович","Георгиевич","Иванович",
                "Игоревич","Ильич","Кириллович","Леонидович","Максимович",
                "Михайлович","Никитич","Николаевич","Олегович","Павлович",
                "Петрович","Романович","Русланович","Сергеевич","Юрьевич"]

FEMALE_MIDDLE= ["Александровна","Алексеевна","Андреевна","Антоновна","Артёмовна",
                "Борисовна","Вадимовна","Васильевна","Викторовна","Витальевна",
                "Владимировна","Владиславовна","Вячеславовна","Георгиевна","Ивановна",
                "Игоревна","Ильинична","Кирилловна","Леонидовна","Максимовна",
                "Михайловна","Никитична","Николаевна","Олеговна","Павловна",
                "Петровна","Романовна","Руслановна","Сергеевна","Юрьевна"]

MALE_FIRST_FALLBACK = [
    "Александр","Алексей","Андрей","Антон","Артём","Борис","Вадим","Василий",
    "Виктор","Виталий","Владимир","Владислав","Вячеслав","Георгий","Григорий",
    "Даниил","Денис","Дмитрий","Евгений","Иван","Игорь","Илья","Кирилл",
    "Константин","Леонид","Максим","Михаил","Никита","Николай","Олег",
    "Павел","Пётр","Роман","Руслан","Сергей","Степан","Тимофей","Фёдор"
]

FEMALE_FIRST_FALLBACK = [
    "Александра","Алина","Алиса","Анастасия","Анна","Валентина","Валерия",
    "Вера","Виктория","Галина","Дарья","Диана","Екатерина","Елена","Елизавета",
    "Жанна","Зинаида","Инна","Ирина","Карина","Кристина","Ксения","Лариса",
    "Людмила","Маргарита","Марина","Мария","Надежда","Наталья","Нина",
    "Оксана","Ольга","Полина","Светлана","София","Тамара","Татьяна","Юлия"
]


# ─────────────────────────────────────────────────────────────────
# 1. ЗАГРУЗКА ДАТАСЕТА
# ─────────────────────────────────────────────────────────────────

def load_kaggle_dataset(csv_path: str):
    """
    Kaggle: rai220/russian-cyrillic-names-and-sex
    Колонки: surname | name | patronymic | sex  (м / ж)
    Скачать: kaggle datasets download -d rai220/russian-cyrillic-names-and-sex
    CSV распакуйте в текущую папку как data.csv
    """
    import pandas as pd
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Приводим названия колонок к нижнему регистру
    df.columns = [c.lower().strip() for c in df.columns]

    # Возможные варианты названия колонки с полом
    sex_col = next((c for c in df.columns if c in ("sex", "gender", "пол", "sex_ru")), None)
    if sex_col is None:
        raise ValueError(f"Не найдена колонка пола. Колонки: {list(df.columns)}")

    records = []
    for _, row in df.iterrows():
        parts = []
        for col in ("surname", "фамилия", "lastname"):
            if col in df.columns and str(row.get(col, "")).strip():
                parts.append(str(row[col]).strip())
                break
        for col in ("name", "имя", "firstname", "first_name"):
            if col in df.columns and str(row.get(col, "")).strip():
                parts.append(str(row[col]).strip())
                break
        for col in ("patronymic", "отчество", "middlename", "middle_name"):
            if col in df.columns and str(row.get(col, "")).strip():
                parts.append(str(row[col]).strip())
                break

        fio   = " ".join(parts).strip()
        sex   = str(row[sex_col]).strip().lower()
        label = 1 if sex in ("ж", "f", "female", "женский", "w") else 0

        if fio:
            records.append((fio, label))

    print(f"[Kaggle] Загружено {len(records)} записей из {csv_path}")
    return records




def generate_synthetic_dataset(n_samples: int = 4000):
    """
    Встроенный fallback: генерация синтетических ФИО.
    Используется только если ни Kaggle, ни HuggingFace недоступны.
    """
    data = []
    for _ in range(n_samples // 2):
        fio = (random.choice(MALE_LAST)   + " " +
               random.choice(MALE_FIRST_FALLBACK) + " " +
               random.choice(MALE_MIDDLE))
        data.append((fio, 0))
    for _ in range(n_samples // 2):
        fio = (random.choice(FEMALE_LAST)  + " " +
               random.choice(FEMALE_FIRST_FALLBACK) + " " +
               random.choice(FEMALE_MIDDLE))
        data.append((fio, 1))
    random.shuffle(data)
    print(f"[Synthetic] Сгенерировано {len(data)} записей.")
    return data


def load_dataset_auto(kaggle_csv: str = "data.csv"):
    """
    Автоматический выбор источника данных:
      1. Если рядом с train.py лежит data.csv  → Kaggle

      3. Иначе — синтетический датасет
    """
    # ── Вариант 1: Kaggle CSV ───────────────────────────────────
    if os.path.exists(kaggle_csv):
        try:
            return load_kaggle_dataset(kaggle_csv)
        except Exception as e:
            print(f"[Kaggle] Ошибка чтения {kaggle_csv}: {e}")


    # ── Вариант 3: синтетика ────────────────────────────────────
    print("[Fallback] Используем встроенный синтетический датасет.")
    return generate_synthetic_dataset(4000)


# ─────────────────────────────────────────────────────────────────
# 2. ТОКЕНИЗАЦИЯ НА УРОВНЕ СИМВОЛОВ
# ─────────────────────────────────────────────────────────────────

def build_vocab(texts):
    """Словарь всех уникальных символов (0 зарезервирован под PAD)."""
    chars = sorted(set("".join(texts)))
    return {c: i + 1 for i, c in enumerate(chars)}


def encode(text, char2idx, max_len=70):
    """Строка → числовой вектор длиной max_len с padding нулями справа."""
    seq = [char2idx.get(c, 0) for c in text[:max_len]]
    seq += [0] * (max_len - len(seq))
    return seq


# ─────────────────────────────────────────────────────────────────
# 3. АРХИТЕКТУРА МОДЕЛИ
# ─────────────────────────────────────────────────────────────────

def build_model(vocab_size, max_len=70, embed_dim=32, lstm_units=64):
    """
    Character-level Bidirectional LSTM.
    Вход : последовательность индексов символов (max_len,)
    Выход: скаляр sigmoid → 0 = мужчина, 1 = женщина
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        # Embedding: индекс символа → плотный вектор
        layers.Embedding(input_dim=vocab_size + 1,
                         output_dim=embed_dim,
                         input_length=max_len,
                         mask_zero=True),       # PAD-токены игнорируются

        # Bidirectional LSTM — читает ФИО в обоих направлениях,
        # благодаря чему лучше улавливает суффиксы (-ова/-ов, -вна/-вич)
        layers.Bidirectional(layers.LSTM(lstm_units, dropout=0.2)),

        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(1, activation="sigmoid")  # вероятность класса "женщина"
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─────────────────────────────────────────────────────────────────
# 4. ОБУЧЕНИЕ И СОХРАНЕНИЕ
# ─────────────────────────────────────────────────────────────────

def main():
    import tensorflow as tf
    from tensorflow import keras

    MAX_LEN = 70     # максимальная длина строки ФИО
    EPOCHS  = 20     # EarlyStopping остановит раньше при отсутствии прогресса
    BATCH   = 64

    # ── Шаг 1: загрузка данных ──────────────────────────────────
    dataset = load_dataset_auto(kaggle_csv="data.csv")

    # Если данных мало, доберём синтетикой до минимума
    if len(dataset) < 2000:
        print(f"Данных мало ({len(dataset)}), добавляем синтетику…")
        dataset += generate_synthetic_dataset(2000)

    random.shuffle(dataset)

    texts  = [d[0] for d in dataset]
    labels = np.array([d[1] for d in dataset], dtype=np.float32)

    print(f"\nВсего примеров: {len(texts)}")
    print(f"  Мужские: {int((labels == 0).sum())}")
    print(f"  Женские: {int((labels == 1).sum())}\n")

    # ── Шаг 2: символьный словарь ───────────────────────────────
    char2idx = build_vocab(texts)
    print(f"Размер словаря: {len(char2idx)} уникальных символов")

    # ── Шаг 3: кодирование ──────────────────────────────────────
    X = np.array([encode(t, char2idx, MAX_LEN) for t in texts])
    y = labels

    split   = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # ── Шаг 4: модель ───────────────────────────────────────────
    model = build_model(vocab_size=len(char2idx), max_len=MAX_LEN)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        )
    ]

    # ── Шаг 5: обучение ─────────────────────────────────────────
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=callbacks
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n✅ Итог — val accuracy: {val_acc:.4f} | val loss: {val_loss:.4f}")

    # ── Шаг 6: сохранение ───────────────────────────────────────
    model.save("gender_model.keras")
    print("Модель сохранена → gender_model.keras")

    with open("char_vocab.json", "w", encoding="utf-8") as f:
        json.dump({"char2idx": char2idx, "max_len": MAX_LEN}, f, ensure_ascii=False)
    print("Словарь сохранён → char_vocab.json")


if __name__ == "__main__":
    main()
