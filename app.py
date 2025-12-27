"""Streamlit приложение для прогнозирования цен автомобилей."""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Константы
ROOT = Path(__file__).resolve().parent
ARTIFACTS_PATH = ROOT / "models" / "car_price_artifacts.pkl"
EPSILON = 1e-6

# Настройка модулей для совместимости с pickle
sys.modules.setdefault("main", sys.modules[__name__])
sys.modules.setdefault("__main__", sys.modules[__name__])


@dataclass
class ModelMetadata:
    """Метаданные модели и конфигурация признаков."""

    feature_names: list[str]
    numeric_cols: list[str]
    cat_cols: list[str]
    categories: dict[str, list[Any]]
    medians: dict[str, float]


def split_name_column(df):
    """Разбивает колонку name на brand, model и version."""
    df = df.copy()

    if "name" not in df.columns:
        return df

    parts = df["name"].astype(str).str.split()
    df["brand"] = parts.str[0].replace({"nan": np.nan})
    df["model"] = parts.str[1].replace({"nan": np.nan})
    df["version"] = parts.str[2:].apply(
        lambda x: " ".join(x) if isinstance(x, list) else ""
    )

    return df.drop(columns=["name"])


def add_brand_model_version(df):
    """Добавляет колонки brand, model, version из колонки name."""
    df = df.copy()

    if "name" not in df.columns:
        return df

    parts = df["name"].astype(str).str.split()
    df["brand"] = parts.str[0].replace({"nan": np.nan})
    df["model"] = parts.str[1].replace({"nan": np.nan})
    df["version"] = parts.apply(
        lambda x: " ".join(x[2:]) if isinstance(x, list) and len(x) > 2 else ""
    )

    # Если name был NaN, то и производные должны быть NaN
    name_is_nan = df["name"].isna()
    df.loc[name_is_nan, ["brand", "model", "version"]] = np.nan

    return df


def add_handcrafted_features(df):
    """Добавляет сконструированные признаки."""
    df = df.copy()

    if "engine" in df.columns:
        df["engine_liters"] = df["engine"] / 1000.0

    if "engine_liters" in df.columns and "max_power" in df.columns:
        safe_engine = df["engine_liters"].replace(0, np.nan) + EPSILON
        df["power_per_liter"] = df["max_power"] / safe_engine

    if "engine_liters" in df.columns and "torque_nm" in df.columns:
        safe_engine = df["engine_liters"].replace(0, np.nan) + EPSILON
        df["torque_per_liter"] = df["torque_nm"] / safe_engine

    if "year" in df.columns:
        ref_year = df["year"].max()
        df["car_age"] = (ref_year - df["year"]).clip(lower=0)

    if "car_age" in df.columns and "km_driven" in df.columns:
        safe_age = df["car_age"].replace(0, np.nan) + EPSILON
        df["km_per_year"] = df["km_driven"] / safe_age

    if "km_driven" in df.columns:
        df["log_km_driven"] = np.log1p(df["km_driven"])

    if "max_power" in df.columns:
        df["log_max_power"] = np.log1p(df["max_power"].clip(lower=0))

    return df


def align_with_features(df, feature_names):
    """Выравнивает колонки DataFrame с ожидаемыми признаками."""
    df = add_brand_model_version(df)

    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan

    return df[feature_names]


def is_int_like(value):
    """Проверяет, можно ли значение представить как целое число."""
    try:
        return float(value).is_integer()
    except (TypeError, ValueError, AttributeError):
        return False


class ArtifactUnpickler(pickle.Unpickler):
    """Unpickler с подменой отсутствующих функций."""

    def __init__(self, file, extra_funcs):
        """Инициализация unpickler."""
        super().__init__(file)
        self.extra_funcs = extra_funcs

    def find_class(self, module, name):
        """Поиск класса или функции по имени."""
        if name in self.extra_funcs:
            return self.extra_funcs[name]
        return super().find_class(module, name)


@st.cache_resource(show_spinner=True)
def load_artifacts(path=ARTIFACTS_PATH):
    """Загружает pipeline и метаданные из файла артефактов."""
    if not path.exists():
        raise FileNotFoundError(f"Файл артефактов не найден: {path}")

    # Функции, которые могут быть в pickle
    extra_funcs = {
        "split_name_column": split_name_column,
        "add_brand_model_version": add_brand_model_version,
        "add_handcrafted_features": add_handcrafted_features,
    }

    with open(path, "rb") as f:
        artifacts = ArtifactUnpickler(f, extra_funcs).load()

    # Извлечение pipeline и словаря артефактов
    pipeline = None
    artifacts_dict = {}

    if hasattr(artifacts, "named_steps"):
        pipeline = artifacts
    elif isinstance(artifacts, dict):
        artifacts_dict = artifacts
        pipeline = artifacts_dict.get("pipeline")

        # Регистрация функций в модуле main
        for name, val in artifacts_dict.items():
            if callable(val):
                setattr(sys.modules["main"], name, val)

    # Валидация pipeline
    if pipeline is None:
        raise KeyError("Pipeline не найден в артефактах")

    if not hasattr(pipeline, "named_steps") or "preprocess" not in pipeline.named_steps:
        raise TypeError("Ожидается sklearn Pipeline с шагом preprocess")

    preprocess = pipeline.named_steps["preprocess"]
    if not hasattr(preprocess, "feature_names_in_"):
        raise AttributeError(
            "У preprocess нет атрибута feature_names_in_. "
            "Pipeline должен быть обучен на DataFrame."
        )

    # Извлечение метаданных
    ohe = artifacts_dict.get("ohe")
    cat_cols = list(artifacts_dict.get("cat_cols", []))
    categories = {}

    if hasattr(ohe, "categories_"):
        categories = {
            col: list(opts) for col, opts in zip(cat_cols, ohe.categories_)
        }

    metadata = ModelMetadata(
        feature_names=list(preprocess.feature_names_in_),
        numeric_cols=list(artifacts_dict.get("numeric_cols", [])),
        cat_cols=cat_cols,
        categories=categories,
        medians=artifacts_dict.get("median_dict", {}),
    )

    return pipeline, metadata


def render_single_car_mode(pipeline, metadata):
    """Отрисовка интерфейса одиночного предсказания."""
    st.header("🧍 Одиночный прогноз")

    with st.form("single_car_form"):
        col1, col2 = st.columns(2)
        inputs = {}

        numeric_cols = set(metadata.numeric_cols)
        cat_cols = set(metadata.cat_cols)

        # Создание виджетов ввода для каждого признака
        for idx, col_name in enumerate(metadata.feature_names):
            target_col = col1 if idx % 2 == 0 else col2

            if col_name in cat_cols:
                # Категориальный признак
                options = metadata.categories.get(col_name, [])
                if options:
                    inputs[col_name] = target_col.selectbox(f"{col_name}", options)
                else:
                    inputs[col_name] = target_col.text_input(f"{col_name}")

            elif col_name in numeric_cols:
                # Числовой признак
                median = metadata.medians.get(col_name, 0.0)
                if median is None:
                    median = 0.0

                step = 1 if is_int_like(median) else 0.1
                default_val = int(median) if is_int_like(median) else float(median)

                inputs[col_name] = target_col.number_input(
                    f"{col_name}", value=default_val, step=step
                )
            else:
                # Неизвестный тип
                inputs[col_name] = target_col.text_input(f"{col_name}")

        submitted = st.form_submit_button("Предсказать цену")

    if submitted:
        df_single = pd.DataFrame([inputs])

        st.subheader("Входные данные")
        st.write(df_single)

        df_single = align_with_features(df_single, metadata.feature_names)

        try:
            y_pred = pipeline.predict(df_single)[0]
            st.success(f"Оценочная цена: **{y_pred:,.0f}**")
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")


def render_batch_mode(pipeline, metadata):
    """Отрисовка интерфейса пакетного предсказания."""
    st.header("📁 Пакетное предсказание по CSV")

    st.markdown(
        """
        Загрузите CSV с колонками, совпадающими с обучающими данными.
        Отсутствующие колонки будут заполнены пропусками.

        **Примеры колонок:** name, year, km_driven, fuel, seller_type,
        transmission, owner, mileage, engine, max_power, seats, torque_nm, torque_rpm
        """
    )

    uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Не удалось прочитать CSV: {e}")
            return

        st.subheader("Первые строки входных данных")
        st.write(df_input.head())

        if st.button("Сделать предсказания"):
            df_for_model = align_with_features(df_input.copy(), metadata.feature_names)

            try:
                preds = pipeline.predict(df_for_model)
                df_result = df_input.copy()
                df_result["predicted_price"] = preds

                st.subheader("Результаты")
                st.write(df_result.head())

                csv_bytes = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Скачать результаты в CSV",
                    data=csv_bytes,
                    file_name="car_price_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Ошибка при предсказании: {e}")


def main():
    """Главная функция приложения."""
    st.set_page_config(
        page_title="Прогноз цены авто", page_icon="🚗", layout="wide"
    )

    st.title("🚗 Прогноз стоимости автомобиля")
    st.markdown(
        """
        Демо-приложение для прогнозирования цен автомобилей.

        - Модель и препроцессинг упакованы в sklearn Pipeline
        - Автоматическая адаптация к признакам из обучающих данных
        - Поддержка одиночных предсказаний и пакетной обработки CSV
        """
    )

    # Загрузка pipeline и метаданных
    try:
        pipeline, metadata = load_artifacts(ARTIFACTS_PATH)
    except Exception as e:
        st.error(f"Не удалось загрузить артефакты: {e}")
        st.stop()

    # Выбор режима работы
    mode = st.sidebar.radio(
        "Режим работы:",
        ("Одиночный автомобиль", "Пакетное предсказание (CSV)"),
    )

    # Отрисовка соответствующего интерфейса
    if mode == "Одиночный автомобиль":
        render_single_car_mode(pipeline, metadata)
    elif mode == "Пакетное предсказание (CSV)":
        render_batch_mode(pipeline, metadata)


if __name__ == "__main__":
    main()
