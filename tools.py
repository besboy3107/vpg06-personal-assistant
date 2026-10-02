"""Инструменты для Haystack-агента.

Инструменты:
  get_cat_fact              — случайный факт о кошках (catfact.ninja)
  get_dog_image_and_analysis — фото собаки + анализ породы через OpenAI Vision
  get_weather               — текущая погода в городе (wttr.in) [НОВЫЙ ИНСТРУМЕНТ]
"""

from __future__ import annotations

import os
import requests
from openai import OpenAI


def get_cat_fact() -> str:
    """Получить случайный интересный факт о кошках."""
    resp = requests.get("https://catfact.ninja/fact", timeout=8)
    resp.raise_for_status()
    return resp.json().get("fact", "Факт недоступен")


def get_dog_image_and_analysis() -> str:
    """Получить случайное фото собаки и определить её породу.

    Возвращает строку вида:
      Изображение: <URL>
      <анализ породы>
    URL нужно передать пользователю как ссылку на фото.
    """
    # Получаем случайное фото с dog.ceo API
    resp = requests.get("https://dog.ceo/api/breeds/image/random", timeout=8)
    resp.raise_for_status()
    image_url = resp.json()["message"]

    # Анализируем породу через OpenAI Vision
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    vision_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Определи породу собаки на фото. "
                        "Расскажи кратко: название породы, страна происхождения, "
                        "характер, интересные особенности. "
                        "Ответ на русском языке, 3-4 предложения."
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
        max_tokens=350,
    )
    analysis = vision_resp.choices[0].message.content.strip()
    return f"Изображение: {image_url}\n\n{analysis}"


def get_weather(city: str) -> str:
    """Получить текущую погоду в указанном городе.

    Args:
        city: Название города на английском языке (например: Moscow, Paris, Tokyo).
    """
    resp = requests.get(
        f"https://wttr.in/{city}?format=j1",
        timeout=8,
        headers={"Accept-Language": "ru"},
    )
    resp.raise_for_status()
    data = resp.json()

    current = data["current_condition"][0]
    desc = current["weatherDesc"][0]["value"]
    temp_c = current["temp_C"]
    feels_c = current["FeelsLikeC"]
    humidity = current["humidity"]
    wind_kmh = current["windspeedKmph"]

    # Читаем человекочитаемое название из ответа
    areas = data.get("nearest_area", [])
    area_name = areas[0]["areaName"][0]["value"] if areas else city

    return (
        f"Погода в {area_name}: {desc}. "
        f"Температура {temp_c}°C (ощущается как {feels_c}°C). "
        f"Влажность {humidity}%, ветер {wind_kmh} км/ч."
    )
