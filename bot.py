"""Telegram-бот с Haystack-агентом и Pinecone-памятью (VPg06).

Архитектура:
  - Haystack Agent обрабатывает запросы и вызывает инструменты.
  - Pinecone хранит только тексты сообщений пользователя (не ответы бота).
  - Краткосрочная история в оперативной памяти (dict по user_id).

Инструменты агента:
  get_cat_fact              — факт о кошках
  get_dog_image_and_analysis — фото собаки + анализ породы
  get_weather               — текущая погода в городе [НОВЫЙ]
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Optional

import telebot
from dotenv import load_dotenv
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from haystack.utils import Secret

from pinecone_manager import PineconeManager
from tools import get_cat_fact, get_dog_image_and_analysis, get_weather

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

load_dotenv()

# ── Конфигурация ─────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
SHORT_TERM_LIMIT = 10  # пар (user+assistant)

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")

# ── Клиенты ──────────────────────────────────────────────────────────────────
memory = PineconeManager()
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode=None)
chat_histories: dict[int, list[ChatMessage]] = {}

# ── Описание инструментов ────────────────────────────────────────────────────
TOOLS = [
    Tool(
        name="get_cat_fact",
        function=get_cat_fact,
        description="Получить случайный интересный факт о кошках.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="get_dog_image_and_analysis",
        function=get_dog_image_and_analysis,
        description=(
            "Получить случайное фото собаки и анализ её породы от AI. "
            "Ответ содержит URL изображения и описание породы. "
            "Обязательно включи URL изображения в свой ответ пользователю."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="get_weather",
        function=get_weather,
        description="Получить текущую погоду в указанном городе.",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Название города на английском, например: Moscow, Paris, Tokyo.",
                }
            },
            "required": ["city"],
        },
    ),
]

SYSTEM_PROMPT_BASE = """Ты — умный персональный ассистент в Telegram.

У тебя есть инструменты:
- get_cat_fact: случайный факт о кошках
- get_dog_image_and_analysis: случайное фото собаки + AI-анализ породы
- get_weather: текущая погода в городе (передай название города на английском)

Правила использования инструментов:
- Если пользователь просит факт о кошках — вызови get_cat_fact.
- Если пользователь просит фото собаки или информацию о породе — вызови get_dog_image_and_analysis. При ответе всегда включай URL изображения дословно — не сокращай и не убирай его.
- Если пользователь спрашивает о погоде — вызови get_weather с названием города.
- На остальные вопросы отвечай самостоятельно.

Если в долговременной памяти есть информация о пользователе — используй её для персонализации.
Отвечай на том же языке, на котором написал пользователь."""


# ── Вспомогательные функции ──────────────────────────────────────────────────

def _build_agent(context: str) -> Agent:
    """Создаёт Haystack-агент с текущим системным промптом (включает память)."""
    system = SYSTEM_PROMPT_BASE
    if context:
        system += f"\n\nИз долговременной памяти о пользователе:\n{context}"

    return Agent(
        chat_generator=OpenAIChatGenerator(
            model=CHAT_MODEL,
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            api_base_url=os.getenv("OPENAI_BASE_URL"),
        ),
        system_prompt=system,
        tools=TOOLS,
        max_agent_steps=5,
    )


def _user_label(user: telebot.types.User) -> str:
    parts = [p for p in (user.first_name, user.last_name) if p]
    return " ".join(parts) if parts else f"user_{user.id}"


def _extract_dog_url(text: str) -> Optional[str]:
    """Ищет URL изображения dog.ceo в тексте ответа агента."""
    match = re.search(
        r"https://images\.dog\.ceo/breeds/[^\s\"'<>]+\.(?:jpg|jpeg|png)",
        text,
        re.IGNORECASE,
    )
    return match.group(0) if match else None


# ── Telegram handlers ────────────────────────────────────────────────────────

@bot.message_handler(commands=["start", "help"])
def start_handler(msg: telebot.types.Message) -> None:
    name = _user_label(msg.from_user)
    bot.reply_to(
        msg,
        f"Привет, {name}! Я твой персональный ассистент.\n\n"
        "Умею:\n"
        "🐱 Рассказывать факты о кошках\n"
        "🐶 Показывать фото собак с анализом породы\n"
        "🌤 Узнавать погоду в любом городе\n"
        "💬 Запоминать наши разговоры через Pinecone\n\n"
        "/reset — очистить историю разговора",
    )


@bot.message_handler(commands=["reset", "clear"])
def reset_handler(msg: telebot.types.Message) -> None:
    chat_histories.pop(msg.from_user.id, None)
    bot.reply_to(msg, "История очищена. Начнём заново!")


@bot.message_handler(content_types=["text"])
def message_handler(msg: telebot.types.Message) -> None:
    user_id = msg.from_user.id
    user_text = msg.text.strip()

    try:
        # 1. Получаем релевантные воспоминания из Pinecone — ТОЛЬКО этого пользователя
        user_filter = {"user_id": {"$eq": str(user_id)}}
        memories = memory.query_by_text(user_text, top_k=5, filter=user_filter)
        context = "\n".join(
            f"- {m['metadata'].get('text', '')}"
            for m in memories
            if m["metadata"].get("text")
        )
        log.info("Найдено воспоминаний: %d для: '%s'", len(memories), user_text[:50])

        # 2. История текущего разговора
        history = chat_histories.setdefault(user_id, [])

        # 3. Запускаем Haystack-агент
        agent = _build_agent(context)
        result = agent.run(messages=history + [ChatMessage.from_user(user_text)])
        response_text = (result.get("last_message") or ChatMessage.from_assistant("...")).text or ""

        # 4. Сохраняем ТОЛЬКО текст пользователя в Pinecone (дедупликация per-user)
        mem_result = memory.upsert_document(
            f"{user_id}-{uuid.uuid4().hex[:12]}",
            user_text,
            metadata={
                "user_id": str(user_id),
                "user_name": _user_label(msg.from_user),
            },
            similarity_filter=user_filter,
        )
        log.info(
            "Память: action=%s | score=%s | '%s'",
            mem_result["action"],
            f"{mem_result['similarity_score']:.3f}" if mem_result["similarity_score"] else "—",
            user_text[:60],
        )

        # 5. Обновляем краткосрочную историю
        history.append(ChatMessage.from_user(user_text))
        history.append(ChatMessage.from_assistant(response_text))
        max_msgs = SHORT_TERM_LIMIT * 2
        if len(history) > max_msgs:
            chat_histories[user_id] = history[-max_msgs:]

        # 6. Отправляем ответ
        dog_url = _extract_dog_url(response_text)
        if dog_url:
            caption = re.sub(r"https?://\S+", "", response_text).strip()[:1024]
            log.info("Отправляем фото: %s", dog_url)
            bot.send_photo(
                msg.chat.id,
                dog_url,
                caption=caption or None,
                reply_to_message_id=msg.message_id,
            )
        else:
            log.info("Отправляем текст в chat_id=%s", msg.chat.id)
            sent = bot.reply_to(msg, response_text[:4096] or "...")
            log.info("Telegram OK: message_id=%s", sent.message_id if sent else None)

    except Exception:
        log.exception("Ошибка при обработке сообщения")
        bot.reply_to(msg, "Произошла ошибка, попробуй ещё раз.")


# ── Запуск ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Бот запущен. Ожидаем сообщения...")
    bot.infinity_polling(skip_pending=True)
