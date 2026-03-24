"""
Асинхронное логирование запросов в Google Sheets.

Настройка:
  GOOGLE_SHEETS_ID   — ID таблицы (из URL: /spreadsheets/d/<ID>/edit)
  GOOGLE_CREDENTIALS_JSON — путь к JSON-файлу сервисного аккаунта
                            ИЛИ сам JSON в виде строки (удобно для Railway/Heroku)

Сервисному аккаунту нужно выдать доступ «Редактор» к таблице.
Первый запуск автоматически создаёт заголовок на первой строке.
"""

import os
import json
import logging
import asyncio
import datetime
from typing import Optional

try:
    import gspread
    from google.oauth2.service_account import Credentials
    _GSPREAD_AVAILABLE = True
except ImportError:
    _GSPREAD_AVAILABLE = False

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
]

_SHEET_HEADERS = [
    "Дата/время",
    "Тип запроса",
    "Исход",
    "Chat ID",
    "Username",
    "Текст вопроса",
]

_worksheet: Optional[object] = None  # gspread.Worksheet
_init_done = False


def _build_worksheet() -> Optional[object]:
    sheets_id = os.getenv("GOOGLE_SHEETS_ID", "").strip()
    creds_raw = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()

    if not sheets_id or not creds_raw:
        return None

    if not _GSPREAD_AVAILABLE:
        logging.warning("sheets_logger: gspread не установлен, логирование отключено")
        return None

    try:
        # creds_raw может быть путём к файлу или JSON-строкой
        if creds_raw.startswith("{"):
            info = json.loads(creds_raw)
        else:
            with open(creds_raw) as f:
                info = json.load(f)

        creds = Credentials.from_service_account_info(info, scopes=_SCOPES)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheets_id)

        try:
            ws = sh.worksheet("Лог запросов")
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title="Лог запросов", rows=1000, cols=len(_SHEET_HEADERS))

        # Добавить заголовок если лист пустой
        if ws.row_count == 0 or not ws.row_values(1):
            ws.insert_row(_SHEET_HEADERS, index=1)

        logging.info("sheets_logger: подключено к Google Sheets (%s)", sheets_id)
        return ws

    except Exception as e:
        logging.warning("sheets_logger: не удалось подключиться — %s", e)
        return None


def _ensure_init() -> None:
    global _worksheet, _init_done
    if not _init_done:
        _worksheet = _build_worksheet()
        _init_done = True


def is_configured() -> bool:
    _ensure_init()
    return _worksheet is not None


def _append_row_sync(query_type: str, resolved: bool, chat_id: int, username: str, text: str) -> None:
    _ensure_init()
    if _worksheet is None:
        return

    outcome = "✅ решено" if resolved else "⬆️ эскалирован"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [ts, query_type, outcome, str(chat_id), username, text[:300]]

    try:
        _worksheet.append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        logging.warning("sheets_logger: ошибка записи — %s", e)


async def log_async(query_type: str, resolved: bool, chat_id: int, username: str, text: str) -> None:
    """Записывает строку в Google Sheets не блокируя event loop."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, _append_row_sync, query_type, resolved, chat_id, username, text
    )
