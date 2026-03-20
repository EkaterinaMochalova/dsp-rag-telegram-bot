import os
import re
import asyncio
import logging
from dotenv import load_dotenv

from openai import OpenAI
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from typing import Optional, Dict, Any, List, Set, Tuple

from inventory_qa import InventoryStore, answer_inventory_question

# chat_id -> state dict
# {
#   "kind": "address_program_collecting" | "address_program_ready",
#   "draft": "<all text merged>",
# }
PENDING: Dict[int, Dict[str, Any]] = {}
_pending_lock = asyncio.Lock()

load_dotenv()
logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")
REQUIRE_MENTION_IN_GROUP = os.getenv("REQUIRE_MENTION_IN_GROUP", "0") == "1"

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing in .env")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing in .env (get it from @BotFather)")
if not VECTOR_STORE_ID:
    raise RuntimeError("VECTOR_STORE_ID is missing in .env (run scripts/index_kb.py and paste it)")

client = OpenAI(api_key=OPENAI_API_KEY)

SUPPORT_TAGS = os.getenv("SUPPORT_TAGS", "")
FINANCE_TAG = os.getenv("FINANCE_TAG", "")
CS_TAGS = os.getenv("CS_TAGS", "")

EMPLOYEE_USERNAMES = set(filter(None, os.getenv("EMPLOYEE_USERNAMES", "").split(",")))
CREATIVE_HELP_REPLY = """Проверьте, пожалуйста, несколько моментов:

• формат файла — должен быть JPG или MP4  
• разрешение / размеры креатива  
• размер файла  

Если всё верно, но креатив не загружается, нам поможет:

• скрин окна загрузки (даже если нет ошибки)  
• ссылка на креатив и на кампанию, для которой вы его загружаете  
• сам файл, который загружаете  

Тогда быстрее разберёмся, в чём проблема 🙏"""


def is_employee(m: types.Message) -> bool:
    u = m.from_user
    if not u:
        return False
    username = (u.username or "").lstrip("@").lower()
    return username in EMPLOYEE_USERNAMES


# ---------- General routing ----------

FINANCE_RE = re.compile(
    r"(?i)\b(счет|сч[её]т|инвойс|invoice|оплат|оплата|плат[её]ж|платеж|"
    r"выстав(ить|ьте)\s+сч[её]т|провест(и|ите)\s+оплат|"
    r"акт|закрывающ(ие|ие документы)|договор|сверк[аи]|"
    r"возврат|refund|баланс|биллинг|billing)\b"
)


def is_finance_question(text: str) -> bool:
    return bool(FINANCE_RE.search(text or ""))


# короткие подтверждения клиента
CONFIRM_RE = re.compile(
    r"(?i)^\s*(ок|okay|ok|хорошо|спасибо|благодарю|подтверждаю|да,?\s*всё\s*ок|верно|супер|отлично)\s*[!.]*\s*$"
)

# Сообщения с ≥2 пронумерованными вопросами ("1. ...\n2. ...") роутим напрямую в RAG
_MULTI_Q_RE = re.compile(r"(?m)^\s*\d+[\.\)]\s+\S")


def is_confirmation(text: str) -> bool:
    return bool(CONFIRM_RE.match(text or ""))


def has_multiple_questions(text: str) -> bool:
    """Возвращает True если сообщение содержит ≥2 пронумерованных вопроса."""
    return len(_MULTI_Q_RE.findall(text or "")) >= 2


ADDRESS_PROGRAM_RE = re.compile(
    r"(?i)\b("
    r"адресн(?:ая|ую)?\s+программ(?:а|у|ы)?|"
    r"адреска|адреску|"
    r"подбор\s+адресн|подберите\s+адресн|"
    r"собери(?:те)?\s+адресн|"
    r"медиа\s*план|медиаплан|"
    r"бриф"
    r")\b"
)


# ---------- RAG (knowledge base answers) ----------

SYSTEM = (
    "Ты — саппорт-ассистент DSP для клиентов.\n"
    "Отвечай на основе результатов file_search и фактов из этого промпта.\n"
    "Если ни в базе знаний, ни в промпте нет ответа — НЕ выдумывай: задай 1–2 уточняющих вопроса или эскалируй.\n\n"
    "Стиль ответа:\n"
    "- Коротко и по делу.\n"
    "- 1–2 предложения + при необходимости 2–4 пункта.\n"
    "- Можно использовать списки.\n"
    "- Не используй символы **, __, * для форматирования и не применяй Markdown.\n"
    "- Без слова «Источник» и без названий документов.\n"
    "- Можно 1–2 уместных эмоджи.\n"
    "- Если пользователь задаёт несколько пронумерованных вопросов — отвечай на каждый "
    "по очереди, сохраняя ту же нумерацию.\n\n"
    "Если проблема техническая (карта не грузится / пустой экран / загрузка крутится), "
    "сначала предложи проверить: VPN, фаервол/AdBlock, кэш/инкогнито, другую сеть. "
    "Только потом переходи к настройкам доступа.\n\n"
    "Маркировка рекламы (ОРД/ERIR): в наружной рекламе (OOH/DOOH) маркировка "
    "рекламных материалов НЕ требуется — это требование распространяется только на "
    "интернет-рекламу. Операторы наружной рекламы не обязаны маркировать размещения, "
    "маркировка РМ остаётся на стороне рекламодателя/агентства только для digital-каналов.\n"
)


def is_called_in_group(m: types.Message, bot_username: str, bot_id: int) -> bool:
    # 1) реплай на сообщение бота
    if m.reply_to_message and m.reply_to_message.from_user and m.reply_to_message.from_user.id == bot_id:
        return True

    # 2) явный @mention бота в тексте
    if m.entities and (m.text or ""):
        text = m.text
        for ent in m.entities:
            if ent.type == "mention":
                raw = text[ent.offset: ent.offset + ent.length]
                if raw.lower() == f"@{bot_username}".lower():
                    return True
    return False


def clean_formatting(text: str) -> str:
    return (text or "").replace("**", "").replace("__", "").strip()


def msg_text(msg: types.Message) -> str:
    return (msg.text or msg.caption or "").strip()


def extract_mentions(m: types.Message) -> Set[str]:
    mentions: Set[str] = set()
    text = m.text or ""
    if not m.entities:
        return mentions
    for ent in m.entities:
        if ent.type == "mention":
            raw = text[ent.offset: ent.offset + ent.length]
            if raw.startswith("@") and len(raw) > 1:
                mentions.add(raw[1:].lower())
    return mentions


def build_thread_messages(m: types.Message, bot_id: int, max_depth: int = 6) -> List[Dict[str, str]]:
    chain: List[types.Message] = []
    cur = m
    depth = 0
    while cur and depth < max_depth:
        chain.append(cur)
        cur = cur.reply_to_message
        depth += 1

    chain.reverse()

    out: List[Dict[str, str]] = []
    for msg in chain:
        t = msg_text(msg)
        if not t:
            continue
        u = msg.from_user
        username = (u.username or "").lstrip("@").lower() if u else ""
        # не включаем сотрудников в контекст
        if username in EMPLOYEE_USERNAMES:
            continue
        role = "assistant" if (u and u.id == bot_id) else "user"
        out.append({"role": role, "content": t})
    return out


def ask_rag(thread_messages: List[Dict[str, str]]) -> str:
    resp = client.responses.create(
        model="gpt-4o",
        input=[{"role": "system", "content": SYSTEM}] + thread_messages,
        tools=[{"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]}],
    )
    return clean_formatting(resp.output_text or "")


def looks_like_unknown(reply: str) -> bool:
    low = (reply or "").lower()
    triggers = [
        "не наш", "нет информации", "не вижу", "недостаточно", "уточнит",
        "не могу ответить", "нет данных", "не удалось"
    ]
    return any(t in low for t in triggers)


# ---------- Address Program parsing + updates ----------

# Budget: supports "10 000 000", "10 млн", "10 миллионов", etc.
BUDGET_ANY_RE = re.compile(
    r"(?is)\b(\d{1,3}(?:[ \u00A0]\d{3})+|\d+)\s*(млн|миллион(?:ов|а)?|млрд|тыс)?\s*(руб\.?|₽|rur|rub)?\b"
)


_NBSP_RE = re.compile(r"[ \u00A0]")


def _normalize_amount(num_str: str, scale: Optional[str]) -> int:
    n = int(_NBSP_RE.sub("", num_str))
    if not scale:
        return n
    s = scale.lower()
    if "млн" in s or "миллион" in s:
        return n * 1_000_000
    if "млрд" in s:
        return n * 1_000_000_000
    if "тыс" in s:
        return n * 1_000
    return n


def is_creative_upload_issue(text: str) -> bool:
    t = (text or "").lower()

    triggers = [
        "не грузится креатив",
        "не загружается креатив",
        "не могу загрузить креатив",
        "креатив не грузится",
        "крео не грузится",
        "не прикрепляется файл",
        "не загружается файл",
        "ошибка загрузки креатива",
    ]

    return any(x in t for x in triggers)

def extract_budget(text: str) -> Optional[str]:
    matches = BUDGET_ANY_RE.findall(text or "")
    if not matches:
        return None
    # берём последнее число, которое похоже на бюджет (>= 100k)
    for num_str, scale, _ in reversed(matches):
        amount = _normalize_amount(num_str, scale)
        if amount >= 100_000:
            return f"{amount:,}".replace(",", " ") + " руб."
    return None


# Period: "1 марта – 20 марта" OR "с 1 марта по 20 марта"
MONTHS = (
    r"январ[ьяе]|феврал[ьяе]|март[ае]?|апрел[ьяе]|ма[йя]|июн[ьяе]|июл[ьяе]|"
    r"август[ае]?|сентябр[ьяе]|октябр[ьяе]|ноябр[ьяе]|декабр[ьяе]"
)

FROM_TO_MONTH_RE = re.compile(
    rf"(?i)\bс\s*(\d{{1,2}})\s*({MONTHS})(?:\s*(\d{{4}}))?\s*по\s*(\d{{1,2}})\s*({MONTHS})(?:\s*(\d{{4}}))?\b"
)

MONTH_RANGE_RE = re.compile(
    rf"(?i)\b(\d{{1,2}})\s*({MONTHS})(?:\s*(\d{{4}}))?\s*(?:-|–|—)\s*(\d{{1,2}})\s*({MONTHS})?(?:\s*(\d{{4}}))?\b"
)


def _last_match(pattern: re.Pattern, text: str) -> Optional[re.Match]:
    last = None
    for m in pattern.finditer(text):
        last = m
    return last


def extract_period(text: str) -> Optional[str]:
    low = (text or "").lower()

    m = _last_match(FROM_TO_MONTH_RE, low)
    if m:
        d1, mon1, y1, d2, mon2, y2 = m.groups()
        year = y2 or y1
        if year:
            return f"{int(d1)} {mon1} {year} – {int(d2)} {mon2} {year}"
        return f"{int(d1)} {mon1} – {int(d2)} {mon2}"

    m = _last_match(MONTH_RANGE_RE, low)
    if m:
        d1, mon1, y1, d2, mon2, y2 = m.groups()
        mon2 = mon2 or mon1
        year = y2 or y1
        if year:
            return f"{int(d1)} {mon1} {year} – {int(d2)} {mon2} {year}"
        return f"{int(d1)} {mon1} – {int(d2)} {mon2}"

    return None


# Schedule: supports "с 9 до 6 ежедневно" and "09:00–20:00"
TIME_RANGE_HHMM_RE = re.compile(r"(?i)\b(\d{1,2}):(\d{2})\s*(?:-|–|—)\s*(\d{1,2}):(\d{2})\b")
TIME_RANGE_RE = re.compile(r"(?i)\bс\s*(\d{1,2})(?::(\d{2}))?\s*(до|-|—)\s*(\d{1,2})(?::(\d{2}))?\b")


_247_RE = re.compile(r"(?i)\b24/7\b")
_DIGIT_RE = re.compile(r"\d")


def has_schedule(text: str) -> bool:
    t = text or ""
    return bool(TIME_RANGE_HHMM_RE.search(t) or TIME_RANGE_RE.search(t) or _247_RE.search(t))


def normalize_schedule(text: str) -> Optional[str]:
    t = (text or "").lower()
    if "24/7" in t:
        return "24/7"

    is_daily = ("ежеднев" in t) or ("каждый день" in t)

    m2 = _last_match(TIME_RANGE_HHMM_RE, t)
    if m2:
        h1, mm1, h2, mm2 = map(int, m2.groups())
        prefix = "ежедневно" if is_daily else "график"
        return f"{prefix} {h1:02d}:{mm1:02d}–{h2:02d}:{mm2:02d}"

    m = _last_match(TIME_RANGE_RE, t)
    if not m:
        if is_daily:
            return "ежедневно"
        return None

    h1 = int(m.group(1))
    mm1 = int(m.group(2) or 0)
    h2 = int(m.group(4))
    mm2 = int(m.group(5) or 0)

    # эвристика: "до 6" в ежедневном окне почти всегда 18:00
    if is_daily and h2 <= 7:
        h2 += 12

    prefix = "ежедневно" if is_daily else "график"
    return f"{prefix} {h1:02d}:{mm1:02d}–{h2:02d}:{mm2:02d}"


# Formats (как у тебя было)
def extract_formats(text: str) -> Optional[str]:
    low = (text or "").lower()
    found: List[str] = []

    if "ситиформ" in low:
        found.append("ситиформаты")
    if "ситиборд" in low or "ситибодр" in low:
        found.append("ситиборды")
    if "медиафасад" in low:
        found.append("медиафасады")
    if "индор" in low:
        found.append("indoor")
    if "аутдор" in low or "outdoor" in low:
        found.append("outdoor")

    places: List[str] = []
    if "транспорт" in low:
        places.append("транспорт")
    if "мфц" in low:
        places.append("МФЦ")
    if "пвз" in low:
        places.append("ПВЗ")
    if "почт" in low:
        places.append("почта")

    if not found and not places:
        return None

    base = ", ".join(found) if found else "форматы из брифа"
    if places:
        base += f" (в т.ч. {', '.join(places)})"
    return base


# Extras: любые доп условия
EXTRA_HINTS_RE = re.compile(
    r"(?i)\b(таргет|аудитор|погод|weather|30\+|dmp|сегмент|retarget|ретаргет|"
    r"по\s+погод|температур|осадк|пробк|traffic|аффинит|интерес)\b"
)


def extract_extras(text: str) -> Optional[str]:
    low = (text or "").lower()
    extras: List[str] = []

    if "погод" in low or "weather" in low:
        extras.append("таргетинг по погоде")
    if "30+" in low or "30 +" in low:
        extras.append("аудитория 30+")
    if "пробк" in low or "traffic" in low:
        extras.append("таргетинг по пробкам")
    if "dmp" in low or "сегмент" in low or "аудитор" in low:
        extras.append("аудиторный таргетинг")

    if not extras and EXTRA_HINTS_RE.search(low):
        return "указано в комментарии"

    if not extras:
        return None

    uniq: List[str] = []
    for x in extras:
        if x not in uniq:
            uniq.append(x)
    return ", ".join(uniq)


# Geo: supports any free text + add/set commands.
GEO_LINE_RE = re.compile(r"(?is)\b(?:гео|география|города|регионы)\s*[:\-]\s*([^\n]+)")
GEO_ADD_RE = re.compile(r"(?is)\bдобав(ь|ьте)\s+в\s+гео\s+([^\n]+)")
GEO_SET_RE = re.compile(r"(?is)\bгео\s+(?:теперь|только)\s+([^\n]+)")


_GEO_SPLIT_RE = re.compile(r"[;,]|(?:\s+и\s+)")
_GEO_STRIP_RE = re.compile(r"(?is)\b(гео|география|города|регионы)\s*[:\-]\s*[^\n]+\n?")


def _split_geo_items(s: str) -> List[str]:
    parts = _GEO_SPLIT_RE.split(s)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def _merge_geo(existing: Optional[str], add_items: List[str], mode: str) -> str:
    if mode == "set" or not existing:
        base = []
    else:
        base = _split_geo_items(existing)

    seen = {x.lower(): x for x in base}
    for it in add_items:
        key = it.lower()
        if key not in seen:
            seen[key] = it
    return ", ".join(seen.values())


def extract_geo(text: str) -> Optional[str]:
    lines = GEO_LINE_RE.findall(text or "")
    if lines:
        return lines[-1].strip()
    return None


def apply_geo_updates(draft: str, new_text: str) -> str:
    current_geo = extract_geo(draft)

    m_add = GEO_ADD_RE.search(new_text or "")
    if m_add:
        items = _split_geo_items(m_add.group(2))
        current_geo = _merge_geo(current_geo, items, mode="add")

    m_set = GEO_SET_RE.search(new_text or "")
    if m_set:
        items = _split_geo_items(m_set.group(1))
        current_geo = _merge_geo(current_geo, items, mode="set")

    m_line = GEO_LINE_RE.search(new_text or "")
    if m_line:
        items = _split_geo_items(m_line.group(1))
        current_geo = _merge_geo(None, items, mode="set")

    if not current_geo:
        return draft

    cleaned = _GEO_STRIP_RE.sub("", draft).strip()
    return (cleaned + "\n" + f"Гео: {current_geo}").strip()


# Missing fields
def address_program_missing_fields(text: str) -> List[str]:
    missing: List[str] = []
    if extract_formats(text) is None:
        missing.append("форматы (можно “все”)")
    if extract_period(text) is None:
        missing.append("период размещения (даты)")
    if not has_schedule(text):
        missing.append("график (дни недели/часы или “24/7”)")
    if extract_budget(text) is None:
        missing.append("бюджет (с НДС/без НДС)")
    if extract_geo(text) is None:
        missing.append("гео (город/регионы)")
    return missing


def build_address_program_confirmation(text: str) -> str:
    geo = extract_geo(text) or "не указан"
    budget = extract_budget(text) or "не указан"
    formats = extract_formats(text) or "не указаны"
    period = extract_period(text) or "не указан (нужны даты)"
    schedule = normalize_schedule(text) or "не указан"
    extras = extract_extras(text)

    low = (text or "").lower()
    nds = "с НДС" if "ндс" in low else "не указано про НДС"
    commission = "с комиссией" if "комисс" in low else "не указано про комиссию"

    extra_line = f"• Дополнительно: {extras}\n" if extras else ""

    return (
        "Приняла запрос на адресную программу. Проверьте, пожалуйста, всё ли верно:\n"
        f"• Гео: {geo}\n"
        f"• Бюджет: {budget} ({nds}, {commission})\n"
        f"• Форматы: {formats}\n"
        f"• Период: {period}\n"
        f"• График: {schedule}\n"
        f"{extra_line}\n"
        "Если всё ок, ответьте “ок” и я передам в КС ✅"
    )


def should_treat_as_brief_update(text: str) -> bool:
    t = (text or "").strip()
    low = t.lower()
    if not t:
        return False

    # Явный вопрос по системе (длинный + '?') лучше отдать RAG
    if "?" in t and len(t) > 40:
        return False

    markers = [
        "ой", "апд", "update",
        "бюдж", "млн", "миллион", "₽", "руб",
        "гео", "казан", "моск", "спб", "город", "регион",
        "период", "дата", "с ", "по ",
        "график", "время", "ежеднев", "24/7",
        "формат", "ситиформ", "ситиборд", "медиафасад", "индор", "аутдор",
        "таргет", "аудитор", "погод", "30+", "dmp", "сегмент", "пробк", "traffic",
    ]
    if any(k in low for k in markers):
        return True

    # короткое сообщение с цифрами обычно правка
    if len(t) <= 160 and _DIGIT_RE.search(t):
        return True

    return False



def is_address_program_request(text: str) -> bool:
    """
    Возвращает True ТОЛЬКО если пользователь явно просит собрать адресную программу/подбор экранов.
    Важно: не триггерим адреску на аналитические вопросы типа "средняя ставка/OTS/топ по городам".
    """
    t = (text or "").lower()

    # явные маркеры адрески
    address_markers = [
        "адресн", "адреска", "адресную программу", "адресная программа", "адресную", "адреса",
        "подбор", "подбери", "подберите", "подобрать", "собери программу", "собрать программу",
        "список экранов", "список адресов", "подбор экранов", "подбор адресов", "Планируется",
        "планируем", "размещение",
        "где размест", "куда размест", "разместить где", "локаци", "точки", "экраны в радиусе",
    ]

    # маркеры аналитики (если есть, это почти наверняка НЕ адреска)
    analytics_markers = [
        "средн", "медиан", "миним", "максим", "сколько", "количество", "топ",
        "ots", "отс", "grp", "грп", "ставк", "minbid", "цена показа", "средняя цена",
        "по билборд", "по ситиформ", "по ситиборд", "по формат", "по город", "по оператор",
    ]

    has_address = any(m in t for m in address_markers)
    has_analytics = any(m in t for m in analytics_markers)

    # если вопрос аналитический, не уводим в адреску
    if has_analytics and not ("адрес" in t or "подбор" in t or "подбери" in t):
        return False

    return has_address and not has_analytics

async def main() -> None:
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    me = await bot.get_me()
    bot_username = (me.username or "").lower()
    bot_id = me.id

    # inventory store: грузим один раз и используем в хендлере
    store = InventoryStore.load()

    dp = Dispatcher()

    @dp.message(CommandStart())
    async def start(m: types.Message) -> None:
        await m.answer("Привет! Я помогу по DSP. Задайте вопрос 🙂")

    @dp.message()
    async def handle(m: types.Message) -> None:
        text = (m.text or "").strip()
        if not text:
            return

        # ignore employees
        if is_employee(m):
            return

        # В группе отвечаем только если позвали (тегом/реплаем), если флаг включен
        if m.chat.type in {"group", "supergroup"} and REQUIRE_MENTION_IN_GROUP:
            if not is_called_in_group(m, bot_username=bot_username, bot_id=bot_id):
                return

        # In groups: молчим, если тегнули кого-то другого (кроме бота)
        if m.chat.type in {"group", "supergroup"}:
            mentions = extract_mentions(m)
            mentions.discard(bot_username)
            if mentions:
                return
            
        # ===== CREATIVE ISSUE (L1 support) =====
        if is_creative_upload_issue(text):
            await m.answer(CREATIVE_HELP_REPLY)
            return

        # Finance routing
        if is_finance_question(text):
            await m.answer(
                f"Похоже на вопрос по счетам/оплате 💳 Подключаю {FINANCE_TAG}.\n"
                "Если можно, пришлите номер кампании/счёта и что нужно сделать."
            )
            return

        pending = PENDING.get(m.chat.id)

        # 1) ready state: accept OK or apply any edits
        if pending and pending.get("kind") == "address_program_ready":
            if is_confirmation(text):
                async with _pending_lock:
                    PENDING.pop(m.chat.id, None)
                await m.answer(
                    "Отлично, передаю в КС на подбор адресной программы ✅\n"
                    f"{CS_TAGS}\n\n"
                    "Если появятся правки, просто напишите их в этот чат."
                )
                return

            if should_treat_as_brief_update(text):
                draft = pending.get("draft", "")

                merged = (draft + "\n" + text).strip()
                merged = apply_geo_updates(merged, text)

                still_missing = address_program_missing_fields(merged)
                if still_missing:
                    async with _pending_lock:
                        PENDING[m.chat.id] = {"kind": "address_program_collecting", "draft": merged}
                    await m.answer(
                        "Приняла правку 👍 Нужно уточнить ещё:\n" + "\n".join(f"• {x}" for x in still_missing)
                    )
                    return

                async with _pending_lock:
                    PENDING[m.chat.id] = {"kind": "address_program_ready", "draft": merged}
                await m.answer(build_address_program_confirmation(merged))
                return
            # если не правка, идём дальше (inventory -> rag)

        # 2) collecting state: keep merging until all fields exist
        if pending and pending.get("kind") == "address_program_collecting":
            draft = pending.get("draft", "")
            merged = (draft + "\n" + text).strip()
            merged = apply_geo_updates(merged, text)

            still_missing = address_program_missing_fields(merged)
            if still_missing:
                async with _pending_lock:
                    PENDING[m.chat.id] = {"kind": "address_program_collecting", "draft": merged}
                await m.answer("Спасибо! Осталось уточнить:\n" + "\n".join(f"• {x}" for x in still_missing))
                return

            async with _pending_lock:
                PENDING[m.chat.id] = {"kind": "address_program_ready", "draft": merged}
            await m.answer(build_address_program_confirmation(merged))
            return

        # --- Inventory analytics (пропускаем для multi-question) ---
        if not has_multiple_questions(text):
            inv_reply = answer_inventory_question(text, store)
            if inv_reply:
                await m.answer(inv_reply)
                return

        

# 3) new address program request (пропускаем для multi-question)
        if not has_multiple_questions(text) and is_address_program_request(text):
            draft = apply_geo_updates(text, text)
            missing = address_program_missing_fields(draft)
            if missing:
                async with _pending_lock:
                    PENDING[m.chat.id] = {"kind": "address_program_collecting", "draft": draft}
                await m.answer(
                    "Чтобы собрать адресную программу, уточните, пожалуйста:\n"
                    + "\n".join(f"• {x}" for x in missing)
                )
                return

            async with _pending_lock:
                PENDING[m.chat.id] = {"kind": "address_program_ready", "draft": draft}
            await m.answer(build_address_program_confirmation(draft))
            return

        # 4) default: RAG answer
        await bot.send_chat_action(m.chat.id, "typing")
        try:
            thread_messages = build_thread_messages(m, bot_id, max_depth=6)
            if not thread_messages:
                thread_messages = [{"role": "user", "content": text}]
            reply = await asyncio.to_thread(ask_rag, thread_messages)
        except Exception as e:
            logging.exception("OpenAI call failed")
            msg = str(e).lower()
            if "unsupported_country_region_territory" in msg or "country, region, or territory not supported" in msg:
                await m.answer(
                    "Сейчас не могу обратиться к базе знаний из-за ограничения по сети/региону 🌐\n"
                    f"Подключаю саппорт: {SUPPORT_TAGS}"
                )
            else:
                await m.answer(f"Не получилось проверить базу знаний. Подключаю саппорт: {SUPPORT_TAGS} 🙏")
            return

        if (not reply) or looks_like_unknown(reply):
            await m.answer(
                f"Уточню и скоро вернусь 🙏 {SUPPORT_TAGS}, помогите, пожалуйста.\n"
                "Пришлите, пожалуйста, ID кампании (или ссылку на неё) и скрины, если есть возможность."
            )
            return

        await m.answer(reply)

    logging.info("Bot started ✅")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())