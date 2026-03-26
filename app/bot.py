import os
import re
import io
import json
import math
import time
import random
import asyncio
import logging
from collections import defaultdict
from dotenv import load_dotenv

import aiohttp
from openai import OpenAI
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart, Command
from aiogram.types import BufferedInputFile
from typing import Optional, Dict, Any, List, Set, Tuple

from inventory_qa import InventoryStore, answer_inventory_question
from geo_ai import find_poi_ai
from geo_nominatim import geocode_query as nominatim_geocode
from overpass_provider import search_overpass
import sheets_logger

# chat_id -> state dict
# {
#   "kind": "address_program_collecting" | "address_program_ready",
#   "draft": "<all text merged>",
#   "created_at": float (unix timestamp),
# }
PENDING: Dict[int, Dict[str, Any]] = {}
_pending_lock = asyncio.Lock()

STATE_TTL_SECONDS = int(os.getenv("STATE_TTL_HOURS", "24")) * 3600

# Metrics: счётчики запросов и исходов
METRICS: Dict[str, int] = defaultdict(int)

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
FEEDBACK_CHANNEL_ID = os.getenv("FEEDBACK_CHANNEL_ID", "")
CALCULATOR_URL = os.getenv("CALCULATOR_URL", "https://omni360.adtech.systems/page103658706.html")

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

# ---------- Local learned facts store ----------

LEARNED_FACTS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "learned_facts.json")


def load_learned_facts() -> List[str]:
    try:
        with open(LEARNED_FACTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [str(x) for x in data if x]
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return []


def save_learned_fact(fact: str) -> None:
    facts = load_learned_facts()
    if fact not in facts:
        facts.append(fact)
    os.makedirs(os.path.dirname(LEARNED_FACTS_FILE), exist_ok=True)
    with open(LEARNED_FACTS_FILE, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)


def _build_system_with_facts() -> str:
    facts = load_learned_facts()
    if not facts:
        return SYSTEM
    facts_block = "\n".join(f"- {f}" for f in facts)
    return (
        SYSTEM
        + "\n\nВАЖНО — уточнённые факты с наивысшим приоритетом (используй именно их, "
        "не добавляй к ним общих пунктов из других источников):\n"
        + facts_block
        + "\n"
    )


# ---------- Geo / selection utilities ----------

PLAN_MAX_PLAYS_PER_HOUR = int(os.getenv("PLAN_MAX_PLAYS_PER_HOUR", "120"))
DEFAULT_RADIUS = float(os.getenv("DEFAULT_RADIUS_KM", "2.0"))

# per-chat last POI and last selection result
LAST_POI: Dict[int, List[Dict]] = {}
LAST_RESULT: Dict[int, Any] = {}


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def find_within_radius(df: Any, center: Tuple[float, float], radius_km: float) -> Any:
    import pandas as pd
    rows = []
    for _, row in df.iterrows():
        try:
            d = haversine_km(center, (float(row["lat"]), float(row["lon"])))
        except Exception:
            continue
        if d <= radius_km:
            rows.append({
                "screen_id": row.get("screen_id", row.get("GID", "")),
                "name": row.get("name", row.get("address", "")),
                "city": row.get("city", ""),
                "format": row.get("format", ""),
                "owner": row.get("owner", ""),
                "lat": row["lat"],
                "lon": row["lon"],
                "distance_km": round(d, 3),
            })
    out = pd.DataFrame(rows)
    return out.sort_values("distance_km").reset_index(drop=True) if not out.empty else out


def spread_select(df: Any, n: int, *, random_start: bool = True, seed: Optional[int] = None) -> Any:
    """Жадный k-center (Gonzalez) с рандомным стартом."""
    import pandas as pd
    if df.empty or n <= 0:
        return df.iloc[0:0]
    n = min(n, len(df))
    if seed is not None:
        random.seed(seed)
    coords = df[["lat", "lon"]].to_numpy()
    if random_start:
        start_idx = random.randrange(len(df))
    else:
        lat_med = float(df["lat"].median())
        lon_med = float(df["lon"].median())
        start_idx = min(range(len(df)), key=lambda i: haversine_km((lat_med, lon_med), (coords[i][0], coords[i][1])))
    chosen = [start_idx]
    dists = [haversine_km((coords[start_idx][0], coords[start_idx][1]), (coords[i][0], coords[i][1])) for i in range(len(df))]
    while len(chosen) < n:
        maxd = max(dists)
        candidates = [i for i, d in enumerate(dists) if d == maxd]
        next_idx = random.choice(candidates)
        chosen.append(next_idx)
        cx, cy = coords[next_idx]
        for i in range(len(df)):
            d = haversine_km((cx, cy), (coords[i][0], coords[i][1]))
            if d < dists[i]:
                dists[i] = d
    res = df.iloc[chosen].copy()
    res["min_dist_to_others_km"] = 0.0
    cc = res[["lat", "lon"]].to_numpy()
    for i in range(len(res)):
        mind = min(haversine_km((cc[i][0], cc[i][1]), (cc[j][0], cc[j][1])) for j in range(len(res)) if j != i) if len(res) > 1 else 0.0
        res.iat[i, res.columns.get_loc("min_dist_to_others_km")] = round(mind, 3)
    return res


def parse_kwargs(parts: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip().lower()] = v.strip().strip('"').strip("'")
    return out


def _format_mask(series: Any, token: str) -> Any:
    col = series.astype(str).str.upper().str.strip()
    t = token.strip().upper()
    if t in {"CITY", "CITY_FORMAT", "CITYFORMAT", "CITYLIGHT"}:
        return col.str.startswith("CITY_FORMAT")
    if t in {"BILLBOARD", "BB"}:
        return col == "BILLBOARD"
    return col == t


def _apply_inv_filters(df: Any, kwargs: Dict[str, str]) -> Any:
    import pandas as pd
    out = df.copy()
    fmt_val = kwargs.get("format") or kwargs.get("formats")
    if fmt_val and "format" in out.columns:
        tokens = [t.strip().upper() for t in re.split(r"[;,|]", str(fmt_val)) if t.strip()]
        col = out["format"].astype(str).str.upper()
        mask = None
        for f in tokens:
            m = _format_mask(col, f)
            mask = m if mask is None else (mask | m)
        if mask is not None:
            out = out[mask]
    own_val = kwargs.get("owner") or kwargs.get("owners")
    if own_val and "owner" in out.columns:
        owners = [t.strip().lower() for t in re.split(r"[;,|]", str(own_val)) if t.strip()]
        col = out["owner"].astype(str).str.lower()
        mask = None
        for o in owners:
            m = col.str.contains(re.escape(o), na=False)
            mask = m if mask is None else (mask | m)
        if mask is not None:
            out = out[mask]
    grp_min_raw = kwargs.get("grp_min")
    if grp_min_raw and "grp" in out.columns:
        try:
            grp_min = float(str(grp_min_raw).replace(",", "."))
            grp_num = pd.to_numeric(out["grp"], errors="coerce")
            out = out[grp_num.ge(grp_min).fillna(False)]
        except Exception:
            pass
    ots_min_raw = kwargs.get("ots_min")
    if ots_min_raw and "ots" in out.columns:
        try:
            ots_min = float(str(ots_min_raw).replace(",", "."))
            ots_num = pd.to_numeric(out["ots"], errors="coerce")
            out = out[ots_num.ge(ots_min).fillna(False)]
        except Exception:
            pass
    return out


def _fill_min_bid(df: Any) -> Any:
    import pandas as pd
    out = df.copy()
    src_col = next((c for c in ("minBid", "min_bid", "min_bid_rub") if c in out.columns), None)
    if src_col:
        vals = pd.to_numeric(out[src_col], errors="coerce")
        median = float(vals.median()) if not vals.dropna().empty else 0.0
        out["min_bid_used"] = vals.fillna(median)
    else:
        out["min_bid_used"] = None
    return out


def _prefer_formats(df: Any, n: int) -> Any:
    import pandas as pd
    if "format" not in df.columns or df.empty:
        return df
    col = df["format"].astype(str).str.upper()
    parts = [
        df[col == "BILLBOARD"],
        df[col == "SUPERSITE"],
        df[col.str.startswith("CITY_FORMAT")],
        df[~col.isin(["BILLBOARD", "SUPERSITE"]) & ~col.str.startswith("CITY_FORMAT")],
    ]
    pool = pd.concat(parts, ignore_index=True)
    return pool.head(max(n * 5, n))


def _as_list_any(sep_str: Optional[str]) -> List[str]:
    if not sep_str:
        return []
    s = sep_str.replace(";", ",").replace("|", ",")
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_hours_windows(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    total = 0
    for p in [x.strip() for x in s.split(",") if x.strip()]:
        if "-" in p:
            try:
                a, b = [int(x) for x in p.split("-", 1)]
                if 0 <= a <= 23 and 0 <= b <= 23:
                    total += (b - a) if b > a else (24 - a + b)
            except Exception:
                pass
    return total or None


async def _geocode_any(query: str, city: Optional[str] = None, limit: int = 5, provider: str = "nominatim") -> List[Dict]:
    """Геокодирование через один из провайдеров."""
    if provider == "openai":
        return await find_poi_ai(query=query, city=city, limit=limit)
    if provider == "overpass":
        return await search_overpass(query, city=city, limit=limit)
    # nominatim (default)
    return await nominatim_geocode(query, city=city, limit=limit)


async def _send_lines(m: types.Message, lines: List[str], header: Optional[str] = None, chunk: int = 60) -> None:
    if header:
        await m.answer(header)
    for i in range(0, len(lines), chunk):
        await m.answer("\n".join(lines[i: i + chunk]))


async def _send_gid_xlsx(m: types.Message, df: Any, filename: str = "screen_ids.xlsx", caption: str = "GID (XLSX)") -> None:
    import pandas as pd
    if df is None or df.empty:
        return
    id_col = next((c for c in ("screen_id", "GID") if c in df.columns), None)
    if id_col is None:
        return
    ids = df[id_col].dropna().astype(str).drop_duplicates().reset_index(drop=True)
    if ids.empty:
        return
    buf = io.BytesIO()
    ids.to_frame().to_excel(buf, index=False)
    buf.seek(0)
    await m.answer_document(BufferedInputFile(buf.getvalue(), filename=filename), caption=caption)


def _df_screen_id(df: Any) -> Optional[str]:
    """Возвращает имя колонки с ID экрана."""
    for c in ("screen_id", "GID"):
        if c in df.columns:
            return c
    return None


def _log_metrics() -> None:
    total = METRICS.get("total", 0)
    if not total:
        return
    resolved = METRICS.get("resolved", 0)
    escalated = METRICS.get("escalated", 0)
    pct = f"{100 * resolved // total}%" if total else "n/a"
    parts = [f"total={total}", f"resolved={resolved}({pct})", f"escalated={escalated}"]
    for key in sorted(METRICS):
        if key.startswith("type."):
            parts.append(f"{key[5:]}={METRICS[key]}")
    logging.info("METRICS | " + " | ".join(parts))


async def track(query_type: str, resolved: bool, m: Optional[types.Message] = None) -> None:
    """Трекает тип запроса и его исход, раз в 10 запросов пишет агрегат в лог."""
    METRICS["total"] += 1
    METRICS[f"type.{query_type}"] += 1
    if resolved:
        METRICS["resolved"] += 1
    else:
        METRICS["escalated"] += 1
    if METRICS["total"] % 10 == 0:
        _log_metrics()
    if m is not None and sheets_logger.is_configured():
        username = (m.from_user.username or "") if m.from_user else ""
        text = (m.text or "").strip()
        asyncio.create_task(sheets_logger.log_async(query_type, resolved, m.chat.id, username, text))


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

_FEEDBACK_RE = re.compile(
    r"(?i)(хотелось\s+бы|было\s+бы\s+(хорош|удобн|круто|класс)|"
    r"не\s+хватает|можно\s+(добавить|сделать|реализовать)|"
    r"предлагаю|а\s+можно\s+ли|планируется\s+ли|будет\s+ли\s+(функц|возможн)|"
    r"добавьте\b|сделайте\b|хочу\s+предложить|пожелани|wish|feature\s+request)"
)

# Детектор запросов на ускорение согласования креативов у операторов
_APPROVAL_ACCEL_RE = re.compile(
    r"(?i)(ускор(ить|ьте|им|яй)?\s+(согласован|модерац|провер)|"
    r"согласован\w*\s+(идёт\s+долго|затянул|задержива|не\s+идёт)|"
    r"помоч(ь|ите)?\s+ускорить|нужно\s+(срочно\s+)?согласовать|"
    r"отправила?\s+на\s+согласован|на\s+согласован\w+\s+операторам)"
)


def is_approval_acceleration(text: str) -> bool:
    return bool(_APPROVAL_ACCEL_RE.search(text or ""))


# Детектор "всё проверили, не помогло" → эскалация в КС
_ESCALATION_RE = re.compile(
    r"(?i)(все\s+(проверил|проверили|равно)|всё\s+(проверил|проверили|равно)|"
    r"по.прежнему\s+(не\s+)?(работает|идут|показ)|"
    r"ничего\s+не\s+(помогло|помогает|изменилось)|"
    r"все\s+равно\s+не|всё\s+равно\s+не|"
    r"до\s+сих\s+пор\s+не|так\s+и\s+не\s+(заработал|пошл|идут|работает)|"
    r"перезапустил|перезапустили|переделал|переделали)"
)


def is_finance_question(text: str) -> bool:
    return bool(FINANCE_RE.search(text or ""))


def is_feedback(text: str) -> bool:
    return bool(_FEEDBACK_RE.search(text or ""))


def is_exhausted_troubleshooting(text: str) -> bool:
    return bool(_ESCALATION_RE.search(text or ""))


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
    "Ты — Omnika, дружелюбный саппорт-ассистент DSP Omni360 для клиентов.\n"
    "Будь тёплой и отзывчивой. Обращайся к пользователю на «вы».\n"
    "Не копируй сухой язык документов — объясняй просто и по-человечески.\n"
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
    system = _build_system_with_facts()
    resp = client.responses.create(
        model="gpt-4o",
        input=[{"role": "system", "content": system}] + thread_messages,
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

# "с 1 по 10 апреля" — месяц только в конце
FROM_TO_SHARED_MONTH_RE = re.compile(
    rf"(?i)\bс\s*(\d{{1,2}})\s*по\s*(\d{{1,2}})\s*({MONTHS})(?:\s*(\d{{4}}))?\b"
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

    m = _last_match(FROM_TO_SHARED_MONTH_RE, low)
    if m:
        d1, d2, mon, year = m.groups()
        if year:
            return f"{int(d1)} {mon} {year} – {int(d2)} {mon} {year}"
        return f"{int(d1)} {mon} – {int(d2)} {mon}"

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

    # "все форматы" / "любые форматы" / "форматы все"
    if re.search(r"\b(все|любые|any)\b.{0,15}\bформат|\bформат.{0,15}\b(все|любые|any)\b", low):
        return "все форматы"

    found: List[str] = []

    if "билборд" in low or "billboard" in low or r"\bbb\b" in low:
        found.append("билборды")
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
        "Собрала всё! 🎉 Проверьте, пожалуйста:\n"
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

    # Если в тексте 2+ вопросительных знака — это вопросы по расчёту, а не новый бриф
    if t.count("?") >= 2:
        return False

    # Структурированный бриф: 3+ из ключевых полей = точно адресная программа,
    # даже если внутри упоминаются аналитические термины (ots, количество и т.п.)
    brief_fields = ["клиент", "период", "бюджет", "гео:", "kpi", "экраны:", "расчёт:", "расчет:"]
    brief_field_count = sum(1 for f in brief_fields if f in t)
    if brief_field_count >= 2:
        return True

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
        await m.answer("Привет! Я Omnika — помощник по DSP Omni360. Спрашивайте, разберёмся вместе 🙂")

    # ===== /near =====
    @dp.message(Command("near"))
    async def cmd_near(m: types.Message) -> None:
        df = store.df
        if df is None or df.empty:
            await m.answer("Инвентарь не загружен.")
            return
        parts = (m.text or "").strip().split()
        if len(parts) < 3:
            await m.answer("Формат: /near lat lon [radius_km]\nПример: /near 55.7143 37.5538 2")
            return
        try:
            lat, lon = float(parts[1]), float(parts[2])
            radius = float(parts[3]) if len(parts) >= 4 and "=" not in parts[3] else DEFAULT_RADIUS
        except Exception:
            await m.answer("Пример: /near 55.7143 37.5538 2")
            return
        res = find_within_radius(df, (lat, lon), radius)
        if res is None or res.empty:
            await m.answer(f"В радиусе {radius} км ничего не найдено.")
            return
        LAST_RESULT[m.chat.id] = res
        lines = [
            f"• {r.get('screen_id', r.get('GID', ''))} — {r.get('name', r.get('address', ''))} "
            f"({r.get('distance_km', '')} км) [{r.get('format', '')} / {r.get('owner', '')}]"
            for _, r in res.iterrows()
        ]
        await _send_lines(m, lines, header=f"Найдено: {len(res)} экр. в радиусе {radius} км", chunk=50)
        await _send_gid_xlsx(m, res, filename="near_screen_ids.xlsx", caption="GID (XLSX)")

    # ===== /near_geo =====
    @dp.message(Command("near_geo"))
    async def cmd_near_geo(m: types.Message) -> None:
        import pandas as pd
        df = store.df
        if df is None or df.empty:
            await m.answer("Инвентарь не загружен.")
            return
        text = (m.text or "").strip()
        tail = text.split()[1:]
        radius_km = DEFAULT_RADIUS
        start_i = 0
        if tail and "=" not in tail[0]:
            try:
                radius_km = float(tail[0].strip("[](){}"))
                start_i = 1
            except Exception:
                pass
        kv = parse_kwargs(tail[start_i:])
        dedup = str(kv.get("dedup", "1")).lower() in {"1", "true", "yes"}
        if "query" in kv:
            q = kv["query"]
            city = kv.get("city")
            limit = int(kv.get("limit", "5") or 5)
            provider = (kv.get("provider") or "nominatim").lower()
            await m.answer(f"Ищу точки «{q}»" + (f" в {city}" if city else "") + "…")
            try:
                pois = await _geocode_any(q, city=city, limit=limit, provider=provider)
            except Exception:
                pois = []
            if not pois:
                pois = await find_poi_ai(query=q, city=city, limit=limit)
            LAST_POI[m.chat.id] = pois
        pois = LAST_POI.get(m.chat.id, [])
        if not pois:
            await m.answer("Сначала найдите точки: /geo <запрос> [city=…] — или /near_geo R query=…")
            return
        await m.answer(f"Подбираю экраны в радиусе {radius_km} км вокруг {len(pois)} точек…")
        frames = []
        for p in pois:
            chunk_df = find_within_radius(df, (p["lat"], p["lon"]), radius_km)
            if chunk_df is not None and not chunk_df.empty:
                chunk_df = chunk_df.copy()
                chunk_df["poi_name"] = p.get("name", "")
                frames.append(chunk_df)
        if not frames:
            await m.answer("В выбранных радиусах экранов не нашлось.")
            return
        res = pd.concat(frames, ignore_index=True)
        filter_kv = {k: v for k, v in kv.items() if k in ("format", "owner", "city")}
        if filter_kv:
            res = _apply_inv_filters(res, filter_kv)
        if res.empty:
            await m.answer("После фильтров ничего не осталось.")
            return
        id_col = _df_screen_id(res)
        if dedup and id_col:
            res = res.drop_duplicates(subset=[id_col]).reset_index(drop=True)
        LAST_RESULT[m.chat.id] = res
        lines = [
            f"• {r.get('screen_id', r.get('GID', ''))} — {r.get('name', '')} "
            f"[{r.get('format', '')}/{r.get('owner', '')}] — {r.get('distance_km', '')} км от «{r.get('poi_name', '')}»"
            for _, r in res.head(20).iterrows()
        ]
        await _send_lines(m, lines, header=f"Найдено {len(res)} экранов рядом с {len(pois)} точками (радиус {radius_km} км)", chunk=50)
        await _send_gid_xlsx(m, res, filename="near_geo_screen_ids.xlsx", caption="GID (XLSX)")

    # ===== /geo =====
    @dp.message(Command("geo"))
    async def cmd_geo(m: types.Message) -> None:
        parts = (m.text or "").strip().split(None, 1)
        if len(parts) < 2:
            await m.answer("Формат: /geo <запрос> [city=…] [provider=nominatim|openai|overpass]\nПример: /geo ТЦ Метрополис city=Москва")
            return
        tail = parts[1]
        kv = parse_kwargs(tail.split())
        # query — всё что не key=value
        query_parts = [p for p in tail.split() if "=" not in p]
        query = " ".join(query_parts).strip()
        if not query:
            await m.answer("Укажите запрос: /geo <место> [city=…]")
            return
        city = kv.get("city")
        limit = int(kv.get("limit", "5") or 5)
        provider = (kv.get("provider") or "nominatim").lower()
        await m.answer(f"Ищу «{query}»" + (f" в {city}" if city else "") + "…")
        pois = await _geocode_any(query, city=city, limit=limit, provider=provider)
        if not pois:
            pois = await find_poi_ai(query=query, city=city, limit=limit)
        if not pois:
            await m.answer("Не нашла подходящих мест. Попробуйте другой запрос или provider=openai")
            return
        LAST_POI[m.chat.id] = pois
        lines = [f"• {p['name']} — {p['lat']:.5f}, {p['lon']:.5f} [{p.get('provider','')}]" for p in pois]
        await _send_lines(m, lines, header=f"Найдено {len(pois)} точек (сохранено для /near_geo):", chunk=50)

    # ===== /pick_city =====
    @dp.message(Command("pick_city"))
    async def cmd_pick_city(m: types.Message) -> None:
        df = store.df
        if df is None or df.empty:
            await m.answer("Инвентарь не загружен.")
            return
        parts = (m.text or "").strip().split()
        if len(parts) < 3:
            await m.answer("Формат: /pick_city Город N [format=…] [owner=…] [shuffle=1]\nПример: /pick_city Москва 20 format=BILLBOARD")
            return
        pos = [p for p in parts[1:] if "=" not in p]
        kv = parse_kwargs([p for p in parts[1:] if "=" in p])
        try:
            n = int(pos[-1])
            city = " ".join(pos[:-1])
        except Exception:
            await m.answer("Пример: /pick_city Москва 20")
            return
        if not city.strip():
            await m.answer("Нужно указать город.")
            return
        if "city" not in df.columns:
            await m.answer("В инвентаре нет столбца city.")
            return
        subset = df[df["city"].astype(str).str.strip().str.lower() == city.strip().lower()]
        if kv:
            subset = _apply_inv_filters(subset, kv)
        if subset.empty:
            await m.answer(f"Не нашла экранов в городе «{city}» с заданными фильтрами.")
            return
        shuffle_flag = str(kv.get("shuffle", "0")).lower() in {"1", "true", "yes"}
        seed = int(kv["seed"]) if str(kv.get("seed", "")).isdigit() else None
        if shuffle_flag:
            subset = subset.sample(frac=1, random_state=None).reset_index(drop=True)
        res = spread_select(subset.reset_index(drop=True), n, random_start=not str(kv.get("fixed", "0")).lower() in {"1", "true"}, seed=seed)
        LAST_RESULT[m.chat.id] = res
        await m.answer(f"Выбрано {len(res)} экранов в «{city}».")
        await _send_gid_xlsx(m, res, filename=f"pick_{city}_screen_ids.xlsx", caption=f"GID по городу «{city}» (XLSX)")

    # ===== /pick_at =====
    @dp.message(Command("pick_at"))
    async def cmd_pick_at(m: types.Message) -> None:
        df = store.df
        if df is None or df.empty:
            await m.answer("Инвентарь не загружен.")
            return
        parts = (m.text or "").strip().split()
        if len(parts) < 4:
            await m.answer("Формат: /pick_at lat lon N [radius_km] [format=…]\nПример: /pick_at 55.75 37.62 30 15 format=BILLBOARD")
            return
        try:
            lat, lon = float(parts[1]), float(parts[2])
            n = int(parts[3])
            radius = float(parts[4]) if len(parts) >= 5 and "=" not in parts[4] else 20.0
            kv = parse_kwargs(parts[5:] if len(parts) > 5 else [])
        except Exception:
            await m.answer("Пример: /pick_at 55.75 37.62 30 15 format=BILLBOARD")
            return
        circle = find_within_radius(df, (lat, lon), radius)
        if circle.empty:
            await m.answer(f"В радиусе {radius} км нет экранов.")
            return
        fmt_arg = kv.get("format")
        if fmt_arg and "format" in circle.columns:
            tokens = [t.strip() for t in re.split(r"[;,|]", fmt_arg) if t.strip()]
            if tokens:
                col = circle["format"].astype(str)
                mask = None
                for tok in tokens:
                    mk = _format_mask(col, tok)
                    mask = mk if mask is None else (mask | mk)
                if mask is not None:
                    circle = circle[mask]
        if circle.empty:
            await m.answer(f"В радиусе {radius} км нет экранов с форматом {fmt_arg!r}.")
            return
        seed = int(kv["seed"]) if str(kv.get("seed", "")).isdigit() else None
        fixed = str(kv.get("fixed", "0")).lower() in {"1", "true"}
        res = spread_select(circle.reset_index(drop=True), n, random_start=not fixed, seed=seed)
        LAST_RESULT[m.chat.id] = res
        lines = [
            f"• {r.get('screen_id', r.get('GID', ''))} — {r.get('name', '')} "
            f"[{r.get('lat', ''):.5f},{r.get('lon', ''):.5f}] [{r.get('format', '')}/{r.get('owner', '')}]"
            for _, r in res.iterrows()
        ]
        await _send_lines(m, lines, header=f"Выбрано {len(res)} экранов равномерно в радиусе {radius} км:", chunk=50)
        await _send_gid_xlsx(m, res, filename="pick_at_screen_ids.xlsx", caption="GID (XLSX)")

    # ===== /plan =====
    @dp.message(Command("plan"))
    async def cmd_plan(m: types.Message) -> None:
        import pandas as pd
        df = store.df
        if df is None or df.empty:
            await m.answer("Инвентарь не загружен.")
            return
        parts = (m.text or "").strip().split()[1:]
        kv: Dict[str, str] = {}
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                kv[k.strip().lower()] = v.strip()
        budget_raw = kv.get("budget") or kv.get("b")
        if not budget_raw:
            await m.answer(
                "Нужно указать бюджет:\n/plan budget=200000 [city=…] [format=…] [owner=…] "
                "[n=10] [days=10] [hours_per_day=8] [ots_min=…] [grp_min=…]"
            )
            return
        try:
            v = budget_raw.lower().replace(" ", "")
            if v.endswith("m"):
                budget_total = float(v[:-1]) * 1_000_000
            elif v.endswith("k"):
                budget_total = float(v[:-1]) * 1_000
            else:
                budget_total = float(v)
        except Exception:
            await m.answer("Не понял бюджет. Пример: budget=200000 или budget=200k")
            return
        city = kv.get("city")
        n = int(kv["n"]) if kv.get("n", "").isdigit() else 10
        days = int(kv["days"]) if kv.get("days", "").isdigit() else 10
        hours_per_day = int(kv["hours_per_day"]) if kv.get("hours_per_day", "").isdigit() else None
        if hours_per_day is None:
            hours_per_day = _parse_hours_windows(kv.get("hours")) or 8
        formats = _as_list_any(kv.get("format") or kv.get("formats"))
        owners = _as_list_any(kv.get("owner") or kv.get("owners"))
        want_top = str(kv.get("top", "0")).lower() in {"1", "true", "yes"}
        pool = df.copy()
        if city and "city" in pool.columns:
            pool = pool[pool["city"].astype(str).str.strip().str.lower() == city.strip().lower()]
        if pool.empty:
            await m.answer("По заданному городу нет экранов.")
            return
        filter_kv: Dict[str, str] = {}
        if formats:
            filter_kv["format"] = ",".join(formats)
        if owners:
            filter_kv["owner"] = ",".join(owners)
        for fld in ("grp_min", "ots_min"):
            if kv.get(fld):
                filter_kv[fld] = kv[fld]
        if filter_kv:
            pool = _apply_inv_filters(pool, filter_kv)
        if pool.empty:
            await m.answer("После применения фильтров экранов не осталось.")
            return
        pool = _fill_min_bid(pool)
        if not formats:
            pool = _prefer_formats(pool, n)
        if want_top and "ots" in pool.columns:
            try:
                ots_vals = pd.to_numeric(pool["ots"], errors="coerce")
                if not ots_vals.dropna().empty:
                    selected = pool.assign(_ots=ots_vals).sort_values("_ots", ascending=False).head(n).drop(columns=["_ots"])
                else:
                    raise ValueError
            except Exception:
                selected = spread_select(pool.reset_index(drop=True), min(n, len(pool)))
        else:
            selected = spread_select(pool.reset_index(drop=True), min(n, len(pool)))
        if selected.empty:
            await m.answer("Не удалось выбрать экраны.")
            return
        n = len(selected)
        budget_per_day_per_screen = budget_total / max(n, 1) / max(days, 1)
        mb = pd.to_numeric(selected.get("min_bid_used"), errors="coerce")
        median_mb = float(mb.dropna().median()) if not mb.dropna().empty else 1.0
        mb = mb.fillna(median_mb).replace(0, median_mb)
        per_day_cap = hours_per_day * PLAN_MAX_PLAYS_PER_HOUR
        slots_per_day = (budget_per_day_per_screen // mb).astype(int).clip(lower=0, upper=per_day_cap)
        total_slots = slots_per_day * days
        planned_cost = total_slots * mb
        out = selected.copy()
        out["budget_per_day"] = round(budget_per_day_per_screen, 2)
        out["min_bid_used"] = mb
        out["planned_slots_per_day"] = slots_per_day
        out["total_slots"] = total_slots
        out["planned_cost"] = planned_cost
        LAST_RESULT[m.chat.id] = selected
        caption = (
            f"План: бюджет={budget_total:,.0f} ₽, n={n}, days={days}, hours/day={hours_per_day}"
        ).replace(",", " ")
        try:
            csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
            await m.answer_document(BufferedInputFile(csv_bytes, filename="plan.csv"), caption=caption)
        except Exception as e:
            await m.answer(f"Не удалось отправить CSV: {e}")
        try:
            xbuf = io.BytesIO()
            with __import__("pandas").ExcelWriter(xbuf, engine="openpyxl") as w:
                out.to_excel(w, index=False, sheet_name="plan")
            xbuf.seek(0)
            await m.answer_document(BufferedInputFile(xbuf.getvalue(), filename="plan.xlsx"), caption="План (XLSX)")
        except Exception:
            pass
        await _send_gid_xlsx(m, selected, filename="plan_gid.xlsx", caption="GID (XLSX)")

    @dp.message(F.text.startswith("/learn"))
    async def learn(m: types.Message) -> None:
        username = (m.from_user.username or "") if m.from_user else ""
        if username not in EMPLOYEE_USERNAMES:
            return
        fact = (m.text or "").removeprefix("/learn").strip()
        if not fact:
            await m.answer("Укажи факт: /learn <текст>")
            return
        try:
            # Сохраняем локально — это даёт гарантированный приоритет в системном промпте
            await asyncio.to_thread(save_learned_fact, fact)
        except Exception:
            logging.exception("Failed to save fact locally")
            await m.answer("Не удалось сохранить факт локально. Проверь логи.")
            return
        # Также загружаем в vector store (для полноты поиска)
        try:
            content = f"Факт из базы знаний:\n{fact}".encode()
            uploaded = await asyncio.to_thread(
                lambda: client.files.create(
                    file=("fact.txt", content, "text/plain"),
                    purpose="assistants",
                )
            )
            await asyncio.to_thread(
                lambda: client.vector_stores.files.create(
                    vector_store_id=VECTOR_STORE_ID,
                    file_id=uploaded.id,
                )
            )
        except Exception:
            logging.exception("Failed to save fact to vector store (non-critical)")
        await m.answer(f"✅ Запомнила: {fact}")

    @dp.message()
    async def handle(m: types.Message) -> None:
        text = (m.text or "").strip()
        if not text:
            return

        # Сотрудники в группах — отвечаем только при явном @упоминании или реплае
        if is_employee(m) and m.chat.type in {"group", "supergroup"}:
            if not is_called_in_group(m, bot_username=bot_username, bot_id=bot_id):
                return

        # В группе (не сотрудники) отвечаем только если позвали, если флаг включен
        if not is_employee(m) and m.chat.type in {"group", "supergroup"} and REQUIRE_MENTION_IN_GROUP:
            if not is_called_in_group(m, bot_username=bot_username, bot_id=bot_id):
                return

        # In groups: молчим, если тегнули кого-то другого (кроме бота)
        if m.chat.type in {"group", "supergroup"}:
            mentions = extract_mentions(m)
            mentions.discard(bot_username)
            if mentions:
                return
            
        # ===== APPROVAL ACCELERATION =====
        if is_approval_acceleration(text):
            await track("approval_accel", resolved=True, m=m)
            await m.answer("Передадим оператору информацию — попросим ускорить согласование! 🙌")
            return

        # ===== CREATIVE ISSUE (L1 support) =====
        if is_creative_upload_issue(text):
            await track("creative", resolved=True, m=m)
            await m.answer(CREATIVE_HELP_REPLY)
            return

        # Эскалация: пользователь всё проверил, проблема осталась → зовём КС
        if is_exhausted_troubleshooting(text):
            await m.answer(
                f"Понятно, давайте подключим коллег из КС — они разберутся! {CS_TAGS} 🙌\n"
                "Пришлите, пожалуйста, ID кампании (или ссылку) и скриншоты, если есть — так будет быстрее."
            )
            return

        # Finance routing
        if is_finance_question(text):
            await track("finance", resolved=False, m=m)
            await m.answer(
                f"Похоже, вопрос про счета или оплату 💳 Подключаю {FINANCE_TAG} — они помогут!\n"
                "Если можно, пришлите номер кампании/счёта и опишите ситуацию."
            )
            return

        # Тихо логируем пожелания по доработкам в канал (бот всё равно отвечает через RAG)
        if FEEDBACK_CHANNEL_ID and is_feedback(text):
            try:
                user = m.from_user
                sender = f"@{user.username}" if user and user.username else (user.full_name if user else "неизвестно")
                chat_name = m.chat.title or "личка"
                fb_msg = (
                    f"💡 *Запрос на доработку*\n"
                    f"От: {sender} | {chat_name}\n\n"
                    f"{text}"
                )
                await bot.send_message(FEEDBACK_CHANNEL_ID, fb_msg, parse_mode="Markdown")
            except Exception:
                logging.exception("Failed to forward feedback to channel")

        pending = PENDING.get(m.chat.id)

        # TTL: сбрасываем зависший стейт после STATE_TTL_SECONDS
        if pending and time.time() - pending.get("created_at", 0) > STATE_TTL_SECONDS:
            async with _pending_lock:
                PENDING.pop(m.chat.id, None)
            pending = None
            await m.answer(
                "Предыдущий подбор адресной программы устарел и был сброшен.\n"
                "Если нужен новый подбор — напишите бриф заново 🙂"
            )

        # 1) ready state: accept OK or apply any edits
        if pending and pending.get("kind") == "address_program_ready":
            if is_confirmation(text):
                async with _pending_lock:
                    PENDING.pop(m.chat.id, None)
                await track("address_program", resolved=False, m=m)
                await m.answer(
                    "Отлично, передаю в КС — они подберут адресную программу! ✅\n"
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
                        PENDING[m.chat.id] = {"kind": "address_program_collecting", "draft": merged, "created_at": time.time()}
                    await m.answer(
                        "Принято! 👍 Осталось ещё кое-что уточнить:\n" + "\n".join(f"• {x}" for x in still_missing)
                    )
                    return

                async with _pending_lock:
                    PENDING[m.chat.id] = {"kind": "address_program_ready", "draft": merged, "created_at": time.time()}
                await m.answer(build_address_program_confirmation(merged))
                return
            # если не правка, идём дальше (inventory -> rag)

        # 2) collecting state: keep merging until all fields exist
        if pending and pending.get("kind") == "address_program_collecting":
            if should_treat_as_brief_update(text):
                draft = pending.get("draft", "")
                merged = (draft + "\n" + text).strip()
                merged = apply_geo_updates(merged, text)

                still_missing = address_program_missing_fields(merged)
                if still_missing:
                    async with _pending_lock:
                        PENDING[m.chat.id] = {"kind": "address_program_collecting", "draft": merged, "created_at": time.time()}
                    await m.answer("Спасибо, почти всё есть! Осталось уточнить:\n" + "\n".join(f"• {x}" for x in still_missing))
                    return

                async with _pending_lock:
                    PENDING[m.chat.id] = {"kind": "address_program_ready", "draft": merged, "created_at": time.time()}
                await m.answer(build_address_program_confirmation(merged))
                return
            # если не правка — идём дальше (inventory -> rag)

        # --- Inventory analytics (пропускаем для multi-question) ---
        if not has_multiple_questions(text):
            inv_reply = answer_inventory_question(text, store)
            if inv_reply:
                await track("inventory", resolved=True, m=m)
                await m.answer(inv_reply)
                return

        # 3) new address program request (пропускаем для multi-question)
        # --- New address program request — проверяем ДО inventory, иначе inventory перехватит бриф ---
        if not has_multiple_questions(text) and is_address_program_request(text):
            is_urgent = bool(re.search(r"(?i)(срочн|до\s+\d+[\-:]\d+|до\s+обед|до\s+конца\s+дня|asap)", text))
            urgent_tip = (
                f"\n\nЕсли очень срочно — можно прикинуть самостоятельно: {CALCULATOR_URL}"
                if is_urgent else ""
            )
            draft = apply_geo_updates(text, text)
            missing = address_program_missing_fields(draft)
            if missing:
                async with _pending_lock:
                    PENDING[m.chat.id] = {"kind": "address_program_collecting", "draft": draft, "created_at": time.time()}
                await m.answer(
                    "Отлично, берусь за адресную программу! 🗺️ Уточните, пожалуйста, несколько деталей:\n"
                    + "\n".join(f"• {x}" for x in missing)
                    + urgent_tip
                )
                return

            async with _pending_lock:
                PENDING[m.chat.id] = {"kind": "address_program_ready", "draft": draft, "created_at": time.time()}
            await m.answer(build_address_program_confirmation(draft) + urgent_tip)
            return

        # --- Inventory analytics (пропускаем для multi-question) ---
        if not has_multiple_questions(text):
            inv_reply = answer_inventory_question(text, store)
            if inv_reply:
                await m.answer(inv_reply)
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
            await track("rag", resolved=False, m=m)
            msg = str(e).lower()
            if "unsupported_country_region_territory" in msg or "country, region, or territory not supported" in msg:
                await m.answer(
                    f"Ой, кажется, есть проблема с сетью с моей стороны 🌐 Подключаю коллег: {SUPPORT_TAGS}"
                )
            else:
                await m.answer(f"Хм, по этому вопросу мне нужна помощь коллег 🙏 Передаю {SUPPORT_TAGS} — разберутся!")
            return

        if (not reply) or looks_like_unknown(reply):
            await track("rag", resolved=False, m=m)
            await m.answer(
                f"Вопрос немного вне моей базы, но коллеги точно помогут! {SUPPORT_TAGS} 🙌\n"
                "Чтобы разобраться быстрее — пришлите ID кампании (или ссылку) и скрины, если есть."
            )
            return

        await track("rag", resolved=True, m=m)
        await m.answer(reply)

    async def _hourly_metrics() -> None:
        """Каждый час пишет метрики в лог и чистит протухшие PENDING."""
        while True:
            await asyncio.sleep(3600)
            _log_metrics()
            now = time.time()
            expired = [cid for cid, s in list(PENDING.items()) if now - s.get("created_at", 0) > STATE_TTL_SECONDS]
            if expired:
                async with _pending_lock:
                    for cid in expired:
                        PENDING.pop(cid, None)
                logging.info("PENDING cleanup: removed %d expired states", len(expired))

    logging.info("Bot started ✅")
    asyncio.create_task(_hourly_metrics())
    await bot.set_my_commands([
        types.BotCommand(command="geo",       description="Найти место на карте: /geo ТЦ Метрополис city=Москва"),
        types.BotCommand(command="near",      description="Экраны рядом с точкой: /near 55.71 37.55 2"),
        types.BotCommand(command="near_geo",  description="Экраны рядом с результатами /geo: /near_geo 1.5"),
        types.BotCommand(command="pick_city", description="Выбрать N экранов по городу: /pick_city Москва 20"),
        types.BotCommand(command="pick_at",   description="Выбрать N экранов в радиусе: /pick_at 55.75 37.62 30 15"),
        types.BotCommand(command="plan",      description="Медиаплан по бюджету: /plan budget=500k city=Москва n=15"),
    ])
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())