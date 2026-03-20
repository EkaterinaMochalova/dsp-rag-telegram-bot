# app/inventory_qa.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import pandas as pd

# ==========
# Canonical formats (for stable filtering/grouping)
# ==========

FORMAT_SYNONYMS: Dict[str, List[str]] = {
    "billboard": [
        "билборд", "билборды", "билбордам", "billboard", "bb", "6x3", "3x6", "щит", "щиты", "board",
    ],
    "cityformat": [
        "ситиформат", "ситиформаты", "cityformat", "city format",
    ],
    "sitiboard": [
        "ситиборд", "ситиборды", "sitiboard", "cityboard", "city board",
    ],
    "mediafacade": [
        "медиафасад", "медиафасады", "mediafacade", "media facade", "фасад", "фасады",
    ],
    "indoor": [
        "индор", "indoor", "в помещении", "внутри помещения", "внутри",
    ],
    "outdoor": [
        "аутдор", "outdoor", "на улице", "наруж", "наружка", "street", "roadside",
    ],
}

FORMAT_CANON_TITLES: Dict[str, str] = {
    "billboard": "билборды",
    "cityformat": "ситиформаты",
    "sitiboard": "ситиборды",
    "mediafacade": "медиафасады",
    "indoor": "indoor",
    "outdoor": "outdoor",
}

def _learn_format_synonyms_from_df(df: pd.DataFrame) -> int:
    """Самообучение по данным CSV.

    Идея: если в инвентаре встречаются новые написания форматов (например,
    "City-Form", "CITY FORM", "Сити-формат"), мы добавляем их в синонимы,
    чтобы фильтры/детекторы начинали их понимать без ручных правок.

    Сейчас добавляем только эвристику для cityformat: формат содержит 'city' и 'form*'
    (или 'сити' и 'форм*').
    """
    if "format" not in df.columns:
        return 0

    added = 0
    # работаем с уникальными значениями, пропускаем пустые
    unique_formats = [str(x) for x in df["format"].dropna().unique().tolist()]
    if not unique_formats:
        return 0

    # текущий набор (в нижнем регистре) для дедупликации
    cur = set(s.lower() for s in FORMAT_SYNONYMS.get("cityformat", []) if s)

    for raw in unique_formats:
        r = raw.strip()
        if not r:
            continue
        t = _norm_text(r)

        # cityformat эвристика
        if (("city" in t) and ("form" in t)) or (("сити" in t) and ("форм" in t)):
            # добавляем оригинал + нормализованный вариант (с пробелами)
            for cand in {r.lower(), t}:
                if cand and cand not in cur:
                    FORMAT_SYNONYMS.setdefault("cityformat", []).append(cand)
                    cur.add(cand)
                    added += 1

    return added


def _detect_format_from_text(q: str) -> Optional[str]:
    t = (q or "").lower()
    for canon, syns in FORMAT_SYNONYMS.items():
        for s in syns:
            if s and s in t:
                return canon
    return None

def _normalize_format_value(v: str) -> str:
    """Приводим значения из столбца format к каноническим форматам.

    Важно: сначала нормализуем текст (пунктуация/пробелы), чтобы ловить
    варианты вида "City-Format", "Сити формат", "CITYFORMAT" и т.п.
    """
    raw = (v or "").strip()
    if not raw:
        return ""

    t = _norm_text(raw)  # "city-format" -> "city format", "сити-формат" -> "сити формат"

    # billboard / board
    if ("bill" in t) or (t in {"bb", "6x3", "3x6"}) or ("билборд" in t) or ("щит" in t) or ("board" in t and "city" not in t):
        return "billboard"

    # cityformat (ловим как англ, так и русские варианты)
    if (
        "cityformat" in t
        or "city format" in t
        or ("city" in t and ("format" in t or "form" in t))
        or "ситиформ" in t
        or ("сити" in t and "формат" in t)
    ):
        return "cityformat"

    # sitiboard / cityboard
    if "sitiboard" in t or "cityboard" in t or "city board" in t or "ситиб" in t:
        return "sitiboard"

    # mediafacade
    if "mediafacade" in t or "media facade" in t or "медиафас" in t or "фасад" in t:
        return "mediafacade"

    # indoor / outdoor
    if "indoor" in t or "индор" in t:
        return "indoor"
    if "outdoor" in t or "аутдор" in t or "roadside" in t or "street" in t or "наруж" in t:
        return "outdoor"

    return raw.lower()


# ==========
# Config
# ==========

DEFAULT_CSV_PATH = os.getenv("INVENTORY_CSV_PATH", "data/inventories.csv")
TOP_N_DEFAULT = 10
MIN_GROUP_SIZE_DEFAULT = 3


# ==========
# Helpers: normalization
# ==========

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[,\.;:!?\(\)\[\]\{\}<>\"'`]+")
_DASHES_RE = re.compile(r"[–—−]")

def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = _DASHES_RE.sub("-", s)
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace("\u00A0", " ", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace("—", "", regex=False)
        .str.replace("-", "", regex=False),
        errors="coerce",
    )

def _pretty_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")

def _pretty_float(x: float, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    fmt = f"{{:.{digits}f}}"
    return fmt.format(float(x)).replace(".", ",")

def _pretty_money(x: float) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{int(round(float(x))):,}".replace(",", " ")


# ==========
# Inventory store (in-memory)
# ==========

@dataclass
class InventoryStore:
    df: pd.DataFrame
    csv_path: str

    @staticmethod
    def load(csv_path: str = DEFAULT_CSV_PATH) -> "InventoryStore":
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Inventory CSV not found: '{csv_path}'. "
                f"Set INVENTORY_CSV_PATH in .env or place the file at that path."
            )
        df = pd.read_csv(csv_path)
        df = _prepare_df(df)
        return InventoryStore(df=df, csv_path=csv_path)

    def reload(self) -> None:
        df = pd.read_csv(self.csv_path)
        self.df = _prepare_df(df)


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for col in ["city", "format", "placement", "installation", "owner", "address"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").map(lambda x: x.strip())
        else:
            df[col] = ""

    _learn_format_synonyms_from_df(df)
    df["_format_canon"] = df["format"].map(_normalize_format_value)
    df["_format_canon_norm"] = df["_format_canon"].map(_norm_text)

    df["_city_norm"] = df["city"].map(_norm_text)
    df["_format_norm"] = df["format"].map(_norm_text)
    df["_owner_norm"] = df["owner"].map(_norm_text)
    df["_placement_norm"] = df["placement"].map(_norm_text)
    df["_installation_norm"] = df["installation"].map(_norm_text)

    for col in ["ots", "minBid", "grp", "lat", "lon", "width_mm", "height_mm", "width_px", "height_px"]:
        if col in df.columns:
            df[col] = _to_num(df[col])

    if "GID" in df.columns:
        df["GID"] = df["GID"].astype(str)
    else:
        df["GID"] = ""

    return df


# ==========
# Query parsing
# ==========

_METRIC_ALIASES = {
    "ots": ["ots", "отс", "ots’", "ots.", "oтс"],
    "minBid": ["minbid", "минставк", "минимальн", "минимальная ставка", "минималка", "ставка", "цена", "bid"],
    "grp": ["grp", "джиарп", "грп", "grps"],
}

_AGG_ALIASES = {
    "avg": ["средн", "average", "avg", "в среднем"],
    "median": ["медиан", "median"],
    "min": ["миним", "min"],
    "max": ["максим", "max"],
    "sum": ["сумм", "итого", "sum"],
    "count": ["сколько", "количество", "count", "шт", "штук", "экранов"],
}

# Слова "минимальное", "количество" и т.п. слишком общие.
# Для любого агрегата требуем хотя бы один инвентарный сигнал,
# чтобы не роутить в inventory фразы вроде "минимальное количество выходов как в МП".
_INVENTORY_SIGNALS = [
    # groupby-измерения
    "по город", "по гео", "по регион", "по формат", "по тип",
    "по оператор", "по подрядчик", "по владельц", "по размещен",
    # инвентарные сущности
    "экран", "поверхност", "конструкц", "оператор", "подрядчик",
    "формат", "регион", "город", "гео", "площадк",
    # метрики инвентаря
    "ставк", "ots", "отс", "grp", "grps", "bid",
    # форматы наружной рекламы
    "билборд", "billboard", "ситиформ", "ситиборд", "медиафасад",
    "суперсайт", "призматрон", "пиллар", "скролл", "лайтбокс",
    "тумб", "брандмауэр", "indoor", "outdoor", "индор", "аутдор",
]

_GROUPBY_ALIASES = {
    "city": ["по город", "по гео", "по региона", "по населен", "разрез город", "в разрезе город"],
    "format": ["по формат", "по тип", "разрез формат", "в разрезе формат"],
    "owner": ["по оператор", "по подрядчик", "по владельц", "по owner", "разрез оператор"],
    "placement": ["по размещен", "по улице/помещ", "по indoor/outdoor", "разрез размещен"],
    "installation": ["по установк", "по движ", "moving", "static", "разрез установк"],
}

_PLACEMENT_HINTS = {
    "outdoor": ["outdoor", "аутдор", "улиц", "на улице", "roadside", "street"],
    "indoor": ["indoor", "индор", "в помещ", "внутри", "помещен", "mall"],
}

_STOP_WORDS = {
    "сколько", "экранов", "в", "во", "на", "по", "для", "и", "или", "а", "с", "со",
    "средний", "средняя", "среднее", "средние", "средн", "миним", "максим", "медиан",
    "ots", "отс", "grp", "грп", "ставка", "minbid", "бюджет", "формат", "оператор", "подрядчик",
    "город", "гео", "разрез", "топ",
    "билборд", "билборды", "ситиформат", "ситиформаты", "ситиборд", "ситиборды",
    "медиафасад", "медиафасады", "индор", "аутдор", "outdoor", "indoor",
}

_TOP_RE = re.compile(r"(?i)\bтоп\s*(\d{1,3})\b")

# более узкий матч: "в <городе>" до следующего "по/на/для/с" или конца
_IN_CITY_RE = re.compile(r"(?i)\b(?:в|во)\s+([а-яёa-z0-9\- ]{2,40}?)(?=\s+(?:по|на|для|с|со)\b|[?.!,]|$)")

# ==========
# RU city heuristics (cases)
# ==========

def _normalize_ru_city_token(token: str) -> str:
    """
    Пытаемся привести "казани" -> "казань", "москве" -> "москва", "петербурге" -> "петербург".
    Без внешних библиотек, только эвристики.
    """
    t = _norm_text(token)

    # частые спец-кейсы
    if t in {"спб", "питер"}:
        return "санкт-петербург"
    if t in {"петербурге", "петербурга", "петербургу", "петербургом"}:
        return "петербург"
    if t.startswith("санкт петербург") or t.startswith("санкт-петербург"):
        return "санкт-петербург"

    # аккуратно снимаем типичные окончания (очень грубо, но помогает)
    for suf, repl in [
        ("е", "а"),     # москвЕ -> москвА (потом матчится как "москва")
        ("у", "а"),
        ("ой", "а"),
        ("ою", "а"),
        ("ом", "а"),
        ("ам", "а"),
        ("ах", "а"),
        ("ы", "а"),
        ("и", ""),      # казанИ -> казан (потом ниже)
        ("а", ""),      # краснодара -> краснодр (не всегда), поэтому ниже есть guard
        ("я", ""),      # тюмени/тюмень - отдельно не решим идеально
    ]:
        if len(t) >= 5 and t.endswith(suf):
            cand = t[:-len(suf)] + repl
            # не ломаем слова совсем в ноль
            if len(cand) >= 4:
                t = cand
            break

    # если получилось "казан" -> "казань" (частый кейс)
    if t.endswith("казан"):
        return "казань"

    return t

def _guess_city_from_phrase(phrase: str, city_vocab_norm: List[str]) -> Optional[str]:
    """
    phrase: то, что поймали после "в/во"
    """
    p = (phrase or "").strip()
    if not p:
        return None

    guess_norm = _normalize_ru_city_token(p)

    # точные/contains/startswith по словарю
    best_norm = _best_match(guess_norm, city_vocab_norm)
    if best_norm:
        return best_norm

    # второй шанс: отрезаем 1-2 буквы (падежи типа "казани", "перми", "твери")
    for cut in (1, 2):
        if len(guess_norm) - cut >= 4:
            best_norm = _best_match(guess_norm[:-cut], city_vocab_norm)
            if best_norm:
                return best_norm

    return None


@dataclass
class ParsedQuery:
    agg: str
    metric: Optional[str]
    groupby: Optional[str]
    top_n: int
    min_group_size: int
    filters: Dict[str, Any]
    raw: str


def is_inventory_question(text: str, store: InventoryStore) -> bool:
    return parse_query(text, store) is not None


def parse_query(text: str, store: InventoryStore) -> Optional[ParsedQuery]:
    t_raw = text or ""
    t = _norm_text(t_raw)
    if not t:
        return None

    # 1) agg — используем \b чтобы "несколько" не матчило "сколько"
    agg = None
    for k, aliases in _AGG_ALIASES.items():
        if any(re.search(r"\b" + re.escape(a), t) for a in aliases):
            agg = k
            break
    if agg is None:
        return None

    # guard: общие слова ("минимальное", "количество" и т.п.) не должны
    # роутить в inventory без явного инвентарного контекста
    if not any(sig in t for sig in _INVENTORY_SIGNALS):
        return None

    # 2) metric — тоже \b чтобы не было ложных совпадений
    metric = None
    if agg != "count":
        for m, aliases in _METRIC_ALIASES.items():
            if any(re.search(r"\b" + re.escape(a), t) for a in aliases):
                metric = m
                break
        if metric is None and agg in ("avg", "median", "min", "max"):
            metric = "ots"

    # 3) groupby
    groupby = None
    for g, aliases in _GROUPBY_ALIASES.items():
        if any(a in t for a in aliases):
            groupby = g
            break

    # 4) top N
    top_n = TOP_N_DEFAULT
    m_top = _TOP_RE.search(t_raw)
    if m_top:
        try:
            top_n = max(1, min(50, int(m_top.group(1))))
        except Exception:
            top_n = TOP_N_DEFAULT

    # 5) filters
    filters: Dict[str, Any] = {}

    for pl_key, hints in _PLACEMENT_HINTS.items():
        if any(h in t for h in hints):
            filters["placement"] = pl_key
            break

    # format canon
    canon = _detect_format_from_text(t_raw)
    if canon:
        filters["format_canon"] = canon
    else:
        fmt_canon = _match_value_from_vocab(t, store.df.get("_format_canon_norm"), store.df.get("_format_canon"))
        if fmt_canon:
            filters["format_canon"] = fmt_canon

    # owner
    own = _match_value_from_vocab(t, store.df.get("_owner_norm"), store.df.get("owner"))
    if own:
        filters["owner"] = own

    # city: first try vocab substring/fuzzy
    city = _match_value_from_vocab(t, store.df.get("_city_norm"), store.df.get("city"))
    if city:
        filters["city"] = city
    else:
        # fallback: "в казани" / "в краснодаре"
        m_city = _IN_CITY_RE.search(t_raw)
        if m_city:
            phrase = m_city.group(1)
            city_vocab = store.df["_city_norm"].dropna().unique().tolist()
            best_norm = _guess_city_from_phrase(phrase, city_vocab)
            if best_norm:
                filters["city"] = _denorm_from_norm(best_norm, store.df, "_city_norm", "city")

    return ParsedQuery(
        agg=agg,
        metric=metric,
        groupby=groupby,
        top_n=top_n,
        min_group_size=MIN_GROUP_SIZE_DEFAULT,
        filters=filters,
        raw=t_raw,
    )


def _match_value_from_vocab(
    query_norm: str,
    vocab_norm_series: Optional[pd.Series],
    orig_series: Optional[pd.Series],
) -> Optional[str]:
    if vocab_norm_series is None or orig_series is None:
        return None

    vocab = vocab_norm_series.dropna().unique().tolist()

    hits = [v for v in vocab if v and len(v) >= 3
            and re.search(r"\b" + re.escape(v) + r"\b", query_norm)]
    if hits:
        best = max(hits, key=len)
        tmp = pd.DataFrame({"n": vocab_norm_series, "o": orig_series})
        return _denorm_from_norm(best, tmp, "n", "o")

    tokens = [w for w in query_norm.split() if len(w) >= 4 and w not in _STOP_WORDS]
    for tok in tokens:
        best = _best_match(tok, vocab)
        if best:
            tmp = pd.DataFrame({"n": vocab_norm_series, "o": orig_series})
            return _denorm_from_norm(best, tmp, "n", "o")

    return None


def _best_match(token: str, vocab: List[str]) -> Optional[str]:
    token = token.strip()
    if not token or len(token) < 3:
        return None

    for v in vocab:
        if v == token:
            return v
    for v in vocab:
        if token in v:
            return v
    for v in vocab:
        if v.startswith(token):
            return v

    return None


def _denorm_from_norm(norm_val: str, df: pd.DataFrame, norm_col: str, orig_col: str) -> Optional[str]:
    try:
        hit = df.loc[df[norm_col] == norm_val, orig_col]
        if len(hit) > 0:
            return hit.value_counts().idxmax()
    except Exception:
        pass
    return None


# ==========
# Execution
# ==========

def answer_inventory_question(text: str, store: InventoryStore) -> Optional[str]:
    pq = parse_query(text, store)
    if not pq:
        return None

    df_f = _apply_filters(store.df, pq.filters)
    if df_f is None or len(df_f) == 0:
        return "Не нашла подходящих экранов по заданным условиям. Проверьте гео/формат/оператора."

    if pq.groupby:
        return _answer_grouped(df_f, pq)
    return _answer_single(df_f, pq)


def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    out = df

    if "city" in filters and "city" in out.columns:
        out = out[out["city"].astype(str) == str(filters["city"])]

    if "owner" in filters and "owner" in out.columns:
        out = out[out["owner"].astype(str) == str(filters["owner"])]

    if "format_canon" in filters and "_format_canon" in out.columns:
        out = out[out["_format_canon"].astype(str) == str(filters["format_canon"])]

    if "placement" in filters and "_placement_norm" in out.columns:
        pl = filters["placement"]
        if pl == "outdoor":
            out = out[out["_placement_norm"].str.contains(r"улиц|outdoor|street|road", regex=True, na=False)]
        elif pl == "indoor":
            out = out[out["_placement_norm"].str.contains(r"помещ|indoor|inside|mall|office", regex=True, na=False)]

    return out


def _answer_single(df: pd.DataFrame, pq: ParsedQuery) -> str:
    n = len(df)

    if pq.agg == "count":
        parts = [f"Нашла экранов: {_pretty_int(n)}"]
        _append_filters_summary(parts, pq.filters)
        return "\n".join(parts)

    metric = pq.metric or "ots"
    if metric not in df.columns:
        return f"В файле нет поля {metric}, не могу посчитать."

    s = df[metric].dropna()
    if len(s) == 0:
        return "По выбранным условиям нет значений метрики, нечего усреднять."

    value = _aggregate_series(s, pq.agg)
    parts = [f"{_agg_title(pq.agg)} {_metric_title(metric)}: {_format_metric(metric, value)} (по {_pretty_int(n)} экранам)"]
    _append_filters_summary(parts, pq.filters)
    return "\n".join(parts)


def _answer_grouped(df: pd.DataFrame, pq: ParsedQuery) -> str:
    groupby = pq.groupby

    if groupby == "format":
        group_col = "_format_canon"
        if group_col not in df.columns:
            return "В файле нет форматов, не могу сделать разрез."
    else:
        group_col = groupby
        if group_col not in df.columns:
            return f"В файле нет поля {group_col}, не могу сделать разрез."

    if pq.agg == "count":
        g = df.groupby(group_col, dropna=False).size().reset_index(name="count")
        g = g[g["count"] >= pq.min_group_size].sort_values("count", ascending=False).head(pq.top_n)
        if len(g) == 0:
            return "Группы получились слишком маленькие. Могу показать без порога, если нужно."
        lines = [f"Топ-{len(g)} по {_group_title(groupby)} (количество экранов):"]
        for _, row in g.iterrows():
            label = _pretty_group_value(groupby, row[group_col])
            lines.append(f"• {label}: {_pretty_int(int(row['count']))}")
        _append_filters_summary(lines, pq.filters)
        return "\n".join(lines)

    metric = pq.metric or "ots"
    if metric not in df.columns:
        return f"В файле нет поля {metric}, не могу посчитать."

    g = (
        df.groupby(group_col, dropna=False)
        .agg(
            n=("GID", "count") if "GID" in df.columns else (metric, "size"),
            metric=(metric, lambda s: _aggregate_series(pd.Series(s).dropna(), pq.agg)),
        )
        .reset_index()
    )

    g = g[g["n"] >= pq.min_group_size]
    if len(g) == 0:
        return "Группы получились слишком маленькие. Могу посчитать без порога MIN_GROUP_SIZE, если нужно."

    g = g.sort_values("metric", ascending=False).head(pq.top_n)

    title = f"{_agg_title(pq.agg)} {_metric_title(metric)}"
    lines = [f"Топ-{len(g)} по {_group_title(groupby)}: {title}"]
    for _, row in g.iterrows():
        label = _pretty_group_value(groupby, row[group_col])
        lines.append(f"• {label}: {_format_metric(metric, row['metric'])} (n={_pretty_int(int(row['n']))})")

    _append_filters_summary(lines, pq.filters)
    return "\n".join(lines)


def _aggregate_series(s: pd.Series, agg: str) -> float:
    if len(s) == 0:
        return float("nan")
    if agg == "avg":
        return float(s.mean())
    if agg == "median":
        return float(s.median())
    if agg == "min":
        return float(s.min())
    if agg == "max":
        return float(s.max())
    if agg == "sum":
        return float(s.sum())
    return float(s.mean())


def _metric_title(metric: str) -> str:
    return {
        "ots": "OTS",
        "minBid": "минимальная ставка",
        "grp": "GRP",
    }.get(metric, metric)


def _group_title(groupby: str) -> str:
    return {
        "city": "городам",
        "format": "форматам",
        "owner": "операторам",
        "placement": "размещению",
        "installation": "установке",
    }.get(groupby, groupby)


def _pretty_group_value(groupby: str, v: Any) -> str:
    if groupby == "format":
        key = (str(v) if v is not None else "").strip().lower()
        return FORMAT_CANON_TITLES.get(key, str(v))
    return str(v)


def _agg_title(agg: str) -> str:
    return {
        "avg": "Среднее",
        "median": "Медиана",
        "min": "Минимум",
        "max": "Максимум",
        "sum": "Сумма",
        "count": "Количество",
    }.get(agg, agg)


def _format_metric(metric: str, value: float) -> str:
    if metric == "minBid":
        return _pretty_money(value)
    if metric in ("ots", "grp"):
        return _pretty_float(value, digits=2)
    return _pretty_float(value, digits=2)


def _append_filters_summary(lines: List[str], filters: Dict[str, Any]) -> None:
    if not filters:
        return

    bits: List[str] = []
    if "city" in filters:
        bits.append(f"гео: {filters['city']}")
    if "format_canon" in filters:
        fmt = str(filters["format_canon"]).strip().lower()
        bits.append(f"формат: {FORMAT_CANON_TITLES.get(fmt, fmt)}")
    if "owner" in filters:
        bits.append(f"оператор: {filters['owner']}")
    if "placement" in filters:
        bits.append(f"размещение: {filters['placement']}")

    if bits:
        lines.append(f"Условия: {', '.join(bits)}")
