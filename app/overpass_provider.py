# overpass_provider.py
import aiohttp
from typing import List, Dict, Optional, Tuple

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]
UA = "OmnikaBot/1.0 (DSP Omni360 support bot)"

CITY_BBOX: dict[str, Tuple[float, float, float, float]] = {
    "москва": (37.2, 55.4, 37.95, 56.05),
    "moscow": (37.2, 55.4, 37.95, 56.05),
    "мск": (37.2, 55.4, 37.95, 56.05),
    "санкт-петербург": (29.4, 59.65, 31.0, 60.3),
    "санкт петербург": (29.4, 59.65, 31.0, 60.3),
    "спб": (29.4, 59.65, 31.0, 60.3),
    "питер": (29.4, 59.65, 31.0, 60.3),
    "химки": (37.30, 55.84, 37.57, 56.02),
    "воронеж": (39.0, 51.5, 39.5, 51.9),
    "казань": (48.9, 55.65, 49.3, 55.95),
}


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _bbox_for_city(city: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    return CITY_BBOX.get(_norm(city))


def _is_category(query: str) -> bool:
    q = _norm(query)
    return any(x in q for x in [
        "аптека", "стадион", "тц", "торгов", "школ", "аэропорт", "вокзал",
        "университет", "кино", "театр", "больниц", "поликлин", "парковка",
        "бизнес-центр", "бц", "новострой", "жк", "мфц", "парк",
    ])


def _tag_filter(query: str) -> str:
    q = _norm(query)
    if "аптека" in q:
        return '["amenity"="pharmacy"]'
    if "стадион" in q:
        return '["leisure"="stadium"]'
    if "тц" in q or "торгов" in q:
        return '["shop"="mall"]'
    if "школ" in q:
        return '["amenity"="school"]'
    return f'["name"~"{query}", i]'


async def search_overpass(query: str, city: Optional[str] = None, limit: int = 10) -> List[Dict]:
    bbox = _bbox_for_city(city)
    if not bbox:
        return []
    min_lon, min_lat, max_lon, max_lat = bbox
    tag = _tag_filter(query)
    ql = f"""
    [out:json][timeout:30];
    (
      node{tag}({min_lat},{min_lon},{max_lat},{max_lon});
      way{tag}({min_lat},{min_lon},{max_lat},{max_lon});
      relation{tag}({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out center {min(int(limit or 10), 50)};
    """
    headers = {"User-Agent": UA}
    for url in OVERPASS_URLS:
        try:
            async with aiohttp.ClientSession(
                headers=headers, timeout=aiohttp.ClientTimeout(total=40)
            ) as sess:
                async with sess.post(url, data=ql.encode("utf-8")) as r:
                    if r.status != 200:
                        _ = await r.text()
                        continue
                    data = await r.json()
            elements = data.get("elements", [])
            out: List[Dict] = []
            for el in elements:
                if "lat" in el and "lon" in el:
                    lat, lon = el["lat"], el["lon"]
                elif "center" in el and el["center"]:
                    lat, lon = el["center"].get("lat"), el["center"].get("lon")
                else:
                    continue
                name = (el.get("tags", {}) or {}).get("name") or query
                out.append({"name": name, "address": name, "lat": lat, "lon": lon, "provider": "overpass"})
            if out:
                return out[: min(len(out), limit or 10)]
        except Exception:
            continue
    return []
