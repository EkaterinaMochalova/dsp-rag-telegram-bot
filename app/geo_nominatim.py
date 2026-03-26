# geo_nominatim.py
import aiohttp
from typing import List, Dict, Optional

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_UA = "OmnikaBot/1.0 (DSP Omni360 support bot)"


async def geocode_query(query: str, city: Optional[str] = None, limit: int = 5) -> List[Dict]:
    if not query:
        return []
    q = query if not city else f"{query}, {city}"
    params = {
        "q": q,
        "format": "jsonv2",
        "addressdetails": 1,
        "limit": min(max(int(limit or 5), 1), 50),
        "accept-language": "ru,en",
    }
    async with aiohttp.ClientSession(
        headers={"User-Agent": NOMINATIM_UA},
        timeout=aiohttp.ClientTimeout(total=20),
    ) as sess:
        async with sess.get(NOMINATIM_URL, params=params) as r:
            if r.status != 200:
                _ = await r.text()
                return []
            data = await r.json()
    out: List[Dict] = []
    for it in data or []:
        try:
            lat = float(it.get("lat", 0))
            lon = float(it.get("lon", 0))
        except Exception:
            continue
        name = it.get("display_name") or it.get("name") or q
        out.append({"name": name, "address": name, "lat": lat, "lon": lon, "provider": "nominatim"})
    return out[: params["limit"]]
