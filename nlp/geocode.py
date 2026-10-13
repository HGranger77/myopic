"""Place name -> (lat, lon) via OpenStreetMap's Nominatim.

Nominatim's usage policy asks for an identifying User-Agent and at most one
request per second, so lookups are throttled and every result (including
"not found") is cached in Postgres by the caller - see
database.operations.get_cached_geocode/set_cached_geocode. Override the
endpoint with MYOPIC_GEOCODER_URL to point at a self-hosted instance."""
import os
import time
from typing import Optional

import requests

GEOCODER_URL = os.environ.get("MYOPIC_GEOCODER_URL", "https://nominatim.openstreetmap.org/search")
_USER_AGENT = "myopic-news-pipeline/1.0 (https://github.com/HGranger77/myopic)"
_MIN_INTERVAL_SECONDS = 1.0
_last_request = 0.0


def geocode(place: str) -> Optional[dict]:
    """Returns {lat, lon, display_name} for the best match, or None if the
    place can't be resolved. Raises on network errors (the caller decides
    whether that's fatal - it must NOT be cached as "not found")."""
    global _last_request
    wait = _MIN_INTERVAL_SECONDS - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    response = requests.get(
        GEOCODER_URL,
        params={"q": place, "format": "jsonv2", "limit": 1},
        headers={"User-Agent": _USER_AGENT},
        timeout=15,
    )
    _last_request = time.monotonic()
    response.raise_for_status()
    results = response.json()
    if not results:
        return None
    top = results[0]
    return {"lat": float(top["lat"]), "lon": float(top["lon"]), "display_name": top.get("display_name")}
