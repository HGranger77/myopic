"""Place name -> (lat, lon) via OpenStreetMap's Nominatim.

Nominatim's usage policy asks for an identifying User-Agent and at most one
request per second, so lookups are throttled and every result (including
"not found") is cached in Postgres by the caller - see
database.operations.get_cached_geocode/set_cached_geocode. Override the
endpoint with MYOPIC_GEOCODER_URL to point at a self-hosted instance."""
import math
import os
import re
import time
from typing import Optional

import requests

GEOCODER_URL = os.environ.get("MYOPIC_GEOCODER_URL", "https://nominatim.openstreetmap.org/search")
_USER_AGENT = "myopic-news-pipeline/1.0 (https://github.com/HGranger77/myopic)"
_MIN_INTERVAL_SECONDS = 1.0
_last_request = 0.0


# Nominatim's place_rank: 4 country, 8 state, 12 county, 16 city, 18-19
# suburb/neighbourhood; 20+ is streets, shops, parks and other POIs - which is
# what a mislabelled non-place ("Kangaroos", "Sporting") resolves to, so
# anything at 20+ is rejected rather than pinned.
MAX_PLACE_RANK = 19

def plausible_place(name: str) -> bool:
    """Cheap pre-geocode screen: drops NER fragments that aren't words (\"n't\", \"'s\")."""
    name = name.strip()
    return (
        len(name) >= 3
        and re.search(r"[A-Za-z]{3}", name) is not None
        and not name.startswith(("'", "\u2019"))
    )


def _radius_km(bbox) -> float:
    south, north, west, east = (float(v) for v in bbox)
    height = (north - south) * 111.0
    width = (east - west) * 111.0 * math.cos(math.radians((north + south) / 2))
    return round(max(height, abs(width)) / 2, 1)


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
        params={"q": place, "format": "jsonv2", "limit": 5},
        headers={"User-Agent": _USER_AGENT},
        timeout=15,
    )
    _last_request = time.monotonic()
    response.raise_for_status()
    for top in response.json():
        is_place = top["category"] == "place" or (
            top["category"] == "boundary" and top["type"] == "administrative"
        )
        if is_place and top["place_rank"] <= MAX_PLACE_RANK:
            return {
                "lat": float(top["lat"]), "lon": float(top["lon"]),
                "display_name": top.get("display_name"),
                "place_rank": top["place_rank"], "radius_km": _radius_km(top["boundingbox"]),
            }
    return None
