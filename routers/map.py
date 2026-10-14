from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from database import operations as ops

router = APIRouter(tags=["map"])
templates = Jinja2Templates(directory="templates")


@router.get("/map", response_class=HTMLResponse)
def map_page(request: Request, date: Optional[str] = None, story: Optional[int] = None):
    """One day's located stories as map markers. With no ?date, shows the most
    recent day that has any mappable story."""
    if date is None:
        latest = ops.get_latest_map_date()
        date = latest.isoformat() if latest else None
    stories = ops.get_map_stories(date=date) if date else []
    markers = [
        {
            "story_id": s["story_id"],
            "lat": s["lat"],
            "lon": s["lon"],
            "location": s["location_text"],
            "precise": bool(s["precise"]),
            "radius_km": s["radius_km"],
            "headline": s["headline"],
            "summary": s["summary_text"],
            "topics": s["topics"],
        }
        for s in stories
    ]
    return templates.TemplateResponse(
        request, "map.html", {"date": date, "markers": markers, "story_count": len(markers), "focus_story": story}
    )
