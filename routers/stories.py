from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from database import operations as ops

router = APIRouter(tags=["stories"])
templates = Jinja2Templates(directory="templates")


@router.get("/stories", response_class=HTMLResponse)
def stories_page(request: Request):
    stories = ops.get_recent_stories_with_summaries()
    for story in stories:
        story["source_articles"] = ops.get_story_source_articles(story["story_id"])
    return templates.TemplateResponse(request, "stories.html", {"stories": stories})
