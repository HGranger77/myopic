from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from database import operations as ops

router = APIRouter(tags=["stories"])
templates = Jinja2Templates(directory="templates")


def _display_name(canonical_name: str) -> str:
    """Uppercases just the first character, leaving the rest untouched -
    Jinja's own capitalize/title filters both lowercase everything after the
    first letter of each word, which would mangle watchlist entities that
    are already correctly cased (acronyms like "AFL"/"NRL"/"AI", or
    multi-word names like "Computer Science"). Only turns a lowercase
    single-word topic like "politics" into "Politics"."""
    return canonical_name[:1].upper() + canonical_name[1:] if canonical_name else canonical_name


_ENTITY_CATEGORIES = ("people", "organizations", "locations")


def _common_entities(source_articles: list[dict]) -> dict:
    """Entities every source covering this story mentions, shown above the
    per-source dropdown. Intersection is exact-match, case-insensitive, per
    category independently - NOT cross-source entity resolution, so "Trump"
    in one source and "Donald Trump" in another won't be recognized as the
    same person and so won't show up as "common". A real limitation of
    keeping this simple, not an oversight."""
    if not source_articles:
        return {cat: [] for cat in _ENTITY_CATEGORIES}

    result = {}
    for cat in _ENTITY_CATEGORIES:
        common_lower = set.intersection(
            *(set(text.lower() for text in article["entities"][cat]) for article in source_articles)
        )
        ordered, seen = [], set()
        for article in source_articles:
            for text in article["entities"][cat]:
                if text.lower() in common_lower and text.lower() not in seen:
                    seen.add(text.lower())
                    ordered.append(text)
        result[cat] = ordered
    return result


@router.get("/stories", response_class=HTMLResponse)
def stories_page(request: Request, topic: Optional[str] = None):
    stories = ops.get_recent_stories_with_summaries(topic=topic)
    for story in stories:
        # Same presentation-only capitalization as the topic filter pills -
        # matched_entities is display text here, never compared against
        # anything (the topic query param matches raw canonical_name in SQL).
        story["matched_entities"] = [_display_name(e) for e in story["matched_entities"] if e]
        story["source_articles"] = ops.get_story_source_articles(story["story_id"])
        for article in story["source_articles"]:
            article["entities"] = ops.get_article_entities(article["id"])
        story["common_entities"] = _common_entities(story["source_articles"])
    # Queried live, every request - any watchlist entity added via the API
    # shows up here (alphabetically, by canonical_name, with a live match
    # count) on the very next page load, no caching or redeploy needed.
    # Filtering/comparison still uses the raw canonical_name (`name`) -
    # `display_name` is presentation-only.
    topics = [
        {"name": t["canonical_name"], "display_name": _display_name(t["canonical_name"]), "count": t["story_count"]}
        for t in ops.get_topic_story_counts()
    ]
    return templates.TemplateResponse(
        request,
        "stories.html",
        {
            "stories": stories,
            "topics": topics,
            "selected_topic": topic,
            "selected_topic_display": _display_name(topic) if topic else None,
        },
    )
