import calendar as calendar_module
from datetime import date as date_cls
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from database import operations as ops

router = APIRouter(tags=["stories"])
templates = Jinja2Templates(directory="templates")

PAGE_SIZE = 20


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
    """Entities a MAJORITY of sources covering this story mention (more than
    half), shown above the per-source dropdown. Not a strict "every single
    source" intersection - that was too brittle once a story had 5+ sources,
    since a genuinely central entity that NER (or the significance filter -
    see nlp.matching.filter_significant_entities) happened to miss in just
    one source's article would silently disappear here even though most
    sources agreed on it. For 1-2 sources, "more than half" still requires
    all of them, so small stories behave the same as before.

    Exact-match, case-insensitive, per category independently - NOT
    cross-source entity resolution, so "Trump" in one source and "Donald
    Trump" in another are counted as two different entities, not merged. A
    real limitation of keeping this simple, not an oversight."""
    if not source_articles:
        return {cat: [] for cat in _ENTITY_CATEGORIES}

    threshold = len(source_articles) // 2 + 1  # strict majority
    result = {}
    for cat in _ENTITY_CATEGORIES:
        counts: dict[str, int] = {}
        display_text: dict[str, str] = {}
        order: list[str] = []
        for article in source_articles:
            for text in article["entities"][cat]:
                key = text.lower()
                counts[key] = counts.get(key, 0) + 1
                if key not in display_text:
                    display_text[key] = text
                    order.append(key)
        result[cat] = [display_text[key] for key in order if counts[key] >= threshold]
    return result


def _query_string(**params) -> str:
    """Builds a `?a=b&c=d` string from whichever of these params are
    non-None, dropping the rest - used to carry the current topic/date
    filter along on pagination and calendar navigation links without every
    caller having to hand-assemble query strings."""
    clean = {k: v for k, v in params.items() if v is not None}
    return f"?{urlencode(clean)}" if clean else ""


def _build_calendar(
    year: int, month: int, story_counts: dict, selected_date: Optional[str], topic: Optional[str], q: Optional[str]
) -> dict:
    """One month's worth of day cells for the /stories page's calendar
    sidepane, plus prev/next month navigation. Days with at least one
    relevant story are links (carrying the current topic/search filters
    forward); days without are inert - there's nothing to show, and a dead
    link would just be confusing. story_counts: {date: count} from
    get_story_dates(), scoped to the same topic/search filters so the
    calendar only lights up days that actually have content for what's
    currently being viewed."""
    weeks = []
    for week in calendar_module.Calendar(firstweekday=0).monthdayscalendar(year, month):
        cells = []
        for day_num in week:
            if day_num == 0:
                cells.append(None)
                continue
            day = date_cls(year, month, day_num)
            day_str = day.isoformat()
            cells.append({
                "day_num": day_num,
                "count": story_counts.get(day, 0),
                "is_today": day == datetime.now(timezone.utc).date(),
                "is_selected": day_str == selected_date,
                "href": f"/stories{_query_string(topic=topic, q=q, date=day_str)}",
            })
        weeks.append(cells)

    prev_month = date_cls(year, month, 1)
    prev_month = date_cls(prev_month.year - 1, 12, 1) if month == 1 else date_cls(year, month - 1, 1)
    next_month = date_cls(year + 1, 1, 1) if month == 12 else date_cls(year, month + 1, 1)

    return {
        "label": date_cls(year, month, 1).strftime("%B %Y"),
        "weeks": weeks,
        "prev_href": f"/stories{_query_string(topic=topic, q=q, date=selected_date, calendar_month=prev_month.strftime('%Y-%m'))}",
        "next_href": f"/stories{_query_string(topic=topic, q=q, date=selected_date, calendar_month=next_month.strftime('%Y-%m'))}",
    }


@router.get("/stories", response_class=HTMLResponse)
def stories_page(
    request: Request,
    topic: Optional[str] = None,
    date: Optional[str] = None,
    q: Optional[str] = None,
    page: int = 1,
    calendar_month: Optional[str] = None,
):
    q = q.strip() if q else None
    q = q or None  # blank or whitespace-only search box submission == no search
    page = max(page, 1)
    stories, total_count = ops.get_recent_stories_with_summaries(
        limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE, topic=topic, date=date, q=q
    )
    for story in stories:
        # Same presentation-only capitalization as the topic filter pills -
        # matched_entities is display text here, never compared against
        # anything (the topic query param matches raw canonical_name in SQL).
        story["matched_entities"] = [_display_name(e) for e in story["matched_entities"] if e]
        story["source_articles"] = ops.get_story_source_articles(story["story_id"])
        for article in story["source_articles"]:
            article["entities"] = ops.get_article_entities(article["id"])
        story["common_entities"] = _common_entities(story["source_articles"])
        story["claim_groups"] = ops.get_story_claim_groups(story["story_id"])
        story["corroborated_count"] = sum(g["status"] == "corroborated" for g in story["claim_groups"])
    # Queried live, every request - any watchlist entity added via the API
    # shows up here (alphabetically, by canonical_name, with a live match
    # count) on the very next page load, no caching or redeploy needed.
    # Filtering/comparison still uses the raw canonical_name (`name`) -
    # `display_name` is presentation-only.
    topics = [
        {
            "name": t["canonical_name"],
            "display_name": _display_name(t["canonical_name"]),
            "count": t["story_count"],
            "href": f"/stories{_query_string(topic=t['canonical_name'], date=date, q=q)}",
        }
        for t in ops.get_topic_story_counts(date=date, q=q)
    ]
    all_topics_href = f"/stories{_query_string(date=date, q=q)}"

    if calendar_month:
        cal_year, cal_month = (int(part) for part in calendar_month.split("-"))
    elif date:
        parsed = datetime.strptime(date, "%Y-%m-%d").date()
        cal_year, cal_month = parsed.year, parsed.month
    else:
        today = datetime.now(timezone.utc).date()
        cal_year, cal_month = today.year, today.month
    story_counts = {row["day"]: row["story_count"] for row in ops.get_story_dates(topic=topic, q=q)}
    calendar_data = _build_calendar(cal_year, cal_month, story_counts, date, topic, q)

    total_pages = max(-(-total_count // PAGE_SIZE), 1)  # ceil division
    return templates.TemplateResponse(
        request,
        "stories.html",
        {
            "stories": stories,
            "topics": topics,
            "all_topics_href": all_topics_href,
            "selected_topic": topic,
            "selected_topic_display": _display_name(topic) if topic else None,
            "selected_date": date,
            "selected_query": q,
            "calendar": calendar_data,
            "page": page,
            "total_pages": total_pages,
            "prev_page_href": f"/stories{_query_string(topic=topic, date=date, q=q, page=page - 1 if page > 1 else None)}",
            "next_page_href": f"/stories{_query_string(topic=topic, date=date, q=q, page=page + 1)}",
            "clear_date_href": f"/stories{_query_string(topic=topic, q=q)}",
            "clear_search_href": f"/stories{_query_string(topic=topic, date=date)}",
        },
    )
