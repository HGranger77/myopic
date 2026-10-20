"""Read-only replay of story clustering: old rules vs new rules on real articles.

Re-clusters every embedded article already in the DB, in discovery order,
under both rule sets and prints how they differ. Writes nothing.

    python scripts/replay_clustering.py                    # summary + diffs
    python scripts/replay_clustering.py --pair 12 345      # distance between two articles
    python scripts/replay_clustering.py --nearest 0.2 --centroid 0.3 --hours 12

Reads the DB named by the usual POSTGRES_* env vars / .env.
"""
import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class Article:
    id: int
    source: str
    headline: str
    discovered_at: datetime
    embedding: np.ndarray  # unit-normalised


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cluster_old(articles: list[Article], window_days: float = 14, max_distance: float = 0.3) -> dict[int, int]:
    """Previous rule: nearest member of any story touched in the last
    window_days, joined if within max_distance. Returns {article_id: story_no}."""
    members: dict[int, list[Article]] = {}
    assignment: dict[int, int] = {}
    for art in sorted(articles, key=lambda a: a.discovered_at):
        best_story, best_dist = None, None
        for story_no, arts in members.items():
            if arts[-1].discovered_at < art.discovered_at - timedelta(days=window_days):
                continue  # story not touched within the window
            d = min(cosine_distance(art.embedding, m.embedding) for m in arts)
            if best_dist is None or d < best_dist:
                best_story, best_dist = story_no, d
        if best_story is None or best_dist > max_distance:
            best_story = len(members)
            members[best_story] = []
        members[best_story].append(art)
        assignment[art.id] = best_story
    return assignment


def cluster_new(
    articles: list[Article], window_hours: float = 24, max_nearest: float = 0.5, max_centroid: float = 0.55
) -> dict[int, int]:
    """Current rule (mirrors ops.find_similar_story): only members within
    +/- window_hours count; story needs nearest <= max_nearest AND
    centroid <= max_centroid; best = smallest sum."""
    members: dict[int, list[Article]] = {}
    assignment: dict[int, int] = {}
    window = timedelta(hours=window_hours)
    for art in sorted(articles, key=lambda a: a.discovered_at):
        best_story, best_score = None, None
        for story_no, arts in members.items():
            near = [m for m in arts if abs(m.discovered_at - art.discovered_at) <= window]
            if not near:
                continue
            nearest = min(cosine_distance(art.embedding, m.embedding) for m in near)
            centroid = cosine_distance(art.embedding, np.mean([m.embedding for m in near], axis=0))
            if nearest > max_nearest or centroid > max_centroid:
                continue
            if best_score is None or nearest + centroid < best_score:
                best_story, best_score = story_no, nearest + centroid
        if best_story is None:
            best_story = len(members)
            members[best_story] = []
        members[best_story].append(art)
        assignment[art.id] = best_story
    return assignment


def group(assignment: dict[int, int]) -> list[frozenset[int]]:
    stories = defaultdict(set)
    for article_id, story_no in assignment.items():
        stories[story_no].add(article_id)
    return [frozenset(s) for s in stories.values()]


def describe(label: str, assignment: dict[int, int]) -> str:
    sizes = [len(s) for s in group(assignment)]
    singles = sum(1 for n in sizes if n == 1)
    return f"{label}: {len(sizes)} stories, {singles} single-article, largest {max(sizes, default=0)}"


def diff_stories(old: dict[int, int], new: dict[int, int], by_id: dict[int, Article]) -> list[str]:
    """Old stories that the new rules split up."""
    lines = []
    for story in sorted(group(old), key=len, reverse=True):
        parts = defaultdict(list)
        for article_id in story:
            parts[new[article_id]].append(article_id)
        if len(parts) > 1:
            lines.append(f"old story of {len(story)} articles split into {len(parts)}:")
            for ids in sorted(parts.values(), key=len, reverse=True):
                for article_id in sorted(ids, key=lambda i: by_id[i].discovered_at):
                    a = by_id[article_id]
                    lines.append(f"    [{a.discovered_at:%m-%d %H:%M}] #{a.id} {a.source}: {a.headline[:90]}")
                lines.append("    --")
    return lines


def _to_array(embedding) -> np.ndarray:
    # newer pgvector returns a Vector wrapper, older versions a numpy array
    raw = embedding.to_numpy() if hasattr(embedding, "to_numpy") else embedding
    return np.asarray(raw, dtype=float)


def _parse_ts(text: str) -> datetime:
    # Postgres text timestamps ("...:52.26341+00") - Python 3.10's fromisoformat
    # wants 6 fractional digits and a "+HH:MM" offset
    m = re.fullmatch(r"(.*?)(?:\.(\d+))?([+-]\d\d)(?::?(\d\d))?", text)
    base, frac, hh, mm = m.groups()
    return datetime.fromisoformat(f"{base}.{(frac or '0').ljust(6, '0')}{hh}:{mm or '00'}")


def load_articles_csv(path: str) -> list[Article]:
    """Columns: id, source, headline, discovered_at, embedding (as '[0.1,...]')."""
    import csv
    import json

    with open(path, newline="") as f:
        return [
            Article(int(r[0]), r[1], r[2], _parse_ts(r[3]), np.asarray(json.loads(r[4]), dtype=float))
            for r in csv.reader(f)
        ]


def load_articles() -> list[Article]:
    from dotenv import load_dotenv

    load_dotenv()
    from database import get_connection

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, source, headline, discovered_at, embedding FROM articles "
                "WHERE embedding IS NOT NULL ORDER BY discovered_at"
            )
            return [
                Article(r["id"], r["source"], r["headline"], r["discovered_at"], _to_array(r["embedding"]))
                for r in cur.fetchall()
            ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", nargs=2, type=int, metavar=("ID_A", "ID_B"), help="print distance between two articles")
    p.add_argument("--csv", help="read articles from a CSV export instead of the DB")
    p.add_argument("--hours", type=float, default=24)
    p.add_argument("--nearest", type=float, default=0.5)
    p.add_argument("--centroid", type=float, default=0.55)
    args = p.parse_args()

    articles = load_articles_csv(args.csv) if args.csv else load_articles()
    by_id = {a.id: a for a in articles}
    if args.pair:
        a, b = (by_id.get(i) for i in args.pair)
        if a is None or b is None:
            print("article not found (or not embedded)", file=sys.stderr)
            return 1
        print(f"#{a.id}: {a.headline}\n#{b.id}: {b.headline}")
        print(f"cosine distance {cosine_distance(a.embedding, b.embedding):.3f}, "
              f"gap {abs(a.discovered_at - b.discovered_at)}")
        return 0

    old = cluster_old(articles)
    new = cluster_new(articles, args.hours, args.nearest, args.centroid)
    print(f"{len(articles)} embedded articles")
    print(describe("old (0.3 single-link, 14d)", old))
    print(describe(f"new ({args.nearest}/{args.centroid}, +/-{args.hours:g}h)", new))
    lines = diff_stories(old, new, by_id)
    print("\n".join(["", *lines]) if lines else "\nno old story was split by the new rules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
