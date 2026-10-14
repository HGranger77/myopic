from database import operations as ops
from nlp.claims import group_claims, is_claim_candidate


def _claim(id, article_id, source, vec):
    return {"id": id, "article_id": article_id, "source": source, "embedding": vec}


def test_candidate_needs_figure_or_quote_and_sane_length():
    long_enough = "The council approved the new road budget on Tuesday night."
    assert not is_claim_candidate(long_enough, {"DATE", "ORG"})
    assert is_claim_candidate(long_enough, {"MONEY"})
    assert is_claim_candidate('The mayor said "we will not back down" at the rally.', set())
    assert not is_claim_candidate("It cost $5.", {"MONEY"})  # too short
    assert not is_claim_candidate("x" * 500, {"MONEY"})  # too long


def test_group_claims_corroborated_across_sources_single_otherwise():
    claims = [
        _claim(1, 10, "abc", [1.0, 0.0]),
        _claim(2, 20, "aljazeera", [0.99, 0.141]),  # ~identical direction
        _claim(3, 10, "abc", [0.0, 1.0]),            # unrelated
    ]
    groups = {tuple(g["claim_ids"]): g for g in group_claims(claims, 0.25)}
    assert groups[(1, 2)]["status"] == "corroborated"
    assert groups[(1, 2)]["source_count"] == 2
    assert groups[(3,)]["status"] == "single_source"


def test_group_claims_same_article_never_corroborates_itself():
    claims = [_claim(1, 10, "abc", [1.0, 0.0]), _claim(2, 10, "abc", [1.0, 0.0])]
    groups = group_claims(claims, 0.25)
    assert len(groups) == 2
    assert all(g["status"] == "single_source" for g in groups)


def test_group_claims_same_source_different_articles_is_not_corroboration():
    claims = [_claim(1, 10, "abc", [1.0, 0.0]), _claim(2, 11, "abc", [1.0, 0.0])]
    (group,) = group_claims(claims, 0.25)
    assert group["status"] == "single_source"


def _relevant_article(url, run_id, story_id):
    article_id, _ = ops.upsert_article("abc_news", url, "H", "home", run_id)
    ops.mark_article_fetched(article_id, "body text")
    ops.set_article_relevance(article_id, True)
    ops.add_article_to_story(story_id, article_id, run_id)
    return article_id


def test_claims_worklist_checked_once_and_groups_roundtrip():
    run_id = ops.create_scrape_run()
    story_id = ops.create_story()
    a1 = _relevant_article("https://example.com/c1", run_id, story_id)
    a2 = _relevant_article("https://example.com/c2", run_id, story_id)
    assert {r["article_id"] for r in ops.get_relevant_articles_missing_claims()} == {a1, a2}

    vec = [1.0] + [0.0] * 383
    ops.insert_claims(a1, [{"sentence": "Twelve people died in the crash.", "embedding": vec}])
    ops.insert_claims(a2, [])  # zero candidates must still drop out of the worklist
    assert ops.get_relevant_articles_missing_claims() == []

    ops.insert_claims(a1, [{"sentence": "Twelve people died in the crash.", "embedding": vec}])  # retry: no dupes
    claims = ops.get_story_claims(story_id)
    assert len(claims) == 1
    # Regression: pgvector returns a Vector/ndarray depending on version, and
    # group_claims needs a plain list - this once crashed check-claims.
    assert isinstance(claims[0]["embedding"], list) and len(claims[0]["embedding"]) == 384

    ops.replace_story_claim_groups(
        story_id, [{"claim_ids": [claims[0]["id"]], "source_count": 1, "status": "single_source"}]
    )
    ops.replace_story_claim_groups(
        story_id, [{"claim_ids": [claims[0]["id"]], "source_count": 1, "status": "single_source"}]
    )
    (group,) = ops.get_story_claim_groups(story_id)
    assert group["status"] == "single_source"
    assert group["claims"][0]["sentence"] == "Twelve people died in the crash."
