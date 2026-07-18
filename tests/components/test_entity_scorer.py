from uuid import uuid4

import numpy as np

from poprox_concepts.domain import Article, CandidateSet, Entity, Mention
from poprox_concepts.domain.profile import AccountInterest, InterestProfile
from poprox_recommender.components.filters.entity import EntityPrefsFilter
from poprox_recommender.components.joiners.score import MinMaxScores
from poprox_recommender.components.scorers.entity_article import EntityArticleScorer

# Shared entity ids so user ratings and article mentions line up by entity_id.
ELON = uuid4()
TESLA = uuid4()
TRUMP = uuid4()
OTHER = uuid4()


def make_profile():
    return InterestProfile(
        click_history=[],
        entity_interests=[
            AccountInterest(entity_id=ELON, entity_name="Elon Musk", entity_type="person", preference=5),
            AccountInterest(entity_id=TESLA, entity_name="Tesla", entity_type="organization", preference=4),
            AccountInterest(entity_id=TRUMP, entity_name="Donald Trump", entity_type="person", preference=1),
        ],
    )


def mention(entity_id, name, entity_type, relevance):
    return Mention(
        source="AP",
        relevance=relevance,
        entity=Entity(entity_id=entity_id, name=name, entity_type=entity_type, source="AP"),
    )


def test_entity_scorer_boosts_liked_and_penalizes_disliked():
    profile = make_profile()

    a_liked = Article(
        article_id=uuid4(),
        headline="Tesla unveils new electric vehicle",
        mentions=[mention(ELON, "Elon Musk", "person", 75.0), mention(TESLA, "Tesla", "organization", 90.0)],
    )
    a_disliked = Article(
        article_id=uuid4(),
        headline="Trump announces new policy",
        mentions=[mention(TRUMP, "Donald Trump", "person", 95.0)],
    )
    a_neutral = Article(
        article_id=uuid4(),
        headline="Unrelated story",
        mentions=[mention(OTHER, "Some Org", "organization", 90.0)],
    )
    a_weak = Article(
        article_id=uuid4(),
        headline="Passing reference to Elon",
        mentions=[mention(ELON, "Elon Musk", "person", 10.0)],  # below relevance threshold -> ignored
    )

    candidates = CandidateSet(articles=[a_liked, a_disliked, a_neutral, a_weak])
    scored = EntityArticleScorer()(candidates, profile)

    scores = {a.article_id: s for a, s in zip(scored.articles, scored.scores)}

    # 3 rated entities; (+2*0.75 + 1*0.90) / 3 / 4 above neutral = 0.5 + 0.2
    assert abs(scores[a_liked.article_id] - 0.7) < 1e-4
    # (-2*0.95) / 3 / 4 below neutral
    assert abs(scores[a_disliked.article_id] - (0.5 - 1.9 / 12)) < 1e-4
    # no rated entities mentioned -> neutral
    assert abs(scores[a_neutral.article_id] - 0.5) < 1e-6
    # weak mention below threshold -> neutral
    assert abs(scores[a_weak.article_id] - 0.5) < 1e-6

    assert scores[a_liked.article_id] > scores[a_neutral.article_id] > scores[a_disliked.article_id]
    assert all(0.0 <= s <= 1.0 for s in scores.values())


def test_entity_scorer_ignores_neutral_ratings():
    # A rating of 3 is "no opinion" and should not affect any score.
    profile = InterestProfile(
        click_history=[],
        entity_interests=[
            AccountInterest(entity_id=ELON, entity_name="Elon Musk", entity_type="person", preference=5),
            AccountInterest(entity_id=OTHER, entity_name="Some Org", entity_type="organization", preference=3),
        ],
    )
    article = Article(
        article_id=uuid4(),
        headline="Some Org story",
        mentions=[mention(OTHER, "Some Org", "organization", 90.0)],
    )
    scored = EntityArticleScorer()(CandidateSet(articles=[article]), profile)
    assert abs(scored.scores[0] - 0.5) < 1e-6


def test_entity_scorer_is_neutral_without_entity_ratings():
    # Only a topic rating -> entity scorer should be a no-op (scores None).
    profile = InterestProfile(
        click_history=[],
        entity_interests=[
            AccountInterest(entity_id=uuid4(), entity_name="Politics", entity_type="topic", preference=5),
        ],
    )
    article = Article(article_id=uuid4(), headline="x", mentions=[mention(ELON, "Elon Musk", "person", 90.0)])
    scored = EntityArticleScorer()(CandidateSet(articles=[article]), profile)
    assert scored.scores is None


def test_entity_filter_drops_strongly_disliked():
    profile = make_profile()

    a_liked = Article(
        article_id=uuid4(),
        headline="Elon Musk news",
        mentions=[mention(ELON, "Elon Musk", "person", 90.0)],
    )
    a_disliked = Article(
        article_id=uuid4(),
        headline="Trump news",
        mentions=[mention(TRUMP, "Donald Trump", "person", 95.0)],
    )
    a_other = Article(
        article_id=uuid4(),
        headline="Other news",
        mentions=[mention(OTHER, "Some Org", "organization", 90.0)],
    )
    a_weak_dislike = Article(
        article_id=uuid4(),
        headline="Brief Trump mention",
        mentions=[mention(TRUMP, "Donald Trump", "person", 50.0)],  # below 76 -> not filtered
    )

    candidates = CandidateSet(articles=[a_liked, a_disliked, a_other, a_weak_dislike])
    kept = EntityPrefsFilter()(candidates, profile)
    kept_ids = {a.article_id for a in kept.articles}

    assert a_liked.article_id in kept_ids
    assert a_disliked.article_id not in kept_ids
    assert a_other.article_id in kept_ids
    assert a_weak_dislike.article_id in kept_ids


def make_articles(n):
    return [Article(article_id=uuid4(), headline=f"article {i}", mentions=[]) for i in range(n)]


def test_minmax_rescales_scores_to_unit_range():
    candidates = CandidateSet(articles=make_articles(3))
    candidates.scores = np.array([2.0, 6.0, 4.0])

    rescaled = MinMaxScores()(candidates)

    assert np.allclose(rescaled.scores, [0.0, 1.0, 0.5])


def test_minmax_passes_through_missing_scores():
    candidates = CandidateSet(articles=make_articles(2))
    candidates.scores = None

    rescaled = MinMaxScores()(candidates)

    assert rescaled.scores is None


def test_minmax_maps_equal_scores_to_neutral():
    candidates = CandidateSet(articles=make_articles(3))
    candidates.scores = np.array([7.0, 7.0, 7.0])

    rescaled = MinMaxScores()(candidates)

    assert np.allclose(rescaled.scores, [0.5, 0.5, 0.5])
