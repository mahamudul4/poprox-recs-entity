from datetime import date
from uuid import uuid4

from poprox_concepts.domain import Article, CandidateSet, Entity, Mention
from poprox_concepts.domain.profile import AccountInterest, InterestProfile
from poprox_recommender.components.selectors.entity_sections import (
    EntityOrTopicalCandidates,
    EntityOrTopicalCandidatesConfig,
)

MUSK = uuid4()
TESLA = uuid4()
SPORTS = uuid4()


def mention(entity_id, name, entity_type, relevance=90.0):
    return Mention(
        source="AP",
        relevance=relevance,
        entity=Entity(entity_id=entity_id, name=name, entity_type=entity_type, source="AP"),
    )


def article(headline, *mentions):
    return Article(article_id=uuid4(), headline=headline, mentions=list(mentions))


def run(candidate_set, profile, sections=None):
    selector = EntityOrTopicalCandidates(EntityOrTopicalCandidatesConfig(min_candidates=3))
    return selector(candidate_set, profile, sections=sections, today=date(2026, 7, 18))


def test_liked_entity_with_enough_articles_seeds_the_section():
    profile = InterestProfile(
        click_history=[],
        entity_interests=[
            AccountInterest(entity_id=MUSK, entity_name="Elon Musk", entity_type="person", preference=5),
            AccountInterest(entity_id=SPORTS, entity_name="Sports", entity_type="topic", preference=5),
        ],
    )
    candidates = CandidateSet(
        articles=[
            article("Musk 1", mention(MUSK, "Elon Musk", "person")),
            article("Musk 2", mention(MUSK, "Elon Musk", "person")),
            article("Musk 3", mention(MUSK, "Elon Musk", "person")),
            article("Sports 1", mention(SPORTS, "Sports", "topic")),
        ]
    )
    seeded = run(candidates, profile)
    assert seeded.seed_entity_id == MUSK
    assert seeded.seed_entity_type == "person"
    assert len(seeded.articles) == 3


def test_entity_section_matches_by_name_across_different_ids():
    # Rated entity id differs from the article mention id (same name) -> must still seed.
    rated_id = uuid4()
    profile = InterestProfile(
        click_history=[],
        entity_interests=[
            AccountInterest(entity_id=rated_id, entity_name="Elon Musk", entity_type="person", preference=5),
        ],
    )
    candidates = CandidateSet(
        articles=[
            article("Musk 1", mention(uuid4(), "Elon Musk", "person")),
            article("Musk 2", mention(uuid4(), "Elon Musk", "person")),
            article("Musk 3", mention(uuid4(), "Elon Musk", "person")),
        ]
    )
    seeded = run(candidates, profile)
    assert seeded.seed_entity_name == "Elon Musk"
    assert len(seeded.articles) == 3


def test_falls_back_to_topic_when_no_entity_has_enough_articles():
    profile = InterestProfile(
        click_history=[],
        entity_interests=[
            AccountInterest(entity_id=MUSK, entity_name="Elon Musk", entity_type="person", preference=5),
            AccountInterest(entity_id=SPORTS, entity_name="Sports", entity_type="topic", preference=5),
        ],
    )
    # Only one Musk article (< min_candidates), but three Sports articles.
    candidates = CandidateSet(
        articles=[
            article("Musk 1", mention(MUSK, "Elon Musk", "person")),
            article("Sports 1", mention(SPORTS, "Sports", "topic")),
            article("Sports 2", mention(SPORTS, "Sports", "topic")),
            article("Sports 3", mention(SPORTS, "Sports", "topic")),
        ]
    )
    seeded = run(candidates, profile)
    assert seeded.seed_entity_id == SPORTS
    assert seeded.seed_entity_type == "topic"


def test_disliked_and_neutral_entities_do_not_seed_sections():
    # Musk rated 3 (neutral) -> not eligible to headline a section even with enough articles.
    profile = InterestProfile(
        click_history=[],
        entity_interests=[
            AccountInterest(entity_id=MUSK, entity_name="Elon Musk", entity_type="person", preference=3),
        ],
    )
    candidates = CandidateSet(
        articles=[
            article("Musk 1", mention(MUSK, "Elon Musk", "person")),
            article("Musk 2", mention(MUSK, "Elon Musk", "person")),
            article("Musk 3", mention(MUSK, "Elon Musk", "person")),
        ]
    )
    seeded = run(candidates, profile)
    # no eligible entity and no topic -> empty
    assert len(seeded.articles) == 0


def test_entity_already_used_as_a_seed_is_skipped():
    from poprox_concepts.domain import ImpressedSection

    profile = InterestProfile(
        click_history=[],
        entity_interests=[
            AccountInterest(entity_id=MUSK, entity_name="Elon Musk", entity_type="person", preference=5),
            AccountInterest(entity_id=TESLA, entity_name="Tesla", entity_type="organization", preference=4),
        ],
    )
    candidates = CandidateSet(
        articles=[
            article("Musk 1", mention(MUSK, "Elon Musk", "person")),
            article("Musk 2", mention(MUSK, "Elon Musk", "person")),
            article("Musk 3", mention(MUSK, "Elon Musk", "person")),
            article("Tesla 1", mention(TESLA, "Tesla", "organization")),
            article("Tesla 2", mention(TESLA, "Tesla", "organization")),
            article("Tesla 3", mention(TESLA, "Tesla", "organization")),
        ]
    )
    prior = ImpressedSection.from_articles([], title="Elon Musk", seed_entity_id=MUSK)
    seeded = run(candidates, profile, sections=[prior])
    # Musk already used -> next liked entity (Tesla) seeds this section
    assert seeded.seed_entity_id == TESLA
