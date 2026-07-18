"""Shared helpers for matching a user's rated entities to article mentions.

Unlike the ~14 curated topics, person/organization/place entities do not share a
stable ``entity_id`` between the row a user rates (e.g. from the web entity
search) and the row an AP article mentions -- they come from different sources,
so their ids differ. We therefore match these entities by normalized *name* and
type, the same way TopicPrefsFilter already matches topics.
"""

import numpy as np

from poprox_concepts.domain import CandidateSet

# entity types a user can rate on the web "Entities" page (everything but topics)
ENTITY_TYPES = ("person", "organization", "place")


def normalize_entity_type(entity_type: str | None) -> str | None:
    # AP has historically used the British "organisation"; treat it as "organization"
    if entity_type == "organisation":
        return "organization"
    return entity_type


def normalize_entity_name(name: str | None) -> str:
    # case- and whitespace-insensitive so "Elon  Musk" == "elon musk"
    return " ".join((name or "").lower().split())


def entity_key(entity_type: str | None, name: str | None) -> tuple:
    """A (type, name) key used to match a rated entity to an article mention."""
    return (normalize_entity_type(entity_type), normalize_entity_name(name))


def select_mentioning_by_name(candidate: CandidateSet, keys: set, min_relevance: float = 76) -> CandidateSet:
    """Keep articles that mention any of the given (type, name) entity keys."""
    kept_articles = []
    kept_scores = []
    has_scores = getattr(candidate, "scores", None) is not None
    for idx, article in enumerate(candidate.articles):
        mentioned = {
            entity_key(m.entity.entity_type, m.entity.name)
            for m in article.mentions
            if m.relevance and m.relevance >= min_relevance
        }
        if keys & mentioned:
            kept_articles.append(article)
            if has_scores:
                kept_scores.append(candidate.scores[idx])

    filtered = CandidateSet(articles=kept_articles)
    filtered.scores = np.array(kept_scores) if kept_scores else None
    return filtered
