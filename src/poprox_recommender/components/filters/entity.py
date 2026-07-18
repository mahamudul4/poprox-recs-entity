import logging

import numpy as np
from lenskit.pipeline import Component

from poprox_concepts.domain import CandidateSet, InterestProfile
from poprox_recommender.components.entity_matching import ENTITY_TYPES, entity_key

logger = logging.getLogger(__name__)

# only act on confident AP mentions; same threshold TopicPrefsFilter uses
RELEVANCE_THRESHOLD = 76


class EntityPrefsFilter(Component):
    config: None

    def __call__(self, candidates: CandidateSet, interest_profile: InterestProfile) -> CandidateSet:
        # keyed by (type, normalized name) since rated entities and article
        # mentions don't share entity_ids -- see components/entity_matching.py
        prefs: dict = {}
        for entity_type in ENTITY_TYPES:
            for interest in interest_profile.interests_by_type(entity_type):
                prefs[entity_key(entity_type, interest.entity_name)] = interest.preference

        if not prefs:
            return candidates

        very_high = {k for k, v in prefs.items() if v == 5}
        high = {k for k, v in prefs.items() if v == 4}
        low = {k for k, v in prefs.items() if v == 2}
        very_low = {k for k, v in prefs.items() if v == 1}

        has_scores = getattr(candidates, "scores", None) is not None

        kept_articles = []
        kept_scores = []
        for idx, article in enumerate(candidates.articles):
            article_entities = {
                entity_key(mention.entity.entity_type, mention.entity.name)
                for mention in article.mentions
                if (mention.relevance or 0) >= RELEVANCE_THRESHOLD
            }

            # Strongly-liked entity present -> always keep.
            if overlap(article_entities, very_high):
                kept_articles.append(article)
                if has_scores:
                    kept_scores.append(candidates.scores[idx])
                continue

            # Strongly-disliked entity present -> drop.
            if overlap(article_entities, very_low):
                continue

            # Otherwise drop only if disliked entities outweigh liked ones.
            if overlap(article_entities, high) < overlap(article_entities, low):
                continue

            kept_articles.append(article)
            if has_scores:
                kept_scores.append(candidates.scores[idx])

        logger.debug(
            "entity filter accepted %d of %d articles for user %s",
            len(kept_articles),
            len(candidates.articles),
            interest_profile.profile_id,
        )

        filtered = CandidateSet(articles=kept_articles)
        filtered.scores = np.array(kept_scores) if kept_scores else None
        return filtered


def overlap(a: set, b: set) -> int:
    return len(a.intersection(b))
