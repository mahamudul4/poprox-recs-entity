import numpy as np
from lenskit.pipeline import Component

from poprox_concepts.domain import CandidateSet, InterestProfile
from poprox_recommender.components.entity_matching import ENTITY_TYPES, entity_key

# AP relevance is on a [0, 100] scale; ignore weak mentions so a passing
# reference doesn't move the score
RELEVANCE_THRESHOLD = 25.0


class EntityArticleScorer(Component):
    """Score articles by how well they match the user's rated entities.

    Entities are matched by normalized name and type (not entity_id), because a
    rated entity and the same entity mentioned in an article are different rows
    with different ids. Scores are on a 0-1 scale with 0.5 as neutral, so they
    can be fused with other 0-1 scores using plain weights.
    """

    config: None

    def __call__(self, candidate_articles: CandidateSet, interest_profile: InterestProfile) -> CandidateSet:
        with_scores = candidate_articles.model_copy()

        # weight each rated entity by its rating: 5 -> +2, 4 -> +1, 2 -> -1, 1 -> -2
        weights: dict = {}
        for entity_type in ENTITY_TYPES:
            for interest in interest_profile.interests_by_type(entity_type):
                weight = interest.preference - 3
                if weight != 0:
                    weights[entity_key(entity_type, interest.entity_name)] = weight

        if not weights:
            # no rated entities -> no signal, ScoreFusion skips it
            with_scores.scores = None
            return with_scores

        scores = []
        for article in candidate_articles.articles:
            raw = 0.0
            for mention in article.mentions:
                relevance = mention.relevance or 0.0
                if relevance < RELEVANCE_THRESHOLD:
                    continue
                key = entity_key(mention.entity.entity_type, mention.entity.name)
                raw += weights.get(key, 0) * relevance / 100.0
            # raw is in [-2, 2] after averaging over rated entities (weights are
            # at most +/-2 and relevance at most 1), so this maps it into [0, 1]
            scores.append(0.5 + raw / len(weights) / 4.0)

        # clip in case an article mentions the same entity more than once
        with_scores.scores = np.clip(np.array(scores, dtype=np.float32), 0.0, 1.0)
        return with_scores
