import hashlib
from datetime import date

import numpy as np
from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import CandidateSet, ImpressedSection, InterestProfile
from poprox_recommender.components.entity_matching import (
    ENTITY_TYPES,
    STRONG_RELEVANCE,
    entity_key,
    select_mentioning_by_name,
)
from poprox_recommender.components.sections.base import select_mentioning

# only entities the user likes (rating 4-5) may headline a section
LIKED_THRESHOLD = 4


class EntityOrTopicalCandidatesConfig(BaseModel):
    min_candidates: int = 3
    random_seed: int = 22


class EntityOrTopicalCandidates(Component):
    """Seed a section from a highly-rated entity when one has enough articles today,
    otherwise fall back to a topic (the default nrms_sections behavior).

    Liked entities are matched to articles by normalized name (see
    components/entity_matching.py); the topic fallback matches by entity_id, the
    same as TopicalCandidates, since topics share canonical ids.
    """

    config: EntityOrTopicalCandidatesConfig

    def __call__(
        self,
        candidate_set: CandidateSet,
        interest_profile: InterestProfile,
        sections: list[ImpressedSection] | None = None,
        today: date | None = None,
        descending: bool | None = True,
    ) -> CandidateSet:
        if descending is None:
            descending = True

        sections = sections or []

        if today is None:
            today = date.today()

        random_seed = self.random_daily_seed(interest_profile.profile_id, today, self.config.random_seed)
        prev_section_seed_ids = [section.seed_entity_id for section in sections]

        # Stage 1: a liked entity (person/organization/place, rating >= 4), matched by name
        liked_entities = [
            interest
            for entity_type in ENTITY_TYPES
            for interest in interest_profile.interests_by_type(entity_type)
            if interest.preference >= LIKED_THRESHOLD
        ]
        seeded = self.seed_from(candidate_set, liked_entities, prev_section_seed_ids, random_seed, descending, True)
        if seeded is not None:
            return seeded

        # Stage 2: fall back to a topic, matched by entity_id like TopicalCandidates
        topics = list(interest_profile.interests_by_type("topic"))
        seeded = self.seed_from(candidate_set, topics, prev_section_seed_ids, random_seed, descending, False)
        if seeded is not None:
            return seeded

        # Stage 3: nothing qualifies -> empty section (dropped downstream by AddSection)
        return CandidateSet(articles=[])

    def seed_from(self, candidate_set, interests, prev_section_seed_ids, random_seed, descending, by_name):
        interests = [i for i in interests if i.entity_id not in prev_section_seed_ids]
        rng = np.random.default_rng(random_seed)
        rng.shuffle(interests)
        interests = sorted(interests, key=lambda i: i.preference, reverse=descending)

        for interest in interests:
            if by_name:
                # a rating-4/5 entity is a strong opinion, so seed its section
                # even from moderate-relevance mentions
                relevant_candidates = select_mentioning_by_name(
                    candidate_set,
                    {entity_key(interest.entity_type, interest.entity_name)},
                    min_relevance=STRONG_RELEVANCE,
                )
            else:
                relevant_candidates = select_mentioning(candidate_set, [interest.entity_id])
            if len(relevant_candidates.articles) >= self.config.min_candidates:
                relevant_candidates.seed_entity_id = interest.entity_id
                relevant_candidates.seed_entity_name = interest.entity_name
                relevant_candidates.seed_entity_type = interest.entity_type
                return relevant_candidates

        return None

    def random_daily_seed(self, profile_id, day, base_seed: int) -> int:
        seed_str = f"{profile_id}_{day.isoformat()}_{base_seed}"
        hash_obj = hashlib.sha256(seed_str.encode("utf-8"))
        return int(hash_obj.hexdigest(), 16)
