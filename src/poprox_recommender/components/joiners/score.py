from collections import defaultdict

import numpy as np
from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import CandidateSet


class MinMaxScores(Component):
    """Rescale scores to the 0-1 range so weighted fusion of different scorers is meaningful."""

    config: None

    def __call__(self, candidates: CandidateSet) -> CandidateSet:
        rescaled = candidates.model_copy()
        if candidates.scores is None or len(candidates.scores) == 0:
            return rescaled

        scores = np.asarray(candidates.scores, dtype=np.float32)
        low, high = scores.min(), scores.max()
        if high > low:
            rescaled.scores = (scores - low) / (high - low)
        else:
            # all scores equal -> no ranking information, treat as neutral
            rescaled.scores = np.full_like(scores, 0.5)
        return rescaled


class ScoreFusionConfig(BaseModel):
    combiner: str = "sum"
    weight1: float = 1
    weight2: float = 1


class ScoreFusion(Component):
    config: ScoreFusionConfig

    def __call__(self, candidates1: CandidateSet, candidates2: CandidateSet) -> CandidateSet:
        combined_score = defaultdict(float)
        combined_article = {}

        if candidates1.scores is not None:
            for article, score in zip(candidates1.articles, candidates1.scores):
                article_id = article.article_id
                combined_score[article_id] += self.config.weight1 * score
                combined_article[article_id] = article

        if candidates2.scores is not None:
            for article, score in zip(candidates2.articles, candidates2.scores):
                article_id = article.article_id
                if self.config.combiner == "sub":
                    combined_score[article_id] -= self.config.weight2 * score
                else:
                    combined_score[article_id] += self.config.weight2 * score
                combined_article[article_id] = article

        merged_scores = []
        merged_articles = []

        if self.config.combiner == "avg":
            denominator = self.config.weight1 + self.config.weight2
        else:
            denominator = 1

        for key, score in combined_score.items():
            merged_articles.append(combined_article[key])
            merged_scores.append(score / denominator)

        return CandidateSet(articles=merged_articles, scores=merged_scores)
