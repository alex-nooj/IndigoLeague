import trueskill

from indigo_league.training.environment.utils.opponent_selector import (
    qualities_to_probabilities,
)


class TestQualitiesToProbabilities:
    def test_sum(self):
        tags = [f"test {i}" for i in range(5)]
        agent_skills = {t: trueskill.Rating() for t in tags}
        probs = qualities_to_probabilities(tags[0], agent_skills)
        assert sum(probs) == 1.0

    def test_len(self):
        tags = [f"test {i}" for i in range(5)]
        agent_skills = {t: trueskill.Rating() for t in tags}
        probs = qualities_to_probabilities(tags[0], agent_skills)
        assert len(probs) == len(tags) - 1
