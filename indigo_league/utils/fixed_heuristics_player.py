from poke_env.player import SimpleHeuristicsPlayer


class FixedHeuristicsPlayer(SimpleHeuristicsPlayer):
    def _stat_estimation(self, mon, stat):
        if mon is None:
            return 1.0
        return super()._stat_estimation(mon, stat)
