import typing

from PyQt5 import QtWidgets

from gui.frame_widget import frame_widget
from gui.tab_base import TabBase
from gui.training.trueskills_tab import TrueskillsTab
from gui.training.win_rates_tab import WinRatesTab
from utils import PokePath


class ResultsTab(TabBase):
    def __init__(self):
        super().__init__()

        right_layout = QtWidgets.QVBoxLayout(self)

        right_layout.addWidget(
            frame_widget(
                TrueskillsTab(PokePath().league_dir / "trueskills.yaml"),
                fixed_width=False,
                fixed_height=False,
            ),
        )
        right_layout.addWidget(WinRatesTab(challengers_dir=PokePath().challenger_dir))

    def on_train(self) -> typing.Dict[str, typing.Any]:
        return {}
