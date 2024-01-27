import typing

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from gui.frame_widget import frame_widget
from gui.tab_base import TabBase
from gui.training.network_tab import NetworkTab
from gui.training.overview_tab import OverviewTab
from gui.training.preprocessing_tab import PreprocessingTab
from gui.training.resume_tab import ResumeTab
from gui.training.rewards_tab import RewardsTab


class TrainingTab(TabBase):
    def __init__(self):
        super().__init__()
        # Build the config grabbers
        self.overview_tab = OverviewTab()
        self.preproc_tab = PreprocessingTab()
        self.rewards_tab = RewardsTab()
        self.network_tab = NetworkTab()
        self.resume_tab = ResumeTab()

        training_main_box = QtWidgets.QVBoxLayout(self)
        training_box = QtWidgets.QHBoxLayout()

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(
            frame_widget(
                self.overview_tab,
                label="General",
            )
        )

        left_layout.addWidget(
            frame_widget(
                self.preproc_tab,
                label="Preprocessing",
            )
        )

        left_layout.addWidget(
            frame_widget(
                self.rewards_tab,
                label="Rewards",
            )
        )

        left_layout.setAlignment(Qt.AlignTop)
        training_box.addLayout(left_layout)

        training_box.addWidget(
            frame_widget(
                self.network_tab,
                label="Network",
            )
        )

        training_main_box.addLayout(training_box)
        training_main_box.addWidget(frame_widget(self.resume_tab, fixed_width=False))

    def on_train(self) -> typing.Dict[str, typing.Any]:
        return {
            **self.overview_tab.start_training(),
            **self.preproc_tab.start_training(),
            **self.rewards_tab.start_training(),
            **self.network_tab.start_training(),
            **self.resume_tab.start_training(),
        }
