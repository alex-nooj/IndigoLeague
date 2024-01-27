import typing

from PyQt5 import QtGui
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import QFormLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QLineEdit

from gui.labelled_frame import PokeWidget

REWARD_FUNCS = {
    "Fainted": 0.3,
    "HP": 0.1,
    "Status": 0.0,
    "Victory": 1.0,
}


class RewardsTab(PokeWidget):
    def __init__(self):
        super().__init__()
        self.reward_funcs = {}
        main_layout = QFormLayout(self)
        my_font = QtGui.QFont()
        my_font.setBold(True)
        active_label = QLabel("Active")
        embedding_label = QLabel("Weight")
        active_label.setFont(my_font)
        embedding_label.setFont(my_font)
        main_layout.addRow(active_label, embedding_label)
        for reward in REWARD_FUNCS:
            checkbox = QCheckBox(reward)
            checkbox.setChecked(True)
            text = QLineEdit()
            text.setValidator(QDoubleValidator())
            text.setText(str(REWARD_FUNCS[reward]))
            self.reward_funcs[f"{reward.lower()}_value"] = {
                "checkbox": checkbox,
                "text": text,
            }
            main_layout.addRow(checkbox, text)
        self.setFixedWidth(300)

    def start_training(self) -> typing.Dict[str, typing.Dict[str, float]]:
        return {
            "rewards": {
                k: float(v["text"].text()) if v["checkbox"].isChecked() else 0.0
                for k, v in self.reward_funcs.items()
            }
        }
