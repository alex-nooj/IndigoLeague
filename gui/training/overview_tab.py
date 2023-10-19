import typing

from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QLabel,
    QComboBox,
    QLineEdit,
    QFormLayout,
)

from gui.labelled_frame import PokeWidget


class OverviewTab(PokeWidget):
    def __init__(self):
        super().__init__()
        main_layout = QFormLayout(self)

        # battle format
        self.battle_combo_box = QComboBox()
        self.battle_combo_box.addItem("gen8ou")
        self.battle_combo_box.addItem("gen8randombattle")
        main_layout.addRow(QLabel("Battle Format:"), self.battle_combo_box)

        # total timesteps
        self.timestep_line_edit = QLineEdit()
        self.timestep_line_edit.setValidator(QIntValidator())
        self.timestep_line_edit.setText("1000000")
        main_layout.addRow(QLabel("Total Timesteps:"), self.timestep_line_edit)

        # save frequency
        self.save_freq_line_edit = QLineEdit()
        self.save_freq_line_edit.setValidator(QIntValidator())
        self.save_freq_line_edit.setText("100000")
        main_layout.addRow(QLabel("Save Frequency:"), self.save_freq_line_edit)

        self.setFixedWidth(300)

    def start_training(self) -> typing.Dict[str, typing.Union[int, str]]:
        return {
            "battle_format": self.battle_combo_box.currentText(),
            "total_timesteps": int(self.timestep_line_edit.text())
            if self.timestep_line_edit.text().isnumeric()
            else 1_000_000,
            "save_freq": int(self.save_freq_line_edit.text())
            if self.save_freq_line_edit.text().isnumeric()
            else 100_000,
        }
