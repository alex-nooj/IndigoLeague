import typing

from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QFormLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QSpinBox

from gui.labelled_frame import PokeWidget


class OverviewTab(PokeWidget):
    def __init__(
        self,
        battle_format: str = "gen8ou",
        total_timesteps: int = 10_000_000,
        save_freq: int = 100_000,
        seq_len: int = 3,
        starting_team_size: int = 1,
    ):
        super().__init__()
        main_layout = QFormLayout(self)

        # battle format
        self.battle_combo_box = QComboBox()
        self.battle_combo_box.addItem("gen8ou")
        self.battle_combo_box.addItem("gen8randombattle")
        self.battle_combo_box.setCurrentText(battle_format)
        main_layout.addRow(QLabel("Battle Format:"), self.battle_combo_box)

        # total timesteps
        self.timestep_line_edit = QLineEdit()
        self.timestep_line_edit.setValidator(QIntValidator())
        self.timestep_line_edit.setText(str(total_timesteps))
        main_layout.addRow(QLabel("Total Timesteps:"), self.timestep_line_edit)

        # save frequency
        self.save_freq_line_edit = QLineEdit()
        self.save_freq_line_edit.setValidator(QIntValidator())
        self.save_freq_line_edit.setText(str(save_freq))
        main_layout.addRow(QLabel("Save Frequency:"), self.save_freq_line_edit)

        self.seq_len_line_edit = QLineEdit()
        self.seq_len_line_edit.setValidator(QIntValidator())
        self.seq_len_line_edit.setText(str(seq_len))
        main_layout.addRow(QLabel("Sequence Length:"), self.seq_len_line_edit)

        self.starting_team_size_spinbox = QSpinBox()
        self.starting_team_size_spinbox.setMinimum(1)
        self.starting_team_size_spinbox.setMaximum(6)
        self.starting_team_size_spinbox.setValue(starting_team_size)
        main_layout.addRow(
            QLabel("Initial Team Size:"), self.starting_team_size_spinbox
        )

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
            "seq_len": int(self.seq_len_line_edit.text()),
            "starting_team_size": int(self.starting_team_size_spinbox.text()),
        }
