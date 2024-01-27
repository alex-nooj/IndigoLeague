import asyncio
import typing

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QFormLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QTextEdit

from battling.callbacks.gui_close_callback import RunnerCheck
from battling.environment.teams.team_builder import AgentTeamBuilder
from gui.tab_base import TabBase
from team_selection.run_genetic_algo import genetic_team_search


class GeneticSearchThread(QtCore.QThread):
    result_signal = QtCore.pyqtSignal(AgentTeamBuilder)

    def __init__(
        self,
        population_size: int,
        n_mutations: int,
        battle_format: str,
        n_gens: int,
        parent=None,
    ):
        super().__init__(parent)
        self.population_size = population_size
        self.n_mutations = n_mutations
        self.battle_format = battle_format
        self.n_gens = n_gens
        self._event_loop = QtCore.QEventLoop()
        self.stop_signal_received = RunnerCheck()

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        team = loop.run_until_complete(
            genetic_team_search(
                self.population_size, self.n_mutations, self.battle_format, self.n_gens
            )
        )
        self.result_signal.emit(team)

    def stop(self):
        self.stop_signal_received.continue_running = False
        self._event_loop.quit()
        self.result_signal.emit(AgentTeamBuilder(self.battle_format, 6))


class TeamTab(TabBase):
    def __init__(
        self,
        battle_format_box: QComboBox,
        population_size: int = 50,
        n_mutations: int = 30,
        n_gens: int = 1,
    ):
        super().__init__()

        left_box = QtWidgets.QHBoxLayout(self)
        left_layout = QFormLayout()
        self.population_size_line_edit = QLineEdit()
        self.population_size_line_edit.setValidator(QIntValidator())
        self.population_size_line_edit.setText(str(population_size))
        left_layout.addRow(QLabel("Population Size:"), self.population_size_line_edit)

        self.n_mutations_line_edit = QLineEdit()
        self.n_mutations_line_edit.setValidator(QIntValidator())
        self.n_mutations_line_edit.setText(str(n_mutations))
        left_layout.addRow(QLabel("Number of Mutations:"), self.n_mutations_line_edit)

        self.battle_combo_box = QComboBox()
        self.battle_combo_box.addItem("gen8ou")
        self.battle_combo_box.addItem("gen8randombattle")
        self.battle_combo_box.setCurrentText(battle_format_box.currentText())
        self.battle_combo_box.currentTextChanged.connect(
            battle_format_box.setCurrentText
        )
        battle_format_box.currentTextChanged.connect(
            self.battle_combo_box.setCurrentText
        )
        left_layout.addRow(QLabel("Battle Format:"), self.battle_combo_box)

        self.n_gens_line_edit = QLineEdit()
        self.n_gens_line_edit.setValidator(QIntValidator())
        self.n_gens_line_edit.setText(str(n_gens))
        left_layout.addRow(QLabel("Number of Generations:"), self.n_gens_line_edit)

        run_button = QPushButton("Run", self)
        run_button.clicked.connect(self.start_search)
        left_layout.addWidget(run_button)

        self.results_textbox = QTextEdit(self)
        self.results_textbox.setReadOnly(True)
        self.results_textbox.setFixedWidth(500)

        left_box.addLayout(left_layout)
        left_box.addWidget(self.results_textbox)
        self.setFixedWidth(1000)
        self.search_thread = None
        self._team = None

    def start_search(self):
        if self.search_thread is None:
            population_size = self.population_size_line_edit.text()
            n_mutations = self.n_mutations_line_edit.text()
            battle_format = self.battle_combo_box.currentText()
            n_gens = self.n_gens_line_edit.text()

            self.search_thread = GeneticSearchThread(
                population_size=int(population_size),
                n_mutations=int(n_mutations),
                battle_format=battle_format,
                n_gens=int(n_gens),
            )
            self.search_thread.result_signal.connect(self.set_team)
            self.search_thread.start()

    def set_team(self, team: AgentTeamBuilder):
        self._team = team
        self.search_thread = None
        self.results_textbox.setText("\n".join(self._team.team))

    def on_train(self) -> typing.Dict[str, AgentTeamBuilder]:
        return {"team": self._team}
