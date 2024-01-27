import pathlib
import sys
import typing

import stable_baselines3.common.callbacks as sb3_callbacks
from omegaconf import OmegaConf
from PyQt5 import QtCore
from PyQt5 import QtWidgets

import main
import utils
from battling.callbacks.gui_close_callback import ControllerCallback
from battling.callbacks.gui_close_callback import RunnerCheck
from battling.callbacks.save_peripherals_callback import SavePeripheralsCallback
from battling.callbacks.success_callback import SuccessCallback
from battling.environment.teams.team_builder import AgentTeamBuilder
from gui.results.results_tab import ResultsTab
from gui.team_building.team_building_tab import TeamTab
from gui.training.training_tab import TrainingTab


class TrainThread(QtCore.QThread):
    finished_signal = QtCore.pyqtSignal()

    def __init__(
        self,
        ops: typing.Dict[str, typing.Dict[str, typing.Any]],
        rewards: typing.Dict[str, float],
        battle_format: str,
        total_timesteps: int,
        save_freq: int,
        seq_len: int,
        starting_team_size: int,
        shared: typing.List[int],
        pi: typing.List[int],
        vf: typing.List[int],
        resume: str,
        team: typing.Optional[AgentTeamBuilder] = None,
        parent=None,
    ):
        super().__init__(parent)

        self.ops = ops
        self.rewards = rewards
        self.battle_format = battle_format
        self.total_timesteps = total_timesteps
        self.save_freq = save_freq
        self.shared = shared
        self.pi = pi
        self.vf = vf
        self.resume = resume
        self.seq_len = seq_len
        self.starting_team_size = starting_team_size
        self.team = team
        self._event_loop = QtCore.QEventLoop()
        self.stop_signal_received = RunnerCheck()
        self.controller_callback = ControllerCallback(self.stop_signal_received)

    def run(self):
        if self.resume is not None and pathlib.Path(self.resume).is_file():
            poke_path, model, env, self.starting_team_size = main.resume_training(
                pathlib.Path(self.resume), self.battle_format, self.rewards
            )
        else:
            poke_path = utils.PokePath(tag=None)

            env, model = main.setup(
                self.ops,
                self.rewards,
                self.battle_format,
                1,
                self.shared,
                self.pi,
                self.vf,
                self.starting_team_size,
                poke_path,
                self.team,
                poke_path.tag,
            )
        main.train(
            env,
            model,
            self.starting_team_size,
            6,
            self.total_timesteps,
            self.save_freq,
            poke_path,
            poke_path.tag,
            callbacks=sb3_callbacks.CallbackList(
                [
                    sb3_callbacks.CheckpointCallback(
                        self.save_freq,
                        save_path=str(poke_path.agent_dir),
                        name_prefix=poke_path.tag,
                    ),
                    SavePeripheralsCallback(
                        poke_path=poke_path, save_freq=self.save_freq
                    ),
                    SuccessCallback(
                        agent_dir=poke_path.agent_dir,
                        league_dir=poke_path.league_dir,
                        tag=poke_path.tag,
                    ),
                    self.controller_callback,
                ]
            ),
        )
        self.finished_signal.emit()

    def stop(self):
        self.stop_signal_received.continue_running = False
        self.controller_callback.send_stop_signal()
        self._event_loop.quit()
        self.finished_signal.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # set the window title and geometry
        self.setWindowTitle("Indigo League")
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        layout = QtWidgets.QVBoxLayout(central_widget)

        # create the tab widget
        self.tabWidget = QtWidgets.QTabWidget()
        layout.addWidget(self.tabWidget)

        # Organize the GUI layout
        self.training_tab = TrainingTab()
        self.team_tab = TeamTab(self.training_tab.overview_tab.battle_combo_box)
        results_tab = ResultsTab()
        # create the "Network" tab
        self.tabWidget.addTab(self.training_tab, "Training")
        self.tabWidget.addTab(self.team_tab, "Team")
        self.tabWidget.addTab(results_tab, "Results")

        h_layout = QtWidgets.QHBoxLayout()

        self.start_training_button = QtWidgets.QPushButton("Start")
        self.start_training_button.clicked.connect(self.start_training)
        self.train_thread = None

        button2 = QtWidgets.QPushButton("Save")
        button2.clicked.connect(self.save_args)
        h_layout.addWidget(self.start_training_button)
        h_layout.addWidget(button2)
        layout.addLayout(h_layout)

    def start_training(self):
        if self.train_thread is None:
            args = {**self.training_tab.on_train(), **self.team_tab.on_train()}
            self.train_thread = TrainThread(**args)
            self.train_thread.finished_signal.connect(self.set_button_start)
            self.train_thread.start()
            self.swap_button()
        else:
            self.train_thread.stop()
            print("Stopping...")
            self.train_thread.wait()
            self.train_thread = None
            self.swap_button()

    def swap_button(self):
        if self.start_training_button.text() == "Start":
            self.set_button_stop()
        else:
            self.set_button_start()

    def set_button_start(self):
        self.start_training_button.setText("Start")

    def set_button_stop(self):
        self.start_training_button.setText("Stop")

    def save_args(self):
        file_path = pathlib.Path(__file__).parent / "main2.yaml"
        OmegaConf.save(
            config=OmegaConf.create(self.training_tab.on_train()), f=file_path
        )

    def closeEvent(self, event):
        if self.train_thread is not None:
            self.train_thread.stop()
            self.train_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
