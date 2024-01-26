import typing

import stable_baselines3.common.callbacks as sb3_callbacks
from PyQt5 import QtCore
from PyQt5 import QtWidgets

import utils
from battling.callbacks.gui_close_callback import ControllerCallback
from battling.callbacks.gui_close_callback import RunnerCheck
from battling.callbacks.save_peripherals_callback import SavePeripheralsCallback
from battling.callbacks.success_callback import SuccessCallback
from battling.environment.teams.team_builder import AgentTeamBuilder
from gui.frame_widget import frame_widget
from gui.training.network_tab import NetworkTab
from gui.training.overview_tab import OverviewTab
from gui.training.preprocessing_tab import PreprocessingTab
from gui.training.resume_tab import ResumeTab
from gui.training.rewards_tab import RewardsTab
from gui.training.trueskills_tab import TrueskillsTab
from gui.training.win_rates_tab import WinRatesTab
from main import setup
from main import train
from utils import PokePath


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

    def run(self):
        poke_path = utils.PokePath(tag=None)
        env, model = setup(
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
            self.resume,
        )

        train(
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
                    ControllerCallback(self.stop_signal_received),
                ]
            ),
        )
        self.finished_signal.emit()

    def stop(self):
        self.stop_signal_received.continue_running = False
        self._event_loop.quit()
        self.finished_signal.emit()


class TrainingTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # Build the config grabbers
        self.overview_tab = OverviewTab()
        self.preproc_tab = PreprocessingTab()
        self.rewards_tab = RewardsTab()
        self.network_tab = NetworkTab()
        self.resume_tab = ResumeTab()
        self._team = None
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

        training_box.addLayout(left_layout)

        training_box.addWidget(
            frame_widget(
                self.network_tab,
                label="Network",
            )
        )
        right_layout = QtWidgets.QVBoxLayout()

        right_layout.addWidget(
            frame_widget(
                TrueskillsTab(PokePath().league_dir / "trueskills.yaml"),
                fixed_width=False,
                fixed_height=False,
            ),
        )
        right_layout.addWidget(WinRatesTab(challengers_dir=PokePath().challenger_dir))
        training_box.addLayout(right_layout)
        training_main_box.addLayout(training_box)
        training_main_box.addWidget(frame_widget(self.resume_tab, fixed_width=False))

        self.start_training_button = QtWidgets.QPushButton("Start", self)
        self.start_training_button.clicked.connect(self.start_training)

        training_main_box.addWidget(self.start_training_button)
        self.train_thread = None

    def start_training(self):
        if self.train_thread is None:
            overview_args = self.overview_tab.start_training()
            preproc_args = self.preproc_tab.start_training()
            reward_args = self.rewards_tab.start_training()
            network_args = self.network_tab.start_training()
            resume_args = self.resume_tab.start_training()
            self.train_thread = TrainThread(
                ops=preproc_args,
                rewards=reward_args,
                battle_format=overview_args["battle_format"],
                total_timesteps=overview_args["total_timesteps"],
                save_freq=overview_args["save_freq"],
                seq_len=overview_args["seq_len"],
                starting_team_size=overview_args["starting_team_size"],
                shared=network_args["shared"],
                pi=network_args["pi"],
                vf=network_args["vf"],
                resume=resume_args["resume"],
                team=self._team,
            )
            self.train_thread.finished_signal.connect(self.set_button_start)
            self.train_thread.start()
            self.swap_button()
        else:
            self.train_thread.stop()
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

    def closeEvent(self, event):
        if self.train_thread is not None:
            self.train_thread.stop()
            self.train_thread.wait()
        event.accept()

    def set_team(self, team: AgentTeamBuilder):
        self._team = team
