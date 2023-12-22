import pathlib
import sys
import typing

import stable_baselines3.common.callbacks as sb3_callbacks
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

from battling.callbacks.gui_close_callback import ControllerCallback
from battling.callbacks.gui_close_callback import RunnerCheck
from battling.callbacks.save_peripherals_callback import SavePeripheralsCallback
from battling.callbacks.success_callback import SuccessCallback
from gui.training.network_tab import NetworkTab
from gui.training.overview_tab import OverviewTab
from gui.training.preprocessing_tab import PreprocessingTab
from gui.training.resume_tab import ResumeTab
from gui.training.rewards_tab import RewardsTab
from gui.training.trueskills_tab import TrueskillsTab
from gui.training.win_rates_tab import WinRatesTab
from main import create_env
from main import new_training
from main import resume_training
from utils import PokePath


def frame_widget(
    widget: QtWidgets.QWidget,
    fixed_width: bool = True,
    fixed_height: bool = True,
    label: str = None,
) -> QtWidgets.QFrame:
    frame = QtWidgets.QFrame()
    frame.setFrameStyle(QtWidgets.QFrame.Box)
    frame.setFrameShadow(QtWidgets.QFrame.Sunken)
    widget_box = QtWidgets.QVBoxLayout()

    if label is not None:
        q_label = QtWidgets.QLabel(label)
        bold_font = QtGui.QFont()
        bold_font.setBold(True)
        bold_font.setUnderline(True)
        q_label.setFont(bold_font)
        q_label.setFixedSize(q_label.sizeHint())
        widget_box.addWidget(q_label)

    widget_box.addWidget(widget)

    frame.setLayout(widget_box)
    if fixed_height and fixed_width:
        frame.setFixedSize(widget_box.sizeHint())
    elif fixed_width:
        frame.setFixedWidth(widget_box.sizeHint().width())
    elif fixed_height:
        frame.setFixedHeight(widget_box.sizeHint().height())

    return frame


async def hello_world():
    print("hello world!")


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
        self._event_loop = QtCore.QEventLoop()
        self.stop_signal_received = RunnerCheck()

    def run(self):
        if self.resume is not None and pathlib.Path(self.resume).is_file():
            tag, n_steps, poke_path, model, env, starting_team_size = resume_training(
                pathlib.Path(self.resume), self.battle_format, self.rewards
            )
            self.total_timesteps -= n_steps
        else:
            env = create_env(
                ops=self.ops,
                rewards=self.rewards,
                seq_len=self.seq_len,
                battle_format=self.battle_format,
                team_size=self.starting_team_size,
                tag=None,
            )

            tag, poke_path, model = new_training(
                env,
                self.seq_len,
                self.shared,
                self.pi,
                self.vf,
                None,
            )

        model.learn(
            total_timesteps=self.total_timesteps,
            callback=sb3_callbacks.CallbackList(
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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # set the window title and geometry
        self.setWindowTitle("Indigo League")

        # create the tab widget
        self.tabWidget = QtWidgets.QTabWidget(self)
        self.setCentralWidget(self.tabWidget)

        # Build the config grabbers
        self.overview_tab = OverviewTab()
        self.preproc_tab = PreprocessingTab()
        self.rewards_tab = RewardsTab()
        self.network_tab = NetworkTab()
        self.resume_tab = ResumeTab()

        # Organize the GUI layout
        training_tab = QtWidgets.QWidget()
        training_main_box = QtWidgets.QVBoxLayout(training_tab)
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

        # create the "Network" tab
        self.tabWidget.addTab(training_tab, "Training")

        # create the "Other" tab
        other_tab = QtWidgets.QWidget()
        self.tabWidget.addTab(other_tab, "Other")

        # create a vertical layout for the "Other" tab
        other_layout = QtWidgets.QVBoxLayout(other_tab)
        other_layout.addWidget(QtWidgets.QPushButton("Button 1"))
        other_layout.addWidget(QtWidgets.QPushButton("Button 2"))
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


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
