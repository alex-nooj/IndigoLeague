import sys

from PyQt5 import QtWidgets

from indigo_league.gui.team_building.team_building_tab import TeamTab
from indigo_league.gui.training.training_tab import TrainingTab


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # set the window title and geometry
        self.setWindowTitle("Indigo League")

        # create the tab widget
        self.tabWidget = QtWidgets.QTabWidget(self)
        self.setCentralWidget(self.tabWidget)

        # Organize the GUI layout
        training_tab = TrainingTab()
        team_tab = TeamTab(training_tab.overview_tab.battle_combo_box)
        # create the "Network" tab
        self.tabWidget.addTab(training_tab, "Training")
        self.tabWidget.addTab(team_tab, "Team")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
