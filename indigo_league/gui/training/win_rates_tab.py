import pathlib

from omegaconf import OmegaConf
from PyQt5.QtChart import QBarCategoryAxis
from PyQt5.QtChart import QBarSeries
from PyQt5.QtChart import QBarSet
from PyQt5.QtChart import QChart
from PyQt5.QtChart import QChartView
from PyQt5.QtChart import QValueAxis
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget


class WinRatesTab(QWidget):
    def __init__(self, challengers_dir: pathlib.Path):
        super().__init__()
        self.challengers_dir = challengers_dir

        self.layout = QVBoxLayout(self)

        self.combo_box = QComboBox()

        self.layout.addWidget(self.combo_box)
        self.update_directory_list()

        self.chart_view = QChartView(self)
        self.layout.addWidget(self.chart_view)

        self.chart = QChart()
        self.chart.setTitle("Win Rates")

        self.series = QBarSeries()
        self.chart.addSeries(self.series)

        self.x_axis = QBarCategoryAxis()
        self.chart.setAxisX(self.x_axis, self.series)

        self.y_axis = QValueAxis()
        self.chart.setAxisY(self.y_axis, self.series)
        self.chart_view.setChart(self.chart)
        self.update()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(5000)  # Update every ten seconds

        self.combo_box.currentIndexChanged.connect(self.handle_directory_selection)
        self.setLayout(self.layout)

    def update_directory_list(self):
        curr_directory_names = sorted(
            [k.stem for k in self.challengers_dir.iterdir() if k.is_dir()]
        )
        curr_text = self.combo_box.currentText()
        self.combo_box.clear()
        self.combo_box.addItems(curr_directory_names)
        if curr_text in curr_directory_names:
            self.combo_box.setCurrentText(curr_text)

    def handle_directory_selection(self, index: int):
        self.update_chart()

    def update(self):
        self.update_directory_list()
        self.update_chart()

    def update_chart(self):
        x_data, y_data = self.read_data()
        if x_data and y_data:
            self.series.clear()

            bar_set = QBarSet("Data")
            self.series.append(bar_set)
            for y_value in y_data:
                bar_set.append(y_value)

            self.x_axis.clear()
            self.x_axis.append(x_data)
            self.y_axis.setRange(0, max(y_data))

    def read_data(
        self,
    ):
        file_path = (
            self.challengers_dir / self.combo_box.currentText() / "win_rates.yaml"
        )
        if file_path is not None and file_path.is_file():
            data = OmegaConf.load(file_path)
        else:
            return None, None
        x_data = [k for k in data]
        y_data = [v for v in data.values()]
        return x_data, y_data
