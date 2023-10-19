import pathlib

from PyQt5.QtChart import QBarCategoryAxis
from PyQt5.QtChart import QBarSeries
from PyQt5.QtChart import QBarSet
from PyQt5.QtChart import QChart
from PyQt5.QtChart import QChartView
from PyQt5.QtChart import QValueAxis
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget
from omegaconf import OmegaConf


class TrueskillsTab(QWidget):
    def __init__(
        self,
        file_path: pathlib.Path,
    ):
        super().__init__()
        self.file_path = file_path

        self.layout = QVBoxLayout(self)
        self.chart_view = QChartView(self)
        self.layout.addWidget(self.chart_view)

        self.chart = QChart()
        self.chart.setTitle(self.file_path.stem.capitalize())

        self.series = QBarSeries()
        self.chart.addSeries(self.series)

        self.x_axis = QBarCategoryAxis()
        self.chart.setAxisX(self.x_axis, self.series)

        self.y_axis = QValueAxis()
        self.chart.setAxisY(self.y_axis, self.series)
        self.chart_view.setChart(self.chart)
        self.update_chart()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_chart)
        self.timer.start(10000)  # Update every ten seconds
        self.setLayout(self.layout)

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

    def read_data(self, ):
        try:
            data = OmegaConf.load(self.file_path)
        except FileNotFoundError:
            print("File not found...")
            return None, None
        x_data = [k for k in data]
        y_data = [v.mu for v in data.values()]
        return x_data, y_data
