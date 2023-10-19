import typing

from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QTableWidget
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtWidgets import QVBoxLayout

from gui.labelled_frame import PokeWidget


class NetworkTable(PokeWidget):
    def __init__(self, label: str, layers: typing.List[int]):
        super().__init__()
        main_layout = QVBoxLayout(self)
        label_widget = QLabel(label)
        my_font = QtGui.QFont()
        my_font.setBold(True)
        label_widget.setFont(my_font)
        self.label = label
        main_layout.addWidget(label_widget)
        # create the table widget
        self.tableWidget = QTableWidget(self)

        # set the column count and headers
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setHorizontalHeaderLabels(["Name", "Output Size", "Delete"])
        font_metrics = self.tableWidget.fontMetrics()
        name_width = font_metrics.width("Linear 10") + 10
        output_width = font_metrics.width("Output Size") + 10
        delete_width = font_metrics.width("Delete") + 10

        self.tableWidget.setColumnWidth(0, name_width)
        self.tableWidget.setColumnWidth(1, output_width)
        self.tableWidget.setColumnWidth(2, delete_width)

        self.setFixedWidth(name_width + output_width + delete_width + 33)
        # add a button to add a new row to the table
        add_row_button = QPushButton("Add Row", self)
        add_row_button.clicked.connect(self.add_row)
        # add some example rows to the table
        for layer in layers:
            self.add_row(str(layer))

        # make the "Name" column immutable
        name_column = self.tableWidget.horizontalHeader().logicalIndex(0)
        for row in range(self.tableWidget.rowCount()):
            item = self.tableWidget.item(row, name_column)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        main_layout.addWidget(self.tableWidget)
        main_layout.addWidget(add_row_button)

    def add_row(self, starting_text: str = ""):
        # get the current row count
        row_count = self.tableWidget.rowCount()

        # add a new row to the table
        self.tableWidget.insertRow(row_count)

        # auto-populate the "name" field with "Linear x"
        name_item = QTableWidgetItem("Linear {}".format(row_count))
        self.tableWidget.setItem(row_count, 0, name_item)
        self.tableWidget.setItem(row_count, 1, QTableWidgetItem(starting_text))

        # add a button to remove the row
        remove_button = QPushButton("X", self.tableWidget)
        remove_button.clicked.connect(lambda: self.remove_row(row_count))
        self.tableWidget.setCellWidget(row_count, 2, remove_button)
        item = self.tableWidget.item(
            row_count, self.tableWidget.horizontalHeader().logicalIndex(0)
        )
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)

    def remove_row(self, row):
        self.tableWidget.removeRow(row)
        for ix, row in enumerate(range(self.tableWidget.rowCount())):
            self.tableWidget.setItem(ix, 0, QTableWidgetItem(f"Linear {ix}"))
            name_item = self.tableWidget.item(row, 0)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)

    def start_training(self) -> typing.Dict[str, typing.List[int]]:
        return {
            self.label.lower(): [
                int(self.tableWidget.item(row, 1).text())
                for row in range(self.tableWidget.rowCount())
                if self.tableWidget.item(row, 1).text().isnumeric()
            ]
        }


class NetworkTab(PokeWidget):
    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout(self)

        self.shared_table = NetworkTable("Shared", [128, 128, 128])
        self.pi_table = NetworkTable(
            "Pi",
            [
                64,
            ],
        )
        self.vf_table = NetworkTable(
            "Vf",
            [
                64,
            ],
        )

        main_layout.addWidget(self.shared_table)
        main_layout.addWidget(self.pi_table)
        main_layout.addWidget(self.vf_table)

    def start_training(self) -> typing.Dict[str, typing.List[int]]:
        return {
            **self.shared_table.start_training(),
            **self.pi_table.start_training(),
            **self.vf_table.start_training(),
        }
