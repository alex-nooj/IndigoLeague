import pathlib
import typing

from PyQt5.QtCore import QFileInfo
from PyQt5.QtWidgets import (
    QCheckBox,
    QLineEdit,
    QFileDialog,
    QHBoxLayout,
)

from gui.labelled_frame import PokeWidget


class ResumeTab(PokeWidget):
    def __init__(self):
        super().__init__()

        # Checkbox
        self.check_box = QCheckBox("Resume", self)
        self.check_box.stateChanged.connect(self.checkbox_changed)

        # Label and Text Box
        self.text_box = QLineEdit(self)

        # Layout
        hbox = QHBoxLayout()
        hbox.addWidget(self.check_box)
        hbox.addWidget(self.text_box)
        self.setLayout(hbox)

    def checkbox_changed(self, state):
        if state == 0:
            self.text_box.clear()
        else:
            self.browse_files()

    def browse_files(self):
        if self.check_box.isChecked():
            file_dialog = QFileDialog()
            file_dialog.setOption(QFileDialog.DontUseNativeDialog)
            file_dialog.setNameFilter("ZIP files (*.zip)")
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            if file_dialog.exec_():
                filename = file_dialog.selectedFiles()[0]
                absolute_path = QFileInfo(filename).absoluteFilePath()
                print(absolute_path)
                self.text_box.setText(absolute_path)

    def start_training(self) -> typing.Dict[str, typing.Union[str, None]]:
        if pathlib.Path(self.text_box.text()).is_file():
            print(self.text_box.text())
            return {"resume": self.text_box.text()}
        else:
            return {"resume": None}
