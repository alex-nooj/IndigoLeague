import typing

from PyQt5 import QtGui
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import QFormLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QLineEdit

from gui.labelled_frame import PokeWidget

EMBEDDING_PROC = [
    "Embed Pokemon IDs",
    "Embed Moves",
    "Embed Abilities",
    "Embed Items",
]

NONEMBEDDING_PROC = [
    "Embed Active Pokemon",
    "Simple Op",
    "Embed Team",
    "Embed Field",
]


PREPROCESSOR_IMPORT_PATH = "battling.environment.preprocessing.ops"


class PreprocessingTab(PokeWidget):
    def __init__(self):
        super().__init__()
        self.preprocessors = {}
        main_layout = QFormLayout(self)
        my_font = QtGui.QFont()
        my_font.setBold(True)
        active_label = QLabel("Active")
        embedding_label = QLabel("Embedding Size")
        active_label.setFont(my_font)
        embedding_label.setFont(my_font)
        main_layout.addRow(active_label, embedding_label)
        for preprocessor in EMBEDDING_PROC:
            checkbox = QCheckBox(preprocessor)
            text = QLineEdit()
            text.setValidator(QIntValidator())
            text.setText("4")
            self.preprocessors[
                f"{PREPROCESSOR_IMPORT_PATH}.{preprocessor.lower().replace(' ', '_')}.{preprocessor.replace(' ', '')}"
            ] = {"checkbox": checkbox, "text": text}
            main_layout.addRow(checkbox, text)

        for preprocessor in NONEMBEDDING_PROC:
            checkbox = QCheckBox(preprocessor)
            checkbox.setChecked(preprocessor == "Simple Op")
            text = QLineEdit()
            text.setEnabled(False)
            self.preprocessors[
                f"battling.environment.preprocessing.ops.{preprocessor.lower().replace(' ', '_')}.{preprocessor.replace(' ', '')}"
            ] = {"checkbox": checkbox, "text": text}
            main_layout.addRow(checkbox, text)
        self.setFixedWidth(300)

    def start_training(self) -> typing.Dict[str, typing.Dict[str, int]]:
        ret_dict = {}
        for k, v in self.preprocessors.items():
            if v["checkbox"].isChecked():
                ret_dict[k] = (
                    {"embedding_size": int(v["text"].text())}
                    if len(v["text"].text()) > 0
                    else {}
                )
        return ret_dict
