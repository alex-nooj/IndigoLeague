import os
import pathlib

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget


class DirectoryComboBox(QWidget):
    def __init__(self, parent=None):
        super(DirectoryComboBox, self).__init__(parent)
        self.comboBox = QComboBox(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.comboBox)

        # Set the initial directory path
        self.directory_path = pathlib.Path(__file__).parent.parent.parent

        # Populate the combo box with directory names
        self.update_directory_list()

    def update_directory_list(self):
        # Clear the existing items in the combo box
        self.comboBox.clear()

        # Get the list of directory names in the specified path
        directory_names = [
            name
            for name in os.listdir(self.directory_path)
            if os.path.isdir(os.path.join(self.directory_path, name))
        ]

        # Add directory names to the combo box
        self.comboBox.addItems(directory_names)

        # Connect the signal to handle item selection
        self.comboBox.currentIndexChanged.connect(self.handle_directory_selection)

    def handle_directory_selection(self, index):
        # Get the selected directory name
        selected_directory = self.comboBox.currentText()

        # Do something with the selected directory, such as print it
        print("Selected Directory:", selected_directory)


if __name__ == "__main__":
    app = QApplication([])
    window = DirectoryComboBox()
    window.show()
    app.exec_()
