import typing
from abc import abstractmethod

from PyQt5 import QtWidgets


class TabBase(QtWidgets.QWidget):
    @abstractmethod
    def on_train(self) -> typing.Dict[str, typing.Any]:
        ...
