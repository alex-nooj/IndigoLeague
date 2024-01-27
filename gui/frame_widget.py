from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


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
    frame.layout().setAlignment(Qt.AlignTop)
    return frame
