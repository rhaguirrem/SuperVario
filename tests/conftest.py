import os
from pathlib import Path

import pytest

from supervario.cli import demo
from supervario.pipeline import load_config


@pytest.fixture
def project(tmp_path):
    return load_config(demo(tmp_path / "demo"))


@pytest.fixture(scope="session")
def qt_app():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt5.QtGui import QFont, QFontDatabase
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    # The Windows offscreen plugin has no system font database.
    font = Path(r"C:\Windows\Fonts\segoeui.ttf")
    if font.exists():
        QFontDatabase.addApplicationFont(str(font))
        app.setFont(QFont("Segoe UI", 9))
    yield app
