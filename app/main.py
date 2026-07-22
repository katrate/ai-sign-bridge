"""
app/main.py
============
Entry point for AI Sign Bridge desktop application.
Run from the project root: python app/main.py
"""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from app.main_window import MainWindow


def main():
    app = QApplication(sys.argv)

    # Set app-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Set app metadata
    app.setApplicationName("AI Sign Bridge")
    app.setOrganizationName("Hackathon")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
