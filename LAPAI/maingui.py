import sys
import json

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QMessageBox,
    QInputDialog
)


class JSONEditor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.data = {}
        self.current_file = None

        self.setWindowTitle("LAPAI JSON Editor")
        self.resize(900, 500)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Key", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)

        layout.addWidget(self.table)

        button_layout = QHBoxLayout()

        self.btn_open = QPushButton("Open")
        self.btn_save = QPushButton("Save")
        self.btn_add = QPushButton("Add")
        self.btn_edit = QPushButton("Edit")
        self.btn_delete = QPushButton("Delete")

        button_layout.addWidget(self.btn_open)
        button_layout.addWidget(self.btn_save)
        button_layout.addWidget(self.btn_add)
        button_layout.addWidget(self.btn_edit)
        button_layout.addWidget(self.btn_delete)

        layout.addLayout(button_layout)

        self.btn_open.clicked.connect(self.open_json)
        self.btn_save.clicked.connect(self.save_json)
        self.btn_add.clicked.connect(self.add_item)
        self.btn_edit.clicked.connect(self.edit_item)
        self.btn_delete.clicked.connect(self.delete_item)

    def refresh_table(self):
        self.table.setRowCount(len(self.data))

        for row, (key, value) in enumerate(self.data.items()):
            self.table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.table.setItem(row, 1, QTableWidgetItem(str(value)))

    def open_json(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open JSON",
            "",
            "JSON Files (*.json)"
        )

        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

            if not isinstance(self.data, dict):
                QMessageBox.warning(
                    self,
                    "Error",
                    "Root JSON Must as object/dictionary."
                )
                self.data = {}
                return

            self.current_file = file_path
            self.refresh_table()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def save_json(self):
        if not self.current_file:
            QMessageBox.warning(
                self,
                "Warning",
                "NO file JSON opened yet."
            )
            return

        try:
            with open(self.current_file, "w", encoding="utf-8") as f:
                json.dump(
                    self.data,
                    f,
                    indent=4,
                    ensure_ascii=False
                )

            QMessageBox.information(
                self,
                "Success",
                "JSON save succeded."
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def add_item(self):
        key, ok = QInputDialog.getText(
            self,
            "add Key",
            "name Key:"
        )

        if not ok or not key:
            return

        value, ok = QInputDialog.getText(
            self,
            "add Value",
            "Value:"
        )

        if not ok:
            return

        self.data[key] = value
        self.refresh_table()

    def edit_item(self):
        row = self.table.currentRow()

        if row < 0:
            return

        key = self.table.item(row, 0).text()

        value, ok = QInputDialog.getText(
            self,
            "Edit Value",
            f"Value for '{key}'",
            text=str(self.data[key])
        )

        if not ok:
            return

        self.data[key] = value
        self.refresh_table()

    def delete_item(self):
        row = self.table.currentRow()

        if row < 0:
            return

        key = self.table.item(row, 0).text()

        reply = QMessageBox.question(
            self,
            "delete",
            f"delete '{key}'?"
        )

        if reply == QMessageBox.Yes:
            del self.data[key]
            self.refresh_table()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = JSONEditor()
    window.show()

    sys.exit(app.exec())
