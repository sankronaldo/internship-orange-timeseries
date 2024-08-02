import os
import sys
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.rinterface_lib import callbacks

from AnyQt.QtWidgets import (
    QPlainTextEdit, QSplitter, QWidget, QVBoxLayout, QPushButton,
    QListView, QSizePolicy, QAction, QMenu, QToolButton, QInputDialog,
    QLabel, QMainWindow, QScrollArea
)
from AnyQt.QtGui import QFont, QTextCursor, QSyntaxHighlighter, QTextCharFormat, QColor, QPixmap
from AnyQt.QtCore import Qt, QItemSelectionModel, QRegExp

from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.settings import Setting
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils import itemmodels
from Orange.base import Learner, Model
import tempfile

# Initialize numpy to R conversion
numpy2ri.activate()


class RSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        keyword_format = QTextCharFormat()
        keyword_format.setForeground(Qt.darkBlue)
        keyword_format.setFontWeight(QFont.Bold)
        keywords = [
            "if", "else", "repeat", "while", "function", "for", "in", "next", "break",
            "TRUE", "FALSE", "NULL", "Inf", "NaN", "NA", "NA_integer_", "NA_real_", "NA_complex_", "NA_character_"
        ]
        for word in keywords:
            pattern = QRegExp("\\b" + word + "\\b")
            self.highlighting_rules.append((pattern, keyword_format))

        builtin_format = QTextCharFormat()
        builtin_format.setForeground(Qt.darkCyan)
        builtins = [
            "print", "cat", "sprintf", "paste", "paste0", "length", "nrow", "ncol", "dim"
        ]
        for word in builtins:
            pattern = QRegExp("\\b" + word + "\\b")
            self.highlighting_rules.append((pattern, builtin_format))

        number_format = QTextCharFormat()
        number_format.setForeground(Qt.darkGreen)
        number_pattern = QRegExp("\\b[+-]?[0-9]*\\.?[0-9]+([eE][+-]?[0-9]+)?\\b")
        self.highlighting_rules.append((number_pattern, number_format))

        string_format = QTextCharFormat()
        string_format.setForeground(Qt.darkRed)
        string_pattern = QRegExp("\".*?\"|'.*?'")
        self.highlighting_rules.append((string_pattern, string_format))

        comment_format = QTextCharFormat()
        comment_format.setForeground(Qt.gray)
        comment_pattern = QRegExp("#[^\n]*")
        self.highlighting_rules.append((comment_pattern, comment_format))

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            expression = QRegExp(pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)


class RScriptEditor(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont("Courier", 10))
        self.highlighter = RSyntaxHighlighter(self.document())


class RConsole(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(False)
        self.prompt = "R> "
        self.insertPlainText(self.prompt)
        self.last_command = ""

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            self.execute_command()
        else:
            super().keyPressEvent(event)

    def execute_command(self):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.movePosition(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
        line = cursor.selectedText()[len(self.prompt):]
        self.last_command = line.strip()

        if self.last_command:
            self.insertPlainText('\n')
            try:
                with self.redirect_stdout_stderr():
                    result = robjects.r(self.last_command)
                    if result is not None:
                        print(result)
            except Exception as e:
                print(f"Error: {str(e)}")

        self.insertPlainText(self.prompt)
        self.ensureCursorVisible()

    def write(self, text):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def redirect_stdout_stderr(self):
        class OutputRedirector:
            def __init__(self, console):
                self.console = console

            def write(self, text):
                self.console.write(text)

        return callbacks.obj_in_module(sys, "stdout", OutputRedirector(self))


class ROutputHandler:
    def __init__(self, console):
        self.console = console

    def write(self, text):
        self.console.write(text)


class PlotWindow(QMainWindow):
    def __init__(self, pixmap):
        super().__init__()
        self.setWindowTitle("R Plot")

        # Create a QLabel to hold the pixmap
        self.label = QLabel()
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)

        # Create a scroll area and set the label as its widget
        scroll = QScrollArea()
        scroll.setWidget(self.label)
        scroll.setWidgetResizable(True)

        # Set the scroll area as the central widget
        self.setCentralWidget(scroll)

        # Set a minimum size for the window
        self.setMinimumSize(400, 300)

    def resizeEvent(self, event):
        # Resize the pixmap to fit the window while maintaining aspect ratio
        pixmap = self.label.pixmap()
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled_pixmap)
        super().resizeEvent(event)


class ScriptItemModel(itemmodels.PyListModel):
    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self[index.row()]["name"]
        return super().data(index, role)


class OWRScript(OWWidget):
    name = "R Script"
    description = "Run R scripts"
    icon = "icons/RScript.svg"
    priority = 3160

    class Inputs:
        data = Input("Data", Table)
        learner = Input("Learner", Learner)
        model = Input("Model", Model)

    class Outputs:
        data = Output("Data", Table)
        learner = Output("Learner", Learner)
        model = Output("Model", Model)

    want_main_area = True
    resizing_enabled = True

    currentScriptIndex = Setting(0)
    scriptText = Setting("")
    splitterState = Setting(None)
    scriptLibrary = Setting([{"name": "New script", "script": "# R script here\n"}])

    def __init__(self):
        super().__init__()

        self.data = None
        self.learner = None
        self.model = None

        self.controlArea.layout().setContentsMargins(4, 4, 4, 4)
        self.mainArea.layout().setContentsMargins(0, 4, 4, 4)

        # GUI
        self.splitCanvas = QSplitter(Qt.Vertical, self.mainArea)
        self.mainArea.layout().addWidget(self.splitCanvas)

        self.text = RScriptEditor(self)
        self.splitCanvas.addWidget(self.text)

        self.console = RConsole(self)
        self.splitCanvas.addWidget(self.console)

        self.controlBox = gui.vBox(self.controlArea, "Script Library")

        self.libraryList = ScriptItemModel(self.scriptLibrary, self)
        self.libraryView = QListView(
            editTriggers=QListView.DoubleClicked | QListView.EditKeyPressed,
            sizePolicy=QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        )
        self.libraryView.setModel(self.libraryList)
        self.libraryView.selectionModel().selectionChanged.connect(self.onSelectedScriptChanged)
        self.controlBox.layout().addWidget(self.libraryView)

        w = gui.hBox(self.controlBox)

        self.addScriptButton = gui.button(w, self, "+", callback=self.onAddScript)
        self.removeScriptButton = gui.button(w, self, "-", callback=self.onRemoveScript)
        self.updateScriptButton = gui.button(w, self, "Update", callback=self.onUpdateScript)

        gui.button(self.controlArea, self, "Run script", callback=self.run)

        self.resize(800, 600)

        # Ensure there's at least one script in the library
        if not self.scriptLibrary:
            self.scriptLibrary.append({"name": "New script", "script": "# R script here\n"})

        # Ensure currentScriptIndex is within bounds
        self.currentScriptIndex = min(self.currentScriptIndex, len(self.scriptLibrary) - 1)

        self.setSelectedScript(self.currentScriptIndex)

        self.r_output_handler = ROutputHandler(self.console)

    @Inputs.data
    def set_data(self, data):
        self.data = data

    @Inputs.learner
    def set_learner(self, learner):
        self.learner = learner

    @Inputs.model
    def set_model(self, model):
        self.model = model

    def onAddScript(self):
        name, ok = QInputDialog.getText(self, "New Script", "Enter script name:")
        if ok and name:
            self.libraryList.append({"name": name, "script": "# R script here\n"})
            self.setSelectedScript(len(self.libraryList) - 1)

    def onRemoveScript(self):
        if len(self.libraryList) > 1:
            index = self.selectedScriptIndex()
            del self.libraryList[index]
            self.setSelectedScript(max(index - 1, 0))

    def onUpdateScript(self):
        index = self.selectedScriptIndex()
        if index is not None:
            self.libraryList[index]["script"] = self.text.toPlainText()

    def selectedScriptIndex(self):
        rows = self.libraryView.selectionModel().selectedRows()
        if rows:
            return rows[0].row()
        return None

    def setSelectedScript(self, index):
        if 0 <= index < len(self.libraryList):
            self.libraryView.setCurrentIndex(self.libraryList.index(index))
            self.currentScriptIndex = index
            self.text.setPlainText(self.libraryList[index]["script"])
        else:
            # If index is out of range, select the first script
            self.setSelectedScript(0)
        self.libraryView.update()

    def onSelectedScriptChanged(self, selected, deselected):
        # Save the current script before switching
        current_index = self.selectedScriptIndex()
        if current_index is not None:
            self.libraryList[current_index]["script"] = self.text.toPlainText()

        index = selected.indexes()
        if index:
            current = index[0].row()
            if current >= len(self.libraryList):
                self.onAddScript()
                return
            self.text.setPlainText(self.libraryList[current]["script"])
            self.currentScriptIndex = current

    def run(self):
        self.console.clear()
        script = self.text.toPlainText()
        self.scriptText = script

        try:
            if self.data is not None:
                self.console.write("Converting Orange data to R dataframe...\n")
                r_data = orange_to_r(self.data)
                self.console.write(f"R dataframe dimensions: {r_data.nrow} rows, {r_data.ncol} columns\n")
                self.console.write("Assigning R dataframe to global environment...\n")
                robjects.globalenv['data'] = r_data
            else:
                self.console.write("No input data provided. Running script without input data.\n")

            if self.learner is not None:
                self.console.write("Learner input provided, but not yet implemented\n")

            if self.model is not None:
                self.console.write("Model input provided, but not yet implemented\n")

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_filename = tmp_file.name

            with self.console.redirect_stdout_stderr():
                robjects.r(f'''
                    png("{tmp_filename}")
                    tryCatch({{
                        {script}
                    }}, finally = {{
                        dev.off()
                    }})
                ''')

            if os.path.getsize(tmp_filename) > 0:
                pixmap = QPixmap(tmp_filename)
                self.plot_window = PlotWindow(pixmap)
                self.plot_window.show()

            if 'result' in robjects.globalenv:
                result = robjects.globalenv['result']
                if isinstance(result, robjects.DataFrame):
                    output_data = r_to_orange(result)
                    self.Outputs.data.send(output_data)
                else:
                    self.console.write("Result variable exists but is not a dataframe. No data output.\n")
            else:
                self.console.write("No 'result' variable found in R environment. No data output.\n")

            self.console.write("Script executed successfully.\n")
        except Exception as e:
            self.console.write(f"Error: {str(e)}\n")
            import traceback
            self.console.write(traceback.format_exc())
        finally:
            if 'tmp_filename' in locals():
                os.remove(tmp_filename)
def orange_to_r(data):
    X = data.X
    Y = data.Y.reshape(-1, 1) if data.Y.ndim == 1 else data.Y
    M = data.metas
    all_data = np.hstack((X, Y, M))

    col_names = [var.name for var in data.domain.attributes +
                 (data.domain.class_vars if data.domain.class_var else []) +
                 data.domain.metas]

    r_dataframe = robjects.DataFrame({
        name: robjects.FloatVector(all_data[:, i]) if all_data[:, i].dtype.kind in 'iufc'
        else robjects.StrVector(all_data[:, i].astype(str))
        for i, name in enumerate(col_names)
    })

    return r_dataframe

def r_to_orange(r_df):
    with robjects.default_converter + numpy2ri.converter:
        pandas_df = robjects.conversion.rpy2py(r_df)

    domain = Domain([ContinuousVariable(name) if pandas_df[name].dtype.kind in 'iufc'
                     else DiscreteVariable(name, values=sorted(pandas_df[name].unique()))
                     for name in pandas_df.columns])

    return Table.from_numpy(domain, pandas_df.values)

if __name__ == "__main__":
    WidgetPreview(OWRScript).run()














# import sys
# import os
# import pandas as pd
# import numpy as np
# import rpy2.robjects as robjects
# from Orange.widgets.data.utils.pythoneditor.editor import PythonEditor
# from PyQt5.QtGui import QFont
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.conversion import localconverter
# from AnyQt.QtWidgets import (
#     QPlainTextEdit, QSplitter, QWidget, QVBoxLayout, QPushButton
# )
# from AnyQt.QtCore import Qt
# from Orange.widgets import gui
# from Orange.widgets.widget import OWWidget, Input, Output
# from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
# from Orange.widgets.settings import Setting
# from Orange.widgets.utils.widgetpreview import WidgetPreview
#
#
# class OWRScript(OWWidget):
#     name = "R Script"
#     description = "Run R scripts"
#     icon = "icons/RScript.svg"
#     priority = 3160
#
#     class Inputs:
#         data = Input("Data", Table)
#
#     class Outputs:
#         data = Output("Data", Table)
#
#     want_main_area = True
#     resizing_enabled = True
#
#     currentScriptIndex = Setting(0)
#     scriptText = Setting("")
#     splitterState = Setting(None)
#
#     def __init__(self):
#         super().__init__()
#         self.data = None
#
#         # GUI
#         self.splitCanvas = QSplitter(Qt.Vertical, self.mainArea)
#         self.mainArea.layout().addWidget(self.splitCanvas)
#
#         self.text = QPlainTextEdit(self)
#         self.text.setPlainText(self.scriptText)
#         self.splitCanvas.addWidget(self.text)
#
#         self.console = QPlainTextEdit(self)
#         self.console.setReadOnly(True)
#         self.splitCanvas.addWidget(self.console)
#
#         w = QWidget()
#         self.controlArea.layout().addWidget(w)
#         w.setLayout(QVBoxLayout())
#         w.layout().addWidget(self.splitCanvas)
#
#         gui.button(w, self, "Run script", callback=self.run)
#
#         self.resize(800, 600)
#
#     @Inputs.data
#     def set_data(self, data):
#         self.data = data
#
#     def run(self):
#         self.console.clear()
#         script = self.text.toPlainText()
#         self.scriptText = script
#
#         if self.data is not None:
#             try:
#                 # Convert Orange Table to pandas DataFrame
#                 domain = self.data.domain
#                 attributes = [var.name for var in domain.attributes]
#                 class_vars = [var.name for var in domain.class_vars]
#                 metas = [var.name for var in domain.metas]
#                 colnames = attributes + class_vars + metas
#
#                 df = pd.DataFrame(self.data.X, columns=attributes)
#                 if self.data.Y.size > 0:
#                     df[domain.class_var.name] = self.data.Y
#                 if self.data.metas.size > 0:
#                     df = pd.concat([df, pd.DataFrame(self.data.metas, columns=metas)], axis=1)
#
#                 # Convert pandas DataFrame to R dataframe
#                 with localconverter(robjects.default_converter + pandas2ri.converter):
#                     r_dataframe = robjects.conversion.py2rpy(df)
#
#                 robjects.globalenv['data'] = r_dataframe
#
#                 # Execute R script
#                 result = robjects.r(script)
#
#                 # Convert R dataframe back to pandas DataFrame
#                 if isinstance(result, robjects.DataFrame):
#                     with localconverter(robjects.default_converter + pandas2ri.converter):
#                         pandas_result = robjects.conversion.rpy2py(result)
#
#                     # Convert pandas DataFrame back to Orange Table
#                     domain = Domain([ContinuousVariable(name) for name in pandas_result.columns])
#                     output_data = Table.from_numpy(domain, pandas_result.values)
#
#                     self.Outputs.data.send(output_data)
#
#                 self.console.appendPlainText("Script executed successfully.")
#             except Exception as e:
#                 self.console.appendPlainText(f"Error: {str(e)}")
#         else:
#             self.console.appendPlainText("No input data available.")
#
#
# if __name__ == "__main__":
#     WidgetPreview(OWRScript).run()