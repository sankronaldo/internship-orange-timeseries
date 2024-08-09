import os
import sys
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
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
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, TimeVariable
from Orange.widgets.settings import Setting
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils import itemmodels

import tempfile

# Activate pandas to R conversion
pandas2ri.activate()

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
        font = QFont("Courier", 12)
        self.setFont(font)
        self.highlighter = RSyntaxHighlighter(self.document())


class RConsole(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(False)
        self.prompt = "R> "
        self.insertPlainText(self.prompt)
        self.command_history = []
        self.history_index = 0

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            self.execute_command()
        elif event.key() == Qt.Key_Up:
            self.show_previous_command()
        elif event.key() == Qt.Key_Down:
            self.show_next_command()
        else:
            super().keyPressEvent(event)

    def execute_command(self):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.movePosition(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
        line = cursor.selectedText()[len(self.prompt):]

        if line.strip():
            self.command_history.append(line.strip())
            self.history_index = len(self.command_history)
            self.insertPlainText('\n')
            try:
                # Check if the command is complete
                robjects.r('tryCatch({' + line + '}, error = function(e) {})')
                # If no error occurs, execute the command
                with self.redirect_stdout_stderr():
                    result = robjects.r(line)
                    if result is not None:
                        self.insertPlainText(str(result) + '\n')
            except Exception as e:
                self.insertPlainText(f"Error: {str(e)}\n")
        else:
            self.insertPlainText('\n')

        self.insertPlainText(self.prompt)
        self.ensureCursorVisible()

    def show_previous_command(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.replace_current_line(self.command_history[self.history_index])

    def show_next_command(self):
        if self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            self.replace_current_line(self.command_history[self.history_index])
        elif self.history_index == len(self.command_history) - 1:
            self.history_index += 1
            self.replace_current_line("")

    def replace_current_line(self, text):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.movePosition(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertText(self.prompt + text)

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

    def clear(self):
        super().clear()
        self.insertPlainText(self.prompt)

class PlotWindow(QMainWindow):
    def __init__(self, pixmap):
        super().__init__()
        self.setWindowTitle("R Plot")
        self.original_pixmap = pixmap
        self.label = QLabel()
        self.label.setPixmap(self.original_pixmap)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.label)
        scroll_area.setWidgetResizable(True)
        self.setCentralWidget(scroll_area)
        self.resize(800, 600)

    def resizeEvent(self, event):
        scaled_pixmap = self.original_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
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
    icon = "icons/final.svg"
    priority = 3160

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    want_main_area = True
    resizing_enabled = True

    currentScriptIndex = Setting(0)
    scriptText = Setting("")
    splitterState = Setting(None)
    scriptLibrary = Setting([{"name": "New script", "script": "# R script here\n"}])

    def __init__(self):
        super().__init__()

        self.data = None

        self.controlArea.layout().setContentsMargins(4, 4, 4, 4)
        self.mainArea.layout().setContentsMargins(0, 4, 4, 4)

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

        self.setSelectedScript(self.currentScriptIndex)

    @Inputs.data
    def set_data(self, data):
        self.data = data

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
            self.setSelectedScript(0)
        self.libraryView.update()

    def onSelectedScriptChanged(self, selected, deselected):
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
                self.console.write("Converting input data to R dataframe...\n")
                r_data = self.orange_to_r(self.data)
                self.console.write(f"R dataframe dimensions: {r_data.nrow} rows, {r_data.ncol} columns\n")
                self.console.write("Assigning R dataframe to global environment...\n")
                robjects.globalenv['orange_data'] = r_data

                robjects.r('data <- orange_data')
                self.console.write("Input data is now accessible as 'data' in your R script.\n")
            else:
                self.console.write("No input data provided. Running script without input data.\n")

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_filename = tmp_file.name

            with self.console.redirect_stdout_stderr():
                robjects.r(f'''
                    png("{tmp_filename}", width=1600, height=1200, res=300)
                    tryCatch({{
                        {script}
                    }}, error = function(e) {{
                        print(paste("Error:", e$message))
                    }}, finally = {{
                        dev.off()
                    }})
                ''')

            if os.path.getsize(tmp_filename) > 0:
                pixmap = QPixmap(tmp_filename)
                self.plot_window = PlotWindow(pixmap)
                self.plot_window.show()

            if robjects.r('exists("result")')[0]:
                result = robjects.r('result')
                if isinstance(result, robjects.vectors.DataFrame):
                    output_data = self.r_to_orange(result)
                    self.Outputs.data.send(output_data)
                    self.console.write("Output data sent successfully.\n")
                else:
                    pass
                    # self.console.write("'result' variable exists but is not a dataframe. No data output.\n")
            else:
                # self.console.write("No 'result' variable found in R environment. No data output.\n")
                pass

            # self.console.write("Script executed successfully.\n")
        except Exception as e:
            self.console.write(f"Error: {str(e)}\n")
        finally:
            if 'tmp_filename' in locals():
                os.remove(tmp_filename)

        self.console.write(self.console.prompt)

    def orange_to_r(self, data):
        if hasattr(data, 'time_variable'):
            # This is a Timeseries object
            time_var = data.time_variable
            domain = data.domain
            attrs = domain.attributes
            class_vars = domain.class_vars
            metas = domain.metas

            # Create a list of all variables
            all_vars = [time_var] + list(attrs) + list(class_vars) + list(metas)

            # Create a dictionary to store the data for each variable
            data_dict = {}

            for var in all_vars:
                if isinstance(var, TimeVariable):
                    # Handle time variable
                    time_data = data.get_column_view(var)[0]
                    pandas_datetime = pd.to_datetime(time_data, unit='s')
                    data_dict[var.name] = pandas_datetime
                else:
                    # Handle other variables
                    col_data = data.get_column_view(var)[0]
                    data_dict[var.name] = col_data

            # Create a pandas DataFrame
            df = pd.DataFrame(data_dict)
        else:
            # This is a regular Table object
            df = pd.DataFrame(data.X, columns=[var.name for var in data.domain.attributes])
            if data.Y.size:
                df[data.domain.class_var.name] = data.Y
            for meta_var, meta_col in zip(data.domain.metas, data.metas.T):
                df[meta_var.name] = meta_col

        # Convert pandas DataFrame to R dataframe
        with localconverter(robjects.default_converter + pandas2ri.converter):
            try:
                r_df = pandas2ri.py2rpy(df)
            except AttributeError:
                # If 'iteritems' is not found, we'll implement a workaround
                from rpy2.robjects import DataFrame
                r_df = DataFrame({col: robjects.vectors.FloatVector(df[col])
                if df[col].dtype.kind in 'iufc' else
                robjects.vectors.StrVector(df[col].astype(str))
                                  for col in df.columns})
        return r_df

    def r_to_orange(self, r_df):
        with localconverter(robjects.default_converter + pandas2ri.converter):
            pandas_df = robjects.conversion.rpy2py(r_df)

        domain = Domain(
            [ContinuousVariable(name) if pandas_df[name].dtype.kind in 'iufc'
             else DiscreteVariable(name, values=sorted(pandas_df[name].unique()))
             for name in pandas_df.columns]
        )

        return Table.from_numpy(domain, pandas_df.values)


if __name__ == "__main__":
    WidgetPreview(OWRScript).run()










# import os
# import sys
# import numpy as np
# import rpy2.robjects as robjects
# from rpy2.robjects import numpy2ri
# from rpy2.rinterface_lib import callbacks
#
# from AnyQt.QtWidgets import (
#     QPlainTextEdit, QSplitter, QWidget, QVBoxLayout, QPushButton,
#     QListView, QSizePolicy, QAction, QMenu, QToolButton, QInputDialog,
#     QLabel, QMainWindow, QScrollArea
# )
# from AnyQt.QtGui import QFont, QTextCursor, QSyntaxHighlighter, QTextCharFormat, QColor, QPixmap
# from AnyQt.QtCore import Qt, QItemSelectionModel, QRegExp
#
# from Orange.widgets import gui
# from Orange.widgets.widget import OWWidget, Input, Output
# from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
# from Orange.widgets.settings import Setting
# from Orange.widgets.utils.widgetpreview import WidgetPreview
# from Orange.widgets.utils import itemmodels
# from Orange.base import Learner, Model
# import tempfile
#
# # Initialize numpy to R conversion
# numpy2ri.activate()
#
#
# class RSyntaxHighlighter(QSyntaxHighlighter):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.highlighting_rules = []
#
#         keyword_format = QTextCharFormat()
#         keyword_format.setForeground(Qt.darkBlue)
#         keyword_format.setFontWeight(QFont.Bold)
#         keywords = [
#             "if", "else", "repeat", "while", "function", "for", "in", "next", "break",
#             "TRUE", "FALSE", "NULL", "Inf", "NaN", "NA", "NA_integer_", "NA_real_", "NA_complex_", "NA_character_"
#         ]
#         for word in keywords:
#             pattern = QRegExp("\\b" + word + "\\b")
#             self.highlighting_rules.append((pattern, keyword_format))
#
#         builtin_format = QTextCharFormat()
#         builtin_format.setForeground(Qt.darkCyan)
#         builtins = [
#             "print", "cat", "sprintf", "paste", "paste0", "length", "nrow", "ncol", "dim"
#         ]
#         for word in builtins:
#             pattern = QRegExp("\\b" + word + "\\b")
#             self.highlighting_rules.append((pattern, builtin_format))
#
#         number_format = QTextCharFormat()
#         number_format.setForeground(Qt.darkGreen)
#         number_pattern = QRegExp("\\b[+-]?[0-9]*\\.?[0-9]+([eE][+-]?[0-9]+)?\\b")
#         self.highlighting_rules.append((number_pattern, number_format))
#
#         string_format = QTextCharFormat()
#         string_format.setForeground(Qt.darkRed)
#         string_pattern = QRegExp("\".*?\"|'.*?'")
#         self.highlighting_rules.append((string_pattern, string_format))
#
#         comment_format = QTextCharFormat()
#         comment_format.setForeground(Qt.gray)
#         comment_pattern = QRegExp("#[^\n]*")
#         self.highlighting_rules.append((comment_pattern, comment_format))
#
#     def highlightBlock(self, text):
#         for pattern, format in self.highlighting_rules:
#             expression = QRegExp(pattern)
#             index = expression.indexIn(text)
#             while index >= 0:
#                 length = expression.matchedLength()
#                 self.setFormat(index, length, format)
#                 index = expression.indexIn(text, index + length)
#
#
# class RScriptEditor(QPlainTextEdit):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setFont(QFont("Courier", 10))
#         self.highlighter = RSyntaxHighlighter(self.document())
#
#
# class RConsole(QPlainTextEdit):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setReadOnly(False)
#         self.prompt = "R> "
#         self.insertPlainText(self.prompt)
#         self.last_command = ""
#
#     def keyPressEvent(self, event):
#         if event.key() == Qt.Key_Return:
#             self.execute_command()
#         else:
#             super().keyPressEvent(event)
#
#     def execute_command(self):
#         cursor = self.textCursor()
#         cursor.movePosition(QTextCursor.End)
#         cursor.movePosition(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
#         line = cursor.selectedText()[len(self.prompt):]
#         self.last_command = line.strip()
#
#         if self.last_command:
#             self.insertPlainText('\n')
#             try:
#                 with self.redirect_stdout_stderr():
#                     result = robjects.r(self.last_command)
#                     if result is not None:
#                         print(result)
#             except Exception as e:
#                 print(f"Error: {str(e)}")
#
#         self.insertPlainText(self.prompt)
#         self.ensureCursorVisible()
#
#     def write(self, text):
#         cursor = self.textCursor()
#         cursor.movePosition(QTextCursor.End)
#         cursor.insertText(text)
#         self.setTextCursor(cursor)
#         self.ensureCursorVisible()
#
#     def redirect_stdout_stderr(self):
#         class OutputRedirector:
#             def __init__(self, console):
#                 self.console = console
#
#             def write(self, text):
#                 self.console.write(text)
#
#         return callbacks.obj_in_module(sys, "stdout", OutputRedirector(self))
#
#
# class ROutputHandler:
#     def __init__(self, console):
#         self.console = console
#
#     def write(self, text):
#         self.console.write(text)
#
#
# class PlotWindow(QMainWindow):
#     def __init__(self, pixmap):
#         super().__init__()
#         self.setWindowTitle("R Plot")
#
#         # Create a QLabel to hold the pixmap
#         self.label = QLabel()
#         self.label.setPixmap(pixmap)
#         self.label.setScaledContents(True)
#
#         # Create a scroll area and set the label as its widget
#         scroll = QScrollArea()
#         scroll.setWidget(self.label)
#         scroll.setWidgetResizable(True)
#
#         # Set the scroll area as the central widget
#         self.setCentralWidget(scroll)
#
#         # Set a minimum size for the window
#         self.setMinimumSize(400, 300)
#
#     def resizeEvent(self, event):
#         # Resize the pixmap to fit the window while maintaining aspect ratio
#         pixmap = self.label.pixmap()
#         scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         self.label.setPixmap(scaled_pixmap)
#         super().resizeEvent(event)
#
#
# class ScriptItemModel(itemmodels.PyListModel):
#     def data(self, index, role=Qt.DisplayRole):
#         if role == Qt.DisplayRole:
#             return self[index.row()]["name"]
#         return super().data(index, role)
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
#         learner = Input("Learner", Learner)
#         model = Input("Model", Model)
#
#     class Outputs:
#         data = Output("Data", Table)
#         learner = Output("Learner", Learner)
#         model = Output("Model", Model)
#
#     want_main_area = True
#     resizing_enabled = True
#
#     currentScriptIndex = Setting(0)
#     scriptText = Setting("")
#     splitterState = Setting(None)
#     scriptLibrary = Setting([{"name": "New script", "script": "# R script here\n"}])
#
#     def __init__(self):
#         super().__init__()
#
#         self.data = None
#         self.learner = None
#         self.model = None
#
#         self.controlArea.layout().setContentsMargins(4, 4, 4, 4)
#         self.mainArea.layout().setContentsMargins(0, 4, 4, 4)
#
#         # GUI
#         self.splitCanvas = QSplitter(Qt.Vertical, self.mainArea)
#         self.mainArea.layout().addWidget(self.splitCanvas)
#
#         self.text = RScriptEditor(self)
#         self.splitCanvas.addWidget(self.text)
#
#         self.console = RConsole(self)
#         self.splitCanvas.addWidget(self.console)
#
#         self.controlBox = gui.vBox(self.controlArea, "Script Library")
#
#         self.libraryList = ScriptItemModel(self.scriptLibrary, self)
#         self.libraryView = QListView(
#             editTriggers=QListView.DoubleClicked | QListView.EditKeyPressed,
#             sizePolicy=QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
#         )
#         self.libraryView.setModel(self.libraryList)
#         self.libraryView.selectionModel().selectionChanged.connect(self.onSelectedScriptChanged)
#         self.controlBox.layout().addWidget(self.libraryView)
#
#         w = gui.hBox(self.controlBox)
#
#         self.addScriptButton = gui.button(w, self, "+", callback=self.onAddScript)
#         self.removeScriptButton = gui.button(w, self, "-", callback=self.onRemoveScript)
#         self.updateScriptButton = gui.button(w, self, "Update", callback=self.onUpdateScript)
#
#         gui.button(self.controlArea, self, "Run script", callback=self.run)
#
#         self.resize(800, 600)
#
#         # Ensure there's at least one script in the library
#         if not self.scriptLibrary:
#             self.scriptLibrary.append({"name": "New script", "script": "# R script here\n"})
#
#         # Ensure currentScriptIndex is within bounds
#         self.currentScriptIndex = min(self.currentScriptIndex, len(self.scriptLibrary) - 1)
#
#         self.setSelectedScript(self.currentScriptIndex)
#
#         self.r_output_handler = ROutputHandler(self.console)
#
#     @Inputs.data
#     def set_data(self, data):
#         self.data = data
#
#     @Inputs.learner
#     def set_learner(self, learner):
#         self.learner = learner
#
#     @Inputs.model
#     def set_model(self, model):
#         self.model = model
#
#     def onAddScript(self):
#         name, ok = QInputDialog.getText(self, "New Script", "Enter script name:")
#         if ok and name:
#             self.libraryList.append({"name": name, "script": "# R script here\n"})
#             self.setSelectedScript(len(self.libraryList) - 1)
#
#     def onRemoveScript(self):
#         if len(self.libraryList) > 1:
#             index = self.selectedScriptIndex()
#             del self.libraryList[index]
#             self.setSelectedScript(max(index - 1, 0))
#
#     def onUpdateScript(self):
#         index = self.selectedScriptIndex()
#         if index is not None:
#             self.libraryList[index]["script"] = self.text.toPlainText()
#
#     def selectedScriptIndex(self):
#         rows = self.libraryView.selectionModel().selectedRows()
#         if rows:
#             return rows[0].row()
#         return None
#
#     def setSelectedScript(self, index):
#         if 0 <= index < len(self.libraryList):
#             self.libraryView.setCurrentIndex(self.libraryList.index(index))
#             self.currentScriptIndex = index
#             self.text.setPlainText(self.libraryList[index]["script"])
#         else:
#             # If index is out of range, select the first script
#             self.setSelectedScript(0)
#         self.libraryView.update()
#
#     def onSelectedScriptChanged(self, selected, deselected):
#         # Save the current script before switching
#         current_index = self.selectedScriptIndex()
#         if current_index is not None:
#             self.libraryList[current_index]["script"] = self.text.toPlainText()
#
#         index = selected.indexes()
#         if index:
#             current = index[0].row()
#             if current >= len(self.libraryList):
#                 self.onAddScript()
#                 return
#             self.text.setPlainText(self.libraryList[current]["script"])
#             self.currentScriptIndex = current
#
#     def run(self):
#         self.console.clear()
#         script = self.text.toPlainText()
#         self.scriptText = script
#
#         try:
#             if self.data is not None:
#                 self.console.write("Converting Orange data to R dataframe...\n")
#                 r_data = orange_to_r(self.data)
#                 self.console.write(f"R dataframe dimensions: {r_data.nrow} rows, {r_data.ncol} columns\n")
#                 self.console.write("Assigning R dataframe to global environment...\n")
#                 robjects.globalenv['data'] = r_data
#             else:
#                 self.console.write("No input data provided. Running script without input data.\n")
#
#             if self.learner is not None:
#                 self.console.write("Learner input provided, but not yet implemented\n")
#
#             if self.model is not None:
#                 self.console.write("Model input provided, but not yet implemented\n")
#
#             with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
#                 tmp_filename = tmp_file.name
#
#             with self.console.redirect_stdout_stderr():
#                 robjects.r(f'''
#                     png("{tmp_filename}")
#                     tryCatch({{
#                         {script}
#                     }}, finally = {{
#                         dev.off()
#                     }})
#                 ''')
#
#             if os.path.getsize(tmp_filename) > 0:
#                 pixmap = QPixmap(tmp_filename)
#                 self.plot_window = PlotWindow(pixmap)
#                 self.plot_window.show()
#
#             if 'result' in robjects.globalenv:
#                 result = robjects.globalenv['result']
#                 if isinstance(result, robjects.DataFrame):
#                     output_data = r_to_orange(result)
#                     self.Outputs.data.send(output_data)
#                 else:
#                     self.console.write("Result variable exists but is not a dataframe. No data output.\n")
#             else:
#                 self.console.write("No 'result' variable found in R environment. No data output.\n")
#
#             self.console.write("Script executed successfully.\n")
#         except Exception as e:
#             self.console.write(f"Error: {str(e)}\n")
#             import traceback
#             self.console.write(traceback.format_exc())
#         finally:
#             if 'tmp_filename' in locals():
#                 os.remove(tmp_filename)
# def orange_to_r(data):
#     X = data.X
#     Y = data.Y.reshape(-1, 1) if data.Y.ndim == 1 else data.Y
#     M = data.metas
#     all_data = np.hstack((X, Y, M))
#
#     col_names = [var.name for var in data.domain.attributes +
#                  (data.domain.class_vars if data.domain.class_var else []) +
#                  data.domain.metas]
#
#     r_dataframe = robjects.DataFrame({
#         name: robjects.FloatVector(all_data[:, i]) if all_data[:, i].dtype.kind in 'iufc'
#         else robjects.StrVector(all_data[:, i].astype(str))
#         for i, name in enumerate(col_names)
#     })
#
#     return r_dataframe
#
# def r_to_orange(r_df):
#     with robjects.default_converter + numpy2ri.converter:
#         pandas_df = robjects.conversion.rpy2py(r_df)
#
#     domain = Domain([ContinuousVariable(name) if pandas_df[name].dtype.kind in 'iufc'
#                      else DiscreteVariable(name, values=sorted(pandas_df[name].unique()))
#                      for name in pandas_df.columns])
#
#     return Table.from_numpy(domain, pandas_df.values)
#
# if __name__ == "__main__":
#     WidgetPreview(OWRScript).run()