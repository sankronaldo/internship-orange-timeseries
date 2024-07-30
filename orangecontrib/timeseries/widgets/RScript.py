import sys
import os
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from Orange.widgets.data.utils.pythoneditor.editor import PythonEditor
from PyQt5.QtGui import QFont
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from AnyQt.QtWidgets import (
    QPlainTextEdit, QSplitter, QWidget, QVBoxLayout, QPushButton
)
from AnyQt.QtCore import Qt
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.settings import Setting
from Orange.widgets.utils.widgetpreview import WidgetPreview


class OWRScript(OWWidget):
    name = "R Script"
    description = "Run R scripts"
    icon = "icons/RScript.svg"
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

    def __init__(self):
        super().__init__()
        self.data = None

        # GUI
        self.splitCanvas = QSplitter(Qt.Vertical, self.mainArea)
        self.mainArea.layout().addWidget(self.splitCanvas)

        self.text = QPlainTextEdit(self)
        self.text.setPlainText(self.scriptText)
        self.splitCanvas.addWidget(self.text)

        self.console = QPlainTextEdit(self)
        self.console.setReadOnly(True)
        self.splitCanvas.addWidget(self.console)

        w = QWidget()
        self.controlArea.layout().addWidget(w)
        w.setLayout(QVBoxLayout())
        w.layout().addWidget(self.splitCanvas)

        gui.button(w, self, "Run script", callback=self.run)

        self.resize(800, 600)

    @Inputs.data
    def set_data(self, data):
        self.data = data

    def run(self):
        self.console.clear()
        script = self.text.toPlainText()
        self.scriptText = script

        if self.data is not None:
            try:
                # Convert Orange Table to pandas DataFrame
                domain = self.data.domain
                attributes = [var.name for var in domain.attributes]
                class_vars = [var.name for var in domain.class_vars]
                metas = [var.name for var in domain.metas]
                colnames = attributes + class_vars + metas

                df = pd.DataFrame(self.data.X, columns=attributes)
                if self.data.Y.size > 0:
                    df[domain.class_var.name] = self.data.Y
                if self.data.metas.size > 0:
                    df = pd.concat([df, pd.DataFrame(self.data.metas, columns=metas)], axis=1)

                # Convert pandas DataFrame to R dataframe
                with localconverter(robjects.default_converter + pandas2ri.converter):
                    r_dataframe = robjects.conversion.py2rpy(df)

                robjects.globalenv['data'] = r_dataframe

                # Execute R script
                result = robjects.r(script)

                # Convert R dataframe back to pandas DataFrame
                if isinstance(result, robjects.DataFrame):
                    with localconverter(robjects.default_converter + pandas2ri.converter):
                        pandas_result = robjects.conversion.rpy2py(result)

                    # Convert pandas DataFrame back to Orange Table
                    domain = Domain([ContinuousVariable(name) for name in pandas_result.columns])
                    output_data = Table.from_numpy(domain, pandas_result.values)

                    self.Outputs.data.send(output_data)

                self.console.appendPlainText("Script executed successfully.")
            except Exception as e:
                self.console.appendPlainText(f"Error: {str(e)}")
        else:
            self.console.appendPlainText("No input data available.")


if __name__ == "__main__":
    WidgetPreview(OWRScript).run()