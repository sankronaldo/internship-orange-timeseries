import Orange.widgets
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QTextEdit, QComboBox, QLabel
from PyQt5.QtCore import Qt

import pandas as pd
import numpy as np
from pycaret.time_series import *
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import sys
import contextlib
import base64
import re

class CaptureOutput:
    def __init__(self, output_widget):
        self.output_widget = output_widget
        self.buffer = StringIO()
        self.current_section = ""
        self.tables_open = False

    def write(self, text):
        self.buffer.write(text)
        self.process_output(text)

    def flush(self):
        pass

    def process_output(self, text):
        if "Setup completed" in text:
            if self.tables_open:
                self.end_table()
            self.current_section = "setup"
            self.output_widget.append("\nSetup Information\n" + "="*50 + "\n")
            self.start_table("setup")
            self.tables_open = True
        elif "Comparing" in text:
            if self.tables_open:
                self.end_table()
            self.current_section = "model_comparison"
            self.output_widget.append("\nModel Comparison\n" + "="*50 + "\n")
            self.start_table("model_comparison")
            self.tables_open = True
        elif "Transformation Pipeline and Model Successfully Saved" in text:
            if self.tables_open:
                self.end_table()
            self.current_section = "final"
            self.output_widget.append("\nFinal Output\n" + "="*50 + "\n")
            self.start_table("final")
            self.tables_open = True

        if self.current_section == "setup":
            self.process_setup(text)
        elif self.current_section == "model_comparison":
            self.process_model_comparison(text)
        else:
            self.output_widget.append(f"{text}\n")

    def start_table(self, section):
        if section == "setup":
            self.output_widget.append("+" + "-"*29 + "+" + "-"*19 + "+")
            self.output_widget.append("| Key" + " "*27 + "| Value" + " "*17 + "|")
            self.output_widget.append("+" + "-"*29 + "+" + "-"*19 + "+")
        elif section == "model_comparison":
            self.output_widget.append("+" + "-"*19 + "+" + "-"*19 + "+" + "-"*17 + "+")
            self.output_widget.append("| Model" + " "*17 + "| Metric" + " "*17 + "| Value" + " "*15 + "|")
            self.output_widget.append("+" + "-"*19 + "+" + "-"*19 + "+" + "-"*17 + "+")
        elif section == "final":
            self.output_widget.append("+" + "-"*29 + "+" + "-"*19 + "+")
            self.output_widget.append("| Detail" + " "*27 + "| Value" + " "*17 + "|")
            self.output_widget.append("+" + "-"*29 + "+" + "-"*19 + "+")

    def end_table(self):
        self.output_widget.append("+" + "-"*29 + "+" + "-"*19 + "+")
        self.tables_open = False

    def process_setup(self, text):
        # Extract key-value pairs and add them as table rows
        matches = re.findall(r'(\w+)\s*:\s*(\S+)', text)
        for key, value in matches:
            self.output_widget.append(f"| {key.ljust(27)} | {value.ljust(17)} |")

    def process_model_comparison(self, text):
        # Split the text into lines and process each line
        lines = text.split('\n')
        for line in lines:
            if line.strip():
                # Replace multiple spaces with table cells
                cells = re.split(r'\s{2,}', line.strip())
                row = "".join([f"| {cell.ljust(17)} " for cell in cells]) + "|"
                self.output_widget.append(row)



class TimeSeriesForecasting(OWWidget):
    name = "PyCaret Model Selection"
    description = "Perform time series forecasting using PyCaret"
    icon = "icons/final.svg"
    priority = 10

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        predictions = Output("Predictions", Orange.data.Table)

    want_main_area = True

    def __init__(self):
        super().__init__()

        self.data = None
        self.predictions = None
        self.target_variable = None

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box, "Please load data")

        self.target_combo = QComboBox(self.controlArea)
        self.target_combo.setEnabled(False)
        self.controlArea.layout().addWidget(QLabel("Select target variable:"))
        self.controlArea.layout().addWidget(self.target_combo)

        self.run_button = QPushButton("Run Forecasting", self.controlArea)
        self.run_button.clicked.connect(self.run_forecasting)
        self.controlArea.layout().addWidget(self.run_button)

        self.output_text = QTextEdit(self.mainArea)
        self.output_text.setReadOnly(True)
        self.output_text.setHtml("")
        self.mainArea.layout().addWidget(self.output_text)

    @Inputs.data
    def set_data(self, data):
        if data is not None:
            self.data = data
            self.info_label.setText(f"Data loaded: {len(self.data)} instances")
            self.update_target_combo()
        else:
            self.data = None
            self.info_label.setText("No data loaded")
            self.target_combo.clear()
            self.target_combo.setEnabled(False)

    def update_target_combo(self):
        self.target_combo.clear()
        if self.data is not None:
            # Display data structure information
            attributes = [var.name for var in self.data.domain.attributes]
            class_vars = [var.name for var in self.data.domain.class_vars]
            metas = [var.name for var in self.data.domain.metas]

            info = "<h3>Data Structure</h3>"
            info += "<table border='1' cellpadding='3' style='border-collapse: collapse;'>"
            info += f"<tr><td><b>Attributes</b></td><td>{', '.join(attributes)}</td></tr>"
            info += f"<tr><td><b>Class variables</b></td><td>{', '.join(class_vars)}</td></tr>"
            info += f"<tr><td><b>Meta attributes</b></td><td>{', '.join(metas)}</td></tr>"
            info += f"<tr><td><b>Shape</b></td><td>{self.data.X.shape}</td></tr>"
            info += "</table>"

            self.output_text.setHtml(info)

            # Populate combo box
            for var in self.data.domain.variables:
                self.target_combo.addItem(var.name)
            self.target_combo.setEnabled(True)
            self.target_variable = self.target_combo.currentText()

    def run_forecasting(self):
        if self.data is None:
            self.output_text.setHtml("<p>Please load data first.</p>")
            return

        # Redirect stdout to capture PyCaret output
        capture = CaptureOutput(self.output_text)
        old_stdout = sys.stdout
        sys.stdout = capture

        try:
            # Convert Orange.data.Table to pandas DataFrame
            df = pd.DataFrame(self.data.X)

            # If there's only one column, use it directly
            if df.shape[1] == 1:
                data = df.iloc[:, 0]
                target = self.data.domain.attributes[0].name
            else:
                # Get the selected target variable
                target = self.target_combo.currentText()

                # Find the index of the target variable
                target_index = next((i for i, var in enumerate(self.data.domain.attributes) if var.name == target),
                                    None)

                if target_index is None:
                    self.output_text.setHtml(
                        f"<p>Error: Selected target variable '{target}' not found in the data.</p>")
                    return

                data = df.iloc[:, target_index]

            # Setup
            s = setup(data, fh=3, session_id=123)

            # Model training and selection
            best = compare_models()

            # Make predictions
            predictions = predict_model(best)

            # Plot the model
            plot_output = plot_model(best, plot='forecast', return_fig=True)

            # Save the plot to a BytesIO object
            buf = BytesIO()
            if isinstance(plot_output, plt.Figure):
                plot_output.savefig(buf, format='png')
            elif isinstance(plot_output, tuple) and len(plot_output) > 0 and isinstance(plot_output[0], plt.Figure):
                plot_output[0].savefig(buf, format='png')
            else:
                self.output_text.append("<p>Note: Unable to display the forecast plot.</p>")
            buf.seek(0)

            # Convert predictions back to Orange.data.Table
            self.predictions = Orange.data.Table.from_numpy(None, predictions.values)

            # Add the plot to the output if we were able to save it
            if buf.getvalue():
                self.output_text.append("<h3>Forecast Plot</h3>")
                plot_data = base64.b64encode(buf.getvalue()).decode()
                self.output_text.append(f'<img src="data:image/png;base64,{plot_data}"/>')

            # Send predictions to output
            self.Outputs.predictions.send(self.predictions)

        except Exception as e:
            self.output_text.append(f"<p>An error occurred: {str(e)}</p>")
            raise  # This will print the full traceback to the console for debugging
        finally:
            # Restore stdout
            sys.stdout = old_stdout


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(TimeSeriesForecasting).run()
