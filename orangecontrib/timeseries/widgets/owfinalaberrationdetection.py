import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from pyod.models.iforest import IForest  # Using Isolation Forest for anomaly detection

class OWTimeSeriesAberration(widget.OWWidget):
    name = "Time Series Aberration Detection"
    description = "Detect and visualize aberrations in time series data"
    icon = "icons/final.svg"
    priority = 10

    class Inputs:
        time_series = Input("Time series", Table)

    class Outputs:
        aberrations = Output("Aberrations", Table)

    want_main_area = True

    # Settings
    contamination = Setting(0.1)  # Proportion of outliers in the data
    target_variable = Setting("")

    def __init__(self):
        super().__init__()

        self.data = None
        self.time_variable = None
        self.model = None
        self.results = None

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box, "No data on input.")

        # Target variable selection
        self.target_combo = gui.comboBox(
            box, self, "target_variable", label="Target Variable:",
            orientation="horizontal", sendSelectedValue=True, callback=self.on_target_variable_changed)

        # Model parameters
        params_box = gui.widgetBox(self.controlArea, "Model Parameters")
        gui.doubleSpin(params_box, self, "contamination", 0.01, 0.5, 0.01, label="Contamination:",
                       callback=self.on_param_changed)

        # Fit button
        self.fit_button = gui.button(self.controlArea, self, "Detect Aberrations", callback=self.fit_model)

        # Set up the main area with plot widget
        self.plot_widget = PlotWidget(background="w")
        self.mainArea.layout().addWidget(self.plot_widget)

    @Inputs.time_series
    def set_data(self, data):
        if data is not None:
            self.data = data
            self.info_label.setText(f"{len(data)} instances on input.")
            self.time_variable = getattr(data, 'time_variable', None)

            # Update target variable combo box options
            self.target_combo.clear()
            self.target_combo.addItem("")
            for var in data.domain.variables:
                if var.is_continuous:
                    self.target_combo.addItem(var.name)

            # Set initial target variable if previously selected
            if self.target_variable in data.domain:
                self.target_combo.setCurrentIndex(self.target_combo.findText(self.target_variable))

            self.on_target_variable_changed()
        else:
            self.data = None
            self.time_variable = None
            self.info_label.setText("No data on input.")
            self.clear_plot()

    def on_target_variable_changed(self):
        self.target_variable = self.target_combo.currentText()
        self.fit_model()

    def on_param_changed(self):
        self.fit_model()

    def fit_model(self):
        if self.data is None or not self.target_variable:
            return

        value_var = self.data.domain[self.target_variable]
        y = self.data.get_column(value_var).reshape(-1, 1)

        # Using Isolation Forest for aberration detection
        self.model = IForest(contamination=self.contamination)
        self.model.fit(y)

        # Predict anomalies
        labels = self.model.labels_  # 0 for inliers, 1 for outliers
        scores = self.model.decision_function(y)  # Anomaly scores

        self.results = {
            "labels": labels,
            "scores": scores
        }
        self.update_plot()
        self.output_aberrations()

    def update_plot(self):
        self.plot_widget.clear()

        if self.results is None:
            return

        labels = self.results["labels"]
        scores = self.results["scores"]
        y = self.data.get_column(self.data.domain[self.target_variable])
        x = np.arange(len(y))

        # Plot the time series
        self.plot_widget.plot(x, y, pen=pg.mkPen(color=(0, 0, 255), width=2))

        # Highlight anomalies
        anomaly_indices = np.where(labels == 1)[0]
        self.plot_widget.plot(x[anomaly_indices], y[anomaly_indices], pen=None,
                              symbol='o', symbolBrush=(255, 0, 0), symbolSize=10)

        self.plot_widget.setLabel('left', self.target_variable)
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setTitle('Aberration Detection')

    def clear_plot(self):
        self.plot_widget.clear()

    def output_aberrations(self):
        if self.results is None:
            self.Outputs.aberrations.send(None)
            return

        labels = self.results["labels"]
        scores = self.results["scores"]
        domain = Domain([ContinuousVariable('Aberration Scores')],
                        metas=[StringVariable('Time'), StringVariable('Label')])
        time_values = self.data.get_column(self.time_variable) if self.time_variable else np.arange(len(labels))
        label_values = np.array(["Aberration" if label == 1 else "Normal" for label in labels])
        aberrations_table = Table(domain, np.atleast_2d(scores).T,
                                  metas=np.vstack((time_values, label_values)).T)
        self.Outputs.aberrations.send(aberrations_table)

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWTimeSeriesAberration).run()
