import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable, StringVariable, TimeVariable
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtCore import Qt
import ruptures as rpt
from pyqtgraph import LinearRegionItem, mkBrush


class OWChangepointDetection(widget.OWWidget):
    name = "Changepoint Detection"
    description = "Detect changepoints in time series data using ruptures"
    icon = "icons/ow_changepoint.svg"
    priority = 10

    class Inputs:
        time_series = Input("Time series", Table)

    class Outputs:
        changepoints = Output("Changepoints", Table)

    want_main_area = True

    # Settings
    target_variable = Setting("")
    algorithm = Setting(0)  # 0: Pelt, 1: Binseg, 2: BottomUp, 3: Window
    penalty = Setting(1.0)

    def __init__(self):
        super().__init__()

        self.data = None
        self.time_variable = None
        self.changepoints = None

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box, "No data on input.")

        # Target variable selection
        self.target_combo = gui.comboBox(
            box, self, "target_variable", label="Target Variable:",
            orientation="horizontal", sendSelectedValue=True, callback=self.on_target_variable_changed)

        # Algorithm selection
        algo_box = gui.widgetBox(self.controlArea, "Algorithm")
        gui.comboBox(algo_box, self, "algorithm", items=["Pelt", "Binseg", "BottomUp", "Window"],
                     label="Algorithm:", orientation="horizontal", callback=self.on_param_changed)

        # Parameters
        param_box = gui.widgetBox(self.controlArea, "Parameters")
        gui.doubleSpin(param_box, self, "penalty", 0.1, 100, 0.1, label="Penalty:", callback=self.on_param_changed)

        # Detect button
        self.detect_button = gui.button(self.controlArea, self, "Detect Changepoints",
                                        callback=self.detect_changepoints)

        # Set up the main area with plot widget
        self.plot_widget = PlotWidget(background="w")
        self.mainArea.layout().addWidget(self.plot_widget)

    @Inputs.time_series
    def set_data(self, data):
        if data is not None:
            self.data = data
            self.info_label.setText(f"{len(data)} instances on input.")
            self.time_variable = data.domain.class_var if isinstance(data.domain.class_var, TimeVariable) else None
            if self.time_variable is None:
                self.time_variable = next((var for var in data.domain.metas if isinstance(var, TimeVariable)), None)

            # Update target variable combo box options
            self.target_combo.clear()
            self.target_combo.addItem("")
            for var in data.domain.variables:
                if var.is_continuous and not isinstance(var, TimeVariable):
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
        self.detect_changepoints()

    def on_param_changed(self):
        self.detect_changepoints()

    def detect_changepoints(self):
        if self.data is None or not self.target_variable:
            return

        try:
            value_var = self.data.domain[self.target_variable]
            signal = self.data.get_column(value_var).astype(float)

            if self.algorithm == 0:  # Pelt
                algo = rpt.Pelt(model='rbf').fit(signal)
                self.changepoints = algo.predict(pen=self.penalty)
                if not self.changepoints:  # If no changepoints detected, try with a lower penalty
                    self.changepoints = algo.predict(pen=self.penalty / 10)
            elif self.algorithm == 1:  # Binseg
                algo = rpt.Binseg(model='rbf').fit(signal)
                self.changepoints = algo.predict(n_bkps=max(1, int(self.penalty)))
            elif self.algorithm == 2:  # BottomUp
                algo = rpt.BottomUp(model='rbf').fit(signal)
                self.changepoints = algo.predict(n_bkps=max(1, int(self.penalty)))
            elif self.algorithm == 3:  # Window
                algo = rpt.Window(width=max(2, int(self.penalty)), model='rbf').fit(signal)
                self.changepoints = algo.predict(n_bkps=max(1, int(self.penalty)))

            self.changepoints = [cp for cp in self.changepoints if cp < len(signal)]

            self.update_plot(signal)
            self.update_info()
            self.output_changepoints()
        except Exception as e:
            self.error(f"An error occurred: {str(e)}")
            self.changepoints = None
            self.clear_plot()

    def update_plot(self, signal):
        self.plot_widget.clear()

        # Plot original signal
        self.plot_widget.plot(signal, pen=pg.mkPen(color=(0, 0, 255), width=2))

        # Add shaded regions
        if self.changepoints:
            regions = [(0, self.changepoints[0])] + \
                      list(zip(self.changepoints[:-1], self.changepoints[1:])) + \
                      [(self.changepoints[-1], len(signal))]

            for i, (start, end) in enumerate(regions):
                color = (255, 200, 200, 50) if i % 2 == 1 else (200, 200, 255, 50)
                region = LinearRegionItem(values=(start, end), brush=mkBrush(color), movable=False)
                self.plot_widget.addItem(region)

        # Plot changepoints
        if self.changepoints:
            for cp in self.changepoints:
                if cp < len(signal):
                    self.plot_widget.addLine(x=cp, pen=pg.mkPen(color=(255, 0, 0), width=2))

        self.plot_widget.setLabel('left', self.target_variable)
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setTitle('Changepoint Detection')

    def update_info(self):
        if self.changepoints is None:
            self.info_label.setText("No changepoints detected or an error occurred.")
            return

        info_text = f"Number of changepoints: {len(self.changepoints)}\n"
        info_text += f"Algorithm: {['Pelt', 'Binseg', 'BottomUp', 'Window'][self.algorithm]}\n"
        info_text += f"Penalty: {self.penalty}"

        self.info_label.setText(info_text)

    def clear_plot(self):
        self.plot_widget.clear()

    def output_changepoints(self):
        if self.changepoints is None or len(self.changepoints) == 0:
            self.Outputs.changepoints.send(None)
            return

        domain = Domain([ContinuousVariable('Changepoint')],
                        metas=[StringVariable('Time')])

        if self.time_variable:
            time_values = self.data.get_column(self.time_variable)
        else:
            time_values = np.arange(len(self.data))

        valid_changepoints = [cp for cp in self.changepoints if cp < len(time_values)]
        changepoint_times = [time_values[cp] for cp in valid_changepoints]

        changepoints_table = Table(domain, np.atleast_2d(valid_changepoints).T,
                                   metas=np.atleast_2d(changepoint_times).T)
        self.Outputs.changepoints.send(changepoints_table)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWChangepointDetection).run()
