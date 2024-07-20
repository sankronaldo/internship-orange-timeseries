import numpy as np
import pandas as pd
from scipy.signal import periodogram, savgol_filter
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, TimeVariable
from Orange.widgets.widget import Input
import pyqtgraph as pg

class OWPeriodogram(widget.OWWidget):
    name = "Periodogram"
    description = "View the raw and smoothed periodogram of a time series"
    icon = "icons/final.svg"
    priority = 10

    class Inputs:
        time_series = Input("Time series", Table)

    want_main_area = True

    # Add settings
    target_variable = Setting("")  # Selected target variable
    show_raw = Setting(True)       # Whether to show raw periodogram
    show_smoothed = Setting(True)  # Whether to show smoothed periodogram
    log_scale = Setting(False)      # Whether to use log scale for y-axis
    smoothing_window = Setting(5) # Window length for Savitzky-Golay filter

    def __init__(self):
        super().__init__()

        self.data = None
        self.plot_widget = None
        self.time_variable = None

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box, "No data on input.")

        # Target variable selection
        self.target_combo = gui.comboBox(
            box, self, "target_variable", label="Target Variable:",
            orientation="horizontal", sendSelectedValue=True, callback=self.on_target_variable_changed)

        # Add checkboxes for raw, smoothed, and log scale options
        gui.checkBox(box, self, "show_raw", "Show Raw Periodogram", callback=self.compute_and_plot_periodogram)
        gui.checkBox(box, self, "show_smoothed", "Show Smoothed Periodogram", callback=self.compute_and_plot_periodogram)
        gui.checkBox(box, self, "log_scale", "Log Scale Y-Axis", callback=self.compute_and_plot_periodogram)

        # Add a slider for smoothing window length
        gui.spin(box, self, "smoothing_window", 5, 201, step=2, label="Smoothing Window Length:",
                 callback=self.compute_and_plot_periodogram)

        # Create a layout for the plots
        self.plot_layout = pg.GraphicsLayoutWidget()
        self.mainArea.layout().addWidget(self.plot_layout)

    @Inputs.time_series
    def set_data(self, data):
        if data is not None:
            self.data = data
            self.info_label.setText(f"{len(data)} instances on input.")
            self.time_variable = data.time_variable

            # Update target variable combo box options
            self.target_combo.clear()
            self.target_combo.addItem("")
            for var in data.domain.variables:
                if var.is_continuous:
                    self.target_combo.addItem(var.name)

            # Set initial target variable if previously selected
            if self.target_variable in data.domain:
                self.target_combo.setCurrentIndex(self.target_combo.findText(self.target_variable))

            self.compute_and_plot_periodogram()
        else:
            self.data = None
            self.time_variable = None
            self.info_label.setText("No data on input.")
            self.clear_plot()

    def on_target_variable_changed(self):
        self.target_variable = self.target_combo.currentText()
        self.compute_and_plot_periodogram()

    def compute_and_plot_periodogram(self):
        if self.data is None or not self.target_variable:
            return

        # Find the time variable and the selected target variable
        time_var = self.time_variable
        value_var = next((var for var in self.data.domain.variables if var.name == self.target_variable), None)

        if time_var is None or value_var is None:
            return

        # Extract time and value data
        time_values = self.data.get_column(time_var)
        y_values = self.data.get_column(value_var)

        # Create pandas Series with the correct time index
        if isinstance(time_var, TimeVariable):
            index = pd.to_datetime(time_values)
        else:
            # If it's not a TimeVariable, use a default date range
            index = pd.date_range(start='1/1/2000', periods=len(time_values), freq='D')

        ts = pd.Series(y_values, index=index)

        # Compute periodogram
        f, Pxx = periodogram(ts, scaling='spectrum')

        # Ensure window length is odd and within valid range
        window_length = min(self.smoothing_window, len(Pxx) // 2 * 2 + 1, len(Pxx))
        if window_length < 5:
            window_length = 5  # Ensure a minimum window length of 5

        # Smooth the periodogram
        Pxx_smooth = savgol_filter(Pxx, window_length=window_length, polyorder=3)

        self.plot_results(f, Pxx, Pxx_smooth)

    def plot_results(self, f, Pxx, Pxx_smooth):
        self.plot_layout.clear()
        self.plot_layout.setBackground('w')

        plot = self.plot_layout.addPlot(row=0, col=0, title="Periodogram")

        # Style improvements
        plot.getAxis('bottom').setPen(pg.mkPen(color='k', width=1))
        plot.getAxis('left').setPen(pg.mkPen(color='k', width=1))
        plot.getAxis('bottom').setTextPen(pg.mkPen(color='k'))
        plot.getAxis('left').setTextPen(pg.mkPen(color='k'))
        plot.setLabel('left', 'Power')
        plot.setLabel('bottom', 'Frequency')
        plot.showGrid(x=True, y=True, alpha=0.3)

        if self.log_scale:
            plot.setLogMode(y=True)
        else:
            plot.setLogMode(y=False)

        if self.show_raw:
            # Plot raw periodogram with a blue line
            plot.plot(f, Pxx, pen=pg.mkPen(color='b', width=2), name="Raw")

        if self.show_smoothed:
            # Plot smoothed periodogram with a red line
            plot.plot(f, Pxx_smooth, pen=pg.mkPen(color='r', width=2), name="Smoothed")

        plot.setMouseEnabled(x=False, y=False)
        plot.enableAutoRange(enable=False)

    def clear_plot(self):
        if self.plot_layout is not None:
            self.plot_layout.clear()

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWPeriodogram).run()






# import numpy as np
# from scipy import signal
# from Orange.widgets import widget, gui
# from Orange.widgets.settings import Setting
# from Orange.data import Table, TimeVariable
# from Orange.widgets.widget import Input
# import pyqtgraph as pg
#
# class OWPeriodogram(widget.OWWidget):
#     name = "claudePeriodogram"
#     description = "Visualize raw and smoothed periodogram of time series data"
#     icon = "icons/Periodogram.svg"
#     priority = 10
#
#     class Inputs:
#         time_series = Input("Time series", Table)
#
#     want_main_area = True
#
#     # Add settings
#     target_variable = Setting("")
#     smoothing_window = Setting(5)  # Default smoothing window size
#
#     def __init__(self):
#         super().__init__()
#
#         self.data = None
#         self.time_variable = None
#
#         # GUI
#         box = gui.widgetBox(self.controlArea, "Info")
#         self.info_label = gui.widgetLabel(box, "No data on input.")
#
#         # Target variable selection
#         self.target_combo = gui.comboBox(
#             box, self, "target_variable", label="Target Variable:",
#             orientation="horizontal", sendSelectedValue=True, callback=self.on_target_variable_changed)
#
#         # Smoothing window control
#         smoothing_box = gui.widgetBox(self.controlArea, "Smoothing")
#         self.smoothing_spin = gui.spin(
#             smoothing_box, self, "smoothing_window", minv=1, maxv=100,
#             label="Smoothing Window:", callback=self.compute_periodogram)
#
#         # Create a layout for the plots
#         self.plot_layout = pg.GraphicsLayoutWidget()
#         self.mainArea.layout().addWidget(self.plot_layout)
#
#     @Inputs.time_series
#     def set_data(self, data):
#         if data is not None:
#             self.data = data
#             self.info_label.setText(f"{len(data)} instances on input.")
#             self.time_variable = data.time_variable
#
#             # Update target variable combo box options
#             self.target_combo.clear()
#             self.target_combo.addItem("")
#             for var in data.domain.variables:
#                 if var.is_continuous:
#                     self.target_combo.addItem(var.name)
#
#             # Set initial target variable if previously selected
#             if self.target_variable in data.domain:
#                 self.target_combo.setCurrentIndex(self.target_combo.findText(self.target_variable))
#
#             self.compute_periodogram()
#         else:
#             self.data = None
#             self.time_variable = None
#             self.info_label.setText("No data on input.")
#             self.clear_plot()
#
#     def on_target_variable_changed(self):
#         self.target_variable = self.target_combo.currentText()
#         self.compute_periodogram()
#
#     def compute_periodogram(self):
#         if self.data is None or not self.target_variable:
#             return
#
#         # Find the time variable and the selected target variable
#         time_var = self.time_variable
#         value_var = next((var for var in self.data.domain.variables if var.name == self.target_variable), None)
#
#         if time_var is None or value_var is None:
#             self.error("Input data must contain a time variable and the selected numeric variable.")
#             return
#
#         # Extract time and value data
#         time_values = self.data.get_column(time_var)
#         y_values = self.data.get_column(value_var)
#
#         # Compute raw periodogram
#         f, Pxx = signal.periodogram(y_values)
#
#         # Compute smoothed periodogram
#         Pxx_smoothed = np.convolve(Pxx, np.ones(self.smoothing_window)/self.smoothing_window, mode='same')
#
#         self.plot_results(f, Pxx, Pxx_smoothed)
#
#     def plot_results(self, f, Pxx, Pxx_smoothed):
#         self.plot_layout.clear()
#         self.plot_layout.setBackground('w')
#
#         # Raw periodogram plot
#         raw_plot = self.plot_layout.addPlot(row=0, col=0, title="Raw Periodogram")
#         raw_plot.plot(f, Pxx, pen=pg.mkPen(color='b', width=2))
#         raw_plot.setLogMode(x=False, y=True)
#         raw_plot.setLabel('left', 'Power')
#         raw_plot.setLabel('bottom', 'Frequency')
#         raw_plot.showGrid(x=True, y=True, alpha=0.3)
#
#         # Smoothed periodogram plot
#         smoothed_plot = self.plot_layout.addPlot(row=1, col=0, title="Smoothed Periodogram")
#         smoothed_plot.plot(f, Pxx_smoothed, pen=pg.mkPen(color='r', width=2))
#         smoothed_plot.setLogMode(x=False, y=True)
#         smoothed_plot.setLabel('left', 'Power')
#         smoothed_plot.setLabel('bottom', 'Frequency')
#         smoothed_plot.showGrid(x=True, y=True, alpha=0.3)
#
#         # Link x-axes of both plots
#         smoothed_plot.setXLink(raw_plot)
#
#     def clear_plot(self):
#         if self.plot_layout is not None:
#             self.plot_layout.clear()
#
# if __name__ == "__main__":
#     from Orange.widgets.utils.widgetpreview import WidgetPreview
#     WidgetPreview(OWPeriodogram).run()