import numpy as np
import pandas as pd
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtGui import QFont
from statsmodels.tsa.stattools import adfuller, kpss


class OWDifferencing(widget.OWWidget):
    name = "Stationarity & Differencing"
    description = "Visualize and apply differencing to time series data"
    icon = "icons/final.svg"
    priority = 130
    category = "Time Series"

    class Inputs:
        time_series = Input("Time series", Table)

    class Outputs:
        differenced_data = Output("Differenced Data", Table)

    want_main_area = True

    target_variable = Setting("")
    regular_diff_order = Setting(0)
    seasonal_diff_order = Setting(0)
    seasonality = Setting(12)

    def __init__(self):
        super().__init__()

        self.data = None
        self.time_variable = None
        self.differenced_data = None

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box, "No data on input.")
        self.adf_label = gui.widgetLabel(box, "")
        self.kpss_label = gui.widgetLabel(box, "")

        # Target variable selection
        self.target_combo = gui.comboBox(
            box, self, "target_variable", label="Target Variable:",
            callback=self.update_view)

        # Differencing controls
        diff_box = gui.widgetBox(self.controlArea, "Differencing Parameters")
        self.regular_diff_spin = gui.spin(
            diff_box, self, "regular_diff_order", 0, 2,
            label="Regular Differencing Order:", callback=self.update_view)
        self.seasonal_diff_spin = gui.spin(
            diff_box, self, "seasonal_diff_order", 0, 2,
            label="Seasonal Differencing Order:", callback=self.update_view)
        self.seasonality_spin = gui.spin(
            diff_box, self, "seasonality", 2, 365,
            label="Seasonality:", callback=self.update_view)

        # Set up the main area with two plot widgets
        self.original_plot = PlotWidget(background="w")
        self.differenced_plot = PlotWidget(background="w")

        gui.vBox(self.mainArea).layout().addWidget(self.original_plot)
        gui.vBox(self.mainArea).layout().addWidget(self.differenced_plot)

    @Inputs.time_series
    def set_data(self, data):
        self.clear_plots()
        if data is not None:
            self.data = data
            self.info_label.setText(f"{len(data)} instances on input.")
            self.time_variable = getattr(data, 'time_variable', None)

            # Update target variable combo box options
            self.target_combo.clear()
            continuous_vars = [var.name for var in data.domain.variables if var.is_continuous]
            self.target_combo.addItems(continuous_vars)

            # Set initial target variable
            if self.target_variable in continuous_vars:
                self.target_combo.setCurrentText(self.target_variable)
            elif continuous_vars:
                self.target_variable = continuous_vars[0]
                self.target_combo.setCurrentIndex(0)
            else:
                self.target_variable = ""

            self.update_view()
        else:
            self.data = None
            self.time_variable = None
            self.target_variable = ""
            self.info_label.setText("No data on input.")
            self.adf_label.setText("")
            self.kpss_label.setText("")

    def update_view(self):
        if self.data is None or not self.target_variable:
            return

        y_values = self.data.get_column(self.target_variable)

        # Check stationarity of original data
        self.check_stationarity(y_values, "Original")

        # Apply differencing
        differenced_values = self.apply_differencing(y_values)

        # Plot original and differenced data
        self.plot_data(self.original_plot, y_values, "Original Data")
        self.plot_data(self.differenced_plot, differenced_values, "Differenced Data")

        # Check stationarity of differenced data
        self.check_stationarity(differenced_values, "Differenced")

        # Send differenced data as output
        domain = Domain([ContinuousVariable("Differenced")])
        self.differenced_data = Table.from_numpy(domain, differenced_values.reshape(-1, 1))
        self.Outputs.differenced_data.send(self.differenced_data)

    def apply_differencing(self, data):
        df = pd.Series(data)

        # Apply regular differencing
        for _ in range(self.regular_diff_order):
            df = df.diff().dropna()

        # Apply seasonal differencing
        for _ in range(self.seasonal_diff_order):
            df = df - df.shift(self.seasonality)
            df = df.dropna()

        return df.values

    def check_stationarity(self, data, data_type):
        if len(data) <= 3:  # Not enough data for the tests
            self.adf_label.setText(f"{data_type}: Not enough data for stationarity tests")
            self.kpss_label.setText("")
            return

        # ADF Test
        adf_result = adfuller(data)
        adf_p_value = adf_result[1]
        adf_stationary = adf_p_value <= 0.05

        # KPSS Test
        kpss_result = kpss(data)
        kpss_p_value = kpss_result[1]
        kpss_stationary = kpss_p_value > 0.05

        self.adf_label.setText(
            f"{data_type} ADF test: {'stationary' if adf_stationary else 'non-stationary'} (p-value: {adf_p_value:.4f})")
        self.kpss_label.setText(
            f"{data_type} KPSS test: {'stationary' if kpss_stationary else 'non-stationary'} (p-value: {kpss_p_value:.4f})")

    def plot_data(self, plot_widget, values, title):
        plot_widget.clear()
        plot_widget.getAxis('bottom').setLabel('Time')
        plot_widget.getAxis('left').setLabel('Value')

        x = np.arange(len(values))
        plot_widget.plot(x, values, pen=pg.mkPen(color=(0, 0, 255), width=2))

        # Set background to white
        plot_widget.setBackground('w')

        # Increase font size and thickness of axis labels
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        plot_widget.getAxis('bottom').setTickFont(font)
        plot_widget.getAxis('left').setTickFont(font)

        # Set title with larger, bold font
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        plot_widget.setTitle(title, color='k', size='14pt')

        # Increase axis line width
        plot_widget.getAxis('bottom').setPen(pg.mkPen(color='k', width=2))
        plot_widget.getAxis('left').setPen(pg.mkPen(color='k', width=2))

    def clear_plots(self):
        self.original_plot.clear()
        self.differenced_plot.clear()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWDifferencing).run()







# import numpy as np
# import pandas as pd
# from Orange.widgets import widget, gui
# from Orange.widgets.settings import Setting
# from Orange.data import Table, Domain, ContinuousVariable
# from Orange.widgets.widget import Input, Output
# from Orange.widgets.visualize.utils.plotutils import PlotWidget
# import pyqtgraph as pg
# from PyQt5.QtGui import QFont
# from statsmodels.tsa.stattools import adfuller
#
#
# class OWDifferencing(widget.OWWidget):
#     name = "Time Series Differencing"
#     description = "Visualize and apply differencing to time series data"
#     icon = "icons/TimeSeriesDiff.svg"
#     priority = 130
#     category = "Time Series"
#
#     class Inputs:
#         time_series = Input("Time series", Table)
#
#     class Outputs:
#         differenced_data = Output("Differenced Data", Table)
#
#     want_main_area = True
#
#     target_variable = Setting("")
#     regular_diff_order = Setting(0)
#     seasonal_diff_order = Setting(0)
#     seasonality = Setting(12)
#
#     def __init__(self):
#         super().__init__()
#
#         self.data = None
#         self.time_variable = None
#         self.differenced_data = None
#
#         # GUI
#         box = gui.widgetBox(self.controlArea, "Info")
#         self.info_label = gui.widgetLabel(box, "No data on input.")
#         self.stationarity_label = gui.widgetLabel(box, "")
#
#         # Target variable selection
#         self.target_combo = gui.comboBox(
#             box, self, "target_variable", label="Target Variable:",
#             callback=self.update_view)
#
#         # Differencing controls
#         diff_box = gui.widgetBox(self.controlArea, "Differencing Parameters")
#         self.regular_diff_spin = gui.spin(
#             diff_box, self, "regular_diff_order", 0, 2,
#             label="Regular Differencing Order:", callback=self.update_view)
#         self.seasonal_diff_spin = gui.spin(
#             diff_box, self, "seasonal_diff_order", 0, 2,
#             label="Seasonal Differencing Order:", callback=self.update_view)
#         self.seasonality_spin = gui.spin(
#             diff_box, self, "seasonality", 2, 365,
#             label="Seasonality:", callback=self.update_view)
#
#         # Set up the main area with two plot widgets
#         self.original_plot = PlotWidget(background="w")
#         self.differenced_plot = PlotWidget(background="w")
#
#         gui.vBox(self.mainArea).layout().addWidget(self.original_plot)
#         gui.vBox(self.mainArea).layout().addWidget(self.differenced_plot)
#
#     @Inputs.time_series
#     def set_data(self, data):
#         self.clear_plots()
#         if data is not None:
#             self.data = data
#             self.info_label.setText(f"{len(data)} instances on input.")
#             self.time_variable = getattr(data, 'time_variable', None)
#
#             # Update target variable combo box options
#             self.target_combo.clear()
#             continuous_vars = [var.name for var in data.domain.variables if var.is_continuous]
#             self.target_combo.addItems(continuous_vars)
#
#             # Set initial target variable
#             if self.target_variable in continuous_vars:
#                 self.target_combo.setCurrentText(self.target_variable)
#             elif continuous_vars:
#                 self.target_variable = continuous_vars[0]
#                 self.target_combo.setCurrentIndex(0)
#             else:
#                 self.target_variable = ""
#
#             self.update_view()
#         else:
#             self.data = None
#             self.time_variable = None
#             self.target_variable = ""
#             self.info_label.setText("No data on input.")
#
#     def update_view(self):
#         if self.data is None or not self.target_variable:
#             return
#
#         y_values = self.data.get_column(self.target_variable)
#
#         # Check stationarity of original data
#         self.check_stationarity(y_values)
#
#         # Apply differencing
#         differenced_values = self.apply_differencing(y_values)
#
#         # Plot original and differenced data
#         self.plot_data(self.original_plot, y_values, "Original Data")
#         self.plot_data(self.differenced_plot, differenced_values, "Differenced Data")
#
#         # Check stationarity of differenced data
#         self.check_stationarity(differenced_values)
#
#         # Send differenced data as output
#         domain = Domain([ContinuousVariable("Differenced")])
#         self.differenced_data = Table.from_numpy(domain, differenced_values.reshape(-1, 1))
#         self.Outputs.differenced_data.send(self.differenced_data)
#
#     def apply_differencing(self, data):
#         df = pd.Series(data)
#
#         # Apply regular differencing
#         for _ in range(self.regular_diff_order):
#             df = df.diff().dropna()
#
#         # Apply seasonal differencing
#         for _ in range(self.seasonal_diff_order):
#             df = df - df.shift(self.seasonality)
#             df = df.dropna()
#
#         return df.values
#
#     def check_stationarity(self, data):
#         if len(data) <= 3:  # Not enough data for the test
#             self.stationarity_label.setText("Not enough data for stationarity test")
#             return
#
#         result = adfuller(data)
#         is_stationary = result[1] <= 0.05
#         status = "stationary" if is_stationary else "non-stationary"
#         self.stationarity_label.setText(f"The series is {status} (p-value: {result[1]:.4f})")
#
#     def plot_data(self, plot_widget, values, title):
#         plot_widget.clear()
#         plot_widget.getAxis('bottom').setLabel('Time')
#         plot_widget.getAxis('left').setLabel('Value')
#
#         x = np.arange(len(values))
#         plot_widget.plot(x, values, pen=pg.mkPen(color=(0, 0, 255), width=2))
#
#         # Set background to white
#         plot_widget.setBackground('w')
#
#         # Increase font size and thickness of axis labels
#         font = QFont()
#         font.setPointSize(12)
#         font.setBold(True)
#         plot_widget.getAxis('bottom').setTickFont(font)
#         plot_widget.getAxis('left').setTickFont(font)
#
#         # Set title with larger, bold font
#         title_font = QFont()
#         title_font.setPointSize(14)
#         title_font.setBold(True)
#         plot_widget.setTitle(title, color='k', size='14pt')
#
#         # Increase axis line width
#         plot_widget.getAxis('bottom').setPen(pg.mkPen(color='k', width=2))
#         plot_widget.getAxis('left').setPen(pg.mkPen(color='k', width=2))
#
#     def clear_plots(self):
#         self.original_plot.clear()
#         self.differenced_plot.clear()
#
#
# if __name__ == "__main__":
#     from Orange.widgets.utils.widgetpreview import WidgetPreview
#
#     WidgetPreview(OWDifferencing).run()