import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QCheckBox

class OWCrossCorrelation(widget.OWWidget):
    name = "CCF - CrossCorrelation"
    description = "Plot the cross-correlation of two time series"
    icon = "icons/ow_ccf.svg"
    priority = 10

    class Inputs:
        time_series1 = Input("Time series 1", Table)
        time_series2 = Input("Time series 2", Table)

    class Outputs:
        ccf_data = Output("CCF Data", Table)

    want_main_area = True

    max_lags = Setting(30)  # Default to 30 lags
    target_variable1 = Setting("")
    target_variable2 = Setting("")

    def __init__(self):
        super().__init__()

        self.data1 = None
        self.data2 = None

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box, "No data on input.")

        # Target variable selection
        self.target_combo1 = gui.comboBox(
            box, self, "target_variable1", label="Target Variable 1:",
            orientation="horizontal", sendSelectedValue=True, callback=self.on_target_variable_changed)

        self.target_combo2 = gui.comboBox(
            box, self, "target_variable2", label="Target Variable 2:",
            orientation="horizontal", sendSelectedValue=True, callback=self.on_target_variable_changed)

        # Add max lags control
        lags_box = gui.widgetBox(self.controlArea, "CCF Parameters")
        self.lags_spin = gui.spin(
            lags_box, self, "max_lags", minv=1, maxv=365,
            label="Max Lags:", callback=self.on_target_variable_changed)

        # Add switch button
        self.switch_button = gui.button(
            self.controlArea, self, "Switch Time Series",
            callback=self.switch_time_series
        )

        # Set up the main area with plot widget
        self.ccf_plot = PlotWidget(background="w")
        gui.vBox(self.mainArea).layout().addWidget(self.ccf_plot)

    @Inputs.time_series1
    def set_data1(self, data):
        self.data1 = data
        self.update_interface()

    @Inputs.time_series2
    def set_data2(self, data):
        self.data2 = data
        self.update_interface()

    def update_interface(self):
        self.update_info_label()
        self.update_target_combos()
        self.plot_results()

    def update_info_label(self):
        if self.data1 is not None and self.data2 is not None:
            self.info_label.setText(f"Time series 1: {len(self.data1)} instances\n"
                                    f"Time series 2: {len(self.data2)} instances")
        elif self.data1 is not None:
            self.info_label.setText(f"Time series 1: {len(self.data1)} instances\n"
                                    f"Time series 2: No data")
        elif self.data2 is not None:
            self.info_label.setText(f"Time series 1: No data\n"
                                    f"Time series 2: {len(self.data2)} instances")
        else:
            self.info_label.setText("No data on input.")

    def update_target_combos(self):
        self.target_combo1.clear()
        self.target_combo2.clear()
        self.target_combo1.addItem("")
        self.target_combo2.addItem("")

        if self.data1 is not None:
            for var in self.data1.domain.variables:
                if var.is_continuous:
                    self.target_combo1.addItem(var.name)

        if self.data2 is not None:
            for var in self.data2.domain.variables:
                if var.is_continuous:
                    self.target_combo2.addItem(var.name)

    def on_target_variable_changed(self):
        self.target_variable1 = self.target_combo1.currentText()
        self.target_variable2 = self.target_combo2.currentText()
        self.plot_results()

    def switch_time_series(self):
        # Switch the data
        self.data1, self.data2 = self.data2, self.data1

        # Switch the target variables
        self.target_variable1, self.target_variable2 = self.target_variable2, self.target_variable1

        # Update the combo boxes
        self.update_target_combos()

        # Set the current index for the combo boxes
        if self.target_variable1 in self.data1.domain:
            self.target_combo1.setCurrentIndex(self.target_combo1.findText(self.target_variable1))
        if self.target_variable2 in self.data2.domain:
            self.target_combo2.setCurrentIndex(self.target_combo2.findText(self.target_variable2))

        # Update the info label
        self.update_info_label()

        # Replot the results
        self.plot_results()

    def calculate_ccf(self, x, y):
        x = (x - np.mean(x)) / (np.std(x) * len(x))
        y = (y - np.mean(y)) / np.std(y)
        ccf = np.correlate(y, x, mode='full')
        return ccf

    def plot_results(self):
        self.clear_plot()
        self.error()  # Clear any previous error messages

        if self.data1 is None or self.data2 is None:
            return
        if not self.target_variable1 or not self.target_variable2:
            return

        try:
            value_var1 = self.data1.get_column(self.target_variable1)
            value_var2 = self.data2.get_column(self.target_variable2)

            # Ensure both series have the same length
            min_length = min(len(value_var1), len(value_var2))
            value_var1 = value_var1[:min_length]
            value_var2 = value_var2[:min_length]

            # Calculate CCF
            ccf_values = self.calculate_ccf(value_var1, value_var2)

            # Generate lags
            lags = np.arange(-min_length + 1, min_length)

            # Ensure we only plot up to max_lags in each direction
            mask = (lags >= -self.max_lags) & (lags <= self.max_lags)
            lags = lags[mask]
            ccf_values = ccf_values[mask]

            # Plot CCF
            self.plot_correlation(lags, ccf_values, min_length)

            # Send CCF data as output
            domain = Domain([ContinuousVariable("CCF")],
                            metas=[ContinuousVariable("Lag")])
            output_data = Table.from_numpy(
                domain,
                ccf_values.reshape(-1, 1),
                metas=lags.reshape(-1, 1)
            )
            self.Outputs.ccf_data.send(output_data)
        except Exception as e:
            self.error(str(e))

    def plot_correlation(self, lags, values, n):
        self.ccf_plot.clear()
        self.ccf_plot.getAxis('bottom').setLabel('Lag')
        self.ccf_plot.getAxis('left').setLabel('Cross-Correlation')

        # Plot vertical lines (sticks)
        for lag, val in zip(lags, values):
            self.ccf_plot.plot([lag, lag], [0, val], pen=pg.mkPen(color=(0, 0, 255), width=2))

        # Plot markers at the top of each stick
        self.ccf_plot.plot(lags, values, pen=None, symbol='o',
                           symbolPen=pg.mkPen(color=(0, 0, 255), width=1),
                           symbolBrush=(0, 0, 255, 200), symbolSize=5)

        # Add zero line
        self.ccf_plot.addLine(y=0, pen=pg.mkPen(color=(0, 0, 0), width=1, style=pg.QtCore.Qt.DashLine))

        # Add 5% significance level lines
        sig_level = 1.96 / np.sqrt(n)  # 5% significance level
        self.ccf_plot.addLine(y=sig_level, pen=pg.mkPen(color=(255, 0, 0), width=2, style=pg.QtCore.Qt.DotLine))
        self.ccf_plot.addLine(y=-sig_level, pen=pg.mkPen(color=(255, 0, 0), width=2, style=pg.QtCore.Qt.DotLine))

        # Set y-axis range
        y_max = max(max(abs(np.min(values)), abs(np.max(values))), sig_level)
        self.ccf_plot.setYRange(-y_max * 1.1, y_max * 1.1)

        # Set x-axis range
        self.ccf_plot.setXRange(lags[0] - 0.5, lags[-1] + 0.5)

        # Set background to white
        self.ccf_plot.setBackground('w')

        # Increase font size and thickness of axis labels
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.ccf_plot.getAxis('bottom').setTickFont(font)
        self.ccf_plot.getAxis('left').setTickFont(font)

        # Set title with larger, bold font
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title = f"Cross-Correlation: {self.target_variable1} vs {self.target_variable2}"
        self.ccf_plot.setTitle(title, color='k', size='12pt')

        # Increase axis line width
        self.ccf_plot.getAxis('bottom').setPen(pg.mkPen(color='k', width=1))
        self.ccf_plot.getAxis('left').setPen(pg.mkPen(color='k', width=1))

    def clear_plot(self):
        if self.ccf_plot:
            self.ccf_plot.clear()

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWCrossCorrelation).run()









# import numpy as np
# from Orange.widgets import widget, gui
# from Orange.widgets.settings import Setting
# from Orange.data import Table, Domain, ContinuousVariable
# from Orange.widgets.widget import Input, Output
# from Orange.widgets.visualize.utils.plotutils import PlotWidget
# import pyqtgraph as pg
# from PyQt5.QtGui import QFont
# from PyQt5.QtWidgets import QCheckBox
#
#
# class OWCrossCorrelation(widget.OWWidget):
#     name = "Cross Correlation"
#     description = "Plot the cross-correlation of two time series"
#     icon = "icons/crosscorrelation.svg"
#     priority = 10
#
#     class Inputs:
#         time_series1 = Input("Time series 1", Table)
#         time_series2 = Input("Time series 2", Table)
#
#     class Outputs:
#         ccf_data = Output("CCF Data", Table)
#
#     want_main_area = True
#
#     max_lags = Setting(30)  # Default to 30 lags
#     target_variable1 = Setting("")
#     target_variable2 = Setting("")
#     reverse_order = Setting(False)
#
#     def __init__(self):
#         super().__init__()
#
#         self.data1 = None
#         self.data2 = None
#
#         # GUI
#         box = gui.widgetBox(self.controlArea, "Info")
#         self.info_label = gui.widgetLabel(box, "No data on input.")
#
#         # Target variable selection
#         self.target_combo1 = gui.comboBox(
#             box, self, "target_variable1", label="Target Variable 1:",
#             orientation="horizontal", sendSelectedValue=True, callback=self.on_target_variable_changed)
#
#         self.target_combo2 = gui.comboBox(
#             box, self, "target_variable2", label="Target Variable 2:",
#             orientation="horizontal", sendSelectedValue=True, callback=self.on_target_variable_changed)
#
#         # Add max lags control
#         lags_box = gui.widgetBox(self.controlArea, "CCF Parameters")
#         self.lags_spin = gui.spin(
#             lags_box, self, "max_lags", minv=1, maxv=365,
#             label="Max Lags:", callback=self.on_target_variable_changed)
#
#         # Add reverse order checkbox
#         self.reverse_checkbox = QCheckBox("Reverse Order")
#         self.reverse_checkbox.setChecked(self.reverse_order)
#         self.reverse_checkbox.stateChanged.connect(self.on_reverse_toggled)
#         lags_box.layout().addWidget(self.reverse_checkbox)
#
#         # Set up the main area with plot widget
#         self.ccf_plot = PlotWidget(background="w")
#         gui.vBox(self.mainArea).layout().addWidget(self.ccf_plot)
#
#     @Inputs.time_series1
#     def set_data1(self, data):
#         self.data1 = data
#         self.update_interface()
#
#     @Inputs.time_series2
#     def set_data2(self, data):
#         self.data2 = data
#         self.update_interface()
#
#     def update_interface(self):
#         self.update_info_label()
#         self.update_target_combos()
#         self.plot_results()
#
#     def update_info_label(self):
#         if self.data1 is not None and self.data2 is not None:
#             self.info_label.setText(f"Time series 1: {len(self.data1)} instances\n"
#                                     f"Time series 2: {len(self.data2)} instances")
#         elif self.data1 is not None:
#             self.info_label.setText(f"Time series 1: {len(self.data1)} instances\n"
#                                     f"Time series 2: No data")
#         elif self.data2 is not None:
#             self.info_label.setText(f"Time series 1: No data\n"
#                                     f"Time series 2: {len(self.data2)} instances")
#         else:
#             self.info_label.setText("No data on input.")
#
#     def update_target_combos(self):
#         self.target_combo1.clear()
#         self.target_combo2.clear()
#         self.target_combo1.addItem("")
#         self.target_combo2.addItem("")
#
#         if self.data1 is not None:
#             for var in self.data1.domain.variables:
#                 if var.is_continuous:
#                     self.target_combo1.addItem(var.name)
#
#         if self.data2 is not None:
#             for var in self.data2.domain.variables:
#                 if var.is_continuous:
#                     self.target_combo2.addItem(var.name)
#
#         if self.data1 is not None and self.target_variable1 in self.data1.domain:
#             self.target_combo1.setCurrentIndex(self.target_combo1.findText(self.target_variable1))
#         if self.data2 is not None and self.target_variable2 in self.data2.domain:
#             self.target_combo2.setCurrentIndex(self.target_combo2.findText(self.target_variable2))
#
#     def on_target_variable_changed(self):
#         self.target_variable1 = self.target_combo1.currentText()
#         self.target_variable2 = self.target_combo2.currentText()
#         self.plot_results()
#
#     def on_reverse_toggled(self, state):
#         self.reverse_order = state == 2  # Qt.Checked is 2
#         self.plot_results()
#
#     def calculate_ccf(self, x, y):
#         x = (x - np.mean(x)) / (np.std(x) * len(x))
#         y = (y - np.mean(y)) / np.std(y)
#         ccf = np.correlate(y, x, mode='full')
#         return ccf
#
#     def plot_results(self):
#         self.clear_plot()
#         self.error()  # Clear any previous error messages
#
#         if self.data1 is None or self.data2 is None:
#             return
#         if not self.target_variable1 or not self.target_variable2:
#             return
#
#         try:
#             value_var1 = self.data1.get_column(self.target_variable1)
#             value_var2 = self.data2.get_column(self.target_variable2)
#
#             if self.reverse_order:
#                 value_var1, value_var2 = value_var2, value_var1
#
#             # Ensure both series have the same length
#             min_length = min(len(value_var1), len(value_var2))
#             value_var1 = value_var1[:min_length]
#             value_var2 = value_var2[:min_length]
#
#             # Calculate CCF
#             ccf_values = self.calculate_ccf(value_var1, value_var2)
#
#             # Generate lags
#             lags = np.arange(-min_length + 1, min_length)
#
#             # Ensure we only plot up to max_lags in each direction
#             mask = (lags >= -self.max_lags) & (lags <= self.max_lags)
#             lags = lags[mask]
#             ccf_values = ccf_values[mask]
#
#             # Plot CCF
#             self.plot_correlation(lags, ccf_values)
#
#             # Send CCF data as output
#             domain = Domain([ContinuousVariable("CCF")],
#                             metas=[ContinuousVariable("Lag")])
#             output_data = Table.from_numpy(
#                 domain,
#                 ccf_values.reshape(-1, 1),
#                 metas=lags.reshape(-1, 1)
#             )
#             self.Outputs.ccf_data.send(output_data)
#         except Exception as e:
#             self.error(str(e))
#
#     def plot_correlation(self, lags, values):
#         self.ccf_plot.clear()
#         self.ccf_plot.getAxis('bottom').setLabel('Lag')
#         self.ccf_plot.getAxis('left').setLabel('Cross-Correlation')
#
#         # Plot vertical lines (sticks)
#         for lag, val in zip(lags, values):
#             self.ccf_plot.plot([lag, lag], [0, val], pen=pg.mkPen(color=(0, 0, 255), width=2))
#
#         # Plot markers at the top of each stick
#         self.ccf_plot.plot(lags, values, pen=None, symbol='o',
#                            symbolPen=pg.mkPen(color=(0, 0, 255), width=1),
#                            symbolBrush=(0, 0, 255, 200), symbolSize=5)
#
#         # Add zero line
#         self.ccf_plot.addLine(y=0, pen=pg.mkPen(color=(0, 0, 0), width=1, style=pg.QtCore.Qt.DashLine))
#
#         # Set y-axis range
#         y_max = max(abs(np.min(values)), abs(np.max(values)))
#         self.ccf_plot.setYRange(-y_max, y_max)
#
#         # Set x-axis range
#         self.ccf_plot.setXRange(lags[0] - 0.5, lags[-1] + 0.5)
#
#         # Set background to white
#         self.ccf_plot.setBackground('w')
#
#         # Increase font size and thickness of axis labels
#         font = QFont()
#         font.setPointSize(10)
#         font.setBold(True)
#         self.ccf_plot.getAxis('bottom').setTickFont(font)
#         self.ccf_plot.getAxis('left').setTickFont(font)
#
#         # Set title with larger, bold font
#         title_font = QFont()
#         title_font.setPointSize(12)
#         title_font.setBold(True)
#         self.ccf_plot.setTitle(f"Cross-Correlation: {self.target_variable1} vs {self.target_variable2}", color='k',
#                                size='12pt')
#
#         # Increase axis line width
#         self.ccf_plot.getAxis('bottom').setPen(pg.mkPen(color='k', width=1))
#         self.ccf_plot.getAxis('left').setPen(pg.mkPen(color='k', width=1))
#
#     def clear_plot(self):
#         if self.ccf_plot:
#             self.ccf_plot.clear()
#
#
# if __name__ == "__main__":
#     from Orange.widgets.utils.widgetpreview import WidgetPreview
#     WidgetPreview(OWCrossCorrelation).run()