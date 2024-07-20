import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets.widget import Input, Output
from statsmodels.tsa.stattools import ccf
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtGui import QFont
from scipy import stats


class OWCCF(widget.OWWidget):
    name = "Cross-Correlation Function"
    description = "Plot the CCF between two time series"
    icon = "icons/final.svg"
    priority = 10

    class Inputs:
        time_series_1 = Input("Time Series 1", Table)
        time_series_2 = Input("Time Series 2", Table)

    class Outputs:
        ccf_data = Output("CCF Data", Table)

    want_main_area = True

    max_lags = Setting(30)  # Default to 30 lags
    target_variable_1 = Setting("")
    target_variable_2 = Setting("")
    significance_level = Setting(0.05)  # Default significance level (5%)

    def __init__(self):
        super().__init__()

        self.data_1 = None
        self.data_2 = None

        # Define fixed colors
        self.blue = pg.mkColor(0, 0, 255)  # Blue
        self.red = pg.mkColor(255, 0, 0)  # Red

        # Initialize color associations
        self.var1_color = self.blue
        self.var2_color = self.red

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box, "No data on input.")

        # Target variable selection for Time Series 1
        self.target_combo_1 = gui.comboBox(
            box, self, "target_variable_1", label="Target Variable 1:",
            orientation="horizontal", sendSelectedValue=True, callback=self.on_target_variable_changed)

        # Target variable selection for Time Series 2
        self.target_combo_2 = gui.comboBox(
            box, self, "target_variable_2", label="Target Variable 2:",
            orientation="horizontal", sendSelectedValue=True, callback=self.on_target_variable_changed)

        # Add max lags control
        lags_box = gui.widgetBox(self.controlArea, "CCF Parameters")
        self.lags_spin = gui.spin(
            lags_box, self, "max_lags", minv=1, maxv=365,
            label="Max Lags:", callback=self.on_target_variable_changed)

        # Add significance level control
        self.significance_spin = gui.doubleSpin(
            lags_box, self, "significance_level", minv=0.01, maxv=0.1, step=0.01,
            label="Significance Level:", callback=self.on_target_variable_changed)

        # Add a button to switch time series order
        self.switch_button = gui.button(
            self.controlArea, self, "Switch Time Series",
            callback=self.switch_time_series
        )

        # Set up the main area with the plot widget
        self.ccf_plot = PlotWidget(background="w")
        gui.vBox(self.mainArea).layout().addWidget(self.ccf_plot)

    @Inputs.time_series_1
    def set_data_1(self, data):
        self.data_1 = data
        self.update_info_label()
        self.update_target_combo(self.target_combo_1, data)
        if data is not None and self.target_variable_1 not in data.domain:
            self.target_variable_1 = ""
        self.plot_results()

    @Inputs.time_series_2
    def set_data_2(self, data):
        self.data_2 = data
        self.update_info_label()
        self.update_target_combo(self.target_combo_2, data)
        if data is not None and self.target_variable_2 not in data.domain:
            self.target_variable_2 = ""
        self.plot_results()

    def update_info_label(self):
        if self.data_1 is not None and self.data_2 is not None:
            self.info_label.setText(f"Time Series 1: {len(self.data_1)} instances\n"
                                    f"Time Series 2: {len(self.data_2)} instances")
        elif self.data_1 is not None:
            self.info_label.setText(f"Time Series 1: {len(self.data_1)} instances\n"
                                    f"Time Series 2: No data")
        elif self.data_2 is not None:
            self.info_label.setText(f"Time Series 1: No data\n"
                                    f"Time Series 2: {len(self.data_2)} instances")
        else:
            self.info_label.setText("No data on input.")

    def update_target_combo(self, combo, data):
        combo.clear()
        combo.addItem("")
        if data is not None:
            for var in data.domain.variables:
                if var.is_continuous:
                    combo.addItem(var.name)

    def on_target_variable_changed(self):
        self.plot_results()

    def switch_time_series(self):
        # Switch the data
        self.data_1, self.data_2 = self.data_2, self.data_1

        # Switch the target variables
        self.target_variable_1, self.target_variable_2 = self.target_variable_2, self.target_variable_1

        # Switch the color associations
        self.var1_color, self.var2_color = self.var2_color, self.var1_color

        # Update the combo boxes
        self.update_target_combo(self.target_combo_1, self.data_1)
        self.update_target_combo(self.target_combo_2, self.data_2)

        # Set the current index for the combo boxes
        if self.target_variable_1 in self.data_1.domain:
            self.target_combo_1.setCurrentIndex(self.target_combo_1.findText(self.target_variable_1))
        if self.target_variable_2 in self.data_2.domain:
            self.target_combo_2.setCurrentIndex(self.target_combo_2.findText(self.target_variable_2))

        # Update the info label
        self.update_info_label()

        # Replot the results
        self.plot_results()

    def plot_results(self):
        if self.data_1 is None or self.data_2 is None:
            self.clear_plot()
            return

        if not self.target_variable_1 or not self.target_variable_2:
            self.clear_plot()
            return

        try:
            y1 = self.data_1.get_column(self.target_variable_1)
        except ValueError:
            self.warning(f"Variable '{self.target_variable_1}' not found in the first input.")
            self.clear_plot()
            return

        try:
            y2 = self.data_2.get_column(self.target_variable_2)
        except ValueError:
            self.warning(f"Variable '{self.target_variable_2}' not found in the second input.")
            self.clear_plot()
            return

        # Ensure both series have the same length
        min_length = min(len(y1), len(y2))
        y1 = y1[:min_length]
        y2 = y2[:min_length]

        # Calculate CCF
        ccf_values = ccf(y1, y2, adjusted=True)

        # Limit the CCF values to the specified max_lags
        if len(ccf_values) > 2 * self.max_lags + 1:
            center = len(ccf_values) // 2
            start = center - self.max_lags
            end = center + self.max_lags + 1
            ccf_values = ccf_values[start:end]
        else:
            # If CCF values are fewer than 2 * max_lags + 1, adjust max_lags
            self.max_lags = (len(ccf_values) - 1) // 2

        # Determine if the order is reversed
        order_reversed = id(self.data_2) < id(self.data_1)

        # Plot CCF
        self.plot_correlation(self.ccf_plot, ccf_values, "Cross-Correlation", order_reversed)

        # Send CCF data as output
        domain = Domain([ContinuousVariable("CCF")],
                        metas=[ContinuousVariable("Lag")])
        ccf_data = Table.from_numpy(
            domain,
            ccf_values.reshape(-1, 1),
            metas=np.arange(-self.max_lags, self.max_lags + 1).reshape(-1, 1)
        )
        self.Outputs.ccf_data.send(ccf_data)

    def plot_correlation(self, plot_widget, values, plot_type, order_reversed):
        plot_widget.clear()
        plot_widget.getAxis('bottom').setLabel('Lag')
        plot_widget.getAxis('left').setLabel(plot_type)

        x = np.arange(-self.max_lags, self.max_lags + 1)

        # Use the color associations directly
        left_color = self.var2_color if order_reversed else self.var1_color
        right_color = self.var1_color if order_reversed else self.var2_color

        # Plot vertical lines (sticks) with different colors
        for i, val in zip(x, values):
            color = left_color if i < 0 else right_color if i > 0 else pg.mkColor(0, 0, 0)
            plot_widget.plot([i, i], [0, val], pen=pg.mkPen(color=color, width=2))

        # Plot markers at the top of each stick with different colors
        plot_widget.plot(x[x < 0], values[x < 0], pen=None, symbol='o',
                         symbolPen=pg.mkPen(color=left_color, width=1),
                         symbolBrush=left_color, symbolSize=5)
        plot_widget.plot(x[x > 0], values[x > 0], pen=None, symbol='o',
                         symbolPen=pg.mkPen(color=right_color, width=1),
                         symbolBrush=right_color, symbolSize=5)
        plot_widget.plot([0], [values[x == 0]], pen=None, symbol='o',
                         symbolPen=pg.mkPen(color=(0, 0, 0), width=1),
                         symbolBrush=(0, 0, 0), symbolSize=5)

        # Add zero line
        plot_widget.addLine(y=0, pen=pg.mkPen(color=(0, 0, 0), width=1, style=pg.QtCore.Qt.DashLine))

        # Add significance levels
        n = min(len(self.data_1), len(self.data_2))
        significance_level = stats.norm.ppf(1 - self.significance_level / 2) / np.sqrt(n)
        plot_widget.addLine(y=significance_level, pen=pg.mkPen(color=(0, 0, 0), width=1, style=pg.QtCore.Qt.DashLine))
        plot_widget.addLine(y=-significance_level, pen=pg.mkPen(color=(0, 0, 0), width=1, style=pg.QtCore.Qt.DashLine))

        # Set y-axis range
        y_max = max(max(abs(values)), significance_level)
        plot_widget.setYRange(-y_max * 1.1, y_max * 1.1)

        # Set x-axis range
        plot_widget.setXRange(x[0] - 0.5, x[-1] + 0.5)

        # Set background to white
        plot_widget.setBackground('w')

        # Increase font size and thickness of axis labels
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        plot_widget.getAxis('bottom').setTickFont(font)
        plot_widget.getAxis('left').setTickFont(font)

        # Set title with larger, bold font
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        var1_name = self.target_variable_2 if order_reversed else self.target_variable_1
        var2_name = self.target_variable_1 if order_reversed else self.target_variable_2
        var1_color_name = "red" if self.var1_color == self.red else "blue"
        var2_color_name = "red" if self.var2_color == self.red else "blue"
        plot_widget.setTitle(f"CCF: {var1_name} ({var1_color_name}) vs {var2_name} ({var2_color_name})", color='k',
                             size='12pt')

        # Increase axis line width
        plot_widget.getAxis('bottom').setPen(pg.mkPen(color='k', width=1))
        plot_widget.getAxis('left').setPen(pg.mkPen(color='k', width=1))

    def clear_plot(self):
        self.ccf_plot.clear()
        self.Outputs.ccf_data.send(None)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWCCF).run()








# import numpy as np
# from Orange.widgets import widget, gui
# from Orange.widgets.settings import Setting
# from Orange.data import Table, Domain, ContinuousVariable
# from Orange.widgets.widget import Input, Output
# from statsmodels.tsa.stattools import ccf
# from Orange.widgets.visualize.utils.plotutils import PlotWidget
# import pyqtgraph as pg
# from PyQt5.QtGui import QFont
# from scipy import stats
#
# class OWCCF(widget.OWWidget):
#     name = "Cross-Correlation Function"
#     description = "Plot the CCF between two time series"
#     icon = "icons/final.svg"
#     priority = 10
#
#     class Inputs:
#         time_series_1 = Input("Time Series 1", Table)
#         time_series_2 = Input("Time Series 2", Table)
#
#     class Outputs:
#         ccf_data = Output("CCF Data", Table)
#
#     want_main_area = True
#
#     max_lags = Setting(30)  # Default to 30 lags
#     target_variable_1 = Setting("")
#     target_variable_2 = Setting("")
#     significance_level = Setting(0.05)  # Default significance level (5%)
#
#     def __init__(self):
#         super().__init__()
#
#         self.data_1 = None
#         self.data_2 = None
#
#         # Define fixed colors for var1 and var2
#         self.var1_color = pg.mkColor(0, 0, 255)  # Blue for var1
#         self.var2_color = pg.mkColor(255, 0, 0)  # Red for var2
#
#         # GUI
#         box = gui.widgetBox(self.controlArea, "Info")
#         self.info_label = gui.widgetLabel(box, "No data on input.")
#
#         # Target variable selection for Time Series 1
#         self.target_combo_1 = gui.comboBox(
#             box, self, "target_variable_1", label="Target Variable 1:",
#             orientation="horizontal", sendSelectedValue=True, callback=self.on_target_variable_changed)
#
#         # Target variable selection for Time Series 2
#         self.target_combo_2 = gui.comboBox(
#             box, self, "target_variable_2", label="Target Variable 2:",
#             orientation="horizontal", sendSelectedValue=True, callback=self.on_target_variable_changed)
#
#         # Add max lags control
#         lags_box = gui.widgetBox(self.controlArea, "CCF Parameters")
#         self.lags_spin = gui.spin(
#             lags_box, self, "max_lags", minv=1, maxv=365,
#             label="Max Lags:", callback=self.on_target_variable_changed)
#
#         # Add significance level control
#         self.significance_spin = gui.doubleSpin(
#             lags_box, self, "significance_level", minv=0.01, maxv=0.1, step=0.01,
#             label="Significance Level:", callback=self.on_target_variable_changed)
#
#         # Add a button to switch time series order
#         self.switch_button = gui.button(
#             self.controlArea, self, "Switch Time Series",
#             callback=self.switch_time_series
#         )
#
#         # Set up the main area with the plot widget
#         self.ccf_plot = PlotWidget(background="w")
#         gui.vBox(self.mainArea).layout().addWidget(self.ccf_plot)
#
#     @Inputs.time_series_1
#     def set_data_1(self, data):
#         self.data_1 = data
#         self.update_info_label()
#         self.update_target_combo(self.target_combo_1, data)
#         if data is not None and self.target_variable_1 not in data.domain:
#             self.target_variable_1 = ""
#         self.plot_results()
#
#     @Inputs.time_series_2
#     def set_data_2(self, data):
#         self.data_2 = data
#         self.update_info_label()
#         self.update_target_combo(self.target_combo_2, data)
#         if data is not None and self.target_variable_2 not in data.domain:
#             self.target_variable_2 = ""
#         self.plot_results()
#
#     def update_info_label(self):
#         if self.data_1 is not None and self.data_2 is not None:
#             self.info_label.setText(f"Time Series 1: {len(self.data_1)} instances\n"
#                                     f"Time Series 2: {len(self.data_2)} instances")
#         elif self.data_1 is not None:
#             self.info_label.setText(f"Time Series 1: {len(self.data_1)} instances\n"
#                                     f"Time Series 2: No data")
#         elif self.data_2 is not None:
#             self.info_label.setText(f"Time Series 1: No data\n"
#                                     f"Time Series 2: {len(self.data_2)} instances")
#         else:
#             self.info_label.setText("No data on input.")
#
#     def update_target_combo(self, combo, data):
#         combo.clear()
#         combo.addItem("")
#         if data is not None:
#             for var in data.domain.variables:
#                 if var.is_continuous:
#                     combo.addItem(var.name)
#
#     def on_target_variable_changed(self):
#         self.plot_results()
#
#     def switch_time_series(self):
#         # Switch the data
#         self.data_1, self.data_2 = self.data_2, self.data_1
#
#         # Switch the target variables
#         self.target_variable_1, self.target_variable_2 = self.target_variable_2, self.target_variable_1
#
#         # Update the combo boxes
#         self.update_target_combo(self.target_combo_1, self.data_1)
#         self.update_target_combo(self.target_combo_2, self.data_2)
#
#         # Set the current index for the combo boxes
#         if self.target_variable_1 in self.data_1.domain:
#             self.target_combo_1.setCurrentIndex(self.target_combo_1.findText(self.target_variable_1))
#         if self.target_variable_2 in self.data_2.domain:
#             self.target_combo_2.setCurrentIndex(self.target_combo_2.findText(self.target_variable_2))
#
#         # Update the info label
#         self.update_info_label()
#
#         # Replot the results
#         self.plot_results()
#
#     def plot_results(self):
#         if self.data_1 is None or self.data_2 is None:
#             self.clear_plot()
#             return
#
#         if not self.target_variable_1 or not self.target_variable_2:
#             self.clear_plot()
#             return
#
#         try:
#             y1 = self.data_1.get_column(self.target_variable_1)
#         except ValueError:
#             self.warning(f"Variable '{self.target_variable_1}' not found in the first input.")
#             self.clear_plot()
#             return
#
#         try:
#             y2 = self.data_2.get_column(self.target_variable_2)
#         except ValueError:
#             self.warning(f"Variable '{self.target_variable_2}' not found in the second input.")
#             self.clear_plot()
#             return
#
#         # Ensure both series have the same length
#         min_length = min(len(y1), len(y2))
#         y1 = y1[:min_length]
#         y2 = y2[:min_length]
#
#         # Calculate CCF
#         ccf_values = ccf(y1, y2, adjusted=True)
#
#         # Limit the CCF values to the specified max_lags
#         if len(ccf_values) > 2 * self.max_lags + 1:
#             center = len(ccf_values) // 2
#             start = center - self.max_lags
#             end = center + self.max_lags + 1
#             ccf_values = ccf_values[start:end]
#         else:
#             # If CCF values are fewer than 2 * max_lags + 1, adjust max_lags
#             self.max_lags = (len(ccf_values) - 1) // 2
#
#         # Determine if the order is reversed
#         order_reversed = id(self.data_2) < id(self.data_1)
#
#         # Plot CCF
#         self.plot_correlation(self.ccf_plot, ccf_values, "Cross-Correlation", order_reversed)
#
#         # Send CCF data as output
#         domain = Domain([ContinuousVariable("CCF")],
#                         metas=[ContinuousVariable("Lag")])
#         ccf_data = Table.from_numpy(
#             domain,
#             ccf_values.reshape(-1, 1),
#             metas=np.arange(-self.max_lags, self.max_lags + 1).reshape(-1, 1)
#         )
#         self.Outputs.ccf_data.send(ccf_data)
#
#     def plot_correlation(self, plot_widget, values, plot_type, order_reversed):
#         plot_widget.clear()
#         plot_widget.getAxis('bottom').setLabel('Lag')
#         plot_widget.getAxis('left').setLabel(plot_type)
#
#         x = np.arange(-self.max_lags, self.max_lags + 1)
#
#         # Assign colors based on order
#         left_color = self.var2_color if order_reversed else self.var1_color
#         right_color = self.var1_color if order_reversed else self.var2_color
#
#         # Plot vertical lines (sticks) with different colors
#         for i, val in zip(x, values):
#             color = left_color if i < 0 else right_color if i > 0 else pg.mkColor(0, 0, 0)
#             plot_widget.plot([i, i], [0, val], pen=pg.mkPen(color=color, width=2))
#
#         # Plot markers at the top of each stick with different colors
#         plot_widget.plot(x[x < 0], values[x < 0], pen=None, symbol='o',
#                          symbolPen=pg.mkPen(color=left_color, width=1),
#                          symbolBrush=left_color, symbolSize=5)
#         plot_widget.plot(x[x > 0], values[x > 0], pen=None, symbol='o',
#                          symbolPen=pg.mkPen(color=right_color, width=1),
#                          symbolBrush=right_color, symbolSize=5)
#         plot_widget.plot([0], [values[x == 0]], pen=None, symbol='o',
#                          symbolPen=pg.mkPen(color=(0, 0, 0), width=1),
#                          symbolBrush=(0, 0, 0), symbolSize=5)
#
#         # Add zero line
#         plot_widget.addLine(y=0, pen=pg.mkPen(color=(0, 0, 0), width=1, style=pg.QtCore.Qt.DashLine))
#
#         # Add significance levels
#         n = min(len(self.data_1), len(self.data_2))
#         significance_level = stats.norm.ppf(1 - self.significance_level / 2) / np.sqrt(n)
#         plot_widget.addLine(y=significance_level, pen=pg.mkPen(color=(0, 0, 0), width=1, style=pg.QtCore.Qt.DashLine))
#         plot_widget.addLine(y=-significance_level, pen=pg.mkPen(color=(0, 0, 0), width=1, style=pg.QtCore.Qt.DashLine))
#
#         # Set y-axis range
#         y_max = max(max(abs(values)), significance_level)
#         plot_widget.setYRange(-y_max * 1.1, y_max * 1.1)
#
#         # Set x-axis range
#         plot_widget.setXRange(x[0] - 0.5, x[-1] + 0.5)
#
#         # Set background to white
#         plot_widget.setBackground('w')
#
#         # Increase font size and thickness of axis labels
#         font = QFont()
#         font.setPointSize(10)
#         font.setBold(True)
#         plot_widget.getAxis('bottom').setTickFont(font)
#         plot_widget.getAxis('left').setTickFont(font)
#
#         # Set title with larger, bold font
#         title_font = QFont()
#         title_font.setPointSize(12)
#         title_font.setBold(True)
#         var1_name = self.target_variable_2 if order_reversed else self.target_variable_1
#         var2_name = self.target_variable_1 if order_reversed else self.target_variable_2
#         plot_widget.setTitle(f"CCF: {var1_name} (blue) vs {var2_name} (red)", color='k', size='12pt')
#
#         # Increase axis line width
#         plot_widget.getAxis('bottom').setPen(pg.mkPen(color='k', width=1))
#         plot_widget.getAxis('left').setPen(pg.mkPen(color='k', width=1))
#
#     def clear_plot(self):
#         self.ccf_plot.clear()
#         self.Outputs.ccf_data.send(None)
#
# if __name__ == "__main__":
#     from Orange.widgets.utils.widgetpreview import WidgetPreview
#     WidgetPreview(OWCCF).run()
#
